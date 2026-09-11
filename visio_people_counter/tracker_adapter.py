"""Обёртка над трекерами движения из пакета `trackers` (SORT / ByteTrack / BoTSORT).

Принцип: конвейер считает людей по motion-blob'ам (`motion_detector.Blob`), а не по
детекциям нейросети, поэтому у объектов НЕТ скорей уверенности. Трекерам передаём
``sv.Detections(xyxy=..., confidence=None)``: пакет `trackers` при ``confidence=None``
трактует все детекции как полностью уверенные (1.0, см.
``trackers.utils.detections.default_confidences``), и потому

* порог активации трека (``track_activation_threshold``) всегда пройден — трек
  создаётся на любой unmatched det;
* двухстадийная ассоциация ByteTrack/BoTSORT (high/low confidence) отключается —
  все трекеры деградируют до одностадийного Kalman-предсказания + IoU-матчинга,
  т.е. ``ByteTrack ≡ SORT`` при наших входах;
* единственное реальное различие между типами остаётся CMC у BoTSORT (нужны кадры,
  а на уровне blob'ов они не передаются — см. ниже).

Для антидубля в счётчиках (задача 14) важно только: один физический объект → один
``track_id``; пропавший на время перекрытия объект сохраняет id до тех пор, пока не
исчерпан ``lost_track_buffer``, после чего трек закрывается и новый объект получает
новый id. Неподтверждённые треки (меньше ``minimum_consecutive_frames`` кадров подряд)
получают ``track_id = -1`` — их в подсчёт не пускаем.

Семантика параметров (проверено по коду пакета, `trackers 2.6.0`):

* ``lost_track_buffer`` задаётся в КАДРАХ 30 FPS и масштабируется трекером:
  ``maximum_frames_without_update = max(1, ceil(frame_rate / 30 * lost_track_buffer))``.
  Т.е. при ``frame_rate=15`` и конфиге ``lost_track_buffer=60`` пропавший объект
  живёт ~30 кадров (~2 c). Эффективный fps потока передаётся через параметр
  конструктора/``set_frame_rate()`` — именно поэтому buffer масштабируется верно.
* ``frame_rate`` — только для масштабирования lost-track buffer (обязателен
  положительный, иначе пакет роняет ValueError); Kalman работает в «шагах кадра».

BoTSORT: в пакете три IoU-порога (first / second / unconfirmed association), а в
конфиге один ``minimum_iou_threshold`` — применяем его к first и unconfirmed, второй
стадийный порог оставляем дефолтным 0.5. CMC у BoTSORT без кадров не работает
(``frame=None`` → шаг CMC молча пропускается пакетом), поэтому на blob-уровне
BoTSORT ведёт себя как SORT; если понадобится компенсация качки камеры, pipeline
(задача 15) передаст кадры отдельным параметром ``update(..., frame=...)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import supervision as sv
from trackers import BoTSORTTracker, ByteTrackTracker, SORTTracker

from .config import Config
from .motion_detector import Blob

#: fps по умолчанию, если pipeline ещё не знает эффективный fps источника.
DEFAULT_FRAME_RATE = 15.0


@dataclass
class TrackedObject:
    """Трекуемый объект (обёртка над одной строкой результата трекера)."""

    track_id: int   # -1 = неподтверждённый трек (меньше minimum_consecutive_frames)
    x: int          # левый верхний угол bbox, px
    y: int          # px
    w: int          # ширина bbox, px
    h: int          # высота bbox, px
    cx: float       # центр по x, px
    cy: float       # центр по y, px
    age_frames: int  # сколько КАДРОВ подряд объект наблюдается с подтверждённым id;
                     # для неподтверждённых (-1) = 0


class TrackerAdapter:
    """Blob'ы → ``sv.Detections(confidence=None)`` → трекер → ``TrackedObject``.

    :param cfg: полный ``Config`` (используется блок ``cfg.tracker``).
    :param frame_rate: эффективный fps потока; ``None`` → :data:`DEFAULT_FRAME_RATE`.
        Перед стартом pipeline подставит фактический fps
        (``processing.effective_fps`` или нативный из источника); пока его нет —
        работаем с 15, и buffer трекера корректно пересчитывается.
    """

    def __init__(self, cfg: Config, frame_rate: float | None = None) -> None:
        self._tcfg = cfg.tracker
        self.frame_rate: float = (
            float(frame_rate) if frame_rate is not None else DEFAULT_FRAME_RATE
        )
        self._tracker = self._build_tracker(self.frame_rate)
        # track_id -> сколько кадров подряд он наблюдался (для age_frames)
        self._age: dict[int, int] = {}

    # ------------------------------------------------------------------ build
    def _build_tracker(self, frame_rate: float) -> SORTTracker | ByteTrackTracker | BoTSORTTracker:
        """Собрать конкретный трекер по ``cfg.tracker``."""
        tcfg = self._tcfg
        common = dict(
            lost_track_buffer=tcfg.lost_track_buffer,
            frame_rate=frame_rate,
            minimum_consecutive_frames=tcfg.minimum_consecutive_frames,
        )
        if tcfg.type == "sort":
            return SORTTracker(minimum_iou_threshold=tcfg.minimum_iou_threshold, **common)
        if tcfg.type == "bytetrack":
            # confidence=None → пакет сам отключает low-stage: ByteTrack ≡ SORT.
            return ByteTrackTracker(
                minimum_iou_threshold=tcfg.minimum_iou_threshold, **common
            )
        if tcfg.type == "botsort":
            # Один порог из конфига → first и unconfirmed ассоциация;
            # second-stage порог — дефолт пакета (0.5).
            return BoTSORTTracker(
                minimum_iou_threshold_first_assoc=tcfg.minimum_iou_threshold,
                minimum_iou_threshold_unconfirmed_assoc=tcfg.minimum_iou_threshold,
                **common,
            )
        # сюда не попасть: Config уже валидирует type
        raise ValueError(f"неизвестный tracker.type: {tcfg.type!r}")

    # ------------------------------------------------------------------ API
    @property
    def tracker_type(self) -> str:
        """``"sort" | "bytetrack" | "botsort"`` (из конфига)."""
        return self._tcfg.type

    @property
    def maximum_frames_without_update(self) -> int:
        """Сколько КАДРОВ (в текущем frame_rate) пропавший трек живёт до закрытия.

        Полезно для тестов/отладки: учитывает масштабирование пакета по fps.
        """
        return self._tracker.maximum_frames_without_update

    def set_frame_rate(self, fps: float) -> None:
        """Пересоздать трекер с новым эффективным fps (вызвать ДО начала потока).

        Пакет `trackers` вычисляет lost-track buffer от ``frame_rate`` один раз в
        конструкторе, поэтому «на лету» fps поменять нельзя — пересобираем трекер.
        Состояние треков при этом сбрасывается (как :meth:`reset`).
        """
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"frame_rate должен быть конечным > 0, получено: {fps!r}")
        self.frame_rate = float(fps)
        self._tracker = self._build_tracker(self.frame_rate)
        self._age.clear()

    def update(self, blobs: list[Blob]) -> list[TrackedObject]:
        """Шаг трекера на одном кадре.

        :param blobs: blob'ы текущего кадра (пустой список = в кадре никого).
            Пустые кадры ОБЯЗАТЕЛЬНЫ передавать — иначе Kalman/счётчик пропусков
            не продвинется и старые треки не закроются.
        :return: один ``TrackedObject`` на каждый blob, в том же порядке, что и вход;
            ``track_id = -1`` — трек ещё не подтверждён (или не смог связаться).
        """
        if blobs:
            xyxy = np.array(
                [[b.x, b.y, b.x + b.w, b.y + b.h] for b in blobs], dtype=np.float32
            )
            dets = sv.Detections(xyxy=xyxy, confidence=None)
        else:
            dets = sv.Detections.empty()  # xyxy/confidence = None — трекеры умеют

        result = self._tracker.update(dets)
        if not blobs:
            return []

        out: list[TrackedObject] = []
        for b, tid in zip(blobs, result.tracker_id):
            tid = int(tid)
            if tid >= 0:
                age = self._age.get(tid, 0) + 1
                self._age[tid] = age
            else:
                age = 0
            out.append(
                TrackedObject(
                    track_id=tid,
                    x=int(b.x), y=int(b.y), w=int(b.w), h=int(b.h),
                    cx=float(b.cx), cy=float(b.cy),
                    age_frames=age,
                )
            )
        return out

    def reset(self) -> None:
        """Полный сброс состояния (вызывается при reconnect потока / смене видео).

        Закрывает все треки и обнуляет счётчик id — после reset первый трек снова
        получит ``track_id = 0`` (пакет `trackers` нумерует с нуля).
        """
        self._tracker.reset()
        self._age.clear()
