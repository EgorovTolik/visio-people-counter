"""Обнаружение движения (background subtraction).

``MotionDetector`` оборачивает ``cv2.createBackgroundSubtractorMOG2`` или
``cv2.createBackgroundSubtractorKNN`` по ``cfg.motion.method`` и выдаёт список
``Blob`` (координаты в px исходного кадра) после:

1. ``apply()`` → fgmask; при ``detect_shadows=true`` (MOG2) пиксели маски со
   значением ``< shadow_threshold`` (тени MOG2 = 127 и прочие «слабые»
   foreground) обнуляются: ``mask = (fgmask >= shadow_threshold) * 255``
   (при ``shadow_threshold <= 127`` остаётся только строгий foreground ==255);
2. морфология ``MORPH_OPEN(morph_open)`` → ``MORPH_CLOSE(morph_close)``;
3. ``connectedComponentsWithStats``;
4. фильтры блока ``objects`` (см. :func:`filter_blobs`): площадь
   (через ``SizeProfile.min/max_area_at(cx)``, либо глобальные доли кадра),
   ``min_bbox_side_px``, ``aspect_ratio_range``, ``min_fill``.

``min_lifetime_frames`` НЕ применяется в самом детекторе — его выполняет
конвейер (``pipeline.Pipeline.step`` / ``gui.GuiPlayer``): подтверждённый трек
попадает в счётчики только после N кадров наблюдения подряд.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass

from .config import Config, ObjectsConfig
from .size_profile import SizeProfile


@dataclass(frozen=True)
class Blob:
    """Связная компонента маски движения (px, координаты исходного кадра)."""
    x: int          # левый край bbox
    y: int          # верхний край bbox
    w: int          # ширина bbox
    h: int          # высота bbox
    area: int       # число пикселей компонента (не w*h)
    cx: float       # центр bbox по x
    cy: float       # центр bbox по y


def filter_blobs(stats: np.ndarray, centroids: np.ndarray, counts: list | np.ndarray,
                 objects_cfg: ObjectsConfig, min_area_at, max_area_at) -> list[Blob]:
    """Отфильтровать компоненты из ``connectedComponentsWithStats``.

    :param stats: массив (n, 5) — [x, y, w, h, bbox_area]; строка 0 — фон.
    :param centroids: массив (n, 2) — центры масс.
    :param counts: число пикселей на компоненту (индекс = номер компонента).
    :param objects_cfg: блок ``objects`` конфига.
    :param min_area_at / max_area_at: callable ``x -> площадь в px²``
        (из ``SizeProfile`` либо константы глобальных долей кадра).

    Порядок проверок: сторона bbox → aspect ratio → заполнение → площадь.
    """
    blobs: list[Blob] = []
    n = len(stats)
    for i in range(1, n):
        x, y, w, h = (int(v) for v in stats[i][:4])
        if w < objects_cfg.min_bbox_side_px or h < objects_cfg.min_bbox_side_px:
            continue  # субпиксельный мусор
        aspect = w / h if h > 0 else 0.0
        if not (objects_cfg.aspect_ratio_range[0] <= aspect <= objects_cfg.aspect_ratio_range[1]):
            continue  # слишком плоский/высокий (машины, полосы)
        area = int(counts[i])
        fill = area / float(w * h) if w * h > 0 else 0.0
        if fill < objects_cfg.min_fill:
            continue  # россыпь шума, ветка — bbox почти пустой
        cx = float(centroids[i][0])
        if not (min_area_at(cx) <= area <= max_area_at(cx)):
            continue  # слишком малый/огромный для этой точки кадра
        blobs.append(Blob(x=x, y=y, w=w, h=h, area=area, cx=cx, cy=float(centroids[i][1])))
    return blobs


class MotionDetector:
    """Субтрактор фона + морфология + connected components + фильтры объектов."""

    def __init__(self, cfg: Config) -> None:
        m = cfg.motion
        if m.method == "mog2":
            self._bg = cv2.createBackgroundSubtractorMOG2(
                history=m.history,
                varThreshold=float(m.var_threshold),
                detectShadows=bool(m.detect_shadows),
            )
        elif m.method == "knn":
            # KNN не помечает тени значением 127 так, как MOG2; маска — 0/255.
            self._bg = cv2.createBackgroundSubtractorKNN(
                history=m.history,
                dist2Threshold=float(m.dist2_threshold),
                detectShadows=bool(m.detect_shadows),
            )
        else:
            # config.py уже валидирует enum; здесь — защита от ручного объекта.
            raise ValueError(f"motion.method: ожидалось 'mog2' или 'knn', получено {m.method!r}")

        # Отсечение теней имеет смысл только для MOG2 (значения 127 в маске).
        self._detect_shadows = bool(m.detect_shadows) and m.method == "mog2"
        self._shadow_threshold = int(m.shadow_threshold)
        self._kernel_open = cv2.getStructuringElement(
            cv2.MORPH_RECT, (int(m.morph_open[0]), int(m.morph_open[1])))
        self._kernel_close = cv2.getStructuringElement(
            cv2.MORPH_RECT, (int(m.morph_close[0]), int(m.morph_close[1])))

        o = cfg.objects
        self._objects = o
        self._min_area_fraction = float(o.min_area_fraction)
        self._max_area_fraction = float(o.max_area_fraction)
        # маска последнего detect() (0/255, после морфологии) — для GUI-overlay
        # (задача 16, cfg.debug.show_mask / calibrate 'm'); до первого detect() — None.
        self.last_mask: "np.ndarray | None" = None

    # ------------------------------------------------------------------

    def detect(self, frame_image: np.ndarray, size_profile: SizeProfile | None = None) -> list[Blob]:
        """Обработать один BGR-кадр и вернуть прошедшие фильтры blob'ы.

        :param frame_image: кадр в исходных координатах (BGR, uint8).
        :param size_profile: опциональный профиль адаптивных порогов площади;
                             если None — глобальные доли из ``cfg.objects``.
        """
        fgmask = self._bg.apply(frame_image)

        if self._detect_shadows:
            # Тени MOG2 имеют значение 127 (< shadow_threshold, обычно 200) —
            # обнуляем все «слабые» пиксели. При threshold <= 127 остаётся
            # только строгий foreground (==255).
            fgmask = np.where(fgmask >= self._shadow_threshold, np.uint8(255), np.uint8(0))

        mask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self._kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel_close)
        self.last_mask = mask

        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        counts = np.bincount(labels.ravel(), minlength=n) if n > 1 else []

        if isinstance(size_profile, SizeProfile):
            min_area_at = size_profile.min_area_at
            max_area_at = size_profile.max_area_at
        else:
            frame_area = float(frame_image.shape[0] * frame_image.shape[1])
            gmin = self._min_area_fraction * frame_area
            gmax = self._max_area_fraction * frame_area
            min_area_at = lambda x, _g=gmin: _g   # noqa: E731
            max_area_at = lambda x, _g=gmax: _g   # noqa: E731

        return filter_blobs(stats, centroids, counts, self._objects,
                            min_area_at, max_area_at)

    def reset(self) -> None:
        """Сбросить модель фона (вызывать при reconnect из pipeline)."""
        self._bg.clear()
