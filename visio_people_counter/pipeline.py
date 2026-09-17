"""Главный конвейер подсчёта (headless, задача 15).

``Pipeline(config_path)`` собирает ВСЕ модули из конфига:

* :class:`~visio_people_counter.video_source.VideoSource` — файл
  (:class:`FileSource`) или HLS/URL (:class:`FfmpegPipeSource`, watchdog-переподключение);
* :class:`~visio_people_counter.motion_detector.MotionDetector`;
* :class:`~visio_people_counter.size_profile.SizeProfile` (w/h берутся из источника после open());
* :class:`~visio_people_counter.tracker_adapter.TrackerAdapter` — ``frame_rate`` = эффективный fps:
  ``processing.effective_fps > 0`` → он; иначе нативный fps источника; если fps неизвестен
  (HLS без стабильного таймкода) → fallback :data:`FALLBACK_FPS` (15) + лог-строка;
* счётчики через :func:`~visio_people_counter.line_counter.build_counters`
  (нормализованные координаты конфига → px на фактическом w/h);
* :class:`~visio_people_counter.event_log.EventLog`.

``run()`` — цикл read → detect → tracker.update → counters.update → event_log:

* при reconnect источника (хук ``on_reconnect``) — ``detector.reset()`` +
  ``tracker.reset()`` + сброс состояний треков в счётчиках; накопленные in/out НЕ обнуляем;
* warmup: первые ~``history/2`` кадров субтрактор «впечатывает» фон — события
  логируются как обычно, но pipeline выставляет флаг :pyattr:`warmup_done` и печатает
  лог-строку о прогреве (без усложнения логики);
* тайминги: fps обработки (EMA + средний), lag ``t_wall − t_video`` при известном
  времени видео; периодическая сводка в stdout каждые ``output.summary_interval_s``;
* SIGINT → graceful stop + финальная сводка; EOF файла → то же (``run() -> 0``);
* ``bench=True`` — замер ms каждого этапа (detect/track/count) на каждом кадре,
  в финале таблица p50/p95.

Файловый источник с ``effective_fps > 0``: кадры пропускаются (каждый N-й),
N = round(нативный fps / effective_fps). Для HLS/URL даунскейл и fps задаёт сам ffmpeg
в :class:`FfmpegPipeSource`.

API для GUI-режима (задача 16): ``Pipeline(cfg)`` → ``build()`` → по одному кадру
``step(frame) -> list[CrossingEvent]`` + публичные компоненты (source/detector/tracker/
counters/size_profile/event_log) и :meth:`BaseCounter.draw <visio_people_counter.line_counter.BaseCounter.draw>`.
"""

from __future__ import annotations

import math
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from .config import Config, describe_frame_range, describe_roi
from .event_log import EventLog
from .line_counter import BaseCounter, CrossingEvent, LineCounter, ZoneCounter, build_counters
from .report import RunMeta, build_report, choose_report_path, write_report
from .motion_detector import MotionDetector
from .size_profile import SizeProfile
from .tracker_adapter import TrackerAdapter
from .video_source import (
    FileSource,
    FfmpegPipeSource,
    VideoSource,
    VideoSourceError,
    is_url,
)

#: fps-заглушка для источников без стабильного таймкода (HLS); см. tracker_adapter.DEFAULT_FRAME_RATE.
FALLBACK_FPS = 15.0


def _emit(msg: str) -> None:
    """Строка лога pipeline в stdout (headless: stdout — единственный канал)."""
    print(f"[pipeline] {msg}", flush=True)


def counting_active(index: int,
                    frame_start: Optional[int] = None,
                    frame_end: Optional[int] = None) -> bool:
    """Активен ли подсчёт (трекер + счётчики) на кадре ``index`` (чистая функция).

    Номера 0-based; обе границы включительно. ``frame_start``/``frame_end`` —
    из ``cfg.processing``; null = без границы:

    * обе null → активен всегда;
    * только start → ``index >= frame_start``;
    * только end → ``index <= frame_end``;
    * оба → ``frame_start <= index <= frame_end``.
    """
    if frame_start is not None and index < frame_start:
        return False
    if frame_end is not None and index > frame_end:
        return False
    return True


def _percentile(values: list[float], q: float) -> float:
    """Перцентиль простого списка (линейная интерполяция); [] → nan."""
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * q
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return float(s[int(k)])
    return s[f] * (c - k) + s[c] * (k - f)


def save_debug_frame(image: np.ndarray, directory: str, frame_index: int) -> Optional[Path]:
    """Сохранить отладочный кадр в ``directory`` (создаётся автоматически).

    :returns: путь к сохранённому файлу или None при неудаче (ошибка записи
        не должна ронять конвейер — только warning в stderr).
    """
    try:
        d = Path(directory).expanduser()
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"frame_{frame_index:06d}.jpg"
        if not cv2.imwrite(str(p), image):
            return None
        return p
    except OSError as e:
        print(f"[pipeline] отладочный кадр {frame_index}: не удалось сохранить: {e}",
              file=sys.stderr, flush=True)
        return None


class Pipeline:
    """Сборка всех модулей из конфига + главный цикл обработки.

    :param config: путь к YAML-конфигу (str/Path) ИЛИ готовый объект
        :class:`~visio_people_counter.config.Config` (CLI передаёт объект, чтобы
        переопределить ``video.path`` флагом ``--video``).
    :param bench: True — замерять время этапов detect/track/count на каждом кадре
        и печатать p50/p95 в финальной сводке.

    Компоненты (source/detector/tracker/counters/size_profile/event_log) создаются
    в :meth:`build` (нужно разрешение/fps источника после ``open()``); ``run()``
    вызывает его сам, если ещё не вызван.
    """

    def __init__(self, config: "Union[str, Path, Config]", bench: bool = False,
                 save_events: bool = False) -> None:
        self.cfg: Config = Config.load(config) if isinstance(config, (str, Path)) else config
        self.bench = bool(bench)
        self.save_events = bool(save_events)

        # --- компоненты (заполняются в build()) ---------------------------------
        self.source: Optional[VideoSource] = None
        self.detector: Optional[MotionDetector] = None
        self.tracker: Optional[TrackerAdapter] = None
        self.size_profile: Optional[SizeProfile] = None
        self.counters: list[BaseCounter] = []
        self.event_log: Optional[EventLog] = None

        # --- состояние run() ------------------------------------------------------
        self.warmup_done: bool = False          # субтрактор прогрет (~history/2 кадров)
        self.frames_processed: int = 0          # кадров, прошедших через конвейер
        self.report_events: list[CrossingEvent] = []  # все события прогона (для markdown-отчёта)
        self.duration_s: float = 0.0            # длительность обработки (wall)
        self.avg_fps: float = 0.0               # frames_processed / duration_s
        self.ema_fps: float = 0.0               # EMA fps обработки
        self.last_lag_s: Optional[float] = None  # lag последнего кадра (t_wall − t_video), если известно
        self.max_lag_s: Optional[float] = None
        self.reconnects: int = 0                # сколько раз источник переподключился

        # базовая точка отсчёта lag (первый кадр с известным t_video)
        self._t_ref_wall: Optional[float] = None
        self._t_ref_video: Optional[float] = None

        self._stop = False                      # SIGINT / graceful stop
        self._bench_detect: list[float] = []
        self._bench_track: list[float] = []
        self._bench_count: list[float] = []
        self._bench_frame: list[float] = []

    # ------------------------------------------------------------------ build
    def _effective_frame_rate(self) -> float:
        """frame_rate трекера: effective_fps > 0 → он; иначе нативный fps источника;
        неизвестный fps (HLS без таймкода) → :data:`FALLBACK_FPS` + лог."""
        eff = self.cfg.processing.effective_fps
        if eff > 0:
            return float(eff)
        native = getattr(self.source, "fps", 0.0) or 0.0
        if native > 0:
            return float(native)
        _emit(f"fps источника неизвестен — использую fallback {FALLBACK_FPS:.0f} fps для трекера")
        return FALLBACK_FPS

    def build(self) -> "Pipeline":
        """Открыть источник и собрать все модули (идемпотентно; повторный вызов игнорируется).

        :raises VideoSourceError: источник открыть нельзя / video.path пусто.
        """
        if self.source is not None:
            return self
        v = self.cfg.video
        if not v.path:
            raise VideoSourceError("video.path пустой — задайте путь в конфиге или --video")

        # ROI (processing.roi) — кроп на уровне источника: весь конвейер видит
        # ROI-кадр как полный; координаты конфига/событий отсчитываются от ROI.
        roi = self.cfg.processing.roi
        if v.type == "file":
            p = Path(v.path).expanduser()
            if not is_url(v.path) and not p.is_file():
                raise VideoSourceError(f"видеофайл не найден: {v.path!r}")
            self.source = FileSource(str(p), loop_file=v.loop_file, roi=roi)
        else:  # hls (URL .m3u8/RTSP/http или локальный файл через ffmpeg-pipe)
            h = v.hls
            self.source = FfmpegPipeSource(
                v.path,
                effective_fps=self.cfg.processing.effective_fps,
                max_width=self.cfg.processing.max_width,
                reconnect_attempts=h.reconnect_attempts,
                reconnect_backoff_s=h.reconnect_backoff_s,
                bad_read_threshold=h.bad_read_threshold,
                on_reconnect=self._on_reconnect,
                roi=roi,
            )

        self.source.open()
        w, h = self.source.width, self.source.height
        if roi is not None:
            _emit(f"ROI применён: {describe_roi(roi)} (кроп на уровне источника; "
                  f"координаты конфига и событий — в системе ROI)")
        _emit(f"источник: {v.type} {v.path!r} → {w}x{h}, fps источника={self.source.fps or 'н/д'}, "
              f"effective_fps={self.cfg.processing.effective_fps or 'нативный'}")

        self.size_profile = SizeProfile(self.cfg.size_profile, self.cfg.objects, w, h)
        self.detector = MotionDetector(self.cfg)
        fr = self._effective_frame_rate()
        self.tracker = TrackerAdapter(self.cfg, frame_rate=fr)
        _emit(f"трекер: {self.cfg.tracker.type} @ frame_rate={fr:.1f} "
              f"(lost_track_buffer ≈ {self.tracker.maximum_frames_without_update} кадров)")
        self.counters = build_counters(self.cfg.counters, w, h, self.size_profile)
        for c in self.counters:
            ctype = "line" if isinstance(c, LineCounter) else ("zone" if isinstance(c, ZoneCounter) else type(c).__name__)
            _emit(f"счётчик: id={c.counter_id} type={ctype} count_mode={c.count_mode}")
        self.event_log = EventLog(self.cfg)
        return self

    def close(self) -> None:
        """Освободить ресурсы (идемпотентно)."""
        if self.source is not None:
            try:
                self.source.close()
            finally:
                self.source = None
        if self.event_log is not None:
            self.event_log.close()
            self.event_log = None

    # ------------------------------------------------------------------ reconnect
    def _on_reconnect(self) -> None:
        """Хук watchdog FfmpegPipeSource: сброс состояния после переподключения."""
        self.reconnects += 1
        _emit(f"reconnect #{self.reconnects}: детектор и трекер сброшены "
              f"(warmup субтрактора заново; счётчики in/out сохранены)")
        if self.detector is not None:
            self.detector.reset()
        if self.warmup_done or self.frames_processed:
            # после переподключения фон снова «не знаком» — прогреваем заново
            self.warmup_done = False
        if self.tracker is not None:
            self.tracker.reset()
        for c in self.counters:
            c.reset()  # сброс last_pos/состояний треков (id трекера снова с нуля)

    # ------------------------------------------------------------------ per-frame step
    def counting_active(self, index: int) -> bool:
        """Активен ли подсчёт на кадре ``index`` (читает ``cfg.processing``).

        См. :func:`counting_active`: обе границы включительно, 0-based номера.
        """
        p = self.cfg.processing
        return counting_active(index, p.frame_start, p.frame_end)

    def step(self, frame) -> list[CrossingEvent]:
        """Обработать один кадр: detect → track → count → log. Возвращает события кадра.

        Вызывается из :meth:`run`; GUI-режиму (задача 16) доступен напрямую — там же
        можно рисовать overlay через ``counter.draw(frame.image)`` по флагам ``cfg.debug``.
        """
        t0 = time.monotonic()
        # детектор — на КАЖДОМ кадре (обучение фона MOG2; интервал не влияет).
        blobs = self.detector.detect(frame.image, self.size_profile)
        t1 = time.monotonic()
        if self.counting_active(frame.index):
            objects = self.tracker.update(blobs)
        else:
            # вне интервала: трекер НЕ обновляем — чистый список, старые треки не
            # «проживают» до старта; счётчики событие получать не будут.
            objects = []
        t2 = time.monotonic()

        events: list[CrossingEvent] = []
        if self.counting_active(frame.index):
            # objects.min_lifetime_frames: подтверждённый трек попадает в счётчики,
            # только когда наблюдался минимум N кадров подряд (анти-вспышка/дубль).
            min_life = int(self.cfg.objects.min_lifetime_frames)
            countable = [o for o in objects if o.track_id == -1 or o.age_frames >= min_life]
            for c in self.counters:
                events.extend(c.update(countable, frame.t_wall,
                                       t_video=frame.t_video, frame_index=frame.index))
        self.report_events.extend(events)
        if events:
            self.event_log.log_events(events)
            for ev in events:
                _emit(f"СОБЫТИЕ {ev.counter_id} {ev.direction} track={ev.track_id} "
                      f"@ ({ev.x_px:.0f},{ev.y_px:.0f}) frame={ev.frame_index}")
        t3 = time.monotonic()

        if self.bench:
            self._bench_detect.append((t1 - t0) * 1e3)
            self._bench_track.append((t2 - t1) * 1e3)
            self._bench_count.append((t3 - t2) * 1e3)
            self._bench_frame.append((t3 - t0) * 1e3)

        # отладочные кадры (cfg.debug.save_debug_frames_dir; каждый N-й).
        dbg = self.cfg.debug
        if dbg.save_debug_frames_dir and \
                frame.index % max(1, int(dbg.debug_frame_step)) == 0:
            save_debug_frame(frame.image, dbg.save_debug_frames_dir, frame.index)

        self.frames_processed += 1
        if not self.warmup_done:
            warmup_frames = max(1, self.cfg.motion.history // 2)
            if self.frames_processed >= warmup_frames:
                self.warmup_done = True
                _emit(f"субтрактор прогрет (~{warmup_frames} кадров; события до прогрева тоже логируются)")

        # lag: насколько обработка отстаёт от таймлайна видео (только при известном t_video).
        if frame.t_video is not None:
            if self._t_ref_wall is None:
                self._t_ref_wall, self._t_ref_video = frame.t_wall, frame.t_video
            lag = (frame.t_wall - self._t_ref_wall) - (frame.t_video - self._t_ref_video)
            self.last_lag_s = lag
            if self.max_lag_s is None or lag > self.max_lag_s:
                self.max_lag_s = lag
        return events

    # ------------------------------------------------------------------ summaries
    def _fps_line(self) -> str:
        fps = f"avg={self.avg_fps:.1f} ema={self.ema_fps:.1f}"
        if self.last_lag_s is not None:
            fps += f" lag={self.last_lag_s:+.2f}s (max {self.max_lag_s:.2f}s)"
        return fps

    def print_summary(self, reason: str = "период") -> None:
        """Промежуточная сводка в stdout."""
        _emit(f"сводка ({reason}): кадров={self.frames_processed} {self._fps_line()} | "
              + self.event_log.summary())

    def _print_final(self, reason: str) -> None:
        if not self.cfg.output.final_summary:
            return
        print("=== ФИНАЛЬНАЯ СВОДКА ===", flush=True)
        print(f"причина остановки : {reason}", flush=True)
        print(f"длительность      : {self.duration_s:.1f} c (wall)", flush=True)
        print(f"обработано кадров : {self.frames_processed}", flush=True)
        p = self.cfg.processing
        print(f"интервал кадров   : {describe_frame_range(p.frame_start, p.frame_end)}", flush=True)
        roi_txt = describe_roi(p.roi)
        extra = "" if p.roi is None else " (координаты событий — в системе ROI)"
        print(f"ROI               : {roi_txt}{extra}", flush=True)
        fps = f"avg={self.avg_fps:.1f} ema={self.ema_fps:.1f}"
        if self.last_lag_s is not None:
            fps += f" lag_last={self.last_lag_s:+.2f}s lag_max={self.max_lag_s:.2f}s"
        print(f"fps обработки     : {fps}", flush=True)
        if self.reconnects:
            print(f"reconnect'ов      : {self.reconnects}", flush=True)
        print(self.event_log.summary(), flush=True)

        if self.bench and self._bench_frame:
            print("=== БЕНЧМАРК (ms, p50/p95) ===", flush=True)
            for name, vals in (("detect", self._bench_detect), ("track", self._bench_track),
                               ("count", self._bench_count), ("frame (итого)", self._bench_frame)):
                print(f"{name:<14}: p50={_percentile(vals, 0.50):8.2f}  "
                      f"p95={_percentile(vals, 0.95):8.2f}", flush=True)

    # ------------------------------------------------------------------ report
    def _write_report(self, reason: str) -> None:
        """Markdown-отчёт в конце прогона (всегда, даже при 0 событий; любой reason).

        Путь — :func:`choose_report_path`: явный ``output.report_path`` либо автопуть
        ``<видео>.report.md`` для файла; HLS/URL без явного пути → отчёт не создаётся.
        Ошибка записи не роняет конвейер (warning в stderr, как у debug-кадров).
        """
        path = choose_report_path(self.cfg)
        if path is None:
            _emit("отчёт не создаётся: нет локального видео и явного output.report_path")
            return
        v = self.cfg.video
        source_name = Path(v.path).name if (v.type == "file" and not is_url(v.path)) else v.path
        fps = float(getattr(self.source, "fps", 0.0) or 0.0) if self.source is not None else 0.0
        duration = float(getattr(self.source, "duration", 0.0) or 0.0) if self.source is not None else 0.0
        meta = RunMeta(
            source=source_name,
            frames_processed=self.frames_processed,
            fps=fps if fps > 0 else None,
            duration_s=duration if duration > 0 else None,
            reason=reason,
        )
        try:
            write_report(path, build_report(
                self.cfg, self.report_events, meta,
                save_events=self.save_events))
        except OSError as e:
            print(f"[pipeline] отчёт {path}: не удалось сохранить: {e}",
                  file=sys.stderr, flush=True)
            return
        print("=== ОТЧЁТ ===", flush=True)
        print(str(path), flush=True)

    # ------------------------------------------------------------------ run
    def _file_skip(self) -> int:
        """Каждый N-й кадр обрабатывать для FileSource при effective_fps > 0."""
        if self.cfg.processing.effective_fps <= 0 or not isinstance(self.source, FileSource):
            return 1
        native = self.source.fps
        if native <= 0:
            return 1
        return max(1, round(native / self.cfg.processing.effective_fps))

    def run(self) -> int:
        """Запустить конвейер до EOF/stop. :returns: 0 — корректное завершение."""
        if self.source is None:
            self.build()

        interval = max(0.0, float(self.cfg.output.summary_interval_s))
        skip = self._file_skip()
        if skip > 1:
            _emit(f"effective_fps={self.cfg.processing.effective_fps:g} → обрабатываю каждый {skip}-й кадр")

        # SIGINT → graceful stop (только на главном потоке).
        prev_handler = None
        in_main_thread = threading.current_thread() is threading.main_thread()
        if in_main_thread:
            try:
                prev_handler = signal.signal(signal.SIGINT, lambda *_a: setattr(self, "_stop", True))
            except (ValueError, OSError):  # pragma: no cover — нестандартная среда
                prev_handler = None

        t_start = time.monotonic()
        last_t = t_start
        next_summary = t_start + interval if interval > 0 else float("inf")
        reason = "EOF"

        try:
            while not self._stop:
                frame = self.source.read()
                if frame is None:
                    # FfmpegPipeSource возвращает None и во время переподключения —
                    # выходим из цикла только когда источник исчерпан.
                    if isinstance(self.source, FfmpegPipeSource) and not self.source.exhausted:
                        time.sleep(0.2)  # backoff уже отоспал внутри read(); не крутим CPU
                        continue
                    break
                if skip > 1 and frame.index % skip != 0:
                    continue

                self.step(frame)

                now = time.monotonic()
                dt = max(1e-6, now - last_t)
                last_t = now
                inst_fps = 1.0 / dt
                self.ema_fps = inst_fps if self.ema_fps <= 0 else 0.9 * self.ema_fps + 0.1 * inst_fps

                if interval > 0 and now >= next_summary:
                    # промежуточные показатели считаем от момента старта run()
                    self.avg_fps = self.frames_processed / max(1e-6, now - t_start)
                    self.print_summary("период")
                    next_summary += interval

            if self._stop:
                reason = "SIGINT (graceful stop)"
        finally:
            self.duration_s = time.monotonic() - t_start
            self.avg_fps = self.frames_processed / max(1e-6, self.duration_s)
            self._print_final(reason)
            self._write_report(reason)  # до close(): нужны fps/duration источника
            self.close()
            if in_main_thread and prev_handler is not None:
                try:
                    signal.signal(signal.SIGINT, prev_handler)
                except (ValueError, OSError):  # pragma: no cover
                    pass
        return 0
