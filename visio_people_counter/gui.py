"""GUI-режим (``--gui``, ``--speed``) — задача 16.

Два компонента:

* :class:`GuiOverlay` — рисует на BGR-кадре отладочную визуализацию по
  ``cfg.debug``: bbox подтверждённых треков + ``track_id``, необработанные
  blob'ы (``show_all_blobs``), микс с маской движения (``show_mask``),
  линии/зоны счётчиков (:meth:`BaseCounter.draw`) + крупные текущие в/out-
  счётчики в углу. Чистая функция от кадра — тестируется БЕЗ окна.
* :class:`GuiPlayer` — окно ``cv2.imshow`` поверх готового
  :class:`~visio_people_counter.pipeline.Pipeline`: пробел — пауза,
  q/ESC — выход, ``+``/``-`` — скорость ×/÷1.5 (диапазон :data:`MIN_SPEED`..:data:`MAX_SPEED`).
  Скорость задаётся CLI ``--speed`` (0.25..8).

Headless-фолбэк: :meth:`GuiPlayer.available` — чистая проверка без создания
окна (env DISPLAY/WAYLAND_DISPLAY + GUI-бэкенд в ``cv2.getBuildInformation()``);
при ``False`` CLI печатает ясную ошибку и возвращает rc=1.
"""

from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

from .config import Config, DebugConfig
from .line_counter import BaseCounter
from .text_overlay import put_text, text_width
from .motion_detector import Blob
from .pipeline import Pipeline, save_debug_frame
from .tracker_adapter import TrackedObject
from .video_source import FfmpegPipeSource

#: допустимый диапазон скорости воспроизведения (CLI --speed и клавиши +/-).
MIN_SPEED = 0.25
MAX_SPEED = 8.0


#: системная папка плагинов Qt5 (там живёт platformtheme gtk3/gtk2)
_SYSTEM_QT_PLUGIN_DIR = "/usr/lib/x86_64-linux-gnu/qt5/plugins"


def apply_qt_env() -> None:
    """Поправить окружение для встроенного Qt в колесе opencv-python.

    OpenCV 5 HighGUI (наш бэкенд — QT5) тянет **свой** Qt из site-packages и не видит
    системные плагины: без platformtheme диалоги (например, QFileDialog встроенной
    фичи «Save current image» по Ctrl+S/иконке дискеты в тулбаре окна) рисуются чужим
    стилем с пустыми подписями папок. Подключаем системный gtk3-плагин, чтобы Qt брал
    шрифты/тему из GTK-сессии.
    """
    if os.path.isdir(_SYSTEM_QT_PLUGIN_DIR):
        cur = os.environ.get("QT_PLUGIN_PATH", "")
        parts = [p for p in cur.split(os.pathsep) if p]
        if _SYSTEM_QT_PLUGIN_DIR not in parts:
            parts.append(_SYSTEM_QT_PLUGIN_DIR)
            os.environ["QT_PLUGIN_PATH"] = os.pathsep.join(parts)
    os.environ.setdefault("QT_QPA_PLATFORMTHEME", "gtk3")


# до первого cv2.imshow/namedWindow (создание QApplication) — на момент импорта модуля
apply_qt_env()


def clamp_speed(speed: float) -> float:
    """Привести скорость к диапазону [:data:`MIN_SPEED`, :data:`MAX_SPEED`]."""
    return max(MIN_SPEED, min(MAX_SPEED, float(speed)))


def _emit(msg: str) -> None:
    print(f"[gui] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Overlay (чистая отрисовка — без imshow)
# ---------------------------------------------------------------------------

class GuiOverlay:
    """Отладочная отрисовка на кадре по флагам ``cfg.debug``.

    :param debug_cfg: блок ``debug`` конфигурации (:class:`~visio_people_counter.config.DebugConfig`).
    """

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    #: минимальный fontScale подписей (ТЗ задачи 16: читаемые, >= 0.6)
    MIN_FONT_SCALE = 0.6

    def __init__(self, debug_cfg: DebugConfig) -> None:
        self.debug = debug_cfg

    # ------------------------------------------------------------------ draw
    def draw(self, frame: np.ndarray, *,
             objects: list[TrackedObject] | None = None,
             blobs: list[Blob] | None = None,
             mask: np.ndarray | None = None,
             counters: list[BaseCounter] | None = None,
             status_text: str = "") -> np.ndarray:
        """Нарисовать overlay на BGR-кадр (in-place) и вернуть его.

        :param frame: кадр (h, w, 3) uint8 — модифицируется на месте.
        :param objects: треки текущего кадра (``cfg.debug.show_bboxes``).
        :param blobs: необработанные blob'ы до трекера (``show_all_blobs``).
        :param mask: маска движения 0/255 того же размера (``show_mask`` → микс 50/50).
        :param counters: счётчики (``show_counters``): линия/зона через
            ``counter.draw`` + крупные текущие in/out в левом верхнем углу.
        :param status_text: строка статуса справа сверху (скорость, PAUSED...).
        """
        if frame is None or frame.size == 0:
            return frame

        # 1) микс с маской движения (JET-цветность, 50/50)
        if self.debug.show_mask and mask is not None:
            m = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET)
            frame[:] = cv2.addWeighted(frame, 0.5, m, 0.5, 0.0)

        # 2) необработанные blob'ы (до трекера/фильтров подтверждённости) — пурпурный
        if self.debug.show_all_blobs and blobs:
            for b in blobs:
                cv2.rectangle(frame, (b.x, b.y), (b.x + b.w, b.y + b.h),
                              (255, 0, 255), 1, lineType=cv2.LINE_AA)

        # 3) bbox треков + track_id: зелёный — подтверждённый, оранжевый — новый (-1)
        if self.debug.show_bboxes and objects:
            for o in objects:
                color = (0, 255, 0) if o.track_id != -1 else (0, 165, 255)
                cv2.rectangle(frame, (o.x, o.y), (o.x + o.w, o.y + o.h), color, 2,
                              lineType=cv2.LINE_AA)
                label = f"#{o.track_id}" if o.track_id != -1 else "new"
                tx = max(0, int(o.x))
                ty = int(o.y) - 6 if int(o.y) - 6 > 14 else int(o.y) + 18
                cv2.putText(frame, label, (tx, ty), self.FONT,
                            max(self.MIN_FONT_SCALE, 0.6), color, 2, cv2.LINE_AA)

        # 4) счётчики: линия/зона (line_counter.draw) + крупные in/out в углу
        if self.debug.show_counters and counters:
            for c in counters:
                c.draw(frame)
            y = 36
            for c in counters:
                # put_text: кириллические id счётчиков тоже отрисовываются (PIL/TTF);
                # чёрная тень включена — читаемость над ярким фоном
                put_text(frame, c.label_text(), (10, y), size_px=22,
                         color=(255, 255, 255))
                y += 34

        if status_text:
            w = frame.shape[1]
            tw = text_width(status_text, size_px=16)
            put_text(frame, status_text, (max(0, w - tw - 10), 12),
                     size_px=16, color=(0, 255, 255))
        return frame


# ---------------------------------------------------------------------------
# GUI-плеер (окно — только здесь)
# ---------------------------------------------------------------------------

class GuiPlayer:
    """Воспроизведение конвейера в окне OpenCV.

    :param pipeline: готовый (или ещё не ``build()``'енный)
        :class:`~visio_people_counter.pipeline.Pipeline` — плеер сам вызывает
        ``build()`` при старте, обрабатывает кадры через публичные компоненты
        (detector/tracker/counters/event_log), как ``Pipeline.step``, но
        дополнительно сохраняет blobs/mask для overlay.
    :param speed: начальная скорость воспроизведения (0.25..8; зажимается).
    :param window_name: название окна.

    Клавиши: **пробел** — пауза/далее; **q**/**ESC** — выход; **+**/**=**,
    **-**/**−** — скорость ×1.5 / ÷1.5 (в пределах 0.25..8); **`,`**/**`.`** —
    масштаб ОТОБРАЖЕНИЯ пресеты 0.5/1/1.5/2 (только экран, обработка не меняется).
    """

    #: Имя окна — только ASCII: GNOME/GTK использует заголовок для имени файла
    #: в диалогах сохранения, и кириллица там превращается в «_».
    DEFAULT_WINDOW = ("visio-people-counter "
                      "[space]=pause [q/ESC]=quit [+/-]=speed [,/.]=scale")

    def __init__(self, pipeline: Pipeline, speed: float = 1.0,
                 window_name: str | None = None,
                 initial_scale: float = 1.0) -> None:
        if initial_scale <= 0:
            raise ValueError(f"initial_scale должен быть > 0, получено {initial_scale!r}")
        self.pipeline = pipeline
        self.cfg: Config = pipeline.cfg
        self.speed: float = clamp_speed(speed)
        self.window_name = window_name or self.DEFAULT_WINDOW
        self.overlay = GuiOverlay(self.cfg.debug)
        #: масштаб ОТОБРАЖЕНИЯ (клавиши `,`/`.`; старт — initial_scale/--scale):
        #: применяем только перед imshow, обработка/детекция — всегда в исходном разрешении
        self.scale: float = initial_scale
        self._stop = False
        self._paused = False
        self._last_view: np.ndarray | None = None  # удержание кадра в паузе

    # ------------------------------------------------------------------ headless
    @staticmethod
    def available() -> bool:
        """Можно ли открыть окно в этой среде? Чистая проверка, БЕЗ создания окна.

        Критерий: задан ``DISPLAY`` или ``WAYLAND_DISPLAY`` И сборка OpenCV
        содержит GUI-бэкенд (Qt/GTK/Cocoa) по ``cv2.getBuildInformation()``.
        :returns: True/False; исключений не бросает.
        """
        try:
            if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
                return False
            info = cv2.getBuildInformation().upper()
            for line in info.splitlines():
                s = line.strip()
                # строка "GUI: QT5/GTK3/..." — перечислены доступные бэкенды
                if s.startswith("GUI:"):
                    value = s.split(":", 1)[1].strip()
                    if value and value != "NO":
                        return True
                # отдельные строки-флаги (Qt5: YES, GTK+: YES, Cocoa: YES...)
                for key in ("QT5", "QT6", "GTK", "COCOA"):
                    if s.startswith(key) and "YES" in s:
                        return True
            return False
        except Exception:
            # если getBuildInformation недоступен — верим только env-переменным
            return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    @staticmethod
    def unavailable_reason() -> str:
        """Человекочитаемая причина недоступности GUI (для CLI)."""
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            return ("нет дисплея для GUI-режима: переменные DISPLAY и "
                    "WAYLAND_DISPLAY не заданы (headless-окружение). "
                    "Запустите без --gui — headless-подсчёт работает без окон.")
        return ("GUI-бэкенд OpenCV недоступен (в cv2.getBuildInformation() нет "
                "Qt/GTK/Cocoa, вероятно opencv-python-headless). "
                "Запустите без --gui — headless-подсчёт работает без окон.")

    # ------------------------------------------------------------------ keys
    def _next_scale(self, direction: int) -> float:
        """Следующий пресет масштаба отображения (логика в calibrate.next_scale)."""
        from .calibrate import next_scale   # ленивый импорт: calibrate импортирует gui (цикл)
        return next_scale(self.scale, direction)

    def _handle_key(self, key: int) -> None:
        """Клавиша cv2.waitKey (& 0xFF). Чистая логика — тестируется без окна."""
        if key in (ord("q"), ord("Q"), 27):          # q / ESC — выход
            self._stop = True
        elif key == 32:                               # пробел — пауза/продолжить
            self._paused = not self._paused
        elif key in (ord("+"), ord("="), 107):        # + / = — быстрее ×1.5
            self.speed = clamp_speed(self.speed * 1.5)
        elif key in (ord("-"), 109):                  # - — медленнее ÷1.5
            self.speed = clamp_speed(self.speed / 1.5)
        elif key == ord(","):                         # `,` — масштаб отображения: уменьшить
            self.scale = self._next_scale(-1)
        elif key == ord("."):                         # `.` — масштаб отображения: увеличить
            self.scale = self._next_scale(1)

    def _scaled_view(self, view: np.ndarray) -> np.ndarray:
        """Кадр для imshow: resize только при scale != 1.0 (обработка не меняется)."""
        if self.scale == 1.0 or view is None or view.size == 0:
            return view
        interp = cv2.INTER_AREA if self.scale < 1.0 else cv2.INTER_LINEAR
        h, w = view.shape[:2]
        return cv2.resize(view, (max(1, int(round(w * self.scale))),
                                 max(1, int(round(h * self.scale)))),
                          interpolation=interp)

    def _status_text(self, proc_fps: float) -> str:
        txt = (f"speed={self.speed:.2f}x  proc={proc_fps:.0f}fps  "
               f"scale={self.scale:g}x")
        if self._paused:
            txt += "   [ПАУЗА]"
        return txt

    # ------------------------------------------------------------------ run
    def _process_frame(self, frame) -> tuple[list[Blob], list[TrackedObject]]:
        """Один кадр через компоненты pipeline (аналог Pipeline.step + blobs/mask)."""
        pipe = self.pipeline
        blobs = pipe.detector.detect(frame.image, pipe.size_profile)
        objects = pipe.tracker.update(blobs)
        # min_lifetime_frames: в счётчики пускаем только треки, подтверждённые
        # минимум N кадров подряд (анти-вспышка; те же правила, что в Pipeline.step)
        min_life = int(pipe.cfg.objects.min_lifetime_frames)
        countable = [o for o in objects if o.track_id == -1 or o.age_frames >= min_life]
        events = []
        for c in pipe.counters:
            events.extend(c.update(countable, frame.t_wall,
                                   t_video=frame.t_video, frame_index=frame.index))
        # markdown-отчёт (задача 10): те же данные, что Pipeline.step копит в headless
        pipe.report_events.extend(events)
        pipe.frames_processed += 1
        if events:
            pipe.event_log.log_events(events)
            for ev in events:
                _emit(f"СОБЫТИЕ {ev.counter_id} {ev.direction} track={ev.track_id} "
                      f"@ ({ev.x_px:.0f},{ev.y_px:.0f}) frame={ev.frame_index}")
        return blobs, objects

    def run(self) -> int:
        """Цикл окна до q/ESC/EOF источника. :returns: 0 — корректное завершение.

        :raises RuntimeError: GUI недоступен (нет дисплея/бэкенда) — CLI должен
            проверить :meth:`available` заранее и печатать ошибку с rc=1.
        """
        if not self.available():
            raise RuntimeError(self.unavailable_reason())

        pipe = self.pipeline
        if pipe.source is None:
            pipe.build()
        src = pipe.source
        fps = float(src.fps or 0.0)
        if fps <= 0:
            fps = float(self.cfg.processing.effective_fps or 25.0)
        _emit(f"GUI-режим: окно {self.window_name!r}, speed={self.speed:g}x "
              f"(+/- ×/÷1.5, ,/. масштаб 0.5-2 только экран, space — пауза, q/ESC — выход)")

        last_proc_dt = 0.0
        try:
            while not self._stop:
                if self._paused:
                    # пауза: не читаем новые кадры, держим последний на экране
                    if self._last_view is not None:
                        cv2.imshow(self.window_name, self._scaled_view(self._last_view))
                    key = cv2.waitKey(1) & 0xFF
                    self._handle_key(key)
                    continue

                frame = src.read()
                if frame is None:
                    # разрыв потока (не EOF): read() внутри уже отспал backoff
                    if isinstance(src, FfmpegPipeSource) and not src.exhausted:
                        time.sleep(0.1)
                        continue
                    break  # EOF файла / источник исчерпан

                t0 = time.monotonic()
                blobs, objects = self._process_frame(frame)
                mask = getattr(pipe.detector, "last_mask", None)
                proc_dt = max(1e-6, time.monotonic() - t0)
                last_proc_dt = 0.9 * last_proc_dt + 0.1 * proc_dt if last_proc_dt else proc_dt

                view = self.overlay.draw(
                    frame.image, objects=objects, blobs=blobs, mask=mask,
                    counters=pipe.counters,
                    status_text=self._status_text(1.0 / max(1e-6, last_proc_dt)))
                # отладочные кадры с overlay (cfg.debug.save_debug_frames_dir)
                dbg = self.cfg.debug
                if dbg.save_debug_frames_dir and \
                        frame.index % max(1, int(dbg.debug_frame_step)) == 0:
                    save_debug_frame(view, dbg.save_debug_frames_dir, frame.index)
                self._last_view = view
                # масштаб ОТОБРАЖЕНИЯ — только перед imshow (debug-кадры и обработка
                # остаются в исходном разрешении)
                cv2.imshow(self.window_name, self._scaled_view(view))
                key = cv2.waitKey(1) & 0xFF
                self._handle_key(key)

                # темп: (1/fps) / speed минус время обработки кадра
                if fps > 0:
                    target = (1.0 / fps) / self.speed - proc_dt
                    if target > 0:
                        time.sleep(target)
        finally:
            cv2.destroyAllWindows()
            # markdown-отчёт (задача 10) — как в headless: до close(), нужны fps/duration
            pipe._write_report("окно закрыто (GUI)")
            pipe.close()
        _emit("GUI-режим: окно закрыто")
        return 0
