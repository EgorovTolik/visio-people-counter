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

import math
import os
import queue
import sys
import threading
import time

import cv2
import numpy as np

from .config import Config, DebugConfig
from .pipeline import Pipeline, counting_active, save_debug_frame
from .line_counter import BaseCounter
from .text_overlay import put_text, text_width
from .motion_detector import Blob
from .tracker_adapter import TrackedObject
from .video_source import FfmpegPipeSource

#: допустимый диапазон скорости воспроизведения (CLI --speed и клавиши +/-).
MIN_SPEED = 0.25
#: верхнее ограничение скорости — очень большое (фактически без ограничения);
#: клавиши +/- ограничиваются им, CLI --speed принимает любое > 0.
MAX_SPEED = 1e6

#: задача 14 — авто-масштаб окна при маленьком кадре (маленький ROI):
#: минимальный «размер окна» для доступного UI (панель кнопок + статусная строка)
MIN_UI_W, MIN_UI_H = 640, 320
#: потолок авто-увеличения
MAX_AUTO_SCALE = 25.0


def auto_ui_scale(src_w: int, src_h: int, user_scale: float) -> float:
    """Масштаб ОТОБРАЖЕНИЯ: максимум(user_scale, того что вписывает кадр в MIN_UI_*).

    Задача 14: при маленьком кадре (маленький ROI) автоматически поднять масштаб
    отображения так, чтобы окно было ≥ ~640×320 и UI оставался доступным.

    - кадр уже больше минимумов по обеим осям → user_scale без изменений;
    - иначе required = max(MIN_UI_W/src_w, MIN_UI_H/src_h), округлённый ВВЕРХ
      до 0.1, но не выше MAX_AUTO_SCALE и не ниже user_scale
      (пользовательское --scale — минимум).

    :param src_w: ширина кадра в px (ROI-размер после кропа); <= 0 → user_scale.
    :param src_h: высота кадра в px; <= 0 → user_scale.
    :param user_scale: пользовательский масштаб (--scale / initial_scale).
    :returns: масштаб отображения в диапазоне [user_scale, MAX_AUTO_SCALE].
    """
    if src_w <= 0 or src_h <= 0:
        return float(user_scale)
    needed = max(MIN_UI_W / float(src_w), MIN_UI_H / float(src_h))
    rounded = math.ceil(needed * 10.0 - 1e-9) / 10.0   # вверх до 0.1 (без FP-шума)
    return min(MAX_AUTO_SCALE, max(float(user_scale), rounded))


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


# ВАЖНО: вызывается НЕ при импорте, а лениво в cv2-драйверах (GuiPlayer.run,
# calibrate.run_calibration) — иначе QT_PLUGIN_PATH/QT_QPA_PLATFORMTHEME попали бы
# в процесс и сломали PySide6-бэкенд (см. gui_qt._sanitize_env_for_pyside).


def clamp_speed(speed: float) -> float:
    """Привести скорость к [MIN_SPEED, MAX_SPEED] (верхний предел очень большой)."""
    return max(MIN_SPEED, min(MAX_SPEED, float(speed)))


def counting_status_text(frame_index: int,
                         frame_start: Optional[int] = None,
                         frame_end: Optional[int] = None) -> str:
    """Строка статуса подсчёта для GUI-окна (чистая функция, без окна).

    * до старта интервала — ``подсчёт: ждём кадр N`` (N = frame_start);
    * внутри интервала (или при отсутствии интервала) — ``""`` (как раньше);
    * после окончания — ``подсчёт завершён (до кадра M)`` (M = frame_end).

    Номера 0-based; границы включительно. Видео продолжает проигрываться.
    """
    if frame_start is not None and frame_index < frame_start:
        return f"подсчёт: ждём кадр {frame_start}"
    if frame_end is not None and frame_index > frame_end:
        return f"подсчёт завершён (до кадра {frame_end})"
    return ""


def _emit(msg: str) -> None:
    print(f"[gui] {msg}", flush=True)


# ---------------------------------------------------------------------------
# counters_status_line — чистая строка счётчиков для Qt-статусбара (задача 20)
# ---------------------------------------------------------------------------

def counters_status_line(counters: list[BaseCounter]) -> str:
    """Сводка счётчиков в одну строку для Qt-статусбара.

    Формат: «счётчики: <id>: in=N out=M | <id2>: in=N out=M» (для ``total``-
    режимов — «<id>: total=N»). Строки счётчиков соединяются ``" | "``;
    пустая строка, если counters нет. Значения берутся актуальные на момент
    вызова (``BaseCounter.label_text()`` + текущие in/out/total). Не рисует —
    только чистая строка для статусбара.
    """
    if not counters:
        return ""
    return "счётчики: " + " | ".join(c.label_text() for c in counters)


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
             status_text: str = "",
             with_text: bool = True) -> np.ndarray:
        """Нарисовать overlay на BGR-кадр (in-place) и вернуть его.

        :param frame: кадр (h, w, 3) uint8 — модифицируется на месте.
        :param objects: треки текущего кадра (``cfg.debug.show_bboxes``).
        :param blobs: необработанные blob'ы до трекера (``show_all_blobs``).
        :param mask: маска движения 0/255 того же размера (``show_mask`` → микс 50/50).
        :param counters: счётчики (``show_counters``): линия/зона через
            ``counter.draw`` + крупные текущие in/out в левом верхнем углу.
        :param status_text: строка статуса справа сверху (скорость, PAUSED...);
            рисуется на кадре только при with_text=True (в Qt-окне — в статусбаре).
        :param with_text: рисовать ли текст счётчиков (in/out) и status_text на
            кадре. False — оставить графику линии/зоны и blob/bbox/mask;
            задачи 20.
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

        # 4) счётчики: линия/зона (line_counter.draw) — всегда; крупные in/out
        #    в углу — только при with_text=True (в Qt-окне текст переносится
        #    в статусбар, задачи 20)
        if self.debug.show_counters and counters:
            for c in counters:
                c.draw(frame)
            if with_text:
                y = 36
                for c in counters:
                    # put_text: кириллические id счётчиков тоже отрисовываются (PIL/TTF);
                    # чёрная тень включена — читаемость над ярким фоном
                    put_text(frame, c.label_text(), (10, y), size_px=22,
                             color=(255, 255, 255))
                    y += 34

        if with_text and status_text:
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

    Публичный API для внешнего драйвера окна (Qt, задача 17): :meth:`handle_key`
    (обёртка над ``_handle_key``), :meth:`tick` (один шаг кадра без imshow;
    cv2-цикл :meth:`run` переписан через него) и :meth:`finalize` (финальная
    сводка + markdown-отчёт, идемпотентна).
    """

    #: Имя окна — только ASCII: GNOME/GTK использует заголовок для имени файла
    #: в диалогах сохранения, и кириллица там превращается в «_».
    DEFAULT_WINDOW = ("visio-people-counter "
                      "[space]=pause [q/ESC]=quit [+/-]=speed [,/.]=scale")

    def __init__(self, pipeline: Pipeline, speed: float = 1.0,
                 window_name: str | None = None,
                 initial_scale: float = 1.0,
                 use_threads: bool = False, queue_size: int = 50) -> None:
        if initial_scale <= 0:
            raise ValueError(f"initial_scale должен быть > 0, получено {initial_scale!r}")
        self.pipeline = pipeline
        self.cfg: Config = pipeline.cfg
        self.speed: float = clamp_speed(speed)
        self.window_name = window_name or self.DEFAULT_WINDOW
        self.overlay = GuiOverlay(self.cfg.debug)
        # thread-pipeline для GUI (задача 21): reader-поток + очередь кадров
        self.use_threads = bool(use_threads)
        self.queue_size = max(1, int(queue_size))
        self._frame_queue: Optional["queue.Queue"] = None
        self._reader_stop = threading.Event()
        self._reader_thread: Optional[threading.Thread] = None
        # задачи 20: рисовать ли текст счётчиков/status_text на кадре. cv2-run и
        # headless — True (текст в кадр); Qt count-окно ставит False (текст в
        # статусбар, на кадре только гракция линии/зоны). Сеттер извне.
        self.overlay_with_text = True
        #: масштаб ОТОБРАЖЕНИЯ (клавиши `,`/`.`; старт — initial_scale/--scale):
        #: применяем только перед imshow, обработка/детекция — всегда в исходном разрешении
        self.scale: float = initial_scale
        self._stop = False
        self._paused = False
        self._last_view: np.ndarray | None = None  # удержание кадра в паузе
        # задача 17 — API для внешнего драйвера (Qt-окно): состояние последнего шага
        self.frame_index: int | None = None   # индекс последнего обработанного кадра
        self.proc_dt: float = 0.0             # время обработки последнего кадра (pacing)
        self.last_message: str = ""           # последнее событие (для статусбара Qt-окна)
        self._proc_ema: float = 0.0           # EMA времени обработки (проц-fps в статусе)
        self._finalized: bool = False         # finalize() идемпотентен

    # ------------------------------------------------------------- публичный API (задача 17)
    def handle_key(self, key: int) -> None:
        """Публичная обёртка над :meth:`_handle_key` (коды cv2 ``waitKey & 0xFF``).

        Для внешнего драйвера окна (Qt-кнопки/клавиатура): логика та же —
        q/ESC — выход, пробел — пауза, +/- — скорость, `,`/`.` — масштаб.
        """
        self._handle_key(key)

    def status_text(self, proc_fps: float, frame_index: int | None = None) -> str:
        """Публичный alias :meth:`_status_text` (строка статуса для статусбара)."""
        return self._status_text(proc_fps, frame_index=frame_index)

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

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        """Формат hh:mm:ss из секунд."""
        s = max(0, int(seconds))
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{sec:02d}"

    def _status_text(self, proc_fps: float, frame_index: int | None = None) -> str:
        txt = (f"speed={self.speed:.2f}x  proc={proc_fps:.0f}fps  "
               f"scale={self.scale:g}x")
        if self._paused:
            txt += "   [ПАУЗА]"
        if frame_index is not None:
            # время/продолжительность: hh:mm:ss
            src = getattr(self.pipeline, 'source', None)
            fps = getattr(src, 'fps', 0) if src else 0
            duration = getattr(src, 'duration', 0) if src else 0
            if fps and frame_index >= 0:
                cur_t = frame_index / fps
                txt += f"   {self._fmt_time(cur_t)}"
                if duration > 0:
                    txt += f"/{self._fmt_time(duration)}"
            # статус подсчёта в интервале кадров (processing.frame_start/frame_end)
            extra = counting_status_text(
                frame_index, self.cfg.processing.frame_start, self.cfg.processing.frame_end)
            if extra:
                txt += f"   {extra}"
        return txt

    # ------------------------------------------------------------- tick/finalize (задача 17)
    def tick(self) -> Optional[np.ndarray]:
        """Один шаг «кадра цикла» БЕЗ окна (используется cv2-run и Qt-окном).

        * пауза → вернуть последний сохранённый view (как cv2-run: удержание кадра);
        * иначе — прочитать кадр из pipeline, ``_process_frame``, overlay,
          ``_scaled_view``; источник открывается лениво (``build()`` + авто-масштаб);
        * EOF / stop → ``None``.

        :returns: кадр для отображения (с учётом ``self.scale``) или ``None`` — завершить.
        """
        pipe = self.pipeline
        if pipe.source is None:
            pipe.build()
            # задача 14: маленький кадр → авто-увеличение масштаба отображения
            # (идемпотентно/монотонно — run() применяет тот же вызов перед циклом)
            self.scale = auto_ui_scale(pipe.source.width, pipe.source.height, self.scale)
            # GUI: seek к time_start/frame_start — просмотр начинается с нужного места
            fs = self.cfg.processing.frame_start
            if fs and fs > 0:
                fps_val = pipe.source.fps or 0
                if fps_val > 0:
                    t_seek = fs / fps_val
                    if pipe.source.seek(t_seek):
                        _emit(f"GUI: перемотка к {t_seek:.1f}s (frame_start={fs})")
            # thread-pipeline: запускаем reader-поток после build() (один раз)
            if self.use_threads and self._frame_queue is None:
                self._start_reader()
        if self._paused:
            return self._scaled_view(self._last_view) if self._last_view is not None else None
        while not self._stop:
            if self._frame_queue is not None:
                # thread-pipeline: кадр из очереди; sentinel None = EOF
                try:
                    frame = self._frame_queue.get(timeout=1.0)
                except queue.Empty:
                    if (self._reader_thread is not None
                            and not self._reader_thread.is_alive()):
                        return None   # reader умер и очередь пуста → EOF
                    continue
                if frame is None:     # sentinel: EOF
                    return None
            else:
                frame = pipe.source.read()
                if frame is None:
                    # разрыв потока (не EOF): read() внутри уже отспал backoff
                    if isinstance(pipe.source, FfmpegPipeSource) and not pipe.source.exhausted:
                        time.sleep(0.1)
                        continue
                    return None   # EOF файла / источник исчерпан
            break
        if self._stop:
            return None

        t0 = time.monotonic()
        blobs, objects = self._process_frame(frame)
        mask = getattr(pipe.detector, "last_mask", None)
        proc_dt = max(1e-6, time.monotonic() - t0)
        self.proc_dt = proc_dt
        self._proc_ema = 0.9 * self._proc_ema + 0.1 * proc_dt if self._proc_ema else proc_dt
        self.frame_index = frame.index

        view = self.overlay.draw(
            frame.image, objects=objects, blobs=blobs, mask=mask,
            counters=pipe.counters,
            status_text=self.status_text(1.0 / max(1e-6, self._proc_ema),
                                         frame_index=frame.index),
            with_text=self.overlay_with_text)
        # отладочные кадры с overlay (cfg.debug.save_debug_frames_dir)
        dbg = self.cfg.debug
        if dbg.save_debug_frames_dir and \
                frame.index % max(1, int(dbg.debug_frame_step)) == 0:
            save_debug_frame(view, dbg.save_debug_frames_dir, frame.index)
        self._last_view = view
        # масштаб ОТОБРАЖЕНИЯ — только для вывода (обработка — в исходном разрешении)
        return self._scaled_view(view)

    def finalize(self, reason: str) -> None:
        """Финальное завершение: финальная сводка + markdown-отчёт с причиной + close.

        Идемпотентна (флаг ``_finalized``): cv2-run и Qt-окно вызывают её в своём
        ``finally``, повторные вызовы игнорируются (отчёт записывается один раз).
        """
        if self._finalized:
            return
        self._finalized = True
        self._stop_reader()
        pipe = self.pipeline
        # markdown-отчёт — до close(): нужны fps/duration источника.
        # Сводка — только если конвейер собран (event_log создан в build());
        # закрытие окна раньше первого кадра (source None) — без сводки.
        if pipe.event_log is not None:
            pipe._print_final(reason)
        pipe._write_report(reason)
        pipe.close()

    # ------------------------------------------------------------------ thread-pipeline (GUI)
    def _start_reader(self) -> None:
        """Запустить reader-поток для GUI: читает кадры из pipe.source в очередь.

        Вызывается один раз после build(). Thread safety: source.read() — только
        здесь; detector/tracker/counters — только в main (tick()).
        """
        self._frame_queue = queue.Queue(maxsize=self.queue_size)
        src = self.pipeline.source

        def _target() -> None:
            while not self._reader_stop.is_set():
                frame = src.read()
                if frame is None:
                    if isinstance(src, FfmpegPipeSource) and not src.exhausted:
                        time.sleep(0.2)
                        continue
                    break
                try:
                    self._frame_queue.put(frame, timeout=5.0)  # type: ignore[union-attr]
                except queue.Full:
                    self._reader_stop.set()
                    break
            try:
                self._frame_queue.put(None, timeout=5.0)       # sentinel EOF  # type: ignore[union-attr]
            except queue.Full:   # pragma: no cover
                pass

        self._reader_thread = threading.Thread(target=_target,
                                               name="vpc-gui-reader", daemon=True)
        self._reader_thread.start()

    def _stop_reader(self) -> None:
        """Остановить reader-поток (идемпотентно)."""
        if self._reader_stop is not None:
            self._reader_stop.set()
        if self._reader_thread is not None and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2.0)

    # ------------------------------------------------------------------ run
    def _process_frame(self, frame) -> tuple[list[Blob], list[TrackedObject]]:
        """Один кадр через компоненты pipeline (аналог Pipeline.step + blobs/mask)."""
        pipe = self.pipeline
        p = pipe.cfg.processing
        # детектор — на КАЖДОМ кадре (обучение фона MOG2 до старта интервала);
        # трекер/счётчики активны только внутри processing.frame_start..frame_end.
        blobs = pipe.detector.detect(frame.image, pipe.size_profile)
        active = counting_active(frame.index, p.frame_start, p.frame_end)
        if active:
            objects = pipe.tracker.update(blobs)
        else:
            # вне интервала: трекер НЕ обновляем (чистый список), счётчики не трогаем
            objects = []
        events = []
        if active:
            # min_lifetime_frames: в счётчики пускаем только треки, подтверждённые
            # минимум N кадров подряд (анти-вспышка; те же правила, что в Pipeline.step)
            min_life = int(pipe.cfg.objects.min_lifetime_frames)
            countable = [o for o in objects if o.track_id == -1 or o.age_frames >= min_life]
            for c in pipe.counters:
                events.extend(c.update(countable, frame.t_wall,
                                       t_video=frame.t_video, frame_index=frame.index))
        # markdown-отчёт (задача 10): те же данные, что Pipeline.step копит в headless
        pipe.report_events.extend(events)
        pipe.frames_processed += 1
        if events:
            pipe.event_log.log_events(events)
            for ev in events:
                # процент обработки (аналог pipeline.py; учитывает frame_end)
                src = getattr(pipe, 'source', None)
                pct = ""
                if src is not None and getattr(src, 'fps', 0) and getattr(src, 'duration', 0):
                    video_total = int(src.duration * src.fps)
                    fe = pipe.cfg.processing.frame_end
                    total = min(fe + 1, video_total) if fe is not None else video_total
                    if total > 0:
                        pct = f" ({pipe.frames_processed}/{total}, " \
                              f"{100.0 * pipe.frames_processed / total:.2f}%)"
                msg = (f"СОБЫТИЕ {ev.counter_id} {ev.direction} track={ev.track_id} "
                       f"@ ({ev.x_px:.0f},{ev.y_px:.0f}) frame={ev.frame_index}{pct}")
                _emit(msg)
                self.last_message = msg   # для статусбара Qt-окна (задача 17)
        return blobs, objects

    def run(self) -> int:
        """Цикл окна до q/ESC/EOF источника. :returns: 0 — корректное завершение.

        :raises RuntimeError: GUI недоступен (нет дисплея/бэкенда) — CLI должен
            проверить :meth:`available` заранее и печатать ошибку с rc=1.
        """
        if not self.available():
            raise RuntimeError(self.unavailable_reason())

        apply_qt_env()   # только для cv2-окна (до создания HighGUI/QApplication)
        pipe = self.pipeline
        if pipe.source is None:
            pipe.build()
        src = pipe.source
        # задача 14: маленький кадр (маленький ROI) → авто-увеличение масштаба
        # отображения, чтобы UI был доступен; user --scale — минимум; статус покажет scale=…x
        self.scale = auto_ui_scale(src.width, src.height, self.scale)
        fps = float(src.fps or 0.0)
        if fps <= 0:
            fps = float(self.cfg.processing.effective_fps or 25.0)
        _emit(f"GUI-режим: окно {self.window_name!r}, speed={self.speed:g}x "
              f"(+/- ×/÷1.5, ,/. масштаб 0.5-2 только экран, space — пауза, q/ESC — выход)")

        try:
            # задача 17: цикл переписан через публичный tick() (тот же pacing) —
            # та же логика, что у Qt-окна; поведение cv2-варианта сохранено.
            while not self._stop:
                view = self.tick()
                if view is None:
                    break   # EOF / источник исчерпан / stop
                cv2.imshow(self.window_name, view)
                key = cv2.waitKey(1) & 0xFF
                self._handle_key(key)

                # темп: (1/fps) / speed минус время обработки кадра (в паузе — без сна,
                # как раньше: waitKey(1) даёт реалтайм-задержку)
                if not self._paused and fps > 0:
                    target = (1.0 / fps) / self.speed - self.proc_dt
                    if target > 0:
                        time.sleep(target)
        finally:
            cv2.destroyAllWindows()
            # финальная сводка + markdown-отчёт (до close(): нужны fps/duration)
            self.finalize("окно закрыто (GUI)")
        _emit("GUI-режим: окно закрыто")
        return 0
