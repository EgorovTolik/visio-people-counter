"""Контроллер калибровки — «мозг» окна калибровки, независимый от способа показа (задача 15).

Вся логика «кадр за кадром» из :func:`calibrate.run_calibration` вынесена в класс
:class:`CalibrationController`, чтобы ОДНОВРЕМЕННО работать под:

* cv2-окном — тонкий драйвер :func:`calibrate.run_calibration` (поведение без изменений);
* Qt-окном (PySide6) — задача 16: тот же контроллер, другой драйвер.

Чистые функции («клик → конфиг», кнопки, уведомления) и :class:`CalibrationState`
ОСТАЮТСЯ в ``calibrate.py`` — контроллер только импортирует их; ничего не переписано.

Цикл драйвера::

    ctrl = CalibrationController(cfg, save_to=...)
    ctrl.open()                    # Pipeline + кэш кадров + seek к frame_start
    while True:
        img = ctrl.step()          # BGR-кадр ИСХОДНОГО разрешения со ВСЕМИ оверлеями
        if img is None:            # quit (q/ESC) или нет ни одного кадра в кэше
            break
        показать(img, scale=ctrl.scale)   # resize под ctrl.scale — обязанность драйвера
        for key in события_клавиш:        # координаты кликов/курсора — В СИСТЕМЕ ИСХОДНОГО
            ctrl.on_key(key)                 # КАДРА (после unscale_mouse), как в cv2-драйвере
    ctrl.close()

События драйвера → контроллер: :meth:`on_click` (клик мышью), :meth:`on_key` (клавиша),
:meth:`on_button` (прямое нажатие кнопки панели по имени — для Qt),
:meth:`on_mouse_move` (позиция курсора для live-превью «мерки роста»).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .config import Config, ConfigError, describe_roi
from .gui import auto_ui_scale
from .text_overlay import put_text
from .motion_detector import MotionDetector
from .pipeline import Pipeline
from .video_source import FfmpegPipeSource, VideoSourceError
from .calibrate import (
    DEFAULT_CACHE_FRAMES,
    MESSAGE_TTL_SECONDS,
    MIN_ROI_FRACTION,
    Button,
    CalibrationState,
    TimeInputBuffer,
    _draw_calibration,
    apply_calibration,
    clamp_seek_time,
    cycle_counter_id,
    delete_current_counter,
    draw_all_counters,
    draw_notification,
    draw_top_panel,
    ensure_counter_kind,
    filter_expired_messages,
    frame_start_seek_time,
    handle_size_click,
    hit_button,
    load_counter_into_state,
    make_new_counter_id,
    next_scale,
    undo_last_size_point,
)


class CalibrationController:
    """«Мозг» калибровки: источник + кэш кадров + состояние + события (без окна).

    :param cfg: загруженный :class:`Config` (video.path уже задан); меняется in-place
        при [a] и принятии ROI.
    :param counter_id: id редактируемого счётчика на старте (по умолчанию ``main_line``).
    :param cache_frames: сколько кадров держать в кэше листа ([n/p]) и после seek; >= 1.
    :param initial_scale: старт масштаба ОТОБРАЖЕНИЯ (--scale); обработка — всегда в
        исходном разрешении кадра; малый кадр → авто-увеличение (задача 14).
    :param window_name: имя окна (использует драйвер: cv2.namedWindow / заголовок Qt).
    :param save_to: куда писать конфиг при [a] (:func:`calibrate.calibration_save_target`
        считает цель в драйвере); None — сохранение недоступно (уведомление).

    Свойства для драйвера (статусбар/название окна): :attr:`frame_index`, :attr:`scale`,
    :attr:`quit_requested`, :attr:`messages`, :attr:`width`, :attr:`height`.
    """

    def __init__(self, cfg: Config, *, counter_id: str = "main_line",
                 cache_frames: int = DEFAULT_CACHE_FRAMES,
                 initial_scale: float = 1.0, window_name: str = "calibrate",
                 save_to: Optional[str | Path] = None) -> None:
        if cache_frames < 1:
            raise ValueError(f"cache_frames: ожидалось >= 1, получено {cache_frames!r}")
        self.cfg = cfg
        self.window_name = window_name
        self.cache_frames = cache_frames
        self.initial_scale = float(initial_scale)
        self._save_target: Optional[Path] = Path(save_to) if save_to is not None else None

        self.state = CalibrationState(counter_id=counter_id)
        self.mask_on = False
        self.show_all = False   # режим «все»: все счётчики из конфига с размерами (только визуал)
        self._scale = float(initial_scale)   # фактический масштаб — после open() (авто-увеличение)

        # источник и кэш кадров (заполняются в open())
        self._pipe: Optional[Pipeline] = None
        self._detector: Optional[MotionDetector] = None   # для 'm' — live-маска движения
        self.w = 0
        self.h = 0
        self.frames: list[np.ndarray] = []
        self.masks: list[Optional[np.ndarray]] = []
        #: ГЛОБАЛЬНЫЕ номера кадров (0-based) — для статуса «кадр N» и seek-позиции
        self.frame_indices: list[int] = []
        self._idx = 0
        # seek (клавиша [t]/кнопка «время») возможен только для файла: у ffmpeg-пайпа
        # (HLS/URL) случайного доступа нет — базовый VideoSource.seek() возвращает False.
        self.seek_supported = False

        # режим ввода времени (seek-секунды)
        self.time_buf = TimeInputBuffer()
        self.time_input_active = False

        # сообщения / прочие флаги
        self.saved_once = False
        self._quit_requested = False
        #: рисовать ли НА КАДРЕ старую панель (кнопки + статусная строка):
        #: cv2-окно — True; Qt-окно — False (есть настоящий QToolBar/статусбар,
        #: наложенная панель только мешает). Управляет step() и hit-test'ом в on_click.
        self.frame_ui = True
        #: активные уведомления ``(текст, expires_at — time.monotonic)``; живут MESSAGE_TTL_SECONDS
        self.pending_msgs: list[tuple[str, float]] = []
        #: раскладка панели кнопок (обновляется каждым step() для hit-test кликов)
        self.buttons: list[Button] = []
        #: позиция курсора в системе ИСХОДНОГО кадра (on_mouse_move) или None;
        #: нужна для live-превью pending «мерки роста»
        self._mouse: Optional[tuple[int, int]] = None

    # ------------------------------------------------------------------ свойства
    @property
    def frame_index(self) -> int:
        """Глобальный номер текущего кадра (0-based; совпадает со статусом «кадр N»)."""
        if self._idx < len(self.frame_indices):
            return self.frame_indices[self._idx]
        return self._idx

    @property
    def scale(self) -> float:
        """Масштаб ОТОБРАЖЕНИЯ (только экран; `,`/`.` — пресеты 0.5/1/1.5/2)."""
        return self._scale

    @scale.setter
    def scale(self, value: float) -> None:
        self._scale = float(value)

    @property
    def quit_requested(self) -> bool:
        """Запрошен выход (q/ESC вне режима «ROI»); step() после этого возвращает None."""
        return self._quit_requested

    @property
    def messages(self) -> list[str]:
        """Активные (не истёкшие) уведомления БЕЗ отрисовки — для Qt-статусбара."""
        return [m[0] for m in filter_expired_messages(self.pending_msgs, time.monotonic())]

    @property
    def width(self) -> int:
        """Ширина обрабатываемого (ROI-)кадра в px; 0 — до open()."""
        return self.w

    @property
    def height(self) -> int:
        """Высота обрабатываемого (ROI-)кадра в px; 0 — до open()."""
        return self.h

    # ------------------------------------------------------------------ жизненный цикл
    def _notify(self, msg: str) -> None:
        """Сообщение на экране: живёт MESSAGE_TTL_SECONDS секунд."""
        self.pending_msgs.append((msg, time.monotonic() + MESSAGE_TTL_SECONDS))

    def open(self) -> None:
        """Открыть источник (``Pipeline(cfg).build()``), загрузить первые
        ``cache_frames`` кадров и сделать seek к ``processing.frame_start`` (только файл) —
        как до выноса в контроллер. Печатает стартовые сообщения в stdout/stderr.

        :raises VideoSourceError: источник не открылся или не прочитан ни один кадр
            (сообщение уже напечатано в stderr).
        :raises ConfigError: ошибка конфига при сборке пайплайна.
        """
        pipe = None
        try:
            pipe = Pipeline(self.cfg).build()
        except (VideoSourceError, ConfigError) as e:
            if pipe is not None:
                pipe.close()  # build мог упасть после source.open() — не утекает ресурс
            print(f"calibrate: ошибка источника: {e}", file=sys.stderr)
            raise
        self._pipe = pipe
        self.w, self.h = pipe.source.width, pipe.source.height

        self._detector = MotionDetector(self.cfg)   # для 'm' — live-маска движения
        self.seek_supported = not isinstance(pipe.source, FfmpegPipeSource)

        # задача 14: маленький кадр (маленький ROI) → авто-увеличение окна для доступного UI;
        # user --scale — минимум; клавиши `,`/`.` продолжают работать поверх авто-значения
        scale_base = max(0.05, self.initial_scale)
        self.scale = auto_ui_scale(self.w, self.h, scale_base)
        if self.scale > scale_base:
            msg = (f"малое изображение {self.w}×{self.h} — окно увеличено до "
                   f"{self.scale:g}x для доступного UI")
            print(f"calibrate: {msg}")
            self._notify(msg)

        if not self._load_cache():   # начальная загрузка первых cache_frames кадров
            print("calibrate: ОШИБКА: не удалось прочитать ни одного кадра", file=sys.stderr)
            raise VideoSourceError("не удалось прочитать ни одного кадра")

        # processing.frame_start — сразу открыть этот кадр (задача 12): seek как [t],
        # только для файла; если fps/seek недоступны или кадр за пределами видео —
        # уведомление и старт с начала.
        fs_frame = self.cfg.processing.frame_start
        if fs_frame is not None:
            if not self.seek_supported:
                print("calibrate: processing.frame_start задан, но для HLS/URL "
                      "случайного доступа нет — переход пропущен")
            else:
                t0 = frame_start_seek_time(fs_frame, pipe.source.fps)
                if t0 is None:
                    print("calibrate: fps источника <= 0 — переход к кадру "
                          f"{fs_frame} (processing.frame_start) пропущен")
                else:
                    t0, _clamped = clamp_seek_time(
                        t0, getattr(pipe.source, "duration", 0.0),
                        self.cache_frames, pipe.source.fps)
                    if self._seek_to(t0):
                        self._notify(f"переход к кадру {fs_frame} (processing.frame_start)")
                    else:
                        self._notify(f"кадр {fs_frame} за пределами видео — старт с начала")

        # ROI задан во входном конфиге: окно сразу на ROI-виде (источник уже кропит)
        if self.cfg.processing.roi is not None:
            print(f"calibrate: ROI применён из конфига: {describe_roi(self.cfg.processing.roi)} "
                  f"(окно показывает ROI-вид; правка — клавиша r / кнопка «roi»)")
            self._notify(f"ROI применён из конфига: {describe_roi(self.cfg.processing.roi)}")

        save_txt = f" в {self._save_target}" if self._save_target is not None else ""
        print(f"calibrate: загрузил {len(self.frames)} кадр(ов) {self.w}x{self.h}, "
              f"counter-id={self.state.counter_id!r}; [n/p] — листать, "
              f"{'[t] — время (seek), ' if self.seek_supported else ''}[a] — сохранить{save_txt}")

    def close(self) -> None:
        """Закрыть источник (идемпотентно). Вызывать ПОСЛЕ закрытия окна драйвером."""
        if self._pipe is not None:
            self._pipe.close()
            self._pipe = None

    # ------------------------------------------------------------------ кадр цикла
    def step(self) -> Optional[np.ndarray]:
        """Один кадр цикла: прочитать из кэша, нарисовать ВСЕ оверлеи и вернуть
        готовый BGR-кадр ИСХОДНОГО разрешения (масштабирование под :attr:`scale` —
        обязанность драйвера перед показом).

        :returns: ``None`` — запрошен выход (q/ESC) или в кэше нет ни одного кадра;
            иначе ``np.ndarray`` (h, w, 3) с панелью, уведомлениями, оверлеями режимов.
        """
        if self._quit_requested or not self.frames:
            return None
        idx = self._idx
        img = _draw_calibration(self.frames[idx], self.state, self.mask_on,
                                self.masks[idx], self.w, self.h,
                                mouse_pos=self._mouse,
                                active_roi=self.cfg.processing.roi)
        # подпись ROI в статусной строке (текущее значение из конфига)
        roi_label = None
        if self.cfg.processing.roi is not None:
            roi_txt = describe_roi(self.cfg.processing.roi)
            roi_label = f"ROI: {roi_txt} (правка — r)" if self.state.mode == "roi" \
                else f"ROI: {roi_txt}"
        if self.frame_ui:
            self.buttons = draw_top_panel(img, self.state.counter_id, self.cfg.counters,
                                          self.state.mode, self.mask_on, show_all=self.show_all,
                                          scale=self.scale,
                                          frame_index=(self.frame_indices[idx]
                                                       if idx < len(self.frame_indices) else None),
                                          roi_label=roi_label)
            panel_y = 72   # сообщения — под строкой кнопок (панель заканчивается ~y=66)
        else:
            self.buttons = []   # Qt: панель не рисуем, клики по ней не ловим
            panel_y = 10        # уведомления — сразу сверху кадра
        # живут MESSAGE_TTL_SECONDS секунд
        self.pending_msgs[:] = filter_expired_messages(self.pending_msgs, time.monotonic())
        for i, (msg, _exp) in enumerate(self.pending_msgs):
            draw_notification(img, msg[:90], x=10, y=panel_y + 20 * i, size_px=14)
        if self.time_input_active:
            # буфер ввода времени дублируется на экране каждый кадр
            buf_str = "".join(str(d) for d in self.time_buf.digits) or "_"
            put_text(img, f"Время (сек): {buf_str} | Enter=OK, ESC/q=отмена",
                     (10, 92), size_px=16, color=(0, 255, 255))
        if self.show_all:
            # режим «все» — поверх всего: все линии/зоны из конфига с размерами;
            # текущий счётчик совпадает по цвету с редактируемым и не дублируется криво
            draw_all_counters(
                img, self.cfg.counters, self.w, self.h, highlight_id=self.state.counter_id,
                size_points=(self.state.size_points
                             or [list(p) for p in self.cfg.size_profile.control_points]))
        return img

    # ------------------------------------------------------------------ события драйвера
    def on_mouse_move(self, x: int, y: int) -> None:
        """Позиция курсора (координаты ИСХОДНОГО кадра — после ``unscale_mouse``):
        используется для live-превью pending «мерки роста» в :meth:`step`."""
        self._mouse = (int(x), int(y))

    def on_click(self, x: int, y: int) -> None:
        """Левый клик (координаты ИСХОДНОГО кадра — после ``unscale_mouse``).

        Сначала — hit-test панели кнопок (клик по кнопке НЕ передаётся в режим);
        иначе — логика текущего режима (:func:`calibrate.handle_size_click`),
        как до выноса.
        """
        if self.frame_ui:
            name = hit_button(self.buttons, x, y)
            if name is not None:
                self.on_button(name)
                return
        try:
            for msg in handle_size_click(self.state, x, y, self.w, self.h):
                self._notify(msg)
        except ValueError as e:
            self._notify(str(e))

    def on_button(self, name: str) -> None:
        """Прямое нажатие кнопки панели по имени (``"line"``, ``"zone"``, ``"size"``,
        ``"roi"``, ``"mask"``, ``"show_all"``, ``"prev"``, ``"next"``, ``"new_line"``,
        ``"new_zone"``, ``"delete"``, ``"undo_size"``, ``"time"``, ``"save"``) —
        для Qt-драйвера; то же действие, что клик по кнопке в cv2-окне."""
        if name == "line":
            self._set_draw_mode("line")
        elif name == "zone":
            self._set_draw_mode("zone")
        elif name == "size":
            self.state.set_mode("size")
        elif name == "roi":
            self._begin_roi_mode()
        elif name == "mask":
            self.mask_on = not self.mask_on
        elif name == "show_all":
            self.show_all = not self.show_all
        elif name == "prev":
            self._switch_counter(-1)
        elif name == "next":
            self._switch_counter(1)
        elif name == "new_line":
            self._new_counter("line")
        elif name == "new_zone":
            self._new_counter("zone")
        elif name == "delete":
            self._do_delete()
        elif name == "undo_size":
            self._do_undo_size_point()
        elif name == "time":
            self._begin_time_input()
        elif name == "save":
            self._do_save()

    def on_key(self, key: int) -> None:
        """Клавиша от драйвера (``cv2.waitKey(1) & 0xFF``; Qt — переводить в ASCII-код).

        То же распределение, что до выноса: режим ввода времени имеет приоритет
        (цифры/Enter/ESC/q/'-' → буфер); иначе l/z/s/r/m/v/n/p/x/b/[ ]/t/a/,/. и
        Enter (зона/size/ROI), q/ESC (выход; в режиме «ROI» — отмена рисования).
        """
        if self.time_input_active:
            # режим ввода времени имеет приоритет: цифры уходят только в буфер
            if key in (ord("q"), ord("Q"), 27):
                self.time_input_active = False
                self.time_buf.reset()
                self._notify("ввод времени отменён")
            elif key == 13:   # Enter — применить seek
                self._apply_time_seek()
            elif 48 <= key <= 57:    # цифры 0-9 → в буфер
                self.time_buf.feed_digit(key - 48)
            elif key in (ord("-"), 8, 127):   # '-' или Backspace (X11=127, Win=8)
                self.time_buf.backspace()
            return
        if key in (ord("q"), ord("Q"), 27):
            if self.state.mode == "roi":
                # ESC в режиме «ROI» — отмена рисования (не выход из окна)
                self.state.cancel_roi()
                self._notify("ROI: отменено (источник не переоткрывался)")
            else:
                self._quit_requested = True
        elif key == ord("l"):
            self._set_draw_mode("line")
        elif key == ord("z"):
            self._set_draw_mode("zone")
        elif key == ord("s"):
            self.state.set_mode("size")
        elif key == ord("r"):
            # повторный r — заново (или правка существующего ROI)
            self._begin_roi_mode()
        elif key == 13:  # Enter — замкнуть зону / завершить size-точки / принять ROI
            if self.state.mode == "zone":
                try:
                    self.state.finish_zone()
                    self._notify(f"зона закрыта: {len(self.state.zone_points)} точек")
                except ValueError as e:
                    self._notify(str(e))
            elif self.state.mode == "size":
                self.state.finish_size()
            elif self.state.mode == "roi":
                try:
                    new_roi = self.state.finish_roi()
                except ValueError as e:
                    self._notify(str(e))
                else:
                    if min(new_roi[2], new_roi[3]) < MIN_ROI_FRACTION:
                        self._notify("ROI слишком маленький (сторона меньше 1% кадра) — "
                                     "кликните углы заново")
                    else:
                        self._apply_roi_change(new_roi)
        elif key == ord("m"):
            self.mask_on = not self.mask_on
        elif key == ord("v"):   # «все» — toggle показа всех счётчиков (только визуал)
            self.show_all = not self.show_all
        elif key == ord("n"):
            if self.frames:
                self._idx = (self._idx + 1) % len(self.frames)
        elif key == ord("p"):
            if self.frames:
                self._idx = (self._idx - 1) % len(self.frames)
        # цифры 1-9 в режиме «размер» больше не используются
        # (рост задаётся двумя кликами «меркой роста»)
        elif key == ord("x"):
            self._do_delete()
        elif key == ord("b"):   # отменить последнюю size-точку (только в режиме «размер»)
            self._do_undo_size_point()
        elif key == ord("["):   # стрелки у waitKey ненадёжны — берём [ ]
            self._switch_counter(-1)
        elif key == ord("]"):
            self._switch_counter(1)
        elif key == ord("t"):
            self._begin_time_input()
        elif key == ord("a"):
            self._do_save()
        elif key == ord(","):   # `,` — масштаб отображения: уменьшить (циклически)
            self.scale = next_scale(self.scale, -1)
        elif key == ord("."):   # `.` — масштаб отображения: увеличить (циклически)
            self.scale = next_scale(self.scale, 1)

    # ------------------------------------------------------------------ кэш и seek
    def _load_cache(self) -> int:
        """Загрузить до ``cache_frames`` кадров от ТЕКУЩЕЙ позиции источника с масками;
        ЗАМЕНЯЕТ списки frames/masks/frame_indices. Возвращает число загруженных кадров."""
        self.frames.clear()
        self.masks.clear()
        self.frame_indices.clear()
        assert self._pipe is not None   # вызывается только после build() в open/_apply_roi_change
        while len(self.frames) < self.cache_frames:
            fr = self._pipe.source.read()
            if fr is None:
                break
            self._detector.detect(fr.image)
            self.frames.append(fr.image.copy())
            self.masks.append(getattr(self._detector, "last_mask", None))
            self.frame_indices.append(fr.index)
        return len(self.frames)

    def _seek_to(self, t_s: float) -> bool:
        """source.seek(t_s) + замена кэша кадрами после метки (общая логика seek).

        Используется и клавишей [t] (:meth:`_apply_time_seek`), и стартовым
        переходом к ``processing.frame_start`` (задача 12). State (линии/зоны/
        sizes/текущий счётчик) НЕ сбрасывается — только кэш кадров. Если после
        метки нет ни одного кадра — возврат в начало, чтобы окно не осталось
        без кадров.

        :returns: True, если после метки кадры есть (idx=0); False — seek не
            удался или за пределами видео (кэш уже возвращён в начало).
        """
        assert self._pipe is not None
        if not self._pipe.source.seek(t_s):
            return False
        n_loaded = self._load_cache()   # читает от новой метки, ЗАМЕНЯЕТ frames/masks
        self._idx = 0
        if n_loaded == 0:
            self._pipe.source.seek(0.0)
            self._load_cache()
            return False
        return True

    def _begin_time_input(self) -> None:
        """[t]/кнопка «время» — включить режим ввода времени для seek.

        Для HLS/URL (нет случайного доступа) режим НЕ включается — только уведомление;
        повторное нажатие при открытом режиме ничего не делает (не дублирует).
        Собранные линии/зоны/счётчик при этом не трогаются.
        """
        if self.time_input_active:
            return
        if not self.seek_supported:
            self._notify("seek недоступен для HLS/URL — только файл видео")
            return
        self.time_buf.reset()
        self.time_input_active = True
        self._notify(f"Время (сек): наберите цифры 0-9 | Enter=OK, ESC/q=отмена")

    def _apply_time_seek(self) -> None:
        """Enter в режиме ввода: source.seek(N) и замена кэша кадрами после метки.

        State (линии/зоны/sizes/текущий счётчик) НЕ сбрасывается — только кэш кадров.
        Если после метки нет ни одного кадра — возврат в начало, чтобы окно не осталось
        без кадров.
        """
        v = self.time_buf.value()
        if v is None:
            self._notify("время не задано — наберите цифры 0-9")
            return
        assert self._pipe is not None
        # время больше длительности файла → конец минус cache_frames (иначе cv2 уйдёт в начало)
        t_seek, clamped = clamp_seek_time(float(v), getattr(self._pipe.source, "duration", 0.0),
                                          self.cache_frames, self._pipe.source.fps)
        if clamped:
            self._notify(f"время {v} с больше длительности файла — seek к {t_seek:.1f} с "
                         f"(конец − {self.cache_frames} кадр(ов))")
        self.time_input_active = False
        if not self._seek_to(t_seek):
            if self.seek_supported:
                self._notify(f"seek к {v} c: после метки нет кадров — вернулся в начало")
            else:
                self._notify("seek недоступен для HLS/URL — только файл видео")
        else:
            self._notify(f"seek к {v} c: загружено {len(self.frames)} кадр(ов)")

    # ------------------------------------------------------------------ счётчики
    def _switch_counter(self, direction: int) -> None:
        """[<]/[>]/[ ] — переключение счётчика с загрузкой его геометрии."""
        try:
            new_id = cycle_counter_id([c.id for c in self.cfg.counters],
                                      self.state.counter_id, direction)
            counter = next(c for c in self.cfg.counters if c.id == new_id)
            load_counter_into_state(self.state, counter)
            self._notify(f"счётчик: {new_id} (геометрия загружена для правки)")
        except ValueError as e:
            self._notify(str(e))

    def _new_counter(self, kind: str) -> None:
        """+линия/+зона — новый счётчик (в cfg попадёт при [a])."""
        new_id = make_new_counter_id({c.id for c in self.cfg.counters}, kind)
        self.state.counter_id = new_id
        self.state.set_mode("line" if kind == "line" else "zone")   # пустые точки, нужный режим
        self._notify(f"новый счётчик {new_id} — нарисуйте {'линию' if kind == 'line' else 'зону'}, "
                     f"[a] сохранит в конфиг")

    def _do_delete(self) -> None:
        """удалить/x — удалить текущий счётчик (логика в delete_current_counter)."""
        msg = delete_current_counter(self.cfg, self.state)
        self._notify(msg)
        print(f"calibrate: {msg}")

    def _do_undo_size_point(self) -> None:
        """[b] — отменить последнюю size-точку (кнопки в панели нет).

        Работает осмысленно только в режиме «размер»: из других режимов
        точки НЕ удаляются, а выдаётся подсказка включить режим размер.
        """
        if self.state.mode != "size":
            self._notify("−точка: включите режим размер (клавиша s / кнопка «размер»)")
            return
        pt = undo_last_size_point(self.state)
        if pt is None:
            self._notify("размер: точек нет")
        else:
            self._notify(f"размер: последняя точка отменена (x={pt[0]:.3f}, "
                         f"y={pt[1]:.3f}, h={int(round(pt[2] * 100))}%)")

    def _do_save(self) -> None:
        """a/«сохранить» — применить и записать в ``save_to`` (semантика без изменений)."""
        try:
            changed = list(apply_calibration(self.cfg, self.state))
            # ROI пишется в конфиг вместе с остальными блоками (как сейчас);
            # если других изменений нет — roi единственное изменение
            if not changed and self.cfg.processing.roi is not None:
                changed.append(f"processing.roi: {describe_roi(self.cfg.processing.roi)}")
            if not changed:
                self._notify("ничего не менялось — нет собранных линий/зон/size-точек")
            elif self._save_target is None:
                self._notify("куда сохранять не задано (save_to) — конфиг не записан")
            else:
                Config.save(self.cfg, self._save_target)
                self.saved_once = True
                print(f"calibrate: сохранено в {self._save_target}:")
                for line in changed:
                    print(f"  - {line}")
                print(f"calibrate: дальше — "
                      f".venv/bin/python -m visio_people_counter count --config {self._save_target}")
        except (ConfigError, OSError) as e:
            self._notify(f"ошибка записи: {e}")

    # ------------------------------------------------------------------ режимы
    def _set_draw_mode(self, kind: str) -> None:
        """Переключить режим рисования; если текущий счётчик другого типа — новый id."""
        new_id = ensure_counter_kind(self.cfg, self.state, kind)
        self.state.set_mode(kind)
        if new_id is not None:
            what = {"line": "линию", "zone": "зону"}[kind]
            self._notify(f"текущий счётчик — другого типа: создан новый {new_id} для {what}")

    def _begin_roi_mode(self) -> None:
        """[r]/кнопка «roi» — режим «ROI»: два клика по углам, Enter — принять.

        Повторный запуск при существующем ROI — правка: текущий прямоугольник
        показывается в окне как стартовые углы (рисуется поверх, клик заново).
        """
        self.state.set_mode("roi")
        old = self.cfg.processing.roi
        if old is not None:
            x, y, w_, h_ = old
            self.state.roi_points = [(x, y), (round(x + w_, 4), round(y + h_, 4))]
            self._notify(f"ROI: правка существующего ({describe_roi(old)}) — кликните 2 угла заново "
                         f"или Enter — оставить как есть")
        else:
            self._notify("ROI: кликните ДВА угла области обработки; Enter — принять, ESC — отмена")

    def _apply_roi_change(self, new_roi: list[float]) -> None:
        """Enter в режиме «ROI»: принять [x,y,w,h] (норм. полный кадр).

        При отличии от текущего ROI: источник переоткрывается с кропом
        (новый Pipeline), текущая позиция по времени сохраняется, если возможно;
        окно сразу показывает ROI-вид (рамка + подпись в статусе). Вызывается
        после проверки минимального размера прямоугольника.
        """
        old = self.cfg.processing.roi
        if list(new_roi) == (list(old) if old is not None else None):
            self.state.mode = None
            self.state.roi_points = []
            self._notify(f"ROI без изменений: {describe_roi(new_roi)}")
            return
        # текущая позиция по времени (best effort; для HLS/без fps — с начала)
        t_cur: Optional[float] = None
        if self._idx < len(self.frame_indices) and self._pipe is not None \
                and self._pipe.source.fps > 0:
            t_cur = float(self.frame_indices[self._idx]) / float(self._pipe.source.fps)
        self.cfg.processing.roi = list(new_roi)
        try:
            new_pipe = Pipeline(self.cfg).build()
        except (VideoSourceError, ConfigError) as e:
            self.cfg.processing.roi = old   # откат: окно продолжает работать со старым источником
            self.state.mode = None
            self.state.roi_points = []
            self._notify(f"ROI не применён (источник не открылся): {e}")
            return
        assert self._pipe is not None
        self._pipe.close()
        self._pipe = new_pipe
        self.w, self.h = self._pipe.source.width, self._pipe.source.height
        self._detector = MotionDetector(self.cfg)   # модель фона под новый размер кадра
        # авто-масштаб при новом (возможно, крошечном) размере: UI должен остаться доступным
        new_scale = auto_ui_scale(self.w, self.h, self.scale)
        if new_scale != self.scale:
            self.scale = new_scale
            print(f"calibrate: малое изображение {self.w}×{self.h} — масштаб окна → {self.scale:g}x")
            self._notify(f"малое изображение {self.w}×{self.h} — окно увеличено до "
                         f"{self.scale:g}x для доступного UI")
        if t_cur is not None and self.seek_supported:
            self._seek_to(t_cur)              # сохранение позиции; при неудаче — возврат в начало
        else:
            self._load_cache()
            self._idx = 0
            if not self.frames:
                self._notify("ROI: после переоткрытия источника кадры не загрузились — "
                             "проверьте поток")
        self.state.mode = None
        self.state.roi_points = []
        print(f"calibrate: ROI изменён: {describe_roi(old) if old is not None else 'нет'} → "
              f"{describe_roi(new_roi)} (источник переоткрыт с кропом, кадры {self.w}x{self.h})")
        self._notify("ROI изменён — проверьте линии/зоны (режим «все»): их координаты "
                     "отсчитываются от ROI")
