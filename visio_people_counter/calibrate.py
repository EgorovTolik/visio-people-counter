"""GUI-калибровка линии/зоны/size-точек мышью → запись в config.yaml (задача 16).

``python -m visio_people_counter calibrate --config config.yaml
   --video PATH_OR_URL [--counter-id main_line]``

Режимы по клавишам И кликабельными кнопками в верхней части кадра
(см. :func:`run_calibration`):

* **l** + 2 клика / кнопка «линия» — линия A→B для выбранного счётчика
  (порядок кликов = направление «in»);
* **z** + N кликов, **Enter** / кнопка «зона» — замкнуть полигон зоны (>= 3 точки);
* **s** + ДВА клика «меркой роста» (верх и низ человека в этом месте кадра):
  центр точки = середина пары, рост = длина отрезка в долях высоты кадра;
  **Enter** / кнопка «size» — завершить набор size-точек; клик по УЖЕ НАРИСОВАННОЙ
  size-точке (hit-радиус ~15 px, только когда pending-клики нет) удаляет её, а
  **b** отменяет ПОСЛЕДНЮЮ size-точку (кнопки нет — конкретную мерку удаляют кликом по ней);
* **r** + ДВА клика по углам прямоугольника / кнопка «roi» — режим «ROI»:
  область обработки ``processing.roi`` в долях ПОЛНОГО кадра; **Enter** — принять
  (источник переоткрывается с кропом, окно показывает ROI-вид с рамкой и подписью,
  координаты линий/зон/size-точек и событий отсчитываются от ROI),
  ESC/повторный **r** — отмена/заново; при изменении ROI — уведомление о том, что
  геометрию счётчиков стоит проверить в режиме «все»;
* **m** / кнопка «маска» — toggle показа текущей маски движения (live-подстройка порога);
* **v** / кнопка «все» — toggle режима просмотра ВСЕХ счётчиков из конфига: каждая
  линия/зона рисуется поверх кадра с размером (линия — длина в px и нормализованная,
  зона — число углов и доля площади кадра); текущий счётчик выделяется. Чисто
  визуальный режим — не влияет на apply/delete/seek;
* **[ ]** / кнопки «<» «>» — предыдущий/следующий счётчик из cfg.counters (циклически,
  геометрия загружается в окно для правки);
* **+линия/+зона** — создать новый счётчик (id line_N / zone_N) и перейти к его рисованию;
* **x** / кнопка «удалить» — удалить текущий счётчик из cfg.counters, очистить его
  точки в state (чтобы сохранение не «воскресило» счётчик) и переключиться на
  соседний или новый той же природы;
* **n/p** — следующий/предыдущий из загруженных кадров;
* **`,` / `.`** — масштаб ОТОБРАЖЕНИЯ окна: пресеты 0.5/1/1.5/2 (уменьшить/
  увеличить, с циклом); только экран — обработка и координаты конфига не меняются;
* **t** / кнопка «время» — seek к заданному времени (секунды, только для файла:
  набрать цифры 0-9, Enter=OK, ESC/q=отмена; кэш заменяется кадрами после метки,
  линии/зоны/счётчик не сбрасываются);
* **a** / кнопка «сохранить» — применить и записать в config.yaml
  (остальные настройки сохраняются), печатает короткий diff «старые → новые координаты»;
* **q/ESC** — выход без записи.

Вся логика «клик → обновление конфига», переключение счётчиков и раскладка/hit-test
кнопок вынесена в чистые функции (:class:`CalibrationState`, :func:`click_to_norm`,
:func:`apply_calibration`, :func:`cycle_counter_id`,
:func:`make_new_counter_id`, :func:`load_counter_into_state`,
:func:`delete_current_counter`, :func:`layout_buttons`, :func:`hit_button`,
:func:`size_point_at_click`, :func:`remove_size_point`, :func:`undo_last_size_point`,
:func:`handle_size_click`)
и тестируется БЕЗ окна (tests/test_gui_calibrate.py).
"""


from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import (
    Config,
    ConfigError,
    LineCounterConfig,
    ZoneCounterConfig,
    describe_roi,
)
from .gui import GuiPlayer, apply_qt_env   # окно/доступность GUI — только в драйвере run_calibration
from .text_overlay import put_text, text_width
from .video_source import VideoSourceError

#: сколько первых кадров видео загрузить в кэш по умолчанию (для HLS — первые кадры);
#: переопределяется параметром cache_frames / CLI --cache-frames.
DEFAULT_CACHE_FRAMES = 100

#: как долго красные сообщения-уведомления держатся на экране.
MESSAGE_TTL_SECONDS = 5.0

#: hit-радиус (в px ЭКРАНА) клика по существующей size-точке в режиме «размер»:
#: клик ближе, чем на этот радиус ПО ОБЕИМ осям (x и y), — точка удаляется.
SIZE_POINT_HIT_RADIUS_PX = 15


def clamp_seek_time(requested_s: float, duration_s: float,
                    cache_frames: int, fps: float) -> tuple[float, bool]:
    """Запрос seek ограничить так, чтобы после метки оставалось место на ``cache_frames``.

    :returns: ``(time_s, clamped)``. Если длительность/fps неизвестны (<= 0) — время не меняется;
        если и «конец минус cache-кадров» <= 0 — ставится 0.0.
    """
    if duration_s <= 0 or fps <= 0 or cache_frames <= 0:
        return requested_s, False
    max_t = max(0.0, duration_s - cache_frames / float(fps))
    if requested_s > max_t:
        return max_t, True
    return requested_s, False


def filter_expired_messages(messages: list[tuple[str, float]],
                            now: float) -> list[tuple[str, float]]:
    """Отбросить уведомления с истёкшим сроком ``(message, expires_at)`` (чистая функция)."""
    return [m for m in messages if m[1] > now]


def draw_notification(img: np.ndarray, text: str, x: int, y: int,
                      size_px: int = 14, color=(0, 0, 255)) -> None:
    """Уведомление: одноцветная скруглённая подложка + текст БЕЗ тени (читаемость)."""
    if img is None or img.size == 0 or not text:
        return
    pad = 5
    tw = text_width(text, size_px)
    x0, y0 = x - pad, y - pad
    x1, y1 = x + tw + pad, y + size_px + pad
    # не вылезать за кадр
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.shape[1] - 1, x1), min(img.shape[0] - 1, y1)
    bg = (255, 255, 255)
    if hasattr(cv2, "roundedRect"):
        cv2.roundedRect(img, (x0, y0), (x1, y1), max(2, size_px // 3), -1,
                        color=bg, lineType=cv2.LINE_AA)
    else:  # pragma: no cover
        cv2.rectangle(img, (x0, y0), (x1, y1), bg, -1)
    put_text(img, text, (max(0, x), max(0, y)), size_px=size_px, color=color,
             shadow=False)


# ---------------------------------------------------------------------------
# Чистые функции: клик → координаты конфига (тестируются без imshow)
# ---------------------------------------------------------------------------

def click_to_norm(x: int, y: int, w: int, h: int) -> tuple[float, float]:
    """Пиксельный клик → нормализованные доли кадра (0..1), зажим и округление.

    :raises ValueError: разрешение кадра <= 0.
    """
    if w <= 0 or h <= 0:
        raise ValueError(f"размер кадра: ожидалось w>0, h>0, получено {w}x{h}")
    nx = min(1.0, max(0.0, float(x) / float(w)))
    ny = min(1.0, max(0.0, float(y) / float(h)))
    return (round(nx, 4), round(ny, 4))


#: минимальная «мерка роста» (доля высоты кадра): короче — пара не добавляется
#: (защита от случайного двойного клика в одной точке).
MIN_SIZE_FRACTION = 0.01


#: максимальное число знаков в буфере ввода времени (seek-секунды).
TIME_INPUT_MAX_DIGITS = 6


@dataclass
class TimeInputBuffer:
    """Буфер ввода времени для seek (чистый класс — без cv2/окна).

    Цифры набираются в конец (лимит :data:`TIME_INPUT_MAX_DIGITS` знаков,
    лишние игнорируются), backspace удаляет последний знак.
    """

    digits: list[int] = field(default_factory=list)

    def feed_digit(self, d: int) -> None:
        """Добавить цифру 0..9 в конец; при лимите знаков — игнорировать.

        :raises ValueError: d не целое 0..9.
        """
        if not isinstance(d, int) or isinstance(d, bool) or not (0 <= d <= 9):
            raise ValueError(f"цифра времени: ожидалось int 0..9, получено {d!r}")
        if len(self.digits) < TIME_INPUT_MAX_DIGITS:
            self.digits.append(d)

    def backspace(self) -> None:
        """Удалить последний знак (пустой буфер — без изменений)."""
        if self.digits:
            self.digits.pop()

    def value(self) -> Optional[int]:
        """Набранное время в секундах (int); пустой буфер → None."""
        return int("".join(str(d) for d in self.digits)) if self.digits else None

    def reset(self) -> None:
        """Очистить буфер."""
        self.digits = []


@dataclass
class CalibrationState:
    """Собранное в окне калибровки (чистое состояние — без cv2)."""

    counter_id: str = ""
    mode: Optional[str] = None   # "line" | "zone" | "size" | "roi" | None
    line_points: list[tuple[float, float]] = field(default_factory=list)   # норм. (x, y)
    zone_points: list[tuple[float, float]] = field(default_factory=list)   # норм. (x, y)
    size_points: list[tuple[float, float, float]] = field(default_factory=list)  # [x_frac, y_frac, h_frac]
    #: клики режиму «ROI» — углы прямоугольника (норм., относительно ПОЛНОГО кадра)
    roi_points: list[tuple[float, float]] = field(default_factory=list)
    #: первый клик «мерки роста» (pending-пара): ждёт второй клик (верх/низ человека)
    size_first_point: Optional[tuple[float, float]] = None

    def set_mode(self, mode: str) -> None:
        """Переключить режим ('l'/'z'/'s'/'r'); переключение очищает НОВЫЙ набор.

        Pending-пара size («первый клик мерки») сбрасывается при ЛЮБОЙ смене режима —
        она не должна переживать выход из режима «размер».
        """
        if mode not in ("line", "zone", "size", "roi"):
            raise ValueError(f"режим калибровки: ожидалось line/zone/size/roi, получено {mode!r}")
        self.size_first_point = None
        self.mode = mode
        if mode == "line":
            self.line_points = []
            self.zone_points = []      # один счётчик — либо линия, либо зона
        elif mode == "zone":
            self.zone_points = []
            self.line_points = []
        elif mode == "roi":
            self.roi_points = []       # заново; существующий ROI подгружает вызывающий

    def cancel_roi(self) -> None:
        """ESC в режиме «ROI»: отмена рисования (остальное состояние не трогается)."""
        self.mode = None
        self.roi_points = []
        self.size_first_point = None

    def handle_click(self, x_norm: float, y_norm: float) -> None:
        """Клик в текущем режиме (координаты уже нормализованы)."""
        if self.mode == "line":
            if len(self.line_points) >= 2:
                self.line_points = []  # третья точка — начать линию заново
            self.line_points.append((x_norm, y_norm))
        elif self.mode == "zone":
            self.zone_points.append((x_norm, y_norm))
        elif self.mode == "roi":
            if len(self.roi_points) >= 2:
                self.roi_points = []   # третий клик — начать прямоугольник заново
            self.roi_points.append((x_norm, y_norm))
        elif self.mode == "size":
            # «мерка роста»: этот клик — первый из пары (второй завершит точку);
            # если pending уже есть — пара НЕ перезаписывается (завершается через
            # handle_size_click, где есть защита от слишком короткой мерки)
            if self.size_first_point is None:
                self.size_first_point = (x_norm, y_norm)

    def finish_zone(self) -> list[tuple[float, float]]:
        """Enter в режиме 'z': закрыть полигон. :raises ValueError: < 3 точек."""
        if len(self.zone_points) < 3:
            raise ValueError(
                f"зона: ожидалось >= 3 клика, получено {len(self.zone_points)}")
        return list(self.zone_points)

    def finish_size(self) -> None:
        """Enter в режиме 's': завершить набор size-точек (pending-пара сбрасывается)."""
        self.size_first_point = None

    def finish_roi(self) -> list[float]:
        """Enter в режиме 'roi': принять прямоугольник → ``[x, y, w, h]`` (норм., полный кадр).

        Порядок кликов не важен. :raises ValueError: меньше 2 кликов.
        """
        if len(self.roi_points) < 2:
            raise ValueError(
                f"ROI: ожидалось 2 клика по углам, получено {len(self.roi_points)}")
        return roi_from_two_clicks(self.roi_points[0], self.roi_points[1])


def roi_from_two_clicks(p1: tuple[float, float], p2: tuple[float, float]) -> list[float]:
    """Два клика по углам прямоугольника (порядок не важен) → нормализованный
    ``[x, y, w, h]`` относительно ПОЛНОГО кадра (доли 0..1, округление до 4 знаков).

    Защита: значения зажимаются так, чтобы ROI не выходил за кадр и не был
    вырожденным; ровно 0 заменяется на :data:`ROI_EPS`, т.к. конфиг требует
    все четыре числа в (0..1].
    """
    # конфиг требует все четыре числа в (0..1] — ровно 0 недопустим, берём ~1 px
    eps = ROI_EPS
    x = max(eps, round(min(float(p1[0]), float(p2[0])), 4))
    y = max(eps, round(min(float(p1[1]), float(p2[1])), 4))
    w = min(round(abs(float(p2[0]) - float(p1[0])), 4), round(1.0 - x, 4))
    h = min(round(abs(float(p2[1]) - float(p1[1])), 4), round(1.0 - y, 4))
    return [x, y, w, h]


#: минимальное смещение ROI от края кадра (доля): конфиг требует (0..1], а клик
#: точно по краю даёт 0 — вместо этого берётся этот epsilon (~1 px на HD-кадре).
ROI_EPS = 0.0001

#: минимальная сторона ROI (доля кадра): меньше — прямоугольник считается
#: случайным двойным кликом и не применяется.
MIN_ROI_FRACTION = 0.01


#: пресеты масштаба ОТОБРАЖЕНИЯ окна (только imshow; обработка и координаты
#: конфигурации всегда в исходном разрешении кадра). Клавиши: `,` — уменьшить,
#: `.` — увеличить (с циклом по концам).
SCALE_PRESETS = (0.5, 1.0, 1.5, 2.0)


def next_scale(current: float, direction: int) -> float:
    """Следующий пресет масштаба отображения (чистая функция).

    :param current: текущий масштаб (может быть и не из :data:`SCALE_PRESETS`).
    :param direction: +1 — следующий БОЛЬШИЙ пресет (больше всех → первый),
        -1 — предыдущий МЕНЬШИЙ пресет (меньше всех → последний).
    :raises ValueError: direction не ±1.

    Примеры: 0.5→1.0→1.5→2.0→0.5 (вверх); 1.0→0.5→2.0→1.5 (вниз);
    не-пресет 1.2: +1 → 1.5, -1 → 1.0.
    """
    if direction not in (-1, 1):
        raise ValueError(f"направление масштаба: ожидалось -1 или 1, получено {direction!r}")
    if direction > 0:
        for p in SCALE_PRESETS:
            if p > current:
                return p
        return SCALE_PRESETS[0]
    for p in reversed(SCALE_PRESETS):
        if p < current:
            return p
    return SCALE_PRESETS[-1]


def unscale_mouse(x: int, y: int, scale: float,
                  w: Optional[int] = None, h: Optional[int] = None) -> tuple[int, int]:
    """Координаты мыши на МАСШТАБИРОВАННОЙ картинке → координаты исходного кадра.

    Деление на ``scale`` с округлением; если заданы ``w``/``h`` (размер исходного
    кадра) — результат зажимается в [0, w-1] / [0, h-1].

    :raises ValueError: scale <= 0.
    """
    if scale <= 0:
        raise ValueError(f"масштаб: ожидалось > 0, получено {scale!r}")
    ux = int(round(x / scale))
    uy = int(round(y / scale))
    if w is not None and w > 0:
        ux = max(0, min(w - 1, ux))
    if h is not None and h > 0:
        uy = max(0, min(h - 1, uy))
    return (ux, uy)


def size_point_at_click(size_points, x_px: int, y_px: int, w: int, h: int,
                        radius_px: int = SIZE_POINT_HIT_RADIUS_PX) -> Optional[int]:
    """Индекс существующей size-точки под пиксельным кликом (x_px, y_px) или None.

    Сравнение в пикселях ЭКРАНА: точка считается «попавшей», если
    ``|x_px − x_frac·w| ≤ radius_px`` И ``|y_px − y_frac·h| ≤ radius_px`` —
    радиус фиксирован на экране, поэтому на широком/высоком кадре он работает
    так же, как и на узком. Возвращает индекс ПЕРВОЙ подходящей точки.
    Пустой список / некорректный размер кадра → None (без исключений).
    """
    if w <= 0 or h <= 0:
        return None
    for i, (x_f, y_f, _h_f) in enumerate(size_points):
        if abs(x_px - x_f * w) <= radius_px and abs(y_px - y_f * h) <= radius_px:
            return i
    return None


def remove_size_point(state: CalibrationState,
                      index: int) -> Optional[tuple]:
    """Удалить size-точку по индексу (чистая функция); вернуть удалённую точку.

    Пустой список или индекс вне 0..len-1 → None, state не меняется.
    """
    if not 0 <= index < len(state.size_points):
        return None
    return state.size_points.pop(index)


def undo_last_size_point(state: CalibrationState) -> Optional[tuple]:
    """Отменить ПОСЛЕДНЮЮ size-точку (клавиша [b]; кнопки в панели нет).

    Возвращает удалённую точку ``[x, y, h]``; пустой список → None (без падений).
    Pending-пара size (``size_first_point``) при этом не трогается.
    """
    if not state.size_points:
        return None
    return state.size_points.pop()


def handle_size_click(state: CalibrationState, x_px: int, y_px: int,
                      w: int, h: int,
                      radius_px: int = SIZE_POINT_HIT_RADIUS_PX) -> list[str]:
    """Клик в режиме «размер» (координаты в px экрана) — чистая функция.

    Механика «мерки роста» (два клика):

    * есть pending ``size_first_point`` → ВТОРОЙ клик завершает пару:
      ``x = (x1+x2)/2``, ``y = (y1+y2)/2`` (центр),
      ``h = |y2-y1| / frame_h`` (доля высоты кадра); тройка ``(x, y, h)``
      добавляется в ``state.size_points``, pending сбрасывается. Если
      ``|y2-y1| < :data:`MIN_SIZE_FRACTION`` (~1% кадра) — точка НЕ добавляется,
      возвращается уведомление «слишком маленькая мерка», pending всё равно
      сбрасывается;
    * иначе клик по существующей size-точке (:func:`size_point_at_click`,
      hit-радиус ``radius_px``) → удалить её (работает только когда pending НЕТ);
    * иначе — ПЕРВЫЙ клик: запомнить ``size_first_point``.

    В других режимах клик просто передаётся в :meth:`CalibrationState.handle_click`
    (size-точки не конфликтуют с точками линии/зоны).

    :returns: список коротких рус. уведомлений (может быть пустым).
    """
    if state.mode != "size":
        state.handle_click(*click_to_norm(x_px, y_px, w, h))
        return []
    x2, y2 = click_to_norm(x_px, y_px, w, h)
    if state.size_first_point is not None:
        x1, y1 = state.size_first_point
        dy = abs(y2 - y1)   # доля высоты кадра (y уже в долях 0..1)
        if dy < MIN_SIZE_FRACTION:
            msgs = ["слишком маленькая мерка (меньше 1% кадра) — пара сброшена"]
        else:
            x = round((x1 + x2) / 2.0, 4)
            y = round((y1 + y2) / 2.0, 4)
            state.size_points.append((x, y, round(dy, 4)))
            msgs = [f"размер: добавлена точка (x={x:.3f}, y={y:.3f}, "
                    f"h={int(round(dy * 100))}%)"]
        state.size_first_point = None
        return msgs
    idx = size_point_at_click(state.size_points, x_px, y_px, w, h,
                              radius_px=radius_px)
    if idx is not None:
        removed = remove_size_point(state, idx)
        return [f"размер: точка удалена (x={removed[0]:.3f}, y={removed[1]:.3f}, "
                f"h={int(round(removed[2] * 100))}%)"]
    state.size_first_point = (x2, y2)
    return []


def _find_counter(cfg: Config, counter_id: str, ctype):
    for c in cfg.counters:
        if c.id == counter_id and isinstance(c, ctype):
            return c
    return None


def apply_calibration(cfg: Config, state: CalibrationState) -> list[str]:
    """Применить собранное состояние к :class:`Config` (in-place).

    * линия (2 точки): обновляет a/b существующего line-счётчика с id
      ``state.counter_id``; если такого счётчика нет — добавляет новый;
    * зона (>= 3 точек после Enter-логике): то же для zone-счётчика;
    * size-точки: ``size_profile.control_points`` (тройки ``[x, y, h]``) +
      ``enabled=True``.

    :returns: список коротких строк diff «старое → новое» (пусто — нечего применить).
    """
    changed: list[str] = []
    cid = state.counter_id

    if len(state.line_points) == 2:
        a, b = state.line_points
        c = _find_counter(cfg, cid, LineCounterConfig)
        if c is None:
            # защита от дубля: id занят счётчиком ДРУГОГО типа — переименовываем,
            # а не дублируем (дублирующиеся id ломают Config.load)
            if any(x.id == cid for x in cfg.counters):
                new_id = make_new_counter_id({x.id for x in cfg.counters}, "line")
                changed.append(f"ВНИМАНИЕ: id {cid!r} уже занят другим счётчиком — "
                               f"линия добавлена как {new_id!r}")
                state.counter_id = new_id
                cid = new_id
            cfg.counters.append(LineCounterConfig(id=cid, a=a, b=b))
            changed.append(f"counters: добавлен line-счётчик {cid!r} (a={a}, b={b})")
        else:
            old_a, old_b = c.a, c.b
            c.a, c.b = a, b
            changed.append(f"{cid}: линия a {old_a} → {a}; b {old_b} → {b}")

    if len(state.zone_points) >= 3:
        poly = list(state.zone_points)
        c = _find_counter(cfg, cid, ZoneCounterConfig)
        if c is None:
            # защита от дубля: id занят счётчиком ДРУГОГО типа — переименовываем
            if any(x.id == cid for x in cfg.counters):
                new_id = make_new_counter_id({x.id for x in cfg.counters}, "zone")
                changed.append(f"ВНИМАНИЕ: id {cid!r} уже занят другим счётчиком — "
                               f"зона добавлена как {new_id!r}")
                state.counter_id = new_id
                cid = new_id
            cfg.counters.append(ZoneCounterConfig(id=cid, polygon=poly))
            changed.append(f"counters: добавлен zone-счётчик {cid!r} "
                           f"(полигон {len(poly)} точек)")
        else:
            old = list(c.polygon)
            c.polygon = poly
            changed.append(f"{cid}: зона полигон {len(old)} → {len(poly)} точек")

    if state.size_points:
        sp = cfg.size_profile
        old_n, old_en = len(sp.control_points), sp.enabled
        sp.control_points = list(state.size_points)
        sp.enabled = True
        en_txt = "" if old_en else f"; enabled {old_en} → True"
        changed.append(f"size_profile: control_points {old_n} → "
                       f"{len(sp.control_points)} точек{en_txt}")

    return changed


def _fmt_pt(p: tuple[float, float]) -> str:
    return f"({p[0]:.3f},{p[1]:.3f})"


# ---------------------------------------------------------------------------
# Мульти-счётчики и кнопки (чистые функции — тестируются без окна)
# ---------------------------------------------------------------------------

#: Кнопка: (name, label, active, x0, y0, x1, y1) — name для hit-test/действий,
#: label — русский текст на кнопке, координаты в px кадра (включительно x0/y0,
#: не включительно x1/y1).
Button = tuple[str, str, bool, int, int, int, int]


def cycle_counter_id(ids: list[str], current: Optional[str], direction: int) -> str:
    """Следующий/предыдущий счётчик из списка id (циклически).

    :param ids: id всех счётчиков из cfg.counters, в порядке конфига.
    :param current: текущий id; может не быть в ``ids`` (новый, ещё не сохранённый счётчик).
    :param direction: +1 — следующий, -1 — предыдущий.
    :raises ValueError: пустой список или направление не ±1.
    """
    if not ids:
        raise ValueError("нет счётчиков — создайте новый кнопкой +линия/+зона")
    if direction not in (-1, 1):
        raise ValueError(f"направление: ожидалось -1 или 1, получено {direction!r}")
    if current in ids:
        i = (ids.index(current) + direction) % len(ids)
    else:
        # новый счётчик ещё не в конфиге — «вперёд» ведёт к первому, «назад» — к последнему
        i = 0 if direction > 0 else len(ids) - 1
    return ids[i]


def make_new_counter_id(existing_ids, kind: str) -> str:
    """Уникальный id нового счётчика: ``line_1``, ``line_2``… / ``zone_1``, …

    n — минимальный свободный номер среди ВСЕХ существующих id (не только по
    префиксу): например, при занятых {"line_1", "line_3"} вернёт "line_2".
    :raises ValueError: kind не line/zone.
    """
    if kind not in ("line", "zone"):
        raise ValueError(f"тип счётчика: ожидалось line/zone, получено {kind!r}")
    taken = set(existing_ids)
    n = 1
    while f"{kind}_{n}" in taken:
        n += 1
    return f"{kind}_{n}"


def ensure_counter_kind(cfg: Config, state: "CalibrationState", kind: str) -> Optional[str]:
    """Не дать рисовать тип ``kind`` поверх счётчика другого типа.

    Если текущий ``state.counter_id`` уже есть в ``cfg.counters`` как ДРУГОЙ тип
    (например, линия line_1, а включили режим «зона») — создаётся новый pending-счётчик
    со свободным id нужного типа, и state переключается на него.

    :returns: новый id, если создан; None — текущий счётчик подходит (или его ещё нет в cfg).
    :raises ValueError: kind не line/zone.
    """
    if kind not in ("line", "zone"):
        raise ValueError(f"тип счётчика: ожидалось line/zone, получено {kind!r}")
    current = next((c for c in cfg.counters if c.id == state.counter_id), None)
    if current is None:
        return None
    cur_kind = "line" if isinstance(current, LineCounterConfig) else "zone"
    if cur_kind == kind:
        return None
    new_id = make_new_counter_id({c.id for c in cfg.counters}, kind)
    state.counter_id = new_id
    return new_id


def load_counter_into_state(state: CalibrationState, counter) -> None:
    """Загрузить геометрию существующего счётчика в :class:`CalibrationState` для правки.

    line → mode="line", line_points=[a, b]; zone → mode="zone", zone_points=polygon.
    Точки другого типа очищаются. size-точки общие (не зависят от счётчика) — не трогаются.
    :raises ValueError: неизвестный тип объекта counter.
    """
    if not isinstance(counter, (LineCounterConfig, ZoneCounterConfig)):
        raise ValueError(f"тип счётчика: ожидалось Line/ZoneCounterConfig, получено {type(counter).__name__!r}")
    state.counter_id = counter.id
    if isinstance(counter, LineCounterConfig):
        state.mode = "line"
        state.line_points = [tuple(counter.a), tuple(counter.b)]
        state.zone_points = []
    else:
        state.mode = "zone"
        state.zone_points = [tuple(p) for p in counter.polygon]
        state.line_points = []
    state.size_first_point = None


def delete_current_counter(cfg: Config, state: CalibrationState) -> str:
    """Удалить текущий счётчик (``state.counter_id``) — чистая функция, без cv2/окна.

    In-place меняет и ``cfg``, и ``state``:

    * если id есть в ``cfg.counters`` — убрать его из списка;
    * очистить в state собранные точки этого счётчика, чтобы :func:`apply_calibration`
      не «воскресил» его при сохранении: line-счётчик (или mode=="line") →
      ``state.line_points = []``, zone (или mode=="zone") → ``state.zone_points = []``;
      если счётчика ещё нет в cfg и mode неизвестен — оба набора точек очищаются;
    * после удаления переключиться: если в ``cfg.counters`` остались счётчики —
      как «назад» (:func:`cycle_counter_id` с direction=-1) с загрузкой его геометрии
      в state; если не осталось — автоматически создать новый pending-счётчик той же
      природы (линия → line, зона → zone; по умолчанию line), чтобы продолжить работу.

    :returns: сообщение для пользователя: ``счётчик <id> удалён``.
    """
    cid = state.counter_id
    removed_kind: Optional[str] = None
    for i, c in enumerate(cfg.counters):
        if c.id == cid:
            removed_kind = ("line" if isinstance(c, LineCounterConfig)
                            else "zone")
            cfg.counters.pop(i)
            break

    # очищаем точки удалённого счётчика — иначе apply_calibration пересоздал бы его
    if removed_kind == "line":
        state.line_points = []
    elif removed_kind == "zone":
        state.zone_points = []
    else:
        # неприменённый «новый»: ориентируемся на текущий mode (иначе — оба набора)
        if state.mode == "line":
            state.line_points = []
        elif state.mode == "zone":
            state.zone_points = []
        else:
            state.line_points = []
            state.zone_points = []

    if cfg.counters:
        # «назад» относительно удалённого id: его уже нет в списке → последний
        new_id = cycle_counter_id([c.id for c in cfg.counters], cid, -1)
        counter = next(c for c in cfg.counters if c.id == new_id)
        load_counter_into_state(state, counter)
    else:
        kind = removed_kind or (state.mode if state.mode in ("line", "zone")
                                else "line")
        state.counter_id = make_new_counter_id({c.id for c in cfg.counters}, kind)
        state.set_mode(kind)

    return f"счётчик {cid} удалён"


# ---------------------------------------------------------------------------
# Режим «все»: все счётчики из конфига с размерами (чистые функции, без окна)
# ---------------------------------------------------------------------------

#: палитра цветов (BGR) режима «все» — выбирается по индексу счётчика в cfg.counters.
#: Небольшая (4 цвета), контрастная и к тому же читаемая текстом на белой подложке
#: меток (draw_notification).
_SHOW_ALL_PALETTE = [
    (255, 140, 0),    # оранжевый
    (230, 0, 230),    # пурпурный
    (200, 100, 0),    # синий
    (210, 210, 0),    # жёлтый
]

#: цвет/стиль ТЕКУЩЕГО (редактируемого) счётчика в режиме «все»: тот же зелёный,
#: что и у редактируемой линии/зоны (не дублируется криво — совпадает с ней),
#: но толще.
_SHOW_ALL_HIGHLIGHT_COLOR = (0, 255, 0)


def line_length_norm(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Нормализованная длина линии (евклидова по долям кадра, БЕЗ aspect ratio).

    ``sqrt((bx-ax)² + (by-ay)²)`` — координаты a/b уже в долях 0..1.
    """
    return math.hypot(b[0] - a[0], b[1] - a[1])


def line_length_px(a: tuple[float, float], b: tuple[float, float],
                   w: int, h: int) -> float:
    """Длина линии в пикселях кадра ``w``x``h``: ``sqrt((dx*w)² + (dy*h)²)``."""
    return math.hypot((b[0] - a[0]) * w, (b[1] - a[1]) * h)


def polygon_area_fraction(polygon, w: int, h: int) -> float:
    """Доля площади полигона от кадра: ``cv2.contourArea`` в px / ``(w*h)``.

    Пустой, вырожденный (< 3 точек / коллинеарные) полигон или некорректный
    размер кадра → ``0.0``.
    """
    if not polygon or w <= 0 or h <= 0:
        return 0.0
    pts = np.array([(int(round(x * w)), int(round(y * h))) for x, y in polygon],
                   dtype=np.int32)
    if len(pts) < 3:
        return 0.0
    return float(cv2.contourArea(pts)) / float(w * h)


#: цвет мерок «размера» в режиме «все» (вне палитры счётчиков)
_SHOW_ALL_SIZE_COLOR = (0, 212, 255)   # BGR — жёлто-оранжевый


def draw_all_counters(img: np.ndarray, counters, w: int, h: int,
                      highlight_id: Optional[str] = None,
                      size_points: Optional[list] = None) -> None:
    """Режим «все»: нарисовать ВСЕ счётчики из конфига поверх кадра (in-place).

    * line — отрезок A→B + точки; у середины метка
      ``{id}: {длина_px}px ({длина_norm:.3f})``;
    * zone — замкнутый контур полигона + вершины; у центроида метка
      ``{id}: {N} угл., S={доля_площади:.1%} кадра``.

    Цвет — палитра :data:`_SHOW_ALL_PALETTE` по индексу счётчика; текущий
    счётчик (``highlight_id``, как правило ``state.counter_id``) рисуется тем же
    зелёным, что и редактируемая линия/зона, но толще, с пометкой «(текущий)».
    Метки — через :func:`draw_notification` (put_text, кириллица).
    Пустой список счётчиков / нулевой кадр — без изменений, без ошибок.
    """
    if img is None or img.size == 0 or not counters:
        return

    def px(p: tuple[float, float]) -> tuple[int, int]:
        return (int(round(p[0] * w)), int(round(p[1] * h)))

    for i, c in enumerate(counters):
        is_current = highlight_id is not None and getattr(c, "id", "") == highlight_id
        color = _SHOW_ALL_HIGHLIGHT_COLOR if is_current else \
            _SHOW_ALL_PALETTE[i % len(_SHOW_ALL_PALETTE)]
        thickness = 4 if is_current else 2

        if isinstance(c, LineCounterConfig):
            pa, pb = px(c.a), px(c.b)
            cv2.line(img, pa, pb, color, thickness, lineType=cv2.LINE_AA)
            for p in (pa, pb):
                cv2.circle(img, p, 4, color, -1)
            anchor = ((pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2)
            label = (f"{c.id}: {line_length_px(c.a, c.b, w, h):.0f}px "
                     f"({line_length_norm(c.a, c.b):.3f})")
        elif isinstance(c, ZoneCounterConfig) and len(c.polygon) >= 2:
            pts = np.array([px(p) for p in c.polygon], dtype=np.int32)
            cv2.polylines(img, [pts], True, color, thickness, lineType=cv2.LINE_AA)
            for p in pts:
                cv2.circle(img, (int(p[0]), int(p[1])), 3, color, -1)
            m = cv2.moments(pts)   # центроид; фолбэк — среднее вершин
            if m["m00"] > 0:
                anchor = (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))
            else:
                cx, cy = pts.mean(axis=0)
                anchor = (int(cx), int(cy))
            label = (f"{c.id}: {len(c.polygon)} угл., "
                     f"S={polygon_area_fraction(c.polygon, w, h):.1%} кадра")
        else:
            continue   # зона без полигона — нечего рисовать

        if is_current:
            label += " (текущий)"
        size = 14
        tw = text_width(label, size)
        lx = min(max(4, anchor[0] - tw // 2), max(4, img.shape[1] - tw - 4))
        ly = min(max(4, anchor[1] - 20), img.shape[0] - 26)
        draw_notification(img, label, x=lx, y=ly, size_px=size, color=color)

    # мерки «размера» — вертикальные отрезки с % высоты кадра (цвет вне палитры счётчиков)
    if size_points:
        for p in size_points:
            try:
                x, y, hfrac = float(p[0]), float(p[1]), float(p[2])
            except (IndexError, TypeError, ValueError):
                continue
            xc, yc = px((x, y))
            hh = max(4, int(round(hfrac * h)))
            cv2.line(img, (xc, yc - hh // 2), (xc, yc + hh // 2),
                     _SHOW_ALL_SIZE_COLOR, 2, lineType=cv2.LINE_AA)
            put_text(img, f"размер: {hfrac:.0%}", (xc + 6, max(14, yc - 8)),
                     size_px=13, color=_SHOW_ALL_SIZE_COLOR)


def counter_status_text(counter_id: str, counters,
                        scale: float = 1.0,
                        frame_index: Optional[int] = None,
                        roi_label: Optional[str] = None) -> str:
    """Строка статуса над кнопками: текущий счётчик, позиция в списке,
    текущий масштаб отображения (``scale=…x``, меняется клавишами `,`/`.`),
    номер текущего кадра (``кадр N``, 0-based — совпадает с индексом
    ``processing.frame_start/frame_end`` для задания интервала подсчёта) и
    подпись ROI (``ROI: x–x+w × y–y+h``, если ``processing.roi`` задан)."""
    txt: Optional[str] = None
    if not counters:
        txt = "Счётчиков нет — нажмите кнопку +линия или +зона"
    else:
        ids = [c.id for c in counters]
        n = len(ids)
        if counter_id in ids:
            i = ids.index(counter_id)
            c = counters[i]
            kind = "линия" if isinstance(c, LineCounterConfig) else "зона"
            txt = f"Счётчик: {counter_id} ({kind}, позиция {i + 1} из {n})"
        else:
            txt = f"Счётчик: {counter_id} (новый — появится в конфиге после [a])"
    if frame_index is not None:
        txt += f"   кадр {frame_index}"
    txt = f"{txt}   scale={scale:g}x"
    if roi_label:
        txt += f"   {roi_label}"
    return txt


#: Порядок и подписи кнопок панели (слева направо); name — ключ действия.
# «−точка» скрыта из панели: конкретную мерку удаляют кликом по ней,
# отмена последней остаётся на клавише [b].
_BUTTON_LABELS: list[tuple[str, str]] = [
    ("line", "линия"),
    ("zone", "зона"),
    ("size", "размер"),
    ("roi", "roi"),
    ("mask", "маска"),
    ("show_all", "все"),
    ("prev", "<"),
    ("next", ">"),
    ("new_line", "+линия"),
    ("new_zone", "+зона"),
    ("delete", "удалить"),
    ("time", "время"),
    ("save", "сохранить"),
]

#: высота строки кнопки в px: шрифт + отступы сверху/снизу
BUTTON_PAD = 6
#: вертикальный зазор между соседними кнопками
_BUTTON_GAP = 4


def layout_buttons(mode: Optional[str], mask_on: bool, *, show_all: bool = False,
                   x0: int = 8, y0: int = 34, font_px: int = 20) -> list[Button]:
    """Раскладка строки кнопок панели (чистая функция, ширина текста — text_width).

    :returns: список ``(name, label, active, x0, y0, x1, y1)`` слева направо.
        Активны кнопки текущего режима (линия/зона/size/roi), маска при mask_on и
        «все» при show_all; остальные — неактивны (они «моментальные» действия).
    """
    active_map = {"line": mode == "line", "zone": mode == "zone",
                  "size": mode == "size", "roi": mode == "roi",
                  "mask": bool(mask_on),
                  "show_all": bool(show_all)}
    out: list[Button] = []
    x = x0
    for name, label in _BUTTON_LABELS:
        tw = text_width(label, font_px) + 2 * BUTTON_PAD
        active = active_map.get(name, False)
        out.append((name, label, active, x, y0, x + tw, y0 + font_px + 2 * BUTTON_PAD))
        x += tw + _BUTTON_GAP
    return out


def hit_button(buttons: list[Button], x: int, y: int) -> Optional[str]:
    """Название кнопки под точкой (x, y) или None — клик мимо всех кнопок.

    Кнопки не пересекаются и лежат только в верхней панели, поэтому клик ниже
    панели автоматически промахивается (не перехватывает рисование линии/зоны).
    """
    for name, _label, _active, bx0, by0, bx1, by1 in buttons:
        if bx0 <= x < bx1 and by0 <= y < by1:
            return name
    return None


def draw_top_panel(img: np.ndarray, counter_id: str, counters,
                   mode: Optional[str], mask_on: bool,
                   show_all: bool = False, scale: float = 1.0,
                   frame_index: Optional[int] = None,
                   roi_label: Optional[str] = None,
                   ui_scale: float = 1.0) -> list[Button]:
    """Нарисовать вверху кадра статусную строку (с ``scale=…x``, ``кадр N`` и
    подписью ROI, если задан) + панель кнопок; вернуть раскладку для hit-test
    в mouse-callback. Текст — только через put_text (кириллица).

    ``ui_scale`` — компенсатор зума: все размеры оверлея умножаются на него,
    чтобы после ресайза img на factor scale on-screen размер оставался постоянным.
    """
    put_text(img, counter_status_text(counter_id, counters, scale,
                                      frame_index=frame_index,
                                      roi_label=roi_label), (10, 8),
             size_px=int(round(20 * ui_scale)), color=(255, 255, 255))
    font = int(round(20 * ui_scale))
    buttons = layout_buttons(mode, mask_on, show_all=show_all, font_px=font)
    # узкий кадр: сужаем шрифт кнопок (не ниже 12px), чтобы вся строка помещалась
    while buttons and buttons[-1][5] > img.shape[1] - 4 and font > 12:
        font -= 2
        buttons = layout_buttons(mode, mask_on, show_all=show_all, font_px=font)
    for _name, label, active, bx0, by0, bx1, by1 in buttons:
        fill = (0, 128, 0) if active else (60, 60, 60)
        cv2.rectangle(img, (bx0, by0), (bx1, by1), fill, -1)
        tw = text_width(label, font)
        tx = bx0 + max(0, ((bx1 - bx0) - tw) // 2)   # текст по центру кнопки
        put_text(img, label, (tx, by0 + BUTTON_PAD), size_px=font,
                 color=(255, 255, 255))
    return buttons


# ---------------------------------------------------------------------------
# Интерактивный цикл (окно — только здесь)
# ---------------------------------------------------------------------------

#: Подсказки на экране (рисуются через text_overlay.put_text — кириллица поддерживается).
_HINTS = {
    None: ("режимы: кнопки или клавиши — [l]иния 2 клика | [z]она N кликов+Enter "
           "| [s]ize мерка роста: 2 клика (верх/низ человека)\n"
           "счётчики: кнопки [<]/[>] (или клавиши [ ]) — смена | +линия/+зона — новый счётчик "
           "| удалить/[x] — удалить текущий счётчик\n"
           "| [m]аска движения | [v]се — все счётчики с размерами | [n/p] кадр вперёд/назад "
           "| [t]время (seek, только файл) | [a]сохранить | [,/.] масштаб окна 0.5-2 (только экран) | [q]выход"),
    "line": ("ЛИНИЯ: кликните точку A, затем B (порядок = направление in); "
             "повторный 'l' или кнопка «линия» — заново"),
    "zone": ("ЗОНА: кликайте углы полигона; Enter — замкнуть (>=3), "
             "'z'/кнопка «зона» — начать заново"),
    "size": ("SIZE: «мерка роста» — 2 клика: верх и низ человека в этом месте кадра\n"
             "центр считается автоматически, рост = длина отрезка в % высоты кадра; "
             "Enter — завершить набор\n"
             "клик по существующей мерке (когда нет ожидающего 1-го клика) — удалить её; "
             "[b] — отменить последнюю"),
    "roi": ("ROI: кликните ДВА угла прямоугольника области обработки; Enter — принять, "
            "r/кнопка «roi» — заново, ESC — отмена\n"
            "после принятия источник переоткрывается с кропом: координаты линий/зон/size-точек "
            "и событий считаются ОТ ROI (проверьте их в режиме [v]се)"),
}




def _dashed_line(img: np.ndarray, p1: tuple[int, int], p2: tuple[int, int],
                 color, thickness: int = 2,
                 dash_px: int = 8, gap_px: int = 6) -> None:
    """Пунктирный отрезок p1→p2 (для pending-«мерки роста»; чистая функция)."""
    x1, y1 = p1
    x2, y2 = p2
    dist = float(math.hypot(x2 - x1, y2 - y1))
    if dist < 1.0:
        return
    dx = (x2 - x1) / dist
    dy = (y2 - y1) / dist
    t = 0.0
    while t < dist:
        seg = min(dash_px, dist - t)
        cv2.line(img,
                 (int(round(x1 + dx * t)), int(round(y1 + dy * t))),
                 (int(round(x1 + dx * (t + seg))), int(round(y1 + dy * (t + seg)))),
                 color, thickness, lineType=cv2.LINE_AA)
        t += dash_px + gap_px


def _draw_calibration(base: np.ndarray, state: CalibrationState, mask_on: bool,
                      mask: Optional[np.ndarray], w: int, h: int,
                      mouse_pos: Optional[tuple[int, int]] = None,
                      active_roi: Optional[list[float]] = None,
                      *, ui_scale: float = 1.0) -> np.ndarray:
    """Отрисовка состояния калибровки на копии кадра (чистая функция).

    ``mouse_pos`` — текущая позиция курсора (для live-превью pending «мерки роста»);
    None — превью не рисуется.
    ``active_roi`` — принятый ROI из конфига: кадр уже является ROI-видом,
    поэтому рисуем рамку по краям кадра (подпись — в статусной строке).
    """
    img = base.copy()
    if mask_on and mask is not None:
        m = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET)
        img[:] = cv2.addWeighted(img, 0.5, m, 0.5, 0.0)

    def px(p: tuple[float, float]) -> tuple[int, int]:
        return (int(round(p[0] * w)), int(round(p[1] * h)))

    # Компенсатор зума: все размеры оверлея умножаются на ui_scale, чтобы после
    # ресайза img на factor scale их on-screen размер оставался постоянным.
    def sz(v: float) -> int:
        return max(1, int(round(v * ui_scale)))

    if state.line_points:
        pts = [px(p) for p in state.line_points]
        for i, p in enumerate(pts):
            cv2.circle(img, p, sz(5), (0, 255, 0), -1)
            cv2.putText(img, "A" if i == 0 else "B", (p[0] + sz(8), p[1] + sz(6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7 * ui_scale, (0, 255, 0), sz(2), cv2.LINE_AA)
        if len(pts) == 2:
            cv2.line(img, pts[0], pts[1], (0, 255, 0), sz(2), lineType=cv2.LINE_AA)

    if state.zone_points:
        pts = [px(p) for p in state.zone_points]
        for p in pts:
            cv2.circle(img, p, sz(4), (0, 200, 255), -1)
        if len(pts) >= 2:
            cv2.polylines(img, [np.array(pts, dtype=np.int32)], False,
                          (0, 200, 255), sz(2), lineType=cv2.LINE_AA)

    for x_f, y_f, h_f in state.size_points:
        # ЗАВЕРШЁННАЯ «мерка роста» = (x_frac, y_frac, доля высоты человека):
        # вертикальный отрезок длиной h*frame_h с центром в (x, y) — «человечек»,
        # стоявший/шедший в этом месте кадра
        x = int(round(x_f * w))
        yc = max(0, min(h - 1, int(round(y_f * h))))
        seg_h = int(round(h_f * h))
        top = max(0, yc - seg_h // 2)
        bottom = min(h - 1, yc + seg_h // 2)
        cv2.line(img, (x, top), (x, bottom), (0, 255, 255), sz(2))
        for ep in ((x, top), (x, bottom)):
            cv2.circle(img, ep, sz(3), (0, 255, 255), -1)
        put_text(img, f"размер: {int(round(h_f * 100))}%", (x + sz(6), max(4, yc - sz(8))),
                 size_px=int(round(14 * ui_scale)), color=(0, 255, 255))

    # PENDING «мерка роста»: первый клик запомнен — рисуем маркер + пунктирное
    # превью до текущего курсора с текущей длиной (px / % высоты кадра)
    first = getattr(state, "size_first_point", None)
    if first is not None:
        fx = int(round(first[0] * w))
        fy = max(0, min(h - 1, int(round(first[1] * h))))
        cv2.circle(img, (fx, fy), sz(5), (0, 165, 255), -1)
        put_text(img, "мерка: кликните 2-ю точку (низ/верх человека)",
                 (max(4, fx + sz(8)), max(4, fy - sz(10))), size_px=int(round(14 * ui_scale)),
                 color=(0, 165, 255))
        if mouse_pos is not None:
            mx, my = mouse_pos
            _dashed_line(img, (fx, fy), (mx, my), (0, 165, 255), thickness=sz(2))
            len_px = abs(my - fy)
            pct = int(round(len_px / h * 100)) if h > 0 else 0
            put_text(img, f"{len_px}px ({pct}% кадра)",
                     (max(4, mx + sz(8)), max(4, my - sz(10))),
                     size_px=int(round(14 * ui_scale)), color=(0, 165, 255))

    # режим «ROI»: два клика по углам — зелёные маркеры + пунктирная рамка-превью
    roi_pts = getattr(state, "roi_points", [])
    if roi_pts:
        rpts = [(int(round(px_ * w)), int(round(py_ * h))) for px_, py_ in roi_pts]
        for p in rpts:
            cv2.circle(img, p, sz(5), (0, 255, 0), -1)
        if len(roi_pts) >= 2:
            (ax, ay), (bx, by) = rpts[0], rpts[1]
            rx0, ry0 = min(ax, bx), min(ay, by)
            rx1, ry1 = max(ax, bx), max(ay, by)
            cv2.rectangle(img, (rx0, ry0), (rx1, ry1), (0, 255, 0), sz(2), lineType=cv2.LINE_AA)
            put_text(img, "ROI: Enter — принять | r — заново | ESC — отмена",
                     (max(4, rx0), max(72, min(ry1 + sz(8), h - sz(20)))),
                     size_px=int(round(14 * ui_scale)), color=(0, 255, 0))

    # принятый ROI: кадр — уже ROI-вид; рамка по краям (внутри px) как напоминание
    if active_roi is not None and state.mode != "roi":
        m = sz(4)
        cv2.rectangle(img, (m, m), (w - 1 - m, h - 1 - m), (0, 255, 0), sz(3),
                      lineType=cv2.LINE_AA)

    # подсказки внизу кадра (верх занят статусной строкой + панелью кнопок)
    hint_lines = _HINTS[state.mode].split("\n")
    y = h - sz(18) - sz(22) * len(hint_lines)
    for line in hint_lines:
        put_text(img, line, (sz(10), max(4, y)), size_px=int(round(16 * ui_scale)),
                 color=(255, 255, 255))
        y += sz(22)
    return img


def resolve_calibrate_config(video_path: Optional[str],
                             explicit_config: Optional[str]
                             ) -> tuple[Optional[Path], Optional[Path]]:
    """(входной конфиг или None=дефолты, цель сохранения) — задача 12.

    Порядок поиска входного конфига при ``calibrate``:

    * ``explicit_config`` задан → ``(explicit, explicit)`` — явный ``--config``
      используется и как вход, и как цель сохранения (как раньше);
    * без явного ``--config`` и рядом с файлом видео есть
      ``<имя_видео>.config.yaml`` (stem без расширения) → ``(авто, авто)``:
      правки пишутся обратно в тот же файл — пользовательские блоки
      (motion/objects/processing, в т.ч. frame_start/frame_end) НЕ затираются;
    * иначе (видео — URL/HLS/не-файл, автоконфига нет или video_path не задан)
      → ``(None, None)``: вход = дефолты, цель сохранения решает
      :func:`calibration_save_target` как раньше.

    Чистая функция — тестируется без окна (tests/test_gui_calibrate.py).
    """
    if explicit_config:
        p = Path(explicit_config)
        return p, p
    if video_path:
        v = Path(video_path)
        if v.is_file():
            auto = v.with_name(f"{v.stem}.config.yaml")
            if auto.is_file():
                return auto, auto
    return None, None


def frame_start_seek_time(frame_start: int, fps: float) -> Optional[float]:
    """Номер кадра ``processing.frame_start`` → время seek в секундах (задача 12).

    ``t = frame_start / fps`` (fps — нативный fps источника). Если fps <= 0
    (длительность/fps неизвестны) → ``None``: переход пропускать с уведомлением.

    :raises ValueError: frame_start не целое или < 0.
    """
    if not isinstance(frame_start, int) or isinstance(frame_start, bool):
        raise ValueError(f"frame_start: ожидалось int, получено {frame_start!r}")
    if frame_start < 0:
        raise ValueError(f"frame_start: ожидалось >= 0, получено {frame_start!r}")
    if fps <= 0:
        return None
    return float(frame_start) / float(fps)


def calibration_save_target(video_type: str, video_path: str,
                            config_path: str | Path) -> Path:
    """Куда calibrate сохраняет конфиг при [a].

    file → рядом с видео: ``<имя_видео>.config.yaml`` (stem без расширения);
    hls/URL → путь из --config («рядом» для потока не определено).
    """
    if video_type == "file":
        p = Path(video_path)
        return p.with_name(f"{p.stem}.config.yaml")
    return Path(config_path)


def load_calibration_config(config_path: str | Path) -> tuple[Optional[Config], Optional[str]]:
    """Входной конфиг calibrate (общий для cv2- и Qt-драйверов — задача 16).

    :returns: ``(cfg, None)`` — конфиг загружен (или дефолты при первом запуске);
        ``(None, сообщение_ошибки)`` — конфиг нечитаем (сообщение уже напечатано в stderr).
    """
    if Path(config_path).is_file():
        try:
            return Config.load(config_path), None
        except ConfigError as e:
            print(f"calibrate: ошибка конфигурации: {e}", file=sys.stderr)
            return None, str(e)
    # первый запуск: файла ещё нет — калибруем по дефолтам и создадим файл при [a]
    print(f"calibrate: файл {config_path} не найден — запускаю с настройками по умолчанию; "
          f"результат будет сохранён рядом с видео (<имя_видео>.config.yaml) при [a] "
          f"(остальные режимы требуют готовый конфиг)")
    return Config.default(), None


def resolve_save_target(cfg: Config, config_path: str | Path,
                        save_to: Optional[str | Path]) -> Path:
    """Куда calibrate сохраняет при [a] (общий для cv2- и Qt-драйверов — задача 16).

    Явный ``save_to`` (--config) → строго туда; иначе рядом с файлом видео
    (``<имя_видео>.config.yaml``); для HLS/URL — фолбэк на --config. Печатает цель.
    """
    if save_to is not None:
        target = Path(save_to)
    else:
        target = calibration_save_target(cfg.video.type, cfg.video.path, config_path)
        if cfg.video.type != "file":
            print(f"calibrate: источник HLS/URL — конфиг будет сохранён в {target} "
                  f"(«рядом с видео» для потока не определено)")
    print(f"calibrate: результат [a] → {target}")
    return target


def run_calibration(config_path: str | Path, video: Optional[str] = None,
                    counter_id: str = "main_line",
                    window_name: str = "calibrate [l/z/s/m/a/q]",
                    save_to: Optional[str | Path] = None,
                    cache_frames: int = DEFAULT_CACHE_FRAMES,
                    initial_scale: float = 1.0) -> int:
    """Интерактивная калибровка.

    :param save_to: если задан (например, явный --config) — результат пишется строго туда;
        иначе по умолчанию рядом с видео: ``<имя_видео>.config.yaml``.
    :param cache_frames: сколько кадров держать в кэше листа ([n/p]) и после seek
        (клавиша [t]/кнопка «время», только для файла); по умолчанию 100.
    :returns: 0 — корректное завершение (запись опциональна).
    """
    if cache_frames < 1:
        raise ValueError(f"cache_frames: ожидалось >= 1, получено {cache_frames!r}")
    if not GuiPlayer.available():
        print(f"calibrate: ОШИБКА: {GuiPlayer.unavailable_reason()}", file=sys.stderr)
        return 1

    # загрузка конфига и цель сохранения — общий helper (cv2- и Qt-драйверы, задача 16)
    cfg, _err = load_calibration_config(config_path)
    if cfg is None:
        return 1
    if video:
        cfg.video.path = video
    if not cfg.video.path:
        print("calibrate: не задан источник видео — укажите --video или video.path в конфиге",
              file=sys.stderr)
        return 1
    save_target = resolve_save_target(cfg, config_path, save_to)

    # Вся логика «кадр за кадром» (источник, кэш кадров, детектор, оверлеи, кнопки,
    # seek, ROI-переоткрытие, save, уведомления) — в контроллере; cv2-окно здесь —
    # только тонкий драйвер (задача 15). Поведение без изменений: те же клавиши,
    # кнопки, сообщения, save, seek и авто-масштаб.
    from .calib_controller import CalibrationController   # ленивый импорт: нет цикла на уровне модулей

    try:
        ctrl = CalibrationController(
            cfg, counter_id=counter_id, cache_frames=cache_frames,
            initial_scale=initial_scale, window_name=window_name,
            save_to=save_target)
        ctrl.open()   # Pipeline + кэш первых кадров + seek к processing.frame_start
    except (VideoSourceError, ConfigError):
        # ошибка источника / ни одного кадра: сообщение уже напечатано в stderr;
        # источник закрываем строго до возврата (как раньше — в finally)
        ctrl.close()
        return 1

    apply_qt_env()   # только для cv2-окна (до создания HighGUI/QApplication)
    cv2.namedWindow(ctrl.window_name)

    def on_mouse(event, x, y, flags, param):
        if event not in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
            return
        # координаты мыши — в системе МАСШТАБИРОВАННОЙ картинки: переводим обратно
        # в координаты исходного кадра ДО hit-test кнопок и обработчиков клика
        # (hit-test панели и логика режимов — внутри ctrl.on_click)
        ux, uy = unscale_mouse(x, y, ctrl.scale, ctrl.width, ctrl.height)
        if event == cv2.EVENT_MOUSEMOVE:
            ctrl.on_mouse_move(ux, uy)   # live-превью pending «мерки роста»
        else:
            ctrl.on_click(ux, uy)

    cv2.setMouseCallback(ctrl.window_name, on_mouse)
    try:
        while True:
            img = ctrl.step()   # кадр ИСХОДНОГО разрешения со ВСЕМИ оверлеями; None — выход/нет кадров
            if img is None or ctrl.quit_requested:
                break
            # масштаб ОТОБРАЖЕНИЯ: overlay нарисован в исходном разрешении, только
            # перед imshow уменьшаем/увеличиваем кадр (обработка не затрагивается)
            if ctrl.scale == 1.0:
                disp = img
            else:
                interp = cv2.INTER_AREA if ctrl.scale < 1.0 else cv2.INTER_LINEAR
                disp = cv2.resize(
                    img, (max(1, int(round(ctrl.width * ctrl.scale))),
                          max(1, int(round(ctrl.height * ctrl.scale)))),
                    interpolation=interp)
            cv2.imshow(ctrl.window_name, disp)
            key = cv2.waitKey(1) & 0xFF
            ctrl.on_key(key)   # распределение клавиш — в контроллере (как раньше)
    finally:
        # источник НЕ закрывается во время окна — он нужен для seek ([t]);
        # close строго после destroyAllWindows
        cv2.destroyAllWindows()
        ctrl.close()

    if not ctrl.saved_once:
        print("calibrate: выход БЕЗ сохранения ([a] не нажимался или изменений не было)")
    return 0
