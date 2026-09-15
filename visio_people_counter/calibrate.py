"""GUI-калибровка линии/зоны/size-точек мышью → запись в config.yaml (задача 16).

``python -m visio_people_counter calibrate --config config.yaml
   --video PATH_OR_URL [--counter-id main_line]``

Режимы по клавишам (см. :func:`run_calibration`):

* **l** + 2 клика — линия A→B для выбранного счётчика (порядок кликов = направление «in»);
* **z** + N кликов, **Enter** — замкнуть полигон зоны (>= 3 точки);
* **s** + клик (x) + цифра **1..9** (высота человека как % высоты кадра:
  1=5%, 2=10%, ..., 9=45%), **Enter** — завершить набор size-точек;
* **m** — toggle показа текущей маски движения (live-подстройка порога);
* **n/p** — следующий/предыдущий из загруженных кадров;
* **a** — применить и записать в config.yaml (остальные настройки сохраняются),
  печатает короткий diff «старые → новые координаты»;
* **q/ESC** — выход без записи.

Вся логика «клик → обновление конфига» вынесена в чистые функции
(:class:`CalibrationState`, :func:`click_to_norm`, :func:`size_digit_to_fraction`,
:func:`apply_calibration`) и тестируется БЕЗ окна (tests/test_gui_calibrate.py).
"""

from __future__ import annotations

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
)
from .gui import GuiPlayer
from .text_overlay import put_text
from .motion_detector import MotionDetector
from .pipeline import Pipeline
from .video_source import VideoSourceError

#: сколько первых кадров видео загрузить для калибровки (для HLS — первые кадры).
MAX_FRAMES = 20


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


def size_digit_to_fraction(digit: int) -> float:
    """Цифра 1..9 → высота человека в долях высоты кадра (1=5% ... 9=45%).

    :raises ValueError: цифра вне 1..9.
    """
    if not isinstance(digit, int) or isinstance(digit, bool) or not (1 <= digit <= 9):
        raise ValueError(f"цифра высоты: ожидалось int 1..9, получено {digit!r}")
    return round(0.05 * digit, 2)


@dataclass
class CalibrationState:
    """Собранное в окне калибровки (чистое состояние — без cv2)."""

    counter_id: str = ""
    mode: Optional[str] = None   # "line" | "zone" | "size" | None
    line_points: list[tuple[float, float]] = field(default_factory=list)   # норм. (x, y)
    zone_points: list[tuple[float, float]] = field(default_factory=list)   # норм. (x, y)
    size_points: list[tuple[float, float]] = field(default_factory=list)   # [x_frac, h_frac]
    _size_x: Optional[float] = None  # x последней 's'-точки, ждёт цифру

    def set_mode(self, mode: str) -> None:
        """Переключить режим ('l'/'z'/'s'); переключение очищает НОВЫЙ набор."""
        if mode not in ("line", "zone", "size"):
            raise ValueError(f"режим калибровки: ожидалось line/zone/size, получено {mode!r}")
        self.mode = mode
        if mode == "line":
            self.line_points = []
            self.zone_points = []      # один счётчик — либо линия, либо зона
        elif mode == "zone":
            self.zone_points = []
            self.line_points = []
            self._size_x = None
        else:  # size: точки накапливаются до Enter
            pass

    def handle_click(self, x_norm: float, y_norm: float) -> None:
        """Клик в текущем режиме (координаты уже нормализованы)."""
        if self.mode == "line":
            if len(self.line_points) >= 2:
                self.line_points = []  # третья точка — начать линию заново
            self.line_points.append((x_norm, y_norm))
        elif self.mode == "zone":
            self.zone_points.append((x_norm, y_norm))
        elif self.mode == "size":
            self._size_x = x_norm      # высоту задаст следующая цифра 1..9

    def set_size_height(self, digit: int) -> None:
        """Цифра 1..9 в режиме 's': фиксирует высоту точки (x — из клика)."""
        if self.mode != "size" or self._size_x is None:
            return
        self.size_points.append((self._size_x, size_digit_to_fraction(digit)))
        self._size_x = None

    def finish_zone(self) -> list[tuple[float, float]]:
        """Enter в режиме 'z': закрыть полигон. :raises ValueError: < 3 точек."""
        if len(self.zone_points) < 3:
            raise ValueError(
                f"зона: ожидалось >= 3 клика, получено {len(self.zone_points)}")
        return list(self.zone_points)

    def finish_size(self) -> None:
        """Enter в режиме 's': завершить набор size-точек."""
        self._size_x = None


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
    * size-точки: ``size_profile.control_points`` + ``enabled=True``.

    :returns: список коротких строк diff «старое → новое» (пусто — нечего применить).
    """
    changed: list[str] = []
    cid = state.counter_id

    if len(state.line_points) == 2:
        a, b = state.line_points
        c = _find_counter(cfg, cid, LineCounterConfig)
        if c is None:
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
# Интерактивный цикл (окно — только здесь)
# ---------------------------------------------------------------------------

#: Подсказки на экране (рисуются через text_overlay.put_text — кириллица поддерживается).
_HINTS = {
    None: "режимы: [l]иния 2 клика | [z]она N кликов+Enter | [s]ize точка+цифра 1-9\n"
          "[m]аска движения | [n/p] кадр вперёд/назад | [a]применить и сохранить | [q]выход",
    "line": "ЛИНИЯ: кликните точку A, затем B (порядок = направление in); повторный 'l' — заново",
    "zone": "ЗОНА: кликайте углы полигона; Enter — замкнуть (>=3), 'z' — начать заново",
    "size": "SIZE: кликните X-точку, затем цифру 1-9 (1=5% ... 9=45% высоты кадра); Enter — завершить",
}




def _draw_calibration(base: np.ndarray, state: CalibrationState, mask_on: bool,
                      mask: Optional[np.ndarray], w: int, h: int) -> np.ndarray:
    """Отрисовка состояния калибровки на копии кадра (чистая функция)."""
    img = base.copy()
    if mask_on and mask is not None:
        m = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET)
        img[:] = cv2.addWeighted(img, 0.5, m, 0.5, 0.0)

    def px(p: tuple[float, float]) -> tuple[int, int]:
        return (int(round(p[0] * w)), int(round(p[1] * h)))

    if state.line_points:
        pts = [px(p) for p in state.line_points]
        for i, p in enumerate(pts):
            cv2.circle(img, p, 5, (0, 255, 0), -1)
            cv2.putText(img, "A" if i == 0 else "B", (p[0] + 8, p[1] + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        if len(pts) == 2:
            cv2.line(img, pts[0], pts[1], (0, 255, 0), 2, lineType=cv2.LINE_AA)

    if state.zone_points:
        pts = [px(p) for p in state.zone_points]
        for p in pts:
            cv2.circle(img, p, 4, (0, 200, 255), -1)
        if len(pts) >= 2:
            cv2.polylines(img, [np.array(pts, dtype=np.int32)], False,
                          (0, 200, 255), 2, lineType=cv2.LINE_AA)

    for x_f, h_f in state.size_points:
        # size-точка = (x_frac, доля высоты человека): вертикальный отрезок
        # нужной длины, центрированный по y=1/2 кадра
        x = int(round(x_f * w))
        yc = h // 2
        seg_h = int(round(h_f * h))
        cv2.line(img, (x, yc - seg_h // 2), (x, yc + seg_h // 2), (0, 255, 255), 2)
        cv2.putText(img, f"{int(round(h_f * 100))}%", (x + 6, yc),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    # подсказки сверху
    y = 10
    for line in _HINTS[state.mode].split("\n"):
        put_text(img, line, (10, y), size_px=16, color=(255, 255, 255))
        y += 24
    return img


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


def run_calibration(config_path: str | Path, video: Optional[str] = None,
                    counter_id: str = "main_line",
                    window_name: str = "calibrate [l/z/s/m/a/q]",
                    save_to: Optional[str | Path] = None) -> int:
    """Интерактивная калибровка.

    :param save_to: если задан (например, явный --config) — результат пишется строго туда;
        иначе по умолчанию рядом с видео: ``<имя_видео>.config.yaml``.
    :returns: 0 — корректное завершение (запись опциональна).
    """
    if not GuiPlayer.available():
        print(f"calibrate: ОШИБКА: {GuiPlayer.unavailable_reason()}", file=sys.stderr)
        return 1

    if Path(config_path).is_file():
        try:
            cfg = Config.load(config_path)
        except ConfigError as e:
            print(f"calibrate: ошибка конфигурации: {e}", file=sys.stderr)
            return 1
    else:
        # первый запуск: файла ещё нет — калибруем по дефолтам и создадим файл при [a]
        cfg = Config.default()
        print(f"calibrate: файл {config_path} не найден — запускаю с настройками по умолчанию; "
              f"результат будет сохранён рядом с видео (<имя_видео>.config.yaml) при [a] "
              f"(остальные режимы требуют готовый конфиг)")
    if video:
        cfg.video.path = video
    if not cfg.video.path:
        print("calibrate: не задан источник видео — укажите --video или video.path в конфиге",
              file=sys.stderr)
        return 1

    # источник + первые кадры (для HLS — первые кадры потока)
    pipe = None
    try:
        pipe = Pipeline(cfg).build()
    except (VideoSourceError, ConfigError) as e:
        if pipe is not None:
            pipe.close()  # build мог упасть после source.open() — не утекает ресурс
        print(f"calibrate: ошибка источника: {e}", file=sys.stderr)
        return 1
    w, h = pipe.source.width, pipe.source.height

    # Куда сохранять при [a]: явный save_to (--config) → строго туда; иначе рядом
    # с файлом видео как <имя_видео>.config.yaml; для HLS/URL фолбэк на --config.
    if save_to is not None:
        save_target = Path(save_to)
    else:
        save_target = calibration_save_target(cfg.video.type, cfg.video.path, config_path)
        if cfg.video.type != "file":
            print(f"calibrate: источник HLS/URL — конфиг будет сохранён в {save_target} "
                  f"(«рядом с видео» для потока не определено)")
    print(f"calibrate: результат [a] → {save_target}")

    detector = MotionDetector(cfg)   # для 'm' — live-маска движения
    frames: list[np.ndarray] = []
    masks: list[Optional[np.ndarray]] = []
    try:
        while len(frames) < MAX_FRAMES:
            fr = pipe.source.read()
            if fr is None:
                break
            detector.detect(fr.image)
            frames.append(fr.image.copy())
            masks.append(getattr(detector, "last_mask", None))
    finally:
        # кадры уже загружены — источник больше не нужен
        pipe.close()

    if not frames:
        print("calibrate: ОШИБКА: не удалось прочитать ни одного кадра", file=sys.stderr)
        return 1
    print(f"calibrate: загрузил {len(frames)} кадр(ов) {w}x{h}, counter-id={counter_id!r}; "
          f"[n/p] — листать, [a] — сохранить в {config_path}")

    state = CalibrationState(counter_id=counter_id)
    mask_on = False
    idx = 0
    saved_once = False
    pending_msgs: list[str] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            try:
                state.handle_click(*click_to_norm(x, y, w, h))
            except ValueError as e:
                pending_msgs.append(str(e))

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)
    try:
        while True:
            base = frames[idx]
            img = _draw_calibration(base, state, mask_on, masks[idx], w, h)
            hint_lines = _HINTS[state.mode].split("\n")
            for i, msg in enumerate(pending_msgs):
                put_text(img, msg[:80], (10, 34 + 24 * len(hint_lines) + i * 20),
                         size_px=14, color=(0, 0, 255))
            pending_msgs.clear()
            cv2.imshow(window_name, img)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            elif key == ord("l"):
                state.set_mode("line")
            elif key == ord("z"):
                state.set_mode("zone")
            elif key == ord("s"):
                state.set_mode("size")
            elif key == 13:  # Enter — замкнуть зону / завершить size-точки
                if state.mode == "zone":
                    try:
                        state.finish_zone()
                        pending_msgs.append(f"зона закрыта: {len(state.zone_points)} точек")
                    except ValueError as e:
                        pending_msgs.append(str(e))
                elif state.mode == "size":
                    state.finish_size()
            elif key == ord("m"):
                mask_on = not mask_on
            elif key == ord("n"):
                idx = (idx + 1) % len(frames)
            elif key == ord("p"):
                idx = (idx - 1) % len(frames)
            elif key in tuple(ord(d) for d in "123456789") and state.mode == "size":
                state.set_size_height(key - ord("0"))
            elif key == ord("a"):
                try:
                    changed = apply_calibration(cfg, state)
                    if not changed:
                        pending_msgs.append(
                        "ничего не менялось — нет собранных линий/зон/size-точек")
                    else:
                        Config.save(cfg, save_target)
                        saved_once = True
                        print(f"calibrate: сохранено в {save_target}:")
                        for line in changed:
                            print(f"  - {line}")
                        print(f"calibrate: дальше — "
                              f".venv/bin/python -m visio_people_counter count --config {save_target}")
                except (ConfigError, OSError) as e:
                    pending_msgs.append(f"ошибка записи: {e}")
    finally:
        cv2.destroyAllWindows()

    if not saved_once:
        print("calibrate: выход БЕЗ сохранения ([a] не нажимался или изменений не было)")
    return 0
