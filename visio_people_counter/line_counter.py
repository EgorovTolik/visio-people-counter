"""Счётчики пересечения: линия (LineCounter) и зона (ZoneCounter).

Реализация по §2 отчёта pi-research/04-implementation-notes.md.

Общий интерфейс :class:`BaseCounter`:

* ``update(objects, t_wall, *, t_video=None, frame_index=-1) -> list[CrossingEvent]`` —
  один кадр: список подтверждённых треков (:class:`~visio_people_counter.tracker_adapter.TrackedObject`)
  и время кадра (wall-clock, монотонный float; monotonic или epoch — неважно, мы
  считаем только РАЗНОСТИ);
* ``counters -> dict`` — текущие счётчики (both: ``{"in", "out", "total"}``;
  total: ``{"total"}``);
* ``draw(frame) -> None`` — простой overlay на BGR-кадр (линия/зона + счётчик).
  Доработка визуализации — задача 16.

Масштабирование координат
-------------------------
Решение: нормализованные координаты из конфига масштабируются **в конструкторе**
счётчика на фактическое разрешение ``w/h``, которое pipeline передаёт при создании
(после ffprobe/первого кадра). Counter не знает про источник кадров; при смене
разрешения нужно пересоздать счётчики (pipeline, задача 15).

Направление линии
-----------------
Линия A→B: ``d(P) = cross(B-A, P-A)`` — знаковая «сторона» точки. Событие возникает,
когда отрезок prev→cur меняет знак d. Конвенция (по §2.2 отчёта 04):

* объект, который ДО пересечения был на стороне ``d > 0`` (левая сторона вектора
  A→B в математической ориентации; при экранных координатах y-down — «верхне-левая»
  сторона линии), считается ``"in"``, из стороны ``d < 0`` — ``"out"``.
* Для типичной диагонали A=верхний-левый, B=нижний-правый это соответствует движению
  «слева-направо по направлению A→B = in», как в ТЗ задачи 14.

Пересечение проверяется по **отрезку prev→cur** (калиманово-сглаженная позиция трека),
а не по одной точке кадра; точка пересечения должна лежать на **отрезке** линии
(параметр ``u`` в [-0.05, 1.05] — небольшой запас на дискретизацию), а не на его
продолжении. Считаются только подтверждённые треки (``track_id != -1``).

Антидубль (по §2.3 отчёта 04)
------------------------------
1. **per-track cooldown** (wall-clock, ``cooldown_s``): повторное пересечение тем же
   track_id внутри окна не считается;
2. **буфер вокруг линии**: пока объект ещё «у линии» (расстояние до прямой <=
   ``buffer_width_scale * h_local``), повторный счёт заблокирован даже после cooldown —
   нужно выйти из буфера и вернуться. ``h_local`` = высота человека в точке пересечения
   из :class:`~visio_people_counter.size_profile.SizeProfile` (если передан и адаптивный);
   иначе фиксированный масштаб: ``0.5 * длина линии`` (решение задачи 14, т.к. без
   профиля «высоты человека» не определена — масштабируем от длины самой линии,
   что переживает смену разрешения);
3. **min_global_gap_s**: новое событие отбрасывается, если до предыдущего события этого
   счётчика меньше ``min_global_gap_s`` И точка пересечения ближе ``0.5 * h_local``
   (разрезает сдвоенные blob'ы одного человека; два человека в разных концах длинной
   линии внутри 0.3 c не глушат друг друга).

ZoneCounter: полигон, point-in-polygon (cv2.pointPolygonTest) по позиции трека;
``"in"`` = снаружи→внутрь, ``"out"`` = внутрь→снаружу. Антидубль тот же: per-track
cooldown + global gap (масштаб h_local для зоны = длина самой длинной стороны полигона).
Блока буфера вокруг границы нет (в конфиге зоны нет buffer_width_scale — по ТЗ задачи 14).

``count_mode`` влияет только на представление ``counters``: "both" — отдельно in/out,
"total" — общий счёт. Сами события обоих направлений фиксируются и логируются всегда.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Sequence

import cv2
import numpy as np

from .config import CounterConfig, LineCounterConfig, ZoneCounterConfig
from .size_profile import SizeProfile
from .tracker_adapter import TrackedObject

#: сколько секунд можно не видеть трек перед удалением его состояния (анти-утечка).
_STATE_PRUNE_AGE_S = 60.0


# ---------------------------------------------------------------------------
# Событие
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CrossingEvent:
    """Пересечение линии / границы зоны одним подтверждённым треком."""

    counter_id: str            # id счётчика из конфига
    direction: str             # "in" | "out"
    track_id: int              # id трека (всегда != -1)
    t_wall: float              # wall-clock времени кадра (определённое pipeline)
    x_px: float                # точка пересечения, px
    y_px: float                # px
    frame_index: int = -1      # индекс кадра для перемотки на запись
    t_video: Optional[float] = None  # время в исходном потоке, если известно

    def to_dict(self) -> dict:
        """Plain-dict для JSONL-лога (event_log.py)."""
        d = {
            "ts_wall": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "t_wall": self.t_wall,
            "counter_id": self.counter_id,
            "direction": self.direction,
            "track_id": self.track_id,
            "frame_index": self.frame_index,
            "x_px": round(float(self.x_px), 1),
            "y_px": round(float(self.y_px), 1),
        }
        if self.t_video is not None:
            d["t_video_s"] = round(float(self.t_video), 3)
        return d


# ---------------------------------------------------------------------------
# Вспомогательная геометрия
# ---------------------------------------------------------------------------

def _cross(v: np.ndarray, w: np.ndarray) -> float:
    """Скалярное (z-компонентное) векторное произведение 2D."""
    return float(v[0] * w[1] - v[1] * w[0])


def _segment_intersection(p1: np.ndarray, p2: np.ndarray,
                          p3: np.ndarray, p4: np.ndarray) -> Optional[np.ndarray]:
    """Пересечение ОТРЕЗКОВ p1p2 и p3p4 или None (параллельны/мимо)."""
    d1 = p2 - p1
    d2 = p4 - p3
    denom = _cross(d1, d2)
    if abs(denom) < 1e-9:
        return None
    t = _cross(p3 - p1, d2) / denom
    s = _cross(p3 - p1, d1) / denom
    if not (-1e-6 <= t <= 1.0 + 1e-6 and -1e-6 <= s <= 1.0 + 1e-6):
        return None
    return p1 + t * d1


@dataclass
class _TrackState:
    """Состояние одного трека внутри счётчика."""

    last_pos: Optional[tuple[float, float]] = None   # позиция в пред. кадре (cx, cy)
    last_event_t: Optional[float] = None             # wall-clock последнего счёта трека
    last_seen_t: float = 0.0                         # когда трек последний раз видели
    last_inside: Optional[bool] = None               # только для ZoneCounter


# ---------------------------------------------------------------------------
# Общий интерфейс
# ---------------------------------------------------------------------------

class BaseCounter(ABC):
    """Общий интерфейс счётчиков пересечения (линия/зона)."""

    def __init__(self, counter_id: str, count_mode: str,
                 cooldown_s: float, min_global_gap_s: float) -> None:
        if count_mode not in ("both", "total"):
            raise ValueError(f"count_mode: ожидалось 'both' или 'total', получено {count_mode!r}")
        if cooldown_s < 0 or min_global_gap_s < 0:
            raise ValueError("cooldown_s и min_global_gap_s должны быть >= 0")
        self.counter_id = counter_id
        self.count_mode = count_mode
        self.cooldown_s = float(cooldown_s)
        self.min_global_gap_s = float(min_global_gap_s)

        self._in_count = 0
        self._out_count = 0
        # глобальный антидребезг: время/точка последнего события счётчика
        self._last_event_t: Optional[float] = None
        self._last_event_pt: Optional[np.ndarray] = None
        self._tracks: dict[int, _TrackState] = {}

    # --- обязательный интерфейс -------------------------------------------------

    @abstractmethod
    def update(self, objects: Sequence[TrackedObject], t_wall: float, *,
               t_video: Optional[float] = None,
               frame_index: int = -1) -> list[CrossingEvent]:
        """Обработать кадр; вернуть события пересечения (обычно пустой список)."""

    @property
    def counters(self) -> dict:
        """Текущие счётчики. both: {"in","out","total"}; total: {"total"}."""
        if self.count_mode == "both":
            return {"in": self._in_count, "out": self._out_count,
                    "total": self._in_count + self._out_count}
        return {"total": self._in_count + self._out_count}

    def draw(self, frame: Optional[np.ndarray]) -> None:  # noqa: B027
        """Нарисовать overlay на BGR-кадр (заглушка; реализуют наследники)."""

    # --- общее состояние ---------------------------------------------------------

    def _global_gap_ok(self, t_wall: float, pt: np.ndarray, ref_scale: float) -> bool:
        """min_global_gap_s + расстояние < 0.5*ref_scale от предыдущего события."""
        if self._last_event_t is None or self._last_event_pt is None:
            return True
        if (t_wall - self._last_event_t) >= self.min_global_gap_s:
            return True
        return float(np.linalg.norm(pt - self._last_event_pt)) > 0.5 * ref_scale

    def _register_event(self, ev: CrossingEvent) -> None:
        if ev.direction == "in":
            self._in_count += 1
        else:
            self._out_count += 1
        self._last_event_t = ev.t_wall
        self._last_event_pt = np.array([ev.x_px, ev.y_px], dtype=float)

    def _prune_stale(self, t_wall: float) -> None:
        """Удалить состояния давно не виденных треков (анти-утечка памяти)."""
        stale = [tid for tid, st in self._tracks.items()
                 if t_wall - st.last_seen_t > _STATE_PRUNE_AGE_S]
        for tid in stale:
            del self._tracks[tid]

    def reset(self) -> None:
        """Сбросить состояния треков (last_pos/last_inside/cooldown-метки).

        Вызывается pipeline при reconnect источника: трекер после ``reset()``
        снова нумерует id с нуля, и старые записи могли бы дать ложное «пересечение»
        от stale last_pos. Накопленные счётчики in/out при этом НЕ обнуляются.
        """
        self._tracks.clear()

    def label_text(self) -> str:
        """Короткая строка «id: in=N out=M» / «id: total=N» для overlay."""
        c = self.counters
        if self.count_mode == "both":
            return f"{self.counter_id}: in={c['in']} out={c['out']}"
        return f"{self.counter_id}: total={c['total']}"


# ---------------------------------------------------------------------------
# Линия
# ---------------------------------------------------------------------------

class LineCounter(BaseCounter):
    """Счётчик по наклонной линии A→B (нормализованные координаты → px в конструкторе)."""

    def __init__(self, cfg: LineCounterConfig, w: int, h: int,
                 size_profile: Optional[SizeProfile] = None) -> None:
        super().__init__(cfg.id, cfg.count_mode, cfg.cooldown_s, cfg.min_global_gap_s)
        if w <= 0 or h <= 0:
            raise ValueError(f"размеры кадра: ожидалось w>0 и h>0, получено {w}x{h}")
        self._a = np.array([cfg.a[0] * w, cfg.a[1] * h], dtype=float)
        self._b = np.array([cfg.b[0] * w, cfg.b[1] * h], dtype=float)
        self._ab = self._b - self._a
        self._len2 = float(self._ab @ self._ab)
        if self._len2 < 1e-6:
            raise ValueError(f"счётчик {cfg.id}: точки A и B совпадают — нулевая линия")
        self.line_len = float(np.sqrt(self._len2))
        self.buffer_width_scale = float(cfg.buffer_width_scale)
        self._sp = size_profile

    # --- локальный масштаб (высота человека / фиксированный от длины линии) -----

    def _ref_scale(self, ip: np.ndarray) -> float:
        """h_local в точке пересечения: из SizeProfile либо 0.5*длина линии."""
        if self._sp is not None and self._sp.adaptive:
            return max(1.0, self._sp.person_height_px(float(ip[0])))
        return 0.5 * self.line_len

    # --- обновление --------------------------------------------------------------

    def update(self, objects: Sequence[TrackedObject], t_wall: float, *,
               t_video: Optional[float] = None,
               frame_index: int = -1) -> list[CrossingEvent]:
        events: list[CrossingEvent] = []
        for obj in objects:
            if obj.track_id == -1:  # неподтверждённый — не считаем и не храним
                continue
            st = self._tracks.get(obj.track_id)
            if st is None:
                st = _TrackState(last_pos=(float(obj.cx), float(obj.cy)))
                self._tracks[obj.track_id] = st
            else:
                cur = np.array([obj.cx, obj.cy], dtype=float)
                ev = self._check_cross(st, obj.track_id, cur, t_wall, t_video, frame_index)
                if ev is not None:
                    events.append(ev)
                # ВАЖНО: last_pos обновляем ВСЕГДА (даже в cooldown), иначе после
                # паузы первый шаг станет «длинным» отрезком и сломает геометрию.
                st.last_pos = (float(obj.cx), float(obj.cy))
            st.last_seen_t = t_wall
        self._prune_stale(t_wall)
        return events

    def _check_cross(self, st: _TrackState, track_id: int, cur: np.ndarray,
                     t_wall: float, t_video: Optional[float],
                     frame_index: int) -> Optional[CrossingEvent]:
        prev = np.array(st.last_pos, dtype=float)
        # 1) стороны прямой для prev и cur
        d_prev = _cross(self._ab, prev - self._a)
        d_cur = _cross(self._ab, cur - self._a)
        if d_prev == 0.0 or d_cur == 0.0 or d_prev * d_cur > 0.0:
            return None  # на одной стороне (или прямо на линии) — нет пересечения
        # 2) точка пересечения отрезков prev->cur × A->B
        t = d_prev / (d_prev - d_cur)
        ip = prev + t * (cur - prev)
        u = float((ip - self._a) @ self._ab) / self._len2
        if not (-0.05 <= u <= 1.05):
            return None  # пересёк ПРОДОЛЖЕНИЕ линии — не считаем
        # 3) антидубль: per-track cooldown + буфер вокруг линии
        ref = self._ref_scale(ip)
        if st.last_event_t is not None:
            dt = t_wall - st.last_event_t
            if dt < self.cooldown_s:
                return None
            dist_to_line = abs(d_cur) / self.line_len
            if dist_to_line <= self.buffer_width_scale * ref:
                return None  # объект ещё «у линии» — ждём ухода из буфера
        # 4) глобальный антидребезг (время + расстояние)
        if not self._global_gap_ok(t_wall, ip, ref):
            return None
        direction = "in" if d_prev > 0.0 else "out"
        ev = CrossingEvent(counter_id=self.counter_id, direction=direction,
                           track_id=track_id, t_wall=t_wall, x_px=float(ip[0]),
                           y_px=float(ip[1]), frame_index=frame_index, t_video=t_video)
        st.last_event_t = t_wall
        self._register_event(ev)
        return ev

    # --- overlay -------------------------------------------------------------------

    def draw(self, frame: Optional[np.ndarray]) -> None:
        if frame is None:
            return
        a = (int(round(self._a[0])), int(round(self._a[1])))
        b = (int(round(self._b[0])), int(round(self._b[1])))
        cv2.line(frame, a, b, (0, 255, 0), 2)
        cv2.putText(frame, self.label_text(), (max(0, a[0] + 8), max(16, a[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Зона
# ---------------------------------------------------------------------------

class ZoneCounter(BaseCounter):
    """Счётчик по полигону: "in" = снаружи→внутрь, "out" = внутрь→снаружу.

    Point-in-polygon — ``cv2.pointPolygonTest`` по позиции трека (cx, cy).
    Точка события — пересечение отрезка prev→cur с ближайшей стороной полигона
    (если не находится — позиция трека в кадре обнаружения).
    """

    def __init__(self, cfg: ZoneCounterConfig, w: int, h: int) -> None:
        super().__init__(cfg.id, cfg.count_mode, cfg.cooldown_s, cfg.min_global_gap_s)
        if w <= 0 or h <= 0:
            raise ValueError(f"размеры кадра: ожидалось w>0 и h>0, получено {w}x{h}")
        self._poly = np.array([(p[0] * w, p[1] * h) for p in cfg.polygon], dtype=float)
        if len(self._poly) < 3:
            raise ValueError(f"счётчик {cfg.id}: ожидалось >= 3 точек полигона")
        self._poly_cv = self._poly.astype(np.float32).reshape(-1, 1, 2)
        # масштаб для global gap — длина самой длинной стороны (документируется в отчёте)
        edges = np.diff(self._poly, axis=0, append=self._poly[:1])
        self._ref_scale_value = float(np.max(np.linalg.norm(edges, axis=1)))

    def _inside(self, pt: np.ndarray) -> bool:
        return cv2.pointPolygonTest(self._poly_cv, (float(pt[0]), float(pt[1])), False) >= 0

    def _boundary_point(self, prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
        """Первая точка пересечения отрезка prev→cur со сторонами полигона."""
        best_t = 1.0
        best_pt = None
        n = len(self._poly)
        for i in range(n):
            p = _segment_intersection(prev, cur, self._poly[i], self._poly[(i + 1) % n])
            if p is not None:
                t = float(np.linalg.norm(p - prev)) / max(1e-9, float(np.linalg.norm(cur - prev)))
                if t < best_t:
                    best_t, best_pt = t, p
        return best_pt if best_pt is not None else cur

    def update(self, objects: Sequence[TrackedObject], t_wall: float, *,
               t_video: Optional[float] = None,
               frame_index: int = -1) -> list[CrossingEvent]:
        events: list[CrossingEvent] = []
        for obj in objects:
            if obj.track_id == -1:
                continue
            st = self._tracks.get(obj.track_id)
            cur = np.array([obj.cx, obj.cy], dtype=float)
            if st is None:
                st = _TrackState(last_pos=(float(obj.cx), float(obj.cy)))
                st.last_inside = self._inside(cur)
                self._tracks[obj.track_id] = st
            else:
                inside = self._inside(cur)
                if st.last_inside is not None and inside != st.last_inside:
                    ev = self._check_boundary(st, obj.track_id, cur, inside,
                                              t_wall, t_video, frame_index)
                    if ev is not None:
                        events.append(ev)
                st.last_pos = (float(obj.cx), float(obj.cy))
                st.last_inside = inside
            st.last_seen_t = t_wall
        self._prune_stale(t_wall)
        return events

    def _check_boundary(self, st: _TrackState, track_id: int, cur: np.ndarray,
                        inside_now: bool, t_wall: float,
                        t_video: Optional[float], frame_index: int) -> Optional[CrossingEvent]:
        # антидубль: per-track cooldown (буфера вокруг границы в конфиге зоны нет)
        if st.last_event_t is not None and (t_wall - st.last_event_t) < self.cooldown_s:
            return None
        prev = np.array(st.last_pos, dtype=float)
        pt = self._boundary_point(prev, cur) if float(np.linalg.norm(cur - prev)) > 1e-9 else cur
        if not self._global_gap_ok(t_wall, pt, self._ref_scale_value):
            return None
        direction = "in" if inside_now else "out"
        ev = CrossingEvent(counter_id=self.counter_id, direction=direction,
                           track_id=track_id, t_wall=t_wall, x_px=float(pt[0]),
                           y_px=float(pt[1]), frame_index=frame_index, t_video=t_video)
        st.last_event_t = t_wall
        self._register_event(ev)
        return ev

    def draw(self, frame: Optional[np.ndarray]) -> None:
        if frame is None:
            return
        pts = np.round(self._poly).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
        x, y = int(self._poly[0][0]), int(self._poly[0][1])
        cv2.putText(frame, self.label_text(), (x + 8, max(16, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Фабрика для pipeline (задача 15)
# ---------------------------------------------------------------------------

def build_counters(counter_cfgs: Sequence[CounterConfig], w: int, h: int,
                   size_profile: Optional[SizeProfile] = None) -> list[BaseCounter]:
    """Создать счётчики из блока ``cfg.counters`` при разрешении кадра w×h."""
    out: list[BaseCounter] = []
    for c in counter_cfgs:
        if isinstance(c, LineCounterConfig):
            out.append(LineCounter(c, w=w, h=h, size_profile=size_profile))
        elif isinstance(c, ZoneCounterConfig):
            out.append(ZoneCounter(c, w=w, h=h))
        else:  # pragma: no cover — Config не создаёт других типов
            raise TypeError(f"неизвестный тип счётчика: {type(c).__name__}")
    return out
