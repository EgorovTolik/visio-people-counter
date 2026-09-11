"""Тесты line_counter.py + event_log.py на чистой геометрии (без видео).

Синтетика: «люди» — TrackedObject с заданным (cx, cy), перемещаются по прямой.
Кадр условно 640×480; линия вертикальная x=320 (A=(0.5,0) → B=(0.5,1)):
движение слева-направо = "in" (конвенция §2 отчёта 04: d_prev > 0 → in).

Проверяется (по ТЗ задачи 14):
* пересечение слева-направо → 1 событие "in"; обратно → "out"; движение ВДОЛЬ
  линии — 0; «пересечение» за пределами отрезка (у продолжения) — 0;
* антидубль: туда-обратно быстрее cooldown_s → максимум 1 событие; медленнее
  (с выходом из буфера) → засчитано;
* min_global_gap_s: два события < 0.3 с друг от друга → второе отброшено;
* неподтверждённый трек (track_id=-1) не считается;
* зона: вход/выход полигона, движение внутри без пересечения границы — 0;
  count_mode=total суммирует оба направления;
* EventLog: JSONL-файл корректен (json.loads по строкам), агрегаты сходятся.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

# корень проекта — родитель каталога tests/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from visio_people_counter.config import (  # noqa: E402
    LineCounterConfig,
    OutputConfig,
    ZoneCounterConfig,
)
from visio_people_counter.event_log import EventLog  # noqa: E402
from visio_people_counter.line_counter import (  # noqa: E402
    CrossingEvent,
    LineCounter,
    ZoneCounter,
)
from visio_people_counter.tracker_adapter import TrackedObject  # noqa: E402

W, H = 640, 480


def make_obj(track_id: int, x: float, y: float) -> TrackedObject:
    """Подтверждённый трек (age_frames>0), центр в (x, y)."""
    return TrackedObject(
        track_id=track_id,
        x=int(x - 30), y=int(y - 60), w=60, h=120,
        cx=float(x), cy=float(y), age_frames=5,
    )


def make_line(**overrides) -> LineCounter:
    """Вертикальная линия x=320 (A верх → B низ), типовые антидубль-параметры."""
    t = {
        "id": "test_line",
        "type": "line",
        "a": (0.5, 0.0),
        "b": (0.5, 1.0),
        "count_mode": "both",
        "cooldown_s": 2.0,
        "buffer_width_scale": 0.75,
        "min_global_gap_s": 0.3,
    }
    t.update(overrides)
    return LineCounter(LineCounterConfig(**t), w=W, h=H)


class TestLineCounterGeometry(unittest.TestCase):
    def test_crossing_left_to_right_is_in(self):
        lc = make_line()
        self.assertEqual(lc.update([make_obj(1, 200, 240)], t_wall=0.0), [])
        evs = lc.update([make_obj(1, 440, 240)], t_wall=0.1, frame_index=7)
        self.assertEqual(len(evs), 1)
        ev = evs[0]
        self.assertIsInstance(ev, CrossingEvent)
        self.assertEqual(ev.direction, "in")
        self.assertEqual(ev.counter_id, "test_line")
        self.assertEqual(ev.track_id, 1)
        self.assertEqual(ev.frame_index, 7)
        # точка пересечения — на линии x=320, y≈240
        self.assertAlmostEqual(ev.x_px, 320.0, delta=1.0)
        self.assertAlmostEqual(ev.y_px, 240.0, delta=1.0)
        self.assertEqual(lc.counters, {"in": 1, "out": 0, "total": 1})

    def test_crossing_right_to_left_is_out(self):
        lc = make_line()
        lc.update([make_obj(1, 440, 240)], t_wall=0.0)
        evs = lc.update([make_obj(1, 200, 240)], t_wall=0.1)
        self.assertEqual(len(evs), 1)
        self.assertEqual(evs[0].direction, "out")
        self.assertEqual(lc.counters["out"], 1)

    def test_motion_along_line_no_events(self):
        lc = make_line()
        # параллельно линии (x=330, с той же стороны) — знаков d нет
        for t in (0.0, 0.1, 0.2):
            self.assertEqual(lc.update([make_obj(1, 330, 100 + 100 * t)], t_wall=t), [])
        # прямо НА линии — тоже 0
        lc2 = make_line()
        for t in (0.0, 0.1, 0.2):
            self.assertEqual(lc2.update([make_obj(1, 320, 100 + 100 * t)], t_wall=t), [])
        self.assertEqual(lc.counters["total"], 0)
        self.assertEqual(lc2.counters["total"], 0)

    def test_crossing_beyond_segment_not_counted(self):
        # короткая горизонтальная линия x=128..256, y=240; пересечение у x=400 —
        # это продолжение линии, не отрезок → 0 событий
        lc = make_line(id="seg", a=(0.2, 0.5), b=(0.4, 0.5))
        self.assertEqual(lc.update([make_obj(1, 400, 200)], t_wall=0.0), [])
        self.assertEqual(lc.update([make_obj(1, 400, 280)], t_wall=0.1), [])
        self.assertEqual(lc.counters["total"], 0)


class TestLineCounterAntiDouble(unittest.TestCase):
    def test_fast_bounce_within_cooldown_max_one_event(self):
        lc = make_line(cooldown_s=2.0)
        seq = [(200, 240), (440, 240), (200, 240), (440, 240)]  # туда-обратно у линии
        all_evs = []
        for i, (x, y) in enumerate(seq):
            all_evs += lc.update([make_obj(1, x, y)], t_wall=0.15 * i)
        self.assertEqual(len(all_evs), 1)  # максимум одно событие
        self.assertEqual(lc.counters["total"], 1)

    def test_slow_return_after_buffer_exit_counted(self):
        lc = make_line(cooldown_s=0.5)
        evs = []
        evs += lc.update([make_obj(1, 200, 240)], t_wall=0.0)
        evs += lc.update([make_obj(1, 440, 240)], t_wall=0.1)   # "in" засчитан
        evs += lc.update([make_obj(1, 600, 240)], t_wall=0.2)    # ушёл из буфера (|600-320|=280 > 180)
        evs += lc.update([make_obj(1, 600, 240)], t_wall=1.5)    # ждёт (cooldown истёк)
        evs += lc.update([make_obj(1, 100, 240)], t_wall=1.6)    # вернулся: "out" засчитан
        self.assertEqual(len(evs), 2)
        self.assertEqual(evs[0].direction, "in")
        self.assertEqual(evs[1].direction, "out")
        self.assertEqual(lc.counters["total"], 2)

    def test_global_gap_drops_second_event(self):
        lc = make_line(min_global_gap_s=0.3)
        evs = []
        evs += lc.update([make_obj(1, 200, 240), make_obj(2, 260, 300)], t_wall=0.0)
        # два трека пересекаются через 0.05 с в ~60 px друг от друга (< 0.5*h_local):
        # второе событие глушится min_global_gap_s
        evs += lc.update([make_obj(1, 440, 240), make_obj(2, 500, 300)], t_wall=0.05)
        self.assertEqual(len(evs), 1)
        self.assertEqual(evs[0].track_id, 1)
        self.assertEqual(lc.counters["total"], 1)

    def test_unconfirmed_track_not_counted(self):
        lc = make_line()
        self.assertEqual(lc.update([make_obj(-1, 200, 240)], t_wall=0.0), [])
        self.assertEqual(lc.update([make_obj(-1, 440, 240)], t_wall=0.1), [])
        self.assertEqual(lc.counters["total"], 0)


class TestZoneCounter(unittest.TestCase):
    def _zone(self, **overrides) -> ZoneCounter:
        t = {
            "id": "test_zone",
            "type": "zone",
            "polygon": [(0.55, 0.6), (0.95, 0.6), (0.95, 0.95), (0.55, 0.95)],
            "count_mode": "total",
            "cooldown_s": 0.5,
            "min_global_gap_s": 0.3,
        }
        t.update(overrides)
        return ZoneCounter(ZoneCounterConfig(**t), w=W, h=H)

    def test_enter_exit_and_stay_inside(self):
        # полигон в px: x 352..608, y 288..456
        zc = self._zone()
        evs = []
        evs += zc.update([make_obj(1, 300, 370)], t_wall=0.0)   # снаружи
        evs += zc.update([make_obj(1, 450, 370)], t_wall=0.1)   # пересёк границу → "in"
        self.assertEqual(len(evs), 1)
        self.assertEqual(evs[0].direction, "in")
        # точка события — на границе x≈352
        self.assertAlmostEqual(evs[0].x_px, 352.0, delta=2.0)
        evs += zc.update([make_obj(1, 500, 370)], t_wall=0.2)   # внутри, без пересечения → 0
        self.assertEqual(len(evs), 1)
        evs += zc.update([make_obj(1, 250, 370)], t_wall=0.6)   # вышел → "out"
        self.assertEqual(len(evs), 2)
        self.assertEqual(evs[1].direction, "out")
        # count_mode=total: оба направления в один счётчик
        self.assertEqual(zc.counters, {"total": 2})

    def test_count_mode_both_separates_directions(self):
        zc = self._zone(count_mode="both")
        zc.update([make_obj(1, 300, 370)], t_wall=0.0)
        zc.update([make_obj(1, 450, 370)], t_wall=0.1)
        self.assertEqual(zc.counters, {"in": 1, "out": 0, "total": 1})

    def test_bounce_at_boundary_within_cooldown_max_one_event(self):
        zc = self._zone(cooldown_s=2.0)
        all_evs = []
        for i, (x, y) in enumerate([(300, 370), (450, 370), (300, 370), (450, 370)]):
            all_evs += zc.update([make_obj(1, x, y)], t_wall=0.1 * i)
        self.assertEqual(len(all_evs), 1)

    def test_unconfirmed_track_not_counted(self):
        zc = self._zone()
        self.assertEqual(zc.update([make_obj(-1, 300, 370)], t_wall=0.0), [])
        self.assertEqual(zc.update([make_obj(-1, 450, 370)], t_wall=0.1), [])
        self.assertEqual(zc.counters["total"], 0)


class TestCrossingEvent(unittest.TestCase):
    def test_to_dict_fields(self):
        ev = CrossingEvent(counter_id="l1", direction="in", track_id=7,
                           t_wall=123.4, x_px=320.123, y_px=240.5,
                           frame_index=99, t_video=15.25)
        d = ev.to_dict()
        self.assertEqual(d["counter_id"], "l1")
        self.assertEqual(d["direction"], "in")
        self.assertEqual(d["track_id"], 7)
        self.assertEqual(d["frame_index"], 99)
        self.assertAlmostEqual(d["x_px"], 320.1, places=1)
        self.assertAlmostEqual(d["t_video_s"], 15.25)
        self.assertIn("ts_wall", d)
        # JSON-сериализуемо
        json.dumps(d)
        # t_video=None → поля нет
        ev2 = CrossingEvent(counter_id="l1", direction="out", track_id=7,
                            t_wall=1.0, x_px=1.0, y_px=1.0)
        self.assertNotIn("t_video_s", ev2.to_dict())


class TestEventLog(unittest.TestCase):
    def _ev(self, counter_id: str, direction: str, track_id: int, t_wall: float,
            x: float = 320.0, y: float = 240.0) -> CrossingEvent:
        return CrossingEvent(counter_id=counter_id, direction=direction,
                             track_id=track_id, t_wall=t_wall, x_px=x, y_px=y,
                             frame_index=int(t_wall * 10), t_video=float(int(t_wall)))

    def test_jsonl_and_aggregates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "events.jsonl"
            el = EventLog(OutputConfig(events_jsonl=str(path)))
            el.log_events([
                self._ev("main_line", "in", 1, 0.0),
                self._ev("main_line", "out", 2, 1.0),
                self._ev("main_line", "in", 3, 2.0),
                self._ev("entry_zone", "in", 4, 3.0),
            ])
            el.close()

            # файл: по одной JSON-строке на событие, все парсятся
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 4)
            for line in lines:
                rec = json.loads(line)
                self.assertIn(rec["direction"], ("in", "out"))
                self.assertIn("track_id", rec)
                self.assertIn("x_px", rec)

            # агрегаты per-counter сходятся
            c = el.counters
            self.assertEqual(c["main_line"], {"in": 2, "out": 1, "total": 3})
            self.assertEqual(c["entry_zone"], {"in": 1, "out": 0, "total": 1})
            # глобальные
            self.assertEqual(el.totals, {"in": 3, "out": 1, "total": 4})

            s = el.summary()
            self.assertIn("main_line", s)
            self.assertIn("entry_zone", s)
            self.assertIn("in=3", s)
            self.assertIn("total=4", s)

    def test_empty_path_no_file_but_aggregates(self):
        el = EventLog(OutputConfig(events_jsonl=""))
        el.log_events([self._ev("line_a", "in", 1, 0.0)])
        self.assertIsNone(el.path)
        self.assertEqual(el.totals, {"in": 1, "out": 0, "total": 1})
        self.assertIn("line_a", el.summary())
        el.close()

    def test_empty_log_summary(self):
        el = EventLog(OutputConfig(events_jsonl=""))
        self.assertEqual(el.totals, {"in": 0, "out": 0, "total": 0})
        self.assertIn("Сводка", el.summary())
        el.close()


if __name__ == "__main__":
    unittest.main()
