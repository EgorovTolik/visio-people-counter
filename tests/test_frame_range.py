"""Тесты интервала кадров (processing.frame_start/frame_end, задача 11).

* чистая функция :func:`visio_people_counter.pipeline.counting_active` —
  4 комбинации границ × включительные границы;
* пайплайн-уровень: детектор вызывается на КАЖДОМ кадре (обучение фона MOG2),
  а трекер/счётчики получают кадры ТОЛЬКО внутри интервала (фейковые
  компоненты, без видео);
* GUI: чистая функция ``counting_status_text`` (ждём кадр N / завершён) и
  статус калибратора «кадр N»;
* markdown-отчёт: строка «интервал кадров» в шапке.
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from visio_people_counter.config import Config  # noqa: E402
from visio_people_counter.pipeline import Pipeline, counting_active  # noqa: E402
from visio_people_counter.report import RunMeta, build_report  # noqa: E402
from visio_people_counter.gui import GuiPlayer, counting_status_text  # noqa: E402
from visio_people_counter.calibrate import counter_status_text  # noqa: E402


def _cfg(frame_start=None, frame_end=None) -> Config:
    return Config.from_dict({
        "video": {"type": "file", "path": "x.mp4"},
        "processing": {"frame_start": frame_start, "frame_end": frame_end},
        "counters": [{
            "id": "main_line", "type": "line",
            "a": [0.25, 0.35], "b": [0.75, 0.85],
        }],
    })


def _frame(index: int) -> SimpleNamespace:
    return SimpleNamespace(
        image=np.zeros((8, 8, 3), np.uint8),
        t_wall=float(index),
        t_video=None,
        index=index,
    )


class TestCountingActivePure(unittest.TestCase):
    """4 комбинации границ × включительность (index == start / == end → True)."""

    def test_no_bounds_always_active(self):
        for i in (-1, 0, 50, 10 ** 6):
            self.assertTrue(counting_active(i), f"i={i}")

    def test_only_start(self):
        f = lambda i: counting_active(i, frame_start=100)  # noqa: E731
        self.assertFalse(f(99))
        self.assertTrue(f(100))
        self.assertTrue(f(12345))

    def test_only_end_inclusive(self):
        f = lambda i: counting_active(i, frame_end=250)  # noqa: E731
        self.assertTrue(f(0))
        self.assertTrue(f(250))     # граница включительно
        self.assertFalse(f(251))

    def test_both_bounds_inclusive(self):
        f = lambda i: counting_active(i, frame_start=100, frame_end=250)  # noqa: E731
        self.assertFalse(f(99))
        self.assertTrue(f(100))
        self.assertTrue(f(200))
        self.assertTrue(f(250))
        self.assertFalse(f(251))

    def test_single_frame_interval(self):
        f = lambda i: counting_active(i, frame_start=7, frame_end=7)  # noqa: E731
        self.assertFalse(f(6))
        self.assertTrue(f(7))
        self.assertFalse(f(8))


class TestPipelineCountingActiveMethod(unittest.TestCase):
    """Pipeline.counting_active читает cfg.processing."""

    def test_method_matches_config(self):
        pipe = Pipeline(_cfg(frame_start=10, frame_end=20))
        for i in range(25):
            expected = (10 <= i <= 20)
            self.assertEqual(pipe.counting_active(i), expected, f"i={i}")

    def test_method_without_range(self):
        pipe = Pipeline(_cfg())
        for i in (0, 1, 999):
            self.assertTrue(pipe.counting_active(i))


class _Fakes:
    """Фейковые компоненты pipeline: детектор/трекер/счётчик считают вызовы."""

    def __init__(self) -> None:
        self.detect_calls = []
        self.tracker_calls = []
        self.counter_frames = []

    def wire(self, pipe: Pipeline) -> None:
        pipe.size_profile = None
        pipe.detector = SimpleNamespace(
            detect=lambda img, sp: (self.detect_calls.append(img is not None), [])[1])
        pipe.tracker = SimpleNamespace(
            update=lambda blobs: (self.tracker_calls.append(len(blobs)), [])[1])
        counter = SimpleNamespace(
            counter_id="main_line",
            label_text=lambda: "m",
            update=lambda objs, t_wall, t_video=0.0, frame_index=0:
                (self.counter_frames.append(frame_index), [])[1],
        )
        pipe.counters = [counter]
        pipe.event_log = SimpleNamespace(log_events=lambda evs: None)


class TestPipelineStepFrameRange(unittest.TestCase):
    """Пайплайн-уровень: вне интервала счётчик/трекер не получают кадры,
    детектор — каждый кадр (обучение фона MOG2)."""

    def _run(self, frame_start, frame_end, n=10):
        pipe = Pipeline(_cfg(frame_start=frame_start, frame_end=frame_end))
        fakes = _Fakes()
        fakes.wire(pipe)
        for i in range(n):
            pipe.step(_frame(i))
        return pipe, fakes

    def test_detector_called_on_every_frame(self):
        _, fakes = self._run(3, 6)
        self.assertEqual(len(fakes.detect_calls), 10)   # ВСЕ кадры — обучение фона

    def test_tracker_and_counters_only_inside_range(self):
        pipe, fakes = self._run(3, 6)
        self.assertEqual(fakes.tracker_calls, [0] * 4)          # только кадры 3..6
        self.assertEqual(fakes.counter_frames, [3, 4, 5, 6])    # границы включительно
        self.assertEqual(pipe.report_events, [])

    def test_only_start_counts_to_end(self):
        _, fakes = self._run(7, None)
        self.assertEqual(fakes.counter_frames, [7, 8, 9])

    def test_only_end_counts_from_beginning(self):
        _, fakes = self._run(None, 2)
        self.assertEqual(fakes.counter_frames, [0, 1, 2])

    def test_no_range_counts_everything(self):
        _, fakes = self._run(None, None)
        self.assertEqual(fakes.counter_frames, list(range(10)))


class TestGuiCountingStatusText(unittest.TestCase):
    """Строка статуса GUI: ждём кадр N до старта, пусто внутри, завершён после."""

    def test_no_range_empty(self):
        for i in (0, 50, 999):
            self.assertEqual(counting_status_text(i), "")

    def test_waiting_before_start(self):
        s = counting_status_text(42, frame_start=100, frame_end=200)
        self.assertIn("ждём кадр 100", s)
        self.assertTrue(s.startswith("подсчёт:"))

    def test_empty_inside_range_and_at_bounds(self):
        for i in (100, 150, 200):     # границы включительно — без надписи
            self.assertEqual(counting_status_text(i, frame_start=100, frame_end=200), "")

    def test_finished_after_end(self):
        s = counting_status_text(201, frame_start=100, frame_end=200)
        self.assertIn("подсчёт завершён", s)
        self.assertIn("до кадра 200", s)

    def test_only_end_finished_after(self):
        self.assertEqual(counting_status_text(5), "")
        self.assertIn("подсчёт завершён (до кадра 5)", counting_status_text(6, frame_end=5))


class TestGuiPlayerStatusText(unittest.TestCase):
    """GuiPlayer._status_text включает статус подсчёта (без окна)."""

    @staticmethod
    def _player(frame_start=None, frame_end=None) -> GuiPlayer:
        pipe = SimpleNamespace(cfg=_cfg(frame_start=frame_start, frame_end=frame_end))
        return GuiPlayer(pipe)

    def test_status_shows_waiting(self):
        p = self._player(frame_start=100, frame_end=200)
        self.assertIn("ждём кадр 100", p._status_text(30.0, frame_index=42))

    def test_status_clean_inside_range(self):
        p = self._player(frame_start=100, frame_end=200)
        self.assertNotIn("подсчёт", p._status_text(30.0, frame_index=150))

    def test_status_shows_finished(self):
        p = self._player(frame_start=100, frame_end=200)
        self.assertIn("подсчёт завершён (до кадра 200)",
                      p._status_text(30.0, frame_index=250))


class TestCalibrateFrameNumberStatus(unittest.TestCase):
    """Статус калибратора содержит номер текущего кадра «кадр N»."""

    def test_frame_number_in_status(self):
        t = counter_status_text("main_line", [], scale=1.0, frame_index=42)
        self.assertIn("кадр 42", t)

    def test_no_frame_number_when_not_given(self):
        t = counter_status_text("main_line", [])
        self.assertNotIn("кадр ", t)


class TestReportFrameRangeLine(unittest.TestCase):
    """Markdown-отчёт: строка интервала в шапке."""

    @staticmethod
    def _text(frame_start=None, frame_end=None) -> str:
        cfg = _cfg(frame_start=frame_start, frame_end=frame_end)
        meta = RunMeta(source="x.mp4", frames_processed=10, reason="EOF")
        return build_report(cfg, [], meta)

    def test_both_bounds(self):
        self.assertIn("интервал кадров: 10–250 (подсчёт)", self._text(10, 250))

    def test_only_start(self):
        self.assertIn("интервал кадров: 10–конец (подсчёт)", self._text(10, None))

    def test_only_end(self):
        self.assertIn("интервал кадров: начало–250 (подсчёт)", self._text(None, 250))

    def test_no_range_whole_video(self):
        self.assertIn("интервал кадров: весь", self._text())


if __name__ == "__main__":
    unittest.main()
