"""Тесты thread-pipeline (задача 21): producer-consumer для headless count.

Синтетические mp4 (cv2.VideoWriter, 'mp4v', без внешних файлов), паттерн из
test_pipeline.py: статичный текстурированный фон + движущийся белый прямоугольник.

Покрыто:
1. базовый прогон use_threads=True — все кадры обработаны, события == legacy;
2. EOF/sentinel — consumer получает None и завершается без зависания (unit на
   _reader_loop с FakeSource + полный прогон до EOF);
3. маленький размер очереди (1) — reader блокируется на put, всё корректно;
4. graceful stop посреди обработки — process завершается, отчёт пишется.
"""

import io
import json
import queue
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from visio_people_counter.config import Config  # noqa: E402
from visio_people_counter.pipeline import Pipeline  # noqa: E402

W, H, FPS = 640, 360, 30.0
P0 = (210.0, 300.0)
PW, PH = 40, 90


def _make_video(path: Path, n_frames: int, hold_frames: int, p1=(390.0, 70.0)) -> None:
    """Синтетический mp4: статичный фон + движущийся белый прямоугольник."""
    rng = np.random.default_rng(42)
    bg = rng.integers(40, 90, size=(H, W, 3), dtype=np.uint8)  # постоянная текстура
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    assert vw.isOpened(), "не удалось открыть VideoWriter для синтетического mp4"
    for f in range(n_frames):
        frame = bg.copy()
        if f < hold_frames:
            pos = (P0[0] + 12.0 * np.sin(f * 0.9), P0[1] + 6.0 * np.cos(f * 0.7))
        else:
            p = (f - hold_frames) / max(1, n_frames - 1 - hold_frames)
            pos = (P0[0] + (p1[0] - P0[0]) * p, P0[1] + (p1[1] - P0[1]) * p)
        x0 = int(round(pos[0] - PW / 2))
        y0 = int(round(pos[1] - PH / 2))
        cv2.rectangle(frame, (x0, y0), (x0 + PW, y0 + PH), (255, 255, 255), -1)
        vw.write(frame)
    vw.release()


def _make_config(video_path: Path, events_path: Path, report_path: Path,
                 effective_fps: float = 0.0) -> Path:
    """Тестовый конфиг (минимальный; недостающие блоки берут дефолты Config).

    YAML-файл именуется по имени events-файла — чтобы несколько прогонов одного
    видео (legacy vs threads) не затирали чужой конфиг.
    """
    cfg = {
        "video": {"type": "file", "path": str(video_path)},
        "processing": {"effective_fps": effective_fps, "max_width": 0},
        # history=200 → warmup = history/2 ≈ 100 кадров (в коротких видео события не
        # наступают — это нормально: сравниваем legacy vs threads на одном и том же)
        "motion": {"history": 200, "var_threshold": 32},
        "counters": [{
            "id": "main_line", "type": "line",
            "a": [0.25, 0.35], "b": [0.75, 0.85],
            "count_mode": "both", "cooldown_s": 2.0, "min_global_gap_s": 0.3,
        }],
        "output": {"events_jsonl": str(events_path),
                   "report_path": str(report_path),
                   "summary_interval_s": 1.0, "final_summary": True},
    }
    p = video_path.parent / f"config_{events_path.stem}.yaml"
    Config.save(Config.from_dict(cfg, str(p)), p)
    return p


def _events_from_jsonl(path: Path) -> list[dict]:
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [(e["counter_id"], e["direction"]) for e in lines]


class _VideoMixin(unittest.TestCase):
    """Общие фикстуры: одно короткое (30 кадров) и одно длинное (200) видео."""

    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory(prefix="vpc_test_thread_")
        cls.dir = Path(cls._td.name)
        cls.short_video = cls.dir / "short.mp4"
        _make_video(cls.short_video, n_frames=30, hold_frames=15)
        cls.long_video = cls.dir / "long.mp4"
        # 200 кадров: пересечение линии после warmup (геометрия test_pipeline.py)
        _make_video(cls.long_video, n_frames=200, hold_frames=70)

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()


class TestThreadedBasic(_VideoMixin):
    """1. Базовый: use_threads=True — все кадры, события совпадают с legacy."""

    def test_all_frames_processed_and_events_match_legacy(self):
        v = self.short_video
        cfg_l = _make_config(v, self.dir / "ev_leg.jsonl", self.dir / "rep_leg.md")
        cfg_t = _make_config(v, self.dir / "ev_thr.jsonl", self.dir / "rep_thr.md")

        legacy = Pipeline(cfg_l)
        with redirect_stdout(io.StringIO()):
            rc_l = legacy.run()

        threaded = Pipeline(cfg_t, use_threads=True, queue_size=10)
        t0 = time.monotonic()
        with redirect_stdout(io.StringIO()) as out:
            rc_t = threaded.run()
        elapsed = time.monotonic() - t0

        self.assertEqual(rc_l, 0)
        self.assertEqual(rc_t, 0)
        # все кадры обработаны в обоих режимах
        self.assertEqual(legacy.frames_processed, 30)
        self.assertEqual(threaded.frames_processed, 30)
        # события совпадают с legacy mode (на коротком видео их нет — но режимы равны)
        self.assertEqual(_events_from_jsonl(self.dir / "ev_thr.jsonl"),
                         _events_from_jsonl(self.dir / "ev_leg.jsonl"))
        # thread-pipeline реально включался и не завис
        self.assertIn("thread-pipeline", out.getvalue())
        self.assertLess(elapsed, 30.0, f"прогон занял {elapsed:.1f} c — зависание?")

    def test_events_match_legacy_on_long_video(self):
        """Длинное видео (пересечение линии): события legacy == threaded."""
        v = self.long_video
        cfg_l = _make_config(v, self.dir / "evL.jsonl", self.dir / "repL.md")
        cfg_t = _make_config(v, self.dir / "evT.jsonl", self.dir / "repT.md")

        legacy = Pipeline(cfg_l)
        with redirect_stdout(io.StringIO()):
            rc_l = legacy.run()
        threaded = Pipeline(cfg_t, use_threads=True, queue_size=10)
        with redirect_stdout(io.StringIO()):
            rc_t = threaded.run()

        self.assertEqual(rc_l, 0)
        self.assertEqual(rc_t, 0)
        ev_l = _events_from_jsonl(self.dir / "evL.jsonl")
        ev_t = _events_from_jsonl(self.dir / "evT.jsonl")
        self.assertGreaterEqual(len(ev_l), 1, "legacy не дал событий — проверка геометрии")
        self.assertEqual(ev_t, ev_l, "threaded-режим дал другие события, чем legacy")


class TestReaderLoopUnit(_VideoMixin):
    """2. EOF/sentinel: unit-тест _reader_loop + полный прогон без зависания."""

    def test_reader_loop_puts_frames_then_sentinel(self):
        cfg = Config.from_dict({"video": {"type": "file", "path": str(self.short_video)}},
                               str(self.dir / "cfg_r.yaml"))
        pipe = Pipeline(cfg)  # без build() — source подставим вручную

        class FakeSource:
            """Имитация источника: N кадров, затем None (EOF)."""
            def __init__(self, n):
                self.left = n
            def read(self):
                if self.left > 0:
                    self.left -= 1
                    return np.zeros((4, 4, 3), dtype=np.uint8)
                return None

        q = queue.Queue(maxsize=2)
        stop = threading.Event()
        pipe.source = FakeSource(5)
        # reader как в production — в отдельном потоке; main снимает кадры (consumer).
        # maxsize=2: reader периодически блокируется на put — проверяем, что это
        # не мешает доставке всех кадров + sentinel.
        t = threading.Thread(target=pipe._reader_loop, args=(q, stop), daemon=True)
        t.start()
        got = [q.get(timeout=10.0) for _ in range(6)]
        t.join(timeout=10.0)
        self.assertFalse(t.is_alive(), "reader не завершился")
        self.assertEqual(len([f for f in got if f is not None]), 5)
        self.assertIsNone(got[-1], "последним должен быть sentinel None")
        self.assertFalse(stop.is_set(), "стоп не нужен: очередь свободна, consumer жив")

    def test_full_run_terminates_on_sentinel_without_hang(self):
        """После последнего кадра consumer получает sentinel и завершается."""
        cfg = _make_config(self.short_video, self.dir / "ev_eof.jsonl",
                           self.dir / "rep_eof.md")
        pipe = Pipeline(cfg, use_threads=True, queue_size=5)
        t0 = time.monotonic()
        with redirect_stdout(io.StringIO()):
            rc = pipe.run()
        elapsed = time.monotonic() - t0
        self.assertEqual(rc, 0)
        self.assertEqual(pipe.frames_processed, 30)  # все кадры дошли до consumer'а
        self.assertLess(elapsed, 20.0, f"завершение после EOF заняло {elapsed:.1f} c")


class TestTinyQueue(_VideoMixin):
    """3. Маленький размер очереди (1): reader блокируется на put — всё корректно."""

    def test_queue_size_1(self):
        cfg = _make_config(self.short_video, self.dir / "ev_q1.jsonl",
                           self.dir / "rep_q1.md")
        pipe = Pipeline(cfg, use_threads=True, queue_size=1)
        with redirect_stdout(io.StringIO()) as out:
            rc = pipe.run()
        self.assertEqual(rc, 0)
        self.assertEqual(pipe.frames_processed, 30)
        self.assertIn("очередь=1", out.getvalue())


class TestGracefulStop(_VideoMixin):
    """4. Graceful stop посреди обработки: завершение + отчёт пишется."""

    def test_stop_mid_run_writes_report(self):
        cfg = _make_config(self.short_video, self.dir / "ev_stop.jsonl",
                           self.dir / "rep_stop.md")
        pipe = Pipeline(cfg, use_threads=True, queue_size=10)

        target = 20

        def _stopper():
            deadline = time.monotonic() + 60.0
            while not pipe._stop and time.monotonic() < deadline:
                if pipe.frames_processed >= target:
                    pipe._stop = True      # эмуляция SIGINT (handler так же ставит флаг)
                    if pipe._reader_stop is not None:
                        pipe._reader_stop.set()
                    return
                time.sleep(0.02)

        t = threading.Thread(target=_stopper, name="vpc-test-stopper", daemon=True)
        t.start()
        with redirect_stdout(io.StringIO()) as out:
            rc = pipe.run()
        t.join(timeout=5.0)

        self.assertEqual(rc, 0)
        # остановились ПОСЕРЕДИНЕ: часть кадров, не все
        self.assertGreaterEqual(pipe.frames_processed, target)
        self.assertLess(pipe.frames_processed, 30,
                        f"остановка сработала слишком поздно: {pipe.frames_processed} кадров")
        # финальная сводка напечатана с причиной SIGINT и отчёт создан
        text = out.getvalue()
        self.assertIn("ФИНАЛЬНАЯ СВОДКА", text)
        self.assertIn("SIGINT", text)
        self.assertTrue((self.dir / "rep_stop.md").is_file(), "отчёт не создан при stop")


if __name__ == "__main__":
    unittest.main()
