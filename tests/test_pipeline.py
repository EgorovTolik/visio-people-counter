"""Тесты pipeline.py на синтетическом mp4 (cv2.VideoWriter, 'mp4v', без внешних файлов).

Видео: 640x360 @ 30 fps, 200 кадров. Статичный текстурированный фон + «человек»
(сплошное прямоугольник 40x90) ~70 кадров «ходит на месте» (чтобы MOG2 не впечатал
его в фон), затем идёт по диагонали через линию a=[0.25,0.35] → b=[0.75,0.85].

Геометрия: A_px=(160,126), B_px=(480,306). Старт P0=(210,300) — сторона d>0
(d(P)=cross(B−A, P−A)), финиш P1=(390,70) — сторона d<0; по конвенции задачи 14
пересечение со стороны d>0 = направление "in". Пересечение происходит на
кадре ~127 (после warmup-удержания).
"""

import io
import json
import sys
import tempfile
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

W, H, FPS, N_FRAMES = 640, 360, 30.0, 200
HOLD_FRAMES = 70          # «ходит на месте» до начала движения (после прогрева MOG2)
P0 = (210.0, 300.0)       # старт (сторона d>0 линии)
P1 = (390.0, 70.0)        # финиш (сторона d<0) — пересечение ≈ кадр 127
PW, PH = 40, 90           # «человек»


def _make_video(path: Path) -> None:
    """Синтетический mp4: статичный фон + движущийся белый прямоугольник."""
    rng = np.random.default_rng(42)
    bg = rng.integers(40, 90, size=(H, W, 3), dtype=np.uint8)  # постоянная текстура
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    assert vw.isOpened(), "не удалось открыть VideoWriter для синтетического mp4"
    for f in range(N_FRAMES):
        frame = bg.copy()
        if f < HOLD_FRAMES:
            # плавное «блуждание» ±12 px: пиксели постоянно меняются → MOG2 не впечатает в фон
            pos = (P0[0] + 12.0 * np.sin(f * 0.9), P0[1] + 6.0 * np.cos(f * 0.7))
        else:
            p = (f - HOLD_FRAMES) / (N_FRAMES - 1 - HOLD_FRAMES)
            pos = (P0[0] + (P1[0] - P0[0]) * p, P0[1] + (P1[1] - P0[1]) * p)
        x0 = int(round(pos[0] - PW / 2))
        y0 = int(round(pos[1] - PH / 2))
        cv2.rectangle(frame, (x0, y0), (x0 + PW, y0 + PH), (255, 255, 255), -1)
        vw.write(frame)
    vw.release()


def _make_config(video_path: Path, events_path: Path, effective_fps: float = 0.0) -> Path:
    """Тестовый конфиг (минимальный; недостающие блоки берут дефолты Config)."""
    cfg = {
        "video": {"type": "file", "path": str(video_path)},
        "processing": {"effective_fps": effective_fps, "max_width": 0},
        # history=200 → warmup = history/2 ≈ 100 кадров (видео 200 кадров)
        "motion": {"history": 200, "var_threshold": 32},
        "counters": [{
            "id": "main_line", "type": "line",
            "a": [0.25, 0.35], "b": [0.75, 0.85],
            "count_mode": "both", "cooldown_s": 2.0, "min_global_gap_s": 0.3,
        }],
        "output": {"events_jsonl": str(events_path),
                   "summary_interval_s": 1.0, "final_summary": True},
    }
    p = video_path.parent / "config.yaml"
    Config.save(Config.from_dict(cfg, str(p)), p)
    return p


class _PipelineTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory(prefix="vpc_test_pipeline_")
        cls.dir = Path(cls._td.name)
        cls.video_path = cls.dir / "synth.mp4"
        _make_video(cls.video_path)

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()


class TestPipelineRun(_PipelineTestBase, unittest.TestCase):
    """Базовый прогон: нативный fps, пересечение линии → событие, финальная сводка, EOF."""

    def test_crossing_event_and_clean_eof(self):
        cfg_path = _make_config(self.video_path, self.dir / "events.jsonl")
        pipe = Pipeline(cfg_path)
        with redirect_stdout(io.StringIO()) as out:
            rc = pipe.run()
        text = out.getvalue()

        # EOF корректный: run() вернул 0, все кадры прочитаны/обработаны
        self.assertEqual(rc, 0)
        self.assertEqual(pipe.frames_processed, N_FRAMES)
        self.assertTrue(pipe.warmup_done)

        # в EventLog ≥1 событие ожидаемого направления "in" (данные из JSONL — единый источник истины)
        lines = [json.loads(l) for l in
                 (self.dir / "events.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        in_events = [e for e in lines if e["counter_id"] == "main_line" and e["direction"] == "in"]
        self.assertGreaterEqual(len(in_events), 1, f"нет 'in'-событий: {lines}")
        # один человек прошёл линию один раз: разумный верхний предел (антидубль работает)
        self.assertLessEqual(len(lines), 3, f"слишком много событий: {lines}")

        # финальная сводка напечатана (строка EventLog.summary + блок ФИНАЛЬНАЯ СВОДКА)
        self.assertIn("ФИНАЛЬНАЯ СВОДКА", text)
        self.assertIn("Сводка", text)
        self.assertIn("main_line", text)

    def test_bench_prints_percentiles(self):
        cfg_path = _make_config(self.video_path, self.dir / "events_bench.jsonl")
        pipe = Pipeline(cfg_path, bench=True)
        with redirect_stdout(io.StringIO()) as out:
            rc = pipe.run()
        text = out.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("БЕНЧМАРК", text)
        for stage in ("detect", "track", "count", "frame (итого)"):
            self.assertIn(stage, text)
        self.assertIn("p50=", text)
        self.assertIn("p95=", text)


class TestPipelineEffectiveFps(_PipelineTestBase, unittest.TestCase):
    """effective_fps=5: кадры пропускаются, длительность не взрывается, fps обработки > 5."""

    def test_effective_fps_5(self):
        cfg_path = _make_config(self.video_path, self.dir / "events_fps5.jsonl", effective_fps=5.0)
        pipe = Pipeline(cfg_path)
        t0 = time.monotonic()
        with redirect_stdout(io.StringIO()):
            rc = pipe.run()
        elapsed = time.monotonic() - t0

        self.assertEqual(rc, 0)
        # пропуск: из 200 кадров обрабатывается ~каждый 6-й (round(30/5))
        self.assertGreater(pipe.frames_processed, 0)
        self.assertLess(pipe.frames_processed, N_FRAMES)
        # длительность не взрывается; fps обработки (frames/wall) > effective_fps
        self.assertLess(elapsed, 60.0, f"прогон занял {elapsed:.1f} c — слишком долго")
        self.assertGreater(pipe.avg_fps, 5.0,
                           f"fps обработки {pipe.avg_fps:.2f} <= effective_fps=5")


if __name__ == "__main__":
    unittest.main()
