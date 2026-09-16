"""Тесты публичного API GuiPlayer (задача 17) + выбор бэкенда count --gui.

Без окна: ``GuiPlayer.tick()`` / ``handle_key`` / ``finalize`` на реальном
Pipeline поверх синтетического mp4; cv2-окно/imshow НЕ поднимается. Плюс CLI:
парсер ``count --backend {qt,cv2}`` и чистая функция ``choose_gui_backend``.
"""

import io
import inspect
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from visio_people_counter.config import Config  # noqa: E402
from visio_people_counter.gui import GuiPlayer  # noqa: E402
from visio_people_counter.pipeline import Pipeline  # noqa: E402

W, H, FPS, N_FRAMES = 320, 180, 30.0, 30


def _make_video(path: Path) -> None:
    """Синтетический mp4: статичная текстура + движущийся белый прямоугольник."""
    rng = np.random.default_rng(7)
    bg = rng.integers(60, 110, size=(H, W, 3), dtype=np.uint8)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    assert vw.isOpened(), "не удалось открыть VideoWriter для синтетического mp4"
    for f in range(N_FRAMES):
        frame = bg.copy()
        x = int(round(40 + 240 * (f / float(N_FRAMES - 1))))
        cv2.rectangle(frame, (x, 60), (x + 40, 150), (255, 255, 255), -1)
        vw.write(frame)
    vw.release()


class _PlayerBase(unittest.TestCase):
    """Общий синтетический mp4; конфиг с явным output.report_path в tmp-папке."""

    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory(prefix="vpc_test_count_backend_")
        cls.dir = Path(cls._td.name)
        cls.video = cls.dir / "synth.mp4"
        _make_video(cls.video)
        cls.n_cfg = 0

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def _make_config(self, report_name: str) -> Config:
        self.n_cfg += 1
        return Config.from_dict({
            "video": {"type": "file", "path": str(self.video)},
            "output": {"report_path": str(self.dir / f"rep_{self._testMethodName}_{self.n_cfg}.md")
                       if report_name is None else str(self.dir / report_name)},
        })


class TestGuiPlayerTick(_PlayerBase):
    """tick(): кадры идут, пауза держит view, q → следующая итерация завершается."""

    def test_tick_advances_frames_and_scales_view(self):
        cfg = self._make_config(None)
        player = GuiPlayer(Pipeline(cfg))
        with redirect_stdout(io.StringIO()):   # build() печатает лог источника
            v1 = player.tick()
            v2 = player.tick()
        self.assertIsNotNone(v1)
        self.assertIsNotNone(v2)
        # кадр 320x180 → авто-масштаб (задача 14) поднял scale до 2.0 → view 640x360
        self.assertEqual(v1.shape[:2], (H * 2, W * 2))
        # два тика → прочитаны кадры 0 и 1: frame-индекс растёт
        self.assertEqual(player.frame_index, 1)
        self.assertGreater(player.proc_dt, 0.0)

    def test_pause_returns_last_view_twice(self):
        cfg = self._make_config(None)
        player = GuiPlayer(Pipeline(cfg))
        with redirect_stdout(io.StringIO()):
            v1 = player.tick()
            idx = player.frame_index
            player.handle_key(32)               # пробел — пауза (публичная обёртка)
            self.assertTrue(player._paused)
            v2 = player.tick()
            v3 = player.tick()
        self.assertIsNotNone(v2)
        self.assertIsNotNone(v3)
        np.testing.assert_array_equal(v2, v3)   # тот же (последний) view дважды
        np.testing.assert_array_equal(v1, v2)
        self.assertEqual(player.frame_index, idx)   # новые кадры не читаются

    def test_q_stops_next_tick(self):
        cfg = self._make_config(None)
        player = GuiPlayer(Pipeline(cfg))
        with redirect_stdout(io.StringIO()):
            self.assertIsNotNone(player.tick())
            player.handle_key(ord("q"))         # q → _stop
            self.assertIsNone(player.tick())    # следующий шаг цикла — None (завершение)


class TestGuiPlayerFinalize(_PlayerBase):
    """finalize(): сводка в stdout + отчёт один раз; идемпотентна."""

    def test_finalize_idempotent_report_once(self):
        cfg = self._make_config("rep_finalize.md")
        report_path = Path(cfg.output.report_path)
        pipe = Pipeline(cfg)
        with redirect_stdout(io.StringIO()):
            pipe.build()
        player = GuiPlayer(pipe)
        with redirect_stdout(io.StringIO()) as out:
            player.finalize("тест finalize")
            first = out.getvalue()
        self.assertIn("ФИНАЛЬНАЯ СВОДКА", first)      # сводка в stdout
        self.assertIn("=== ОТЧЁТ ===", first)
        self.assertTrue(report_path.is_file())
        content1 = report_path.read_text(encoding="utf-8")
        self.assertIn("тест finalize", content1)

        with redirect_stdout(io.StringIO()) as out2:
            player.finalize("тест finalize")          # повторный вызов — игнорится
        self.assertNotIn("ФИНАЛЬНАЯ СВОДКА", out2.getvalue())
        self.assertNotIn("=== ОТЧЁТ ===", out2.getvalue())
        self.assertEqual(report_path.read_text(encoding="utf-8"), content1)   # один раз

    def test_cv2_run_rewritten_through_tick(self):
        """run() (cv2) переписан ВНУТРЕННЕ через tick() (source-assert, окно не поднимается)."""
        src = inspect.getsource(GuiPlayer.run)
        self.assertIn("self.tick()", src)
        self.assertIn("self.finalize(", src)
        # авто-масштаб (задача 14) остаётся в run() — как раньше
        self.assertIn("auto_ui_scale(src.width, src.height, self.scale)", src)


class TestCliCountBackend(unittest.TestCase):
    """CLI: count --backend {qt,cv2} парсится; choose_gui_backend — чистая функция."""

    @staticmethod
    def _parse(extra: list[str]):
        from visio_people_counter.__main__ import build_parser
        return build_parser().parse_args(["count"] + extra)

    def test_count_backend_parses(self):
        ns = self._parse(["--gui", "--backend", "qt"])
        self.assertTrue(ns.gui)
        self.assertEqual(ns.backend, "qt")
        self.assertEqual(self._parse([]).backend, None)   # дефолт — не задан
        self.assertEqual(self._parse(["--backend", "cv2"]).backend, "cv2")

    def test_count_backend_bad_choice_is_argparse_error(self):
        with self.assertRaises(SystemExit):
            self._parse(["--gui", "--backend", "gtk"])

    def test_choose_gui_backend_matrix(self):
        from visio_people_counter.__main__ import choose_gui_backend
        # без явного --backend: qt при PySide6, иначе cv2
        self.assertEqual(choose_gui_backend(None, True), "qt")
        self.assertEqual(choose_gui_backend(None, False), "cv2")
        # явный qt: нужен PySide6 (иначе None → CLI rc=1)
        self.assertEqual(choose_gui_backend("qt", True), "qt")
        self.assertIsNone(choose_gui_backend("qt", False))
        # явный cv2 — работает в любом случае
        self.assertEqual(choose_gui_backend("cv2", True), "cv2")
        self.assertEqual(choose_gui_backend("cv2", False), "cv2")


if __name__ == "__main__":
    unittest.main()
