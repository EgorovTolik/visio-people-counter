"""Тесты переноса текста счётчиков/статуса из кадра в Qt-статусбар (задача 20):

* :func:`counters_status_line` — чистая строка счётчиков для статусбара;
* :meth:`GuiOverlay.draw(..., with_text)` — False снимает put_text in/out и
  status_text с кадра, но оставляет графику линии/зоны (counter.draw);
* :meth:`GuiPlayer.tick` передаёт ``with_text=self.overlay_with_text``;
* Qt count-окно: ``overlay_with_text=False`` + строка счётчиков в статусбаре.

Offscreen. Паттерн как в tests/test_count_qt.py / test_gui_qt_text.py: один
QApplication на модуль, синтетический mp4 320x180 @ 30 fps, exec() не вызывается
— окна ведутся ручными tick_once(). Чистые функции (counters_status_line,
GuiOverlay.draw) работают и без PySide6, но весь модуль guarded'ится наличием
PySide6 ради Qt-окна — как у соседей по tests/.
"""

import importlib.util
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if importlib.util.find_spec("PySide6") is None:
    # PySide6 не установлен — пропускаем ВЕСЬ модуль (Qt-бэкенд опциональный)
    class _SkipModule(unittest.TestCase):
        @unittest.skip("PySide6 не установлен в .venv (.venv/bin/pip install PySide6)")
        def test_pyside_available(self):
            pass


else:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    import cv2  # noqa: E402
    import numpy as np  # noqa: E402
    from PySide6.QtWidgets import QApplication  # noqa: E402

    import visio_people_counter.gui_qt as gui_qt  # noqa: E402
    from visio_people_counter.config import (  # noqa: E402
        Config, DebugConfig, LineCounterConfig, ZoneCounterConfig)
    from visio_people_counter.gui import (  # noqa: E402
        GuiOverlay, GuiPlayer, counters_status_line)
    from visio_people_counter.line_counter import (  # noqa: E402
        LineCounter, ZoneCounter)
    from visio_people_counter.pipeline import Pipeline  # noqa: E402

    W, H, FPS, N_FRAMES = 320, 180, 30.0, 40   # короткое видео: EOF быстро

    _APP = None
    _TD = None
    _VIDEO: Path | None = None
    _WINDOWS: list = []


    def _make_video(path: Path) -> None:
        """Синтетический mp4: статичная текстура + движущийся белый прямоугольник."""
        rng = np.random.default_rng(17)
        bg = rng.integers(60, 110, size=(H, W, 3), dtype=np.uint8)
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        assert vw.isOpened(), "не удалось открыть VideoWriter для синтетического mp4"
        for f in range(N_FRAMES):
            frame = bg.copy()
            x = int(round(40 + 240 * (f / float(N_FRAMES - 1))))
            cv2.rectangle(frame, (x, 60), (x + 40, 150), (255, 255, 255), -1)
            vw.write(frame)
        vw.release()


    def _make_config(path: Path, report_name: str) -> Config:
        """Минимальный конфиг с line + zone счётчиками и явным report_path."""
        d = {
            "video": {"type": "file", "path": str(path)},
            "processing": {"max_width": 0},
            "motion": {"history": 50, "var_threshold": 32},
            "counters": [
                {"id": "main_line", "type": "line",
                 "a": [0.2, 0.8], "b": [0.8, 0.2],
                 "count_mode": "both", "cooldown_s": 2.0, "min_global_gap_s": 0.3},
                {"id": "main_zone", "type": "zone",
                 "polygon": [(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)],
                 "count_mode": "total", "cooldown_s": 2.0, "min_global_gap_s": 0.3},
            ],
            "output": {"report_path": str(Path(_TD.name) / report_name)},
        }
        p = path.parent / f"cfg_{report_name}.yaml"
        return Config.from_dict(d, str(p))


    def setUpModule():
        global _APP, _TD, _VIDEO
        _APP = QApplication.instance() or QApplication([])
        _TD = tempfile.TemporaryDirectory(prefix="vpc_test_count_text_")
        _VIDEO = Path(_TD.name) / "synth.mp4"
        _make_video(_VIDEO)


    def tearDownModule():
        global _TD
        for w in _WINDOWS:
            try:
                if not w.is_closed:
                    with redirect_stdout(io.StringIO()):
                        w.close()
                w.deleteLater()
            except RuntimeError:
                pass
        _APP.processEvents()
        if _TD is not None:
            _TD.cleanup()
            _TD = None


    # ------------------------------------------------------------------ чистые

    class TestCountersStatusLinePure(unittest.TestCase):
        """counters_status_line — задачи 20 (чистая функция, без окна)."""

        def _counters(self) -> list:
            line = LineCounter(LineCounterConfig(
                id="main_line", a=(0.2, 0.8), b=(0.8, 0.2),
                count_mode="both", cooldown_s=2.0, min_global_gap_s=0.3), W, H)
            zone = ZoneCounter(ZoneCounterConfig(
                id="main_zone", polygon=[(0.4, 0.4), (0.6, 0.4),
                                         (0.6, 0.6), (0.4, 0.6)],
                count_mode="total", cooldown_s=2.0, min_global_gap_s=0.3), W, H)
            return [line, zone]

        def test_line_contains_ids_and_inout(self):
            s = counters_status_line(self._counters())
            # формат «счётчики: <id>: in=N out=M | ...» — видны id и in/out
            self.assertIn("main_line", s)
            self.assertIn("in=", s)
            self.assertIn("out=", s)
            self.assertIn("main_zone", s)
            self.assertTrue(s.startswith("счётчики:"))
            # разделитель между счётчиками — « | »
            self.assertIn("|", s)

        def test_empty_when_no_counters(self):
            self.assertEqual(counters_status_line([]), "")
            self.assertEqual(counters_status_line(None), "")

        def test_values_are_live_at_call_time(self):
            """Чистая функция берёт актуальные in/out на момент вызова."""
            line = LineCounter(LineCounterConfig(
                id="live", a=(0.2, 0.8), b=(0.8, 0.2),
                count_mode="both", cooldown_s=2.0, min_global_gap_s=0.3), W, H)
            self.assertIn("in=0 out=0", counters_status_line([line]))
            line._in_count = 1   # имитируем одно пересечение «внутрь» (публичный счётчик)
            s = counters_status_line([line])
            self.assertIn("in=1", s)

        def test_pipeline_counters(self):
            """Через Pipeline: pipe.counters → строка с обоими id (тест-паттерн)."""
            cfg = _make_config(_VIDEO, "pipeline_report.md")
            with redirect_stdout(io.StringIO()):
                pipe = Pipeline(cfg)
                pipe.build()
            s = counters_status_line(pipe.counters)
            self.assertIn("main_line", s)
            self.assertIn("main_zone", s)
            pipe.close()


    # -------------------------------------------------- GuiOverlay.draw(with_text)

    class TestGuiOverlayDrawText(unittest.TestCase):
        """GuiOverlay.draw: with_text=False снимает текст, но оставляет графику."""

        def _overlay(self) -> GuiOverlay:
            return GuiOverlay(DebugConfig(show_counters=True, show_bboxes=False))

        def _counters(self) -> list:
            line = LineCounter(LineCounterConfig(
                id="main_line", a=(0.2, 0.8), b=(0.8, 0.2),
                count_mode="both", cooldown_s=2.0, min_global_gap_s=0.3), W, H)
            zone = ZoneCounter(ZoneCounterConfig(
                id="main_zone", polygon=[(0.4, 0.4), (0.6, 0.4),
                                         (0.6, 0.6), (0.4, 0.6)],
                count_mode="total", cooldown_s=2.0, min_global_gap_s=0.3), W, H)
            return [line, zone]

        @staticmethod
        def _white_count(img: np.ndarray, rows, cols) -> int:
            """Чисто-БЕЛЫХ пикселей (все каналы яркие) — только PIL-текст."""
            sub = img[rows[0]:rows[1], cols[0]:cols[1]]
            if sub.size == 0:
                return 0
            return int((sub.astype(int).min(axis=2) > 200).sum())

        @staticmethod
        def _cyan_count(img: np.ndarray, rows, cols) -> int:
            """СИНИХ пикселей (status_text — cyan (0,255,255)) в области."""
            sub = img[rows[0]:rows[1], cols[0]:cols[1]]
            if sub.size == 0:
                return 0
            b, g, r = sub[:, :, 0].astype(int), sub[:, :, 1].astype(int), sub[:, :, 2].astype(int)
            return int(((g > 200) & (b > 200) & (r < 100)).sum())

        @staticmethod
        def _green_count(img: np.ndarray, rows, cols) -> int:
            """ЗЕЛЁНЫХ пикселей — гракция линии/зоны (counter.draw)."""
            sub = img[rows[0]:rows[1], cols[0]:cols[1]]
            if sub.size == 0:
                return 0
            b, g, r = sub[:, :, 0].astype(int), sub[:, :, 1].astype(int), sub[:, :, 2].astype(int)
            return int(((g > 200) & (r < 60) & (b < 60)).sum())

        def test_with_text_true_has_counters_and_status(self):
            ov = self._overlay()
            frame = np.full((H, W, 3), 128, dtype=np.uint8)
            view = ov.draw(frame.copy(), counters=self._counters(),
                           status_text="speed=1.00x proc=30fps scale=1x")
            # текст счётчиков в левом верхнем углу (белый PIL-текст)
            self.assertGreater(self._white_count(view, (0, 120), (0, 140)), 0)
            # status_text справа сверху (синий — cyan)
            self.assertGreater(self._cyan_count(view, (0, 24), (W - 160, W)), 0)

        def test_with_text_false_no_text_but_graphics(self):
            ov = self._overlay()
            counters = self._counters()
            img_true = ov.draw(np.full((H, W, 3), 128, np.uint8), counters=counters,
                               status_text="speed=1.00x", with_text=True)
            img_false = ov.draw(np.full((H, W, 3), 128, np.uint8), counters=counters,
                                status_text="speed=1.00x", with_text=False)
            # с текстом — есть белый текст in/out в углу; без — его нет
            self.assertGreater(self._white_count(img_true, (0, 120), (0, 140)), 0)
            self.assertEqual(self._white_count(img_false, (0, 120), (0, 140)), 0)
            # с текстом — cyan status_text справа; без — нет
            self.assertGreater(self._cyan_count(img_true, (0, 24), (W - 160, W)), 0)
            self.assertEqual(self._cyan_count(img_false, (0, 24), (W - 160, W)), 0)
            # гракция линии/зоны (зелёные линии counter.draw) — в обоих случаях
            self.assertGreater(self._green_count(img_true, (0, H), (0, W)), 0)
            self.assertGreater(self._green_count(img_false, (0, H), (0, W)), 0)

        def test_draw_returns_frame_and_in_place(self):
            ov = self._overlay()
            frame = np.full((H, W, 3), 128, np.uint8)
            out = ov.draw(frame, counters=self._counters(), with_text=True)
            self.assertIs(out, frame)   # in-place


    # -------------------------------------------------- GuiPlayer.tick(with_text)

    class TestGuiPlayerTickText(unittest.TestCase):
        """GuiPlayer.tick передаёт overlay_with_text в overlay.draw."""

        def _bright_frac(self, view: np.ndarray) -> float:
            if view is None or view.size == 0:
                return 0.0
            return float((view.astype(int).max(axis=2) > 200).sum() /
                         (view.shape[0] * view.shape[1]))

        def test_tick_false_has_less_text_than_true(self):
            cfg = _make_config(_VIDEO, "tick_report.md")
            player_off = GuiPlayer(Pipeline(cfg), speed=1.0)
            player_off.overlay_with_text = False
            with redirect_stdout(io.StringIO()):
                view_off = player_off.tick()

            cfg2 = _make_config(_VIDEO, "tick_report2.md")
            player_on = GuiPlayer(Pipeline(cfg2), speed=1.0)   # True по умолчанию
            with redirect_stdout(io.StringIO()):
                view_on = player_on.tick()

            self.assertFalse(player_off.overlay_with_text)
            self.assertTrue(player_on.overlay_with_text)
            self.assertIsNotNone(view_off)
            self.assertIsNotNone(view_on)
            # кадр с текстом ярче (белый текст in/out + cyan статуса)
            self.assertGreater(self._bright_frac(view_on), self._bright_frac(view_off))

        def test_tick_true_default_renders_text(self):
            cfg = _make_config(_VIDEO, "tick3.md")
            player = GuiPlayer(Pipeline(cfg), speed=1.0)
            with redirect_stdout(io.StringIO()):
                view = player.tick()
            self.assertGreater(self._bright_frac(view), 0.005)   # текст на кадре


    # -------------------------------------------------- Qt count-окно (задача 20)

    class _CountQtTextBase(unittest.TestCase):
        """Синтетическое видео + хелпер создания плеера/окна."""

        def setUp(self):
            self.win = None
            self.player = None
            self.n_cfg = 0

        def tearDown(self):
            if self.win is not None:
                try:
                    with redirect_stdout(io.StringIO()):
                        self.win.close()
                except RuntimeError:
                    pass
                _WINDOWS.append(self.win)
                self.win = None
            if self.player is not None:
                with redirect_stdout(io.StringIO()):
                    self.player.pipeline.close()
                self.player = None

        def _open_window(self, report_name: str):
            cfg = _make_config(_VIDEO, report_name)
            with redirect_stdout(io.StringIO()):
                self.player = GuiPlayer(Pipeline(cfg), speed=1.0)
                self.win = gui_qt.CountQtWindow(self.player)
            return self.win

        @staticmethod
        def _grab_bright_frac(win: gui_qt.CountQtWindow,
                              rows, cols) -> float:
            pm = win.canvas.grab()
            from PySide6.QtGui import QImage
            img = pm.toImage().convertToFormat(QImage.Format.Format_RGB888)
            arr = np.ascontiguousarray(img.bits()).astype(np.uint8)
            h, w = img.height(), img.width()
            arr = arr.reshape(h, w, 3)
            sub = arr[rows[0]:rows[1], cols[0]:cols[1]]
            if sub.size == 0:
                return 0.0
            return float((sub.astype(int).max(axis=2) > 200).sum() /
                         (sub.shape[0] * sub.shape[1]))


    class TestCountQtWindowTextOverlay(_CountQtTextBase):
        """Qt count-окно: overlay_with_text=False + счётчики в статусбаре."""

        def test_overlay_flag_false(self):
            win = self._open_window("qt_flag.md")
            self.assertFalse(win.player.overlay_with_text)   # окно сняло текст с кадра

        def test_statusbar_has_counters_line(self):
            win = self._open_window("qt_sb.md")
            with redirect_stdout(io.StringIO()):
                win.tick_once()
            msg = win.statusBar().currentMessage()
            self.assertIn("счётчики", msg)          # строка счётчиков из pipe.counters
            self.assertIn("main_line", msg)
            self.assertIn("main_zone", msg)
            self.assertIn("in=", msg)

        def test_frame_without_text_overlay(self):
            """Canvas-кадр Qt-окна БЕЗ text-overlay: яркость как у player.tick(False),
            а не с текстом (player.tick(True) заметно ярче). Контент видео одинаков,
            поэтому разница — только в нарисованном тексте."""
            cfg = _make_config(_VIDEO, "qt_frame_ref.md")
            ref = GuiPlayer(Pipeline(cfg), speed=1.0)
            ref.overlay_with_text = False
            with redirect_stdout(io.StringIO()):
                view_ref = ref.tick()

            win = self._open_window("qt_frame.md")
            with redirect_stdout(io.StringIO()):
                win.tick_once()
            frac_win = self._grab_bright_frac(win, (0, H), (0, W))
            frac_ref = float((view_ref.astype(int).max(axis=2) > 200).sum() /
                             (view_ref.shape[0] * view_ref.shape[1]))
            # кадр окна без text-overlay совпадает по яркости с player.tick(False)
            self.assertAlmostEqual(frac_win, frac_ref, delta=0.02)
            # но заметно меньше, чем кадр с текстом (player_tick(True))
            cfg2 = _make_config(_VIDEO, "qt_frame_on.md")
            on = GuiPlayer(Pipeline(cfg2), speed=1.0)
            with redirect_stdout(io.StringIO()):
                view_on = on.tick()
            frac_on = float((view_on.astype(int).max(axis=2) > 200).sum() /
                            (view_on.shape[0] * view_on.shape[1]))
            self.assertGreater(frac_on, frac_win + 0.01)


if __name__ == "__main__":
    unittest.main()
