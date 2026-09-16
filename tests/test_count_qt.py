"""Тесты Qt-окна подсчёта (gui_qt.CountQtWindow, задача 17) — offscreen.

Без PySide6 модуль пропускается целиком (Qt-бэкенд опциональный). Паттерн
test_gui_qt.py: синтетический mp4 320x180 @ 30 fps; один QApplication на модуль;
exec() в тестах НЕ вызывается (кроме одного дымового run_count_qt), окно ведётся
ручными tick_once(). Логика — в GuiPlayer (общий с cv2-режимом): окно только
кнопки → handle_key и canvas ← tick().
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
    # offscreen ДО импорта PySide6 (без дисплея)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    import cv2  # noqa: E402
    import numpy as np  # noqa: E402
    from PySide6.QtWidgets import QApplication  # noqa: E402

    import visio_people_counter.gui_qt as gui_qt  # noqa: E402
    from visio_people_counter.config import Config  # noqa: E402
    from visio_people_counter.gui import GuiPlayer  # noqa: E402
    from visio_people_counter.pipeline import Pipeline  # noqa: E402

    W, H, FPS, N_FRAMES = 320, 180, 30.0, 40   # короткое видео: EOF наступает быстро

    _APP = None
    _TD = None
    _VIDEO: Path | None = None
    _WINDOWS: list = []   # окна, живущие до конца модуля (deleteLater в tearDownModule)


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


    def setUpModule():
        global _APP, _TD, _VIDEO
        _APP = QApplication.instance() or QApplication([])
        _TD = tempfile.TemporaryDirectory(prefix="vpc_test_count_qt_")
        _VIDEO = Path(_TD.name) / "synth.mp4"
        _make_video(_VIDEO)


    def tearDownModule():
        global _TD
        for w in _WINDOWS:
            try:
                if not w.is_closed:
                    with redirect_stdout(io.StringIO()):
                        w.close()   # closeEvent → player.finalize (печать в stdout)
                w.deleteLater()
            except RuntimeError:
                pass   # C++ объект уже освобождён — ничего не делаем
        _APP.processEvents()
        if _TD is not None:
            _TD.cleanup()
            _TD = None


    class _CountQtBase(unittest.TestCase):
        """Общий синтетический видеофайл + хелпер создания плеера/окна."""

        def setUp(self):
            self.win: gui_qt.CountQtWindow | None = None
            self.player: GuiPlayer | None = None
            self.n_cfg = 0

        def tearDown(self):
            if self.win is not None:
                try:
                    with redirect_stdout(io.StringIO()):
                        self.win.close()   # идемпотентно (finalize тоже)
                except RuntimeError:
                    pass
                _WINDOWS.append(self.win)
                self.win = None
            # плеер закрыт в closeEvent окна (finalize → pipe.close()); страховка
            if self.player is not None:
                with redirect_stdout(io.StringIO()):
                    self.player.pipeline.close()
                self.player = None

        def _make_config(self, report_name: str) -> Config:
            """Минимальный конфиг с явным output.report_path (tmp-папка)."""
            self.n_cfg += 1
            return Config.from_dict({
                "video": {"type": "file", "path": str(_VIDEO)},
                "output": {"report_path": str(Path(_TD.name) / report_name)},
            })

        def _open_window(self, cfg: Config):
            """GuiPlayer (без cv2-run!) + CountQtWindow; stdout заглушен."""
            with redirect_stdout(io.StringIO()):   # Pipeline.build() печатает лог
                self.player = GuiPlayer(Pipeline(cfg), speed=1.0)
                self.win = gui_qt.CountQtWindow(self.player)
            return self.win

        def _report_path(self, cfg: Config) -> Path:
            return Path(cfg.output.report_path)


    class TestCountQtWindowCreation(_CountQtBase):
        """Окно создаётся на синтетическом видео; canvas и статусбар."""

        def test_window_created_and_ticks_advance_frame(self):
            win = self._open_window(self._make_config("report_create.md"))
            player = win.player
            for _ in range(3):
                with redirect_stdout(io.StringIO()):
                    win.tick_once()   # тик вручную (без exec/таймера)
            self.assertFalse(win.canvas._pixmap.isNull())   # кадр на canvas
            self.assertGreaterEqual(player.frame_index, 2)  # frame прошёл
            self.assertFalse(win.is_closed)
            # статусбар: строка статуса плеера
            self.assertIn("speed=", win.statusBar().currentMessage())
            # таймер после первого тика: fps=30, speed=1.0 → ~33 мс
            self.assertTrue(20 <= win.timer.interval() <= 40)


    class TestCountQtWindowButtons(_CountQtBase):
        """Кнопки тулбара → player.handle_key (те же коды, что клавиши cv2)."""

        def test_pause_button(self):
            win = self._open_window(self._make_config("report_pause.md"))
            player = win.player
            with redirect_stdout(io.StringIO()):
                win.tick_once()
                idx = player.frame_index
                win.actions["pause"].trigger()
            self.assertTrue(player._paused)
            self.assertTrue(win.actions["pause"].isChecked())   # подсветка checkable
            with redirect_stdout(io.StringIO()):
                win.tick_once()
                win.tick_once()
            self.assertEqual(player.frame_index, idx)   # пауза: кадры не читаются
            # снятие паузы
            with redirect_stdout(io.StringIO()):
                win.actions["pause"].trigger()
                win.tick_once()
            self.assertFalse(player._paused)
            self.assertGreater(player.frame_index, idx)

        def test_speed_buttons(self):
            win = self._open_window(self._make_config("report_speed.md"))
            player = win.player
            before = player.speed
            with redirect_stdout(io.StringIO()):
                win.tick_once()   # первый тик: build() → fps известен таймеру
                win.actions["faster"].trigger()
            self.assertAlmostEqual(player.speed, before * 1.5)
            # интервал таймера пересчитан: fps=30, speed=1.5 → ~22 мс
            self.assertTrue(10 <= win.timer.interval() <= 30)
            with redirect_stdout(io.StringIO()):
                win.actions["slower"].trigger()
            self.assertAlmostEqual(player.speed, before)

        def test_scale_buttons(self):
            win = self._open_window(self._make_config("report_scale.md"))
            player = win.player
            before = player.scale
            with redirect_stdout(io.StringIO()):
                win.actions["scale_up"].trigger()
            self.assertNotEqual(player.scale, before)
            with redirect_stdout(io.StringIO()):
                win.actions["scale_down"].trigger()
            self.assertEqual(player.scale, before)   # цикл пресетов 0.5/1/1.5/2

        def test_quit_finalizes_and_closes(self):
            cfg = self._make_config("report_quit.md")
            win = self._open_window(cfg)
            player = win.player
            with redirect_stdout(io.StringIO()):
                win.tick_once()
                win.actions["quit"].trigger()      # «Выход» → handle_key(q)
            self.assertTrue(player._stop)
            with redirect_stdout(io.StringIO()) as out:
                win.tick_once()                    # None из tick() → finalize + close
            self.assertTrue(win.is_closed)
            self.assertTrue(player._finalized)
            text = out.getvalue()
            self.assertIn("ФИНАЛЬНАЯ СВОДКА", text)
            report = self._report_path(cfg)
            self.assertTrue(report.is_file())       # отчёт записан
            self.assertIn("окно закрыто (GUI)", report.read_text(encoding="utf-8"))


    class TestCountQtWindowEof(_CountQtBase):
        """EOF: короткий источник исчерпан → окно завершается, отчёт записан."""

        def test_eof_closes_window_and_writes_report(self):
            cfg = self._make_config("report_eof.md")
            win = self._open_window(cfg)
            player = win.player
            with redirect_stdout(io.StringIO()) as out:
                for _ in range(N_FRAMES + 5):
                    if win.is_closed:
                        break
                    win.tick_once()
            self.assertTrue(win.is_closed)
            self.assertGreaterEqual(player.frame_index, N_FRAMES - 1)
            self.assertTrue(player._finalized)
            self.assertIn("ФИНАЛЬНАЯ СВОДКА", out.getvalue())   # финальная сводка
            report = self._report_path(cfg)
            self.assertTrue(report.is_file())                   # report.md записан
            self.assertIn("окно закрыто (GUI)", report.read_text(encoding="utf-8"))


    class TestRunCountQtSmoke(_CountQtBase):
        """Дым: run_count_qt доводит окно на синтетическом видео до EOF (exec())."""

        def test_run_count_qt_reaches_eof(self):
            cfg = self._make_config("report_smoke.md")
            pipe = Pipeline(cfg)
            with redirect_stdout(io.StringIO()) as out:
                rc = gui_qt.run_count_qt(pipe, speed=2.0, initial_scale=1.0)
            self.assertEqual(rc, 0)
            # финальная сводка и отчёт (запись — до close источника)
            self.assertIn("ФИНАЛЬНАЯ СВОДКА", out.getvalue())
            report = self._report_path(cfg)
            self.assertTrue(report.is_file())
            self.assertGreater(pipe.frames_processed, 0)


if __name__ == "__main__":
    unittest.main()
