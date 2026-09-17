"""Тесты переноса текста из кадра в Qt (задача 19): чистые функции генерации текста,
``_draw_calibration(with_text=False)`` рисует только графику, Qt-окно показывает текст
в статусбаре/QLabel под кадром. Offscreen; без PySide6 — пропуск всего модуля.

Паттерн как в tests/test_gui_qt.py: один QApplication на модуль + синтетический mp4,
exec() не вызывается — только processEvents/tick.
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
    class _SkipModule(unittest.TestCase):
        @unittest.skip("PySide6 не установлен в .venv (.venv/bin/pip install PySide6)")
        def test_pyside_available(self):
            pass


else:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    import cv2  # noqa: E402
    import numpy as np  # noqa: E402
    from PySide6.QtCore import QPoint, Qt  # noqa: E402
    from PySide6.QtGui import QImage  # noqa: E402
    from PySide6.QtTest import QTest  # noqa: E402
    from PySide6.QtWidgets import QApplication  # noqa: E402

    import visio_people_counter.gui_qt as gui_qt  # noqa: E402
    from visio_people_counter.calib_controller import CalibrationController  # noqa: E402
    from visio_people_counter.calibrate import (  # noqa: E402
        CalibrationState, _draw_calibration, hint_lines, line_hint,
        pending_size_hint, roi_mode_hint, size_point_label,
    )
    from visio_people_counter.config import Config  # noqa: E402

    W, H, FPS, N_FRAMES = 320, 180, 30.0, 300

    _APP = None
    _TD = None
    _VIDEO: Path | None = None
    _WINDOWS: list = []


    def _make_video(path: Path) -> None:
        """Синтетический mp4: статичная текстура + движущийся белый прямоугольник."""
        rng = np.random.default_rng(7)
        bg = rng.integers(60, 110, size=(H, W, 3), dtype=np.uint8)
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        assert vw.isOpened(), "не удалось открыть VideoWriter для синтетического mp4"
        for f in range(N_FRAMES):
            frame = bg.copy()
            x = int(round(40 + 240 * (f / float(N_FRAMES - 1))))
            y = int(round(60 + 30 * np.sin(f * 0.3)))
            cv2.rectangle(frame, (x, y), (x + 40, y + 90), (255, 255, 255), -1)
            vw.write(frame)
        vw.release()


    def setUpModule():
        global _APP, _TD, _VIDEO
        _APP = QApplication.instance() or QApplication([])
        _TD = tempfile.TemporaryDirectory(prefix="vpc_test_qt_text_")
        _VIDEO = Path(_TD.name) / "synth.mp4"
        _make_video(_VIDEO)


    def tearDownModule():
        global _TD
        for w in _WINDOWS:
            try:
                if not w.is_closed:
                    w.close()
                w.deleteLater()
            except RuntimeError:
                pass
        _APP.processEvents()
        if _TD is not None:
            _TD.cleanup()
            _TD = None


    # ------------------------------------------------------------------ чистые функции

    class TestHintTextPure(unittest.TestCase):
        """Чистые функции генерации текста — задачи 19 (без окна)."""

        def test_hint_lines_line(self):
            lines = hint_lines("line")
            self.assertIsInstance(lines, list)
            self.assertTrue(any("ЛИНИЯ" in ln or "линия" in ln for ln in lines))

        def test_size_point_label(self):
            self.assertEqual(size_point_label((0.3, 0.4, 0.2)), "размер: 20%")
            self.assertEqual(size_point_label((0.1, 0.1, 0.357)), "размер: 36%")

        def test_roi_mode_hint(self):
            self.assertIn("Enter", roi_mode_hint())
            self.assertIn("отмена", roi_mode_hint())

        def test_pending_size_hint_with_mouse(self):
            s = CalibrationState()
            s.size_first_point = (0.2, 0.3)   # pending — ждём второй клик
            out = pending_size_hint(s, W, H, mouse_pos=(100, 90))
            self.assertIsInstance(out, str)
            self.assertIn("px", out)      # длина в пикселях
            self.assertIn("%", out)       # доля высоты кадра

        def test_pending_size_hint_without_mouse(self):
            s = CalibrationState()
            s.size_first_point = (0.2, 0.3)
            out = pending_size_hint(s, W, H, mouse_pos=None)
            self.assertIsInstance(out, str)
            self.assertIn("мерка", out)
            self.assertNotIn("px", out)   # без курсора — только подсказка о втором клике

        def test_pending_size_hint_none_when_no_pending(self):
            s = CalibrationState()
            s.size_first_point = None
            self.assertIsNone(pending_size_hint(s, W, H, mouse_pos=(10, 10)))

        def test_line_hint(self):
            s = CalibrationState()
            s.mode = "line"
            s.line_points = [(0.1, 0.2)]
            out = line_hint("line", s)
            self.assertIsInstance(out, str)
            self.assertIn("линия", out.lower())
            self.assertIn("/2", out)
            # ни одной точки — None
            s.line_points = []
            self.assertIsNone(line_hint("line", s))
            # не line-режим — None
            s.mode = "zone"
            s.line_points = [(0.1, 0.2)]
            self.assertIsNone(line_hint("zone", s))


    # -------------------------------------------------- _draw_calibration(with_text=…)

    class TestDrawCalibrationText(unittest.TestCase):
        """_draw_calibration: with_text=False рисует только графику (без put_text)."""

        def _base(self) -> np.ndarray:
            return np.full((H, W, 3), 128, dtype=np.uint8)   # однородный серый кадр

        def _state_line_two(self) -> CalibrationState:
            s = CalibrationState()
            s.mode = "line"
            # точки по центру — подписи A/B рисуются там же (не в нижней полосе)
            s.line_points = [(0.45, 0.5), (0.55, 0.5)]
            return s

        @staticmethod
        def _white_count(img: np.ndarray, rows: tuple[int, int], cols: tuple[int, int]) -> int:
            """Чисто-БЕЛЫХ пикселей (все каналы яркие) в области — только текст, не
            цветная графика (линии/круги зелёные/жёлтые тут же отпадают)."""
            sub = img[rows[0]:rows[1], cols[0]:cols[1]]
            if sub.size == 0:
                return 0
            min_ch = sub.astype(int).min(axis=2)
            return int((min_ch > 200).sum())

        def test_bottom_hints_text_present_only_when_with_text(self):
            s = self._state_line_two()
            region = (int(H * 0.7), H), (5, 90)   # нижняя левая полоса — там подсказки режима
            img_true = _draw_calibration(self._base(), s, False, None, W, H, with_text=True)
            img_false = _draw_calibration(self._base(), s, False, None, W, H, with_text=False)
            self.assertGreater(self._white_count(img_true, *region), 0)   # текст подсказки
            self.assertEqual(self._white_count(img_false, *region), 0)    # графики там нет

        def test_line_points_graphics_preserved_without_text(self):
            """При with_text=False графика (линия A→B) всё ещё рисуется."""
            s = self._state_line_two()
            img_true = _draw_calibration(self._base(), s, False, None, W, H, with_text=True)
            img_false = _draw_calibration(self._base(), s, False, None, W, H, with_text=False)
            # зелёная линия (BGR: G=255 в канале 1) — в обоих случаях
            for img in (img_true, img_false):
                center = img[int(H * 0.45):int(H * 0.55), int(W * 0.4):int(W * 0.6)]
                self.assertGreater(int((center[:, :, 1].astype(int) > 200).sum()), 0)

        def test_ab_labels_absent_without_text(self):
            """Подписи A/B (зелёные) рисуются ТОЛЬКО при with_text=True; графика (линия)
            присутствует в обоих случаях — задачи 19 (A/B переносятся в Qt-подсказку):"""
            s = self._state_line_two()
            img_true = _draw_calibration(self._base(), s, False, None, W, H, with_text=True)
            img_false = _draw_calibration(self._base(), s, False, None, W, H, with_text=False)

            @staticmethod
            def _green(img: np.ndarray) -> int:
                g = img[:, :, 1].astype(int) > 200
                r = img[:, :, 2].astype(int) < 60
                b = img[:, :, 0].astype(int) < 60
                return int((g & r & b).sum())

            # линия A→B (графика) есть в обоих случаях
            self.assertGreater(_green(img_false), 0)
            # подписи A/B добавляют зеленые пиксели только при with_text=True
            self.assertGreater(_green(img_true), _green(img_false))


    # ------------------------------------------------------------------- Qt-окно

    class _QtTextBase(unittest.TestCase):
        """Синтетическое видео + хелперы контроллера/окна (как test_gui_qt.py)."""

        def setUp(self):
            self.ctrl: CalibrationController | None = None
            self.win: gui_qt.CalibrateQtWindow | None = None
            self.n_cfg = 0

        def tearDown(self):
            if self.win is not None:
                try:
                    self.win.close()
                except RuntimeError:
                    pass
                _WINDOWS.append(self.win)
                self.win = None
            if self.ctrl is not None:
                self.ctrl.close()
                self.ctrl = None

        def _make_config(self) -> Config:
            d = {"video": {"type": "file", "path": str(_VIDEO)}}
            self.n_cfg += 1
            p = Path(_TD.name) / f"cfg_{type(self).__name__}_{self._testMethodName}_{self.n_cfg}.yaml"
            cfg = Config.from_dict(d, str(p))
            Config.save(cfg, p)
            return cfg

        def _open_window(self, cfg: Config | None = None, **kw):
            cfg = cfg if cfg is not None else self._make_config()
            kw.setdefault("cache_frames", 50)
            ctrl = CalibrationController(cfg, **kw)
            with redirect_stdout(io.StringIO()):
                ctrl.open()
                self.win = gui_qt.CalibrateQtWindow(ctrl)
            self.ctrl = ctrl
            return self.win

        @staticmethod
        def _grab_bright_frac(win: gui_qt.CalibrateQtWindow,
                              rows: tuple[int, int], cols: tuple[int, int]) -> float:
            """Доля ярких (белый текст) пикселей в области на grabbed кадре canvas."""
            pm = win.canvas.grab()
            img = pm.toImage().convertToFormat(QImage.Format.Format_RGB888)
            arr = np.ascontiguousarray(img.bits()).astype(np.uint8)
            h, w = img.height(), img.width()
            arr = arr.reshape(h, w, 3)
            sub = arr[rows[0]:rows[1], cols[0]:cols[1]]
            if sub.size == 0:
                return 0.0
            return float((sub.astype(int).max(axis=2) > 200).sum() / (sub.shape[0] * sub.shape[1]))


    class TestCalibrateQtTextWidgets(_QtTextBase):
        """Qt-окно при frame_ui=False: текст — в статусбаре и hint_label, кадр без text."""

        def test_statusbar_hint_and_messages(self):
            win = self._open_window()
            ctrl = win.ctrl
            win.actions["line"].trigger()   # режим line → подсказка «ЛИНИЯ»
            win.tick()
            # статусбар: counter_status_text содержит «кадр»
            self.assertIn("кадр", win.statusBar().currentMessage())
            # hint_label под кадром не пустой и показывает подсказку режима
            self.assertTrue(win.hint_label.text())
            self.assertIn("ЛИНИЯ", win.hint_label.text())

        def test_messages_in_statusbar_widget(self):
            win = self._open_window()
            ctrl = win.ctrl
            with redirect_stdout(io.StringIO()):   # +линия печатает в stdout
                win.actions["new_line"].trigger()
            win.tick()
            msgs = ctrl.messages
            self.assertTrue(msgs)
            # активные уведомления — в постоянном виджете статусбара (задача 19)
            self.assertIn(msgs[-1], win._msg_widget.text())

        def test_canvas_has_no_text_overlay(self):
            """frame_ui=False: на кадре БЕЗ текста меньше белых пикселей, чем frame_ui=True
            (cv2-режим), где текст рисуется на кадре — задачи 19."""
            win = self._open_window()
            ctrl = win.ctrl
            with redirect_stdout(io.StringIO()):
                win.actions["line"].trigger()   # режим line — подсказки появятся при frame_ui=True
                win.tick()                      # frame_ui=False: только графика
                frac_false = self._grab_bright_frac(
                    win, (int(H * ctrl.scale * 0.7), int(H * ctrl.scale)), (5, 90))
                ctrl.frame_ui = True
                win.tick()                      # текст на кадре (подсказки режима)
                frac_true = self._grab_bright_frac(
                    win, (int(H * ctrl.scale * 0.7), int(H * ctrl.scale)), (5, 90))
            self.assertGreater(frac_true, frac_false)   # текст на кадре добавляет белые пиксели
            self.assertLess(frac_false, 0.2)    # без текста — существенно меньше (видео не в счёт)


    class TestCalibrateQtCv2LikeText(_QtTextBase):
        """Qt-окно при frame_ui=True (cv2-режим): текст на кадре, hint_label пустой."""

        def test_frame_with_text_hint_label_empty(self):
            win = self._open_window()
            ctrl = win.ctrl
            # режим cv2-окна: текст рисуется на кадре как раньше
            ctrl.frame_ui = True
            with redirect_stdout(io.StringIO()):
                win.tick()
            # подсказки под кадром пустые — весь текст уже на кадре
            self.assertEqual(win.hint_label.text(), "")
            # нижняя левая полоса кадра: есть белый текст (подсказка режима)
            scale = ctrl.scale
            frac_true = self._grab_bright_frac(
                win, (int(H * scale * 0.7), int(H * scale)), (5, 90))
            self.assertGreater(frac_true, 0.01)

        def test_frame_text_more_than_without(self):
            """Переключение frame_ui True→False убирает текст с кадра (строго меньше)."""
            win = self._open_window()
            ctrl = win.ctrl
            with redirect_stdout(io.StringIO()):
                win.tick()   # frame_ui=False — только графика
                frac_false = self._grab_bright_frac(
                    win, (int(H * ctrl.scale * 0.7), int(H * ctrl.scale)), (5, 90))
                ctrl.frame_ui = True
                win.tick()   # текст на кадре
                frac_true = self._grab_bright_frac(
                    win, (int(H * ctrl.scale * 0.7), int(H * ctrl.scale)), (5, 90))
            self.assertGreater(frac_true, frac_false)


if __name__ == "__main__":
    unittest.main()
