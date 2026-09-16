"""Тесты Qt-бэкенда окна калибратора (gui_qt.py, задача 16) — offscreen.

Без PySide6 модуль пропускается целиком (Qt-бэкенд опциональный). Синоним:
синтетический mp4 320x180 @ 30 fps, 300 кадров (паттерн test_calib_controller.py);
один QApplication на модуль; exec() в тестах НЕ вызывается — только processEvents/tick.
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
    from PySide6.QtCore import QEvent, QPoint, Qt  # noqa: E402
    from PySide6.QtGui import QKeyEvent  # noqa: E402
    from PySide6.QtTest import QTest  # noqa: E402
    from PySide6.QtWidgets import QApplication  # noqa: E402

    import visio_people_counter.gui_qt as gui_qt  # noqa: E402
    from visio_people_counter.calib_controller import CalibrationController  # noqa: E402
    from visio_people_counter.config import Config  # noqa: E402

    W, H, FPS, N_FRAMES = 320, 180, 30.0, 300   # 10 секунд

    _APP = None
    _TD = None
    _VIDEO: Path | None = None
    _WINDOWS: list = []   # окна, живущие до конца модуля (deleteLater в tearDownModule)


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
        """Один QApplication на модуль + один общий синтетический mp4."""
        global _APP, _TD, _VIDEO
        _APP = QApplication.instance() or QApplication([])
        _TD = tempfile.TemporaryDirectory(prefix="vpc_test_gui_qt_")
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
                pass   # C++ объект уже освобождён — ничего не делаем
        _APP.processEvents()
        if _TD is not None:
            _TD.cleanup()
            _TD = None


    def _kevent(key, text="", mods=Qt.KeyboardModifier.NoModifier) -> QKeyEvent:
        """Сконструированный QKeyEvent (KeyPress) для qt_key_to_cv2."""
        return QKeyEvent(QEvent.Type.KeyPress, key, mods, text)


    class _QtBase(unittest.TestCase):
        """Общий синтетический видеофайл + хелперы создания контроллера/окна."""

        def setUp(self):
            self.ctrl: CalibrationController | None = None
            self.win: gui_qt.CalibrateQtWindow | None = None
            self.n_cfg = 0

        def tearDown(self):
            if self.win is not None:
                # close() → closeEvent → ctrl.close(); идемпотентно и для уже закрытых
                try:
                    self.win.close()
                except RuntimeError:
                    pass
                _WINDOWS.append(self.win)
                self.win = None
            if self.ctrl is not None:
                self.ctrl.close()   # если окно не создавалось — закрываем сами
                self.ctrl = None

        def _make_config(self) -> Config:
            """Минимальный конфиг (file-видео); недостающие блоки — дефолты Config."""
            d = {"video": {"type": "file", "path": str(_VIDEO)}}
            self.n_cfg += 1
            p = Path(_TD.name) / f"cfg_{type(self).__name__}_{self._testMethodName}_{self.n_cfg}.yaml"
            cfg = Config.from_dict(d, str(p))
            Config.save(cfg, p)
            return cfg

        def _open_window(self, cfg: Config | None = None, **kw):
            """Контроллер (open()) + CalibrateQtWindow; stdout заглушен (Pipeline печатает)."""
            cfg = cfg if cfg is not None else self._make_config()
            kw.setdefault("cache_frames", 50)
            ctrl = CalibrationController(cfg, **kw)
            with redirect_stdout(io.StringIO()):
                ctrl.open()
                self.win = gui_qt.CalibrateQtWindow(ctrl)
            self.ctrl = ctrl
            return self.win


    class TestKeyMapping(_QtBase):
        """qt_key_to_cv2: QKeyEvent → код on_key (формат waitKey&0xFF)."""

        def test_printable_chars(self):
            self.assertEqual(gui_qt.qt_key_to_cv2(
                _kevent(int(Qt.Key_L), "l")), ord("l"))          # 108
            self.assertEqual(gui_qt.qt_key_to_cv2(
                _kevent(int(Qt.Key_5), "5")), ord("5"))          # 53
            self.assertEqual(gui_qt.qt_key_to_cv2(
                _kevent(int(Qt.Key_Comma), ",")), ord(","))

        def test_special_keys(self):
            self.assertEqual(gui_qt.qt_key_to_cv2(
                _kevent(int(Qt.Key_Return), "\r")), 13)          # Enter
            self.assertEqual(gui_qt.qt_key_to_cv2(
                _kevent(int(Qt.Key_Escape), "")), 27)            # ESC
            self.assertEqual(gui_qt.qt_key_to_cv2(
                _kevent(int(Qt.Key_Space), " ")), 32)            # Space
            self.assertEqual(gui_qt.qt_key_to_cv2(
                _kevent(int(Qt.Key_Backspace), "")), 8)

        def test_ctrl_combo_is_none(self):
            ev = _kevent(int(Qt.Key_L), "l", Qt.KeyboardModifier.ControlModifier)
            self.assertIsNone(gui_qt.qt_key_to_cv2(ev))          # Ctrl — игнорируется


    class TestWindowCreation(_QtBase):
        """Окно создаётся на открытом контроллере; размер canvas и статусбар."""

        def test_window_canvas_size_and_statusbar(self):
            win = self._open_window()
            ctrl = win.ctrl
            # авто-масштаб (задача 14) поднял scale; canvas = width*scale × height*scale
            self.assertEqual(ctrl.width, W)
            self.assertEqual(ctrl.height, H)
            expected = (int(W * ctrl.scale), int(H * ctrl.scale))
            self.assertEqual((win.canvas.size().width(), win.canvas.size().height()), expected)
            # 320x180 → авто-масштаб до 2.0 (как в test_calib_controller)
            self.assertEqual(expected, (640, 360))
            # статусбар содержит «кадр»
            self.assertIn("кадр", win.statusBar().currentMessage())
            # таймер: fps источника 30 → интервал ~33 мс
            self.assertTrue(1 <= win.timer.interval() <= 40)


    class TestTick(_QtBase):
        """tick(): кадр показывается, frame_index ходит [n]/[p], quit закрывает окно."""

        def test_ticks_show_frame_and_n_advances(self):
            win = self._open_window()
            ctrl = win.ctrl
            first = ctrl.frame_index
            for _ in range(3):
                win.tick()   # тик вручную (без exec/таймера) — кадр на canvas
            self.assertFalse(win.canvas._pixmap.isNull())
            self.assertFalse(ctrl.quit_requested)
            # листание [n] через клавиатуру окна → frame_index вырос; следующий тик — живой кадр
            win.keyPressEvent(_kevent(int(Qt.Key_N), "n"))
            self.assertEqual(ctrl.frame_index, first + 1)
            win.tick()   # следующий тик — живой кадр (окно не закрылось)
            self.assertFalse(win.is_closed)

        def test_quit_closes_window_and_source(self):
            win = self._open_window()
            ctrl = win.ctrl
            self.assertFalse(win.is_closed)
            win.keyPressEvent(_kevent(int(Qt.Key_Q), "q"))       # q → quit_requested
            self.assertTrue(ctrl.quit_requested)
            win.tick()                                            # None из step() → closeWindow()
            self.assertTrue(win.is_closed)
            self.assertIsNone(ctrl._pipe)   # closeEvent → ctrl.close(): источник закрыт


    class TestButtons(_QtBase):
        """Кнопки тулбара: trigger() → ctrl.on_button; подсветка checkable."""

        def test_line_action_enters_mode(self):
            win = self._open_window()
            win.actions["line"].trigger()
            self.assertEqual(win.ctrl.state.mode, "line")
            self.assertTrue(win.actions["line"].isChecked())   # подсветка активного режима

        def test_show_all_toggles(self):
            win = self._open_window()
            self.assertFalse(win.ctrl.show_all)
            win.actions["show_all"].trigger()
            self.assertTrue(win.ctrl.show_all)
            self.assertTrue(win.actions["show_all"].isChecked())
            win.actions["show_all"].trigger()
            self.assertFalse(win.ctrl.show_all)
            self.assertFalse(win.actions["show_all"].isChecked())


    class TestMouse(_QtBase):
        """Клики по canvas (QTest.mouseClick) → unscale_mouse → ctrl.on_click."""

        def test_two_line_clicks_via_canvas(self):
            win = self._open_window()
            ctrl = win.ctrl
            win.actions["line"].trigger()
            # координаты CANVAS (scale=2.0): canvas-px = исходные * scale;
            # ниже панели кнопок (панель кончается ~y=66 в координатах кадра)
            QTest.mouseClick(win.canvas, Qt.MouseButton.LeftButton, pos=QPoint(100, 300))
            self.assertEqual(len(ctrl.state.line_points), 1)
            # клик попал в on_click в исходных координатах: (100,300)/2.0 = (50,150)
            self.assertEqual(ctrl.state.line_points[0], (round(50 / W, 4), round(150 / H, 4)))
            QTest.mouseClick(win.canvas, Qt.MouseButton.LeftButton, pos=QPoint(400, 330))
            self.assertEqual(len(ctrl.state.line_points), 2)


    class TestRoiResize(_QtBase):
        """ROI: применение → следующий tick обновляет размер canvas под новый кроп."""

        def test_roi_apply_resizes_canvas(self):
            win = self._open_window()
            ctrl = win.ctrl
            with redirect_stdout(io.StringIO()):   # переоткрытие Pipeline печатает
                win.actions["roi"].trigger()
                QTest.mouseClick(win.canvas, Qt.MouseButton.LeftButton, pos=QPoint(120, 240))
                QTest.mouseClick(win.canvas, Qt.MouseButton.LeftButton, pos=QPoint(600, 320))
                win.keyPressEvent(_kevent(int(Qt.Key_Return), "\r"))   # Enter — принять

            self.assertIsNotNone(ctrl.cfg.processing.roi)
            # источник переоткрыт с кропом: кадр меньше исходного
            self.assertLess(ctrl.width, W)
            self.assertLess(ctrl.height, H)
            win.tick()   # следующий тик — кадр нового размера, canvas пересчитан
            expected = (int(ctrl.width * ctrl.scale), int(ctrl.height * ctrl.scale))
            self.assertEqual((win.canvas.size().width(), win.canvas.size().height()), expected)


    class TestPysideLazyImport(unittest.TestCase):
        """Остальной код проекта не тянет PySide6 (проверка в чистом интерпретаторе)."""

        def test_package_import_does_not_pull_pyside(self):
            import subprocess
            code = ("import sys; import visio_people_counter, "
                    "visio_people_counter.calibrate, visio_people_counter.__main__; "
                    "assert 'PySide6' not in sys.modules")
            r = subprocess.run([sys.executable, "-c", code],
                               capture_output=True, text=True, cwd=str(ROOT))
            self.assertEqual(r.returncode, 0,
                             f"imпорт без PySide6 упал: {r.stderr}")


if __name__ == "__main__":
    unittest.main()
