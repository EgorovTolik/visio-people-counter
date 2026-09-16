"""Тесты авто-масштаба окна при маленьком кадре (задача 14).

ВСЕ тесты — без реального окна: чистая функция ``gui.auto_ui_scale`` +
интеграционная проверка, что шаг «выбор scale» подключён в calibrate и
GuiPlayer.run (source-assert, окно не поднимается).
"""

from __future__ import annotations

import inspect
import unittest

from visio_people_counter.gui import (
    MAX_AUTO_SCALE,
    MIN_UI_H,
    MIN_UI_W,
    GuiPlayer,
    auto_ui_scale,
)


class AutoUiScaleTest(unittest.TestCase):
    """Чистая функция выбора масштаба отображения."""

    def test_large_frame_keeps_user_scale(self):
        # кадр больше минимумов по обеим осям → user_scale без изменений
        self.assertEqual(auto_ui_scale(1920, 1080, 1.0), 1.0)
        self.assertEqual(auto_ui_scale(1280, 720, 0.5), 0.5)   # дым из спецификации
        self.assertEqual(auto_ui_scale(640, 320, 1.0), 1.0)    # ровно минимумы

    def test_tiny_frame_200x80(self):
        # max(640/200=3.2, 320/80=4.0) = 4.0 → 4.0 (дым из спецификации)
        self.assertEqual(auto_ui_scale(200, 80, 1.0), 4.0)

    def test_width_dominates(self):
        # 640/200=3.2 > 320/400=0.8 → 3.2
        self.assertEqual(auto_ui_scale(200, 400, 1.0), 3.2)

    def test_rounded_up_to_0_1(self):
        # 640/300 = 2.1333… → вверх до 0.1 → 2.2
        self.assertEqual(auto_ui_scale(300, 300, 1.0), 2.2)
        # 320/130 = 2.4615… → 2.5
        self.assertEqual(auto_ui_scale(640, 130, 1.0), 2.5)

    def test_cap_at_max_auto_scale(self):
        # 640/20=32 > потолка → MAX_AUTO_SCALE (25.0)
        self.assertEqual(auto_ui_scale(20, 20, 1.0), MAX_AUTO_SCALE)
        self.assertEqual(MAX_AUTO_SCALE, 25.0)

    def test_50x50(self):
        # max(640/50=12.8, 320/50=6.4) → 12.8 (ниже потолка — потолок не срабатывает)
        self.assertEqual(auto_ui_scale(50, 50, 1.0), 12.8)

    def test_zero_dimensions_return_user_scale(self):
        # деление на 0/неподготовленный источник → user_scale без изменений
        self.assertEqual(auto_ui_scale(0, 320, 1.0), 1.0)
        self.assertEqual(auto_ui_scale(640, 0, 1.5), 1.5)
        self.assertEqual(auto_ui_scale(0, 0, 2.0), 2.0)

    def test_user_scale_is_floor(self):
        # пользовательский --scale — минимум: авто не ниже user_scale
        self.assertEqual(auto_ui_scale(200, 80, 4.5), 4.5)   # > required 4.0
        self.assertEqual(auto_ui_scale(1920, 1080, 2.0), 2.0)

    def test_result_bounded(self):
        for w in (0, 1, 50, 200, 640, 1920):
            for h in (0, 1, 80, 320, 1080):
                for user in (0.05, 1.0, 2.0, 8.0):
                    s = auto_ui_scale(w, h, user)
                    self.assertGreaterEqual(s, user)
                    self.assertLessEqual(s, MAX_AUTO_SCALE)


class AutoUiScaleIntegrationTest(unittest.TestCase):
    """Шаг «выбор scale» подключён в calibrate и count --gui (без окна)."""

    def test_calibrate_uses_auto_ui_scale(self):
        # задача 15: логика выбора масштаба — в контроллере (open/_apply_roi_change),
        # run_calibration остался тонким cv2-драйвером
        from visio_people_counter.calib_controller import CalibrationController
        src = inspect.getsource(CalibrationController)
        self.assertIn("auto_ui_scale(self.w, self.h, scale_base)", src)
        self.assertIn("малое изображение", src)   # уведомление в окне + print
        from visio_people_counter import calibrate
        drv = inspect.getsource(calibrate.run_calibration)
        self.assertIn("CalibrationController", drv)   # драйвер создаёт контроллер

    def test_gui_player_run_uses_auto_ui_scale(self):
        src = inspect.getsource(GuiPlayer.run)
        self.assertIn(
            "auto_ui_scale(src.width, src.height, self.scale)", src)


class AutoUiScaleConstantsTest(unittest.TestCase):
    """Константы минимального UI из спецификации."""

    def test_min_ui(self):
        self.assertEqual((MIN_UI_W, MIN_UI_H), (640, 320))


if __name__ == "__main__":
    unittest.main()
