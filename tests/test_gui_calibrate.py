"""Тесты GUI-overlay и чистых функций калибровки (задача 16).

ВСЕ тесты — без реального окна: GuiOverlay.draw работает на синтетических
кадрах, calibrate-логика — через CalibrationState/apply_calibration,
GuiPlayer.available() — чистая проверка env + cv2.getBuildInformation().
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from visio_people_counter.calibrate import (
    CalibrationState,
    apply_calibration,
    click_to_norm,
    size_digit_to_fraction,
)
from visio_people_counter.config import (
    Config,
    LineCounterConfig,
    ZoneCounterConfig,
)
from visio_people_counter.gui import GuiOverlay, GuiPlayer
from visio_people_counter.line_counter import LineCounter, ZoneCounter
from visio_people_counter.motion_detector import Blob
from visio_people_counter.tracker_adapter import TrackedObject

W, H = 480, 320


def _frame(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 60, size=(H, W, 3), dtype=np.uint8)


def _debug_cfg(**over):
    """Блок debug (DebugConfig) с переопределениями."""
    d = {"show_bboxes": True, "show_all_blobs": False, "show_mask": False,
         "show_counters": True}
    d.update(over)
    return Config.from_dict({"debug": d}).debug


def _line_counter(w=W, h=H) -> LineCounter:
    return LineCounter(LineCounterConfig(id="main_line", a=(0.2, 0.2), b=(0.8, 0.8)),
                       w=w, h=h)


def _objects() -> list[TrackedObject]:
    return [
        TrackedObject(track_id=5, x=100, y=80, w=40, h=90, cx=120.0, cy=125.0, age_frames=3),
        TrackedObject(track_id=-1, x=200, y=150, w=30, h=70, cx=215.0, cy=185.0, age_frames=0),
    ]


def _blobs() -> list[Blob]:
    return [Blob(x=90, y=70, w=60, h=110, area=3200, cx=120.0, cy=125.0)]


# ---------------------------------------------------------------------------
# GuiOverlay
# ---------------------------------------------------------------------------

class TestGuiOverlay(unittest.TestCase):

    def test_all_debug_options_combined_does_not_crash(self):
        cfg = _debug_cfg(show_bboxes=True, show_all_blobs=True, show_mask=True,
                         show_counters=True)
        frame = _frame()
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[60:180, 90:150] = 255
        out = GuiOverlay(cfg).draw(
            frame, objects=_objects(), blobs=_blobs(), mask=mask,
            counters=[_line_counter()], status_text="speed=1.00x")
        self.assertIs(out, frame)  # in-place
        self.assertEqual(frame.shape, (H, W, 3))

    def test_no_options_frame_unchanged(self):
        cfg = _debug_cfg(show_bboxes=False, show_all_blobs=False,
                         show_mask=False, show_counters=False)
        frame = _frame()
        before = frame.copy()
        GuiOverlay(cfg).draw(frame)
        np.testing.assert_array_equal(frame, before)

    def test_line_pixels_differ_from_background(self):
        """Счётчик реально нарисован: пиксели линии отличаются от фона (по маске)."""
        cfg = _debug_cfg(show_bboxes=False, show_all_blobs=False,
                         show_mask=False, show_counters=True)
        bg = np.full((H, W, 3), 50, dtype=np.uint8)   # однородный фон — предсказуемо
        frame = bg.copy()
        GuiOverlay(cfg).draw(frame, counters=[_line_counter()])
        changed = (frame.astype(int) - bg.astype(int)).sum(axis=2) > 30
        self.assertGreater(int(changed.sum()), 50, "линия/подпись не нарисованы")
        # середина линии a=(0.2,0.2)->b=(0.8,0.8) = центр кадра — зелёная линия
        px = frame[H // 2, W // 2].astype(int)
        self.assertGreater(px[1], 150)                 # G-канал
        self.assertLess(px[0], 120)                    # B-канал

    def test_line_not_drawn_when_counters_disabled(self):
        cfg = _debug_cfg(show_bboxes=False, show_all_blobs=False,
                         show_mask=False, show_counters=False)
        bg = np.full((H, W, 3), 50, dtype=np.uint8)
        frame = bg.copy()
        GuiOverlay(cfg).draw(frame, counters=[_line_counter()])
        np.testing.assert_array_equal(frame, bg)

    def test_show_mask_mixes_frame(self):
        cfg = _debug_cfg(show_bboxes=False, show_all_blobs=False,
                         show_mask=True, show_counters=False)
        frame = _frame(seed=1)
        before = frame.copy()
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[40:200, 100:300] = 255
        GuiOverlay(cfg).draw(frame, mask=mask)
        self.assertFalse(np.array_equal(frame, before), "show_mask не изменил кадр")

    def test_bboxes_and_blobs_drawn(self):
        cfg = _debug_cfg(show_bboxes=True, show_all_blobs=True,
                         show_mask=False, show_counters=False)
        bg = np.full((H, W, 3), 50, dtype=np.uint8)
        frame = bg.copy()
        GuiOverlay(cfg).draw(frame, objects=_objects(), blobs=_blobs())
        changed = (frame.astype(int) - bg.astype(int)).sum(axis=2) > 30
        self.assertGreater(int(changed.sum()), 50)

    def test_zone_counter_drawn(self):
        cfg = _debug_cfg(show_bboxes=False, show_all_blobs=False,
                         show_mask=False, show_counters=True)
        bg = np.full((H, W, 3), 50, dtype=np.uint8)
        frame = bg.copy()
        zc = ZoneCounter(ZoneCounterConfig(id="z1", polygon=[
            (0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)]), w=W, h=H)
        GuiOverlay(cfg).draw(frame, counters=[zc])
        changed = (frame.astype(int) - bg.astype(int)).sum(axis=2) > 30
        self.assertGreater(int(changed.sum()), 50)


# ---------------------------------------------------------------------------
# calibrate: чистые функции «клик → конфиг»
# ---------------------------------------------------------------------------

class TestCalibrationPure(unittest.TestCase):

    def test_click_to_norm(self):
        self.assertEqual(click_to_norm(120, 160, W, H), (0.25, 0.5))
        # зажим в 0..1
        self.assertEqual(click_to_norm(-5, 4000, W, H), (0.0, 1.0))
        with self.assertRaises(ValueError):
            click_to_norm(10, 10, 0, H)

    def test_size_digit_to_fraction(self):
        self.assertEqual(size_digit_to_fraction(1), 0.05)
        self.assertEqual(size_digit_to_fraction(2), 0.10)
        self.assertEqual(size_digit_to_fraction(9), 0.45)
        for bad in (0, 10, -1):
            with self.assertRaises(ValueError):
                size_digit_to_fraction(bad)

    def test_two_clicks_line_added_to_config(self):
        cfg = Config.from_dict({})
        st = CalibrationState(counter_id="main_line")
        st.set_mode("line")
        st.handle_click(*click_to_norm(96, 64, W, H))     # (0.2, 0.2)
        st.handle_click(*click_to_norm(384, 256, W, H))   # (0.8, 0.8)
        changed = apply_calibration(cfg, st)
        self.assertEqual(len(changed), 1)
        self.assertEqual(len(cfg.counters), 1)
        c = cfg.counters[0]
        self.assertIsInstance(c, LineCounterConfig)
        self.assertEqual(c.id, "main_line")
        # координаты нормализованы и валидны (собирается в реальный счётчик)
        for p in (c.a, c.b):
            self.assertTrue(0.0 <= p[0] <= 1.0 and 0.0 <= p[1] <= 1.0)
        LineCounter(c, w=W, h=H)  # не бросает → валидна

    def test_line_update_existing_counter(self):
        cfg = Config.from_dict({"counters": [{
            "id": "m", "type": "line", "a": [0.1, 0.1], "b": [0.9, 0.9]}]})
        st = CalibrationState(counter_id="m")
        st.set_mode("line")
        st.handle_click(0.3, 0.4)
        st.handle_click(0.7, 0.6)
        changed = apply_calibration(cfg, st)
        self.assertEqual(len(cfg.counters), 1)          # обновлён, не добавлен
        self.assertEqual(cfg.counters[0].a, (0.3, 0.4))
        self.assertEqual(cfg.counters[0].b, (0.7, 0.6))
        self.assertIn("(0.3", changed[0])               # diff содержит новые координаты

    def test_four_clicks_zone(self):
        cfg = Config.from_dict({})
        st = CalibrationState(counter_id="gate")
        st.set_mode("zone")
        for pt in ((0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7)):
            st.handle_click(*pt)
        poly = st.finish_zone()
        self.assertEqual(len(poly), 4)
        apply_calibration(cfg, st)
        c = cfg.counters[0]
        self.assertIsInstance(c, ZoneCounterConfig)
        self.assertEqual(c.polygon, [tuple(p) for p in poly])
        ZoneCounter(c, w=W, h=H)  # валидный полигон

    def test_zone_fewer_than_three_points_rejected(self):
        st = CalibrationState(counter_id="z")
        st.set_mode("zone")
        st.handle_click(0.5, 0.5)
        with self.assertRaises(ValueError):
            st.finish_zone()

    def test_size_point_recorded_and_applied(self):
        cfg = Config.from_dict({})
        st = CalibrationState(counter_id="m")
        st.set_mode("size")
        st.handle_click(0.42, 0.9)      # x фиксируется кликом (y не используется)
        st.set_size_height(3)           # → 15% высоты кадра
        self.assertEqual(st.size_points, [(0.42, 0.15)])
        apply_calibration(cfg, st)
        self.assertTrue(cfg.size_profile.enabled)
        self.assertIn((0.42, 0.15), cfg.size_profile.control_points)

    def test_mode_switch_clears_other_geometry(self):
        st = CalibrationState(counter_id="m")
        st.set_mode("line")
        st.handle_click(0.1, 0.1)
        st.set_mode("zone")             # переключение сбрасывает набросок линии
        self.assertEqual(st.line_points, [])
        st.handle_click(0.2, 0.2)
        st.handle_click(0.3, 0.3)
        st.set_mode("line")
        self.assertEqual(st.zone_points, [])

    def test_apply_empty_state_changes_nothing(self):
        cfg = Config.from_dict({})
        changed = apply_calibration(cfg, CalibrationState(counter_id="m"))
        self.assertEqual(changed, [])
        self.assertEqual(cfg.counters, [])


class TestConfigRoundtrip(unittest.TestCase):
    """Config.save → load сохраняет всё (после калибровки)."""

    def test_save_load_roundtrip(self):
        cfg = Config.from_dict({})
        st = CalibrationState(counter_id="main_line")
        st.set_mode("line")
        st.handle_click(0.2, 0.2)
        st.handle_click(0.8, 0.8)
        st.size_points.append((0.5, 0.2))
        apply_calibration(cfg, st)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.yaml")
            Config.save(cfg, path)
            cfg2 = Config.load(path)

        c = next(x for x in cfg2.counters if x.id == "main_line")
        self.assertIsInstance(c, LineCounterConfig)
        self.assertEqual(c.a, (0.2, 0.2))
        self.assertEqual(c.b, (0.8, 0.8))
        self.assertIn((0.5, 0.2), cfg2.size_profile.control_points)
        self.assertTrue(cfg2.size_profile.enabled)
        # остальные блоки не потерялись
        self.assertEqual(cfg2.motion.method, "mog2")
        self.assertEqual(cfg2.tracker.type, "sort")


# ---------------------------------------------------------------------------
# headless: available() без исключений и без окна
# ---------------------------------------------------------------------------

class TestGuiPlayerHeadless(unittest.TestCase):

    def test_available_returns_bool_without_window(self):
        res = GuiPlayer.available()
        self.assertIsInstance(res, bool)

    def test_available_false_without_display_env(self):
        saved_disp = os.environ.pop("DISPLAY", None)
        saved_way = os.environ.pop("WAYLAND_DISPLAY", None)
        try:
            self.assertFalse(GuiPlayer.available())
        finally:
            if saved_disp is not None:
                os.environ["DISPLAY"] = saved_disp
            if saved_way is not None:
                os.environ["WAYLAND_DISPLAY"] = saved_way

    def test_unavailable_reason_is_clear(self):
        msg = GuiPlayer.unavailable_reason()
        self.assertTrue("GUI" in msg or "gui" in msg)


if __name__ == "__main__":
    unittest.main()
