"""Тесты GUI-overlay и чистых функций калибровки (задача 16).

ВСЕ тесты — без реального окна: GuiOverlay.draw работает на синтетических
кадрах, calibrate-логика — через CalibrationState/apply_calibration,
GuiPlayer.available() — чистая проверка env + cv2.getBuildInformation().
"""

from __future__ import annotations

import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from visio_people_counter.calibrate import (
    CalibrationState,
    apply_calibration,
    click_to_norm,
    counter_status_text,
    cycle_counter_id,
    draw_top_panel,
    hit_button,
    layout_buttons,
    delete_current_counter,
    load_counter_into_state,
    make_new_counter_id,
    size_digit_to_fraction,
    SIZE_POINT_HIT_RADIUS_PX,
    handle_size_click,
    remove_size_point,
    size_point_at_click,
    undo_last_size_point,
    TimeInputBuffer,
    line_length_px,
    line_length_norm,
    polygon_area_fraction,
    draw_all_counters,
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
        st.handle_click(0.42, 0.9)      # клик фиксирует ОБЕ координаты (важен и x, и y)
        st.set_size_height(3)           # → 15% высоты кадра
        self.assertEqual(st.size_points, [(0.42, 0.9, 0.15)])
        apply_calibration(cfg, st)
        self.assertTrue(cfg.size_profile.enabled)
        self.assertIn((0.42, 0.9, 0.15), cfg.size_profile.control_points)

    def test_size_click_y_is_stored(self):
        """Регрессия задачи 05: клик в size-режиме запоминает и y (2D-профиль)."""
        st = CalibrationState(counter_id="m")
        st.set_mode("size")
        st.handle_click(0.1, 0.2)
        self.assertEqual((st._size_x, st._size_y), (0.1, 0.2))
        st.set_size_height(5)           # → 25% высоты кадра
        self.assertEqual(st.size_points, [(0.1, 0.2, 0.25)])
        st.finish_size()
        self.assertIsNone(st._size_x)
        self.assertIsNone(st._size_y)

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


# ---------------------------------------------------------------------------
# calibrate: мульти-счётчики и панель кнопок (чистые функции, без окна)
# ---------------------------------------------------------------------------

class TestCycleCounterId(unittest.TestCase):

    def test_cycle_forward_and_back_wraps(self):
        ids = ["main_line", "line_1", "zone_1"]
        self.assertEqual(cycle_counter_id(ids, "main_line", 1), "line_1")
        self.assertEqual(cycle_counter_id(ids, "line_1", 1), "zone_1")
        self.assertEqual(cycle_counter_id(ids, "zone_1", 1), "main_line")   # цикл
        self.assertEqual(cycle_counter_id(ids, "line_1", -1), "main_line")
        self.assertEqual(cycle_counter_id(ids, "main_line", -1), "zone_1")   # цикл назад

    def test_single_counter_stays(self):
        self.assertEqual(cycle_counter_id(["only"], "only", 1), "only")
        self.assertEqual(cycle_counter_id(["only"], "only", -1), "only")

    def test_empty_list_raises(self):
        with self.assertRaises(ValueError):
            cycle_counter_id([], "x", 1)

    def test_bad_direction_raises(self):
        with self.assertRaises(ValueError):
            cycle_counter_id(["a"], "a", 0)

    def test_current_not_in_ids_new_counter(self):
        # новый несaved-счётчик: «вперёд» — к первому, «назад» — к последнему
        ids = ["line_1", "zone_9"]
        self.assertEqual(cycle_counter_id(ids, "line_77", 1), "line_1")
        self.assertEqual(cycle_counter_id(ids, "line_77", -1), "zone_9")


class TestMakeNewCounterId(unittest.TestCase):

    def test_free_number_not_only_by_prefix(self):
        # минимальный свободный n среди ВСЕХ id: line_2 (не line_3)
        self.assertEqual(make_new_counter_id(set(), "line"), "line_1")
        self.assertEqual(
            make_new_counter_id({"line_1", "zone_1", "line_3"}, "line"), "line_2")

    def test_sequential(self):
        taken = set()
        for expect in ("line_1", "line_2", "line_3"):
            self.assertEqual(make_new_counter_id(taken, "line"), expect)
            taken.add(expect)

    def test_zone_kind_and_collision_with_other_ids(self):
        self.assertEqual(make_new_counter_id(set(), "zone"), "zone_1")
        # id из другого семейства не мешает, но совпадение с полным id учитывается
        self.assertEqual(make_new_counter_id({"line_2"}, "line"), "line_1")

    def test_bad_kind_raises(self):
        with self.assertRaises(ValueError):
            make_new_counter_id(set(), "rect")


class TestLoadCounterIntoState(unittest.TestCase):

    def test_load_line_geometry(self):
        st = CalibrationState(counter_id="other")
        st.set_mode("zone")
        st.handle_click(0.5, 0.5)
        st.handle_click(0.6, 0.6)
        c = LineCounterConfig(id="m", a=(0.2, 0.3), b=(0.8, 0.9))
        load_counter_into_state(st, c)
        self.assertEqual(st.counter_id, "m")
        self.assertEqual(st.mode, "line")
        self.assertEqual(st.line_points, [(0.2, 0.3), (0.8, 0.9)])
        self.assertEqual(st.zone_points, [])     # другой тип очищен

    def test_load_zone_geometry(self):
        st = CalibrationState(counter_id="other")
        st.set_mode("line")
        st.handle_click(0.1, 0.1)
        poly = [(0.3, 0.3), (0.7, 0.3), (0.7, 0.7)]
        c = ZoneCounterConfig(id="z", polygon=poly)
        load_counter_into_state(st, c)
        self.assertEqual(st.counter_id, "z")
        self.assertEqual(st.mode, "zone")
        self.assertEqual(st.zone_points, poly)
        self.assertEqual(st.line_points, [])

    def test_unknown_counter_type_raises(self):
        st = CalibrationState()
        with self.assertRaises(ValueError):
            load_counter_into_state(st, object())


class TestCounterStatusText(unittest.TestCase):

    def test_no_counters_hint(self):
        t = counter_status_text("line_1", [])
        self.assertIn("+линия", t)
        self.assertIn("+зона", t)

    def test_position_and_kind(self):
        counters = [LineCounterConfig(id="a", a=(0.1, 0.1), b=(0.9, 0.9)),
                    ZoneCounterConfig(id="b", polygon=[(0.2, 0.2), (0.8, 0.2),
                                                       (0.8, 0.8)])]
        self.assertIn("a (линия, позиция 1 из 2)", counter_status_text("a", counters))
        self.assertIn("b (зона, позиция 2 из 2)", counter_status_text("b", counters))

    def test_new_counter_not_in_config(self):
        counters = [LineCounterConfig(id="a")]
        self.assertIn("новый", counter_status_text("line_9", counters))


class TestButtonLayoutAndHitTest(unittest.TestCase):

    def _buttons(self, **kw):
        return layout_buttons(kw.pop("mode", None), kw.pop("mask_on", False), **kw)

    def test_order_and_labels(self):
        btns = self._buttons(mode="line")
        self.assertEqual([b[0] for b in btns],
                         ["line", "zone", "size", "mask", "show_all",
                          "prev", "next", "new_line", "new_zone", "delete",
                          "undo_size", "time", "save"])
        labels = {b[0]: b[1] for b in btns}
        self.assertEqual(labels["line"], "линия")
        self.assertEqual(labels["show_all"], "все")
        self.assertEqual(labels["size"], "размер")
        self.assertEqual(labels["new_line"], "+линия")
        self.assertEqual(labels["delete"], "удалить")
        self.assertEqual(labels["undo_size"], "−точка")
        self.assertEqual(labels["time"], "время")
        self.assertEqual(labels["save"], "сохранить")

    def test_active_flags(self):
        btns = {b[0]: b for b in self._buttons(mode="zone", mask_on=True)}
        self.assertTrue(btns["zone"][2])
        self.assertTrue(btns["mask"][2])
        for name in ("line", "size", "show_all", "prev", "next",
                    "new_line", "new_zone", "delete", "undo_size", "time", "save"):
            self.assertFalse(btns[name][2], f"{name} должен быть неактивным")

    def test_show_all_button_active(self):
        btns = {b[0]: b for b in self._buttons(mode=None, show_all=True)}
        self.assertTrue(btns["show_all"][2])
        self.assertEqual(btns["show_all"][1], "все")

    def test_geometry_row_no_overlap(self):
        btns = self._buttons(x0=8, y0=34)
        for (n1, _l1, _a1, x01, y01, x11, y11), \
            (n2, _l2, _a2, x02, y02, x12, y12) in zip(btns, btns[1:]):
            self.assertLessEqual(x11, x02, f"кнопки {n1}/{n2} пересекаются")
        for _n, _l, _a, x0, y0, x1, y1 in btns:
            self.assertGreater(x1 - x0, 8)     # кнопка не нулевая по ширине
            self.assertEqual(y1 - y0, 32)      # 20px шрифт + 2*6 отступов
            self.assertEqual((y0, y1), (34, 66))
        self.assertEqual(btns[0][3], 8)        # первая кнопка начинается с x0=8

    def test_hit_center_and_edges(self):
        btns = self._buttons()
        b = next(b for b in btns if b[0] == "mask")
        _name, _l, _a, x0, y0, x1, y1 = b
        self.assertEqual(hit_button(btns, (x0 + x1) // 2, (y0 + y1) // 2), "mask")
        self.assertEqual(hit_button(btns, x0, y0), "mask")                 # левый-верхний угол
        self.assertIsNone(hit_button(btns, x1, y0))                        # правый край — мимо
        self.assertIsNone(hit_button(btns, x0, y1))                        # нижний край — мимо

    def test_hit_below_panel_is_miss(self):
        """Клик ниже панели (область линии/зоны) НЕ перехватывается кнопками."""
        btns = self._buttons()
        bottom = max(b[6] for b in btns)
        self.assertIsNone(hit_button(btns, 100, bottom + 50))
        self.assertIsNone(hit_button(btns, -1, 40))

    def test_hit_first_and_last(self):
        btns = self._buttons()
        first, last = btns[0], btns[-1]
        self.assertEqual(hit_button(btns, first[3] + 2, first[4] + 2), "line")
        self.assertEqual(hit_button(btns, last[5] - 2, last[6] - 2), "save")

    def test_draw_top_panel_paints_active_button_green(self):
        import numpy as np
        img = np.zeros((360, 900, 3), np.uint8)
        counters = [LineCounterConfig(id="m", a=(0.1, 0.1), b=(0.9, 0.9))]
        btns = draw_top_panel(img, "m", counters, mode="line", mask_on=False)
        self.assertEqual(len(btns), 13)   # с кнопками «все» и «−точка»
        b = next(b for b in btns if b[0] == "line")
        _n, _l, active, x0, y0, x1, y1 = b
        self.assertTrue(active)
        px = img[y0 + 3, (x0 + x1) // 2].astype(int)
        # активный фон (0,128,0): G≈128, B мало (текст белый не в этой точке — запас)
        self.assertGreater(px[1], 90)
        self.assertLess(px[0], 60)


class TestConfigRoundtrip(unittest.TestCase):
    """Config.save → load сохраняет всё (после калибровки)."""

    def test_save_load_roundtrip(self):
        cfg = Config.from_dict({})
        st = CalibrationState(counter_id="main_line")
        st.set_mode("line")
        st.handle_click(0.2, 0.2)
        st.handle_click(0.8, 0.8)
        st.size_points.append((0.5, 0.6, 0.2))
        apply_calibration(cfg, st)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.yaml")
            Config.save(cfg, path)
            cfg2 = Config.load(path)

        c = next(x for x in cfg2.counters if x.id == "main_line")
        self.assertIsInstance(c, LineCounterConfig)
        self.assertEqual(c.a, (0.2, 0.2))
        self.assertEqual(c.b, (0.8, 0.8))
        self.assertIn((0.5, 0.6, 0.2), cfg2.size_profile.control_points)
        self.assertTrue(cfg2.size_profile.enabled)
        # остальные блоки не потерялись
        self.assertEqual(cfg2.motion.method, "mog2")
        self.assertEqual(cfg2.tracker.type, "sort")


# ---------------------------------------------------------------------------
# headless: available() без исключений и без окна
# ---------------------------------------------------------------------------

class TestCounterKindGuard(unittest.TestCase):
    """Один id — один счётчик: нельзя нарисовать зону под id существующей линии."""

    def _cfg_with_line1(self):
        from visio_people_counter.config import Config, LineCounterConfig
        cfg = Config.default()
        cfg.counters = [LineCounterConfig(id="line_1", a=(0.1, 0.5), b=(0.9, 0.5))]
        return cfg

    def test_ensure_same_kind_no_change(self):
        from visio_people_counter.calibrate import CalibrationState, ensure_counter_kind
        cfg = self._cfg_with_line1()
        st = CalibrationState(counter_id="line_1")
        self.assertIsNone(ensure_counter_kind(cfg, st, "line"))
        self.assertEqual(st.counter_id, "line_1")

    def test_ensure_other_kind_creates_new_id(self):
        from visio_people_counter.calibrate import CalibrationState, ensure_counter_kind
        cfg = self._cfg_with_line1()
        st = CalibrationState(counter_id="line_1")
        new_id = ensure_counter_kind(cfg, st, "zone")
        self.assertEqual(new_id, "zone_1")
        self.assertEqual(st.counter_id, "zone_1")

    def test_ensure_bad_kind_raises(self):
        from visio_people_counter.calibrate import CalibrationState, ensure_counter_kind
        cfg = self._cfg_with_line1()
        with self.assertRaises(ValueError):
            ensure_counter_kind(cfg, CalibrationState(counter_id="line_1"), "size")

    def test_apply_does_not_duplicate_id(self):
        """Регрессия: зона, нарисованная под id линии, не создаёт дубль id."""
        import tempfile
        from pathlib import Path
        from visio_people_counter.calibrate import (
            CalibrationState, apply_calibration)
        from visio_people_counter.config import Config
        cfg = self._cfg_with_line1()
        st = CalibrationState(counter_id="line_1")
        st.set_mode("zone")
        st.zone_points = [(0.3, 0.2), (0.9, 0.5), (0.4, 0.8)]
        changed = apply_calibration(cfg, st)
        ids = [c.id for c in cfg.counters]
        self.assertEqual(len(ids), len(set(ids)), f"дубли id: {ids}, diff: {changed}")
        # roundtrip: конфиг сохранился и снова читается (Config.load проверяет дубли)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "c.yaml"
            Config.save(cfg, p)
            cfg2 = Config.load(p)
            self.assertEqual(len(cfg2.counters), 2)


class TestClampSeekTime(unittest.TestCase):
    """seek за конец файла → «конец минус cache_frames», иначе — без изменений."""

    def _clamp(self, *a):
        from visio_people_counter.calibrate import clamp_seek_time
        return clamp_seek_time(*a)

    def test_within_range_unchanged(self):
        self.assertEqual(self._clamp(10.0, 100.0, 20, 25.0), (10.0, False))

    def test_beyond_end_clamped_to_end_minus_cache(self):
        # 100 с − 20/25 с = 99.2
        self.assertEqual(self._clamp(200.0, 100.0, 20, 25.0), (99.2, True))

    def test_unknown_duration_or_fps_no_clamp(self):
        self.assertEqual(self._clamp(999.0, 0.0, 20, 25.0), (999.0, False))
        self.assertEqual(self._clamp(999.0, 100.0, 20, 0.0), (999.0, False))

    def test_short_video_clamps_to_zero(self):
        self.assertEqual(self._clamp(50.0, 0.3, 20, 25.0), (0.0, True))


class TestNotificationBackground(unittest.TestCase):
    def test_white_background_and_blue_text(self):
        import numpy as np
        from visio_people_counter.calibrate import draw_notification
        img = np.zeros((60, 200, 3), dtype=np.uint8)
        draw_notification(img, "тест", x=10, y=10, size_px=14)
        # подложка белая (в отступе, вне глифов текста)
        self.assertGreater(int(np.mean(img[7, 8])), 240)
        # текст нарисован: в области есть пиксели, заметно отличные от белого
        region = img[10:30, 10:80]
        nonwhite = np.any(region < 200, axis=2).sum()
        self.assertGreater(nonwhite, 5)

    def test_empty_or_none_safe(self):
        from visio_people_counter.calibrate import draw_notification
        draw_notification(None, "тест", 0, 0)  # не падает


class TestMessageTtl(unittest.TestCase):
    def test_filter_expired(self):
        from visio_people_counter.calibrate import filter_expired_messages
        msgs = [("живое", 100.0), ("истекшее", 50.0)]
        self.assertEqual(filter_expired_messages(msgs, now=60.0), [("живое", 100.0)])
        self.assertEqual(filter_expired_messages(msgs, now=40.0), msgs)

    def test_ttl_is_five_seconds(self):
        from visio_people_counter import calibrate
        self.assertGreaterEqual(calibrate.MESSAGE_TTL_SECONDS, 5.0)


class TestCalibrationSaveTarget(unittest.TestCase):
    """Путь сохранения calibrate: по умолчанию <имя_видео>.config.yaml рядом с видео."""

    def test_file_video_template(self):
        from visio_people_counter.calibrate import calibration_save_target
        p = calibration_save_target("file", "videos/demo.mp4", "config.yaml")
        self.assertEqual(p, Path("videos/demo.config.yaml"))

    def test_nested_path_and_weird_name(self):
        from visio_people_counter.calibrate import calibration_save_target
        p = calibration_save_target("file", "/a/b/моя камера (1).avi", "config.yaml")
        self.assertEqual(p, Path("/a/b/моя камера (1).config.yaml"))

    def test_hls_fallback_to_config(self):
        from visio_people_counter.calibrate import calibration_save_target
        p = calibration_save_target("hls", "https://cam/x/stream.m3u8", "my/conf.yaml")
        self.assertEqual(p, Path("my/conf.yaml"))


class TestApplyQtEnv(unittest.TestCase):
    """apply_qt_env: встроенный Qt из колеса opencv должен видеть системный gtk3-плагин."""

    def test_sets_plugin_path_and_theme_idempotent(self):
        import visio_people_counter.gui as gui
        saved = {k: os.environ.get(k) for k in ("QT_PLUGIN_PATH", "QT_QPA_PLATFORMTHEME")}
        try:
            for k in saved:
                os.environ.pop(k, None)
            gui.apply_qt_env()
            self.assertIn(gui._SYSTEM_QT_PLUGIN_DIR, os.environ["QT_PLUGIN_PATH"])
            self.assertEqual(os.environ["QT_QPA_PLATFORMTHEME"], "gtk3")
            before = os.environ["QT_PLUGIN_PATH"]
            gui.apply_qt_env()  # повторный вызов — без дублей
            self.assertEqual(os.environ["QT_PLUGIN_PATH"], before)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


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


class TestTextOverlay(unittest.TestCase):
    """text_overlay: кириллица через PIL/TTF, фолбэк — ASCII без ошибок."""

    def test_cyrillic_renders_pixels(self):
        import numpy as np
        from visio_people_counter.text_overlay import put_text
        frame = np.zeros((120, 512, 3), np.uint8)
        put_text(frame, "СЧЁТ: вх=3 вых=1", (10, 10), size_px=20,
                 color=(0, 255, 0))
        green = int(((frame[..., 1] > 60) & (frame[..., 0] < 60)
                     & (frame[..., 2] < 60)).sum())
        self.assertGreater(green, 300, "кириллический текст не отрисовался")

    def test_text_color_bgr_respected(self):
        import numpy as np
        from visio_people_counter.text_overlay import put_text
        frame = np.zeros((60, 200, 3), np.uint8)
        put_text(frame, "тест", (5, 5), size_px=18, color=(255, 0, 0))  # BGR: красный
        red_px = int(((frame[..., 2] > 60) & (frame[..., 0] < 60)).sum())
        self.assertGreater(red_px, 30)

    def test_text_outside_frame_does_not_crash(self):
        import numpy as np
        from visio_people_counter.text_overlay import put_text
        frame = np.zeros((40, 40, 3), np.uint8)
        for org in ((-50, -50), (39, 39), (1000, 1000)):
            put_text(frame, "за краем", org, size_px=16)  # без исключений

    def test_fallback_path_ascii(self):
        import numpy as np
        from visio_people_counter import text_overlay as to
        frame = np.zeros((60, 200, 3), np.uint8)
        to._fallback_put_text(frame, "линия in=1", (5, 5), size_px=18,
                              color_bgr=(255, 255, 255), shadow=True)
        self.assertGreater(int((frame.sum(axis=2) > 60).sum()), 30)

    def test_counter_label_with_cyrillic_id_renders(self):
        # id счётчика с русскими буквами не должен ломать/прятать подпись
        import numpy as np
        from visio_people_counter.config import LineCounterConfig
        from visio_people_counter.line_counter import LineCounter
        lc = LineCounter(LineCounterConfig(id="вход", a=(0.25, 0.35), b=(0.75, 0.85)),
                         w=640, h=360)
        frame = np.zeros((360, 640, 3), np.uint8)
        lc.draw(frame)  # не бросает исключение
        self.assertGreater(int((frame.sum(axis=2) > 60).sum()), 100)

    def test_calibrate_hints_drawn_with_cyrillic(self):
        import numpy as np
        from visio_people_counter.calibrate import _HINTS, _draw_calibration
        state = type("S", (), {"mode": None, "line_points": [], "zone_points": [],
                               "size_points": []})()
        frame = np.zeros((360, 640, 3), np.uint8)
        out = _draw_calibration(frame, state, mask_on=False, mask=None,
                                w=640, h=360)
        # подсказки (русские) отрисованы белым
        white = int(((out[..., 0] > 150) & (out[..., 1] > 150)
                     & (out[..., 2] > 150)).sum())
        self.assertGreater(white, 200)
        self.assertIn("[l]иния", _HINTS[None])


# ---------------------------------------------------------------------------
# calibrate: буфер ввода времени для seek (чистый класс, без cv2) — задача 03
# ---------------------------------------------------------------------------

class TestTimeInputBuffer(unittest.TestCase):
    def test_empty_value_is_none(self):
        buf = TimeInputBuffer()
        self.assertIsNone(buf.value())
        self.assertEqual(buf.digits, [])

    def test_feed_digits_appends_to_end(self):
        buf = TimeInputBuffer()
        for d in (1, 2, 3):
            buf.feed_digit(d)
        self.assertEqual(buf.digits, [1, 2, 3])
        self.assertEqual(buf.value(), 123)

    def test_leading_zeros_kept_in_value(self):
        buf = TimeInputBuffer()
        for d in (0, 5):
            buf.feed_digit(d)
        self.assertEqual(buf.value(), 5)

    def test_limit_six_digits_ignored_beyond(self):
        buf = TimeInputBuffer()
        for d in (1, 2, 3, 4, 5, 6, 7, 8):
            buf.feed_digit(d)
        self.assertEqual(buf.digits, [1, 2, 3, 4, 5, 6])
        self.assertEqual(buf.value(), 123456)

    def test_backspace_removes_last_only(self):
        buf = TimeInputBuffer()
        for d in (9, 0, 9):
            buf.feed_digit(d)
        buf.backspace()
        self.assertEqual(buf.digits, [9, 0])
        self.assertEqual(buf.value(), 90)

    def test_backspace_on_empty_is_noop(self):
        buf = TimeInputBuffer()
        buf.backspace()
        self.assertIsNone(buf.value())

    def test_reset_clears(self):
        buf = TimeInputBuffer()
        for d in (1, 2):
            buf.feed_digit(d)
        buf.reset()
        self.assertEqual(buf.digits, [])
        self.assertIsNone(buf.value())
        # после reset можно набирать заново
        buf.feed_digit(7)
        self.assertEqual(buf.value(), 7)

    def test_feed_digit_rejects_non_digits(self):
        buf = TimeInputBuffer()
        for bad in (-1, 10, 2.5, "3", True):
            with self.assertRaises(ValueError):
                buf.feed_digit(bad)
        self.assertIsNone(buf.value())


# ---------------------------------------------------------------------------
# calibrate: CLI --cache-frames и параметр run_calibration — задача 03
# ---------------------------------------------------------------------------

class TestCliCacheFrames(unittest.TestCase):
    def _parse(self, extra: list[str]):
        from visio_people_counter.__main__ import build_parser
        return build_parser().parse_args(["calibrate"] + extra)

    def test_default_is_20(self):
        self.assertEqual(self._parse([]).cache_frames, 100)

    def test_valid_value(self):
        self.assertEqual(self._parse(["--cache-frames", "5"]).cache_frames, 5)

    def test_less_than_one_is_argparse_error(self):
        for bad in ("0", "-3"):
            with self.assertRaises(SystemExit):
                self._parse(["--cache-frames", bad])

    def test_non_int_is_argparse_error(self):
        with self.assertRaises(SystemExit):
            self._parse(["--cache-frames", "abc"])

    def test_run_calibration_rejects_bad_cache_frames_without_gui(self):
        from visio_people_counter.calibrate import run_calibration
        # проверка аргумента — до создания окна/источника (без GUI и cv2-окна)
        with self.assertRaises(ValueError):
            run_calibration("config.yaml", video="x.mp4", cache_frames=0)


# ---------------------------------------------------------------------------
# calibrate: удаление текущего счётчика (чистая функция, без окна)
# ---------------------------------------------------------------------------

class TestDeleteCurrentCounter(unittest.TestCase):

    def test_delete_existing_line_counter(self):
        cfg = Config.from_dict({"counters": [
            {"id": "lineA", "type": "line", "a": [0.1, 0.1], "b": [0.9, 0.9]},
            {"id": "zoneB", "type": "zone",
             "polygon": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8]]}]})
        st = CalibrationState(counter_id="lineA")
        load_counter_into_state(st, cfg.counters[0])   # как переключение [<] в окне
        msg = delete_current_counter(cfg, st)
        self.assertEqual(msg, "счётчик lineA удалён")
        self.assertEqual([c.id for c in cfg.counters], ["zoneB"])
        # state переключён на соседа с его геометрией (направление «назад»)
        self.assertEqual(st.counter_id, "zoneB")
        self.assertEqual(st.mode, "zone")
        self.assertEqual(st.line_points, [])           # точки удалённого очищены

    def test_delete_existing_zone_counter(self):
        cfg = Config.from_dict({"counters": [
            {"id": "lineA", "type": "line", "a": [0.1, 0.1], "b": [0.9, 0.9]},
            {"id": "zoneB", "type": "zone",
             "polygon": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8]]}]})
        st = CalibrationState(counter_id="zoneB")
        load_counter_into_state(st, cfg.counters[1])
        msg = delete_current_counter(cfg, st)
        self.assertEqual(msg, "счётчик zoneB удалён")
        self.assertEqual([c.id for c in cfg.counters], ["lineA"])
        self.assertEqual(st.counter_id, "lineA")
        self.assertEqual(st.mode, "line")
        self.assertEqual(st.zone_points, [])           # точки удалённой зоны очищены

    def test_delete_counter_not_in_cfg(self):
        """Неприменённый «новый» счётчик: ничего не падает, точки в state очищены."""
        cfg = Config.from_dict({"counters": [
            {"id": "lineA", "type": "line", "a": [0.1, 0.1], "b": [0.9, 0.9]}]})
        st = CalibrationState(counter_id="line_9")
        st.set_mode("line")
        st.handle_click(0.3, 0.4)
        msg = delete_current_counter(cfg, st)
        self.assertIn("line_9", msg)
        self.assertEqual([c.id for c in cfg.counters], ["lineA"])  # cfg не менялся
        # переключились на соседа: в state — ЕГО геометрия, а не набросок line_9
        self.assertEqual(st.counter_id, "lineA")
        self.assertEqual(st.line_points, [(0.1, 0.1), (0.9, 0.9)])

    def test_apply_calibration_does_not_resurrect_deleted(self):
        """Регрессия: после удаления сохранение НЕ пересоздаёт счётчик из точек state.

        Единственный line-счётчик, 2 его точки загружены в state (как при правке).
        Если бы удаление не очистило line_points — apply_calibration добавил бы
        «lineA» заново; после очистки — diff пуст и cfg остаётся пустым.
        """
        cfg = Config.from_dict({"counters": [
            {"id": "lineA", "type": "line", "a": [0.1, 0.1], "b": [0.9, 0.9]}]})
        st = CalibrationState(counter_id="lineA")
        load_counter_into_state(st, cfg.counters[0])   # 2 точки линии в state
        self.assertEqual(len(st.line_points), 2)
        delete_current_counter(cfg, st)
        self.assertEqual(st.line_points, [])           # точки удалённого очищены
        changed = apply_calibration(cfg, st)           # новый pending пуст — нечего применить
        self.assertEqual(changed, [])
        self.assertNotIn("lineA", [c.id for c in cfg.counters],
                         "apply_calibration «воскресил» удалённый счётчик")

    def test_delete_last_counter_creates_new_pending_of_same_kind(self):
        cfg = Config.from_dict({"counters": [
            {"id": "lineA", "type": "line", "a": [0.1, 0.1], "b": [0.9, 0.9]}]})
        st = CalibrationState(counter_id="lineA")
        load_counter_into_state(st, cfg.counters[0])
        delete_current_counter(cfg, st)
        self.assertEqual(cfg.counters, [])
        # автоматически создан новый pending той же природы (линия → line_N)
        self.assertEqual(st.counter_id, "line_1")
        self.assertEqual(st.mode, "line")
        self.assertEqual(st.line_points, [])

    def test_delete_last_zone_counter_creates_new_zone(self):
        cfg = Config.from_dict({"counters": [
            {"id": "zoneA", "type": "zone",
             "polygon": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8]]}]})
        st = CalibrationState(counter_id="zoneA")
        load_counter_into_state(st, cfg.counters[0])
        delete_current_counter(cfg, st)
        self.assertEqual(st.counter_id, "zone_1")
        self.assertEqual(st.mode, "zone")

    def test_delete_not_in_cfg_without_mode_clears_both_point_sets(self):
        """Счётчик не в cfg и mode=None — очищаются оба набора точек."""
        cfg = Config.from_dict({})
        st = CalibrationState(counter_id="lineA")
        st.line_points = [(0.1, 0.1), (0.9, 0.9)]
        st.zone_points = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8)]
        msg = delete_current_counter(cfg, st)
        self.assertEqual(msg, "счётчик lineA удалён")
        self.assertEqual(st.line_points, [])
        self.assertEqual(st.zone_points, [])
        # счётчиков не осталось → создан новый по умолчанию (линия)
        self.assertEqual(st.counter_id, "line_1")


# ---------------------------------------------------------------------------
# calibrate: режим «все» — все счётчики с размерами (чистые функции, без окна)
# ---------------------------------------------------------------------------

class TestLineLength(unittest.TestCase):
    def test_norm_diagonal(self):
        # (0,0)-(1,1): нормализованная длина = sqrt(1+1), БЕЗ учёта aspect ratio
        self.assertAlmostEqual(line_length_norm((0.0, 0.0), (1.0, 1.0)), math.sqrt(2))

    def test_norm_horizontal(self):
        self.assertAlmostEqual(line_length_norm((0.25, 0.35), (0.75, 0.85)),
                               math.hypot(0.5, 0.5), places=10)

    def test_px_square_frame(self):
        # (0,0)-(1,1) на 100x100: sqrt((1*100)² + (1*100)²) = 100*sqrt2
        self.assertAlmostEqual(line_length_px((0.0, 0.0), (1.0, 1.0), 100, 100),
                               100 * math.sqrt(2))

    def test_px_asymmetric_frame_200x100(self):
        # асимметричный кадр: dx масштабируется по w=200, dy — по h=100
        self.assertAlmostEqual(line_length_px((0.0, 0.0), (1.0, 1.0), 200, 100),
                               math.hypot(200.0, 100.0))
        # горизонтальная половина кадра: px = dx * w
        self.assertAlmostEqual(line_length_px((0.25, 0.5), (0.75, 0.5), W, H), 240.0)

    def test_px_zero_length(self):
        self.assertEqual(line_length_px((0.5, 0.5), (0.5, 0.5), W, H), 0.0)


class TestPolygonAreaFraction(unittest.TestCase):
    SQUARE = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)]

    def test_quarter_square(self):
        self.assertAlmostEqual(polygon_area_fraction(self.SQUARE, W, H), 0.25,
                               delta=0.01)
        # на другом размере кадра — та же доля
        self.assertAlmostEqual(polygon_area_fraction(self.SQUARE, 640, 360), 0.25,
                               delta=0.01)

    def test_empty_and_degenerate(self):
        self.assertEqual(polygon_area_fraction([], W, H), 0.0)
        self.assertEqual(polygon_area_fraction([(0.0, 0.0), (1.0, 1.0)], W, H),
                         0.0)
        # коллинеарные точки — нулевая площадь
        collinear = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]
        self.assertLess(polygon_area_fraction(collinear, W, H), 1e-6)

    def test_bad_frame_size(self):
        self.assertEqual(polygon_area_fraction(self.SQUARE, 0, 0), 0.0)


class TestDrawAllCounters(unittest.TestCase):
    COUNTERS = [
        LineCounterConfig(id="line_1", a=(0.2, 0.3), b=(0.8, 0.7)),
        ZoneCounterConfig(id="zone_1",
                          polygon=[(0.25, 0.25), (0.75, 0.25),
                                   (0.75, 0.75), (0.25, 0.75)]),
    ]

    def test_paints_lines_zones_and_labels(self):
        img = np.zeros((H, W, 3), np.uint8)
        draw_all_counters(img, self.COUNTERS, W, H)
        self.assertGreater(int((img.sum(axis=2) > 60).sum()), 100,
                           "линии/зоны/метки не нарисованы")

    def test_empty_list_no_change(self):
        img = np.zeros((H, W, 3), np.uint8)
        before = img.copy()
        draw_all_counters(img, [], W, H)          # не падает, кадр не тронут
        np.testing.assert_array_equal(img, before)

    def test_highlight_does_not_break(self):
        for hid in ("zone_1", "line_1", "no_such_id"):
            img = np.zeros((H, W, 3), np.uint8)
            draw_all_counters(img, self.COUNTERS, W, H, highlight_id=hid)
            self.assertGreater(int((img.sum(axis=2) > 60).sum()), 100)

    def test_zone_without_polygon_skipped(self):
        img = np.zeros((H, W, 3), np.uint8)
        counters = [ZoneCounterConfig(id="empty"),   # polygon=[] — не падает
                    LineCounterConfig(id="ok", a=(0.1, 0.1), b=(0.9, 0.9))]
        draw_all_counters(img, counters, W, H)
        self.assertGreater(int((img.sum(axis=2) > 60).sum()), 50)

    def test_none_img_safe(self):
        draw_all_counters(None, self.COUNTERS, W, H)   # без исключений


# ---------------------------------------------------------------------------
# calibrate: удаление отдельных/последних size-точек (чистые функции, без окна)
# ---------------------------------------------------------------------------

class TestSizePointHitAndDelete(unittest.TestCase):
    """Клик по существующей size-точке → удаление; [b]/«−точка» → отмена последней."""

    def _state(self, points):
        st = CalibrationState(counter_id="m")
        st.set_mode("size")
        st.size_points = [tuple(p) for p in points]
        return st

    # --- hit-тест: size_point_at_click ---

    def test_hit_exact_on_point(self):
        # точки (0.25, 0.5) и (0.75, 0.3) на кадре 480x320 → px (120,160) и (360,96)
        pts = [(0.25, 0.5, 0.2), (0.75, 0.3, 0.3)]
        self.assertEqual(size_point_at_click(pts, 120, 160, W, H), 0)
        self.assertEqual(size_point_at_click(pts, 360, 96, W, H), 1)

    def test_hit_within_radius(self):
        pts = [(0.25, 0.5, 0.2)]   # px (120, 160)
        self.assertEqual(size_point_at_click(pts, 130, 170, W, H), 0)    # +10/+10 px
        self.assertEqual(size_point_at_click(pts, 105, 145, W, H), 0)    # ровно −15/−15 px
        # x в радиусе, но y вне (Δy=40 > 15) — мимо: условие И по обеим осям
        self.assertIsNone(size_point_at_click(pts, 130, 200, W, H))

    def test_miss_outside_radius(self):
        pts = [(0.25, 0.5, 0.2)]
        self.assertIsNone(size_point_at_click(pts, 140, 160, W, H))      # +20 px > 15
        self.assertIsNone(size_point_at_click(pts, 120, 300, W, H))      # далеко по y

    def test_wide_frame_same_pixel_radius(self):
        """На широком кадре тот же ПИКСЕЛЬНЫЙ радиус работает в пикселях."""
        pts = [(0.5, 0.5, 0.2)]                       # центр 1920x1080 → px (960, 540)
        self.assertEqual(size_point_at_click(pts, 960 + 15, 540, 1920, 1080), 0)
        self.assertEqual(size_point_at_click(pts, 960, 540 - 15, 1920, 1080), 0)
        self.assertIsNone(size_point_at_click(pts, 960 + 16, 540, 1920, 1080))

    def test_custom_radius_and_defaults(self):
        pts = [(0.25, 0.5, 0.2)]                      # px (120, 160)
        self.assertEqual(size_point_at_click(pts, 130, 160, W, H,
                                             radius_px=30), 0)
        self.assertIsNone(size_point_at_click(pts, 128, 160, W, H,
                                              radius_px=5))
        self.assertEqual(SIZE_POINT_HIT_RADIUS_PX, 15)

    def test_empty_and_bad_frame_size(self):
        self.assertIsNone(size_point_at_click([], 120, 160, W, H))
        self.assertIsNone(size_point_at_click([(0.5, 0.5, 0.2)], 10, 10, 0, 0))

    # --- удаление по индексу / отмена последней ---

    def test_remove_middle_point_keeps_others(self):
        st = self._state([(0.1, 0.2, 0.1), (0.4, 0.5, 0.2), (0.8, 0.9, 0.3)])
        pt = remove_size_point(st, 1)
        self.assertEqual(pt, (0.4, 0.5, 0.2))
        self.assertEqual(st.size_points, [(0.1, 0.2, 0.1), (0.8, 0.9, 0.3)])

    def test_remove_out_of_range_returns_none(self):
        st = self._state([(0.1, 0.2, 0.1)])
        self.assertIsNone(remove_size_point(st, 5))
        self.assertIsNone(remove_size_point(CalibrationState(counter_id="m"), -1))
        self.assertEqual(st.size_points, [(0.1, 0.2, 0.1)])   # state не тронут

    def test_undo_last_removes_last_only(self):
        st = self._state([(0.1, 0.2, 0.1), (0.4, 0.5, 0.2)])
        pt = undo_last_size_point(st)
        self.assertEqual(pt, (0.4, 0.5, 0.2))
        self.assertEqual(st.size_points, [(0.1, 0.2, 0.1)])   # другие на месте

    def test_undo_on_empty_list_no_crash(self):
        st = CalibrationState(counter_id="m")
        st.set_mode("size")
        self.assertIsNone(undo_last_size_point(st))
        self.assertEqual(st.size_points, [])

    # --- интеграция: клик в size-режиме (handle_size_click) ---

    def test_click_on_existing_point_deletes_it(self):
        st = self._state([(0.25, 0.5, 0.2), (0.7, 0.4, 0.3)])
        removed = handle_size_click(st, 126, 160, W, H)   # в радиусе первой точки
        self.assertEqual(removed, (0.25, 0.5, 0.2))
        self.assertEqual(st.size_points, [(0.7, 0.4, 0.3)])
        self.assertIsNone(st._size_x)                     # новая точка НЕ запомнена
        self.assertIsNone(st._size_y)

    def test_click_miss_adds_pending_point(self):
        st = self._state([(0.25, 0.5, 0.2)])
        removed = handle_size_click(st, 400, 100, W, H)   # далеко от существующей
        self.assertIsNone(removed)
        self.assertEqual(st.size_points, [(0.25, 0.5, 0.2)])
        # клик запомнен как нормализованная ожидающая-цифру точка (округление click_to_norm)
        self.assertEqual((st._size_x, st._size_y), (round(400 / W, 4), round(100 / H, 4)))

    def test_click_in_other_mode_does_not_delete_size_points(self):
        """Hit-логика size активна только в режиме «размер»."""
        st = CalibrationState(counter_id="m")
        st.set_mode("line")
        st.size_points = [(0.25, 0.5, 0.2)]
        removed = handle_size_click(st, 120, 160, W, H)   # прямо «на» size-точке
        self.assertIsNone(removed)
        self.assertEqual(st.size_points, [(0.25, 0.5, 0.2)])   # не удалена
        self.assertEqual(st.line_points, [(0.25, 0.5)])        # клик ушёл в линию

    def test_undo_size_button_in_panel_layout(self):
        btns = layout_buttons("size", False)
        names = [b[0] for b in btns]
        self.assertIn("undo_size", names)
        # после «удалить», до «время»/«сохранить» — раскладка не сломана
        self.assertLess(names.index("delete"), names.index("undo_size"))
        self.assertLess(names.index("undo_size"), names.index("time"))
        self.assertEqual({b[0]: b[1] for b in btns}["undo_size"], "−точка")
        # панель не пересекается и помещается в кадр (авто-сужение работает)
        img = np.zeros((360, 480, 3), np.uint8)
        out = draw_top_panel(img, "m", [], mode="size", mask_on=False)
        self.assertEqual(len(out), 13)


if __name__ == "__main__":
    unittest.main()
