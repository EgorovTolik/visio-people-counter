"""Тесты config.py: загрузка example-конфига, валидация битых значений."""

import sys
import unittest
from pathlib import Path

# корень проекта — родитель каталога tests/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from visio_people_counter.config import (  # noqa: E402
    Config, ConfigError, LineCounterConfig, ZoneCounterConfig,
)

EXAMPLE = ROOT / "config.example.yaml"


class TestConfigLoad(unittest.TestCase):
    def test_example_config_loads(self):
        cfg = Config.load(EXAMPLE)
        self.assertEqual(cfg.video.type, "file")
        self.assertEqual(cfg.video.path, "videos/demo.mp4")
        self.assertFalse(cfg.video.loop_file)
        self.assertEqual(cfg.video.hls.reconnect_attempts, 0)
        self.assertEqual(cfg.video.hls.reconnect_backoff_s, 5.0)
        self.assertEqual(cfg.video.hls.bad_read_threshold, 10)
        self.assertEqual(cfg.processing.effective_fps, 0.0)
        self.assertEqual(cfg.processing.max_width, 0)
        self.assertIsNone(cfg.processing.frame_start)
        self.assertIsNone(cfg.processing.frame_end)
        self.assertEqual(cfg.motion.method, "mog2")
        self.assertEqual(cfg.motion.morph_open, (3, 3))
        self.assertEqual(cfg.motion.morph_close, (9, 15))
        self.assertEqual(cfg.objects.aspect_ratio_range, (0.2, 2.5))
        self.assertFalse(cfg.size_profile.enabled)
        self.assertEqual(cfg.tracker.type, "sort")
        self.assertEqual(len(cfg.counters), 2)
        line = cfg.counters[0]
        self.assertIsInstance(line, LineCounterConfig)
        self.assertEqual(line.id, "main_line")
        self.assertEqual(line.a, (0.25, 0.35))
        self.assertEqual(line.b, (0.75, 0.85))
        zone = cfg.counters[1]
        self.assertIsInstance(zone, ZoneCounterConfig)
        self.assertEqual(len(zone.polygon), 4)
        self.assertEqual(cfg.output.events_jsonl, "results/events.jsonl")
        self.assertTrue(cfg.debug.show_bboxes)

    def test_minimal_config_passes(self):
        # конфиг без опциональных блоков (size_profile.enabled=false и т.п.) проходит
        cfg = Config.from_dict({
            "video": {"type": "file", "path": "x.mp4"},
            "counters": [
                {"id": "l1", "type": "line", "a": [0.2, 0.3], "b": [0.8, 0.9]},
            ],
        })
        self.assertFalse(cfg.size_profile.enabled)
        self.assertEqual(cfg.size_profile.control_points, [])
        self.assertEqual(cfg.motion.method, "mog2")
        self.assertEqual(cfg.video.hls.reconnect_attempts, 0)
        self.assertEqual(len(cfg.counters), 1)

    def test_empty_config_passes_with_defaults(self):
        cfg = Config.from_dict({})
        self.assertEqual(cfg.video.type, "file")
        self.assertEqual(cfg.counters, [])

    def test_save_roundtrip(self):
        import tempfile
        cfg = Config.load(EXAMPLE)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "rt.yaml"
            Config.save(cfg, p)
            cfg2 = Config.load(p)
        self.assertEqual(cfg.to_dict(), cfg2.to_dict())


class TestConfigValidation(unittest.TestCase):
    def _load_broken(self, mutate) -> None:
        """Взять example-конфиг, применить mutate(dict) и загрузить.

        Ожидаемый исход — ConfigError (проверяется самим тестом).
        """
        import copy, yaml
        raw = copy.deepcopy(yaml.safe_load(EXAMPLE.read_text(encoding="utf-8")))
        mutate(raw)
        return Config.from_dict(raw)

    def test_coordinate_out_of_range(self):
        def m(d):
            d["counters"][0]["a"] = [1.5, 0.35]
        try:
            self._load_broken(m)
        except ConfigError as e:
            self.assertIn("counters[0].a", str(e))
            self.assertIn("0..1", str(e))

    def test_even_morph_kernel(self):
        def m(d):
            d["motion"]["morph_open"] = [4, 4]
        try:
            self._load_broken(m)
        except ConfigError as e:
            self.assertIn("motion.morph_open", str(e))

    def test_zero_morph_kernel(self):
        def m(d):
            d["motion"]["morph_close"] = [0, 9]
        with self.assertRaises(ConfigError):
            self._load_broken(m)

    def test_unknown_motion_method(self):
        def m(d):
            d["motion"]["method"] = "optical_flow"
        try:
            self._load_broken(m)
        except ConfigError as e:
            self.assertIn("motion.method", str(e))
            self.assertIn("mog2", str(e))

    def test_unknown_video_type(self):
        def m(d):
            d["video"]["type"] = "rtsp"
        with self.assertRaises(ConfigError) as cm:
            self._load_broken(m)
        self.assertIn("video.type", str(cm.exception))

    def test_bad_counter_type(self):
        def m(d):
            d["counters"][0]["type"] = "circle"
        with self.assertRaises(ConfigError) as cm:
            self._load_broken(m)
        self.assertIn("counters[0].type", str(cm.exception))

    def test_zone_polygon_too_few_points(self):
        def m(d):
            d["counters"][1]["polygon"] = [[0.1, 0.1], [0.2, 0.2]]
        with self.assertRaises(ConfigError) as cm:
            self._load_broken(m)
        self.assertIn("polygon", str(cm.exception))

    def test_wrong_type_value(self):
        def m(d):
            d["motion"]["history"] = "пятьсот"
        try:
            self._load_broken(m)
        except ConfigError as e:
            self.assertIn("motion.history", str(e))

    def test_unknown_key_rejected(self):
        def m(d):
            d["video"]["typo_path"] = "x.mp4"
        try:
            self._load_broken(m)
        except ConfigError as e:
            self.assertIn("typo_path", str(e))

    def test_missing_file(self):
        with self.assertRaises(ConfigError) as cm:
            Config.load("/nonexistent/path/config.yaml")
        self.assertIn("не найден", str(cm.exception))


class TestSizeProfilePoints(unittest.TestCase):
    """size_profile.control_points: тройки [x, y, h] + legacy-миграция пар (задача 05)."""

    def test_triples_accepted(self):
        sp = Config.from_dict({"size_profile": {
            "enabled": True,
            "control_points": [[0.1, 0.9, 0.45], [0.8, 0.3, 0.08]]}}).size_profile
        self.assertEqual(sp.control_points, [(0.1, 0.9, 0.45), (0.8, 0.3, 0.08)])

    def test_legacy_pairs_migrated_to_y_half(self):
        sp = Config.from_dict({"size_profile": {
            "enabled": True,
            "control_points": [[0.1, 0.45], [0.9, 0.08]]}}).size_profile
        self.assertEqual(sp.control_points, [(0.1, 0.5, 0.45), (0.9, 0.5, 0.08)])

    def test_bad_points_rejected(self):
        for bad in ([0.1], [1.5, 0.5, 0.2], [0.1, 1.2, 0.3],
                    [0.1, 0.5, 0.0], [0.1, 0.5, 1.1], "x"):
            with self.assertRaises(ConfigError):
                Config.from_dict({"size_profile": {"control_points": [bad]}})

    def test_save_writes_triples_and_roundtrips(self):
        import tempfile
        import yaml
        cfg = Config.from_dict({"size_profile": {
            "enabled": True,
            "control_points": [[0.1, 0.9, 0.4], [0.8, 0.3, 0.1]]}})
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "c.yaml"
            Config.save(cfg, p)
            raw = yaml.safe_load(p.read_text(encoding="utf-8"))
            for pt in raw["size_profile"]["control_points"]:
                self.assertEqual(len(pt), 3, "в YAML должны писаться только тройки")
            cfg2 = Config.load(p)
        self.assertEqual(cfg2.size_profile.control_points,
                         [(0.1, 0.9, 0.4), (0.8, 0.3, 0.1)])


class TestConfigDefault(unittest.TestCase):
    """Config.default() — первый запуск calibrate без файла."""

    def test_default_is_valid_and_has_line_counter(self):
        cfg = Config.default()
        self.assertEqual(len(cfg.counters), 1)
        c = cfg.counters[0]
        self.assertEqual(c.id, "main_line")
        self.assertEqual(c.type, "line")
        for x, y in (c.a, c.b):
            self.assertTrue(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0)

    def test_default_save_load_roundtrip(self):
        import tempfile
        cfg = Config.default()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "config.yaml"
            Config.save(cfg, p)
            loaded = Config.load(p)
        self.assertEqual(loaded.to_dict(), cfg.to_dict())


class TestProcessingFrameRange(unittest.TestCase):
    """processing.frame_start/frame_end: null/null, только start/end, ошибки."""

    @staticmethod
    def _load(frame_start="__omit__", frame_end="__omit__") -> Config:
        processing: dict = {}
        if frame_start != "__omit__":
            processing["frame_start"] = frame_start
        if frame_end != "__omit__":
            processing["frame_end"] = frame_end
        return Config.from_dict({
            "video": {"type": "file", "path": "x.mp4"},
            "processing": processing,
        })

    def test_both_null_default(self):
        cfg = self._load(None, None)
        self.assertIsNone(cfg.processing.frame_start)
        self.assertIsNone(cfg.processing.frame_end)

    def test_missing_keys_are_none(self):
        cfg = self._load()
        self.assertIsNone(cfg.processing.frame_start)
        self.assertIsNone(cfg.processing.frame_end)

    def test_only_start(self):
        cfg = self._load(100, None)
        self.assertEqual(cfg.processing.frame_start, 100)
        self.assertIsNone(cfg.processing.frame_end)

    def test_only_end_inclusive_bound_kept(self):
        cfg = self._load(None, 250)
        self.assertIsNone(cfg.processing.frame_start)
        self.assertEqual(cfg.processing.frame_end, 250)

    def test_both_equal_ok(self):
        cfg = self._load(42, 42)
        self.assertEqual((cfg.processing.frame_start, cfg.processing.frame_end), (42, 42))

    def test_zero_is_valid(self):
        cfg = self._load(0, 0)
        self.assertEqual(cfg.processing.frame_start, 0)
        self.assertEqual(cfg.processing.frame_end, 0)

    def test_start_greater_than_end_raises(self):
        with self.assertRaises(ConfigError) as ctx:
            self._load(250, 100)
        self.assertIn("frame_start", str(ctx.exception))

    def test_negative_raises(self):
        for kw in ({"frame_start": -1}, {"frame_end": -5}):
            with self.assertRaises(ConfigError) as ctx:
                self._load(**kw)
            self.assertIn(">= 0", str(ctx.exception))

    def test_non_int_raises(self):
        for bad in (1.5, "100", True, [100]):
            with self.assertRaises(ConfigError) as ctx:
                self._load(bad, None)
            self.assertIn("processing.frame_start", str(ctx.exception))

    def test_unknown_key_still_rejected(self):
        with self.assertRaises(ConfigError) as ctx:
            Config.from_dict({
                "video": {"type": "file", "path": "x.mp4"},
                "processing": {"frame_startt": 100},
            })
        self.assertIn("неизвестные ключи", str(ctx.exception))

    def test_save_load_roundtrip_with_range(self):
        import tempfile
        cfg = self._load(50, 120)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "config.yaml"
            Config.save(cfg, p)
            loaded = Config.load(p)
        self.assertEqual(loaded.processing.frame_start, 50)
        self.assertEqual(loaded.processing.frame_end, 120)


if __name__ == "__main__":
    unittest.main()
