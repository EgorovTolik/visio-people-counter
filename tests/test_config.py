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


if __name__ == "__main__":
    unittest.main()
