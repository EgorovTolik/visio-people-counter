"""Тесты YOLO-детектора (задача 22).

ВСЕ тесты mock-based: реальная модель на диск не грузится, ``ultralytics``
подменяется фейковым модулем в ``sys.modules`` (torch не подтягивается):

* Config: method="yolo" парсится; yolo.model/conf/device — дефолты и кастом;
  невалидный method / conf вне (0..1] → ConfigError;
* YoloDetector.init: модель загружается через (mock) YOLO(model_path);
  ImportError (нет ultralytics) → RuntimeError с подсказкой pip install;
* detect(): mock-predict → фиксированные boxes.xyxy/conf → проверяем list[Blob];
* Pipeline.build() с method="yolo" → detector is YoloDetector (mock source +
  mock YOLO); по умолчанию — MotionDetector.
"""

import io
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from visio_people_counter.config import Config, ConfigError, ProcessingConfig  # noqa: E402
from visio_people_counter.pipeline import Pipeline  # noqa: E402
from visio_people_counter import pipeline as pipeline_mod  # noqa: E402
from visio_people_counter.motion_detector import Blob, MotionDetector  # noqa: E402
from visio_people_counter.yolo_detector import YoloDetector  # noqa: E402


class _FakeTensor:
    """Заглушка torch-тензора: .cpu().numpy() → ndarray."""

    def __init__(self, arr):
        self._arr = np.asarray(arr, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


def _fake_ultralytics():
    """Фейковый модуль ultralytics: YOLO(path) → общий mock-модель.

    :returns: (модуль, mock-модель, список загруженных model_path).
    """
    loaded_paths = []
    model = MagicMock(name="yolo_model")

    def _yolo(path):
        loaded_paths.append(path)
        return model

    mod = types.ModuleType("ultralytics")
    mod.YOLO = _yolo
    return mod, model, loaded_paths


def _cfg_dict(**processing_extra):
    processing = {"method": "yolo"}
    processing.update(processing_extra)
    return {
        "video": {"type": "file", "path": "dummy.mp4"},
        "processing": processing,
        "counters": [{
            "id": "main_line", "type": "line",
            "a": [0.25, 0.35], "b": [0.75, 0.85],
        }],
    }


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfigYolo(unittest.TestCase):

    def test_default_method_mog2_and_empty_yolo(self):
        p = ProcessingConfig.from_dict({})
        self.assertEqual(p.method, "mog2")
        self.assertEqual(p.yolo, {})

    def test_yolo_defaults_for_missing_keys(self):
        # непустой блок yolo → отсутствующие ключи добираются дефолтами
        p = ProcessingConfig.from_dict({"method": "yolo", "yolo": {"conf": 0.5}})
        self.assertEqual(p.method, "yolo")
        self.assertEqual(p.yolo, {"model": "yolov8n.pt", "conf": 0.5, "device": "cpu",
                                  "backend": "torch"})

    def test_yolo_block_absent_or_empty_stays_empty_dict(self):
        # блок yolo не задан или пустой → {} (YoloDetector сам применяет дефолты)
        for d in ({"method": "yolo"}, {"method": "yolo", "yolo": {}}):
            self.assertEqual(ProcessingConfig.from_dict(d).yolo, {})

    def test_yolo_custom_values(self):
        p = ProcessingConfig.from_dict({
            "method": "yolo",
            "yolo": {"model": "/models/yolov8s.pt", "conf": 0.7, "device": "cuda:0"},
        })
        self.assertEqual(p.yolo["model"], "/models/yolov8s.pt")
        self.assertEqual(p.yolo["conf"], 0.7)
        self.assertEqual(p.yolo["device"], "cuda:0")

    def test_mog2_with_yolo_block_still_parses(self):
        p = ProcessingConfig.from_dict({"method": "mog2", "yolo": {"conf": 0.5}})
        self.assertEqual(p.method, "mog2")
        self.assertEqual(p.yolo["conf"], 0.5)

    def test_invalid_method_raises_config_error(self):
        with self.assertRaises(ConfigError):
            ProcessingConfig.from_dict({"method": "haar"})

    def test_conf_out_of_range_raises_config_error(self):
        with self.assertRaises(ConfigError):
            ProcessingConfig.from_dict({"method": "yolo", "yolo": {"conf": 1.5}})
        with self.assertRaises(ConfigError):
            ProcessingConfig.from_dict({"method": "yolo", "yolo": {"conf": 0.0}})

    def test_unknown_yolo_key_raises_config_error(self):
        with self.assertRaises(ConfigError):
            ProcessingConfig.from_dict({"method": "yolo", "yolo": {"iou": 0.5}})

    def test_full_config_roundtrip(self):
        cfg = Config.from_dict(_cfg_dict(yolo={"conf": 0.5}))
        self.assertEqual(cfg.processing.method, "yolo")
        self.assertEqual(cfg.processing.yolo,
                         {"model": "yolov8n.pt", "conf": 0.5, "device": "cpu",
                          "backend": "torch"})


# ---------------------------------------------------------------------------
# YoloDetector — init / detect (mock-модель)
# ---------------------------------------------------------------------------

class TestYoloDetector(unittest.TestCase):

    def _make_detector(self, model=None, yolo_block=None):
        fake_mod, m, paths = _fake_ultralytics()
        cfg = Config.from_dict(_cfg_dict(yolo=yolo_block or {}))
        with patch.dict(sys.modules, {"ultralytics": fake_mod}):
            det = YoloDetector(cfg)
        if model is not None:
            det._model = model
        return det, m, paths

    def test_init_loads_model_with_configured_path(self):
        _, _, paths = self._make_detector(yolo_block={"model": "my_model.pt"})
        self.assertEqual(paths, ["my_model.pt"])

    def test_last_mask_is_none(self):
        det, _, _ = self._make_detector()
        self.assertIsNone(det.last_mask)

    def test_import_error_gives_pip_hint(self):
        cfg = Config.from_dict(_cfg_dict())
        # sys.modules["ultralytics"] = None → `from ultralytics import ...` бросает ImportError
        with patch.dict(sys.modules, {"ultralytics": None}):
            with self.assertRaises(RuntimeError) as ctx:
                YoloDetector(cfg)
        self.assertIn("pip install ultralytics", str(ctx.exception))

    def test_detect_returns_blobs_from_mock_boxes(self):
        model = MagicMock(name="yolo_model")
        boxes = SimpleNamespace(
            xyxy=_FakeTensor([[10.0, 20.0, 50.0, 80.0], [100.3, 40.7, 141.9, 200.8]]),
            conf=_FakeTensor([0.91, 0.62]),
        )
        model.predict.return_value = [SimpleNamespace(boxes=boxes)]
        det, _, _ = self._make_detector(model=model)

        image = np.zeros((360, 640, 3), dtype=np.uint8)
        blobs = det.detect(image)

        self.assertEqual(len(blobs), 2)
        for b in blobs:
            self.assertIsInstance(b, Blob)
        x1, y1, w1, h1 = blobs[0].x, blobs[0].y, blobs[0].w, blobs[0].h
        self.assertEqual((x1, y1, w1, h1), (10, 20, 40, 60))
        self.assertEqual(blobs[0].area, 2400)
        self.assertAlmostEqual(blobs[0].cx, 30.0)
        self.assertAlmostEqual(blobs[0].cy, 50.0)
        # второй bbox: x=round(100.3)=100, y=round(40.7)=41, w=round(41.6)=42, h=round(160.1)=160
        self.assertEqual((blobs[1].x, blobs[1].y, blobs[1].w, blobs[1].h), (100, 41, 42, 160))

    def test_detect_calls_predict_with_expected_args(self):
        model = MagicMock(name="yolo_model")
        model.predict.return_value = []
        det, _, _ = self._make_detector(model=model, yolo_block={"conf": 0.7, "device": "cuda:0"})

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        det.detect(image)

        args, kwargs = model.predict.call_args
        self.assertIs(args[0], image)
        self.assertEqual(kwargs["classes"], [0])
        self.assertEqual(kwargs["conf"], 0.7)
        self.assertEqual(kwargs["device"], "cuda:0")
        self.assertFalse(kwargs["verbose"])

    def test_detect_empty_boxes_returns_empty_list(self):
        model = MagicMock(name="yolo_model")
        boxes = SimpleNamespace(
            xyxy=_FakeTensor(np.zeros((0, 4))),
            conf=_FakeTensor(np.zeros(0)),
        )
        model.predict.return_value = [SimpleNamespace(boxes=boxes)]
        det, _, _ = self._make_detector(model=model)

        blobs = det.detect(np.zeros((100, 100, 3), dtype=np.uint8))
        self.assertEqual(blobs, [])

    def test_reset_is_noop(self):
        det, _, _ = self._make_detector()
        self.assertIsNone(det.reset())


# ---------------------------------------------------------------------------
# Pipeline.build() — выбор детектора по processing.method
# ---------------------------------------------------------------------------

class TestPipelineBuildDetectorChoice(unittest.TestCase):

    class FakeSource:
        """Минимальная заглушка VideoSource (open/read/close, фиксированный w/h/fps)."""
        width = 320
        height = 240
        fps = 15.0
        duration = 0.0

        def __init__(self, *args, **kwargs):
            pass

        def open(self):
            pass

        def read(self):
            return None

        def close(self):
            pass

    def _build_pipeline(self, cfg):
        with tempfile.TemporaryDirectory(prefix="vpc_test_yolo_") as td:
            vpath = Path(td) / "fake.mp4"
            vpath.write_bytes(b"not a real video; source is mocked")
            cfg.video.path = str(vpath)
            with patch.object(pipeline_mod, "FileSource", self.FakeSource), \
                    redirect_stdout(io.StringIO()):
                pipe = Pipeline(cfg)
                pipe.build()
        return pipe

    def test_method_yolo_selects_yolo_detector(self):
        fake_mod, model, paths = _fake_ultralytics()
        cfg = Config.from_dict(_cfg_dict())
        with patch.dict(sys.modules, {"ultralytics": fake_mod}):
            pipe = self._build_pipeline(cfg)
        self.assertIsInstance(pipe.detector, YoloDetector)
        self.assertEqual(paths, ["yolov8n.pt"])
        self.assertIsNone(pipe.detector.last_mask)

    def test_default_method_selects_motion_detector(self):
        cfg = Config.from_dict(_cfg_dict(method="mog2"))
        pipe = self._build_pipeline(cfg)
        self.assertIsInstance(pipe.detector, MotionDetector)


if __name__ == "__main__":
    unittest.main()
