"""Тесты YOLO ONNX-бэкенда (задача 23).

ВСЕ тесты mock-based: onnxruntime и модель на диск не грузятся. Фейковый
модуль ``onnxruntime`` подставляется в ``sys.modules`` (как ultralytics в
test_yolo_detector.py); сессия — фиксированный output-массив:

* Config: backend="onnx" парсится; дефолт "torch"; невалидный → ConfigError;
* YoloOnnxDetector.init без onnxruntime (sys.modules → None) → RuntimeError с
  подсказкой pip install;
* detect() с mock-сессией: фильтрация по class=0/conf, маппинг letterbox → кадр;
* Letterbox roundtrip: 640×480 → 640×640 → bbox обратно → те же пиксели;
* Pipeline.build() с backend="onnx" → detector is YoloOnnxDetector (mock).

Формат output в тестах — фактический для yolov8n.onnx: [1, 84, 8400]
(cx,cy,w,h + 80 классов COCO), без встроенного NMS.
"""

import io
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from visio_people_counter.config import Config, ConfigError, ProcessingConfig  # noqa: E402
from visio_people_counter.pipeline import Pipeline  # noqa: E402
from visio_people_counter import pipeline as pipeline_mod  # noqa: E402
from visio_people_counter.motion_detector import MotionDetector  # noqa: E402
from visio_people_counter.yolo_onnx import (  # noqa: E402
    IMG_SIZE, YoloOnnxDetector, letterbox, unletterbox_box, nms_xyxy,
)


# ---------------------------------------------------------------------------
# Фейковый onnxruntime
# ---------------------------------------------------------------------------

class _FakeSession:
    """Минимальная заглушка ort.InferenceSession."""

    def __init__(self, output_array):
        self._out = np.asarray(output_array, dtype=np.float32)
        self.calls: list[np.ndarray] = []

    def get_inputs(self):
        return [SimpleNamespace(name="images", shape=[1, 3, IMG_SIZE, IMG_SIZE])]

    def run(self, output_names, feed):
        assert output_names is None
        self.calls.append(np.asarray(feed["images"]))
        return [self._out]


def _fake_ort(session):
    """Фейковый модуль onnxruntime: InferenceSession(...) → переданная сессия.

    :returns: (модуль, список (path, providers) всех созданных сессий).
    """
    created = []

    def _session(path, sess_options=None, providers=None):
        created.append((str(path), list(providers)))
        return session

    mod = types.ModuleType("onnxruntime")
    mod.InferenceSession = _session
    mod.get_available_providers = lambda: ["CPUExecutionProvider"]
    mod.SessionOptions = type("SessionOptions", (), {})
    mod.GraphOptimizationLevel = SimpleNamespace(ORT_ENABLE_ALL=99)
    return mod, created


def _raw_output(rows):
    """Собрать raw-выход [1, 84, N] из строк (cx, cy, w, h, person_score).

    Остальные классы COCO — нули; conf других классов не влияет на детекцию.
    """
    n = len(rows)
    out = np.zeros((1, 84, max(n, 1)), dtype=np.float32)
    for i, (cx, cy, w, h, score) in enumerate(rows):
        out[0, :5, i] = [cx, cy, w, h, score]
    return out


def _cfg_dict(**processing_extra):
    processing = {"method": "yolo", "yolo": {}}
    processing.update(processing_extra)
    return {
        "video": {"type": "file", "path": "dummy.mp4"},
        "processing": processing,
        "counters": [{
            "id": "main_line", "type": "line",
            "a": [0.25, 0.35], "b": [0.75, 0.85],
        }],
    }


def _make_detector(session, yolo_block=None):
    """YoloOnnxDetector с фейковым onnxruntime → (detector, created)."""
    fake_mod, created = _fake_ort(session)
    cfg = Config.from_dict(_cfg_dict(yolo=yolo_block or {}))
    with patch.dict(sys.modules, {"onnxruntime": fake_mod}):
        det = YoloOnnxDetector(cfg)
    return det, created


# ---------------------------------------------------------------------------
# Config — поле backend
# ---------------------------------------------------------------------------

class TestConfigBackend(unittest.TestCase):

    def test_default_backend_torch(self):
        p = ProcessingConfig.from_dict({"method": "yolo", "yolo": {"conf": 0.5}})
        self.assertEqual(p.yolo["backend"], "torch")

    def test_backend_onnx_parsed(self):
        p = ProcessingConfig.from_dict({
            "method": "yolo",
            "yolo": {"model": "/models/yolov8n.onnx", "conf": 0.5,
                     "device": "cuda:0", "backend": "onnx"},
        })
        self.assertEqual(p.yolo["model"], "/models/yolov8n.onnx")
        self.assertEqual(p.yolo["device"], "cuda:0")
        self.assertEqual(p.yolo["backend"], "onnx")

    def test_invalid_backend_raises_config_error(self):
        with self.assertRaises(ConfigError):
            ProcessingConfig.from_dict({"method": "yolo", "yolo": {"backend": "openvino"}})

    def test_empty_yolo_block_stays_empty(self):
        # блок не задан → {} (дефолты — в pipeline/detector'е)
        self.assertEqual(ProcessingConfig.from_dict({"method": "yolo"}).yolo, {})


# ---------------------------------------------------------------------------
# YoloOnnxDetector — init
# ---------------------------------------------------------------------------

class TestYoloOnnxInit(unittest.TestCase):

    def test_init_loads_session_with_model_path(self):
        session = _FakeSession(_raw_output([]))
        det, created = _make_detector(session, yolo_block={"model": "my_model.onnx"})
        self.assertEqual(created, [("my_model.onnx", ["CPUExecutionProvider"])])
        self.assertIs(det._sess, session)

    def test_init_default_model_onnx(self):
        session = _FakeSession(_raw_output([]))
        _, created = _make_detector(session)
        # пустой блок yolo → детектор сам подставляет yolov8n.onnx
        self.assertEqual(created[0][0], "yolov8n.onnx")

    def test_import_error_gives_pip_hint(self):
        cfg = Config.from_dict(_cfg_dict())
        # sys.modules["onnxruntime"] = None → `import onnxruntime` бросает ImportError
        with patch.dict(sys.modules, {"onnxruntime": None}):
            with self.assertRaises(RuntimeError) as ctx:
                YoloOnnxDetector(cfg)
        msg = str(ctx.exception)
        self.assertIn("pip install onnxruntime", msg)
        self.assertIn("onnxruntime-gpu", msg)

    def test_last_mask_is_none(self):
        det, _ = _make_detector(_FakeSession(_raw_output([])))
        self.assertIsNone(det.last_mask)


# ---------------------------------------------------------------------------
# detect() — mock-сессия: фильтрация и маппинг координат
# ---------------------------------------------------------------------------

class TestYoloOnnxDetect(unittest.TestCase):

    def test_detect_maps_letterbox_boxes_to_frame(self):
        # кадр 640×480: letterbox scale=1.0, pad_y=80 (верх/низ по 80)
        session = _FakeSession(_raw_output([
            (320.0, 80.0 + 240.0, 200.0, 300.0, 0.91),   # центр кадра → bbox (220..420, 90..390)
            (100.0, 100.0, 50.0, 60.0, 0.80),            # bbox letterbox (75..125, 70..130) → кадр (75..125, -10..50)→clip
        ]))
        det, _ = _make_detector(session, yolo_block={"conf": 0.4})
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        blobs = det.detect(image)

        self.assertEqual(len(blobs), 2)
        b = blobs[0]
        self.assertEqual((b.x, b.y, b.w, b.h), (220, 90, 200, 300))
        self.assertEqual(b.area, 60000)
        self.assertAlmostEqual(b.cx, 320.0)
        self.assertAlmostEqual(b.cy, 240.0)
        b2 = blobs[1]
        # x: (75..125)/1.0; y: (70-80)..(130-80) = -10..50 → clip по верху до 0
        self.assertEqual(b2.x, 75)
        self.assertEqual(b2.y, 0)

    def test_detect_filters_by_conf(self):
        session = _FakeSession(_raw_output([
            (320.0, 320.0, 100.0, 100.0, 0.9),
            (100.0, 100.0, 50.0, 50.0, 0.3),   # ниже порога 0.4 → отбрасывается
        ]))
        det, _ = _make_detector(session, yolo_block={"conf": 0.4})
        blobs = det.detect(np.zeros((640, 640, 3), dtype=np.uint8))
        self.assertEqual(len(blobs), 1)
        self.assertAlmostEqual(blobs[0].cx, 320.0)

    def test_detect_empty_output(self):
        session = _FakeSession(_raw_output([]))
        det, _ = _make_detector(session)
        self.assertEqual(det.detect(np.zeros((480, 640, 3), dtype=np.uint8)), [])

    def test_input_array_shape_and_normalization(self):
        session = _FakeSession(_raw_output([]))
        det, _ = _make_detector(session, yolo_block={"model": "m.onnx"})
        image = np.full((480, 640, 3), 127, dtype=np.uint8)
        det.detect(image)

        x = session.calls[0]
        self.assertEqual(x.shape, (1, 3, IMG_SIZE, IMG_SIZE))
        self.assertEqual(x.dtype, np.float32)
        self.assertTrue(np.all((x >= 0.0) & (x <= 1.0)))
        # 127/255 ≈ 0.498 — пиксели кадра нормализованы; padding = 114/255
        self.assertAlmostEqual(float(x[0, 0, 80, 320]), 127.0 / 255.0, places=5)
        self.assertAlmostEqual(float(x[0, 0, 0, 0]), 114.0 / 255.0, places=5)

    def test_supports_post_nms_output_format(self):
        # вариант B: [1, N, 6] = x1,y1,x2,y2,conf,class_id (post-NMS экспорт)
        out = np.array([[[100., 200., 300., 400., 0.9, 0.],
                         [0., 0., 50., 50., 0.8, 7.]]], dtype=np.float32)  # class 7 ≠ person
        session = _FakeSession(out)
        det, _ = _make_detector(session)
        blobs = det.detect(np.zeros((640, 640, 3), dtype=np.uint8))
        self.assertEqual(len(blobs), 1)
        self.assertEqual((blobs[0].x, blobs[0].y, blobs[0].w, blobs[0].h), (100, 200, 200, 200))

    def test_nms_merges_overlapping_boxes(self):
        # два почти идентичных bbox (дубль) → NMS оставляет один (с большим score)
        session = _FakeSession(_raw_output([
            (320.0, 320.0, 100.0, 200.0, 0.95),
            (322.0, 321.0, 102.0, 201.0, 0.85),   # IoU ≈ 0.9 → отбрасывается
        ]))
        det, _ = _make_detector(session)
        blobs = det.detect(np.zeros((640, 640, 3), dtype=np.uint8))
        self.assertEqual(len(blobs), 1)


# ---------------------------------------------------------------------------
# Letterbox roundtrip
# ---------------------------------------------------------------------------

class TestLetterboxRoundtrip(unittest.TestCase):

    def test_640x480_roundtrip(self):
        # изображение 640×480 → letterbox к 640×640 → bbox в 640×640 → обратно
        h, w = 480, 640
        image = np.zeros((h, w, 3), dtype=np.uint8)
        canvas, scale, pad_x, pad_y = letterbox(image, IMG_SIZE)
        self.assertEqual(canvas.shape, (IMG_SIZE, IMG_SIZE, 3))
        self.assertAlmostEqual(scale, 1.0)
        self.assertEqual((pad_x, pad_y), (0, 80))

        # «обнаруженный» bbox в исходных пикселях кадра
        orig = (100.0, 50.0, 300.0, 350.0)  # x1,y1,x2,y2
        # прямой маппинг кадр → letterbox: x_lb = x*scale + pad
        lb_box = ((orig[0] * scale + pad_x), (orig[1] * scale + pad_y),
                  (orig[2] * scale + pad_x), (orig[3] * scale + pad_y))
        # обратное преобразование — те же пиксели
        back = unletterbox_box(*lb_box, scale, pad_x, pad_y, w, h)
        for got, want in zip(back, orig):
            self.assertAlmostEqual(got, want, places=6)

    def test_wide_frame_roundtrip(self):
        # 1280×720 → scale=min(640/1280, 640/720)=0.5, dh=360, pad_x=0, pad_y=(640-360)/2=140
        w, h = 1280, 720
        canvas, scale, pad_x, pad_y = letterbox(np.zeros((h, w, 3), np.uint8), IMG_SIZE)
        self.assertAlmostEqual(scale, 0.5)
        self.assertEqual((pad_x, pad_y), (0, 140))
        orig = (120.0, 60.0, 900.0, 700.0)
        lb_box = (o * scale + p for o, p in zip(orig, (pad_x, pad_y, pad_x, pad_y)))
        back = unletterbox_box(*lb_box, scale, pad_x, pad_y, w, h)
        for got, want in zip(back, orig):
            self.assertAlmostEqual(got, want, places=6)

    def test_clips_to_frame_bounds(self):
        # bbox, ушедший за границу кадра → обрезка по [0,w]×[0,h]
        back = unletterbox_box(-30.0, -20.0, 700.0, 500.0,
                               scale=1.0, pad_x=0, pad_y=80, width=640, height=480)
        self.assertEqual(back, (0.0, 0.0, 640.0, 420.0))

    def test_nms_keeps_disjoint_boxes(self):
        boxes = np.array([[0., 0., 10., 10.], [50., 50., 60., 60.]])
        keep = nms_xyxy(boxes, np.array([0.9, 0.8]))
        self.assertEqual(sorted(keep), [0, 1])


# ---------------------------------------------------------------------------
# Pipeline.build() — выбор детектора по backend
# ---------------------------------------------------------------------------

class TestPipelineBuildOnnxBackend(unittest.TestCase):

    class FakeSource:
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
        with tempfile.TemporaryDirectory(prefix="vpc_test_onnx_") as td:
            vpath = Path(td) / "fake.mp4"
            vpath.write_bytes(b"not a real video; source is mocked")
            cfg.video.path = str(vpath)
            with patch.object(pipeline_mod, "FileSource", self.FakeSource), \
                    redirect_stdout(io.StringIO()):
                pipe = Pipeline(cfg)
                pipe.build()
        return pipe

    def test_backend_onnx_selects_yolo_onnx_detector(self):
        session = _FakeSession(_raw_output([]))
        fake_mod, created = _fake_ort(session)
        cfg = Config.from_dict(_cfg_dict(
            yolo={"model": "yolov8n.onnx", "conf": 0.5, "backend": "onnx"}))
        with patch.dict(sys.modules, {"onnxruntime": fake_mod}):
            pipe = self._build_pipeline(cfg)
        self.assertIsInstance(pipe.detector, YoloOnnxDetector)
        self.assertEqual(created[0][0], "yolov8n.onnx")
        self.assertIsNone(pipe.detector.last_mask)

    def test_backend_default_stays_torch(self):
        # backend не задан → torch-ветка (ultralytics тоже фейковый)
        ultr = types.ModuleType("ultralytics")
        ultr.YOLO = lambda path: object()
        cfg = Config.from_dict(_cfg_dict())  # yolo {} → дефолт backend="torch"
        with patch.dict(sys.modules, {"ultralytics": ultr}):
            pipe = self._build_pipeline(cfg)
        from visio_people_counter.yolo_detector import YoloDetector
        self.assertIsInstance(pipe.detector, YoloDetector)

    def test_default_method_selects_motion_detector(self):
        cfg = Config.from_dict(_cfg_dict(method="mog2"))
        pipe = self._build_pipeline(cfg)
        self.assertIsInstance(pipe.detector, MotionDetector)


if __name__ == "__main__":
    unittest.main()
