"""YOLO-детектор людей через ONNX Runtime (задача 23, альтернатива torch).

``YoloOnnxDetector`` — тот же интерфейс, что и
:class:`~visio_people_counter.yolo_detector.YoloDetector`: ``detect(frame) →
list[Blob]``, ``last_mask → None``. Трекер и счётчики не знают, какой бэкенд
работает. Детектируются только люди (COCO class=0).

Почему ONNX Runtime: GTX 1060 (Pascal, CC 6.1) не поддерживается torch ≥2.3, а
torch 2.2 конфликтует с numpy 2 / OpenCV 5. ``onnxruntime`` — отдельный
runtime без PyTorch; для GPU ставится ``onnxruntime-gpu`` и задаётся
``device: cuda`` (execution providers: CUDA + fallback на CPU).

Формат вывода (проверено диагностикой, см. pi-research/23-task-report.md):
у экспорта yolov8n.onnx **нет встроенного NMS** — output ``[1, 84, 8400]``:
строки 0–3 = cx, cy, w, h (letterbox 640×640), строки 4–83 = классы COCO.
Поэтому здесь: фильтр class=0 + conf-порог + собственный NMS (IoU 0.5).
Для моделей, экспортированных c post-NMS (output ``[1, N, 6]`` =
x1, y1, x2, y2, conf, class_id), поддерживается и этот вариант.

Импорт ``onnxruntime`` ленивый: модуль импортируется без него; пакет
подтягивается только при создании :class:`YoloOnnxDetector`. При отсутствии —
понятная ошибка с подсказкой установки. Зависимости модуля: onnxruntime +
numpy + cv2 (ultralytics/torch НЕ импортируются).

YOLO работает на ROI-кадре (источник уже кропнут) — координаты совпадают с
остальными модулями.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import Config
from .motion_detector import Blob

#: размер стороны letterbox-канваса (на котором обучена модель)
IMG_SIZE = 640
#: цвет padding'а, используемый ultralytics при pre-processing
_PAD_VALUE = 114
#: порог IoU для NMS над «сырыми» детекциями
_NMS_IOU = 0.5


def letterbox(image: np.ndarray, new_size: int = IMG_SIZE):
    """Letterbox-преобразование кадра до ``new_size × new_size``.

    Пропорциональный resize (без искажения аспекта) + центрированный padding
    серым (114). Возвращает канвас в исходном BGR/uint8 (RGB-конверсию делает
    вызывающий, сразу перед нормализацией).

    :returns: (canvas, scale, pad_x, pad_y) — canvas ``[new_size, new_size, 3]``,
              масштаб ``min(new/w, new/h)``, смещения padding'а в px канваса.
    """
    h, w = image.shape[:2]
    scale = min(new_size / float(w), new_size / float(h))
    dw = max(1, int(round(w * scale)))
    dh = max(1, int(round(h * scale)))
    if (dw, dh) != (w, h):
        resized = cv2.resize(image, (dw, dh), interpolation=cv2.INTER_LINEAR)
    else:
        resized = image
    pad_x = (new_size - dw) // 2
    pad_y = (new_size - dh) // 2
    canvas = np.full((new_size, new_size, 3), _PAD_VALUE, dtype=np.uint8)
    canvas[pad_y:pad_y + dh, pad_x:pad_x + dw] = resized
    return canvas, scale, pad_x, pad_y


def unletterbox_box(x1: float, y1: float, x2: float, y2: float,
                    scale: float, pad_x: int, pad_y: int,
                    width: int, height: int):
    """Обратное преобразование bbox из letterbox-канваса в координаты кадра.

    :returns: (x1, y1, x2, y2) в px исходного кадра, обрезанные по границам
              ``[0, width] × [0, height]``.
    """
    x1 = (x1 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    x2 = (x2 - pad_x) / scale
    y2 = (y2 - pad_y) / scale
    return (max(0.0, min(float(width), x1)), max(0.0, min(float(height), y1)),
            max(0.0, min(float(width), x2)), max(0.0, min(float(height), y2)))


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = _NMS_IOU) -> list[int]:
    """Жадный NMS над bbox'ами xyxy; возвращает индексы оставшихся (по убыванию score)."""
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0].astype(np.float64)
    y1 = boxes[:, 1].astype(np.float64)
    x2 = boxes[:, 2].astype(np.float64)
    y2 = boxes[:, 3].astype(np.float64)
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.astype(np.float64).argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter)
        order = rest[iou <= iou_threshold]
    return keep


class YoloOnnxDetector:
    """Детекция людей через ONNX Runtime → list[Blob] (формат MotionDetector)."""

    @staticmethod
    def _setup_cuda_env() -> None:
        """Автонастройка окружения для CUDA EP (cuDNN/cuBLAS из nvidia pip-пакетов).

        LD_LIBRARY_PATH в os.environ не влияет на уже запущенный process (ld.so кэш).
        Поэтому: 1) обновляем env (для дочерних процессов), 2) явно preload .so через
        ctypes.CDLL(RTLD_GLOBAL) — это делает символы доступными для dlopen (onnxruntime).
        """
        import ctypes
        import glob
        import os
        import site
        nvidia_dirs = []
        for sp in site.getsitepackages():
            base = os.path.join(sp, "nvidia")
            if not os.path.isdir(base):
                continue
            for sub in ("cudnn", "cublas", "cufft", "curand"):
                lib_dir = os.path.join(base, sub, "lib")
                if os.path.isdir(lib_dir) and lib_dir not in nvidia_dirs:
                    nvidia_dirs.append(lib_dir)
        # CUDA toolkit
        for cuda_path in ("/usr/local/cuda-12.8", "/usr/local/cuda-12", "/usr/local/cuda"):
            lib = os.path.join(cuda_path, "lib64")
            if os.path.isdir(lib):
                nvidia_dirs.append(lib)
                break
        if not nvidia_dirs:
            return
        # 1) env (для дочерних / последующих запусков)
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        new_parts = [d for d in nvidia_dirs if d not in existing]
        if new_parts:
            os.environ["LD_LIBRARY_PATH"] = ":".join(new_parts + ([existing] if existing else []))
        # 2) preload .so (RTLD_GLOBAL — символы видны всем последующим dlopen)
        rtld_global = getattr(ctypes, "RTLD_GLOBAL", 0x0000100)
        for d in nvidia_dirs:
            for so in sorted(glob.glob(os.path.join(d, "libcudnn*.so*"))
                             + glob.glob(os.path.join(d, "libcublas*.so*"))):
                try:
                    ctypes.CDLL(so, mode=rtld_global)
                except OSError:   # уже загружен или несовместим — пропускаем
                    pass

    def __init__(self, cfg: Config) -> None:
        y = cfg.processing.yolo or {}
        self.model_path: str = str(y.get("model") or "yolov8n.onnx")
        self.conf: float = float(y.get("conf", 0.4))
        self.device: str = str(y.get("device", "cpu"))

        # auto-setup CUDA env before importing onnxruntime
        dev_l = self.device.lower()
        if dev_l in ("cuda", "gpu") or dev_l.startswith("cuda:"):
            self._setup_cuda_env()

        try:
            import onnxruntime as ort  # лениво: тяжёлый, ставится опционально
        except ImportError as e:
            raise RuntimeError(
                "onnxruntime не установлен — не могу создать YoloOnnxDetector. "
                "Установите пакет в venv проекта: pip install onnxruntime "
                "(для GPU: pip install onnxruntime-gpu и device: cuda)"
            ) from e

        self._ort = ort
        dev = self.device.lower()
        if dev in ("cuda", "gpu") or dev.startswith("cuda:"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        available = set(ort.get_available_providers())
        # fallback на CPU, если CUDA-провайдера в этой сборке onnxruntime нет
        self.providers: list[str] = [p for p in providers if p in available] \
            or ["CPUExecutionProvider"]

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess = ort.InferenceSession(self.model_path, sess_options=sess_options,
                                          providers=self.providers)
        self._input_name: str = self._sess.get_inputs()[0].name

    @property
    def last_mask(self) -> None:
        """Маски движения нет (в отличие от MotionDetector) — GUI не рисует микс."""
        return None

    # ------------------------------------------------------------------

    def _parse_output(self, out: np.ndarray):
        """Вывод сессии → (boxes_xyxy, scores) людей в координатах letterbox-канваса.

        Поддерживаемые варианты экспорта:
        * ``[1, 4+C, N]`` — raw YOLOv8 без NMS: строки 0–3 = cx,cy,w,h,
          строка 4 = class 0 (person), далее остальные классы COCO;
        * ``[1, N, 6]``   — post-NMS: [x1, y1, x2, y2, conf, class_id].
        """
        arr = np.asarray(out)
        if arr.ndim == 3 and arr.shape[2] == 6:
            # post-NMS: [1, N, 6] = [x1, y1, x2, y2, conf, class_id]
            dets = arr.reshape(-1, 6)
            person = dets[dets[:, 5] == 0]
            return person[:, :4].astype(np.float64), person[:, 4].astype(np.float64)
        # raw: [1, C, N] → (N, C); без squeeze (пустой вывод (1,84,1) не схлопывается)
        t = arr[0].T  # (N, 4+C)
        cx, cy, w, h = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
        boxes = np.stack([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], axis=1)
        scores = t[:, 4].astype(np.float64)  # class 0 (person)
        return boxes.astype(np.float64), scores

    def detect(self, image: np.ndarray, size_profile=None) -> list[Blob]:
        """Обработать один BGR-кадр и вернуть bbox'ы людей как list[Blob].

        :param image: кадр в исходных координатах (BGR, uint8; ROI-кадр).
        :param size_profile: принимается для совместимости интерфейса с
                             MotionDetector; YOLO не использует адаптивные пороги
                             площади (фильтрация — порог ``conf`` модели + NMS).
        """
        h, w = image.shape[:2]
        canvas, scale, pad_x, pad_y = letterbox(image, IMG_SIZE)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])

        out = self._sess.run(None, {self._input_name: x})[0]
        boxes, scores = self._parse_output(out)

        mask = scores >= self.conf
        if not mask.any():
            return []
        boxes, scores = boxes[mask], scores[mask]
        keep = nms_xyxy(boxes, scores, _NMS_IOU)

        blobs: list[Blob] = []
        for i in keep:
            x1, y1, x2, y2 = unletterbox_box(float(boxes[i, 0]), float(boxes[i, 1]),
                                             float(boxes[i, 2]), float(boxes[i, 3]),
                                             scale, pad_x, pad_y, w, h)
            bw = int(round(x2 - x1))
            bh = int(round(y2 - y1))
            if bw <= 0 or bh <= 0:
                continue  # вырожденный bbox (совсем уехал за границу кадра)
            blobs.append(Blob(
                x=int(round(x1)), y=int(round(y1)), w=bw, h=bh,
                area=bw * bh, cx=x1 + bw / 2.0, cy=y1 + bh / 2.0,
            ))
        return blobs

    def reset(self) -> None:
        """No-op: у сессии ONNX нет состояния (pipeline вызывает при reconnect)."""
