"""YOLO-детектор людей (альтернатива MOG2, задача 22).

``YoloDetector`` оборачивает ``ultralytics.YOLO`` и возвращает тот же список
:class:`~visio_people_counter.motion_detector.Blob`, что и
:class:`~visio_people_counter.motion_detector.MotionDetector` — трекер и счётчики
не знают, какой именно детектор работает. Детектируются только люди (COCO
class=0); при плотной группе каждый человек получает отдельный bbox. Маски
движения нет (``last_mask → None``) — GUI-overlay просто не рисует микс с маской.

Импорт ``ultralytics`` ленивый: сам модуль импортируется без torch; пакет
подтягивается только при создании :class:`YoloDetector` (pipeline делает это,
когда ``processing.method: "yolo"``). При отсутствии пакета — понятная ошибка
с подсказкой установки.

YOLO работает на ROI-кадре (источник уже кропнут) — координаты совпадают с
остальными модулями.
"""

from __future__ import annotations

import numpy as np

from .config import Config
from .motion_detector import Blob


class YoloDetector:
    """Детекция людей через ultralytics YOLO → list[Blob] (формат MotionDetector)."""

    def __init__(self, cfg: Config) -> None:
        y = cfg.processing.yolo or {}
        self.model_path: str = str(y.get("model", "yolov8n.pt"))
        self.conf: float = float(y.get("conf", 0.4))
        self.device: str = str(y.get("device", "cpu"))

        try:
            from ultralytics import YOLO  # лениво: тянет torch
        except ImportError as e:
            raise RuntimeError(
                "ultralytics не установлен — не могу создать YoloDetector. "
                "Установите пакет в venv проекта: pip install ultralytics"
            ) from e

        self._model = YOLO(self.model_path)

    @property
    def last_mask(self) -> None:
        """Маски движения нет (в отличие от MotionDetector) — GUI не рисует микс."""
        return None

    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray, size_profile=None) -> list[Blob]:
        """Обработать один BGR-кадр и вернуть bbox'ы людей как list[Blob].

        :param image: кадр в исходных координатах (BGR, uint8; ROI-кадр).
        :param size_profile: принимается для совместимости интерфейса с
                             MotionDetector; YOLO не использует адаптивные пороги
                             площади (фильтрация — порог ``conf`` модели).
        """
        results = self._model.predict(
            image, classes=[0], conf=self.conf, device=self.device, verbose=False
        )

        blobs: list[Blob] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            # пустая детекция → xyxy (0×4) → цикл не даёт blob'ов
            xyxy = np.asarray(boxes.xyxy.cpu().numpy())
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = (float(v) for v in xyxy[i])
                w = int(round(x2 - x1))
                h = int(round(y2 - y1))
                if w <= 0 or h <= 0:
                    continue  # вырожденный bbox
                blobs.append(Blob(
                    x=int(round(x1)), y=int(round(y1)), w=w, h=h,
                    area=w * h, cx=x1 + w / 2.0, cy=y1 + h / 2.0,
                ))
        return blobs

    def reset(self) -> None:
        """No-op: у YOLO нет состояния (pipeline вызывает при reconnect)."""
