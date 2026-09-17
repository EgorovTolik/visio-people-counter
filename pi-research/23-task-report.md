# Задача 23 — ONNX Runtime backend для YOLO-детектора: отчёт

Статус: **выполнено**. Все тесты проходят (451 OK), реальный смоук с `yolov8n.onnx` работает.

## 1. Диагностика формата ONNX output (сделана ПЕРВОЙ, до реализации)

Диагностические скрипты (`/tmp/diag_yolo_onnx.py`, `/tmp/diag_yolo_onnx2.py`, не в тестах):

- onnxruntime **1.30.0**, available providers: `AzureExecutionProvider`,
  `CPUExecutionProvider` (GPU-сборка не установлена — CUDA EP будет включаться
  автоматически, если владелец поставит `onnxruntime-gpu`).
- **INPUT**: `images`, shape `[1, 3, 640, 640]`, float32.
- **OUTPUT**: `output0`, shape **`[1, 84, 8400]`**, float32.

**Вывод: это ВАРИАНТ A — raw YOLOv8 БЕЗ встроенного NMS.**
84 = 4 (cx, cy, w, h) + 80 классов COCO; 8400 ячеек anchor-free сетки.
Проверено на пустом кадре (max person score ≈ 0.001) и на шумовом.
Последствия для реализации:

1. NMS делаем сами (жадный, IoU 0.5) — иначе десятки дублей на одного человека;
2. берём строку 4 как class 0 (person), остальные классы игнорируем;
3. координаты в letterbox-канвасе 640×640 → обратное преобразование в кадр.

Для совместимости с другими экспортами поддерживается и вариант B
(`[1, N, 6]` = x1,y1,x2,y2,conf,class_id, post-NMS) — одна ветка в `_parse_output`.

## 2. Что реализовано

### `visio_people_counter/yolo_onnx.py` (новый модуль)
- `YoloOnnxDetector(cfg)` — читает `cfg.processing.yolo.{model,conf,device}`;
  ленивый `import onnxruntime`; при ImportError → RuntimeError с подсказкой
  `pip install onnxruntime` / `onnxruntime-gpu`.
- Providers: `device=cpu` → `[CPUExecutionProvider]`; `cuda|gpu|cuda:N` →
  `[CUDAExecutionProvider, CPUExecutionProvider]` + отсечение недоступных EP
  (fallback на CPU), итоговый список в `det.providers`.
- `detect(image)` → BGR→RGB, letterbox 640×640 (resize с сохранением аспекта,
  pad 114), /255, HWC→CHW, batch dim → float32 `[1,3,640,640]` → `sess.run`.
- Post-processing: class=0 + conf-порог → NMS (IoU 0.5) → unletterbox с clip
  по границам кадра → `list[Blob]` (тот же dataclass из motion_detector.py,
  area = w*h — как в torch-бэкенде).
- Хелперы на модульном уровне (тестируемы): `letterbox()`, `unletterbox_box()`,
  `nms_xyxy()`. `last_mask → None`, `reset()` no-op.
- **Импорт только onnxruntime + numpy + cv2** — проверено: после импорта модуля
  в `sys.modules` нет ни `ultralytics`, ни `torch`.

### `visio_people_counter/config.py`
- В блок `processing.yolo` добавлено поле `backend`: `"torch"` (дефолт) |
  `"onnx"`; невалидное значение → ConfigError. При непустом блоке yolo дефолт
  добирается и попадает в dict: `{model, conf, device, backend}`.

### `visio_people_counter/pipeline.py`
- `build()`: при `method=="yolo"` читает `backend`; `"onnx"` → ленивый импорт
  `YoloOnnxDetector`, иначе — прежний `YoloDetector`. В логе строка про модель +
  providers. Torch-ветка и MOG2-путь не тронуты, `yolo_detector.py` не менялся.

### Прочее
- `requirements.txt`: добавлен `onnxruntime>=1.16` (optional, CPU) +
  закомментированный `onnxruntime-gpu>=1.16`.
- `config.example.yaml`: в блок `yolo` — `backend: torch` с пояснениями.

## 3. Тесты (`tests/test_yolo_onnx.py`, 20 тестов, все mock-based)

Не зависят от наличия onnxruntime/модели: фейковый модуль `onnxruntime` в
`sys.modules`, фиксированный output-массив `[1,84,N]` (фактический формат):

1. Config: дефолт backend="torch", парсинг "onnx", невалидный → ConfigError;
2. init без onnxruntime (sys.modules→None) → RuntimeError с pip-подсказкой;
   init с фейковой сессией — путь к модели и providers передаются корректно;
3. detect() с mock-сессией: маппинг letterbox→кадр (640×480, pad_y=80),
   фильтрация по conf, пустой вывод, shape/нормализация входного тензора
   (127/255 в кадре, 114/255 в padding), post-NMS формат `[1,N,6]`,
   NMS сливает дубли;
4. Letterbox roundtrip: 640×480 и 1280×720 — bbox кадр→letterbox→кадр
   совпадает с исходными пикселями (places=6); clip по границам кадра;
5. Pipeline.build(): backend="onnx" → `detector is YoloOnnxDetector`;
   дефолт → torch-ветка; method="mog2" → MotionDetector.

Обновлены 2 ассерта в `tests/test_yolo_detector.py` (дефолтный dict yolo теперь
включает `"backend": "torch"`).

## 4. Проверки

| Команда | Результат |
|---|---|
| `.venv/bin/python -m unittest discover -s tests -t .` | **Ran 451 tests — OK** (было 84+; +20 новых, −0) |
| импорт `visio_people_counter.yolo_onnx` | OK, torch/ultralytics в sys.modules не появляются |
| смоук: реальная `yolov8n.onnx`, серый кадр 640×480, CPU | сессия за 0.1 с, detect ≈ **33 мс/кадр**, 0 blob'ов (как ожидается на сером фоне), providers=['CPUExecutionProvider'] |

## 5. Ограничения / риски

- GPU не проверен: в .venv стоит CPU-сборка onnxruntime; `device: cuda`
  потребует `pip install onnxruntime-gpu` (записано в requirements.txt).
  GTX 1060 (CC 6.1) поддерживается CUDA EP — но на практике, а не здесь.
- NMS IoU=0.5 — константа по умолчанию YOLO; при необходимости вынести в конфиг
  можно в следующей задаче (не делал — вне scope).
- `backend: onnx` без явного `model` использует дефолт конфига `yolov8n.pt`;
  если файл отсутствует — ошибка от onnxruntime. Рекомендация владельцу: при
  onnx-бэкенде явно указывать `model: yolov8n.onnx`.
- Коммит НЕ сделан (правило владельца — только после подтверждения).

## 6. Изменённые файлы

- `visio_people_counter/yolo_onnx.py` (новый)
- `visio_people_counter/config.py`
- `visio_people_counter/pipeline.py`
- `tests/test_yolo_onnx.py` (новый, 20 тестов)
- `tests/test_yolo_detector.py` (2 ассерта под новый дефолт backend)
- `requirements.txt`, `config.example.yaml`
