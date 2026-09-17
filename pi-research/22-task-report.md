# Задача 22 — YOLO как альтернативный детектор: отчёт (worker)

Дата: реализация по спецификации `pi-tasks/22-yolo-detector.md`.

## Что сделано

### 1. Новый модуль `visio_people_counter/yolo_detector.py`
- Класс `YoloDetector`:
  - `__init__(cfg)` — читает `cfg.processing.yolo.{model,conf,device}` (дефолты
    `yolov8n.pt` / `0.4` / `cpu`); импорт `ultralytics.YOLO` **ленивый** (внутри
    `__init__`, не при import модуля → torch не подтягивается, если метод "mog2");
    при ImportError — `RuntimeError` с подсказкой `pip install ultralytics`.
  - `detect(image, size_profile=None) -> list[Blob]` — `predict(image, classes=[0],
    conf=..., device=..., verbose=False)`; из `result.boxes.xyxy` (N×4) строит
    `Blob(x, y, w, h, area=w*h, cx, cy)` — тот же dataclass из motion_detector.py;
    пустая детекция → `[]`. size_profile принимается для совместимости интерфейса
    (YOLO не использует адаптивные пороги площади — фильтрация через conf).
  - `last_mask` property → None (GUI-overlay: `gui.py:474` берёт mask через
    `getattr(..., "last_mask", None)` и при None просто не рисует микс с маской —
    изменений в GUI не потребовалось).
  - `reset()` — no-op (pipeline вызывает его при reconnect; у YOLO нет состояния).

### 2. `visio_people_counter/config.py`
- `ProcessingConfig`: новые поля `method: str = "mog2"` и
  `yolo: dict = field(default_factory=dict)`.
- `from_dict`: `method` — enum {"mog2","yolo"} (иначе ConfigError); блок `yolo` —
  опциональный dict, ключи {model, conf, device}: непустой блок нормализуется с
  дефолтами (conf обязан быть в (0..1], model/device — непустые str), неизвестные
  ключи → ConfigError; пустой/отсутствующий блок → `{}` (симметричный save/load
  roundtrip, дефолты применяет сам YoloDetector).

### 3. `visio_people_counter/pipeline.py`
- `build()`: после создания SizeProfile — выбор детектора:
  `processing.method == "yolo"` → ленивый `from .yolo_detector import YoloDetector`
  + лог-строка с model/conf/device; иначе — как раньше `MotionDetector(self.cfg)`.
- MotionDetector и MOG2-путь не тронуты; трекер/счётчики работают с list[Blob] без изменений.

### 4. Документация/зависимости
- `config.example.yaml`: в блок processing добавлены `method: mog2` и документированный
  блок `yolo:` (model/conf/device).
- `requirements.txt`: `ultralytics>=8.0` с комментарием «optional: YOLO-детектор».

## Тесты — `tests/test_yolo_detector.py` (18 тестов, все mock-based)
Реальный ultralytics/torch НЕ подтягиваются: в `sys.modules` подменяется фейковый
модуль `ultralytics` (`patch.dict`), модель — MagicMock; tensor-заглушка с
`.cpu().numpy()`.

- **Config** (9): дефолт method="mog2"/yolo={}; непустой блок → дефолты недостающих
  ключей; кастомные model/conf/device; пустой/отсутствующий блок → {}; невалидный
  method → ConfigError; conf вне (0..1] → ConfigError; неизвестный ключ yolo →
  ConfigError; roundtrip через `Config.from_dict`.
- **YoloDetector** (7): init грузит модель с заданным путём; `last_mask is None`;
  ImportError (sys.modules["ultralytics"]=None) → RuntimeError c «pip install
  ultralytics»; detect() с фиксированными boxes.xyxy/conf → точные x/y/w/h/area/cx/cy
  двух Blob'ов; predict вызван с classes=[0], conf, device, verbose=False; пустые
  boxes → []; reset() no-op.
- **Pipeline.build** (2): method="yolo" + mock FileSource + mock YOLO →
  `detector is YoloDetector` и путь модели передаётся в YOLO(); по умолчанию
  (method="mog2") → `MotionDetector`.

## Проверка
```bash
.venv/bin/python -m unittest tests.test_yolo_detector   # Ran 18 tests — OK
.venv/bin/python -m unittest discover -s tests -t .     # Ran 430 tests — OK (полный прогон)
```
Полный прогон доводился до зелёного: один промежуточный сбой
`tests.test_config.TestConfigDefault.test_default_save_load_roundtrip` (пустой
`yolo: {}` после save/load нормализовался в дефолты) устранён симметричной
семантикой «пустой блок → {}».

## Файлы
- новые: `visio_people_counter/yolo_detector.py`, `tests/test_yolo_detector.py`
- изменены: `visio_people_counter/config.py`, `visio_people_counter/pipeline.py`,
  `config.example.yaml`, `requirements.txt`

Коммитов НЕ делалось (правило владельца: только после явного подтверждения).

## Остаточные риски / замечания
1. Реальная модель yolov8n.pt не прогонялась (требование задачи — mock-based):
   первый запуск с method="yolo" скачает ~6 МБ веса; рекомендуется владельцу один
   live-прогон на `videos/` для калибровки conf.
2. YOLO-детектор не применяет фильтры блока `objects` (площадь/aspect/min_fill) —
   по спецификации фильтрация идёт через `conf`; при желании можно добавить
   post-filter, но это расширение scope.
3. Warmup-логика pipeline («субтрактор прогрет») остаётся для yolo без смысла,
   но безвредна (события логируются всегда).
