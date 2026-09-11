# visio-people-counter

Подсчёт трафика людей по видеонаблюдению: локальный файл (mp4/avi/mkv) или
HLS-поток (`.m3u8`). Без классификации объектов — motion-based конвейер
(background subtraction → трекер → линия/зона пересечения), работающий на CPU.

## Установка

Требуется Python 3.10+ и системные `ffmpeg`/`ffprobe`. Окружение проекта — `.venv`:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
```

Все команды запускаются через `.venv/bin/python`.

## Конфигурация

Эталон схемы конфига — [`config.example.yaml`](config.example.yaml) (все блоки,
ключи и дефолты с комментариями: `video`, `processing`, `motion`, `objects`,
`size_profile`, `tracker`, `counters`, `output`, `debug`). Скопируйте и подстройте:

```bash
cp config.example.yaml config.yaml
```

Все координаты (линия, зона, size-профиль) заданы в нормализованных долях кадра
0..1 — конфиг переживает смену разрешения камеры. Быстрый способ получить
координаты линии/зоны — GUI-калибровка `calibrate` (см. ниже).

## Режимы работы

### 1. Диагностика входа (`probe`)

Разрешение/fps/длительность/кодек по ffprobe; для файла дополнительно проверяется
открытие через `cv2.VideoCapture`:

```bash
.venv/bin/python -m visio_people_counter probe --video videos/demo.mp4
.venv/bin/python -m visio_people_counter probe --video https://cam.example.com/live/stream.m3u8
```

### 2. Подсчёт (`count`) — headless по умолчанию

Без окон, на максимальной скорости обработки; события печатаются в stdout и
пишутся в JSONL (`output.events_jsonl`); при обрыве HLS-потока — автоматическое
переподключение (watchdog, `video.hls.*`):

```bash
.venv/bin/python -m visio_people_counter count --config config.yaml
# разовый прогон другого видео/потока (--video переопределяет video.path из конфига)
.venv/bin/python -m visio_people_counter count --config config.yaml --video videos/demo.mp4
.venv/bin/python -m visio_people_counter count --config config.yaml --video https://cam.example.com/live/stream.m3u8
# бенчмарк: время этапов detect/track/count, p50/p95 в финале
.venv/bin/python -m visio_people_counter count --config config.yaml --bench
```

Завершение: EOF файла → финальная сводка; стрим → `Ctrl+C` (SIGINT) — graceful stop
с финальной сводкой.

### 3. GUI-режим (`count --gui`)

Окно с overlay по блоку `debug` конфига: bbox треков с `track_id`, необработанные
blob'ы, микс с маской движения, линии/зоны счётчиков и крупные в/out-счётчики.
`--speed` задаёт скорость воспроизведения (0.25..8; по умолчанию 1.0) — работает
только в GUI-режиме. В headless-окружении (нет DISPLAY/WAYLAND_DISPLAY) команда
завершается с понятной ошибкой и кодом 1.

```bash
.venv/bin/python -m visio_people_counter count --config config.yaml --gui --speed 0.5
```

**Клавиши GUI:**

| Клавиша | Действие |
|---|---|
| пробел | пауза / продолжить (последний кадр удерживается) |
| q / ESC | выход (финальная обработка) |
| + / = | скорость ×1.5 (до 8.0) |
| − | скорость ÷1.5 (до 0.25) |

### 4. Калибровка (`calibrate`)

GUI-режим: рисование линии/зоны/size-точек мышью; результат записывается прямо в
`config.yaml` (остальные блоки сохраняются), в stdout печатается короткий diff
«старые → новые координаты».

**Первый запуск без конфига:** если `config.yaml` ещё нет, calibrate стартует с настройками
по умолчанию (один line-счётчик `main_line`) и создаст файл при сохранении [a].
Остальные режимы (`count`) требуют существующий конфиг:

```bash
.venv/bin/python -m visio_people_counter calibrate --config config.yaml --video videos/demo.mp4
# другой id счётчика (по умолчанию main_line)
.venv/bin/python -m visio_people_counter calibrate --config config.yaml --video videos/demo.mp4 --counter-id entry_zone
```

**Клавиши калибровки:**

| Клавиша | Действие |
|---|---|
| `l`, затем 2 клика | линия A→B (порядок кликов = направление «in»); повторный `l` или 3-й клик — заново |
| `z`, N кликов, Enter | полигон зоны (≥3 точек) |
| `s`, клик, цифра 1–9 | size-точка: x из клика, высота человека = цифра×5% высоты кадра (1=5% … 9=45%); Enter — завершить набор |
| `m` | toggle маски движения (видно, что детектится; для подбора `motion.var_threshold` и т.п.) |
| `n` / `p` | следующий / предыдущий из загруженных кадров (первые ≤20) |
| `a` | применить и записать в config.yaml (можно повторно после доводки) |
| q / ESC | выход без сохранения |

## Тесты

Тесты на stdlib `unittest` (в т.ч. с синтетическим mp4, GUI/calibrate-логика
тестируется без реального окна):

```bash
.venv/bin/python -m unittest discover -s tests -t .
```

## Структура проекта

```
visio_people_counter/
├── __main__.py         # CLI: probe / count [--gui --speed --bench] / calibrate
├── config.py           # Config: dataclass + загрузка/сохранение YAML, валидация (схема = config.example.yaml)
├── video_source.py     # FileSource (cv2.VideoCapture), FfmpegPipeSource (ffmpeg-pipe + watchdog-переподключение), ffprobe
├── motion_detector.py  # MOG2/KNN background subtraction + тени/морфология + фильтры объектов → Blob'ы
├── size_profile.py     # адаптивные пороги площади по контрольным точкам (высота человека в точке кадра)
├── tracker_adapter.py  # Blob'ы ↔ sv.Detections(confidence=None) → SORT/ByteTrack/BoTSORT (пакет trackers), reset()
├── line_counter.py     # LineCounter + ZoneCounter: пересечение линии/зоны, антидубль (cooldown/буфер/global gap)
├── event_log.py        # JSONL-лог событий + агрегаты in/out/total и сводки
├── pipeline.py         # главный headless-цикл: источник → детекция → трекер → счётчики, тайминги/fps/lag, SIGINT, --bench
└── gui.py              # GUI-режим: overlay (cfg.debug) + окно с паузой/скоростью; headless-фолбэк
tests/                  # unittest-тесты всех модулей (без реального видео/окна)
config.example.yaml     # эталон схемы конфигурации
```

## Архитектура (конвейер)

Кадровый конвейер: **источник → детекция движения → трекинг → счётчики → лог**.
Модули готовились последовательно, за каждым — отчёт в `pi-research/`:

| Модуль | Ответственность | Отчёт |
|---|---|---|
| `config.py` | Config: dataclass + загрузка/сохранение YAML, валидация по схеме | [11](pi-research/11-task-report.md) |
| `video_source.py` | FileSource (cv2), FfmpegPipeSource (ffmpeg-pipe + watchdog переподключения) | [11](pi-research/11-task-report.md) |
| `motion_detector.py`, `size_profile.py` | MOG2/KNN + тени/морфология/фильтры → Blob'ы; size-профиль по контрольным точкам | [12](pi-research/12-task-report.md) |
| `tracker_adapter.py` | Blob'ы ↔ `sv.Detections(confidence=None)` → SORT/ByteTrack/BoTSORT, устойчивые `track_id`, `reset()` | [13](pi-research/13-task-report.md) |
| `line_counter.py`, `event_log.py` | LineCounter + ZoneCounter, антидубль; JSONL-лог событий и сводки | [14](pi-research/14-task-report.md) |
| `pipeline.py` | главный цикл: источник → детекция → трекер → счётчики, тайминги/fps/lag, сводки, SIGINT, --bench | [15](pi-research/15-task-report.md) |
| `gui.py`, `calibrate.py` | GUI-режим (--gui/--speed, overlay) и мышиная калибровка линии/зоны/size-точек → config.yaml | [16](pi-research/16-task-report.md) |

Детали архитектуры — в [`pi-research/04-implementation-notes.md`](pi-research/04-implementation-notes.md);
итоговое код-ревью и список ограничений — в [`pi-research/17-review-report.md`](pi-research/17-review-report.md).
