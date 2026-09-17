# 18 — Детальный разбор подмодулей конвейера `visio-people-counter`

**Дата:** 2025-09-16
**Цель:** посמודульный технический анализ исходников пакета `visio_people_counter/`
(файлы `.py`, без запуска тестов). Разбор охватывает каждый публичный модуль:
назначение, структуру данных, алгоритмы, обработку крайних случаев и дизайн-решения.

## Оглавление
1. [Обзор архитектуры](#1-обзор-архитектуры)
2. [`__main__.py` — CLI](#2-mainpy--cli)
3. [`config.py` — загрузка и валидация конфига](#3-configpy--загрузка-и-валидация-конфига)
4. [`video_source.py` — источники кадров](#4-video_sourcepy--источники-кадров)
5. [`motion_detector.py` — выделение движения](#5-motion_detectorpy--выделение-движения)
6. [`size_profile.py` — адаптивные пороги площади](#6-size_profilepy--адаптивные-пороги-площади)
7. [`tracker_adapter.py` — трекеры SORT/ByteTrack/BoTSORT](#7-tracker_adapterpy--трекеры-sortbytetrackbotsort)
8. [`line_counter.py` — счётчики пересечения](#8-line_counterpy--счётчики-пересечения)
9. [`pipeline.py` — главный конвейер](#9-pipelinepy--главный-конвейер)
10. [`event_log.py` и `report.py` — выходные данные](#10-event_logpy-и-reportpy--выходные-данные)
11. [`gui.py` / `gui_qt.py`, `calibrate*.py`, `text_overlay.py`](#11-guipy--gui_qtpy-calibratepy-text_overlaypy--дополнительно)
12. [Ключевые дизайн-решения](#12-ключевые-дизайн-решения)
13. [Заметки по надёжности и потенциальные риски](#13-заметки-по-надёжности-и-потенциальные-риски)

---

## 1. Обзор архитектуры

Программа — консольный конвейер подсчёта людей по видео (файл mp4/avi/mkv или HLS
`.m3u8`), motion-based, на CPU, без нейросетевой классификации. Схема данных:

```
        ┌───────────────┐   build()    ┌──────────────────────────────────────┐
  YAML ─┤  Config       ├────────────▶ │ Pipeline.run() / .step(frame)          │
 (YAML) └───────────────┘              │                                          │
                                      │  source.read() → Frame                   │
                                      │    ├── FileSource        (cv2.VideoCapture)│
                                      │    ├── FfmpegPipeSource (ffmpeg pipe, HLS)│
                                      │    ▼                                       │
                                      │  detector.detect(frame) → list[Blob]      │
                                      │    └── SizeProfile (адаптивные пороги)     │
                                      │    ▼                                       │
                                      │  tracker.update(blobs) → list[TrackedObject]│
                                      │    └── SORT / ByteTrack / BoTSORT          │
                                      │    ▼                                       │
                                      │  counters.update(objects) → [CrossingEvent]│
                                      │    ├── LineCounter   (наклонная линия A→B) │
                                      │    └── ZoneCounter (полигон)               │
                                      └──────────────────────────────────────┘
                                              │ events
                                       event_log.jsonl        report.md
                                              GUI: gui.py / gui_qt.py (overlay через counter.draw)
```

Общие принципы, объединяющие модули:
- **Разделение headless/GUI.** `Pipeline.step()` — чистая логика, возвращает события;
  рисование overlay делает вызывающий код через `counter.draw(frame)`. Это позволяет
  гонять конвейер без окна и переиспользовать его в GUI.
- **Нормализованные координаты 0..1.** Линии, полигоны, size-точки заданы в долях кадра;
  масштабирование в пиксели выполняется в конструкторе счётчика/детектора после открытия
  источника (когда известно фактическое w/h).
- **Кроп ROI на уровне источника.** При наличии `processing.roi` весь остальной код видит
  ROI-кадр как «полный»; все координаты настроек и событий — в «системе ROI».
- **Отказоустойчивость вывода.** Ошибки записи debug-кадров/отчёта не роняют конвейер,
  только warning в stderr.

---

## 2. `__main__.py` — CLI

Точка входа: `python -m visio_people_counter <probe|count|calibrate>`.

### Структура
- `build_parser()` — argparse с тремя подкомандами и кастомным `_FMT` (RawDescriptionHelp)
  для красивого epilog'а. `main(argv=None)` диспетчеризует через `args.func`.
- `--version` читает `__version__` из пакета.

### Команды
- **`probe --video PATH|URL`** — `ffprobe_info()` (разрешение/fps/длительность/кодек),
  определяет тип (`_kind_label`: m3u8/hls → hls, иначе URL или file); для локального файла
  дополнительно открывает `cv2.VideoCapture` и читает первый кадр. Возвращает rc=1 при ошибке.
- **`count --config … [--video …] [--bench] [--gui [--speed --scale --backend]]`** — основной
  сценарий. `--config` обязателен (дефолт `config.yaml`); `--video` переопределяет `video.path`.
- **`calibrate …`** — GUI-калибровка линии/зоны/size-точек.

### Ключевые детали дизайна
- **Ленивый импорт Qt.** PySide6 тянется только по запросу (`--backend qt`):
  `_pyside_available()` проверяет `importlib.util.find_spec("PySide6")`, а реальный импорт
  `from gui_qt import run_count_qt / run_calibration_qt` — внутри ветки `if backend == "qt"`.
  Это значит, что headless-прогон работает без установленного Qt.
- **Выбор бэкенда — чистая функция `choose_gui_backend(requested, available)`** → `"qt"|"cv2"|None`.
  Логика: `None` → qt если PySide6 есть, иначе cv2; явный qt без PySide6 → None (ошибка);
  явный cv2 всегда рабочий. Функция тестируется без импорта GUI.
- **CLI-типы как валидаторы:** `_speed_value` (0.25..8), `_scale_value` (0.05..8),
  `_cache_frames_value` (>=1) — через `argparse.ArgumentTypeError`, дают валидные сообщения об ошибке.

### Заметки
- В `cmd_count` порядок проверок: загрузка конфига → проверка пути/существования файла → build
  pipeline → GUI или headless. Это позволяет отловить опечатки в конфиге до старта ffmpeg/cv2.

---

## 3. `config.py` — загрузка и валидация конфига

Dataclass'ы с дефолтами, **точно соответствующими** `config.example.yaml`. Схема 1:1 по блокам:
`video / processing / motion / objects / size_profile / tracker / counters / output / debug`.

### Механика валидации
Набор маленьких хелперов с путём `блок.ключ` в сообщении об ошибки:
- `_get_str/bool/int/float/enum/range/pair/pair_list/size_points/roi/morph_kernel` — каждый
  проверяет тип, диапазон и возвращает дефолт при отсутствии ключа.
- `_check_unknown_keys(d, allowed, path)` — отклоняет неизвестные ключи (защита от опечаток).
- **Bool ≠ int.** В `_get_int/_get_float`/`_get_optional_int` явно: `isinstance(v, int) and not isinstance(v, bool)`.
  Критично, т.к. в Python `True == 1`.
- `_get_enum` ограничивает множество значений (например `motion.method ∈ {mog2, knn}`,
  `tracker.type ∈ {sort, bytetrack, botsort}`).

### Особенности блоков

**VideoConfig / HlsConfig.** `type ∈ {file, hls}`; блок `hls` — watchdog переподключения:
`reconnect_attempts` (0 = бесконечно, -1 = нельзя), `reconnect_backoff_s`, `bad_read_threshold`.

**ProcessingConfig.** Самое ёмкое по семантике:
- `effective_fps` (0 = нативный) — управляет пропуском кадров у файла (`_file_skip`) и `-r` у ffmpeg.
- `frame_start` / `frame_end` — **интервал кадров 0-based** (обе границы включительно), в котором
  работают трекер + счётчики; детектор работает на всех кадрах (обучение фона MOG2). Семантика:
  обе null = всё видео; только start = с него до конца; только end = от начала до `end` включ.;
  обе = `[start, end]`. Проверка `frame_start <= frame_end`.
- `roi` — прямоугольник `[x, y, w, h]` в долях ПОЛНОГО кадра. Валидация: 4 числа ∈ (0..1],
  `x + w <= 1`, `y + h <= 1`.

**MotionConfig.** `method={mog2,knn}`, `history=500`, `var_threshold=32` (mog2),
`dist2_threshold=4.0` (knn), `detect_shadows`, `shadow_threshold=200`, морфо-ядра
`morph_open=(3,3)`, `morph_close=(9,15)` — ядра целые нечётные > 0 (`_get_morph_kernel`).

**ObjectsConfig.** Фильтры blob'ов: доли площади кадра `min/max_area_fraction`,
`min_bbox_side_px=8`, `aspect_ratio_range=(0.2, 2.5)`, `min_fill=0.25` (area/bbox_area),
`min_lifetime_frames=3`.

**SizeProfileConfig.** `enabled`, контрольные точки `[x_доля, y_доля, h_доля]`,
`k_min=0.2`, `k_max=4.0`. Миграция старых пар `[x, h]` → `[x, 0.5, h]` с warning в stdout;
при сохранении пишутся только тройки (`h > 0`).

**TrackerConfig.** `type={sort,bytetrack,botsort}`, `lost_track_buffer=60`,
`minimum_consecutive_frames=2`, `minimum_iou_threshold=0.3`.

**LineCounterConfig / ZoneCounterConfig.** Унифицированы полями антидубля: `cooldown_s`,
`min_global_gap_s`; у линии дополнительно `a/b` (нормализованные), `count_mode={both,total}`,
`buffer_width_scale`; у зоны — `polygon` (>= 3 точек) и дефолтный `count_mode=total`.
Фабрика в `config.from_dict` проверяет уникальный non-empty `id`, тип `line|zone`, чужие ключи.

**OutputConfig / DebugConfig.** `output`: `events_jsonl`(="" = не писать), `report_path`,
`summary_interval_s`(0 = только финал), `final_summary`. `debug`: флаги overlay,
`save_debug_frames_dir`, `debug_frame_step=10`.

### Корневой Config
- `load(path)` — читает YAML (`yaml.safe_load`), при битом → `ConfigError("битый YAML")`,
  собирает `from_dict`.
- `default()` — дефолты с одним line-счётчиком `main_line` (нужен для калибровки).
- `save(cfg, path)` — сериализует в YAML (`sort_keys=False`, allow_unicode).

### Заметки
- Валидация строгая и предсказуемая — хорошее поле для unit-тестов (`test_config.py`).

---

## 4. `video_source.py` — источники кадров

Абстракция `VideoSource` (ABC) с методом `read() -> Frame | None`. Один общий тип кадра:

```python
@dataclass
class Frame:
    image: np.ndarray          # BGR uint8 (h, w, 3)
    index: int                 # порядковый номер в текущем проходе (0-based)
    t_wall: float              # time.monotonic() получения
    t_video: Optional[float]   # время в таймлайне видео (сек), если известно
```

### `VideoSource` ABC
Публичные свойства: `width/height/fps/duration`, абстрактные `open/read/close`.
`seek(seconds)` по умолчанию возвращает False (случайный доступ недоступен для pipe/HLS);
переопределяется в FileSource. Контекстный менеджер (`__enter__/__exit`).

### FileSource (`cv2.VideoCapture`)
- `open()`: открывает, читает w/h/fps; при ROI — NumPy-срез через `roi_crop_pixels`, и
  `_w/_h` становятся размером ROI (весь конвейер работает от него).
- `read()`: `cap.read()`; в режиме loop после EOF — перемотка `CAP_PROP_POS_FRAMES=0`.
  ROI-срез делается копией (`frame[...].copy()`), чтобы не резать исходный буфер OpenCV.
  `t_video = index / fps` (или None, если fps неизвестен).
- `seek(sec)`: `CAP_PROP_POS_MSEC`, точность «до ключевого кадра»; `_index = round(seconds*fps)`.
- `duration`: `FRAME_COUNT/fps`, фолбэк `CAP_PROP_DURATION/1000`.

### FfmpegPipeSource (HLS/URL)
- `open()`: `ffprobe_info()` → разрешение/fps/кодек. Затем:
  - ROI: пиксели считаются один раз из разрешения ffprobe до старта ffmpeg (`crop=w:h:x:y`);
    если срез неприменим — фолбэк без кропа + warning.
  - Даунскейл `-s`: только если итоговый `(w,h) != base_size`, высота чётная (bgr24/rawvideo любят чётные).
  - `-r N` при `effective_fps > 0`.
- `_build_cmd()`: `ffmpeg -hide_banner -loglevel error -rw_timeout 5000000 -i … [crop] [-r N] [-s S] -f rawvideo -pix_fmt bgr24 pipe:1`.
  `-rw_timeout` (~5 с) не даёт `read` висеть вечно при обрыве сети.
- **Watchdog переподключения** — центральная часть надёжности HLS:
  - `_handle_bad_read()`: неудачное чтение (короткое/пустое или ffmpeg с ненулевым кодом) инкрементит `_fails`;
    до `bad_read_threshold` — ждём; при достижении порога → `_kill_proc()` + backoff (`sleep`) + respawn + хук `on_reconnect`.
  - `_can_reconnect()`: `<0` → нельзя, `0` → бесконечно, иначе лимит по `_restarts`.
  - EOF файла через pipe: пустое чтение и `rc == 0` → `_eof = True`, graceful close.
- `read()`: читает ровно `w*h*3` байт; при коротком чтении — проверка кода/обработка разрыва, возвращает None
  (pipeline понимает `None` как «недоступен сейчас», а не EOF, пока `exhausted == False`).
- `exhausted`: True, если `_eof` или `_stopped` — дальше read вернёт None.

### `ffprobe_info()` + `roi_crop_pixels()`
- `ffprobe_info` — чистая обёртка над subprocess: парсит JSON, извлекает width/height/fps (из
  `r_frame_rate` → `/`-разделение → float)/codec/duration/format. Ошибки → `VideoSourceError`.
- `roi_crop_pixels` — чистая функция округления ROI в пиксели с защитой от вырожденных 0-срезов:
  координаты зажимаются в границы, w/h ∈ [1 … край].

### Заметки по надёжности
- `_close_stdout()` и `__del__` — страховка от `ResourceWarning`/утечки процесса ffmpeg.
- Потенциальный риск: при очень коротком чтении (частичный кадр) `np.frombuffer` даст буфер
  меньше нужного → reshape упадёт с ValueError; код читает только `frame_bytes`, но не проверяет
  длину raw перед reshape (см. §13).

---

## 5. `motion_detector.py` — выделение движения

Обёртка над `cv2.createBackgroundSubtractorMOG2/KNN`. Возвращает `list[Blob]`:

```python
@dataclass(frozen=True)
class Blob:
    x, y, w, h: int     # bbox в px исходного кадра
    area: int           # число пикселей компоненты (не w*h)
    cx, cy: float       # центр bbox
```

### `MotionDetector.__init__`
- Выбирает субтрактор по `cfg.motion.method`. MOG2: `history`, `varThreshold`, `detectShadows`.
  KNN: `history`, `dist2Threshold`, `detectShadows`. (Кнопка «неизвестный метод» — защита от ручного конфига.)
- `_detect_shadows` истинен **только для MOG2** (KNN не помечает тени значением 127).
- Ядра морфологии через `getStructuringElement(MORPH_RECT, …)`.
- Хранит `last_mask` — маску последнего detect() для GUI-overlay (`cfg.debug.show_mask`).

### `detect(frame_image, size_profile)`
Последовательность:
1. `fgmask = self._bg.apply(frame_image)`.
2. Обрезка теней (MOG2): `np.where(fgmask >= shadow_threshold, 255, 0)` — при threshold ≤ 127 остаётся строгий foreground ==255.
3. `MORPH_OPEN(open_kernel)` → `MORPH_CLOSE(close_kernel)`. Результат в `last_mask`.
4. `connectedComponentsWithStats(mask, 8)` → `(n, labels, stats, centroids)`. Пиксели на компоненту —
   `np.bincount(labels.ravel(), minlength=n)` (быстрее повторного `cv2.countNonZero`).
5. `filter_blobs(...)`:

### `filter_blobs()` — порядок проверок
1. **Стора bbox** ≥ `min_bbox_side_px` (субпиксельный мусор отсекается).
2. **Aspect ratio** ∈ `[aspect_ratio_range]` (машины, полосы отпадают).
3. **Fill** = area/(w·h) ≥ `min_fill` (россыпь шума, ветка — bbox почти пустой).
4. **Площадь** через `min_area_at/cx,cy <= area <= max_area_at`:
   - если передан `SizeProfile` → адаптивные пороги в точке центра;
   - иначе глобальные доли `min/max_area_fraction * (w·h)` (замкнутые lambdas).

Возвращает отфильтрованные `Blob`. **`min_lifetime_frames` НЕ здесь** — его применяет конвейер.

### `reset()`
`self._bg.clear()` — сброс модели фона; вызывается pipeline при reconnect, чтобы MOG2 заново обучился.

### Заметки
- Детектор работает на каждом кадре (в т.ч. вне интервала подсчёта) — это сознательно, для обучения фона.
- Адаптивная площадь (`SizeProfile`) привязана к центру blob'а; если центр вне зоны действия профиля — clip.

---

## 6. `size_profile.py` — адаптивные пороги площади

Строит поверхность ожидаемой высоты человека `h(x, y)` и выводит пороги площади как функцию точки кадра.

### `HeightSurface` (чистый класс, numpy)
Работает в нормализованных координатах 0..1; `height_at(nx, ny)` клампит x,y к [0,1].
Подгонка по контрольным точкам `(x, y, h)`:
- **n == 1** → константа.
- **n == 2** → линейная интерполяция вдоль отрезка: проекция точки на отрезок (t=clamp(0..1)),
  за краями — крайнее значение; вырожденный отрезок → константа h0.
- **n == 3** → плоскость `a + b·x + c·y` (least squares; для неколлинеарных — точное решение).
- **n >= 4** → биквадрик `a + b·x + c·y + d·x·y` (numpy `np.linalg.lstsq`).

### `fit_height_surface(points)`
Чистая функция-фабрика, возвращает `HeightSurface`; пустой список → ValueError.

### `SizeProfile.__init__`
- Глобальные пороги: `global_min_area = min_area_fraction * w*h`, аналогично max — используются при отключённом профиле.
- `pts = [p for p in control_points if p[2] > 0]` (только с h > 0); `_surface = fit_height_surface(pts)` или None.
- `adaptive` property: True iff `_surface is not None`.

### Публичные методы (y опционален; None = середина кадра — назад-совместимость)
- `person_height_px(x, y)`: без профиля → высота кадра h (нейтральная величина, влияет только на buffer);
  с профилем → `max(1.0, surface.height_at(x/w, y/h) * h)`.
- `min_area_at/cx,cy` = `k_min * h_px²` / `k_max * h_px²` (адаптивно) или глобальные.
- `buffer_width_px(x, y, scale=0.75)` = `scale * person_height_px` — ширина буфера вокруг линии.

### Заметки
- Площадь человека масштабируется с квадратом ожидаемой высоты: ближние объекты (выше в px) допускают большую площадь blob'а.
- `_MIN_PERSON_HEIGHT_PX=1.0` защищает от деления/нормализации на ноль.

---

## 7. `tracker_adapter.py` — трекеры SORT/ByteTrack/BoTSORT

Обёртка над пакетом `trackers` (`SORTTracker`, `ByteTrackTracker`, `BoTSORTTracker`) + `supervision`.

### Ключевой инсайт (задокументирован в docstring)
Конвейер считает по **motion-blob'ам**, у которых нет confidence. Поэтому:
- Передаётся `sv.Detections(xyxy=..., confidence=None)` → пакет трактует всё как уверенность 1.0
  (`trackers.utils.detections.default_confidences`), и потому:
  - порог активации трека всегда пройден — трек на任何 unmatched det;
  - двухстадийная ассоциация ByteTrack/BoTSORT (high/low) отключается → **ByteTrack ≡ SORT**;
  - BoTSORT без кадров CMC тоже ≈ SORT (шаг CMC молча пропускается при `frame=None`).
- Единственное реальное различие типов остаётся — семантика буферов/id, но на blob-уровне они эквивалентны.

Это осознанная деградация: для motion-tracking не нужны heavy-механики ByteTrack/BoTSORT, а выбор `type`
в основном влияет на то, как легко переключиться, если в будущем понадобятся кадры (задача CMC качки).

### `TrackedObject`
```python
@dataclass
class TrackedObject:
    track_id: int       # -1 = неподтверждённый трек
    x, y, w, h: int     # bbox px
    cx, cy: float       # центр
    age_frames: int     # кадров подряд с подтверждённым id (для -1 = 0)
```

### `TrackerAdapter.__init__(cfg, frame_rate=None)`
- `frame_rate`: effective_fps или fallback `DEFAULT_FRAME_RATE=15`.
- `_build_tracker()`: общие параметры `lost_track_buffer`, `frame_rate`, `minimum_consecutive_frames`;
  у BoTSORT один порог из конфига применяется к first/unconfirmed ассоциации, second-stage — дефолт 0.5.

### Масштабирование lost-track buffer
`maximum_frames_without_update = max(1, ceil(frame_rate/30 * lost_track_buffer))`. Т.е. буфер задаётся
в кадрах 30 FPS и правильно пересчитывается под реальный fps — поэтому fps обязателен (иначе `ValueError`).

### `update(blobs)`
- Есть blob'ы → `xyxy` numpy float32 → `sv.Detections(xyxy, confidence=None)`; нет → `sv.Detections.empty()`.
  **Пустые кадры ОБЯЗАТЕЛЬНЫ** передавать — иначе Kalman/счётчик пропусков не продвинется и старые треки
  не закроются.
- `_tracker.update(dets)` → `result.tracker_id`; для каждого blob вычисляется `age_frames` через словарь `_age`.
- Неподтверждённые (`tid < 0`) получают age=0.

### `set_frame_rate(fps)` / `reset()`
- `set_frame_rate`: пересоздаёт трекер с новым fps (пакет вычисляет buffer в конструкторе, «на лету» нельзя);
  валидация `fps > 0`; сбрасывает `_age`.
- `reset()`: `_tracker.reset()` + очистка `_age` — при reconnect/id заново с нуля.

### Заметки
- Возвращённый список по порядку входных blob'ов (zip), что упрощает сопоставление в pipeline.
- `maximum_frames_without_update` и `tracker_type` — property для отладки/тестов.

---

## 8. `line_counter.py` — счётчики пересечения

Реализация по §2 отчёта `04-implementation-notes.md`. Общий интерфейс `BaseCounter`:
`update(objects, t_wall, *, t_video, frame_index) -> [CrossingEvent]`, property `counters`, `draw(frame)`.

### `CrossingEvent` (frozen dataclass)
`counter_id, direction ("in"/"out"), track_id (!= -1), t_wall, x_px, y_px, frame_index, t_video`.
`to_dict()` → plain-dict для JSONL (`ts_wall` UTC ISO + поля).

### Геометрия пересечения (вспомогательные функции)
- `_cross(v, w)` — z-компонентное 2D-векторное произведение.
- `_segment_intersection(p1,p2,p3,p4)` — пересечение **отрезков** (не прямых); возвращает точку или None,
  с допуском на дискретизацию (`t,s ∈ [-1e-6, 1+1e-6]`).

### `_TrackState`
`last_pos`, `last_event_t` (wall последнего счёта трека), `last_seen_t`, `last_inside` (только зона).

### Направление линии
`d(P) = cross(B−A, P−A)` — знаковая сторона точки. Событие — когда отрезок **prev→cur** меняет знак d.
Конвенция: объект из стороны `d > 0` → `"in"`, из `d < 0` → `"out"`; для диагонали A=вл-верх, B=нп-право это
«движение по направлению A→B = in».

### LineCounter
- Конструктор: нормализованные `a/b` → px на w/h; `_ab = B−A`, `_len2 = |ab|²`; проверка «нулевой линии» (A==B).
- `_ref_scale(ip)`: h_local из SizeProfile (`person_height_px`) если адаптивен, иначе `0.5 * line_len`.
- `update()`:
  - пропуск `track_id == -1` (не считаем и не храним);
  - первый раз — создаём `_TrackState(last_pos=cur)`; далее — `_check_cross`;
  - **last_pos обновляем ВСЕГДА** (даже в cooldown), иначе после паузы первый шаг станет «длинным» отрезком
    и сломает геометрию пересечения.
- `_check_cross()`:
  1. стороны prev/cur: `d_prev==0 || d_cur==0 || d_prev*d_cur>0` → None (на одной стороне).
  2. точка пересечения отрезков; параметр `u = ((ip−A)·AB)/|AB|²`; если `u ∉ [-0.05, 1.05]` → пересёк продолжение — не считаем.
  3. антидубль: per-track cooldown (`dt < cooldown_s` → None); буфер вокруг линии
     (`dist_to_line = |d_cur|/line_len <= buffer_width_scale * ref` → None).
  4. `_global_gap_ok(t_wall, ip, ref)`: если до последнего события счётчика < `min_global_gap_s` и расстояние
     ≤ `0.5*ref` → отбросить (разрезает сдвоенные blob'ы одного человека).
  - direction = `"in"` если `d_prev > 0`.

### ZoneCounter
- Конструктор: полигон из нормализованных точек → px; проверка >= 3 точек; `_poly_cv` для cv2.
- `_ref_scale_value`: длина самой длинной стороны полигона (масштаб для global gap).
- `_inside(pt)` = `cv2.pointPolygonTest(poly, pt, False) >= 0`.
- `_boundary_point(prev, cur)` — первая точка пересечения prev→cur со сторонами полигона.
- **Семантика "in" (задача 09):** снаружи→внутрь ИЛИ появление нового трека уже внутри:
  - новый трек: `last_inside = _inside(cur)`; если True → `_check_boundary(..., inside_now=True)`.
    prev == cur, поэтому точка события = позиция объекта; anti-dub через global gap (per-track cooldown не применяется — last_event_t None).
  - существующий: смена `last_inside` → `_check_boundary`.
- `_check_boundary()`: per-track cooldown + `_global_gap_ok`; direction = "in" если inside_now, иначе "out".

### Антидребезг в BaseCounter
- `_register_event(ev)`: инкремент in/out + запоминание `_last_event_t/_last_event_pt` (глобальный антидребезг счётчика).
- `_global_gap_ok(t_wall, pt, ref_scale)`: True, если прошло ≥ `min_global_gap_s` ИЛИ расстояние до последнего события > `0.5*ref_scale`.
- `_prune_stale(t_wall)`: удаляет состояния треков, невидимых > `_STATE_PRUNE_AGE_S=60 с` (анти-утечка памяти).
- `reset()`: чистит `_tracks`, **не обнуляя** in/out (критично при reconnect — счётчики сохраняются).

### `count_mode`
`"both"` → `{in, out, total}`, `"total"` → `{total}`. События обоих направлений логируются всегда;
`count_mode` влияет только на представление `counters`.

### `draw(frame)`
LineCounter рисует зелёную линию + текст (`put_text`, кириллица через Pillow); ZoneCounter — жёлтый полигон.

### Заметки
- Антидубль продуман до мелочей: 3 уровня для линии (cooldown + буфер + global gap) и 2 для зоны,
  с учётом того, что у зоны нет buffer_width_scale в конфиге (по ТЗ задачи 14).
- `min_global_gap_s` обязателен даже для новых треков в зоне — иначе два трека рядом дадут дубль.

---

## 9. `pipeline.py` — главный конвейер

Связывает все модули и управляет циклом обработки.

### `Pipeline.__init__(config, bench=False)`
Принимает путь к YAML **или** готовый `Config` (CLI передаёт объект, чтобы переопределить `video.path` `--video`).
Состояние run: `warmup_done`, `frames_processed`, `report_events`, fps (avg/ema), lag (last/max), `reconnects`, флаги bench.

### `_effective_frame_rate()`
`eff = processing.effective_fps`; >0 → он; иначе нативный fps источника; неизвестен (HLS без таймкода) →
FALLBACK_FPS=15 + лог-строка. Именно этот rate идёт в трекер (для верного масштабирования buffer).

### `build()` — сборка (идемпотентна)
1. Проверка `video.path`; выбор источника (`FileSource` для файла, `FfmpegPipeSource` для hls/URL).
   ROI передаётся источнику → кроп на уровне источника.
2. `source.open()` → w,h; логирование (ROI, разрешение, fps, effective_fps).
3. Порядок создания: `SizeProfile(cfg.size_profile, cfg.objects, w, h)` → `MotionDetector(cfg)` →
   `TrackerAdapter(cfg, frame_rate=eff)` → `build_counters(cfg.counters, w, h, size_profile)` → `EventLog(cfg)`.
4. Лог каждого счётчика (id/type/count_mode).

### `_on_reconnect()` — хук watchdog
`reconnects += 1`; сброс `detector.reset()`, `tracker.reset()`, `c.reset()` у всех счётчиков
(сбрасываем состояния треков, т.к. id заново с нуля); `warmup_done=False` (фон снова «не знаком»).
**Накопленные in/out НЕ обнуляем.**

### `step(frame) -> [CrossingEvent]` — один кадр
1. `t0`; `blobs = detector.detect(frame.image, size_profile)` (на любом кадре).
2. Если `counting_active(index)`: `objects = tracker.update(blobs)`; **вне интервала** → `objects=[]`
   (трекер не обновляем, старые треки не «проживают» до старта; счётчики их не видят).
3. Если в интервале: фильтрация по `age_frames >= min_lifetime_frames` (анти-вспышка/дубль),
   затем `counter.update(...)` для каждого счётчика → события.
4. События: в `report_events`, `event_log.log_events()`, лог в stdout (`СОБЫТИЕ …`).
5. Bench-тайминги detect/track/count/frame (если `bench=True`).
6. Debug-кадры (`save_debug_frame` каждый N-й, по `cfg.debug`).
7. Warmup: первые `history/2` кадров — ставим `warmup_done`, лог о прогреве.
8. Lag: `(t_wall − ref_wall) − (t_video − ref_video)` при известном t_video; max_lag.
9. `frames_processed += 1`.

### `counting_active(index)` — чистая функция
Обе границы inclusive, 0-based; null = без границы. Модульная версия `_counting_active` тестируется отдельно.

### `_file_skip()`
Для FileSource при `effective_fps > 0`: `N = round(native/eff)`, обрабатываем каждый N-й кадр.
Для HLS даунскейл/fps задаёт сам ffmpeg (skip=1).

### `run() -> int` — цикл до EOF/stop
- `build()` если не собран.
- `interval = summary_interval_s`; `skip = _file_skip()`.
- **SIGINT → graceful stop**: устанавливает `_stop=True` (только на главном потоке, через `signal.signal`,
  старый handler восстанавливается в `finally`).
- Цикл: `read()`; None + FfmpegPipeSource не exhausted → sleep(0.2) backoff и continue (не выходим);
  иначе break при исчерпании. Skip кадров по `_file_skip`. `step(frame)`. EMA fps из instantaneous dt.
  Периодические сводки (`print_summary`).
- В `finally`: `duration_s`, `avg_fps`, `_print_final(reason)`, `_write_report(reason)` (до close — нужен fps/duration),
  `close()`, восстановление SIGINT. Возвращает 0.

### `_print_final()` / `_write_report()`
- Финальная сводка: причина, длительность, обработано кадров, интервал кадров, ROI, fps (+lag), reconnects,
  сводка event_log; при bench — таблица p50/p95 detect/track/count/frame (через `_percentile`).
- Markdown-отчёт через `choose_report_path` + `build_report`; путь None (HLS без явного report_path) → не создаётся.
  Ошибка записи = warning в stderr, не exception.

### Заметки
- Разделение «детектор на всех кадрах, трекер/счётчики — только в интервале» экономит ресурсы вне окна интереса,
  но скорость прогона определяется детектором (MOG2 обучается всегда).
- Graceful-обработка повсюду: SIGINT, reconnect, EOF — всё через try/finally с восстановлением состояния.

---

## 10. `event_log.py` и `report.py` — выходные данные

### EventLog
- Конструктор из `OutputConfig | Config`: открывает `output.events_jsonl` на дописывание (каталог создаётся);
  пустой путь → файл не открывается, агрегаты всё равно считаются.
- `log(event)` / `log_events(events)`: инкремент per-counter in/out + запись JSONL (`json.dumps(..., ensure_ascii=False)`), flush после каждой строки.
- Свойства `counters` (per-counter `{in,out,total}`) и `totals` (глобальные).
- `summary()` — человекочитаемая строка для stdout (сортировка по id счётчиков).
- Контекстный менеджер; `close()` закрывает файл.

### Report
- `choose_report_path(cfg)`: явный `report_path` > авто `<видео>.report.md` (только для локального файла,
  без суффикса → `<имя>.report.md`); HLS/URL без пути → None (отчёт не создаётся).
- `build_report(cfg, events, meta)` — **чистая функция** (тестируется без pipeline): markdown с шапкой
  (дата запуска, длительность/кадры/fps, интервал кадров, ROI + примечание про систему координат),
  таблицей «Итоги по счётчикам» со строкой ВСЕГО и таблицей событий per счётчика (# | время(с) | направление | track_id | x_px | y_px | кадр).
- `write_report(path, text)` — запись с созданием родительских каталогов; существующий файл перезаписывается.
- `_fmt_num` убирает хвост `.0` у целых (25 → 25, 13.6 → 13.6).

### Заметки
- Отчёт и JSONL дополняют друг друга: markdown — человекочитаемый итог, JSONL — машиночитаемые сырые события.
- `t_video` в отчёте форматируется до мс; при None → «н/д».

---

## 11. `gui.py` / `gui_qt.py`, `calibrate*.py`, `text_overlay.py` — дополнительно

Эти модули не участвуют в headless-конвейере, но важны для полноты картины:

### `text_overlay.py`
`put_text(frame, text, pos, size_px, color)` — рисует текст с кириллицей через Pillow (cv2.putText её не умеет);
при отсутствии Pillow/шрифта — фолбэк в ASCII. Единый центр для overlay счётчиков.

### `gui.py` (`GuiPlayer`) — cv2-окно
- `available()` / `unavailable_reason()` — проверка доступности GUI без импорта (DISPLAY/Wayland + наличие зависимостей).
- Игрок окна: гоняет `Pipeline.step(frame)` по одному кадру, рисует overlay через `counter.draw()`,
  управляет скоростью (`--speed`) и масштабом (`--scale`), пресеты `,`/`.`.

### `gui_qt.py` — PySide6-окно
- `run_count_qt(pipe, speed, initial_scale)` / `run_calibration_qt(...)`. Отдельный бэкенд;
  в последних коммитах добавлен `paintEvent` для VideoCanvas и скрытие старой панели «на кадре».

### `calibrate.py` / `calib_controller.py`
- Разделение: логика калибровки — в `CalibrationController`, тонкий драйiver окна (cv2/Qt) — в `calibrate.py`.
- `resolve_calibrate_config(video, config)`: явный --config → он (вход и цель); без опции → ищет
  `<имя_видео>.config.yaml` рядом с видео (нашёл — вход И цель, правки не затирают другие блоки); иначе дефолты.
- Кнопки/действия: добавление/удаление точек линии/зоны/size, seek по времени, show-all blobs, кэш кадров (`--cache-frames`, ~6 МБ/кадр при 1080p).

### Заметки
- GUI-модули импортируются лениво (см. `__main__.py`), чтобы не тянуть Qt без запроса.

---

## 12. Ключевые дизайн-решения

1. **Headless/GUI разделение через `step()` + `draw()`.** Конвейер — чистая логика, окно лишь рисует overlay.
   Позволяет гонять на сервере без X и переиспользовать тот же pipeline в GUI.
2. **Нормализация 0..1 + масштабирование в конструкторе.** Конфиг переживает смену разрешения камеры;
   счётчики пересоздаются при смене w/h (pipeline.build).
3. **Кроп ROI на уровне источника.** Все координаты живут в «системе ROI»; события и отчёт указывают, что
   x_px/y_px — относительно кропа.
4. **motion-blob'ы без confidence → ByteTrack ≡ SORT.** Осознанная деградация heavy-трекеров; выбор `type`
   оставлен на будущее (возможность передать кадры для CMC BoTSORT).
5. **Адаптивная площадь через поверхность высоты человека.** Ближе к камере человек крупнее → допустимая площадь blob'а растёт с квадратом высоты; отпадает необходимость ручной перенастройки под камеру.
6. **Многоуровневый антидубль** (cooldown + буфер линии + global gap + `min_lifetime_frames` в конвейере).
7. **Graceful-обработка повсюду:** SIGINT, reconnect HLS, EOF, ошибки записи — через try/finally и warning'и.
8. **Интервал кадров экономит ресурсы**, но детектор всегда активен (обучение фона) — это задокументированный компромисс скорости/точности.

---

## 13. Заметки по надёжности и потенциальные риски

- **Короткое чтение ffmpeg (`video_source.py`).** `read()` читает ровно `w*h*3` байт; при ошибке вызывает
  `_handle_bad_read()` и возвращает None, но перед `np.frombuffer(raw).reshape(...)` длина raw не перепроверяется
  в «успешном» ветке — теоретически частичный кадр вызовет ValueError reshape. На практике watchdog ловит смерть процесса,
  а полный фрейм от ffmpeg идёт порциями по `w*h*3`. Рекомендуется defensive-проверка `len(raw) == frame_bytes` перед reshape.
- **HLS: `t_video = index/fps`** при неизвестном fps → None; lag тогда не считается (требует известного t_video). Корректно, но в отчёте время будет «н/д».
- **Warmup после reconnect.** При переподключении `warmup_done=False`, и первые `history/2` кадров события логируются,
  но трекер ещё не стабилен — возможен всплеск ложных событий сразу после reconnect. Практически приемлемо (счётчики in/out не обнуляются).
- **`_segment_intersection` на продолжении линии.** Допуск `u ∈ [-0.05, 1.05]` намеренный (дискретизация), но для очень короткой линии буфер может «ловить» объекты мимо — зависит от `buffer_width_scale`.
- **Memory: `_tracks` в счётчиках** прунятся по `last_seen_t > 60 с`, что ограничивает утечку при долгом стриме; при reconnect id сбрасываются, но состояния чистятся через `reset()`.
- **Bool vs int в конфиге** тщательно защищён — надёжная валидация.

### Рекомендации (некритичные)
- Добавить defensive check длины raw перед reshape в `FfmpegPipeSource.read()` (см. выше).
- При желании — логировать «ложное событие после reconnect» с флагом warmup для диагностики.

---

## Итог

Пакет `visio_people_counter/` — аккуратно спроектированный motion-based конвейер подсчёта трафика:
строгая валидация конфига (`config.py`), надёжные источники кадров с watchdog'ом HLS
(`video_source.py`), классическая CV-психология движения (`motion_detector.py`, `size_profile.py`),
осознанная деградация трекеров под motion-blob'ы (`tracker_adapter.py`) и проработанная геометрия
пересечения с многоуровневым антидублем (`line_counter.py`). Headless/GUI разделение, нормализованные
координаты и отказоустойчивый вывод делают модули независимо тестируемыми (см. `tests/`), а логика
интервала кадров, warmup'а и reconnect'ов отражена в документации модулей.
