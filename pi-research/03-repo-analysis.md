# Задача 03: Клонирование и архитектурный разбор референс-репозиториев

Контекст: отчёты 01 (методы) и 02 (отбор решений). Проект — подсчёт человеческого трафика,
ввод HLS URL или файл, уличная камера с широким углом; трекинг — защита от двойного счёта.
Клоны лежат в `references/repos/` (в .gitignore), shallow `--depth 1`.

---

## 1. Клонирование: что склонировалось, HEAD

| Репо | Статус | HEAD (shallow) | Дата коммита | Лицензия |
|---|---|---|---|---|
| JovanSk/yolov8-line-crossing-counter | OK | `49ed0b5` | 2026-06-09 | MIT (заявлено в README; файла LICENSE в репо нет) |
| roboflow/trackers | OK | `076ecef` | 2026-09-10 | Apache-2.0 (файл LICENSE есть, версия пакета 2.6.0) |
| FoundationVision/ByteTrack | OK | `d1bf019` | 2022-12-11 | MIT |
| Gupu25/PeopleCounter | OK | `df0057a` | 2020-03-04 | **отсутствует** — read-only референс логики, код НЕ копировать |

Все 4 клонировались успешно с первой/второй попытки (у ByteTrack и PeopleCounter первая
параллельная попытка обрывалась, повторный последовательный `git clone --depth 1` прошёл).

---

## 2. JovanSk/yolov8-line-crossing-counter

### Структура
Односессионный демо-проект: один основной скрипт + медиа.
- `process_and_count.py` — **215 LOC**, весь пайплайн и вся логика подсчёта;
- `README.md`, `requirements.txt`, `video/input.mp4`, `screenshots/`.

### Пайплайн
- **Ввод**: только локальный файл — `cap = cv2.VideoCapture(video_path)`, где
  `video_path = "video/input.mp4"` (строка на 37-й). URL/HLS не поддерживаются, никакого
  ffmpeg-pipe. Камера по индексу в принципе возможна (`cv2.VideoCapture`), но не используется.
- **Детекция + трекинг**: YOLOv8 через ultralytics, трекинг встроенный:
  `results = model.track(frame, persist=True)` — это BoT-SORT из коробки ultralytics
  (поэтому в requirements есть `lap`). Модель `YOLO("yolov8s.pt")`.
- Вывод: `cv2.VideoWriter` → `output/counting_video.mp4`, плюс отрисовка CCTV-хедера,
  счётчика-панели, угловых боксов (`draw_corner_box`).

### Логика подсчёта (ядро — ~17 строк)
Линия горизонтальная на середине кадра: `line_y = height // 2`. Направление одно
(сверху вниз). Антидубль — через два словаря: история предыдущих Y по track_id и set
уже посчитанных ID. Фрагмент, `process_and_count.py` (главный цикл):

```python
# tracking memory for y coordinates with track_id as a key
previous_positions = {}
count = 0
counted_ids = set()
...
for r in results:
    for box in r.boxes:
        if box.id is None:
            continue
        cls = int(box.cls[0])
        if cls != 0:               # только "person" (COCO class 0)
            continue
        track_id = int(box.id[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        ...
        # previous y position
        if track_id in previous_positions:
            prev_cy = previous_positions[track_id]
            # line cross detection
            if prev_cy < line_y and cy >= line_y:
                if track_id not in counted_ids:
                    count += 1
                    counted_ids.add(track_id)
        previous_positions[track_id] = cy
```

Разбор механизма:
- **пересечение**: «скок» центаоида через линию между двумя кадрами (`prev_cy < line_y and cy >= line_y`) — классический sign-change по Y;
- **антидубль**: `counted_ids` (ID после пересечения никогда не считается повторно) + требование, чтобы у трека был предыдущий кадр в памяти (`previous_positions`);
- **направление**: жёстко одно (вниз). На «вверх» и на общий трафик логика не расширяется — но добавится симметричным условием;
- **разные размеры объектов**: не обрабатываются специально — YOLO сам масштабирует, центаоид берётся от bbox;
- **слабости**: при потере track_id (ID switch) объект пересчитывается уже невозможен только потому что новый ID ещё не в `counted_ids` — т.е. ID switch = повторный счёт; нет cooldown по времени; линейная проверка «скок за кадр» пропускает, если объект скакнул далеко; `counted_ids` растёт бесконечно (для длинного стрима нужен прунинг).

### Трекер
BoT-SORT из ultralytics (`model.track(persist=True)`), параметры не выкручиваются вообще — всё по умолчанию. Для нашего проекта это «чёрный ящик».

### Зависимости (`requirements.txt`)
```
ultralytics>=8.4,<9   # тянет за собой torch/torchvision, heavy (GPU-стек)
opencv-python>=4.11,<5
numpy>=1.26,<2
torch>=2.1,<3         # тяжёлое: ~2 ГБ с CUDA-колесами
torchvision>=0.16,<1
lap>=0.5              # Hungarian algorithm для встроенного трекера
```
Тяжёлые: torch/torchvision/ultralytics (GPU/DL). Всё остальное — стандартный OpenCV-стек.

### Лицензия и ограничения
MIT по README. Практически ограничений нет; но это учебный демо-скрипт: нет CLI, нет HLS,
один счётчик, одно направление, оффлайн-прогон. Берём архитектуру, не код.

---

## 3. roboflow/trackers (v2.6.0)

### Структура
Полноценная библиотеку (`src/trackers/`, ~11 000 LOC), clean-room ре-имплементации трекеров:
- `core/base.py` (695 LOC) — общий каркас `BaseTracker`: Kalman, таймстемпы, prune;
- `core/bytetrack/tracker.py` (314) — `ByteTrackTracker`;
- `core/botsort/tracker.py` (490) — `BoTSORTTracker` (+ CMC);
- `core/sort/`, `core/ocsort/`, `core/cbiou/`, `core/mcbyte/` — остальные;
- `utils/`: `kalman_filter.py`, `iou.py` (IoU/GIoU/DIoU/CIoU), `state_representations.py`;
- `io/video.py` — `frames_from_source()`; `cli/` — track/tune/eval; `tune/` — автоподбор параметров через Optuna; `demo/app.py` (Gradio).

### Пайплайн / ввод стрима
Сама библиотека трекером является и видео не читает, но есть утилитарный итератор:

```python
# src/trackers/io/video.py
def frames_from_source(source: str | Path | int) -> Iterator[tuple[int, np.ndarray]]:
    """Yield numbered BGR frames from video files, webcams, network streams,
    or image directories."""
```

Поддерживает файл, индекс камеры, **RTSP/HTTP URL** (через `cv2.VideoCapture`) и каталог
картинок. HLS напрямую не умеет — как и всё на cv2.VideoCapture (см. отчёт 01 §8: для HLS
нужен ffmpeg-pipe, который мы напишем сами).

### Трекеры и параметры (ключевое для нас)
`ByteTrackTracker.__init__` (дефолты из `core/bytetrack/tracker.py`):
- `lost_track_buffer=30` (кадров при 30 fps — буфер «потерянных» треков), `frame_rate=30.0`;
- `track_activation_threshold=0.7` — мин. confidence для создания нового трека;
- `minimum_consecutive_frames=2` — до подтверждения трек получает `tracker_id = -1`;
- `minimum_iou_threshold=0.1`, `high_conf_det_threshold=0.6`.

Алгоритм `update()` (фрагмент, `core/bytetrack/tracker.py`): двухстадийная ассоциация —
сначала high-confidence детекты против всех треков (Kalman-предикт + IoU + Hungarian
`scipy.optimize.linear_sum_assignment`), затем low-confidence детекты против оставшихся
треков — «оживление» в окклюзии:

```python
high_mask = confidences >= self.high_conf_det_threshold
...
# Step 1: associate high-confidence detections to all tracks
iou_matrix = self.iou.compute(predicted_boxes, high_boxes)
matched, unmatched_tracks, unmatched_high = self._get_associated_indices(iou_matrix, self.minimum_iou_threshold)
...
# Step 2: associate low-confidence detections to remaining tracks
remaining_boxes = predicted_boxes[unmatched_tracks]
iou_matrix = self.iou.compute(remaining_boxes, low_boxes)
matched, _, unmatched_low = self._get_associated_indices(iou_matrix, self.minimum_iou_threshold)
```

Важные детали:
- **чистый CPU-стек**: Kalman + IoU + `linear_sum_assignment`, без torch в ядре;
- работа с `sv.Detections` (supervision) — единый формат «боксы + tracker_id» на весь пайплайн;
- поддержка absolute `timestamp` (важно для стрима: не считаем кадры, а время),
  `reset()` при смене видео/сцены;
- у BoTSORT дополнительно CMC (`enable_cmc=True`, sparseOptFlow) — компенсация движения камеры;
- есть `search_space` + CLI `trackers tune` — готовый автоподбор под конкретное видео (Optuna).

### Зависимости (`pyproject.toml`)
Ядро лёгкое: `numpy>=2.0.2, supervision>=0.26.1, scipy>=1.13.1, opencv-python>=4.8.0, rich, requests, jsonargparse, pydeprecate`. **torch — только в опциональных extra** (`mask`, `detection`). Demo тянет `inference-models` + torch (тяжёлое), но для нас это не нужно: детектор свой (MOG2 или YOLO).

### Лицензия
Apache-2.0 — можно свободно использовать, включая коммерчески; атрибуция в NOTICE не обязательна, лицензию сохранить желательно. Это **лучший кандидат на «готовый трекер» для лёгкого пайплайна**.

---

## 4. FoundationVision/ByteTrack (оригинальная реализация)

### Структура
Исследовательский репо вокруг YOLOX: детектор + набор MOT-трекеров + бенчмарки MOT17/MOT20.
- **ядро трекера**: `yolox/tracker/byte_tracker.py` (классы `STrack`, `BYTETracker`),
  `yolox/tracker/matching.py` (180 LOC — IoU distance + Hungarian),
  `yolox/tracker/kalman_filter.py` (269), `yolox/tracker/basetrack.py` (STrack-статы: Tracked/Lost/Removed);
- демо: `tools/demo_track.py`; оценка: `tools/track*.py`, `yolox/evaluators/MOTevaluator.py`;
- в `tutorials/` — порты ByteTrack под JDE, FairMOT, CSTrack, TransTrack.

### Пайплайн / ввод
`tools/demo_track.py`: `cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)` —
файл или камера по индексу; URL формально пройдёт в VideoCapture (RTSP), HLS — нет.
Цикл: YOLOX-inference → `tracker.update(outputs[0], [h, w], exp.test_size)` → фильтр выводов:

```python
# tools/demo_track.py
for t in online_targets:
    tlwh = t.tlwh
    tid = t.track_id
    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh   # default 1.6
    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical: # min_box_area default 10
        online_tlwhs.append(tlwh) ...
```

Это единственная «адаптация к размерам» в репо — отбрасывание крошечных и слишком вытянутых боксов.

### Трекер (алгоритм, `yolox/tracker/byte_tracker.py`, BYTETracker.update)
- параметры через args: `track_thresh=0.6` (high), low-ярус `(0.1, 0.6]`, `match_thresh=0.9`
  (порог IoU-дистанции), `track_buffer=30` → `buffer_size = int(frame_rate/30 * track_buffer)`;
- статы трека: Tracked / Lost / Removed, Kalman по `(cx, cy, a, h)` — **аспект и высота в явлении**,
  поэтому трекер устойчив к разным размерам объектов (широкий угол);
- три стадии ассоциации: high-score → все треки (dists = iou_distance + fuse_score),
  low-score → оставшиеся Tracked, unconfirmed треки → непривязанные high-детекты (thresh 0.7).

### Зависимости
Тяжёлый исследовательский стек: `torch>=1.7, torchvision, onnx/onnxruntime==1.8.x, motmetrics,
filterpy, h5py, tensorboard, thop` + YOLOX-веса (нужно скачивать). Python 3.7-era код,
насовременных torch будет болеть. **Для нас это алгоритмический референс, не библиотека.**

### Лицензия
MIT. Но код застыл в 2022: использовать как «взять файл и подключить» — плохая идея;
брать идею двухстадийной ассоциации уже реализована свежо в roboflow/trackers (см. п.3).

---

## 5. Gupu25/PeopleCounter (⚠️ без лицензии — только логика)

### Структура
Два файла, **297 LOC** суммарно: `PeopleCounter.py` (218) — пайплайн, `Person.py` (79) — классы трека.

### Пайплайн
- **Ввод**: `cap = cv.VideoCapture('Test Files/TestVideo.avi')` (комментирован `cv.VideoCapture(0)`); URL/HLS не поддерживаются;
- **Детекция движения**: MOG2 с тенями + двойной apply + порог 200 против серых теней:

```python
# PeopleCounter.py
fgbg = cv.createBackgroundSubtractorMOG2(detectShadows = True)
kernelOp  = np.ones((3,3), np.uint8)    # open
kernelCl  = np.ones((11,11), np.uint8)  # close
...
fgmask  = fgbg.apply(frame)
fgmask2 = fgbg.apply(frame)
ret, imBin  = cv.threshold(fgmask,  200, 255, cv.THRESH_BINARY)   # убирает тени (серый=127)
ret, imBin2 = cv.threshold(fgmask2, 200, 255, cv.THRESH_BINARY)
mask  = cv.morphologyEx(imBin,  cv.MORPH_OPEN,  kernelOp)
mask  = cv.morphologyEx(mask,   cv.MORPH_CLOSE, kernelCl)
```

Порог площади: `areaTH = frameArea / 250` (≈0.4% кадра) — **масштабируется с размером кадра**,
что частично решает проблему «разный размер объектов» на широком угле.

### Логика подсчёта — ДВЕ линии + ограничительные полосы
Геометрия: красная линия `line_down = 3h/5`, синяя `line_up = 2h/5`, плюс границы зоны
`up_limit = h/5`, `down_limit = 4h/5`. Объекты учитываются только внутри полосы
(`if cy in range(up_limit, down_limit)`), после пересечения и выхода за свою границу
маркируются done. Направление определяется тем, какую из двух линий пересёк трек.

Антидубль и направление — в `Person.py`:

```python
# Person.py: MyPerson
def going_UP(self, mid_start, mid_end):
    if len(self.tracks) >= 2:
        if self.state == '0':
            if self.tracks[-1][1] < mid_end and self.tracks[-2][1] >= mid_end:  # cruzo la linea
                state = '1'
                self.dir = 'up'
                return True
        else:
            return False
    else:
        return False
```

Привязка «детект → трек» (весь «трекер» — O(n) ближнее-соседство по bbox):

```python
# PeopleCounter.py
for i in persons:
    if abs(x - i.getX()) <= w and abs(y - i.getY()) <= h:
        new = False
        i.updateCoords(cx, cy)          # обновляет трек, сбрасывает age
        if i.going_UP(line_down, line_up):
            cnt_up += 1
        elif i.going_DOWN(line_down, line_up):
            cnt_down += 1
        break
...
if i.timedOut():                        # age > max_p_age (5 кадров) — удаляем трек
    persons.pop(persons.index(i))
```

Разбор механизма:
- **пересечение**: sign-change по Y между двумя последними точками трека, отдельно для каждой линии;
- **антидубль**: флаг `state '0'→'1'` (один снос на трек) + удаление трека за границей полосы.
  ⚠️ **баг оригинала**: в `going_UP/going_DOWN` присваивается локальная `state = '1'`, а не
  `self.state = '1'` — флаг никогда не фиксируется, защита от повторного пересечения внутри
  зоны фактически не работает (спасает только removal за границей). **Не воспроизводить.**
- **направление**: два счётчика up/down; общий трафик получается суммой;
- **трекер**: наивный — ближайший bbox, max_age=5 кадров; при сближении двух людей происходит
  перепутывание (не помогает антидубль). Это и есть та причина, по которой в нашем проекте
  трекинг берём нормальный (SORT/ByteTrack), а MOG2-пайплайн оставляем как «лёгкий» режим.
- **разные размеры**: только areaTH от площади кадра; на широком угле дальние люди могут
  не дотянуть до порога — порог придётся делать адаптивным/двухуровневым.

### Зависимости
`pip install opencv-python` (README); формального requirements.txt нет. Легчайший из всех.

### Лицензия
**Отсутствует.** Код не копировать; используем как иллюстрацию «лёгкого» MOG2-пайплайна
и схемы двухлинейного counting line с ограничительными зонами.

---

## 6. Сравнительная таблица механизмов

| Механизм | A: yolov8-line-crossing-counter | B: roboflow/trackers | C: ByteTrack (orig.) | D: Gupu25/PeopleCounter |
|---|---|---|---|---|
| **Ввод стрима** | только файл (`cv2.VideoCapture(path)`) | `frames_from_source()`: файл/камера/RTSP-HTTP/каталог; HLS нет (нужен ffmpeg-pipe) | файл/камера в demo_track.py; HLS нет | только файл/камера 0 |
| **Детекция движения** | YOLOv8 (`ultralytics`, class person) | не детектор — трекер; демо: rfdetr (torch) | YOLOX (torch) | MOG2 + threshold(200) + open(3×3)/close(11×11), areaTH = frame/250 |
| **Counting line** | 1 горизонтальная линия `y=h/2`; sign-change центаоида | нет (трекер, не counter) | нет (трекер, не counter) | 2 линии (2h/5, 3h/5) + полосы ограничений h/5–4h/5; sign-change по треку |
| **Антидубль** | `counted_ids` set по track_id + история prev_cy | подтверждение трека `minimum_consecutive_frames=2`, ID=-1 до подтверждения | статы Tracked/Lost/Removed, buffer 30 кадров | флаг state (с багом — не работает!) + удаление трека за границей; max_age=5 |
| **Направление** | одно (вниз), hardcode | n/a | n/a | два: up/down, отдельными счётчиками |
| **Трекер** | BoT-SORT из ultralytics, `persist=True`, параметры не тронуты | ByteTrack/SORT/BoTSORT/OCSORT/CBIoU/MCByte, clean-room, Kalman + linear_sum_assignment, CMC в BoTSORT, tune через Optuna | оригинальный BYTE: 2-стадийная ассоциация (high>0.6, low 0.1–0.6), match_thresh=0.9, KF по (cx,cy,a,h) | наивный nearest-bbox, max_age=5 |
| **Адаптация к размерам** | YOLO инвариантен; отдельно не решено | size-агностичен (IoU+KF); `tune` под видео | KF хранит aspect/height; фильтр min_box_area=10 и aspect<1.6 в демо | areaTH ∝ площади кадра |
| **CPU-only?** | нет (torch) | **ядро да** (numpy/scipy), torch только в extra | нет (torch) | **да** (OpenCV) |
| **Лицензия** | MIT | Apache-2.0 | MIT | **нет лицензии** |

---

## 7. Заимствования для нашего проекта + «пишем сами»

### Берём из референсов (механизм → источник)
1. **Из A (JovanSk)** — шаблон counting line: sign-change центаоида между кадрами +
   `counted_ids` (set отсечённых track_id) как антидубль. Файл/функция: `process_and_count.py`,
   блок «COUNTING» в главном цикле. Доработаем: оба направления,Cooldown вместо пожизненного
   set, прунинг по возрасту трека.
2. **Из B (roboflow/trackers)** — **самого трекера**: `ByteTrackTracker` из
   `src/trackers/core/bytetrack/tracker.py` (+ `BaseTracker`, Kalman, IoU). Apache-2.0, CPU-only,
   интерфейс `tracker.update(sv.Detections) -> sv.Detections(tracker_id=...)`. Дефолты
   (`lost_track_buffer=30, track_activation_threshold=0.7, minimum_consecutive_frames=2`) —
   стартовая точка; при необходимости подкрутим через их же `search_space`/Optuna.
   Также `frames_from_source()` (`src/trackers/io/video.py`) — заготовка чтения RTSP/файла.
3. **Из C (ByteTrack)** — только алгоритмическая справка: двухстадийная ассоциация,
   Kalman по `(cx, cy, aspect, h)`, buffer_size ∝ frame_rate (`yolox/tracker/byte_tracker.py`).
   Код не переносим (zastyle 2022, torch-стек).
4. **Из D (Gupu25)** — только схема: MOG2 + двойной apply + threshold(200) против теней +
   open/close + `areaTH = frameArea/250` как «лёгкий» без-DL режим детекции; и геометрия
   двухлинейного счёта (вход/выход) с ограничительными полосами, чтобы трекер не считал
   движение вне ROI. Код НЕ копировать (нет лицензии).

### Пишем сами (нигде нет в готовом виде)
1. **HLS-ввод**: ffmpeg → raw-frames pipe (отчёт 01 §8.3), абстракция `VideoSource` с единым
   интерфейсом файл/HLS/RTSP + обработка разрыва стрима и ресинка (ни одно репо не умеет HLS).
2. **Нормальная counting-line логика**: произвольная наклонная линия (в A — только горизонталь,
   в D — только горизонтальные пары), проекция трека на линию, оба направления + режим «общий»,
   cooldown по времени вместо пожизненного `counted_ids`, устойчивая к ID-switch
   (окно подавления после пересечения: не считать новый ID, если старый ушёл в lost).
3. **Адаптивный порог размера** для MOG2-режима на широком углу (двухуровневый areaTH / по Y-полосам кадра).
4. **Служба подсчёта**: состояние счётчика (up/down/total), логирование событий пересечения с
   таймстампами, выгрузка в файл/CSV — в A/D это `print` + `log.txt`.
5. **Обвязка трекера под наш пайплайн**: конвертация детектов MOG2/YOLO → `sv.Detections`,
   `tracker.reset()` при смене источника, маппинг ID=-1 (неподтверждённый) в логику линии.

### Итоговая архитектура (кратко)
`VideoSource(ffmpeg-pipe HLS | файл | RTSP)` → детекция: режим «лёгкий» MOG2 (схема D, свой код)
или «точный» YOLO (как в A) → `ByteTrackTracker` из roboflow/trackers (B) → наша counting-line
(логика A + геометрия D + cooldown) → счётчик/лог.
