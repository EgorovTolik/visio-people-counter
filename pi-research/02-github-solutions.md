# Open-source решения подсчёта людей на GitHub (отбор для задачи 03)

Дата прогона: 2026. Все данные получены live через `curl` к GitHub REST API
(без авторизации), raw.githubusercontent.com и HEAD-запросы. Ничего не клонировалось — это задача 03.

---

## 1. Методика поиска

**GitHub Search API** (`https://api.github.com/search/repositories?q=...&sort=stars&order=desc&per_page=10`):

| # | Запрос | Найдено всего |
|---|--------|---------------|
| a | `people counter opencv` | 230 |
| b | `pedestrian counting video` | 55 |
| c | `background subtraction people counting line` | 3 (мало, слишком узко) |
| d | `yolo people counter tracking` | 43 |
| e | `crowd counting street camera` | 2 (шумные результаты, почти бесполезен) |
| f | `footfall counter opencv` | 15 (в основном учебные репо ≤1★) |
| g | `object counting line crossing video` | 18 (авто-счётчики + пара line-crossing) |
| h | `multiple object tracking SORT kalman` | 3 (доп. запрос после блокировки DDG — см. ниже) |
| i | `bytetrack` | 2243 (доп. запрос: нашлись крупные трекер-репо) |

**Rate limits:** без авторизации search = 10/мин, core = 60/час. Соблюдалось:
- между search-запросами `sleep 8`, всего **9 search-запросов** (в пределах лимита);
- core-запросы — только 3 (`GET /repos/sarful/...`, `/repos/phschiele/manokward`, `/repos/roboflow/supervision`) + 1 проверка `rate_limit`; остаток на момент конца прогона: search=10, core=57.

**README** — только через `raw.githubusercontent.com/<owner>/<repo>/HEAD|main|master/README.md`
(не расходует API-лимит). **DuckDuckGo web-поиск НЕ удался**: оба HTML-эндпоинта
(`html.duckduckgo.com/html/` и `lite.duckduckgo.com/lite/`) вернули anomaly/captcha-страницу
(~14 КБ, ни одного `result__a`) — датацентровый IP заблокирован. Компенсировано доп.
GitHub-поисками h/i (в разделе 5).

**Дубликаты/форки:** `Stereoscopic-memory180/yolov8-line-crossing-counter` и
`sangjune20012-collab/yolov8-crosszone-counter` — копии/варианты одного паттерна, в детали не взяты;
`ifzhang/ByteTrack` переименован → основной репо теперь `FoundationVision/ByteTrack`.

---

## 2. Таблица кандидатов

| Репо | Stars | Язык | Лицензия | Последний push | Суть (1 строка) | URL |
|---|---|---|---|---|---|---|
| roboflow/supervision | 49958 | Python | MIT | 2026-09 | Библиотека: детекции, трекеры, «zone counting» — готовые count-утилиты | https://github.com/roboflow/supervision |
| mikel-brostrom/boxmot | 8293 | Python | **AGPL-3.0** ⚠️ | 2026-09 | SOTA трекеры (ByteTrack, BoT-SORT и др.), но copyleft — не брать в продукт | https://github.com/mikel-brostrom/boxmot |
| FoundationVision/ByteTrack | 6678 | Python | MIT | 2024-06 | Оригинал ByteTrack (ECCV 2022), «тяжёлое» референс-решение | https://github.com/FoundationVision/ByteTrack |
| roboflow/trackers | 3765 | Python | Apache-2.0 | 2026-09 | Чистые переписанные SORT/ByteTrack/BoT-SORT, без GPU, pip-install | https://github.com/roboflow/trackers |
| shaoshengsong/DeepSORT | 1004 | C++ | none | 2026-09 | DeepSORT на C++17 + ONNX — не наш стек | https://github.com/shaoshengsong/DeepSORT |
| nathanrooy/rpi-urban-mobility-tracker | 129 | Python (Jupyter) | GPL-3.0 | 2024-08 | DeepSORT + MobileNet(TFLite), подсчёт пешеходов/авто на RPi, Docker | https://github.com/nathanrooy/rpi-urban-mobility-tracker |
| Gupu25/PeopleCounter | 76 | Python | **нет** ⚠️ | 2020-10 | Чистый OpenCV: MOG2 + морфология + ДВЕ линии счёта (in/out) — наш пайплайн 1:1 | https://github.com/Gupu25/PeopleCounter |
| jeffskinnerbox/people-counter | 67 | Python | NOASSERTION ⚠️ | 2023-02 | Код по туториалу femb.com.mx: MOG2 → морфология → контуры → счёт | https://github.com/jeffskinnerbox/people-counter |
| koba/overhead-camera-people-counter | 64 | C++ | GPL-3.0 | 2020-02 | Подсчёт с верхней (обзорной) камеры, C++ — не наш стек | https://github.com/koba/overhead-camera-people-counter |
| narayananramu/opencv-people-counter | 51 | Python | **нет** ⚠️ | 2017-08 | Старый учебный счётчик OpenCV (README отсутствует, репо без описания) | https://github.com/narayananramu/opencv-people-counter |
| akshun4/People-Counter | 46 | Python | **нет** ⚠️ | 2019-04 | HOG+SVM+NMS+SIFT, ROI-зона, in/out — детекторный вариант без YOLO | https://github.com/akshun4/People-Counter |
| chs74515/PeopleCounter | 45 | Python | **нет** ⚠️ | 2019-05 | Детекция + трекинг + подсчёт людей (курсовая работа) | https://github.com/chs74515/PeopleCounter |
| mgalushka/pedestrians-traffic-calc | 44 | Processing | MIT | 2019-06 | Подсчёт пешеходов с веб-камеры (Processing, не Python/OpenCV) | https://github.com/mgalushka/pedestrians-traffic-calc |
| sarful/People-counter-opencv-python3 | 32 | Python | **нет** ⚠️ | 2020-04 | Люди-счётчик OpenCV/py3 (README пустой, малoinформативно) | https://github.com/sarful/People-counter-opencv-python3 |
| noorkhokhar99/People-Counter-using-YOLOv8-and-Object-Tracking-People-Counting-Entering-Leaving- | 17 | Python | MIT | 2023-04 | YOLOv8 + трекинг, entering/leaving счётчик (видео-проект) | https://github.com/noorkhokhar99/People-Counter-using-YOLOv8-and-Object-Tracking-People-Counting-Entering-Leaving- |
| JovanSk/yolov8-line-crossing-counter | 3 | Python | MIT (badge) | 2026-06 | YOLOv8 + ByteTrack (`model.track(persist=True)`) + line crossing, один скрипт 215 строк | https://github.com/JovanSk/yolov8-line-crossing-counter |

⚠️ — «нет лицензии» = по умолчанию все права у автора; для референса читать можно,
код копировать нельзя. Для задачи 03 такие репо не рекомендуются как основа.

---

## 3. Детальные заметки по лучшим кандидатам (README / исходники)

### 3.1 JovanSk/yolov8-line-crossing-counter — **самый близкий к нашей задаче**
- Пайплайн: `YOLO("yolov8s.pt")` → `model.track(frame, persist=True)` (ByteTrack по умолчанию в ultralytics) → горизонтальная counting line (`line_y = height // 2`) с наложением и стрелкой направления.
- Де-дупликация: `counted_ids = set()` — ID трека учитывается один раз (механизм из раздела 4.3 отчёта 01).
- Весь код — **один файл** `process_and_count.py` (~215 строк), есть `requirements.txt`, структура тривиальная; overlay в CCTV-стиле (ID, bbox, линия, счётчик, timestamp/FPS).
- Зависимости: ultralytics + opencv; GPU не требуется (yolov8s работает на CPU). Минусы: одна горизонтальная линия на середине кадра (для нас нужно перенести в произвольный отрезок prev→cur из раздела 4.1), только «вход», in/out отдельно не ведётся — но это доработка в десятки строк.

### 3.2 Gupu25/PeopleCounter — **чистый OpenCV, наш «лёгкий» пайплайн 1:1**
- Проверил исходник `PeopleCounter.py`: `cv.createBackgroundSubtractorMOG2(detectShadows=True)`, две линии (in/out) на y=2h/5 и y=3h/5, морфология OPEN (ядро op) + CLOSE, обработка тени через вторую маску (`imBin2`), счёт по направлению.
- Это практически дословная реализация конвейера из отчёта 01 (разделы 1–4) — лучший «эталон лёгкого» для сравнения и заимствования логики пересечения/направления.
- **Минусы:** лицензии нет (только референс), Python 2-стиль импортов, без трекера (де-дубль через эвристики), мёртв с 2020.

### 3.3 jeffskinnerbox/people-counter
- README — фактически конспект 9-частьного туториала femb.com.mx: установка → поток → рисование → **background subtraction** → морфология → контуры → «определение человека» → отслеживание движения → подсчёт. + ссылки на приёмы pyimagesearch по FPS (`cv2.VideoCapture`, ускорение обработки).
- Лицензия NOASSERTION (нестандартный файл) — референс-уровень. Полезно: пошаговая логика «лёгкого» конвейера и советы по скорости.

### 3.4 nathanrooy/rpi-urban-mobility-tracker
- DeepSORT + MobileNet v1 (TFLite, квантованный) или PedNet; CLI `umt` с `-camera/-video/-imageseq`, порог детекции, опция Coral TPU; Docker-установка; тестовые данные MOT PETS09-S2L1.
- Архитектурно — «верхний» вариант (детектор+трекер), но стек TF-Lite/RPi/Docker и **GPL-3.0** → не основа, а референс по организации CLI/конвейера и параметрам DeepSORT.

### 3.5 roboflow/trackers (+ supervision) — **источник трекера для лёгкого пайплайна**
- `trackers` (Apache-2.0): clean-room реализации SORT, ByteTrack, OC-SORT, BoT-SORT, C-BIoU, McByte; единый интерфейс `tracker.update(detections)`; нативно говорит на `supervision.Detections`; Python ≥ 3.10, чистый NumPy — **без GPU и без inference-библиотек**; есть компенсация движения камеры (BoT-SORT/McByte) — релевантно нашей «качке камеры» из раздела 9 отчёта 01.
- `supervision` (MIT, ~50k★, активен): «from data loading to real-time zone counting» — готовые count-утилиты и обёртки трекеров; в задаче 03 использовать как библиотеку/референс де-дупликации.
- Это замена «пропавшим» классическим репо SORT (см. раздел 4): phschiele/manokward, abergal/SORT и longcw/DeepSORT теперь **404**.

### 3.6 FoundationVision/ByteTrack — «тяжёлое» референс-решение
- Оригинал ByteTrack (arXiv:2110.06864, ECCV 2022; SOTA MOT17): ассоциация всех box'ов включая слабые det'ы, работа с tracklet'ами для окклюзий. MIT. Репозиторий переименован из ifzhang/ByteTrack (старые ссылки редиректят).
- Для нас — источник архитектуры «связки» детектора и трекера; в проде берём ByteTrack через ultralytics `model.track()` или через roboflow/trackers, а не этот репо напрямую (его dev-зависимости тяжёлые: mmdetection-стек).

### 3.7 noorkhokhar99/People-Counter-using-YOLOv8...
- YOLOv8 + object tracking, счёт entering/leaving; MIT; README минимальный (видео на YouTube), код — учебный. Запасной второй «детекторный» вариант, если JovanSk окажется неудобным.

---

## 4. Live-проверка источников из отчёта 01

| URL | HEAD | GET (с догоном редиректов) | Вердикт |
|---|---|---|---|
| https://docs.opencv.org/4.x/d1/d53/classcv_1_1BackgroundSubtractorMOG2.html | 301 → `/4.13.0/...` | **403** | ⚠️ Сайт жив, но путь `4.x` теперь редиректит на версионный `4.13.0`; прямой доступ с нашего IP блокируется (403, вероятно Cloudflare/datacenter-IP). Для справки: использовать `https://docs.opencv.org/4.13.0/d1/d53/classcv_1_1BackgroundSubtractorMOG2.html` (откроется в браузере) |
| https://docs.opencv.org/4.x/de/d0e/classcv_1_1BackgroundSubtractorKNN.html | 301 → `/4.13.0/...` | **403** | ⚠️ То же: `https://docs.opencv.org/4.13.0/de/d0e/classcv_1_1BackgroundSubtractorKNN.html` |
| https://docs.opencv.org/4.x/d9/dfc/group__tracking.html | 301 → `/4.13.0/...` | **403** | ⚠️ То же: `https://docs.opencv.org/4.13.0/d9/dfc/group__tracking.html` |
| https://supervision.roboflow.com/ | **200** | 200 | ✅ доступен; репо roboflow/supervision активно (push 2026-09, MIT) |
| https://docs.ultralytics.com/ | **200** | 200 | ✅ доступен |
| https://github.com/ifzhang/ByteTrack | 301 → FoundationVision/ByteTrack | **200** | ⚠️ Репозиторий переименован: корректный URL теперь **https://github.com/FoundationVision/ByteTrack** (6678★, MIT) |
| https://github.com/abergal/SORT | **404** | 404 | ❌ **репо удалено**. Классические репо SORT/DeepSORT мертвы: `abergal/SORT` — 404, `longcw/DeepSORT` — 404, популярный fork `phschiele/manokward` — 404. Заменитель: **roboflow/trackers** (чистая реализация SORT, Apache-2.0) |

Важные правки к отчёту 01: п. 9 источников (`abergal/SORT`) — мёртв; п. 11 (`ifzhang/ByteTrack`)
— переименован в `FoundationVision/ByteTrack`; URL документации OpenCV `/4.x/...` — редиректят
на `/4.13.0/...`.

---

## 5. Найденное через web-поиск сверх GitHub-API

DuckDuckGo (html и lite) **заблокировал запросы с нашего IP** (anomaly-страница, парсинг
`result__a` дал 0 результатов) — web-поиск через DDG в этом прогоне фактически недоступен.
Компенсация — два дополнительных GitHub-API-запроса (h: `multiple object tracking SORT kalman`,
i: `bytetrack`), которые дали главное «сверх top-stars по people counting»:

- **roboflow/trackers** (3765★, Apache-2.0) — не попадал бы ни в один запрос по «people counting», но это сейчас канонический лёгкий источник SORT/ByteTrack после смерти классических репо;
- **FoundationVision/ByteTrack** (6678★) — актуальное имя оригинала ByteTrack;
- mikel-brostrom/boxmot (8293★, AGPL-3.0) — учтён и отклонён по лицензии;
- мелкие: `marwankefah/Kalman_Tracking_Single_Camera` (14★, неофициальная SORT-реализация, Python) — возможный минимальный референс именно «SORT на bbox'ах», если roboflow/trackers покажется избыточным.

---

## 6. ИТОГОВАЯ РЕКОМЕНДАЦИЯ для задачи 03 (клонировать эти репо)

1. **JovanSk/yolov8-line-crossing-counter** — https://github.com/JovanSk/yolov8-line-crossing-counter
   MIT, Python, один скрипт; **максимально близко к задаче**: YOLOv8 + ByteTrack + counting line + де-дубль по track_id. Нет тяжёлых GPU-зависимостей (yolov8s на CPU) — закрывает обязательное условие «хотя бы одно лёгкое». Базовый каркас для детекторного варианта; доработки под нас: произвольный отрезок линии с prev→cur пересечением (раздел 4.1 отчёта 01), в обе стороны, HLS-вход через ffmpeg pipe.
2. **roboflow/trackers** — https://github.com/roboflow/trackers
   Apache-2.0, чистые реализации SORT/ByteTrack без GPU и inference-зависимостей; **ядро «лёгкого» конвейера**: MOG2 → морфология → компоненты → `SORTTracker.update(detections)` → counting line. Заменитель мёртвых репо SORT (abergal/SORT, phschiele/manokward — 404). Читать в первую очередь `src/trackers/sort.py`.
3. **FoundationVision/ByteTrack** — https://github.com/FoundationVision/ByteTrack
   MIT; «тяжёлое» референс-решение (оригинал ECCV 2022) для сравнения архитектуры детектор+трекер, в т.ч. логики работы со слабыми детекциями и окклюзиями. Клонировать как reference, не как основа.

**Рекомендуемые дополнения (библиотеки/референсы, не обязательны к клону в 03):**
- `roboflow/supervision` (MIT) — готовые count-утилиты и формат Detections;
- `Gupu25/PeopleCounter` (без лицензии!) — только как **read-only референс** логики MOG2+линии, наш пайплайн в 1:1; код не копировать.

Критерии выбора соблюдены: ≥1 лёгкий Python+OpenCV без GPU (JovanSk + trackers); лицензии MIT/Apache у всех трёх; одно тяжёлое референс-решение YOLO+ByteTrack (FoundationVision).
