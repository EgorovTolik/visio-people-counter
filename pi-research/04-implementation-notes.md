# Задача 04: Implementation notes — архитектура решения для подсчёта трафика

Опора: отчёты 01 (методы), 02 (отбор репо), 03 (разбор кода) + прямой просмотр кода в
`references/repos/{trackers,ByteTrack,yolov8-line-crossing-counter,PeopleCounter}`.
Все пути к файлам ниже проверены `ls`/`grep` на момент написания (HEAD'ы — см. отчёт 03).

Стек: Python + OpenCV, numpy 1.26.4, ffmpeg 6.1.1, **CPU only**. Ввод: HLS URL или файл.
Трекинг — вспомогательный механизм против двойного счёта (основная детекция — движение).

---

## 1. Целевая архитектура: модули и dataflow

### 1.1 Структура проекта (предложение)

```
visio_people_counter/
├── __main__.py            # CLI-вход (argparse/click): count / calibrate / probe
├── config.py              # Config: dataclass + загрузка/сохранение YAML, валидация, нормализация координат
├── video_source.py        # VideoSource: абстракция «источник кадров» (HLS-ffmpeg-pipe | файл)
├── motion_detector.py     # MotionDetector: MOG2 + морфология + connectedComponents → Bbox'ы
├── tracker_adapter.py     # TrackerAdapter: bbox'ы blob'ов ↔ sv.Detections, обёртка ByteTrackTracker/SORTTracker
├── line_counter.py        # LineCounter: наклонная линия, направление, антидубль, события
├── size_profile.py        # SizeProfile: интерполяция min/max area по сегментам линии (см. §3)
├── event_log.py           # EventLog: лог событий (JSONL/CSV), счётчики in/out/total, экспорт
├── calibrate.py           # calib-режим: GUI на cv2.setMouseCallback → заполнение config.yaml
└── pipeline.py            # Pipeline: связывает всё в цикл обработки кадра + watchdog
```

### 1.2 Ответственность модулей

| Модуль | Классы | Ответственность |
|---|---|---|
| `video_source.py` | `VideoSource` (ABC), `FfmpegPipeSource`, `FileSource` | Единственный источник кадров: `read() -> Frame` (frame BGR, frame_index, t_wall, t_video если известно). HLS — ffmpeg-процесс + rawvideo-pipe; файл — `cv2.VideoCapture`. Watchdog переподключения живёт здесь (§4). Не знает о детекции. |
| `motion_detector.py` | `MotionDetector` | MOG2 (`detectShadows=True`) → отсечение теней (threshold 200) → MORPH_OPEN/CLOSE → `connectedComponentsWithStats` → фильтр по площади/aspect/fill → список `Blob(x,y,w,h,area)` в пикселях исходного кадра. Порог minArea — из `SizeProfile` (§3). Держит состояние субтрактора (`reset()` при переподключении/смене сцены). |
| `tracker_adapter.py` | `TrackerAdapter` | Конвертация: bbox'ы blob'ов → `sv.Detections(xyxy=..., confidence=None)` → `ByteTrackTracker.update(...)` → обратно в список объектов с `track_id` (и `-1` для неподтверждённых). При `confidence=None` ByteTrack деградирует до SORT-режима одностадийного IoU-матча — **это именно наш случай** (документировано в docstring `ByteTrackTracker`, `references/repos/trackers/src/trackers/core/bytetrack/tracker.py`). Обёртка `reset()` при смене источника. |
| `line_counter.py` | `LineCounter` | Линия (2 точки), направление, пересечение по prev→cur, антидубль-набор (§2). Вывод: список `CrossingEvent` на кадр. Счётчики in/out/total — здесь же или в `event_log`. |
| `size_profile.py` | `SizeProfile` | Контрольные точки «высота человека в точке t линии = H px» → интерполяция min/max area, buffer_width, min_event_dist по позиции пересечения (§3). |
| `config.py` | `Config`, `Line`, `SizePoint` | YAML: координаты **нормализованные 0..1** (масштабируются на фактическое разрешение из ffprobe — при смене камеры/разрешения конфиг не ломается); absolute px хранятся только для size-контрольных точек как «px при эталонном разрешении W_ref» или сразу в % от высоты кадра (рекомендую % H). |
| `event_log.py` | `EventLog` | JSONL-лог событий (§2.4), агрегаты in/out/total, периодическая сводка (каждые N сек: +in/+out), опц. HTTP-дамп результата (открытый вопрос, §7). |
| `calibrate.py` | — | calib-GUI (§3.3): рисование линии/ROI/size-точек мышью → запись в config.yaml. |
| `pipeline.py` | `Pipeline` | Главный цикл: read → detect → track → count → log; тайминги (fps обработки, lag); обработка EOF/переподключений; graceful shutdown по SIGINT с финальной сводкой. |

### 1.3 Dataflow (ASCII)

```
                        ┌────────────────────────────── pipeline.py ─────────────────────────────────┐
 HLS URL / file path    │                                                                            │
        │               ▼                                                                            │
 ┌───────────────┐  Frame{img, idx, t_wall}      ┌──────────────────┐  List[Blob]                    │
 │ VideoSource   │ ────────────────────────────► │ MotionDetector   │ ──────────────────┐            │
 │ (ffmpeg-pipe  │                               │ MOG2 → shadow →  │                   ▼            │
 │  | file)      │ ◄── watchdog: N bad reads ──► │ OPEN/CLOSE → CCW │   sv.Detections   ┌────────────────────┐
 └───────────────┘     restart ffmpeg (backoff)  │ + area filter    │ (xyxy, conf=None) │  TrackerAdapter    │
                                                 └──────────────────┘ ────────────────► │ ByteTrackTracker  │
                                                                                        │ (roboflow/trackers)│
                                                                                        └─────────┬──────────┘
                                                                                        Detections + tracker_id
                                                                                                  │ (после подтверждения, id != -1)
                                                                                                  ▼
                                                        ┌───────────────────────────────  ┌─────────────┐
                                                        │ List[CrossingEvent]             │ LineCounter │
                                                        └───────────────────────────────◄─┴─────────────┘
                                                                                    prev→cur × наклонная линия,
                                                                                    направление ±1, антидубль
                                                                                        │
                                                                                        ▼
                                                                                   ┌─────────────┐   JSONL / CSV / stdout
                                                                                   │  EventLog   │ ────────────────────►
                                                                                   │ in/out/total│
                                                                                   └─────────────┘
```

Опциональный второй режим детекции — YOLO+ByteTrack (уточнение в §6/§7): тот же
`TrackerAdapter` и `LineCounter`, вместо `MotionDetector` — `Detector` (ultralytics).

### 1.4 Трекер: параметры для CPU (проверено по коду `references/repos/trackers`)

Репо `roboflow/trackers` v2.6.0 (Apache-2.0), пакет `trackers`, экспортирует
`ByteTrackTracker, SORTTracker, BoTSORTTracker, OCSORTTracker`
(`references/repos/trackers/src/trackers/__init__.py`). Ядро — чистый NumPy +
`scipy.optimize.linear_sum_assignment` (Hungarian) + Kalman; **torch в ядре нет** —
в `references/repos/trackers/pyproject.toml` базовые зависимости: numpy, supervision,
scipy, opencv-python, rich, requests, jsonargparse, pydeprecate; torch встречается
только в опциональных extra (`mask`, `detection`). Supervision (MIT) — формат данных.

Дефолты из кода (проверены):

`ByteTrackTracker.__init__` (`references/repos/trackers/src/trackers/core/bytetrack/tracker.py:93-101`):
- `lost_track_buffer=30` — буфер потерянных треков в кадрах при 30 fps (масштабируется по `frame_rate`: `maximum_time_without_update = lost_track_buffer/30.0` сек);
- `frame_rate=30.0`;
- `track_activation_threshold=0.7` — не используется, если `confidence=None` (у нас conf нет → любой blob может создать трек; параметр фактически отключён);
- `minimum_consecutive_frames=2` — до этого tracklet получает `tracker_id=-1` (мусорные одиночные blobs не попадают в LineCounter — **важно**: считаем только подтверждённые треки);
- `minimum_iou_threshold=0.1`;
- `high_conf_det_threshold=0.6` — без confidence тоже отключён (одностадийный матч).

`SORTTracker.__init__` (`references/repos/trackers/src/trackers/core/sort/tracker.py:80-85`):
- `lost_track_buffer=30`, `frame_rate=30.0`, `track_activation_threshold=0.25`,
  `minimum_consecutive_frames=3`, `minimum_iou_threshold=0.3`.

**Рекомендация для нашего случая (blob'ы без confidence, effective fps ≈ 10–15):**
- стартовать с `SORTTracker(lost_track_buffer=60, frame_rate=<fактический fps>, minimum_consecutive_frames=2, minimum_iou_threshold=0.3)` — при `confidence=None` ByteTrack ≡ SORT, а у SORT ниже порог активации и выше min IoU (меньше ложных треков от шумовых blob'ов);
- `lost_track_buffer` в 1.5–2× от дефолта: дальние мелкие объекты часто «пропадали» из маски на 3–6 кадров; при fps=15 это ~2–4 c буфера — приемлемо для уличного трафика (человек на линии не исчезает надолго);
- `frame_rate` **обязательно передавать фактический** effective fps (из ffprobe / из замера), иначе buffer масштабируется неверно;
- запасной вариант при массовых ID-switch'ах: `BoTSORTTracker(enable_cmc=True)` — в коде CMC (sparseOptFlow, `references/repos/trackers/src/trackers/core/botsort/tracker.py:120`) включён по умолчанию и компенсирует качку камеры (риск №1 из §7); цена — несколько ms extra на кадр.
- автоподбор: у трекеров есть `search_space` (ClassVar, виден в `bytetrack/tracker.py:82-88`) и CLI `trackers tune` (Optuna) — можно прогнать по нашему ролику, чтобы не гадать порогами.

---

## 2. Детали LineCounter

### 2.1 Геометрия

Линия задана двумя точками `A=(x1,y1)`, `B=(x2,y2)` в пикселях (в конфиге —
нормализованные 0..1). Направление «+1» (in) — сторона, куда смотрит вектор A→B
(направление задаётся порядком точек при калибровке; оператор рисует от «входа» к «выходу»).

Для точки `P` — **знаковая проекция**:
- вдоль линии: `u(P) = dot(P-A, B-A) / |B-A|²` (0..1 — внутри отрезка);
- перпендикулярная сторона: `d(P) = cross(B-A, P-A)` (знак = сторона прямой).

Пересечение считаем по **отрезку движения** `P_prev → P_cur` (калиманово-сглаженное
положение трека), а не по одной точке кадра — иначе при пропуске кадров пересечение теряется.

### 2.2 Псевдокод (конкретный Python)

```python
@dataclass
class TrackState:
    last_pos: tuple[float, float] | None = None   # позиция в пред. кадре (cx, cy)
    state: int = NORMAL                            # NORMAL | COOLING
    last_event_t: float = 0.0                      # wall-clock последнего счёта трека

class LineCounter:
    def __init__(self, a_px, b_px, size_profile, *,
                 buffer_width_scale=0.75,   # ширина "буфера" вокруг линии в долях локальной высоты человека
                 cooldown_wall_s=2.0,       # wall-clock пауза per track после счёта
                 min_global_gap_s=0.30):    # мин. интервал МЕЖДУ событиями (антидребезг всего счётчика)
        self.a, self.b = np.asarray(a_px), np.asarray(b_px)
        self.ab = self.b - self.a
        self.len2 = float(self.ab @ self.ab)
        self.size_profile = size_profile          # §3: локальные h(t), buffer_width, min_dist
        self.track_states: dict[int, TrackState] = {}
        self.last_event = None                    # (t_wall, point_px, direction)

    def update(self, tracks, t_wall):
        """tracks: list[(track_id, cx, cy)] подтверждённых треков (id != -1)."""
        events = []
        for tid, cx, cy in tracks:
            st = self.track_states.setdefault(tid, TrackState())
            cur = np.array([cx, cy])
            if st.last_pos is not None:
                ev = self._check_cross(st, cur, t_wall)
                if ev:
                    events.append(ev)
            st.last_pos = cur
        # прунинг: забытые треки (не пришли > N кадров — трекер их уже потерял)
        seen = {tid for tid, _, _ in tracks}
        self._prune_stale(seen, t_wall)
        return events

    def _check_cross(self, st, cur, t_wall):
        prev = np.asarray(st.last_pos, dtype=float)
        # 1) антидубль: cooldown по wall-clock per track
        if st.state == COOLING and (t_wall - st.last_event_t) < self.cooldown_wall_s:
            return None
        # 2) стороны прямой для prev и cur
        d_prev = cross(self.ab, prev - self.a)
        d_cur  = cross(self.ab, cur  - self.a)
        if d_prev == 0 or d_cur == 0 or d_prev * d_cur > 0:
            return None                                   # на одной стороне — нет пересечения
        # 3) точка пересечения отрезков (prev->cur) × (A->B), параметр t вдоль движения
        t = d_prev / (d_prev - d_cur)
        ip = prev + t * (cur - prev)                      # intersection point
        u = float(np.dot(ip - self.a, self.ab) / self.len2)
        if not (-0.05 <= u <= 1.05):                      # пересёк ПРОДОЛЖЕНИЕ линии — не считаем
            return None                                   # (небольшой запас ±5% на дискретизацию)
        # 4) зона "буфера": локальная геометрия в точке пересечения
        h_local   = self.size_profile.person_height_px(u)   # px, §3
        buf_width = self.buffer_width_scale * h_local       # px по обе стороны линии
        if st.state == COOLING and abs(d_cur) / norm(self.ab) <= buf_width:
            return None                                   # всё ещё "у линии" — ждём ухода из буфера
        # 5) глобальный антидребезг: два разных события слишком близко в пространстве/времени
        if self.last_event is not None:
            t0, p0, _ = self.last_event
            if (t_wall - t0) < self.min_global_gap_s and \
               np.linalg.norm(ip - p0) < 0.5 * h_local:    # сдвоенный blob одного человека
                return None
        direction = +1 if d_prev > 0 else -1               # в обе стороны (знак векторного произведения)
        st.state, st.last_event_t = COOLING, t_wall
        self.last_event = (t_wall, ip, direction)
        return CrossingEvent(track_id=..., point_px=ip, u=u, direction=direction, ...)
```

Важные нюансы реализации:
- **состояние `last_pos` обновляем ВСЕГДА** (даже при cooldown) — иначе после паузы первый же шаг превращается в «длинный» отрезок prev→cur и дробит геометрию;
- позиция трека берётся из **Kalman-оценки трекера** (`sv.Detections.xyxy` после `update()`), а не из сырого centroid'а blob'а — сглаживание режет скачки;
- `tracker_id == -1` (неподтверждённый tracklet) в LineCounter **не попадает**;
- при потере трека (id исчезает > max_age у трекера) его `TrackState` удаляется после grace-period ≈ `cooldown_wall_s`; это сознательный компромисс: ID-switch → новый id → теоретически двойной счёт, но глобальный антидребезг (п. 5) гасит большинство случаев;
- при переподключении стрима (`VideoSource` restart): сбросить `last_pos` всех треков и вызвать `tracker.reset()` — иначе первый кадр после рестарта строит «отрезок» от старой позиции.

### 2.3 Антидубль: что какой случай режет

| Механизм | Режет |
|---|---|
| per-track cooldown (wall-clock, 1–3 c) | «дребезг» центра масс на границе линии после счёта |
| буфер вокруг линии (ширина ~0.75·h_local px) | повторный учёт, пока объект ещё визуально «у линии»; выход из буфера = разрешение нового счёта (человек реально отошёл и вернулся — засчитываем) |
| глобальный min gap (время + дистанция вдоль/поперёк линии < 0.5·h_local) | сдвоенные blob'ы (голова+тело), два близко идущих человека, сливающихся в один компонент |
| только подтверждённые track_id (`minimum_consecutive_frames`) | одиночные шумовые вспышки, тени-артефакты, живущие 1 кадр |

### 2.4 Логирование событий (поля `CrossingEvent` в JSONL)

```json
{"ts_wall": "2026-…T12:34:56.789Z", "frame_index": 18234, "t_video_s": 1215.6,
 "track_id": 42, "direction": +1, "counters": {"in": 37, "out": 21},
 "point_px": [913, 402], "u_on_line": 0.48, "h_local_px": 64,
 "bbox_px": [890, 350, 936, 470], "area_px": 2210,
 "prev_event_gap_s": 3.41}
```

Обязательно: `track_id`, координаты точки пересечения, направление (+1/−1), wall-clock и
index кадра (для перемотки на запись). Желательно: bbox/area объекта (аудит «кого посчитали»),
`t_video` (время в исходном потоке, если известно), дистанция до предыдущего события.
В stdout — человекочитаемая строка + периодическая сводка (in/out/total, fps обработки, lag).

---

## 3. Адаптация к разным размерам объектов

### 3.1 Механизм: size-профиль по сегментам линии

Проблема: «дальний» пешеход на широком угле в 10–30× меньше по площади, чем ближний
(отчёт 01 §7). Фиксированный minArea либо теряет дальних, либо пропускает шум.

**Формат конфигурации (YAML):** оператор задаёт N контрольных точек вдоль линии —
«в точке t от A до B человек имеет высоту H (% от высоты кадра)»:

```yaml
line:                      # нормализованные координаты 0..1 (от входа к выходу)
  a: [0.12, 0.68]
  b: [0.87, 0.41]
size_profile:              # t — доля вдоль A→B; h_pct — высота человека в % H кадра
  - {t: 0.0, h_pct: 32}    # ближний край линии: крупно
  - {t: 0.5, h_pct: 14}
  - {t: 1.0, h_pct: 6}     # дальний край: мелко
min_area_fill: 0.22        # blob заполняет ~22% от "квадрата" высоты человека (калибровать)
max_area_fill: 3.0         # max = 3× квадрата ближней высоты (режет машины/тени-полосы)
```

**Интерполяция:** `h(t)` — кусочно-линейная между контрольными точками (t отсортированы;
вне диапазона — крайние значения). Для точки пересечения с параметром `u`:

```python
def person_height_px(self, u: float) -> float:
    h_pct = interp(u, self.t_pts, self.h_pcts)     # np.interp
    return (h_pct / 100.0) * frame_h

# пороги площади в точке пересечения:
min_area_px(u) = min_area_fill * person_height_px(u) ** 2
max_area_px(u) = max_area_fill * person_height_px(0)**2   # global, не зависит от u
```

Ключевая идея: **порог площади привязан к «квадрату локальной высоты человека»**, а не к
абсолютным px — смена разрешения камеры (и даже другой камере того же типа) не ломает
настройку, т.к. всё в % H кадра. Отдельно держать глобальный `min_area_px_floor`
(например 25–60 px) как защиту от субпиксельного мусора на самом дальнем конце.

Применение: (а) фильтр blob'ов в `MotionDetector` — по **проекции центра blob'а на линию**
вычислить u и сравнить area с min/max для этой точки (blob'ы далеко от линии фильтруем
глобальными порогами); (б) `LineCounter` — buf_width, min gap (§2.2).

### 3.2 Что ещё масштабировать с размером (и что — нет)

| Параметр | Масштабирование | Обоснование |
|---|---|---|
| min/max area blob'а | ∝ h(u)² (таблица выше) | основной механизм |
| buffer_width вокруг линии (§2.3) | ∝ h(u) (0.5–1.0 × h_local) | «у линии» для дальнего = меньше px |
| min distance между событиями | ∝ h(u) (0.5–1.0 × h_local) | сдвоенные blob'ы крупнее там, где объект крупнее |
| cooldown wall-clock per track | **НЕ** масштабировать (время — время) | 1.5–3 c везде; дальние «медленнее в пикселях», но дребезг маски по времени одинаков |
| морфологические ядра OPEN/CLOSE | фиксированные МАЛЕНЬКИЕ (open 3×3, close ~9×11 px при ≤720p); **не** масштабировать под h(u) | большой close съест дальних людей целиком; если дальние «рвутся» — второй проход с ядром попоменьше/двойной close, а не увеличение (отчёт 01 §7 п.3) |
| `minimum_consecutive_frames` трекера | нет | 2–3 кадра везде |

### 3.3 UX калибровки (`calibrate.py`, cv2.setMouseCallback — достаточно)

Команда: `python -m visio_people_counter.calibrate --source <file|hls-url> --out config.yaml`.
Режимы в одном окне (клавиши переключают инструмент; кадр из источника, для файла — можно
перемотать стрелками):

1. **L**ine: ЛКМ — точка A, второе нажатие — точка B; пока линия «активна» — зажать и тянуть
   конец для корректировки (рисовалка поверх последнего кадра, `waitKey(1)`-цикл).
2. **S**ize: клик по линии добавляет control point; затем оператор мышью тянет **вертикальную
   мерку в этой точке** от ног до макушки видимого на кадре человека → программа записывает
   `h_pct = (пиксели мерки / H) * 100`. Минимум 2 точки, рекомендуется 3–5.
3. **R**OI (опц.): замкнутый полигон «зона интереса» — blob'ы вне ROI отбрасываются
   (ветки у края кадра, соседняя дорога). point-in-polygon `cv2.pointPolygonTest`.
4. **T**est: мгновенный прогон конвейера на текущем источнике с отрисовкой bbox + area +
   track_id + линии — оператор видит, какие люди проходят фильтр, и крутит `min_area_fill`
   колесом мыши до приемлемого результата (live-подстройка одного числового параметра).
5. **S**ave / ESC: сохранить config.yaml (нормализованные координаты; size-точки в % H) и выйти.

Технически — стандартный `cv2.setMouseCallback(win, on_event)` + словарь состояния
режима; отрисовка каждый кадр поверх последнего полученного кадра (для HLS просто latest
frame из пайплайна; для файла — замороженный кадр текущего seek). Сложностей нет, ~150 строк.

---

## 4. Ввод HLS / файл

### 4.1 ffprobe перед стартом (w/h/fps/кодек)

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate,codec_name \
  -of json "https://host/cam/index.m3u8"
```

Python-обёртка (`json.loads`): `w, h, fps_num/fps_den`. Для HLS r_frame_rate иногда даёт
0/NaN (HLS-контейнер без стабильного таймкода) — фолбэк: замерить wall-clock на первые 100
кадров. Полученные w/h/fps идут в конфиг рантайма (не хардкод) и в `frame_rate` трекера.

### 4.2 Точная команда ffmpeg для pipe

```python
cmd = [
    "ffmpeg", "-hide_banner", "-loglevel", "error",
    # сетевые таймауты (~5 c), чтобы read не висел вечно при обрыве:
    "-rw_timeout", "5000000",
    "-i", url,                    # HLS m3u8 (или .ts); для https с токеном:
    # "-headers", "Authorization: Bearer ...\r\n",
    "-r", str(effective_fps),     # 10–15: сознательно режем fps (§4.5, §5)
    "-s", f"{proc_w}x{proc_h}",   # до-скейл в ffmpeg (быстрее, чем cv2.resize; proc ≤ 720p по §5)
    "-f", "rawvideo", "-pix_fmt", "bgr24",
    "pipe:1",
]
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
frame_bytes = proc_w * proc_h * 3
```

Чтение: `raw = proc.stdout.read(frame_bytes)`; короткое/пустое чтение → считать
«неудачный read» (watchdog). `np.frombuffer(raw, np.uint8).reshape(h, w, 3)`.
Скейл в ffmpeg (`-s`) предпочтительнее cv2.resize: экономит CPU на декодере и даёт
меньше пикселей для MOG2 (§5); при этом координаты конфига (нормализованные) масштабируются
на `proc_w×proc_h`, а не исходные — единый масштаб по всему конвейеру.

### 4.3 Watchdog переподключения (псевдокод)

```python
class FfmpegPipeSource(VideoSource):
    def __init__(self, url, cfg):
        self._proc = None; self._fails = 0
        self.max_fails = cfg.watchdog_max_fails      # ~10–20 подряд
        self.restart_backoff_s = (1.0, 5.0, 30.0)    # экспоненциальный backoff до 30 c

    def _spawn(self):
        cmd = build_ffmpeg_cmd(self.url, ...)        # §4.2
        self._proc = subprocess.Popen(cmd, stdout=PIPE, stderr=DEVNULL)
        self._fails = 0

    def read(self):
        if self._proc is None:
            self._spawn()
        try:
            raw = self._proc.stdout.read(self.frame_bytes)
            ok = len(raw) == self.frame_bytes
        except (OSError, BrokenPipeError):
            ok = False
        if not ok or self._proc.poll() is not None:  # короткий read ИЛИ ffmpeg умер сам
            self._fails += 1
            log.warning("bad read %d/%d", self._fails, self.max_fails)
            if self._fails >= self.max_fails:
                self._restart()                       # и СООБЩАЕМ конвейеру о разрыве
            return None
        self._fails = 0
        return Frame(np.frombuffer(raw, np.uint8).reshape(self.h, self.w, 3), t_wall=time.monotonic())

    def _restart(self):
        p, self._proc = self._proc, None
        if p: 
            try: p.kill()
            except OSError: pass
        delay = min(30.0, 2 ** (self._retries % 5))   # backoff
        time.sleep(delay); self._retries += 1
        self.on_gap()                                  # колбэк в Pipeline (§4.4)
```

Дополнительно: раз в N сек проверять `lag` (wall-clock между моментами получения кадров vs
ожидаемый 1/fps) — если ffmpeg «завис, но жив» (pipe полон старых кадров), это тоже
перезапуск. Для http-сегментов можно добавить `-reconnect 1 -reconnect_streamed 1
-reconnect_delay_max 5` (ffmpeg сам тресает HTTP-сегменты) — это снижает частоту наших
полных рестартов, но **не заменяет** watchdog (HLS-playlist, DNS, TLS всё ещё падают).

### 4.4 Поведение на разрыве / EOF

- **Файл (EOF)**: `proc.stdout.read` вернёт 0 байт, ffmpeg завершится с кодом 0 →
  graceful stop: финальная сводка счётчиков, exit 0. Рестартов файла НЕТ (это не стрим);
  опция `--loop` для тестов.
- **HLS**: любой gap > watchdog-порога → рестарт ffmpeg. При рестарте конвейер обязан:
  `MotionDetector.reset()` (MOG2 перестроит фон за ~history кадров — первые ~15 c после
  переподключения помечать «warmup», события в warmup логировать с флагом),
  `TrackerAdapter.reset()`, сброс `last_pos` в LineCounter. Счётчики in/out **не** обнулять
  (это накопитель за всё время работы).
- **Короткие рывки fps** (джиттер HLS): логику строить только на wall-clock, никогда
  «кадр = 1/30 c» (грабли из отчёта 01 §8.2).

### 4.5 Задержка стрима: нужна ли «live»?

HLS по своей природе задерживает на длительность сегмента(ов) + буфер клиента:
реалистично **5–30 c** (зависит от сервера, не лечится параметрами ffmpeg с нашей стороны;
`-r`/pipe влияют только на наш догон). Для задач подсчёта трафика это почти всегда
некритично — результат нужен как статистика, а не realtime-операция. **Рекомендация:**
закладывать 10–30 c задержки в ТЗ; если заказчику нужен «live» мониторинг <5 c — просить у
камеры RTSP/HTTP-stream источник вместо HLS (RTSP через тот же ffmpeg-pipe, latency 1–3 c).
Открытый вопрос для владельца — §7.

---

## 5. Производительность на CPU

### 5.1 Оценка по кадрам (порядки величин, современный desktop/server CPU 4–8 ядер;
обязательно замерить на целевой машине бенчмарком `pipeline.py --bench`)

Операции на кадр **в масштабе обработки proc ≤ 1280×720** (после `-s` в ffmpeg):

| Операция | 1920×1080 (ориентир) | 1280×720 | Примечание |
|---|---|---|---|
| `MOG2.apply` (history=500, detectShadows=True) | ~10–25 ms | ~6–14 ms | главная статья; падает с числом пикселей |
| threshold 200 + MORPH_OPEN(3×3) + MORPH_CLOSE(~9×11) | ~3–6 ms | ~2–4 ms | 3 прохода по маске |
| `connectedComponentsWithStats` + фильтр blob'ов (N≤50) | ~3–8 ms | ~2–5 ms | |
| сборка `sv.Detections` | <1 ms | <1 ms | |
| SORT/ByteTrack (`linear_sum_assignment` на N×N, Kalman), N=10–30 объектов | ~2–6 ms | ~2–5 ms | Hungarian на 30×30 — доли ms; растёт квадратично, до N≈100 всё ещё <20 ms |
| LineCounter + лог (M событий/кадр ≈ 0) | <0.5 ms | <0.5 ms | |
| **Итого** | **~20–45 ms** | **~13–29 ms** | → теоретически 25–60 fps на 1080p, 35–70 fps на 720p |

Проверено по коду: в ядре `roboflow/trackers` **torch отсутствует** (только numpy/scipy/
supervision/opencv — `references/repos/trackers/pyproject.toml`, базовые deps); тяжёлого в
обновлении трека нет. supervision здесь — тонкая обёртка над numpy, не bottleneck.

### 5.2 Выводы

- **10–15 fps обработки более чем достаточно**: пешеход пересекает линию за сотни кадров;
  Kalman трекера + prev→cur отрезок (§2) устойчивы к такому дискретизации. Работать на
  full 30 fps смысла нет — только грузит CPU и растит pipe-буфер при просадках.
- **Ставим effective `-r 15`** (в ffmpeg, §4.2). Это сразу вдвое уменьшает нагрузку vs 30 fps.
- **Даунскейл к 1280×720 имеет смысл по умолчанию** (`-s 1280x720`): −44% пикселей →
  MOG2/морфология/CC заметно быстрее; потери в детекции дальних людей компенсируются
  size-профилем (в % H). На слабой машине (RPi-класс, старый офисный CPU) — уходить до
  960×540@10 fps: там весь конвейер ~5–10× медленнее desktop, и только так держим realtime.
- Ресурс трекера не лимитирует: при N≤30 треков это единицы ms; лимит — MOG2.
  Альтернатива для слабой CPU: `detectShadows=False` + морф-отсечение теней (дешевле).
- **Защита от роста задержки**: pipe не умеет выбрасывать старые кадры (отчёт 01 §8.3) —
  при просадке конвейера ffmpeg буферит. Меры: `-r 15`, proc ≤720p, watchdog с контролем lag
  (§4.3): если lag > ~30 c — рестарт ffmpeg = сброс задержки до базовой HLS-latency.

### 5.3 Бенчмарк (встроить в проект)

`python -m visio_people_counter.pipeline --source <file> --bench`: по кадрам логировать
ms на каждом этапе (time.monotonic вокруг вызовов), вывести p50/p95 и effective fps;
прогнать по 1–2 минутам реального ролика. Критерий приёмки: p95 одного кадра < 66 ms при
`-r 15`.

---

## 6. Что берём из референсов / что пишем сами

### 6.1 Берём (с кодом — только лицензионно чистые)

| # | Механизм | Источник (точный путь) | Лицензия |
|---|---|---|---|
| 1 | **Самого трекера**: `SORTTracker` / `ByteTrackTracker` (`update(sv.Detections) → sv.Detections(tracker_id=...)`, Kalman, Hungarian, `reset()`, timestamps, tune через `search_space`) | `references/repos/trackers/src/trackers/core/sort/tracker.py`, `references/repos/trackers/src/trackers/core/bytetrack/tracker.py` (установить `pip install trackers supervision`) | Apache-2.0 ✅ |
| 2 | Формат данных `sv.Detections` (`xyxy`, `tracker_id`, `confidence=None` → одностадийный матч) | библиотека roboflow/supervision (MIT), используется в п.1 | MIT ✅ |
| 3 | Шаблон «sign-change позиции трека + set посчитанных id» как базовая логика линии | `references/repos/yolov8-line-crossing-counter/process_and_count.py` строки 50, 58–62, 162–170 (`line_y = height // 2`, `previous_positions`, `counted_ids`) | MIT ✅ (кода там ~17 строк — берём паттерн, не файл) |
| 4 | Идея двухстадийной ассоциации и Kalman по `(cx,cy,aspect,h)` как **алгоритмическая справка** при отладке ID-switch | `references/repos/ByteTrack/yolox/tracker/byte_tracker.py` (строки ~153–210: track_thresh, buffer_size ∝ frame_rate, linear_assignment) | MIT ✅ (код НЕ переносим — 2022-стек с torch; дубль п.1) |
| 5 | Заготовка чтения файла/URL через `cv2.VideoCapture` как референс API | `references/repos/trackers/src/trackers/io/video.py:25` (`frames_from_source`) | Apache-2.0 ✅ (для HLS всё равно пишем свой, §4) |

### 6.2 Только идеи, код НЕ копировать (⚠️ без лицензии)

| # | Идея | Источник |
|---|---|---|
| 1 | Схема «лёгкого» режима: MOG2(detectShadows=True) + двойной apply + `threshold(fgmask, 200)` против серых теней(127) + OPEN(3×3)/CLOSE(11×11) + `areaTH = frameArea/250` (масштабирование порога от площади кадра — предтеча нашего size-профиля, §3) | `references/repos/PeopleCounter/PeopleCounter.py` строки 39, 72–77, 103–107 |
| 2 | Геометрия двухлинейного счёта (вход/выход) с ограничительными полосами «не считать вне ROI» | `references/repos/PeopleCounter/PeopleCounter.py` (линии 2h/5, 3h/5, up_limit/down_limit) |
| 3 | Антипаттерн для НЕ воспроизведения: присваивание локальной переменной вместо `self.state` в `Person.going_UP` — флаг фиксации пересечения в оригинале не работает | `references/repos/PeopleCounter/Person.py` (`going_UP`/`going_DOWN`) |

### 6.3 Пишем сами (нигде в готовом виде нет)

1. **VideoSource с HLS через ffmpeg-pipe** + ffprobe-обёртка + watchdog/backoff +
   warmup-флаг после переподключения (§4). Ни одно репо не умеет HLS (отчёт 03 §6).
2. **LineCounter**: наклонная линия из двух точек, пересечение по отрезку prev→cur с
   проверкой попадания в отрезок линии, ОБА направления + общий счётчик, антидубль-набор
   «state machine per track + wall-clock cooldown + буфер вокруг линии + глобальный gap»
   (§2). У JovanSk — одна горизонтальная линия и одно направление; у Gupu25 — две
   горизонтальные и сломанный флаг.
3. **SizeProfile**: адаптивные min/max area по сегментам линии, интерполяция h(t),
   масштабирование buf_width/min-gap (§3).
4. **calib-GUI** (`cv2.setMouseCallback`): линия мышью, size-мерки, ROI, live-подстройка
   порога, сохранение YAML (§3.3).
5. **Config/CLI**: YAML с нормализованными координатами, `argparse`/`click`:
   `count --source … --config …`, `calibrate`, `probe`; JSONL-лог событий + сводки (§2.4).
6. **TrackerAdapter** как связка: blob'ы без confidence → `sv.Detections(confidence=None)`,
   фильтр `tracker_id != -1` перед LineCounter, `reset()` при разрывах (§1.4).
7. Опционально (по итогам A/B): режим YOLO-детекции (ultralytics `model.track`) — тогда
   трекер тот же адаптер, но с confidence'ами и классом person; тянет torch (~2 ГБ) —
   держать отдельным extra-режимом, не в базовом CPU-конвейере.

---

## 7. Риски и открытые вопросы для владельца проекта

### 7.1 Топ-5 рисков (с mitigation'ами)

| # | Риск | Влияние | Mitigation |
|---|---|---|---|
| 1 | **Качка/дрейф камеры** (ветер на кронштейне): фон «дышит», всё становится движением; линии смещаются относительно сцены | Счёт превращается в шум, false positives ×N | Жёсткий монтаж (главное, до софта); программно: BoTSORT с CMC (`enable_cmc=True`, `references/repos/trackers/src/trackers/core/botsort/tracker.py`) компенсирует глобальное движение; ROI + увеличенный minArea; детекция «скачка фона» (средняя яркость/SSIM между N кадрами без движения) → алерт оператору. При сильной качке задача решается только стабилизацией монтажа |
| 2 | **ID-switch при перекрытиях** на линии: трекер меняет id в момент пересечения → двойной счёт (новый id ещё не в cooldown) | Прямые дубли, главный риск точности | Kalman-буфер `lost_track_buffer` (1.5–2× дефолт), prev→cur по сглаженной позиции, глобальный anti-retrigger gap (§2.2 п.5: новое событие <0.3 c и <0.5·h_local от предыдущего не учитывается), при массовых сдвигах — BoTSORT; мониторинг: логить пары событий в «подозрительном» окне для аудита |
| 3 | **HLS-задержка и обрывы** (5–30 c latency, рестарты при смене сегмента/сети) | Просадки покрытия стрима, пропуски пересечений в gap'ах | watchdog + backoff (§4.3), warmup-флаг на событиях после рестарта, контроль lag и принудительный сброс буфера; **запросить у камеры RTSP как основной источник** (latency 1–3 c) — главный вопрос §7.2 |
| 4 | **Размытие/motion blur + тени + смена освещения** (рассвет/закат, day/night-режим камеры) | Blob'ы рвутся/слипаются → пропуски и дубли; резкий переключ day/night ломает MOG2 | prev→cur отрезок уже гасит blur; двойной close / median-blur перед apply (вкл. в конфиг); `detectShadows=True` + threshold 200; детекция скачка яркости → `MOG2.reset()` + warmup; при необходимости два набора порогов day/night в конфиге |
| 5 | **Не-люди в кадре**: машины, велосипеды, животные, качающиеся деревья/флаги | Засорение счёта (машины — крупные blob'ы) | maxArea ∝ h_local² (§3), ROI-полигон (вырезать дорогу/вегетацию при калибровке), aspect-ratio фильтр как опция, min life трека (`minimum_consecutive_frames`); край — A/B с YOLO-режимом (класс person) на тех же роликах; в ТЗ зафиксировать: считаем «движение людей», а не «людей» |

Дополнительно (не в топ-5): дрейф счётчиков при длительной работе (прунинг словарей
track_states/counted — по возрасту трека), рост pipe-буфера (§5.2), ночная работа ИК-камеры.

### 7.2 Вопросы к заказчику

1. **Источник**: доступен ли RTSP/HTTP-стрим камеры вместо HLS? (определяет задержку 3 c vs 30 c). Нужен ли строгий «live» (<5 c) или ок статистика с задержкой 10–30 c?
2. **Линия vs зона**: одна линия пересечения (наше базовое решение) или ROI-зона «площадь»? Нужно ли два направления (вход/выход отдельно) + общий — подтверждаем в ТЗ.
3. **Что считаем**: именно людей (тогда YOLO-режим и torch в стеке — ~2 ГБ, медленнее) или «пешее движение» (MOG2-конвейер, лёгкий)? Что делать с велосипедами/животными — отдельно или в общий счёт?
4. **Интерфейс результатов**: CLI + JSONL/CSV-лог достаточно? Нужен ли HTTP API / периодический дамп (какой транспорт: pull endpoint? push на URL? MQTT?) и визуализация live (стрим с отрисовкой линий/bbox'ов)?
5. **Сценарий эксплуатации**: непрерывный стрим 24/7 или офлайн-прогон записей? Один инстанс = одна камера или несколько камер параллельно (определяет CPU-бюджет и мультипроцессность)? Какова целевая машина (CPU, RAM) — для выбора proc-разрешения (§5.2)?
6. **Приёмка точности**: есть ли эталонные ролики с разметкой «правильного» счёта для валидации? Какой допуск считается приемлемым (±N% per час)?

---

## Приложение: проверенные факты по коду референсов (на HEAD'ы из отчёта 03)

- `trackers` v2.6.0, Apache-2.0; базовые deps без torch (`pyproject.toml`): numpy>=2.0.2,
  supervision>=0.26.1, scipy>=1.13.1, opencv-python>=4.8.0, rich, requests, jsonargparse,
  pydeprecate; torch — только в extra `mask`/`detection`.
- Дефолты `ByteTrackTracker`: lost_track_buffer=30, frame_rate=30.0,
  track_activation_threshold=0.7, minimum_consecutive_frames=2, minimum_iou_threshold=0.1,
  high_conf_det_threshold=0.6 (`core/bytetrack/tracker.py:93-101`); `maximum_time_without_update
  = lost_track_buffer/30.0` с; docstring явно говорит: `confidence is None` → поведение как SORT.
- Дефолты `SORTTracker`: lost_track_buffer=30, frame_rate=30.0, track_activation_threshold=0.25,
  minimum_consecutive_frames=3, minimum_iou_threshold=0.3 (`core/sort/tracker.py:80-85`).
- `BoTSORTTracker(..., enable_cmc=True)` — CMC по sparseOptFlow включён по умолчанию
  (`core/botsort/tracker.py:120`).
- JovanSk: `line_y = height // 2` (строка 50), `previous_positions`/`counted_ids` (58–62),
  пересечение `prev_cy < line_y and cy >= line_y` + guard по `counted_ids` (162–170);
  трекинг — BoT-SORT из ultralytics (`model.track(persist=True)`); requirements тянет torch.
- Gupu25: `areaTH = frameArea/250` (строка 39), MOG2 detectShadows=True (72), ядра
  open 3×3 / close 11×11 (75–77), двойной apply + threshold 200 (103–104); баг флага в
  `Person.going_UP`/`going_DOWN` (локальная `state`, не `self.state`).
- ByteTrack (orig.): `buffer_size = int(frame_rate / 30.0 * args.track_buffer)`
  (`yolox/tracker/byte_tracker.py:155`), пороги через args: track_thresh=0.6, match_thresh=0.9.
