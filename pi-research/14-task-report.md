# Задача 14: line_counter.py (линия + зона) + event_log.py — отчёт

Дата: выполнена в `.venv` проекта (Python 3.12.3, cv2 5.0.0, numpy 2.5.3).
Ничего не коммитится (`git status`: только те же untracked-файлы, что после задач 11–13).
API задач 11–13 (config.py, motion_detector.py, tracker_adapter.py, size_profile.py)
**не изменялся** — счётчики только читают `cfg.counters` / `cfg.output` и принимают
`list[TrackedObject]`. Изменён только README.md (таблица модулей: статус «готово»).

## Файлы

| Файл | Содержимое |
|---|---|
| `visio_people_counter/line_counter.py` | `CrossingEvent` (dataclass + `to_dict()`), `BaseCounter(ABC)` (`update()`, `counters -> dict`, `draw(frame)`), `LineCounter` (наклонная линия A→B, оба направления, антидубль), `ZoneCounter` (полигон, point-in-polygon, in/out), фабрика `build_counters(counter_cfgs, w, h, size_profile=None)` |
| `visio_people_counter/event_log.py` | `EventLog(OutputConfig или Config)`: JSONL-запись (`""` = не писать; каталог создаётся сам, режим append + flush), агрегаты per-counter и глобальные, `summary() -> str`, `close()`/контекстный менеджер |
| `tests/test_counters.py` | 16 тестов (unittest, чистая геометрия на синтетических `TrackedObject`, без видео) |
| `README.md` | таблица архитектуры: line_counter/event_log → «готово (отчёт 14)» |

## Решения по геометрии и антидублю (что реализовано из §2 отчёта 04)

### Масштабирование координат
Нормализованные координаты конфига масштабируются **в конструкторе** счётчика на
фактическое разрешение `w/h`, которое pipeline передаёт при создании
(`LineCounter(cfg, w, h, size_profile=None)`, `ZoneCounter(cfg, w, h)`). Counter не
знает про источник кадров; при смене разрешения пересоздают счётчики (задача 15).

### Линия (LineCounter) — псевдокод §2.2 реализован построчно
- Сторона точки: `d(P) = cross(B−A, P−A)`; событие только при смене знака d между
  prev и cur (пересечение по **отрезку** prev→cur, калиманово-сглаженная позиция cx,cy).
- Точка пересечения `ip = prev + t·(cur−prev)`, параметр `t = d_prev/(d_prev−d_cur)`;
  считаем только если `u(ip) ∈ [−0.05, 1.05]` — т.е. пересечение **на отрезке** линии,
  а не на его продолжении (±5% запас на дискретизацию).
- Направление: `d_prev > 0` → `"in"`, иначе `"out"` (конвенция §2.2). Для типичной
  диагонали A=верхне-левый, B=нижне-правый это = «слева-направо по A→B — in».
- Считаются только подтверждённые треки (`track_id != -1`); их состояние не хранится.
- `last_pos` обновляется **всегда** (даже в cooldown) — иначе после паузы первый шаг
  становится «длинным» отрезком (§2.2, нюансы).
- Прунинг состояний: трек не виден > 60 s → `_TrackState` удаляется (анти-утечка).

### Антидубль (§2.3) — все три механизма + подтверждённые треки
1. **per-track cooldown** (`cooldown_s`, wall-clock): повторное пересечение тем же
   track_id внутри окна отбрасывается;
2. **буфер вокруг линии**: пока объект «у линии» (расстояние до прямой ≤
   `buffer_width_scale × h_local`) повторный счёт заблокирован даже после cooldown —
   нужно выйти из буфера и вернуться. `h_local` берётся из `SizeProfile.person_height_px`
   (если передан И адаптивный); **иначе фиксированный масштаб = 0.5 × длина линии**
   (решение задачи 14: без профиля «высоты человека» не определена, а масштаб от длины
   линии переживает смену разрешения);
3. **min_global_gap_s**: новое событие отбрасывается, если прошло < gap c И точка
   пересечения ближе `0.5 × h_local` к предыдущему событию счётчика (режет сдвоенные
   blob'ы; два человека в разных концах длинной линии внутри 0.3 c не глушат друг друга).

### Зона (ZoneCounter)
- Point-in-polygon — `cv2.pointPolygonTest` (полигон ≥3 точек, нормализованный → px);
  `"in"` = снаружи→внутрь, `"out"` = внутрь→снаружу.
- Точка события — пересечение отрезка prev→cur с ближайшей стороной полигона
  (чистая геометрия; если не находится — позиция трека).
- Антидубль: per-track cooldown + global gap (масштаб h_local для зоны = длина самой
  длинной стороны полигона). Буфера вокруг границы нет — в конфиге зоны нет
  `buffer_width_scale` (по ТЗ задачи 14).

### count_mode
Влияет только на представление `counters`: `"both"` → `{"in", "out", "total"}`,
`"total"` → `{"total"}`. События обоих направлений фиксируются/логируются всегда —
разделение доступно в JSONL и в агрегатах `EventLog`.

### CrossingEvent / EventLog (§2.4)
Поля: `counter_id, direction, track_id, t_wall, x_px, y_px, frame_index, t_video`
(`None` = нет). `to_dict()` для JSONL добавляет `ts_wall` (ISO-UTC в момент записи);
`t_video_s` пишется только если известно. `EventLog` принимает блок `output` либо
корневой `Config`; агрегаты per-counter in/out/total + глобальные; `summary()` —
строка вида `Сводка | main_line: in=.. out=.. total=.. || ВСЕГО: in=.. out=.. total=..`.

## API для задачи 15 (pipeline)

```python
from visio_people_counter.line_counter import build_counters, BaseCounter, CrossingEvent
from visio_people_counter.event_log import EventLog

counters = build_counters(cfg.counters, w, h, size_profile=size_profile)  # после первого кадра/ffprobe
elog     = EventLog(cfg.output)            # или EventLog(cfg)
for frame in source:                        # Frame(img, frame_index, t_wall, t_video?)
    objs = tracker.update(blobs)
    for c in counters:
        events = c.update(objs, t_wall, t_video=frame.t_video, frame_index=frame.frame_index)
        elog.log_events(events)             # JSONL + агрегаты
print(elog.summary())                       # каждые summary_interval_s и при финале
c.counters                                  # {"in":..,"out":..,"total":..} / {"total":..}
c.draw(img)                                 # overlay (заготовка, доработка задача 16)
```

`update(objects, t_wall)` — обязательные параметры; `t_video`, `frame_index` —
опциональные kwargs. При переподключении стрима pipeline должен **пересоздать**
счётчики (или хотя бы трекер+дetection reset), чтобы сбросить `last_pos` (§2.2) —
накопленные in/out в `EventLog` при этом не обнуляются.

## Вывод тестов

`.venv/bin/python -m unittest tests.test_counters -v`: **16/16 OK** (0.002 s):
геометрия линии (in/out/вдоль/продолжение отрезка), антидубль (bounce < cooldown → 1,
выход из буфера → засчитано, global gap < 0.3 c → второе отброшено), трек −1 не
считается, зона (вход/выход/движение внутри → 0, total суммирует, both разделяет,
bounce у границы), `CrossingEvent.to_dict`, EventLog (JSONL парсится по строкам,
агрегаты сходятся, пустой путь = без файла).
Полный набор: `.venv/bin/python -m unittest discover -s tests`: **61/61 OK** — задачи 11–13 не сломаны.
Смоук: `build_counters` на config.example.yaml (1280×720 + SizeProfile) строит
LineCounter+ZoneCounter, `draw()` рисует overlay без ошибок.
