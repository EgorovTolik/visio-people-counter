# Задача 12: motion_detector.py + size_profile.py — отчёт

Дата: выполнена в `.venv` проекта (cv2 5.0.0, numpy 2.5.3). Ничего не коммитится
(`git status`: только untracked-файлы, как и после задачи 11). API задач 11
(config.py, video_source.py) **не изменялся**.

## Файлы

| Файл | Содержимое |
|---|---|
| `visio_people_counter/size_profile.py` | `SizeProfile(sp_cfg, objects_cfg, w, h)`: `person_height_px(x)` (линейная интерполяция контрольных точек по x, за краями — крайнее значение), `min_area_at(x)=k_min·h²`, `max_area_at(x)=k_max·h²`, `buffer_width_px(x, scale)=scale·h`, свойство `.adaptive`; при отключённом/пустом профиле (<2 точек) — глобальные пороги `objects.min/max_area_fraction · w·h` |
| `visio_people_counter/motion_detector.py` | `Blob(x,y,w,h,area,cx,cy)` (px исходного кадра, `area` = число пикселей компонента, не bbox); `MotionDetector(cfg)`: MOG2(`history`,`varThreshold`,`detectShadows`) или KNN(`history`,`dist2Threshold`) по `cfg.motion.method`; `detect(frame, size_profile=None)` → apply → тени → MORPH_OPEN(morph_open) → MORPH_CLOSE(morph_close) → connectedComponentsWithStats(8) → фильтры; `filter_blobs(stats, centroids, counts, objects_cfg, min_area_at, max_area_at)` — вынесен в отдельную (тестируемую) функцию; `reset()` |
| `tests/test_motion.py` | 15 тестов: движущийся «человек» MOG2/KNN (IoU>0.3), стоящий объект исчезает, тени E2E, фильтры E2E + unit, SizeProfile (интерполяция/край/вкл-выкл/buffer), reset |

## Результаты тестов

```
$ .venv/bin/python -m unittest discover -s tests
....................................
----------------------------------------------------------------------
Ran 36 tests in 4.058s

OK
```

(21 теста задачи 11 без изменений + 15 новых; время на новые ~2 c, всё на
синтетике, детерминировано: seed фиксирован, cv2-операции детерминированны.)

## Выбранные дефолты / нюансы

### Как режутся тени
* Только для MOG2 (в маске есть значение 127 — «тень»). Отсечение:
  `mask = np.where(fgmask >= shadow_threshold, 255, 0)` — т.е. все пиксели
  < `shadow_threshold` (тени=127 и прочие слабые) обнуляются; при
  `shadow_threshold <= 127` остаётся ровно `==255`. При `detect_shadows=false`
  отсечения нет — тени выглядят как движение и режутся морфологией/фильтрами.
* KNN: в cv2 5.0 `createBackgroundSubtractorKNN(..., detectShadows=True)` тоже
  умеет выдавать пиксели 127, но мы для KNN отсечение **не применяем**
  (флаг `_detect_shadows = detect_shadows AND method=="mog2"`): такие пиксели
  считаются foreground. Если в A/B-экспериментах это помешает — флаг на месте.

### Порядок фильтров (в `filter_blobs`)
1. `min_bbox_side_px` по w и h; 2. `aspect_ratio_range` (w/h);
3. `min_fill` (area/(w·h)); 4. площадь: `min_area_at(cx) <= area <= max_area_at(cx)`
   (через `SizeProfile` либо глобальные доли кадра).
* `min_lifetime_frames` **не применяется здесь** — уровень трекера (задача 13), как и ТЗ.

### Поведение MOG2 в cv2 5.0, которое важно знать (проверено экспериментально)
* Warmup: при `history=100` стабильная детекция движущегося объекта начинается
  примерно с кадра ~25–30; тесты дают 45–55 кадров warmup.
* **Imprinting**: пиксели, которые объект закрывает много кадров подряд (или
  стоит на месте), со временем усваиваются моделью фона — стоящий объект
  перестаёт детектироваться через ~30–60 кадров (это и есть предмет теста
  `test_standing_object_disappears`), а у движущегося «стачивается» задний край.
  В синтетике тестовой сцены человек идёт по циклу периода 48 кадров шагом
  10 px — за warmup пиксель не успевает запомнить объект, и blob стабилен.
* Широкие объекты при быстрой «прыгающей» траектории MOG2 дробит на фрагменты;
  поэтому E2E-тест aspect-фильтра сделан через `filter_blobs` на синтетических
  CC-stats (детерминированно), а E2E «широкий объект виден целиком» проверен в
  отладке на KNN — в тесте оставлен unit-путь.

### Прочее
* `MotionDetector.reset()` вызывает `self._bg.clear()` — у cv2 5.0 у
  BackgroundSubtractor **нет метода `reset()`** (только `clear()`). Вызывать при
  reconnect из pipeline (хук `on_reconnect` FfmpegPipeSource, задача 15).
* `Blob.area` считается через `np.bincount(labels.ravel())` — это площадь
  компонента в пикселях (для min_fill и area-фильтров), bbox-area не подходит.
* `SizeProfile` в адаптивном режиме требует включённый блок **и ≥2** контрольных
  точек; иначе прозрачно работает глобальный режим из `cfg.objects`.
  `person_height_px` в нефункциональном режиме возвращает высоту кадра
  (нейтрально — влияет только на `buffer_width_px`).
* Точки контроля сортируются по x; дубликаты x-координат не ломают интерполяцию.

## API для задач 13+

```python
from visio_people_counter.motion_detector import MotionDetector, Blob, filter_blobs
det = MotionDetector(cfg)                      # cfg — Config из задачи 11
blobs: list[Blob] = det.detect(frame_image, size_profile=None)  # BGR uint8
# Blob: .x .y .w .h (int, px), .area (int, пиксели), .cx .cy (float)
det.reset()                                    # при reconnect

from visio_people_counter.size_profile import SizeProfile
sp = SizeProfile(cfg.size_profile, cfg.objects, w=src.width, h=src.height)
h_px   = sp.person_height_px(x_px)             # для buffer линии (задача 14)
bw_px  = sp.buffer_width_px(x_px, scale)       # scale — counter.buffer_width_scale
sp.min_area_at(cx) / sp.max_area_at(cx)        # уже используются в detect()
```

Для трекера (задача 13): bbox'ы = `(x, y, x+w, y+h)`; confidence отсутствует —
`sv.Detections(confidence=None)`; `min_lifetime_frames` и
`minimum_consecutive_frames` применяются на уровне адаптера трека.
