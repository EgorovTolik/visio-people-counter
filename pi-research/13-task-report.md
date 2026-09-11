# Задача 13: tracker_adapter.py — отчёт

Дата: выполнена в `.venv` проекта (supervision 0.30.2, trackers 2.6.0, numpy 2.5.3).
Ничего не коммитится (`git status`: только untracked-файлы, как после задач 11–12).
API задач 11/12 (config.py, motion_detector.py) **не изменялся** — адаптер только
читает `cfg.tracker` и принимает `list[Blob]`.

## Файлы

| Файл | Содержимое |
|---|---|
| `visio_people_counter/tracker_adapter.py` | `TrackedObject` (dataclass: track_id, x,y,w,h,cx,cy, age_frames), `TrackerAdapter(cfg, frame_rate=None)`: строит SORT/ByteTrack/BoTSORT по `cfg.tracker.type`; `update(blobs) -> list[TrackedObject]`; `reset()`; `set_frame_rate(fps)`; свойства `tracker_type`, `maximum_frames_without_update`, `frame_rate`; константа `DEFAULT_FRAME_RATE = 15.0` |
| `tests/test_tracker.py` | 9 тестов (unittest, чистая синтетика без cv2/видео) |
| `README.md` | секция «Структура» → «Архитектура (конвейер)»: таблица модулей по мере готовности со ссылками на отчёты 11–13 и задачи 14–16 |

## Почему confidence=None — кратко (полное объяснение в docstring модуля)

Blob'ы motion-детектора не имеют скорей, поэтому `sv.Detections(xyxy=..., confidence=None)`.
Пакет при `confidence is None` трактует все det как 1.0 (`trackers.utils.detections.default_confidences`)
→ порог активации трека всегда пройден (трек создаётся на любой unmatched det), а
двухстадийная low-confidence ассоциация ByteTrack/BoTSORT отключается: **ByteTrack ≡ SORT**
на наших входах. Реальное различие типов остаётся только CMC у BoTSORT — но он требует
кадры, а на blob-уровне кадры не передаются (`frame=None` → шаг CMC молча пропускается).

## Проверенные параметры трекеров (коды референса, references/repos/trackers/src/trackers)

Все три конструктора принимают `lost_track_buffer`, `frame_rate`,
`minimum_consecutive_frames` — и все три имеют `reset()`:

* **SORT** — `core/sort/tracker.py:80`
  `__init__(lost_track_buffer=30, frame_rate=30.0, track_activation_threshold=0.25, minimum_consecutive_frames=3, minimum_iou_threshold=0.3, ...)`;
  `reset()` — строка 255 (`tracks=[]`, сброс счётчика id).
* **ByteTrack** — `core/bytetrack/tracker.py:90`
  `__init__(lost_track_buffer=30, frame_rate=30.0, track_activation_threshold=0.7, minimum_consecutive_frames=2, minimum_iou_threshold=0.1, high_conf_det_threshold=0.6, ...)`;
  докстринг строки 43–48: *«When input detections carry no confidence (`detections.confidence is None`),
  ByteTrack falls back to a single-stage IoU match equivalent to SORT»*.
* **BoTSORT** — `core/botsort/tracker.py:110`
  `__init__(lost_track_buffer=30, frame_rate=30.0, ..., minimum_iou_threshold_first_assoc=0.2, minimum_iou_threshold_second_assoc=0.5, minimum_iou_threshold_unconfirmed_assoc=0.3, enable_cmc=True, ...)`;
  докстринг строки 184–187: *«When `frame=None` and `enable_cmc=True`, CMC is silently skipped for that step»*.
* **Масштабирование buffer** — `core/base.py:327` `_compute_maximum_frames_without_update`:
  `lost_track_buffer` считается в кадрах **30 FPS**: `max(1, ceil(frame_rate / 30 * lost_track_buffer))`.
  Т.е. при fps=15 и конфиге 60 трек живёт ~30 кадров (~2 с), при fps=30 — 60 кадров (2 с).
* **Нумерация id** — `core/base.py:646–651`: первый трек получает `track_id = 0` (не 1);
  неподтверждённые/несвязанные — `-1`. `reset()` обнуляет счётчик.

## Решения по маппингу конфига → трекера

* `cfg.tracker.minimum_iou_threshold` передаётся в SORT/ByteTrack как есть; у BoTSORT
  (3 порога, в конфиге один) — в `minimum_iou_threshold_first_assoc` и
  `minimum_iou_threshold_unconfirmed_assoc`, second-stage остаётся дефолтным 0.5.
* `frame_rate`: конструктор принимает эффективный fps; по умолчанию `DEFAULT_FRAME_RATE = 15.0`
  (pipeline ещё не знает fps на момент создания). Пакет считает buffer в конструкторе,
  поэтому «на лету» fps менять нельзя — `set_frame_rate(fps)` пересобирает трекер
  (сбрасывая состояние; вызывать до начала потока / при reconnect вместе с `reset()`).
* Порядок результата: пакет возвращает ровно по одному выводу на каждый входной det, в том же
  порядке → адаптер зипит blob'ы с `result.tracker_id` (координаты — исходные из Blob).
* `age_frames`: ведёт сам адаптер (track_id → счётчик кадров подряд); для неподтверждённых (-1) = 0.

## Результаты тестов

```
$ .venv/bin/python -m unittest tests.test_tracker -v
test_all_tracker_types_smoke ... ok
test_default_frame_rate ... ok
test_empty_frames_do_not_crash_and_close_tracks ... ok
test_id_preserved_through_occlusion ... ok
test_new_object_gets_new_id ... ok
test_reset_clears_state_and_ids_restart ... ok
test_set_frame_rate_rescales_buffer_and_resets ... ok
test_short_gap_keeps_track ... ok
test_unconfirmed_single_frame_is_minus_one ... ok
----------------------------------------------------------------------
Ran 9 tests in 0.046s

OK

$ .venv/bin/python -m unittest discover -s tests   # ВСЕ тесты проекта
.............................................
----------------------------------------------------------------------
Ran 45 tests in 4.095s

OK
```

Что проверяют (всё на синтетике: прямоугольники 60x120, постоянное движение по прямой):
* `test_id_preserved_through_occlusion` — A пропадает 9 кадров (≤ buffer=30 при fps 15) →
  тот же id после возврата; B весь период с одним другим id.
* `test_new_object_gets_new_id` — объект C, появившийся на кадре 25, получает новый id; A не меняется.
* `test_unconfirmed_single_frame_is_minus_one` — 1 кадр → -1; 2-й подряд кадр → подтверждение (id≥0).
* `test_empty_frames_do_not_crash_and_close_tracks` — 50 пустых кадров (> buffer) не падают,
  после них объект на том же месте получает **новый** id (старый трек закрыт).
* `test_short_gap_keeps_track` — разрыв 3 кадра → тот же id.
* `test_reset_clears_state_and_ids_restart` — после reset() первый трек снова id=0, age сброшен.
* `test_set_frame_rate_rescales_buffer_and_resets` — buffer пересчитывается (15→30 fps: 30→60 кадров),
  состояние сбрасывается; fps≤0 → ValueError.
* `test_all_tracker_types_smoke` — sort/bytetrack/botsort: id≥0 со 2-го кадра, стабилен 25 кадров.

## API для задачи 14+

```python
from visio_people_counter.tracker_adapter import TrackerAdapter, TrackedObject, DEFAULT_FRAME_RATE

trk = TrackerAdapter(cfg)                    # cfg — Config (блок cfg.tracker); fps по умолчанию 15
trk.set_frame_rate(effective_fps)            # ДО начала потока: фактический fps источника
objs: list[TrackedObject] = trk.update(blobs)   # blobs из MotionDetector.detect(); [] ОБЯЗАТЕЛЕН при пустом кадре
# TrackedObject: .track_id (int, -1 = неподтверждён — НЕ пускать в подсчёт),
#                .x .y .w .h (px), .cx .cy (float), .age_frames
trk.maximum_frames_without_update            # сколько кадров живёт пропавший трек при текущем fps
trk.reset()                                  # при reconnect (хук on_reconnect FfmpegPipeSource) + set_frame_rate при смене fps
```

Для счётчиков (задача 14): брать только `track_id >= 0`; cooldown/антидубль — по track_id.
`age_frames` можно использовать как замену/усиление `objects.min_lifetime_frames`
(на уровне трека, а не blob'а).

## Риски / заметки

* BoTSORT без кадров = SORT + другой набор IoU-порогов; CMC реально заработает только
  если задача 15/16 расширит `update(blobs, frame=...)`. Задел есть (у всех трекеров
  `update(dets, frame=None)`), в отчёте задачи 15 это нужно учесть.
* Семантика `lost_track_buffer` из комментария config.example.yaml («сколько КАДРОВ») не
  совпадает с пакетом: пакет считает в кадрах 30 FPS и масштабирует по fps. При fps=15
  конфиг 60 → ~2 с, а не 4 с. Значения конфига менять не стал (это вопрос владельца);
  при желании «N секунд» = `lost_track_buffer ≈ 30 * N` (не зависит от fps).
* Kalman-предсказание хорошо держит прямолинейное движение; на резких поворотах при
  длинном перекрытии возможны id-switch — лечится увеличением lost_track_buffer и
  уменьшением minimum_iou_threshold, это настройки конфига, не баг.
