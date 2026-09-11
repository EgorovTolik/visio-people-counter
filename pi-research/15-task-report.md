# Задача 15: pipeline.py + CLI headless-режим `count` — отчёт

Дата: выполнена в `.venv` проекта (Python 3.12.3, cv2 5.0.0, numpy, trackers 2.6.0).
Ничего не коммитится (`git status`: те же untracked-дерева, что после задач 11–14;
артефакт `results/events.jsonl` от CLI-прогона удалён).

## Файлы

| Файл | Содержимое |
|---|---|
| `visio_people_counter/pipeline.py` (новый) | `Pipeline(config_path_or_Config, bench=False)`: `build()` — источник (file/hls по `cfg.video.type`) → w/h/fps → SizeProfile, MotionDetector, TrackerAdapter (frame_rate = effective_fps > 0 ? он : нативный fps источника; неизвестный fps → fallback 15 + лог), счётчики через `build_counters` (норм. координаты → px на w/h), EventLog. `run()` — цикл read → detect → track → count → log, warmup-флаг+лог, тайминги (EMA/avg fps, lag), периодические сводки, SIGINT graceful + финальная сводка, EOF → rc 0; `bench=True` — p50/p95 по этапам. `step(frame)` — один кадр (API для GUI) |
| `visio_people_counter/__main__.py` | CLI `count`: `--config`, `--video` (переопределяет `video.path`), `--bench`; валидация конфига и наличия файла ДО старта с понятными ошибками (rc=1); headless без окон. `calibrate`/`--gui` — как были «в разработке» |
| `tests/test_pipeline.py` (новый) | 3 теста на синтетическом mp4 640x360@30fps, 200 кадров: статичный текстурированный фон + «человек» (прямоугольник 40x90) «ходит на месте» ~70 кадров (MOG2 не впечатывает в фон), затем идёт по диагонали через линию a=[0.25,0.35]→b=[0.75,0.85], пересечение ≈ кадр 125–127 |
| `README.md` | секция CLI: `count` готов (примеры incl. `--video`, HLS, `--bench`); `calibrate`/`--gui` — «в разработке»; таблица архитектуры обновлена |

### Минимальные ДОБАВЛЕНИЯ в API задач 11–14 (ничего не менялось, только плюсом)
* `FfmpegPipeSource.exhausted -> bool` (video_source.py): pipeline обязан отличать
  «None = временный разрыв/переподключение» от «None = источник исчерпан». До этого
  это знание было приватно (`_eof`/`_stopped`).
* `BaseCounter.reset()` (line_counter.py, наследуется Line/Zone): сброс `_tracks`
  (last_pos/last_inside/cooldown-метки) при reconnect — трекер после `reset()` снова
  нумерует id с нуля, stale записи могли дать ложное «пересечение» от старого last_pos.
  Накопленные in/out НЕ обнуляются (по §4 отчёта 04).

## Решения по конвейеру

* **Сборка**: `__init__` только грузит конфиг; все компоненты — в `build()` после
  `source.open()`, т.к. SizeProfile/счётчики нужны w/h/fps из ffprobe/cv2.
  `run()` сам вызывает `build()`. CLI передаёт объект `Config` (после `--video`).
* **effective_fps для файла**: FileSource не умеет `-r`; pipeline пропускает кадры
  (каждый N-й, N = round(native/effective)). Для HLS/URL даунскейл и `-r` делает сам ffmpeg.
* **EOF vs разрыв**: `FileSource.read()→None` = EOF; `FfmpegPipeSource.read()→None` —
  либо разрыв (read() внутри уже отспал backoff, pipeline спит 0.2 c и повторяет),
  либо исчерпание (`exhausted=True`) → выход + финальная сводка.
* **reconnect** (хук `on_reconnect`): `detector.reset()` + `tracker.reset()` +
  `counter.reset()`, warmup-флаг сбрасывается, в stdout — лог-строка; счётчики in/out сохранены.
* **warmup**: первые ~`motion.history/2` обработанных кадров — прогрева субтрактора;
  события логируются ВСЕ (без усложнения), pipeline только выставляет `Pipeline.warmup_done`
  и печатает строку о прогреве.
* **Тайминги**: fps обработки = EMA(0.9) инстантного + средний frames/wall; lag =
  `(t_wall − t_ref_wall) − (t_video − t_ref_video)` с базой от первого кадра с известным
  t_video (для файла при обработке быстрее реалтайма lag отрицателен — мы «впереди» таймлайна).
* **Сводки**: периодическая в stdout каждые `output.summary_interval_s` (0 = только финал);
  финальная — при EOF/SIGINT/исчерпании reconnect'ов: причина, длительность, кадров,
  avg/ema fps, lag, строка `EventLog.summary()`, счёт reconnect'ов.
* **SIGINT**: handler на время `run()` ставит флаг `_stop`; цикл выходит до следующего
  кадра → финальная сводка; старый handler восстанавливается в `finally`.
* **--bench**: на каждом кадре `time.monotonic()` вокруг detect/track/count (+итог),
  списки ms, в финале таблица p50/p95 (линейная интерполяция перцентиля).

## Тесты (unittest, без внешних видеофайлов)

```
$ .venv/bin/python -m unittest tests.test_pipeline -v
test_effective_fps_5 ... ok
test_bench_prints_percentiles ... ok
test_crossing_event_and_clean_eof ... ok
Ran 3 tests in 4.002s — OK

$ .venv/bin/python -m unittest discover -s tests   # весь набор
Ran 64 tests in 7.805s — OK
```

Проверяется: `run() == 0` (EOF корректный), обработано ровно 200 кадров,
`warmup_done`, в EventLog ≥1 событие `main_line direction="in"` (JSONL — источник
истиной; верхний предел ≤3 — антидубль работает), финальная сводка напечатана
(«ФИНАЛЬНАЯ СВОДКА» + «Сводка | main_line...»), bench печатает p50/p95 по 4 этапам,
proгон с `effective_fps=5`: < 200 кадров (пропуск), wall < 60 c, avg fps обработки > 5.

## Реальный прогон CLI на синтетике (фрагмент stdout)

Видео: тот же синтетический mp4 640x360@30fps; конфиг из /tmp (линия a/b как в тесте).

```
$ .venv/bin/python -m visio_people_counter count --config /tmp/vpc_demo_*/config.yaml --bench
[pipeline] источник: file '/tmp/vpc_demo_.../synth.mp4' → 640x360, fps источника=30.0, effective_fps=нативный
[pipeline] трекер: sort @ frame_rate=30.0 (lost_track_buffer ≈ 60 кадров)
[pipeline] счётчик: id=main_line type=line count_mode=both
[pipeline] субтрактор прогрет (~100 кадров; события до прогрева тоже логируются)
[pipeline] СОБЫТИЕ main_line in track=1 @ (289,199) frame=125
[pipeline] сводка (период): кадров=148 avg=146.9 ema=75.6 lag=-3.91s (max 0.00s) | Сводка | main_line: in=1 out=0 total=1 || ВСЕГО: in=1 out=0 total=1
=== ФИНАЛЬНАЯ СВОДКА ===
причина остановки : EOF
длительность      : 1.8 c (wall)
обработано кадров : 200
fps обработки     : avg=114.2 ema=68.6 lag_last=-4.90s lag_max=0.00s
Сводка | main_line: in=1 out=0 total=1 || ВСЕГО: in=1 out=0 total=1
=== БЕНЧМАРК (ms, p50/p95) ===
detect        : p50=    6.86  p95=   13.96
track         : p50=    0.67  p95=    1.05
count         : p50=    0.03  p95=    0.05
frame (итого) : p50=    7.64  p95=   15.03
RC=0
```

JSONL: `{"counter_id": "main_line", "direction": "in", "track_id": 1, "frame_index": 125,
"x_px": 289.3, "y_px": 198.7, "t_video_s": 4.167}` — ровно одно событие в ожидаемой
точке (аналитически ≈ (287,198)) и направлении.

Проверены CLI-ошибки: `--video /tmp/nope.mp4` → `видеофайл не найден` (rc=1);
битый конфиг (`counters: not-a-list`) → сообщение валидатора config.py (rc=1);
прогон через эталонный `config.example.yaml --video <synth>` работает (2 счётчика,
событие только на main_line).

## Тайминги (бенчмарк на 640x360, desktop CPU)

p95 полного кадра ≈ 15 ms — с запасом в ~4× от критерия §5.3 отчёта 04 (< 66 ms при
`-r 15`). Главный этап — MOG2 (p95 detect ≈ 14 ms); трекер и счётчики — доли ms.

## API для задачи 16 (GUI-режим)

* `Pipeline(cfg, bench=False)`; `pipe.build()` → открыт источник и собраны компоненты:
  `pipe.source` (`.read() -> Frame`, `.width/.height/.fps`), `pipe.detector`,
  `pipe.tracker`, `pipe.size_profile`, `pipe.counters` (list, у каждого `draw(frame)` —
  базовый overlay линии/зоны+счётчика), `pipe.event_log`.
* Один кадр: `events = pipe.step(frame) -> list[CrossingEvent]` (detect→track→count→log
  + warmup-флаг + bench-замеры). GUI рисует сам по `cfg.debug.*` (show_bboxes/show_all_blobs/
  show_mask/show_counters, save_debug_frames_dir+debug_frame_step) — headless эти флаги игнорирует.
* Сводки: `pipe.print_summary(reason)`; финальный блок — внутренний (`_print_final`,
  выводится в `run()`); атрибуты-показатели: `frames_processed`, `duration_s`, `avg_fps`,
  `ema_fps`, `last_lag_s`/`max_lag_s`, `warmup_done`, `reconnects`.
* Реагирование на reconnect уже внутри источника (хук `on_reconnect` → reset detector/
  tracker/counters + warmup заново) — GUI-режиму ничего дополнительно не нужно.
* CLI: добавить в `p_count` флаги `--gui/--speed` и ветку «GUI-цикл» (cv2.imshow поверх
  того же `step()`), `calibrate` — отдельная команда (задача 16).

## Риски/остатки

* Пропуск кадров для FileSource при effective_fps>0 — грубый (каждый N-й, без интерполяции
  фазы); для HLS всё делает ffmpeg (`-r`). На точность счёта не влияет (трекер Kalman).
* `--bench` хранит таймы всех кадров в памяти (для часовых стримов — миллионы float'ов;
  при необходимости заменить на квинтиль/резервирование).
* lag для файла при обработке быстрее реалтайма отрицателен («впереди» таймлайна) — осознанно.
