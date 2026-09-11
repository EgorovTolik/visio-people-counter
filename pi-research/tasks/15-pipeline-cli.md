# Задача 15: pipeline.py + CLI headless-режим `count`

## Читать ПЕРЕД началом (в этом порядке)
1. /home/anatoliy/PiProjects/visio-people-counter/pi-research/05-development-plan.md — план и правила.
2. /home/anatoliy/PiProjects/visio-people-counter/config.example.yaml — весь конфиг (pipeline связывает все блоки).
3. Отчёты pi-research/11-task-report.md … 14-task-report.md — API всех готовых модулей; код visio_people_counter/*.py читать обязательно перед сборкой.
4. pi-research/04-implementation-notes.md — §4.5 (EOF/разрыв), §5.3 (бенчмарк --bench).

## Среда
ТОЛЬКО /home/anatoliy/PiProjects/visio-people-counter/.venv/bin/python. Ничего не устанавливать, ничего не коммитить.

## Делать (в /home/anatoliy/PiProjects/visio-people-counter/)
1. `pipeline.py`:
   - `Pipeline(config_path)`: строит всё из конфига — VideoSource (file/hls по cfg.video.type),
     MotionDetector, SizeProfile (w/h из источника), TrackerAdapter (frame_rate = effective fps:
     effective_fps>0 → он, иначе нативный fps источника; для HLS с effective_fps=0 использовать 15 как fallback и логировать),
     счётчики через build_counters (нормализованные координаты → px на w/h), EventLog;
   - `run()`: цикл read → detect → tracker.update → counters.update → event_log;
     при reconnect источника (хук on_reconnect): detector.reset() + tracker.reset();
     warmup: первые ~history/2 кадров события НЕ логируем в агрегаты? — нет, просто не усложнять:
     логировать всё, но добавить флаг Pipeline.warmup_done и лог-строку о прогреве субтрактора;
   - тайминги: fps обработки (EMA), lag (t_wall − t_video при известном); периодическая сводка
     в stdout каждые output.summary_interval_s; SIGINT → graceful stop + финальная сводка (EventLog.final summary,
     длительность, обработано кадров, fps); EOF файла → то же;
   - `--bench`-режим: на каждом N-м кадре замерять время этапов (detect/track/count) и в финале печатать p50/p95 —
     реализация простая (список таймов, без тяжёлых структур).
2. CLI (`__main__.py`): `count` команда:
   - `python -m visio_people_counter count --config config.yaml [--video PATH_OR_URL] [--bench]`;
   - `--video` переопределяет cfg.video.path (удобно для разовых прогонов); валидация конфига до старта,
     понятные ошибки; headless: без окон, максимальная скорость.
3. Тесты tests/test_pipeline.py (unittest):
   - синтезировать mp4 ~200 кадров 640x360: статичный фон + «человек» (прямоугольник 40x90) идёт по диагонали
     через заданную в тестовом конфиге линию; после warmup (~100 кадров) и прохождения — EventLog содержит ≥1
     событие с ожидаемым направлением; финальная сводка печатается; EOF корректный (run() вернул 0);
   - прогон с effective_fps=5: длительность теста не взрывается, fps обработки > 5.
4. README.md: обновить секцию CLI — `count` готов (примеры), `calibrate`/`--gui` помечены «в разработке».

## Правила
- Все bash с timeout; без интерактивных команд; ничего не коммитить.
- Не менять API задач 11–14 без крайней необходимости — если изменил, отметить в отчёте.
- Тесты должны проходить БЕЗ внешних видеофайлов (всё синтезировать).

## Отчёт
/home/anatoliy/PiProjects/visio-people-counter/pi-research/15-task-report.md: файлы, вывод тестов и реального
прогона count на синтетике (фрагмент stdout), тайминги, API для задачи 16 (GUI-режим).

## Формат ответа (СТРОГО)
ТОЛЬКО короткое резюме (10–20 строк + результат тестов) + путь к отчёту.
