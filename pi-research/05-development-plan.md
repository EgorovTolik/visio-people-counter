# План разработки visio-people-counter

Основа: отчёты pi-research/01–04, config.example.yaml (схема конфига — эталон для config.py).
Среда: `.venv` (system-site-packages) — cv2 5.0.0, numpy 2.5.3, supervision 0.30.2, trackers 2.6.0, PyYAML; ffmpeg/ffprobe системные.
Правила: субагенты строго по одному; код пишется в `visio_people_counter/`; коммиты — только после подтверждения владельца; `references/repos/Gupu25/PeopleCounter` — без лицензии, копировать код нельзя (только идеи).

## Задачи (последовательно)
1. **11-skeleton-source**: структура пакета, config.py (YAML→dataclass + валидация по schema из config.example.yaml), video_source.py (FileSource, FfmpegPipeSource с ffprobe и watchdog переподключения), CLI-команда `probe`, requirements.txt, README.md (заготовка). Смоук-тесты без реального видео.
2. **12-motion**: motion_detector.py (MOG2/KNN + тени + морфология + CCW + фильтры objects.*), size_profile.py (интерполяция контрольных точек → min/max area в точке).
3. **13-tracker**: tracker_adapter.py (bbox'ы blob'ов ↔ sv.Detections(confidence=None), SORT/ByteTrack/BoTSORT из `trackers`, reset()).
4. **14-counters**: line_counter.py (наклонная линия, prev→cur, оба направления, антидубль: cooldown per track + buffer + global gap) + zone counter (полигон, in/out) + event_log.py (JSONL, агрегаты, сводки).
5. **15-pipeline-cli**: pipeline.py (главный цикл, тайминги, SIGINT graceful), CLI headless-режим `count` (по умолчанию: без окна, полная скорость), режимы из ТЗ: --speed только для GUI.
6. **16-gui-calibrate**: `--gui` (окно, overlay bbox/линии/счётчики, --speed 0.25..4), `calibrate` (cv2.setMouseCallback: линия/зона/size-точки → запись в config.yaml).
7. **17-review**: прогон reviewer-агента по коду + фиксы; итоговая сводка владельцу.

## Критерии приёмки каждой задачи
- код импортируется и модульные смоук-тесты (`python -m visio_people_counter ... --selftest` или pytest) проходят в .venv;
- конфиг config.yaml (копия example) проходит валидацию;
- ничего не коммитится, references/ не трогается.
