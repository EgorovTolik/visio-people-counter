# Задача 17: Финальное код-ревью + мелкие фиксы — отчёт

Дата: выполнена в `.venv` проекта (Python 3.12, cv2 5.0.0/QT5, numpy, trackers 2.6.0, PyYAML; ffmpeg/ffprobe системные).
Ничего не коммитится и не стагается (`git diff --cached` пуст; дерево остаётся untracked, как после задач 11–16).
`references/` и `pi-research/tasks/` не тронуты.

## 1. Ревью по чек-листу — что проверено

- **Соответствие ТЗ**: headless по умолчанию (без окон, без ограничения скорости); `--gui` + `--speed`
  (0.25..8 — именно диапазон из ТЗ задачи 16; план 05 имел 0.25..4, ТЗ/код/CLI-валидация согласованы на 0.25..8);
  `calibrate`; счётчики line/zone, count_mode both/total; фильтры объектов вместо классификации;
  HLS через ffmpeg-pipe + watchdog-переподключение; YAML-конфиг с полной валидацией — всё на месте.
- **Корректность**: пересечение линии по отрезку prev→cur (знак d, точка на отрезке u∈[−0.05,1.05],
  направление d>0→"in"), зона через pointPolygonTest + точка на ближайшей стороне; антидубль трёхуровневый
  (per-track cooldown, буфер вокруг линии, min_global_gap с расстоянием); трекер — reset при reconnect,
  frame_rate масштабирует lost_track_buffer (проверено по пакету и тестами); тени MOG2 через shadow_threshold;
  размерные пороги — SizeProfile/глобальные доли. Утечки: `FileSource.close` → release,
  `FfmpegPipeSource.close/__del__` → kill+wait+закрытие stdout, GUI/calibrate → destroyAllWindows в finally,
  SIGINT → graceful stop с восстановлением handler. Всё корректно; мелочи — ниже.
- **Безопасность/надёжность**: ffmpeg/ffprobe — только subprocess со СПИСКОМ аргументов (ни одной строковой команды);
  `-rw_timeout` для сетевых входов; JSONL пишется с flush на событие и close при завершении (проверено: файл полный);
  битый конфиг / отсутствующий файл / пустой video.path — понятные сообщения + rc=1 (E2E-проверено).
- **Качество/консистентность**: найдены и исправлены 11 проблем (таблица ниже), в т.ч. две «мёртвых»
  конфиг-ключа (`objects.min_lifetime_frames`, `debug.save_debug_frames_dir`+`debug_frame_step`) — теперь реально
  используются кодом; README доведён до финала и сверен с фактическими argparse-флагами.

## 2. Найденные проблемы: фиксы и TODO

| # | Файл:строка (до фикса) | Суть | Статус |
|---|---|---|---|
| 1 | `visio_people_counter/calibrate.py:365` | **Критично**: `return exit_rc` — переменная не определена; любое корректное завершение `calibrate` падало с NameError (rc=1 вместо 0) | **фикс**: `return 0` |
| 2 | `calibrate.py:197–203, 246` | при ошибке `Pipeline.build()` после `source.open()` источник не закрывался; мёртвая переменная `hint_lines` до цикла | **фикс**: `pipe.close()` в except + удаление |
| 3 | `pipeline.py` (step) | конфиг-ключ `objects.min_lifetime_frames` парсил/валидировался, но нигде НЕ применялся (мёртвый ключ; docstring'ы ссылались на «уровень трекера», где его тоже не было) | **фикс**: в `Pipeline.step` и `GuiPlayer._process_frame` подтверждённый трек попадает в счётчики только после ≥N кадров наблюдения подряд (`age_frames >= min_lifetime_frames`) — анти-вспышка/антидубль короткоживущих треков (см. отчёт 13: age_frames как усиление min_lifetime) |
| 4 | `pipeline.py` / `gui.py` | конфиг-ключи `debug.save_debug_frames_dir` + `debug_frame_step` не использовались нигде (README/example обещали сохранение кадров) | **фикс**: новый helper `pipeline.save_debug_frame()`; headless — каждый N-й обработанный кадр, `--gui` — кадр с overlay. E2E: при step=50 и 200 кадрах сохранено ровно 4 валидных JPG |
| 5 | `pipeline.py:318` (run) | неиспользуемая переменная `events = self.step(frame)` | **фикс**: вызов без присваивания |
| 6 | `pipeline.py`, `__main__.py` | дублирование `_is_url()` (2 копии, в т.ч. с избыточным startswith — все префиксы и так содержат `://`) | **фикс**: одна публичная `video_source.is_url()`, импортируется в pipeline и __main__ |
| 7 | `__main__.py:153–158` (main) | при отсутствии подкоманды `build_parser()` вызывался дважды (парсер с subparsers нельзя переиспользовать для parse+help корректно) | **фикс**: один экземпляр парсера |
| 8 | `gui.py:115–119` | частный доступ `c._label_text()` из gui в line_counter; тень рисовалась ПОСЛЕ текста (накрывала его смещённо, а не под ним) | **фикс**: публичный `BaseCounter.label_text()`; сначала чёрная тень, затем белый текст |
| 9 | `config.py:4`, `line_counter.py:60` | опечатки с латиницей в docstring/комментарии («этalon», «определённo») | **фикс** |
| 10 | `config.example.yaml` (coment'ы) | комментарии расходились с реальностью: min_lifetime_frames («попасть в трекер»), botsort («с CMC» — на blob'ах CMC не активен, ведёт себя как SORT), reconnect_attempts −1 («для файла это поведение по умолчанию»), debug-блок | **фикс**: комментарии приведены к фактическому поведению |
| 11 | `README.md` | не в финальном состоянии: «в разработке (задача 16)» для готовых --gui/calibrate, нет клавиш и структуры проекта | **фикс**: финальная версия — установка, все 4 режима с примерами, таблицы клавиш GUI и calibrate, структура проекта, конфиг → ссылка на `config.example.yaml` |

### TODO / ограничения (архитектурные — НЕ переписывалось)

1. **BoTSORT без CMC**: пакету не передаются кадры (`frame=None`), поэтому CMC-шаг молча
   пропускается — на blob'ах BoTSORT ≡ SORT; ByteTrack без confidence тоже ≡ SORT (см. docstring
   `tracker_adapter.py`). Если понадобится компенсация качки камеры — передать кадр в
   `update(..., frame=...)` (требует изменения API адаптера).
2. **Stale-DISPLAY**: `GuiPlayer.available()` при устаревшей `DISPLAY=:0` без X-сервера вернёт True,
   и `cv2.imshow` упадёт штатной ошибкой OpenCV. Чистая проверка (без создания окна) дальше не может —
   осознанное решение задачи 16; на реальных headless-машинах DISPLAY обычно не задан → понятная ошибка.
3. **FileSource + effective_fps** — пропуск каждого N-го кадра без интерполяции фазы (грубая, но для
   счёта людей точность не страдает; у HLS всё делает ffmpeg `-r`).
4. **--bench** держит ms по всем кадрам в памяти (часовой стрим → миллионы float'ов); при необходимости —
   заменить на квантиль/окно.
5. `FfmpegPipeSource.close()`: если после `kill()` `wait(timeout=5)` всё же таймаутится, процесс остаётся
   zombie до GC (`__del__` как страховка). Крайне редкий случай.
6. Интерактивные циклы GUI/calibrate в headless-окружении не прогонялись (нет дисплея) — см. ручную проверку ниже.

## 3. Итоговый результат тестов

```
$ .venv/bin/python -m unittest discover -s tests -t .
Ran 84 tests in 6.903s
OK
```

(84 теста — весь набор задач 11–16; фиксы задачи 17 их не ломают: трекер-тесты не затронуты, т.к.
min_lifetime_frames применяется на уровне pipeline/счётчиков, а не в адаптере.)

## 4. E2E-смоук (синтетика 640x360@30fps, 200 кадров, «человек» пересекает линию ≈ кадр 125)

Видео сгенерировано как в tests/test_pipeline.py (`cv2.VideoWriter`, mp4v) во временный каталог /tmp.

- **probe**: `probe --video /tmp/vpc17/synth.mp4` → тип file, 640x360, fps 30.000, 6.7 c, mpeg4,
  `cv2.VideoCapture: OK (кадров по метаданным: 200)`, **rc=0**.
- **count headless**: `count --config <cfg>` → `СОБЫТИЕ main_line in track=1 @ (289,199) frame=125`,
  финальная сводка `main_line: in=1 out=0 total=1`, EOF, **rc=0**; JSONL ровно одно событие:
  `{"counter_id":"main_line","direction":"in","track_id":1,"frame_index":125,"x_px":289.3,"y_px":198.7,"t_video_s":4.167}` —
  точка совпадает с аналитической (≈(287,198)), антидубль работает (1 событие).
- **ffmpeg-pipe путь** (`video.type: hls` на локальный файл через ffmpeg, `reconnect_attempts: -1`):
  200 кадров, graceful EOF, то же одно событие, **rc=0**.
- **--gui без дисплея** (`env -u DISPLAY -u WAYLAND_DISPLAY count --gui`):
  `count --gui: ОШИБКА: нет дисплея для GUI-режима: переменные DISPLAY и WAYLAND_DISPLAY не заданы (headless-окружение).
  Запустите без --gui — headless-подсчёт работает без окон.`, **rc=1**. То же для `calibrate` (**rc=1**).
- **CLI-ошибки**: битый конфиг (`counters: not-a-list`) → `count: ошибка конфигурации: counters: ожидалось список счётчиков…` rc=1;
  отсутствующее видео → `видеофайл не найден: '/tmp/nope.mp4'` rc=1; `--speed 99` → argparse-ошибка
  `--speed: ожидалось 0.25..8, получено 99.0`; без подкоманды — help + rc=1.
- **Эталонный конфиг**: `count --config config.example.yaml --video <synth>` — оба счётчика (line+zone)
  собраны, событие только на main_line, rc=0.
- **Новое: debug-кадры** — при `save_debug_frames_dir` + `debug_frame_step: 50` headless сохранил ровно
  frame_000000/000050/000100/000150.jpg (640x360, читаются cv2.imread).

## 5. Что владельцу стоит проверить вручную (пошагово) — только в среде С дисплеем

1. `cp config.example.yaml config.yaml`; поправить `video.path` на реальное видео/стрим.
2. **Калибровка**: `.venv/bin/python -m visio_people_counter calibrate --config config.yaml --video <видео>`:
   - `l` → 2 клика (A→B, направление «in»); зелёная линия с подписями A/B;
   - `z` → 4+ кликов → Enter — замкнутый оранжевый полигон зоны;
   - `s` → клик по x дальнего человека → цифра (напр., 2 = 10% высоты) — жёлтая «мерка роста»;
   - `m` — маска движения: убедиться, что люди выделяются; если нет — подправить `motion.var_threshold`,
     `objects.min_area_fraction` в конфиге и перезапустить calibrate;
   - `a` — в stdout diff «старые → новые»; проверить в config.yaml: координаты 0..1, остальные блоки не тронуты;
   - **проверить, что при выходе (q) команда завершается с rc=0** (до фикса задачи 17 здесь был NameError).
3. **GUI**: `.venv/bin/python -m visio_people_counter count --config config.yaml --gui --speed 0.5`:
   окно с bbox+track_id, линией/зоной и крупными in/out; пробел — пауза (счётчики на застывшем кадре);
   `+`/`−` меняют скорость в статусной строке; при пересечении stdout печатает «СОБЫТИЕ …»; q — выход, rc=0.
4. **Реальное видео**: прогнать `count` headless на 5–15 мин видеозаписи: сравнить счётчики с ручным
   пересмотром; при ложных событиях поднимать `objects.min_area_fraction`/`cooldown_s`;
   при «рассыпании» треков — `tracker.lost_track_buffer` (60–120) и/или `size_profile` (включить + точки).
5. **HLS-стрим**: `count --video <.m3u8>` в фоне; проверить, что при обрыве сети идёт переподключение
   (лог-строка «reconnect #N: детектор и трекер сброшены»), а `Ctrl+C` даёт финальную сводку;
   JSONL растёт на 1 строку за событие.
6. **Отладочные кадры**: задать `debug.save_debug_frames_dir: debug/` + `debug_frame_step: 30`, прогнать
   headless — в папке каждый 30-й кадр (новое поведение задачи 17).

## 6. Файлы, изменённые задачей 17

- `visio_people_counter/calibrate.py` (фикс NameError, close при ошибке build, чистка)
- `visio_people_counter/pipeline.py` (min_lifetime_frames, save_debug_frame, is_url, чистка)
- `visio_people_counter/gui.py` (label_text, фильтр min_lifetime, debug-кадры с overlay)
- `visio_people_counter/line_counter.py` (публичный label_text, опечатка)
- `visio_people_counter/video_source.py` (is_url)
- `visio_people_counter/__main__.py` (is_url, один парсер)
- `visio_people_counter/config.py` (опечатка в docstring)
- `visio_people_counter/motion_detector.py` (docstring про min_lifetime_frames)
- `config.example.yaml` (комментарии → реальное поведение)
- `README.md` (финальная версия)
