# Задача 11: Скетелет проекта, config.py, video_source.py, CLI probe — отчёт

Дата: выполнена в `.venv` проекта (cv2 5.0.0, numpy 2.5.3, PyYAML; ffmpeg/ffprobe системные).
Ничего не коммитится (`git status`: только untracked-файлы).

## Что сделано (файлы)

| Файл | Содержимое |
|---|---|
| `visio_people_counter/__init__.py` | версия пакета (`__version__ = "0.1.0"`) |
| `visio_people_counter/__main__.py` | CLI argparse: `probe` (реализована), `count`/`calibrate` — заглушки «не реализовано» + exit 2; `--version` |
| `visio_people_counter/config.py` | dataclass'ы 1:1 со всеми блоками config.example.yaml + `Config.load(path)`, `Config.from_dict(d)`, `Config.save(cfg, path)`, `cfg.to_dict()`; `ConfigError` с сообщениями `блок.ключ: ожидалось X, получено Y` |
| `visio_people_counter/video_source.py` | `Frame` (image BGR, index, t_wall, t_video), `VideoSource` (ABC: open/read/close + width/height/fps, context manager), `FileSource` (cv2.VideoCapture, loop_file = перемотка в начало), `FfmpegPipeSource` (ffprobe → ffmpeg-pipe, watchdog по §4.3 отчёта 04, хук `on_reconnect`, graceful EOF файла через pipe), утилита `ffprobe_info()` |
| `tests/test_config.py` | 14 тестов: example-конфиг загружается; минимальный/пустой конфиг проходит; save→load roundtrip; битые значения (координата 1.5, ядра [4,4]/[0,9], method "optical_flow", type "rtsp"/"circle", polygon из 2 точек, str вместо int, неизвестный ключ) → `ConfigError` с текстом |
| `tests/test_video_source.py` | 7 тестов на синтетическом mp4 (960x540, 30 fps, 30 кадров, 'mp4v', движущийся квадрат): FileSource читает все кадры + loop_file; FfmpegPipeSource с тем же файлом — та же последовательность кадров (±1), effective_fps=10 → ~10 кадров, max_width=640 → 640x360 (высота чётная); ffprobe-сводка; недоступный вход + reconnect_attempts=-1 → VideoSourceError |
| `README.md` | заготовка: что это, установка (.venv), примеры CLI (probe/count/gui/calibrate, «в разработке»), ссылка на config.example.yaml, структура модулей |

## Результаты тестов

```
$ .venv/bin/python -m unittest discover -s tests
.....................
----------------------------------------------------------------------
Ran 21 tests in 1.701s

OK
```

Смоук CLI (на временном mp4 640x360, 30 fps):

```
$ .venv/bin/python -m visio_people_counter probe --video /tmp/.../demo.mp4
=== visio-people-counter probe ===
вход          : /tmp/.../demo.mp4
тип           : file
контейнер     : mov,mp4,m4a,3gp,3g2,mj2
разрешение    : 640x360
fps           : 30.000
длительность  : 1.0 c
кодек         : mpeg4
cv2.VideoCapture: OK (открывается; кадров по метаданным: 30)     # rc=0

$ .venv/bin/python -m visio_people_counter count --config config.yaml
count: не реализовано (задача 15-pipeline-cli)                    # rc=2
$ .venv/bin/python -m visio_people_counter calibrate
calibrate: не реализовано (задача 16-gui-calibrate)               # rc=2
```

## Отклонения от спецификации и почему

1. **Скорость в тесте effective_fps=10**: wall-time декода файла ffmpeg'ом НЕ равен
   длительности видео (ffmpeg с файловым входом режет fps быстрее реального времени,
   ~0.2 c на 1 c видео). Тест проверяет ожидаемое w/h (640x360), число кадров (~10 при
   -r 10) и что конвейер не медленнее реального времени (elapsed < 5 c). Жёсткий
   wall-clock-порог «≥0.5 c» некорректен для файловых входов — отброшен.
2. **`-s WxH` вместо `-vf scale`** в команде ffmpeg: по §4.2 отчёта 04 (`-s`); высота
   округляется до чётной на стороне Python (расчёт до запуска ffmpeg).
3. **Backoff watchdog — фиксированный `reconnect_backoff_s`** (из конфига hls.*), а не
   экспоненциальный из псевдокода §4.3: задача прямо указывает «backoff reconnect_backoff_s».
   Экспоненту легко добавить в задаче 15, если понадобится.
4. **t_video для FfmpegPipeSource** считается как `index / fps` (effective_fps или нативный
   из ffprobe), а не из таймкода: при effective_fps>0 таймлайн ffmpeg — это именно
   обработанная последовательность, что согласуется с логикой cooldown'ов (wall-clock в t_wall).
5. **CLI-команда `gui`** не выделена отдельной сабкомандой — по ТЗ задача 16 GUI-режим это
   флаг `count --gui`; в README пример дан как `count --gui`.
6. Валидация конфига строгая: неизвестные ключи (включая опечатки) отклоняются с перечнем
   разрешённых — осознанный выбор для «ОЧЕНЬ полезных» сообщений об ошибках.

## Что важно для задач 12–16 (имена API)

- `from visio_people_counter.config import Config, ConfigError`
  - `cfg = Config.load("config.yaml")`; блоки: `cfg.video(.type,.path,.hls{reconnect_attempts,reconnect_backoff_s,bad_read_threshold},.loop_file)`,
    `cfg.processing(effective_fps,max_width)`, `cfg.motion(method,history,var_threshold,dist2_threshold,detect_shadows,shadow_threshold,morph_open,morph_close)`,
    `cfg.objects(min_area_fraction,max_area_fraction,min_bbox_side_px,aspect_ratio_range,min_fill,min_lifetime_frames)`,
    `cfg.size_profile(enabled,control_points,k_min,k_max)`,
    `cfg.tracker(type,lost_track_buffer,minimum_consecutive_frames,minimum_iou_threshold)`,
    `cfg.counters[]` — `LineCounterConfig(id,a,b,count_mode,cooldown_s,buffer_width_scale,min_global_gap_s)` /
    `ZoneCounterConfig(id,polygon,count_mode,cooldown_s,min_global_gap_s)` (кортежи/списки нормализованных пар),
    `cfg.output(events_jsonl,summary_interval_s,final_summary)`, `cfg.debug(...)`.
- `from visio_people_counter.video_source import Frame, VideoSource, FileSource, FfmpegPipeSource, ffprobe_info, VideoSourceError`
  - `Frame(image: np BGR, index: int, t_wall: float, t_video: float|None)`;
  - `read() -> Frame | None`: **None = EOF ИЛИ «недоступен сейчас» (разрыв/обработанный watchdog-цикл)** —
    pipeline (задача 15) обязан обрабатывать None как «пропустить и читать дальше», а не как конец.
    Для FileSource без loop None — окончательный EOF. Для FfmpegPipeSource при reconnect_attempts=-1
    или исчерпанных попытках read() тоже будет возвращать None постоянно — pipeline должен
    различать по необходимости (можно проверить `src.probe_info`/состояние через close).
  - `FfmpegPipeSource(path_or_url, effective_fps=0, max_width=0, reconnect_attempts=0,
    reconnect_backoff_s=5.0, bad_read_threshold=10, on_reconnect=None)` — хук `on_reconnect`
    вызывается ПОСЛЕ успешного respawn: сюда повесить `MotionDetector.reset()` / `TrackerAdapter.reset()`.
  - `width/height` = размеры ОБРАБАТЫВАЕМОГО кадра (после max_width) — все нормализованные
    координаты конфига масштабировать на них, не на исходные.
  - `ffprobe_info(path_or_url) -> {width,height,fps,codec_name,duration,format_name}` — переиспользовать в calibrate/probe.
- CLI: сабкоманды регистрируются в `build_parser()` (`__main__.py`); заглушки `count`/`calibrate`
  принимают уже `--config` и `--video` (переопределение video.path) — задачи 15/16 дописывают логику,
  аргументы менять не надо.

## Риски / заметки

- HLS live (m3u8) в тестовой среде нет — путь FfmpegPipeSource проверен на файловом входе через ffmpeg;
  поведение watchdog при реальных обрывах сети проверить можно будет только в задаче 15 (бенчмарк/смоук).
- `FileSource.fps` берётся из метаданных cv2; для экзотических контейнеров может быть 0 → t_video=None.
