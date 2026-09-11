# Задача 11: Скетелет проекта, config.py, video_source.py, CLI probe

## Читать ПЕРЕД началом (в этом порядке)
1. /home/anatoliy/PiProjects/visio-people-counter/pi-research/05-development-plan.md — план и правила.
2. /home/anatoliy/PiProjects/visio-people-counter/config.example.yaml — ЭТАЛОН схемы конфига (все ключи, типы, дефолты).
3. /home/anatoliy/PiProjects/visio-people-counter/pi-research/04-implementation-notes.md — раздел 4 (ввод HLS/файл: команда ffmpeg, ffprobe, watchdog) и §1.1–1.2 (ответственность модулей).

## Среда
Работать ТОЛЬКО через .venv проекта: /home/anatoliy/PiProjects/visio-people-counter/.venv/bin/python
(там cv2 5.0.0, numpy 2.5.3, PyYAML; ffmpeg/ffprobe — системные). Ничего не устанавливать.

## Делать (проект: /home/anatoliy/PiProjects/visio-people-counter/)
1. Пакет `visio_people_counter/`: `__init__.py` (версия), `__main__.py` — CLI на argparse с
   сабкомандами: `probe`, `count`, `calibrate`. `count` и `calibrate` пока заглушки
   (печатают "не реализовано" и exit 2) — их сделают задачи 15/16.
2. `config.py`: dataclass'ы 1:1 со всеми блоками config.example.yaml (video/hls, processing,
   motion, objects, size_profile, tracker, counters (line+zone), output, debug).
   - `Config.load(path) -> Config`: чтение YAML, валидация типов/диапазонов (координаты 0..1,
     ядра морфологии — нечётные >0, type-поля из разрешённых множеств), ОЧЕНЬ полезные
     сообщения об ошибках (блок.ключ: что ожидалось vs получено);
   - `Config.save(cfg, path)`: сериализация обратно в YAML с комментариями не обязательна;
   - дефолты — точно как в config.example.yaml; конфиг без опциональных блоков
     (size_profile.enabled=false и т.п.) должен проходить;
   - нормализованные координаты: при обращении из кода модули сами масштабируют на w/h кадра
     (в config.py достаточно валидации).
3. `video_source.py`:
   - `@dataclass Frame: image (np BGR), index, t_wall, t_video (float|None)`;
   - `VideoSource(ABC)`: `open()`, `read() -> Frame | None` (None = EOF/недоступен сейчас),
     `close()`, свойства `width/height/fps`;
   - `FileSource`: cv2.VideoCapture, опция loop_file (повтор с reset'ом субтрактора НЕ нужен —
     просто перемотка; событие повторного начала можно не сигнализировать);
   - `FfmpegPipeSource`: ffprobe → w/h/fps/кодек; команда ffmpeg: `-loglevel error`, вход = путь/URL,
     при effective_fps>0 → `-r N`, при max_width>0 → scale (высота чётная), `-f rawvideo
     -pix_fmt bgr24 pipe:1`; чтение через proc.stdout.read(w*h*3) + frombuffer;
   - watchdog по §4.3 отчёта 04: bad_read_threshold подряд неудачных read/пустых → kill ffmpeg,
     backoff reconnect_backoff_s, повтор (reconnect_attempts: 0 = бесконечно, -1 = не переподключаться);
     хук `on_reconnect` (callable) — потом pipeline повесит reset детектора/трекера;
   - EOF файла через pipe = graceful close.
4. CLI `probe --video <path|url>`: ffprobe-сводка (разрешение, fps, длительность, кодек,
   тип: file/hls) + для файла — проверка cv2.VideoCapture открытия.
5. `tests/` на stdlib unittest (pytest НЕ установлен):
   - test_config.py: валидный example-конфиг загружается; битые значения (координата 1.5,
     нечётное ядро [4,4], неизвестный method) → ConfigError с текстом;
   - test_video_source.py: создать СИНТЕТИЧЕСКИЙ mp4 (~30 кадров, cv2.VideoWriter 'mp4v',
     движущийся квадрат) во временную папку → FileSource читает все кадры, loop_file работает;
     FfmpegPipeSource с тем же файлом как входом (ffmpeg -i file...) — та же последовательность
     кадров, effective_fps=10 и max_width=640 дают ожидаемые w/h/скорость.
6. `README.md`: заготовка — что это, установка (.venv), быстрые примеры CLI (probe/count/gui/
   calibrate — с пометкой "в разработке"), ссылка на config.example.yaml.

## Правила
- Все bash — с timeout; find/grep -maxdepth 4; без интерактивных команд.
- НИЧЕГО не коммитить в git. Не трогать references/, pi-research/ (кроме своего отчёта), .venv/.
- Код читаемый, аннотации типов, docstring'ы на классах/публичных методах; комментарии — русский/английский как удобнее, consistently.

## Отчёт
Написать /home/anatoliy/PiProjects/visio-people-counter/pi-research/11-task-report.md:
что сделано (файлы), результаты тестов (вывод unittest), отклонения от спецификации и почему, что важно для задач 12–16 (имена API).

## Формат ответа (СТРОГО)
ТОЛЬКО короткое резюме (10–20 строк: что сделано + результат тестов) + путь к файлу отчёта.
