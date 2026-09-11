# Задача 13: tracker_adapter.py

## Читать ПЕРЕД началом (в этом порядке)
1. /home/anatoliy/PiProjects/visio-people-counter/pi-research/05-development-plan.md — план и правила.
2. /home/anatoliy/PiProjects/visio-people-counter/config.example.yaml — блок tracker.
3. /home/anatoliy/PiProjects/visio-people-counter/pi-research/11-task-report.md и 12-task-report.md — существующие API (Config, Blob).
4. Код трекера в референсе: references/repos/trackers/src/trackers/__init__.py и
   references/repos/trackers/src/trackers/core/sort/tracker.py, bytetrack/tracker.py, botsort/tracker.py
   (параметры конструкторов; у нас confidence=None → ByteTrack ≡ SORT-режим).

## Среда
ТОЛЬКО /home/anatoliy/PiProjects/visio-people-counter/.venv/bin/python (там supervision 0.30.2, trackers 2.6.0).
Ничего не устанавливать, ничего не коммитить, references/ — только чтение.

## Делать (в /home/anatoliy/PiProjects/visio-people-counter/)
1. `tracker_adapter.py`:
   - `@dataclass TrackedObject: track_id (int, -1 = неподтверждённый), x, y, w, h, cx, cy, age_frames`;
   - `TrackerAdapter(cfg)`: строит трекер по cfg.tracker.type: sort→SORTTracker, bytetrack→ByteTrackTracker,
     botsort→BoTSORTTracker; параметры: lost_track_buffer, minimum_consecutive_frames,
     minimum_iou_threshold, frame_rate=эффективный fps (параметр конструктора `frame_rate`,
     pipeline передаст фактический fps — предусмотреть инициализацию с fps по умолчанию 15 и метод
     `set_frame_rate(fps)`/создание с ним);
   - `update(blobs: list[Blob]) -> list[TrackedObject]`:
     bbox'ы → sv.Detections(xyxy=..., confidence=None) (пустой список → пустые detections,
     проверить корректное поведение трекера на пустых кадрах!);
     detections = tracker.update(dets); вернуть TrackedObject с tracker_id и координатами;
   - `reset()` — пересоздать/сбросить трекер (вызывается при reconnect потока);
   - docstring: почему confidence=None (blob'ы без скорей) → трекеры деградируют до IoU-ассоциации.
2. Тесты tests/test_tracker.py (unittest):
   - синтетика: 30–60 кадров, 1–2 движущихся «человека» по прямой + на несколько кадров пропадает
     один (simulating перекрытие) → track_id сохраняется после пропадания ≤ lost_track_buffer,
     новые объекты получают новые id; неподтверждённые одиночные кадры дают track_id=-1;
   - пустые списки blob'ов (50 кадров) не роняют адаптер и «старые» треки закрываются;
   - reset() после обрыва: id начинаются заново, состояние чистое.
3. Обновить README.md: секция "архитектура" — перечислить модули по мере готовности (ссылки на отчёты 11–13).

## Правила
- Все bash с timeout; без интерактивных команд; ничего не коммитить.
- Не менять API задач 11/12 без крайней необходимости — если изменил, отметить в отчёте.

## Отчёт
/home/anatoliy/PiProjects/visio-people-counter/pi-research/13-task-report.md: файлы, вывод тестов,
проверенные параметры трекеров (строки кода референса), API для задачи 14+.

## Формат ответа (СТРОГО)
ТОЛЬКО короткое резюме (10–20 строк + результат тестов) + путь к отчёту.
