# Задача 14: line_counter.py (линия + зона) + event_log.py

## Читать ПЕРЕД началом (в этом порядке)
1. /home/anatoliy/PiProjects/visio-people-counter/pi-research/05-development-plan.md — план и правила.
2. /home/anatoliy/PiProjects/visio-people-counter/config.example.yaml — блок counters (line/zone, count_mode, антидубль), output.
3. /home/anatoliy/PiProjects/visio-people-counter/pi-research/04-implementation-notes.md — §2 (геометрия линии, псевдокод пересечения prev→cur, антидубль, поля события) и §2.4 (поля JSONL).
4. Отчёты 11–13 в pi-research/ + код visio_people_counter/ (Blob, TrackedObject, Config).

## Среда
ТОЛЬКО /home/anatoliy/PiProjects/visio-people-counter/.venv/bin/python. Ничего не устанавливать, ничего не коммитить.

## Делать (в /home/anatoliy/PiProjects/visio-people-counter/)
1. `line_counter.py`:
   - общий интерфейс: `BaseCounter(ABC)` с `update(objects: list[TrackedObject], t_wall) -> list[CrossingEvent]`,
     `counters -> dict` (счётчики), `draw(frame) -> None` (overlay для GUI — линия/зона + счётчики;
     реализация может быть простой, доработает задача 16);
   - `LineCounter`: наклонная линия A→B (px; из нормализованных координат конфига масштабирует pipeline/сам
     при init с w/h — решите и задокументируйте), направление: пересечение по вектору A→B = "in", обратное = "out";
     пересечение проверяется по отрезку prev→cur (калиманово-сглаженная позиция трека cx,cy) + точка пересечения
     должна лежать на ОТРЕЗКЕ линии, а не на его продолжении; считаются только подтверждённые треки (track_id != -1);
     антидубль по §2.3 отчёта 04: per-track cooldown_s (wall-clock) + буфер вокруг линии (buffer_width_scale ×
     person_height из SizeProfile, если передан; иначе фиксированный масштаб от длины линии — задокументировать),
     min_global_gap_s между любыми событиями счётчика;
   - `ZoneCounter`: полигон (нормализованные точки → px), point-in-polygon (cv2.pointPolygonTest) по позиции трека;
     "in" = пересечение границы снаружи→внутрь, "out" = внутрь→снаружу; тот же антидубль (cooldown per track +
     global gap); count_mode total/both как у линии.
   - `CrossingEvent` dataclass: counter_id, direction ("in"/"out"), track_id, t_wall, t_video (float|None),
     x_px, y_px (точка пересечения/пересечения границы), frame_index; метод to_dict() для JSONL.
2. `event_log.py`:
   - `EventLog(cfg)`: запись событий в output.events_jsonl (JSON, по строке на событие; "" = не писать),
     агрегаты per-counter in/out/total + глобальные; `summary()` → строка сводки для stdout
     (используется pipeline раз в summary_interval_s и при финале); thread-безопасность не нужна.
3. Тесты tests/test_counters.py (unittest), геометрия на чистых списках TrackedObject (без видео):
   - объект пересёк линию слева-направо по A→B → 1 событие "in"; обратно → "out"; движение ВДОЛЬ линии — 0 событий;
     «пересечение» за пределами отрезка (у продолжения) — 0;
   - антидубль: объект туда-обратно на месте у линии быстрее cooldown_s → максимум 1 событие; медленнее (с выходом из буфера) → засчитано;
   - min_global_gap_s: два события < 0.3 c друг от друга → второе отброшено;
   - неподтверждённый трек (-1) не считается;
   - зона: вход/выход полигона, движение внутри без пересечения границы — 0; count_mode=total суммирует оба направления;
   - EventLog: JSONL-файл корректен (json.loads по строкам), агрегаты сходятся.

## Правила
- Все bash с timeout; без интерактивных команд; ничего не коммитить.
- Не менять API задач 11–13 без крайней необходимости — если изменил, отметить в отчёте.

## Отчёт
/home/anatoliy/PiProjects/visio-people-counter/pi-research/14-task-report.md: файлы, вывод тестов,
решения по геометрии/антидублю (что именно реализовано из §2), API для задачи 15 (pipeline).

## Формат ответа (СТРОГО)
ТОЛЬКО короткое резюме (10–20 строк + результат тестов) + путь к отчёту.
