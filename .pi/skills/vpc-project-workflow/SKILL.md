---
name: vpc-project-workflow
description: Правила работы в проекте visio-people-counter (подсчёт человеческого трафика на видео). Использовать ВСЕГДА при работе в этом проекте: субагенты строго последовательно, задачи/результаты в pi-research/, методика веб-/GitHub-исследований, окружение .venv, правила git и коммитов только после подтверждения владельца.
---

# Workflow проекта visio-people-counter

Проект: консольный инструмент подсчёта человеческого трафика на видео (HLS URL или файл).
Python + OpenCV; детекция движения + трекер (вспомогательно, против двойного счёта);
уличная камера с широким углом → сильный разброс размеров объектов.

## 1. Окружение (ВАЖНО)
- ВСЕ python-команды — через venv проекта: `.venv/bin/python` (создан с `--system-site-packages`).
  Внутри: cv2 5.0.0 (wheel от supervision), numpy 2.5.3, supervision 0.30.2, trackers 2.6.0, PyYAML.
- ffmpeg/ffprobe — системные. Системный python «externally managed» (PEP 668) — НИКОГДА не ставить пакеты в систему.
- Новые зависимости: сначала обсудить с владельцем; установка — только в `.venv` + запись в `requirements.txt`.
- Тесты: `.venv/bin/python -m unittest discover -s tests -t .` (stdlib unittest, pytest НЕ установлен).
- Видео для тестов владельца кладутся в `videos/` (в .gitignore).

## 2. Субагенты — ОБЯЗАТЕЛЬНЫЕ правила владельца
1. **Все многошаговые действия выполнять через субагентов**, строго **по одному**, НИКАКОЙ параллели
   (LLM-сервер однопоточный). Запускать блокирующим вызовом (`async: false`), ждать завершения, потом следующий.
2. Задачи субагентам — отдельным файлом-спецификацией в `pi-research/tasks/NN-<topic>.md`.
   Спецификация обязана быть самодостаточной: что читать ПЕРЕД началом (отчёты/код), точный scope,
   структура отчёта, точный путь результата, правила bash (timeout на все команды, find/grep -maxdepth 4,
   без интерактивных команд).
3. Субагент сохраняет **полные результаты в отдельный файл** `pi-research/NN-<topic>.md` / `<NN>-task-report.md`,
   а в ответ возвращает **ТОЛЬКО** короткое резюме (10–20 строк) + путь к файлу.
4. После каждого субагента ПРОВЕРИТЬ: файл существует и имеет вес; если результат плохой — перезапустить
   с уточнённой задачей-исправлением.
5. Финальный синтез/интеграцию (планы, стратегия, ответы владельцу) делает главный агент сам.
6. Типы субагентов: `researcher` — веб-исследование; `worker` — код/bash (git clone, тесты);
   `reviewer`/`planner`/`scout` — по назначению.

## 3. Методика исследований
- Веб: DuckDuckGo MCP может быть недоступен → curl к `https://html.duckduckgo.com/html/?q=...`
  (UA браузера, парсинг grep `result__a`); при блокировке IP компенсировать GitHub-API-поисками.
- GitHub REST API через curl (без авторизации): **rate limits** — ~10 search/мин и 60 core/час:
  sleep 7–8 сек между search-запросами; README брать с `raw.githubusercontent.com` (не расходует core-лимит);
  не жечь лимит больше необходимого.
- Клонировать референсные репозитории ТОЛЬКО в `references/repos/` (уже в .gitignore), shallow:
  `git clone --depth 1`. Репо без лицензии — read-only, код НЕ копировать (только идеи).
- Отчёты исследований нумеруются последовательно: `pi-research/01…NN-*.md`; дублировать между отчётами нельзя.

## 4. Git — правила владельца
- **Коммитить ТОЛЬКО после явного подтверждения владельца.** Push — тоже только когда разрешил.
- Исследование и код — отдельными коммитами (исследование → `pi-research/`, разработка → остальное).
- Дистант: origin = git@github.com:EgorovTolik/visio-people-counter.git, ветка master (push в master разрешён владельцем).
- В репозиторий не попадают: `references/repos/`, `.venv/`, `videos/` — уже в .gitignore.
- GitHub host key добавлен в known_hosts (`accept-new`); авторизация по ssh-ключу EgorovTolik.

## 5. Карта проекта (состояние на v0.1.0)
```
visio_people_counter/  пакет:
  __main__.py      CLI: probe / count [--gui --speed] [--bench] / calibrate
  config.py        YAML dataclass'ы + валидация (эталон схемы — config.example.yaml)
  video_source.py  FileSource, FfmpegPipeSource (ffprobe, watchdog переподключения)
  motion_detector.py MOG2/KNN → тени → морфология → connectedComponents → фильтры
  size_profile.py  адаптивные пороги под разный размер объектов (контрольные точки)
  tracker_adapter.py SORT/ByteTrack/BoTSORT (roboflow/trackers), confidence=None
  line_counter.py  LineCounter (наклонная A→B, prev→cur, in/out, антидубль) + ZoneCounter (полигон)
  event_log.py     JSONL событий, агрегаты in/out/total, сводки
  pipeline.py      сборка из конфига, главный цикл, SIGINT-graceful, --bench
  gui.py / calibrate.py — GUI-overlay+плеер; калибровка линии/зоны/size-точек мышью → config.yaml
tests/             unittest (84 теста на синтетике, E2E пересечения линии)
pi-research/       отчёты 01–05 + tasks/ (спецификации) + 11…17-task-report.md
```
Новые фазы работы: продолжать нумерацию задач в `pi-research/tasks/` с 18+, результаты — рядом.

## 6. Быстрые команды
```bash
cd /home/anatoliy/PiProjects/visio-people-counter
.venv/bin/python -m unittest discover -s tests -t .                 # все тесты
.venv/bin/python -m visio_people_counter probe --video videos/x.mp4
.venv/bin/python -m visio_people_counter count --config config.yaml            # headless
.venv/bin/python -m visio_people_counter count --config config.yaml --gui --speed 1
.venv/bin/python -m visio_people_counter calibrate --config config.yaml --video videos/x.mp4
```
