# Задача 02: Поиск и отбор open-source решений подсчёта людей на GitHub

## Контекст
Читай ПЕРЕД началом: /home/anatoliy/PiProjects/visio-people-counter/pi-research/01-techniques.md
(обзор методов: MOG2 → морфология → компоненты → трекер → counting line; HLS через ffmpeg pipe).

Задача проекта — как в том файле. Инструменты: Python + OpenCV, камера фиксированная уличная,
широкий угол (разный размер объектов), ввод: HLS URL или файл. Трекинг — вспомогательный
механизм против двойного счёта.

## Твоя задача
1. **GitHub-поиск живых решений** через `curl` к GitHub REST API (gh CLI нет, авторизации нет):
   - endpoint: https://api.github.com/search/repositories?q=...&sort=stars&order=desc&per_page=10
   - запросы (каждый — отдельный curl, можно менять формулировки/операторы in:name,description,readme):
     a) `people counter opencv`
     b) `pedestrian counting video`
     c) `background subtraction people counting line`
     d) `yolo people counter tracking`
     e) `crowd counting street camera` (и что ещё сочтёшь уместным)
   - ВАЖНО, rate limits: без авторизации ~10 search-запросов/мин и 60 core-запросов/час.
     Не жги лимит: между запросами sleep 7–8 сек; core-запросы (GET по репо) — минимум,
     не более ~15–20 за весь прогон.
   - Собирай 10–15 разных кандидатов из всех запросов. Для каждого: full_name, stars, язык,
     лицензия, дата последнего коммита (updated_at/pushed_at), описание 1 строкой, URL.
   - Проверь дубликаты/форки — в отчёт возьми только основной репо.

2. **Безопасные по лимиту детали** для ~6 самых интересных кандидатов:
   - README можно брать с raw.githubusercontent.com (это НЕ расходует API core-лимит):
     `curl -sL https://raw.githubusercontent.com/<owner>/<repo>/HEAD/README.md | head -200`
   - Структуру дерева репо (без API, через git trees? — нет, это API; вместо этого:
     по содержимому README понять стек и наличие counting line / трекера).

3. **Live-проверка источников из отчёта 01** (отчёт 01 писался без live-доступа к вебу):
   проверь `curl -sI` (HEAD, timeout 15) на ключевые URL и отметь доступность:
   - https://docs.opencv.org/4.x/d1/d53/classcv_1_1BackgroundSubtractorMOG2.html
   - https://docs.opencv.org/4.x/de/d0e/classcv_1_1BackgroundSubtractorKNN.html
   - https://docs.opencv.org/4.x/d9/dfc/group__tracking.html
   - https://supervision.roboflow.com/
   - https://docs.ultralytics.com/
   - https://github.com/ifzhang/ByteTrack
   - https://github.com/abergal/SORT
   Плюс 2–3 свежих web-поиска через DuckDuckGo HTML (curl, UA браузера, примеры запросов:
   "people counting line opencv tutorial github", "yolov8 bytetrack people counter stream")
   — чтобы подхватить решения, не попавшие в top по звёздам. Парсинг: grep `result__a`.

4. **Рекомендация для задачи 03** (клонирование и разбор кода): выбери 2–3 репо по критериям:
   - максимально близко к нашей задаче (counting line / люди / уличная камера);
   - Python + OpenCV без тяжёлых GPU-зависимостей (как минимум ОДНО такое обязательно);
   - активные или хотя бы рабочие; лицензия совместимая с открытой разработкой (MIT/Apache/BSD);
   - плюс одно "тяжёлое" референс-решение (YOLO+ByteTrack), если есть, — для сравнения архитектуры.

## Требования к выводу
Файл: /home/anatoliy/PiProjects/visio-people-counter/pi-research/02-github-solutions.md
Язык: русский (названия репо/API — как есть). Разделы:
1. Методика поиска (запросы, ограничения)
2. Таблица кандидатов: Репо | Stars | Язык | Лицензия | Последний коммит | Суть (1 строка) | URL
3. Детальные заметки по 5–6 лучшим кандидатам (что нашли в README: пайплайн, трекер,
   counting line, зависимости)
4. Live-проверка источников из отчёта 01: таблица URL → доступен/нет
5. Найденное через web-поиск сверх GitHub-API
6. ИТОГОВАЯ РЕКОМЕНДАЦИЯ: точный список репо (full_name + URL) для клонирования в задаче 03
   с обоснованием каждого выбора

## Правила
- Все curl — с `timeout 20`/`--max-time`; никаких интерактивных команд; sleep между API-запросами.
- Ничего не клонировать в этой задаче (это задача 03).
- Писать только в указанный файл результата.

## Формат ответа (СТРОГО)
ТОЛЬКО краткое резюме (10–20 строк, включая финальный список репо для задачи 03) + путь к файлу.
