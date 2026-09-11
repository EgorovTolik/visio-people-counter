# Задача 16: GUI-режим (--gui, --speed) + calibrate

## Читать ПЕРЕД началом (в этом порядке)
1. /home/anatoliy/PiProjects/visio-people-counter/pi-research/05-development-plan.md — план и правила.
2. /home/anatoliy/PiProjects/visio-people-counter/config.example.yaml — блоки debug, counters.
3. Отчёты pi-research/11…15-task-report.md; код visio_people_counter/*.py (особенно pipeline.py, line_counter.draw).
4. pi-research/04-implementation-notes.md — §3.3 (UX калибровки: setMouseCallback, линия мышью, мерка роста, live-подстройка порога).

## Среда
ТОЛЬКО /home/anatoliy/PiProjects/visio-people-counter/.venv/bin/python. Ничего не устанавливать, ничего не коммитить.
ВНИМАНИЕ: окружение может быть headless (без X11) — GUI-код должен дегралировать с ПОНЯТНОЙ ошибкой
("нет дисплея для GUI-режима"), а тесты не должны падать без окна (проверка через cv2.getBuildInformation()
или try/except при создании окна; логику вынести в чистые функции, тестируемые без imshow).

## Делать (в /home/anatoliy/PiProjects/visio-people-counter/)
1. `gui.py` (новый модуль):
   - `GuiOverlay`: рисует на кадре по cfg.debug: bbox подтверждённых треков + track_id, необработанные blob'ы
     (show_all_blobs), микс с маской движения (show_mask), линии/зоны счётчиков + текущие в/out-счётчики
     крупно в углу; подписи читаемые (fontScale ≥ 0.6);
   - `GuiPlayer`: окно cv2.imshow, клавиши: пробел — пауза, q/ESC — выход, +/− — скорость ×/÷1.5,
     r — перемотка? НЕ нужна (стрим) — только для файла опционально; скорость задаётся --speed (0.25..8);
   - headless-фолбэк: `GuiPlayer.available() -> bool` (нет DISPLAY/WAYLAND и нет окна → False),
     при недоступности CLI печатает ясную ошибку и rc=1.
2. `calibrate.py`:
   - `python -m visio_people_counter calibrate --config config.yaml --video PATH_OR_URL [--counter-id main_line]`;
   - открывает кадр(ы) видео (для HLS — первые кадры), режимы по клавишам:
     'l' + 2 клика — нарисовать линию A→B для выбранного счётчика (порядок кликов = направление);
     'z' + N кликов, Enter — замкнуть полигон зоны;
     's' + клик — поставить size-точку (x фиксируется кликом, высоту человека задаёт последующая цифра 1-9
       как % высоты кадра: 1=5%, 2=10%… 9=45% — простой и предсказуемый ввод; Enter завершает);
     'm' — toggle показа текущей маски движения (live-подстройка, чтобы видеть, что детектится);
     'c' + параметр из строки? НЕ вводить текст: вместо этого 'f1'..'f4' не использовать — переключение
       min_area_fraction пресетами клавишами 1..9 НЕ нужно; достаточно 'm' + 'l'/'z'/'s';
     'a' — применить и записать в config.yaml (Config.save, с сохранением остальных настроек), 'q' — выход без записи.
   - при 'a': обновить координаты линии/зоны size-точек в конфиге; если line-счётчика нет в конфиге — добавить по id из --counter-id;
     распечатать что изменилось (diff коротко: старые → новые координаты).
3. CLI: `count` получает флаги `--gui [--speed FLOAT]`; headless остаётся ПО УМОЛЧАНИЮ.
4. Тесты tests/test_gui_calibrate.py (unittest, БЕЗ реального окна):
   - GuiOverlay.draw на синтетическом кадре: не падает, все опции cfg.debug комбинируются;
     счётчики/линии действительно нарисованы (пиксели линии отличаются от фона — проверка по маске);
   - calibrate: чистые функции «клик → обновление конфига» тестируются без imshow:
     две координаты кликов → LineCounter-координаты в конфиге нормализованы и валидны;
     полигон из 4 кликов → ZoneCounter; size-точка записывается; Config.save→load сохраняет всё (roundtrip).
   - headless: GuiPlayer.available() возвращает bool без исключений.

## Правила
- Все bash с timeout; без интерактивных команд; ничего не коммитить.
- Не менять API задач 11–15 без крайней необходимости — если изменил, отметить в отчёте.

## Отчёт
/home/anatoliy/PiProjects/visio-people-counter/pi-research/16-task-report.md: файлы, вывод тестов,
таблица клавиш GUI/calibrate, что проверить вручную владельцу (пошагово), ограничения headless-окружения.

## Формат ответа (СТРОГО)
ТОЛЬКО короткое резюме (10–20 строк + результат тестов) + путь к отчёту.
