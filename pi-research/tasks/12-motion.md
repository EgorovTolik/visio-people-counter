# Задача 12: motion_detector.py + size_profile.py

## Читать ПЕРЕД началом (в этом порядке)
1. /home/anatoliy/PiProjects/visio-people-counter/pi-research/05-development-plan.md — план и правила.
2. /home/anatoliy/PiProjects/visio-people-counter/config.example.yaml — блоки motion, objects, size_profile (значения/диапазоны).
3. /home/anatoliy/PiProjects/visio-people-counter/pi-research/11-task-report.md — что уже есть: API config.py и video_source.py (ИСПОЛЬЗУЙ существующие dataclass'ы конфига, не изобретай свои).
4. Код: visio_people_counter/config.py (имена полей), пиши код в том же стиле.

## Среда
ТОЛЬКО /home/anatoliy/PiProjects/visio-people-counter/.venv/bin/python. Ничего не устанавливать, ничего не коммитить.

## Делать (в /home/anatoliy/PiProjects/visio-people-counter/)
1. `size_profile.py`:
   - `SizeProfile` строится из Config (control_points: список [x_frac, person_h_frac], k_min, k_max) и текущих w/h кадра;
   - API: `person_height_px(x_px)` — высота человека в px по x-координате (линейная интерполяция между контрольными точками, за краями — крайнее значение);
   - `min_area_at(x_px) = k_min * h_px^2`, `max_area_at(x_px) = k_max * h_px^2`;
   - если size_profile отключён/пустой → методы возвращают глобальные пороги из cfg.objects (min/max_area_fraction * w*h);
   - `buffer_width_px(x_px, scale)` — ширина буфера вокруг линии = scale * person_height_px.
2. `motion_detector.py`:
   - `@dataclass Blob: x, y, w, h, area, cx, cy` (px, исходные координаты кадра);
   - `MotionDetector(cfg)`: создаёт MOG2 (history, var_threshold, detectShadows) или KNN (dist2Threshold) по cfg.motion.method;
   - `detect(frame_image, size_profile: SizeProfile|None) -> list[Blob]`:
     apply → при detect_shadows: пиксели < shadow_threshold (тени=127 и прочие слабые) обнулить
     (математически: mask = (fgmask >= shadow_threshold).astype(uint8)*255, при shadow_threshold<=127 — просто ==255);
     MORPH_OPEN(morph_open) → MORPH_CLOSE(morph_close) → connectedComponentsWithStats;
     фильтры по cfg.objects: min/max площадь (через size_profile.min_area_at(cx) / max_area_at(cx)),
     min_bbox_side_px, aspect_ratio_range, min_fill; min_lifetime_frames — СЮДА не входит
     (это уровень трекера/adapter'а, задача 13);
   - `reset()` — сброс субтрактора (вызывать при reconnect из pipeline);
   - порядок фильтров и все параметры — строго по config.example.yaml.
3. Тесты в tests/test_motion.py (unittest):
   - синтезировать сцену: статичный «фон» (шум 5% пикселей) + движущийся прямоугольник-«человек»;
     после ~history warmup кадров детектор находит blob рядом с квадратом (IoU bbox > 0.3),
     стоящий объект через N кадров не даёт blob'а;
   - тени: отрисовать тёмный «отблеск» рядом с движением — при detect_shadows=true он НЕ попадает в blob
     (проверить на синтетике, если не воспроизводится стабильно — задокументировать);
   - фильтры: слишком маленький/слишком широкий blob'ы отфильтровываются;
   - SizeProfile: проверка интерполяции на контрольных точках и краях, включённый vs выключенный.

## Правила
- Все bash с timeout; без интерактивных команд; ничего не коммитить; не трогать references/, .venv/.
- Не менять API задач 11 (config.py, video_source.py) без крайней необходимости — если изменил, отметить в отчёте.

## Отчёт
/home/anatoliy/PiProjects/visio-people-counter/pi-research/12-task-report.md: файлы, вывод тестов,
выбранные дефолты/нюансы (например, как именно режутся тени), API для задачи 13+.

## Формат ответа (СТРОГО)
ТОЛЬКО короткое резюме (10–20 строк + результат тестов) + путь к отчёту.
