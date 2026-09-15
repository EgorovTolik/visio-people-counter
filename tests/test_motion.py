"""Тесты motion_detector.py + size_profile.py на синтетических сценах.

Сцена «стандартная»: статичный фон (серый 180) с шумом ~5% тёмных пикселей
+ движущийся тёмный прямоугольник-«человек» (80x220). Все параметры подобраны
так, чтобы детектор вёл себя стабильно и детерминированно:

* «человек» движется с шагом 10 px/кадр по циклу периода 48 кадров — за время
  warmup (~30 кадров) пиксель фона не успевает «впечатать» объект в модель;
* история субтрактора 100 — достаточно для warmup без долгого ожидания.

Замечание о MOG2: объект, который покрывает одни и те же пиксели много кадров
подряд (или стоит на месте), со временем усваивается моделью фона и перестаёт
детектироваться — это и проверяет ``test_standing_object_disappears``.
"""

import sys
import unittest
from pathlib import Path

# корень проекта — родитель каталога tests/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from visio_people_counter.config import Config, SizeProfileConfig  # noqa: E402
from visio_people_counter.motion_detector import Blob, MotionDetector, filter_blobs  # noqa: E402
from visio_people_counter.size_profile import (  # noqa: E402
    SizeProfile,
    fit_height_surface,
)

W, H = 640, 360
PERSON_W, PERSON_H, PERSON_Y = 81, 221, 90   # cv2.rectangle — координаты включительно


def make_noise_scene(seed: int = 7):
    """Фабрика кадров: статичный фон 180 + ~5% тёмного «шума» (значения 20-40)."""
    rng = np.random.default_rng(seed)

    def frame() -> np.ndarray:
        f = np.full((H, W, 3), 180, np.uint8)
        m = rng.random((H, W)) < 0.05
        vals = (20 + rng.integers(0, 20, int(m.sum()))).astype(np.uint8)
        f[m] = np.repeat(vals[:, None], 3, axis=1)
        return f

    return frame


def person_pos(i: int) -> int:
    """x левого края «человека» в кадре i (период 48 кадров)."""
    return (20 + i * 10) % 480


def draw_person(f: np.ndarray, px: int) -> None:
    cv2.rectangle(f, (px, PERSON_Y), (px + PERSON_W - 1, PERSON_Y + PERSON_H - 1), (40, 40, 40), -1)


def person_iou(b: Blob, px: int) -> float:
    """IoU bbox'а blob'а и «истинного» прямоугольника человека при x=px."""
    iw = min(b.x + b.w, px + PERSON_W) - max(b.x, px)
    ih = min(b.y + b.h, PERSON_Y + PERSON_H) - max(b.y, PERSON_Y)
    inter = max(0, iw) * max(0, ih)
    if inter <= 0:
        return 0.0
    return inter / (b.w * b.h + PERSON_W * PERSON_H - inter)


# ---------------------------------------------------------------------------
# Сцены с движением
# ---------------------------------------------------------------------------

class TestMog2MovingObject(unittest.TestCase):
    def test_moving_person_detected_after_warmup(self):
        """После warmup MOG2 находит blob рядом с «человеком» (IoU > 0.3)."""
        frames = make_noise_scene(7)
        det = MotionDetector(Config.from_dict(
            {"motion": {"method": "mog2", "history": 100}}))
        hits = 0
        for i in range(55):
            f = frames()
            px = person_pos(i)
            draw_person(f, px)
            if any(person_iou(b, px) > 0.3 for b in det.detect(f)):
                hits += 1
        self.assertGreaterEqual(hits, 10, "человек не детектируется после warmup")

    def test_standing_object_disappears(self):
        """Стоящий на месте объект со временем перестаёт давать blob'ы."""
        frames = make_noise_scene(7)
        det = MotionDetector(Config.from_dict(
            {"motion": {"method": "mog2", "history": 100}}))
        px_static = person_pos(45)
        for i in range(45):  # warmup с движением
            f = frames()
            draw_person(f, person_pos(i))
            det.detect(f)
        standing_counts = []
        for _ in range(150):  # объект замирает
            f = frames()
            draw_person(f, px_static)
            standing_counts.append(len(det.detect(f)))
        self.assertEqual(sum(standing_counts[-80:]), 0,
                         "стоящий объект всё ещё детектируется")


class TestShadowExclusion(unittest.TestCase):
    def test_dark_shadow_not_merged_into_blob(self):
        """Тёмный «отблеск» (150) рядом с движением:

        * detect_shadows=true  → тень НЕ попадает в blob (ширина ~ человека);
        * detect_shadows=false → тень сливается с человеком (blob заметно шире).
        """
        frames = make_noise_scene(7)
        det_on = MotionDetector(Config.from_dict(
            {"motion": {"method": "mog2", "history": 100,
                        "detect_shadows": True, "shadow_threshold": 200}}))
        det_off = MotionDetector(Config.from_dict(
            {"motion": {"method": "mog2", "history": 100, "detect_shadows": False}}))
        w_on, w_off = 0, 0
        for i in range(45):
            f = frames()
            px = person_pos(i)
            draw_person(f, px)
            # «тень»: тёмный прямоугольник (150 < 180) справа от человека
            cv2.rectangle(f, (px + PERSON_W, PERSON_Y),
                          (px + PERSON_W + 49, PERSON_Y + PERSON_H - 1), (150,) * 3, -1)
            blobs_on = det_on.detect(f)    # warmup-кадры тоже идут в субтрактор
            blobs_off = det_off.detect(f)  # оба детектора получают один и тот же кадр
            if i >= 30:
                w_on = max(w_on, max((b.w for b in blobs_on), default=0))
                w_off = max(w_off, max((b.w for b in blobs_off), default=0))
        self.assertGreaterEqual(w_off, 95, "контроль: без тени-отсечения blob должен быть шире")
        self.assertLessEqual(w_on, PERSON_W + 24, "тень попала в blob при detect_shadows=true")


class TestKnnMethod(unittest.TestCase):
    def test_knn_detects_moving_person(self):
        frames = make_noise_scene(7)
        det = MotionDetector(Config.from_dict(
            {"motion": {"method": "knn", "history": 100, "dist2_threshold": 4.0}}))
        hits = 0
        for i in range(60):
            f = frames()
            px = person_pos(i)
            draw_person(f, px)
            if any(person_iou(b, px) > 0.3 for b in det.detect(f)):
                hits += 1
        self.assertGreaterEqual(hits, 10, "KNN не детектирует движущегося объект")


# ---------------------------------------------------------------------------
# Фильтры объектов (end-to-end на синтетике + unit-тест filter_blobs)
# ---------------------------------------------------------------------------

class TestObjectFiltersE2E(unittest.TestCase):
    def test_too_small_blob_filtered(self):
        """9x9 «точка» видна в маске, но отбрасывается фильтром площади.

        Контроль: с ослабленными порогами (min_area_fraction=0,
        min_bbox_side_px=4) тот же blob проходит — т.е. фильтрация, а не
        недетектирование.
        """
        cfg = Config.from_dict({})
        frames = make_noise_scene(7)
        det = MotionDetector(cfg)
        for i in range(50):
            f = frames()
            px = person_pos(i)
            cv2.rectangle(f, (px, 150), (px + 8, 158), (40,) * 3, -1)   # 9x9
            self.assertEqual(det.detect(f), [])

        cfg_relaxed = Config.from_dict(
            {"objects": {"min_area_fraction": 0.0, "min_bbox_side_px": 4}})
        frames = make_noise_scene(7)
        det_relaxed = MotionDetector(cfg_relaxed)
        seen = []
        for i in range(50):
            f = frames()
            px = person_pos(i)
            cv2.rectangle(f, (px, 150), (px + 8, 158), (40,) * 3, -1)
            if i >= 29:
                seen += det_relaxed.detect(f)
        self.assertGreaterEqual(len(seen), 3, "контроль: blob должен детектироваться")


def _make_cc(components):
    """Синтетический вывод connectedComponentsWithStats.

    :param components: список (x, y, w, h, pixels).
    """
    n = len(components) + 1
    stats = np.zeros((n, 5), dtype=np.int32)
    cents = np.zeros((n, 2), dtype=np.float32)
    counts = np.zeros(n, dtype=np.int64)
    for i, (x, y, w, h, pix) in enumerate(components, start=1):
        stats[i] = (x, y, w, h, w * h)
        cents[i] = (x + w / 2.0, y + h / 2.0)
        counts[i] = pix
    return stats, cents, counts


class TestFilterBlobsUnit(unittest.TestCase):
    def setUp(self):
        self.objects = Config.from_dict({}).objects   # дефолты config.example.yaml

    def test_wide_blob_filtered_by_aspect(self):
        stats, cents, counts = _make_cc([(100, 100, 200, 40, 7600)])  # w/h = 5.0 > 2.5
        self.assertEqual(filter_blobs(stats, cents, counts, self.objects,
                                      lambda x, y: 0.0, lambda x, y: float("inf")), [])
        wide_ok = Config.from_dict({"objects": {"aspect_ratio_range": [0.2, 6.0]}}).objects
        blobs = filter_blobs(stats, cents, counts, wide_ok,
                             lambda x, y: 0.0, lambda x, y: float("inf"))
        self.assertEqual(len(blobs), 1)
        self.assertEqual((blobs[0].x, blobs[0].w), (100, 200))

    def test_sparse_blob_filtered_by_min_fill(self):
        stats, cents, counts = _make_cc([(50, 50, 100, 100, 500)])    # fill = 0.05 < 0.25
        self.assertEqual(filter_blobs(stats, cents, counts, self.objects,
                                      lambda x, y: 0.0, lambda x, y: float("inf")), [])
        low_fill = Config.from_dict({"objects": {"min_fill": 0.0}}).objects
        self.assertEqual(len(filter_blobs(stats, cents, counts, low_fill,
                                          lambda x, y: 0.0, lambda x, y: float("inf"))), 1)

    def test_area_bounds(self):
        stats, cents, counts = _make_cc([(100, 100, 40, 40, 1600)])   # area = 1600 px²
        # слишком малый для min_area_at
        self.assertEqual(filter_blobs(stats, cents, counts, self.objects,
                                      lambda x, y: 2000.0, lambda x, y: float("inf")), [])
        # слишком большой для max_area_at
        self.assertEqual(filter_blobs(stats, cents, counts, self.objects,
                                      lambda x, y: 0.0, lambda x, y: 1500.0), [])
        # в пределах → проходит, поля Blob корректны
        blobs = filter_blobs(stats, cents, counts, self.objects,
                             lambda x, y: 1000.0, lambda x, y: 2000.0)
        self.assertEqual(len(blobs), 1)
        b = blobs[0]
        self.assertIsInstance(b, Blob)
        self.assertEqual((b.x, b.y, b.w, b.h, b.area), (100, 100, 40, 40, 1600))
        self.assertAlmostEqual(b.cx, 120.0)
        self.assertAlmostEqual(b.cy, 120.0)

    def test_min_bbox_side(self):
        stats, cents, counts = _make_cc([(0, 0, 5, 50, 200)])          # w = 5 < 8 px
        self.assertEqual(filter_blobs(stats, cents, counts, self.objects,
                                      lambda x, y: 0.0, lambda x, y: float("inf")), [])


# ---------------------------------------------------------------------------
# SizeProfile (2D: h(x, y), задача 05)
# ---------------------------------------------------------------------------

class TestSizeProfile(unittest.TestCase):
    def _profile(self, enabled=True,
                 points=((0.0, 0.5, 0.4), (0.5, 0.5, 0.3), (1.0, 0.5, 0.2))):
        sp_cfg = SizeProfileConfig(enabled=enabled, control_points=list(points),
                                   k_min=0.2, k_max=4.0)
        return SizeProfile(sp_cfg, Config.from_dict({}).objects, w=W, h=H)

    def test_interpolation_at_control_points_and_edges(self):
        sp = self._profile()
        self.assertTrue(sp.adaptive)
        yc = H // 2
        self.assertAlmostEqual(sp.person_height_px(0, yc), 0.4 * H)      # первая точка
        self.assertAlmostEqual(sp.person_height_px(W // 2, yc), 0.3 * H)  # средняя
        self.assertAlmostEqual(sp.person_height_px(W, yc), 0.2 * H)       # последняя
        self.assertAlmostEqual(sp.person_height_px(-50, yc), 0.4 * H)     # за левым краем (clip)
        self.assertAlmostEqual(sp.person_height_px(W + 999, yc), 0.2 * H)  # за правым краем (clip)
        self.assertAlmostEqual(sp.person_height_px(W // 4, yc), 0.35 * H)  # линейно между 0 и 0.5

    def test_y_dependent_thresholds(self):
        """Регрессия: пороги зависят не только от x, но и от y (2D-профиль)."""
        sp = self._profile(points=((0.5, 0.2, 0.4), (0.5, 0.8, 0.2)))
        xc = W // 2
        self.assertGreater(sp.min_area_at(xc, int(0.2 * H)),
                           sp.min_area_at(xc, int(0.8 * H)))
        self.assertGreater(sp.max_area_at(xc, int(0.2 * H)),
                           sp.max_area_at(xc, int(0.8 * H)))

    def test_min_max_area(self):
        sp = self._profile()
        h_mid = 0.3 * H   # px при x = W/2, y = H/2
        self.assertAlmostEqual(sp.min_area_at(W // 2, H // 2), 0.2 * h_mid ** 2)
        self.assertAlmostEqual(sp.max_area_at(W // 2, H // 2), 4.0 * h_mid ** 2)
        # пороги убывают вглубь кадра вместе с высотой человека
        self.assertGreater(sp.min_area_at(0, H // 2), sp.min_area_at(W, H // 2))

    def test_disabled_falls_back_to_global_thresholds(self):
        sp = self._profile(enabled=False)
        self.assertFalse(sp.adaptive)
        frame_area = float(W * H)
        for x in (0, W // 2, W):
            self.assertAlmostEqual(sp.min_area_at(x, H // 2), 0.0005 * frame_area)
            self.assertAlmostEqual(sp.max_area_at(x, H // 2), 0.25 * frame_area)

    def test_enabled_but_empty_points_falls_back_to_global(self):
        sp = self._profile(enabled=True, points=())
        self.assertFalse(sp.adaptive)
        self.assertAlmostEqual(sp.min_area_at(W // 2, H // 2), 0.0005 * float(W * H))

    def test_single_point_adaptive_constant(self):
        """Адаптивный режим включается уже с 1 точкой; высота — константа."""
        sp = self._profile(points=((0.3, 0.4, 0.3),))
        self.assertTrue(sp.adaptive)
        for x, y in ((0, 0), (W // 2, H // 2), (W, H)):
            self.assertAlmostEqual(sp.person_height_px(x, y), 0.3 * H)

    def test_clamp_min_height_one_px(self):
        sp = self._profile(points=((0.5, 0.5, 0.001),))
        self.assertGreaterEqual(sp.person_height_px(W // 2, H // 2), 1.0)

    def test_y_default_is_frame_center(self):
        """Назад-совместимый вызов без y — середина кадра по y."""
        sp = self._profile(points=((0.5, 0.2, 0.4), (0.5, 0.8, 0.2)))
        self.assertAlmostEqual(sp.person_height_px(W // 2),
                               sp.person_height_px(W // 2, H * 0.5))

    def test_buffer_width(self):
        sp = self._profile()
        self.assertAlmostEqual(sp.buffer_width_px(W // 2, H // 2, 0.75), 0.75 * 0.3 * H)
        self.assertEqual(sp.buffer_width_px(0, H // 2, 0.0), 0.0)
        with self.assertRaises(ValueError):
            sp.buffer_width_px(W // 2, H // 2, -1.0)

    def test_clips_out_of_frame_coordinates(self):
        sp = self._profile()
        # за краями кадра (x < 0 / x > w / y < 0 / y > h) — граница, без падения
        self.assertAlmostEqual(sp.person_height_px(-500, -500),
                               sp.person_height_px(0, 0))
        self.assertAlmostEqual(sp.person_height_px(W * 3, H * 2),
                               sp.person_height_px(W, H))


# ---------------------------------------------------------------------------
# fit_height_surface: чистая функция подгонки поверхности h(x, y)
# ---------------------------------------------------------------------------

class TestFitHeightSurface(unittest.TestCase):
    def test_four_points_biquadratic_exact(self):
        a, b, c, d = 0.35, -0.15, -0.20, 0.10
        pts = [(x, y, a + b * x + c * y + d * x * y)
               for x, y in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))]
        f = fit_height_surface(pts)
        for x, y in ((0.25, 0.75), (0.5, 0.5), (0.9, 0.1), (0.1, 0.3)):
            self.assertAlmostEqual(f.height_at(x, y),
                                   a + b * x + c * y + d * x * y, places=8)

    def test_five_points_on_same_surface_still_exact(self):
        a, b, c, d = 0.3, -0.1, -0.25, 0.05
        pts = [(x, y, a + b * x + c * y + d * x * y)
               for x, y in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0),
                            (1.0, 1.0), (0.5, 0.25))]
        f = fit_height_surface(pts)
        self.assertAlmostEqual(f.height_at(0.7, 0.4),
                               a + b * 0.7 + c * 0.4 + d * 0.28, places=8)

    def test_three_points_plane_exact(self):
        a, b, c = 0.4, -0.2, -0.1
        pts = [(x, y, a + b * x + c * y) for x, y in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))]
        f = fit_height_surface(pts)
        self.assertAlmostEqual(f.height_at(0.5, 0.25), a + b * 0.5 + c * 0.25, places=8)

    def test_two_points_segment_interpolation(self):
        p0, p1 = (0.2, 0.3, 0.4), (0.6, 0.7, 0.2)
        f = fit_height_surface([p0, p1])
        # t=0 и t=1 — сами точки
        self.assertAlmostEqual(f.height_at(0.2, 0.3), 0.4, places=9)
        self.assertAlmostEqual(f.height_at(0.6, 0.7), 0.2, places=9)
        # середина отрезка (t=0.5)
        self.assertAlmostEqual(f.height_at(0.4, 0.5), 0.3, places=9)
        # за краями отрезка — крайнее значение
        self.assertAlmostEqual(f.height_at(0.0, 0.1), 0.4, places=9)   # t<0 → t=0
        self.assertAlmostEqual(f.height_at(0.9, 0.9), 0.2, places=9)   # t>1 → t=1

    def test_one_point_constant(self):
        f = fit_height_surface([(0.3, 0.7, 0.25)])
        for x, y in ((0.0, 0.0), (0.5, 0.5), (1.0, 1.0)):
            self.assertAlmostEqual(f.height_at(x, y), 0.25)

    def test_query_outside_frame_is_clipped(self):
        f = fit_height_surface([(0.0, 0.0, 0.4), (1.0, 1.0, 0.1)])
        self.assertAlmostEqual(f.height_at(-5.0, -5.0), 0.4)
        self.assertAlmostEqual(f.height_at(9.0, 2.0), 0.1)

    def test_empty_points_rejected(self):
        with self.assertRaises(ValueError):
            fit_height_surface([])


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset(unittest.TestCase):
    def test_reset_smoke(self):
        frames = make_noise_scene(7)
        for method in ("mog2", "knn"):
            det = MotionDetector(Config.from_dict({"motion": {"method": method, "history": 100}}))
            for i in range(10):
                f = frames()
                draw_person(f, person_pos(i))
                det.detect(f)
            det.reset()  # не должно бросать исключение
            f = frames()
            draw_person(f, person_pos(10))
            self.assertIsInstance(det.detect(f), list)


if __name__ == "__main__":
    unittest.main()
