"""Тесты tracker_adapter.py на синтетических blob'ах (без видео и без cv2).

Синтетика: «люди» — прямоугольники 60x120, движущиеся по прямой с постоянной
скоростью; Kalman-трекеры при такой траектории ведут себя детерминированно.

Проверяется (по ТЗ задачи 13):
* track_id сохраняется после пропадания объекта на время ≤ lost_track_buffer
  (перекрытие) и новые объекты получают новые id;
* неподтверждённые (меньше minimum_consecutive_frames кадров подряд) — track_id=-1;
* пустые списки blob'ов не роняют адаптер, а «старые» треки закрываются
  после исчерпания буфера;
* reset() полностью очищает состояние: id начинаются заново с 0.
"""

import sys
import unittest
from pathlib import Path

# корень проекта — родитель каталога tests/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from visio_people_counter.config import Config  # noqa: E402
from visio_people_counter.motion_detector import Blob  # noqa: E402
from visio_people_counter.tracker_adapter import (  # noqa: E402
    DEFAULT_FRAME_RATE,
    TrackerAdapter,
)

W = 640  # условная ширина кадра (для ограничения траекторий)


def make_blob(x: int, y: int = 140, w: int = 60, h: int = 120) -> Blob:
    return Blob(x=x, y=y, w=w, h=h, area=w * h, cx=x + w / 2.0, cy=y + h / 2.0)


def make_adapter(tracker_type="sort", frame_rate=None, **tracker_overrides) -> TrackerAdapter:
    """Адаптер с типовым блоком tracker (lost_track_buffer=60, min_consec=2, iou=0.3)."""
    t = {
        "type": tracker_type,
        "lost_track_buffer": 60,
        "minimum_consecutive_frames": 2,
        "minimum_iou_threshold": 0.3,
    }
    t.update(tracker_overrides)
    return TrackerAdapter(Config.from_dict({"tracker": t}), frame_rate=frame_rate)


class TestTrackerAdapter(unittest.TestCase):
    # ------------------------------------------------------------- fps/buffer

    def test_default_frame_rate(self):
        ad = make_adapter()  # frame_rate не передан
        self.assertEqual(ad.frame_rate, DEFAULT_FRAME_RATE)
        # пакет масштабирует buffer: max(1, ceil(15/30 * 60)) = 30 кадров при 15 fps
        self.assertEqual(ad.maximum_frames_without_update, 30)

    def test_set_frame_rate_rescales_buffer_and_resets(self):
        ad = make_adapter()  # 15 fps → buffer 30 кадров
        for f in range(10):
            ad.update([make_blob(40 + 5 * f)])
        ad.set_frame_rate(30.0)
        self.assertEqual(ad.frame_rate, 30.0)
        # ceil(30/30 * 60) = 60 кадров при 30 fps
        self.assertEqual(ad.maximum_frames_without_update, 60)
        # состояние сброшено: объект снова неподтверждён в первый кадр
        res = ad.update([make_blob(40)])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].track_id, -1)
        with self.assertRaises(ValueError):
            ad.set_frame_rate(0.0)

    # ------------------------------------------------------- подтверждение id

    def test_unconfirmed_single_frame_is_minus_one(self):
        ad = make_adapter()  # minimum_consecutive_frames=2
        res = ad.update([make_blob(100)])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].track_id, -1)     # 1-й кадр — трек не подтверждён
        self.assertEqual(res[0].age_frames, 0)
        # второй подряд кадр в том же месте → трек подтверждается
        res2 = ad.update([make_blob(105)])
        self.assertGreaterEqual(res2[0].track_id, 0)
        self.assertEqual(res2[0].age_frames, 1)

    def test_id_preserved_through_occlusion(self):
        """A пропадает на 9 кадров (≤ buffer) → id сохраняется; B не трогается."""
        ad = make_adapter(frame_rate=15.0)  # buffer = 30 кадров
        GAP = range(30, 39)  # 9 кадров перекрытия
        ids_a, ids_b = {}, {}
        for f in range(50):
            blobs = []
            if f not in GAP:
                blobs.append(make_blob(40 + 5 * f, y=60))    # A — вправо
            blobs.append(make_blob(500 - 3 * f, y=240))      # B — влево
            res = ad.update(blobs)
            self.assertEqual(len(res), len(blobs))
            if f == 0:   # первый кадр — треки ещё неподтверждены (-1)
                self.assertTrue(all(r.track_id < 0 for r in res))
                continue
            idx = 0
            if f not in GAP:
                ids_a[f] = res[idx].track_id
                idx += 1
            ids_b[f] = res[idx].track_id
        # A подтвержден до пропадания и после — тот же id (≥ 1)
        self.assertGreaterEqual(ids_a[25], 0)
        self.assertEqual(ids_a[40], ids_a[25])
        # B всё время с одним id, отличным от A
        self.assertGreaterEqual(ids_b[1], 0)
        self.assertEqual(set(ids_b.values()), {ids_b[1]})
        self.assertNotIn(ids_b[1], set(ids_a.values()))

    def test_new_object_gets_new_id(self):
        """Новый объект, появившийся позже, получает свой id; старые не меняются."""
        ad = make_adapter(frame_rate=15.0)
        a_ids, c_ids = {}, {}
        for f in range(40):
            blobs = [make_blob(40 + 4 * f, y=60)]          # A — с кадра 0
            if f >= 25:
                blobs.append(make_blob(450 - 4 * (f - 25), y=240))  # C — с кадра 25
            res = ad.update(blobs)
            if f == 0:  # A ещё неподтверждён
                self.assertLess(res[0].track_id, 0)
                continue
            a_ids[f] = res[0].track_id
            if f == 25:  # первый кадр C — неподтверждён
                self.assertLess(res[1].track_id, 0)
            elif f >= 26:  # C подтверждён на своём втором кадре
                c_ids[f] = res[1].track_id
        self.assertGreaterEqual(a_ids[5], 0)
        self.assertEqual(set(a_ids.values()), {a_ids[5]})   # A всё время с одним id
        self.assertGreaterEqual(c_ids[26], 0)
        self.assertNotIn(c_ids[26], set(a_ids.values()))    # C — новый id
        self.assertEqual(set(c_ids.values()), {c_ids[26]})

    # ------------------------------------------------- пустые кадры и закрытие

    def test_empty_frames_do_not_crash_and_close_tracks(self):
        ad = make_adapter(frame_rate=15.0)  # buffer = 30 кадров
        for f in range(20):
            res = ad.update([make_blob(40 + 5 * f)])
        old_id = res[0].track_id
        self.assertGreaterEqual(old_id, 0)
        # 50 пустых кадров (> buffer=30) — адаптер не роняет, треки закрываются
        for _ in range(50):
            self.assertEqual(ad.update([]), [])
        # новый объект на том же месте получает НОВЫЙ id (старый трек закрыт)
        for f in range(3):
            res = ad.update([make_blob(40 + 5 * f)])
        new_id = res[0].track_id
        self.assertGreaterEqual(new_id, 0)
        self.assertNotEqual(new_id, old_id)

    def test_short_gap_keeps_track(self):
        """Краткий разрыв (3 кадра) — трек НЕ закрывается, id тот же."""
        ad = make_adapter(frame_rate=15.0)
        ids = {}
        for f in range(24):
            if 8 <= f < 11:
                ad.update([])
                continue
            res = ad.update([make_blob(40 + 5 * f)])
            if f > 0:  # кадр 0 — неподтверждённый (-1), не участвует в сравнении
                ids[f] = res[0].track_id
        self.assertGreaterEqual(ids[6], 0)
        self.assertEqual(set(ids.values()), {ids[6]})

    # ------------------------------------------------------------------ reset

    def test_reset_clears_state_and_ids_restart(self):
        ad = make_adapter(frame_rate=15.0)
        for f in range(10):
            res = ad.update([make_blob(40 + 5 * f)])
        first_id = res[0].track_id
        self.assertEqual(first_id, 0)  # первый трек в «жизни» получает id=0
        # имитируем обрыв потока: reset()
        ad.reset()
        res = ad.update([make_blob(40)])
        self.assertLess(res[0].track_id, 0)          # первый кадр — неподтверждён
        self.assertEqual(ad._age, {})                # возраст-счётчик чистый
        res = ad.update([make_blob(45)])
        self.assertEqual(res[0].track_id, 0)         # счётчик id начался заново
        self.assertEqual(res[0].age_frames, 1)       # age тоже сброшен

    # ------------------------------------------------- все три типа трекеров

    def test_all_tracker_types_smoke(self):
        for ttype in ("sort", "bytetrack", "botsort"):
            with self.subTest(tracker=ttype):
                ad = make_adapter(ttype, frame_rate=15.0)
                self.assertEqual(ad.tracker_type, ttype)
                ids = {}
                for f in range(25):
                    res = ad.update([make_blob(40 + 5 * f)])
                    if f >= 1:
                        ids[f] = res[0].track_id
                self.assertGreaterEqual(ids[1], 0)
                self.assertEqual(set(ids.values()), {ids[1]})  # id стабилен


if __name__ == "__main__":
    unittest.main()
