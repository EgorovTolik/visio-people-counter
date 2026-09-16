"""Тесты video_source.py на синтетическом mp4 (cv2.VideoWriter, 'mp4v').

Временное видео: 960x540, 30 fps, 30 кадров, движущийся белый квадрат.
Файл создаётся один раз на класс (setUpClass).
"""

import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from visio_people_counter.video_source import (  # noqa: E402
    FileSource, FfmpegPipeSource, VideoSourceError, ffprobe_info, roi_crop_pixels,
)

W, H, FPS, N_FRAMES = 960, 540, 30.0, 30


def _make_video(path: Path) -> None:
    """Синтетический mp4: тёмный фон + движущийся белый квадрат."""
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    assert vw.isOpened(), "не удалось открыть VideoWriter для синтетического mp4"
    for i in range(N_FRAMES):
        frame = np.full((H, W, 3), 30, dtype=np.uint8)
        x = 10 + i * (W - 220) // (N_FRAMES - 1)   # квадрат едет слева направо
        cv2.rectangle(frame, (x, H // 2 - 60), (x + 160, H // 2 + 60), (255, 255, 255), -1)
        vw.write(frame)
    vw.release()


class _VideoMixin:
    """Создаёт временное видео перед классом и удаляет после."""

    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory(prefix="vpc_test_video_")
        cls.video_path = Path(cls._td.name) / "synth.mp4"
        _make_video(cls.video_path)

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()


class TestFileSource(_VideoMixin, unittest.TestCase):
    def test_reads_all_frames(self):
        with FileSource(str(self.video_path)) as src:
            self.assertEqual(src.width, W)
            self.assertEqual(src.height, H)
            self.assertAlmostEqual(src.fps, FPS, delta=0.5)
            frames = []
            while True:
                f = src.read()
                if f is None:
                    break
                frames.append(f)
        self.assertEqual(len(frames), N_FRAMES)
        for i, f in enumerate(frames):
            self.assertEqual(f.index, i)
            self.assertEqual(f.image.shape, (H, W, 3))
            self.assertGreater(f.image.mean(), 20)   # не чёрный (есть квадрат)

    def test_loop_file(self):
        with FileSource(str(self.video_path), loop_file=True) as src:
            total = 0
            saw_wrap = False
            prev_index = -1
            for _ in range(N_FRAMES * 2 + 5):
                f = src.read()
                if f is None:
                    break
                if f.index < prev_index:
                    saw_wrap = True
                prev_index = f.index
                total += 1
        self.assertGreaterEqual(total, N_FRAMES)      # дошёл хотя бы до первого EOF
        self.assertTrue(saw_wrap, "loop_file должен перемотать в начало")

    def test_duration_known_and_close_to_expected(self):
        with FileSource(str(self.video_path)) as src:
            d = src.duration
            # 30 кадров @ 30 fps = ~1 с; контейнер может добавить дробь — допускаем разброс
            self.assertGreater(d, 0.5)
            self.assertLess(d, 2.0)

    def test_missing_file(self):
        with self.assertRaises(VideoSourceError):
            FileSource("/nonexistent/nope.mp4").open()

    def test_seek_before_open_returns_false(self):
        src = FileSource(str(self.video_path))
        self.addCleanup(src.close)
        self.assertFalse(src.seek(1.0), "seek до open() должен вернуть False")

    def test_seek_zero_reads_from_start(self):
        with FileSource(str(self.video_path)) as src:
            # после seek(0) — полная последовательность с index=0 (как без seek)
            self.assertTrue(src.seek(0.0))
            frames = []
            while True:
                f = src.read()
                if f is None:
                    break
                frames.append(f)
        self.assertEqual(len(frames), N_FRAMES)
        self.assertEqual(frames[0].index, 0)
        # t_video первого кадра — в начале таймлайна
        self.assertIsNotNone(frames[0].t_video)
        self.assertLess(frames[0].t_video, 0.1)

    def test_seek_midway_starts_later(self):
        """seek(0.5) на видео 1 c @30fps: чтение начинается позже → до EOF меньше кадров."""
        with FileSource(str(self.video_path)) as src:
            self.assertTrue(src.seek(0.5))
            frames = []
            while True:
                f = src.read()
                if f is None:
                    break
                frames.append(f)
        # до ключевого кадра: начало в районе 15-го кадра (±2 на точность декодера)
        self.assertGreater(len(frames), 0, "после seek(0.5) не прочиталось ни одного кадра")
        self.assertLess(len(frames), N_FRAMES,
                        "seek(0.5) должен начать чтение позже начала файла")
        self.assertEqual(frames[0].index, int(round(0.5 * FPS)))   # _index сброшен по fps
        if frames[0].t_video is not None:
            # t_video согласуется с меткой (±0.3 c — точность «до ключевого кадра»)
            self.assertAlmostEqual(frames[0].t_video, 0.5, delta=0.3)

    def test_seek_then_close_and_reopen(self):
        """seek не ломает повторное открытие: после close/open чтение снова с начала."""
        src = FileSource(str(self.video_path))
        self.addCleanup(src.close)
        src.open()
        self.assertTrue(src.seek(0.7))
        src.read()
        src.close()
        self.assertFalse(src.seek(1.0), "seek после close() → False")
        src.open()
        f = src.read()
        self.assertIsNotNone(f)
        self.assertEqual(f.index, 0)


class TestSeekUnsupported(unittest.TestCase):
    """Базовый VideoSource.seek: ffmpeg-пайп/HLS — случайного доступа нет → False."""

    def test_ffmpeg_pipe_source_seek_returns_false(self):
        src = FfmpegPipeSource("https://example.com/stream.m3u8")
        self.addCleanup(src.close)
        self.assertFalse(src.seek(5.0))
        # и без open/close — False, без побочных эффектов

    def test_abc_default_seek_is_false(self):
        """Подкласс, не переопределяющий seek, наследует базовый False (как pipe)."""
        from visio_people_counter.video_source import VideoSource

        class _NoSeekSource(VideoSource):
            def open(self) -> None:
                pass

            def read(self):
                return None

            def close(self) -> None:
                pass

            @property
            def width(self) -> int:
                return 0

            @property
            def height(self) -> int:
                return 0

            @property
            def fps(self) -> float:
                return 0.0

        self.assertFalse(_NoSeekSource().seek(1.0))


class TestFfmpegPipeSource(_VideoMixin, unittest.TestCase):
    def test_ffprobe_info(self):
        info = ffprobe_info(str(self.video_path))
        self.assertEqual(info["width"], W)
        self.assertEqual(info["height"], H)
        self.assertIsNotNone(info["fps"])
        self.assertGreater(info["fps"], 0)
        self.assertIn("codec_name", info)

    def test_same_frame_sequence_as_file(self):
        with FfmpegPipeSource(str(self.video_path)) as src:
            self.assertEqual(src.width, W)
            self.assertEqual(src.height, H)
            frames = []
            while True:
                f = src.read()
                if f is None:
                    break
                frames.append(f)
        # та же последовательность кадров (±1 на границах декодера)
        self.assertAlmostEqual(len(frames), N_FRAMES, delta=2)
        for i, f in enumerate(frames):
            self.assertEqual(f.index, i)
        # контент: квадрат светлее фона
        first = frames[0].image
        self.assertGreater(first[:, :, 0].max(), 200)

    def test_effective_fps_and_max_width(self):
        t0 = time.monotonic()
        with FfmpegPipeSource(
            str(self.video_path),
            effective_fps=10,
            max_width=640,
        ) as src:
            # 960 -> 640, высота 540*640/960 = 360 (чётная)
            self.assertEqual(src.width, 640)
            self.assertEqual(src.height, 360)
            self.assertEqual(src.fps, 10.0)
            frames = []
            while True:
                f = src.read()
                if f is None:
                    break
                frames.append(f)
        elapsed = time.monotonic() - t0
        # 30 кадров @30fps = 1 c видео; при -r 10 получаем ~10 кадров
        self.assertTrue(8 <= len(frames) <= 12, f"ожидалось ~10 кадров, получено {len(frames)}")
        for f in frames:
            self.assertEqual(f.image.shape, (360, 640, 3))
        # скорость: ffmpeg с файловым входом режет fps быстрее реального времени —
        # 1 c видео не должно занимать заметно больше секунды wall-time
        self.assertLess(elapsed, 5.0)

    def test_unavailable_input_no_reconnect(self):
        # несуществующий вход + reconnect_attempts=-1: ffmpeg умирает сразу,
        # watchdog не переподключается -> read() стабильно возвращает None
        src = FfmpegPipeSource(
            "/nonexistent/stream.m3u8",
            reconnect_attempts=-1,
            reconnect_backoff_s=0.05,
            bad_read_threshold=2,
        )
        self.addCleanup(src.close)
        with self.assertRaises(VideoSourceError):
            src.open()   # ffprobe не находит вход


class TestRoiCropPixels(unittest.TestCase):
    """Задача 13: roi_crop_pixels — доли кадра → пиксельный срез (чистая функция)."""

    def test_basic(self):
        self.assertEqual(roi_crop_pixels([0.25, 0.25, 0.5, 0.5], 960, 540),
                         (240, 135, 480, 270))

    def test_rounding(self):
        # 0.3333*100 = 33.33 → 33; срез не выходит за кадр
        x, y, w, h = roi_crop_pixels([0.3333, 0.6667, 0.25, 0.25], 100, 100)
        self.assertEqual((x, y), (33, 67))
        self.assertEqual((w, h), (25, 25))
        self.assertLessEqual(x + w, 100)
        self.assertLessEqual(y + h, 100)

    def test_degenerate_sub_pixel_becomes_one_px(self):
        # доля меньше пикселя → не 0, а минимум 1 px (защита от вырожденного среза)
        x, y, w, h = roi_crop_pixels([0.5, 0.5, 0.0004, 0.0004], 960, 540)
        self.assertGreaterEqual(w, 1)
        self.assertGreaterEqual(h, 1)

    def test_clamp_to_frame_edges(self):
        # x + w > 1 в пикселях из-за округления → ширина сжимается до края кадра
        x, y, w, h = roi_crop_pixels([0.99, 0.05, 0.2, 0.2], 100, 50)
        self.assertEqual(x, 99)
        self.assertLessEqual(w, 100 - x)
        self.assertGreaterEqual(w, 1)

    def test_full_frame(self):
        self.assertEqual(roi_crop_pixels([0.001, 0.001, 0.999, 0.999], 1000, 1000),
                         (1, 1, 999, 999))

    def test_bad_frame_size_raises(self):
        with self.assertRaises(ValueError):
            roi_crop_pixels([0.1, 0.1, 0.5, 0.5], 0, 100)

    def test_bad_roi_length_raises(self):
        with self.assertRaises(ValueError):
            roi_crop_pixels([0.1, 0.1, 0.5], 100, 100)


class TestFileSourceRoi(_VideoMixin, unittest.TestCase):
    """Задача 13: FileSource с roi — source.width/height = размер ROI, кроп NumPy-срезом."""

    def test_width_height_are_roi_size(self):
        with FileSource(str(self.video_path), roi=[0.25, 0.25, 0.5, 0.5]) as src:
            self.assertEqual(src.width, W // 2)     # 480
            self.assertEqual(src.height, H // 2)    # 270
            f = src.read()
            self.assertIsNotNone(f)
            self.assertEqual(f.image.shape, (H // 2, W // 2, 3))

    def test_crop_matches_numpy_slice_of_full_frame(self):
        with FileSource(str(self.video_path)) as full:
            frame_full = full.read().image
        x, y, cw, ch = roi_crop_pixels([0.25, 0.25, 0.5, 0.5], W, H)
        expected = frame_full[y:y + ch, x:x + cw]
        with FileSource(str(self.video_path), roi=[0.25, 0.25, 0.5, 0.5]) as src:
            cropped = src.read().image
        self.assertTrue(np.array_equal(cropped, expected))

    def test_no_roi_unchanged(self):
        with FileSource(str(self.video_path)) as src:
            self.assertEqual(src.width, W)
            self.assertEqual(src.height, H)
            self.assertIsNone(src.roi_px)
            f = src.read()
            self.assertEqual(f.image.shape, (H, W, 3))

    def test_roi_px_attribute_set(self):
        with FileSource(str(self.video_path), roi=[0.25, 0.25, 0.5, 0.5]) as src:
            self.assertEqual(src.roi_px, (W // 4, H // 4, W // 2, H // 2))

    def test_frames_count_same_as_without_roi(self):
        with FileSource(str(self.video_path), roi=[0.1, 0.1, 0.8, 0.8]) as src:
            n = 0
            while src.read() is not None:
                n += 1
        self.assertEqual(n, N_FRAMES)


class TestFfmpegPipeSourceRoi(_VideoMixin, unittest.TestCase):
    """Задача 13: FfmpegPipeSource с roi — кроп через ffmpeg -vf crop на уровне источника."""

    def test_width_height_and_frame_are_roi_size(self):
        src = FfmpegPipeSource(str(self.video_path), roi=[0.25, 0.25, 0.5, 0.5])
        self.addCleanup(src.close)
        src.open()
        self.assertEqual((src.width, src.height), (W // 2, H // 2))
        f = src.read()
        self.assertIsNotNone(f)
        self.assertEqual(f.image.shape, (H // 2, W // 2, 3))

    def test_no_roi_unchanged(self):
        src = FfmpegPipeSource(str(self.video_path))
        self.addCleanup(src.close)
        src.open()
        self.assertEqual((src.width, src.height), (W, H))
        self.assertIsNone(src.roi_px)


if __name__ == "__main__":
    unittest.main()
