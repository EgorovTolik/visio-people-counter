"""Тесты calib_controller.py (задача 15) на синтетическом mp4 (cv2.VideoWriter, 'mp4v').

Контроллер проверяется headless, БЕЗ cv2-окна: open/step/on_click/on_key/on_button.
cv2-драйвер run_calibration окном не поднимается (нет дисплея) — он остался тонким
и покрывается синтаксисом + CLI calibrate --help (см. pi-tasks/report-15-calib-controller.md).

Видео: 320x180 @ 30 fps, 300 кадров (10 секунд), статичный фон + «человек»
(белый прямоугольник) — движется, чтобы MOG2 не впечатывал его в модель фона.
"""

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from visio_people_counter.calib_controller import CalibrationController  # noqa: E402
from visio_people_counter.config import Config  # noqa: E402
from visio_people_counter.video_source import VideoSourceError  # noqa: E402

W, H, FPS, N_FRAMES = 320, 180, 30.0, 300   # 10 секунд


def _make_video(path: Path) -> None:
    """Синтетический mp4: статичная текстура + движущийся белый прямоугольник."""
    rng = np.random.default_rng(7)
    bg = rng.integers(60, 110, size=(H, W, 3), dtype=np.uint8)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    assert vw.isOpened(), "не удалось открыть VideoWriter для синтетического mp4"
    for f in range(N_FRAMES):
        frame = bg.copy()
        x = int(round(40 + 240 * (f / float(N_FRAMES - 1))))
        y = int(round(60 + 30 * np.sin(f * 0.3)))
        cv2.rectangle(frame, (x, y), (x + 40, y + 90), (255, 255, 255), -1)
        vw.write(frame)
    vw.release()


class _ControllerBase(unittest.TestCase):
    """Общий синтетический видеофайл + хелперы создания/открытия контроллера."""

    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory(prefix="vpc_test_calibctrl_")
        cls.dir = Path(cls._td.name)
        cls.video_path = cls.dir / "synth.mp4"
        _make_video(cls.video_path)

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def setUp(self):
        self.ctrl: CalibrationController | None = None
        self.n_cfg = 0   # уникальные имена конфигов внутри общего каталога

    def tearDown(self):
        if self.ctrl is not None:
            self.ctrl.close()
            self.ctrl = None

    def _make_config(self, extra: dict | None = None) -> Config:
        """Минимальный конфиг (file-видео); недостающие блоки — дефолты Config."""
        d = {"video": {"type": "file", "path": str(self.video_path)}}
        if extra:
            d.update(extra)
        self.n_cfg += 1
        p = self.dir / f"cfg_{self._id()}.yaml"
        cfg = Config.from_dict(d, str(p))
        Config.save(cfg, p)
        return cfg

    def _id(self) -> str:
        return f"{type(self).__name__}_{self._testMethodName}"

    def _open(self, cfg: Config, **kw) -> CalibrationController:
        """CalibrationController + open() с заглушкой stdout (Pipeline печатает)."""
        kw.setdefault("cache_frames", 50)
        ctrl = CalibrationController(cfg, **kw)
        with redirect_stdout(io.StringIO()):
            ctrl.open()
        self.ctrl = ctrl
        return ctrl


class TestOpenStep(_ControllerBase):
    """open/step: кадр исходного размера, листание [n], выход → None."""

    def test_open_step_frame_and_quit(self):
        cfg = self._make_config()
        ctrl = self._open(cfg)
        # источник открыт: размер кадра = размеру источника (ROI не задан)
        self.assertEqual((ctrl.width, ctrl.height), (W, H))
        img = ctrl.step()
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.shape, (H, W, 3))
        self.assertFalse(ctrl.quit_requested)

        # [n] — следующий из загруженных кадров: глобальный frame_index растёт
        first = ctrl.frame_index
        ctrl.on_key(ord("n"))
        self.assertEqual(ctrl.frame_index, first + 1)
        # [p] — назад
        ctrl.on_key(ord("p"))
        self.assertEqual(ctrl.frame_index, first)

        # q — выход: quit_requested, дальше step() → None (источник «исчерпан» для драйвера)
        ctrl.on_key(ord("q"))
        self.assertTrue(ctrl.quit_requested)
        self.assertIsNone(ctrl.step())

    def test_open_bad_source_raises(self):
        d = {"video": {"type": "file", "path": str(self.dir / "нет.mp4")}}
        p = self.dir / f"cfg_{self._id()}_bad.yaml"
        cfg = Config.from_dict(d, str(p))
        ctrl = CalibrationController(cfg)
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(VideoSourceError):
                ctrl.open()


class TestButtonsKeys(_ControllerBase):
    """on_button / on_key: режимы, show_all, масштаб (пресеты)."""

    def test_on_button_line_and_show_all(self):
        cfg = self._make_config()
        ctrl = self._open(cfg)
        ctrl.on_button("line")
        self.assertEqual(ctrl.state.mode, "line")
        ctrl.on_button("show_all")
        self.assertTrue(ctrl.show_all)
        ctrl.on_button("show_all")   # toggle обратно
        self.assertFalse(ctrl.show_all)

    def test_key_v_toggles_show_all(self):
        cfg = self._make_config()
        ctrl = self._open(cfg)
        self.assertFalse(ctrl.show_all)
        ctrl.on_key(ord("v"))
        self.assertTrue(ctrl.show_all)
        ctrl.on_key(ord("v"))
        self.assertFalse(ctrl.show_all)

    def test_scale_preset_keys(self):
        from visio_people_counter.calibrate import next_scale
        cfg = self._make_config()
        ctrl = self._open(cfg)   # 320x180 → авто-масштаб до 2.0 (задача 14)
        s0 = ctrl.scale
        ctrl.on_key(ord(","))    # уменьшить: 2.0 → 1.5
        self.assertEqual(ctrl.scale, next_scale(s0, -1))
        ctrl.on_key(ord("."))    # увеличить обратно
        self.assertEqual(ctrl.scale, next_scale(next_scale(s0, -1), 1))


class TestClickPanel(_ControllerBase):
    """on_click: клик по области кнопки панели → действие (hit_button)."""

    def test_click_first_panel_button_enters_line_mode(self):
        cfg = self._make_config()
        ctrl = self._open(cfg)
        img = ctrl.step()   # отрисовка панели — раскладка buttons обновлена
        self.assertIsInstance(img, np.ndarray)
        self.assertTrue(ctrl.buttons)
        name, _label, _active, x0, y0, x1, y1 = ctrl.buttons[0]
        self.assertEqual(name, "line")   # первая кнопка панели — «линия»
        ctrl.on_click((x0 + x1) // 2, (y0 + y1) // 2)
        self.assertEqual(ctrl.state.mode, "line")

    def test_click_second_button_zone_and_miss(self):
        cfg = self._make_config()
        ctrl = self._open(cfg)
        ctrl.step()
        _n, _l, _a, x0, y0, x1, y1 = ctrl.buttons[1]
        self.assertEqual(_n, "zone")
        ctrl.on_click((x0 + x1) // 2, (y0 + y1) // 2)
        self.assertEqual(ctrl.state.mode, "zone")

        # клик в пустой области кадра (ниже панели) — режим не меняется, точка линии/зоны
        ctrl.state.set_mode("line")
        ctrl.on_click(5, H - 5)   # мимо кнопок
        self.assertEqual(ctrl.state.mode, "line")
        self.assertEqual(len(ctrl.state.line_points), 1)   # клик ушёл в режим


class TestSeekTimeInput(_ControllerBase):
    """[t] + цифры + Enter — seek (только файл); ESC — отмена ввода."""

    def test_time_input_seek_moves_frame_index(self):
        cfg = self._make_config()
        ctrl = self._open(cfg)   # cache_frames=50, видео 10 с @30 fps
        self.assertEqual(ctrl.frame_index, 0)

        ctrl.on_key(ord("t"))    # включить режим ввода времени
        self.assertTrue(ctrl.time_input_active)
        ctrl.on_key(ord("5"))    # «5» секунд
        img = ctrl.step()        # в режиме ввода на кадре дублируется буфер (оверлей нарисован)
        self.assertIsInstance(img, np.ndarray)
        ctrl.on_key(13)          # Enter — применить seek

        self.assertFalse(ctrl.time_input_active)
        # FileSource.seek: индекс = round(t * fps) = 150 (точность «до ключевого кадра»
        # не влияет на счётчик index); запас на декодер ±60 кадров избыточен, но безопасен
        self.assertGreaterEqual(ctrl.frame_index, 100)
        self.assertLessEqual(ctrl.frame_index, 200)

    def test_time_input_esc_cancels(self):
        cfg = self._make_config()
        ctrl = self._open(cfg)
        ctrl.on_key(ord("t"))
        ctrl.on_key(ord("7"))
        ctrl.on_key(27)          # ESC — отмена (не выход: вне режима ROI)
        self.assertFalse(ctrl.time_input_active)
        self.assertEqual(ctrl.time_buf.digits, [])
        self.assertEqual(ctrl.frame_index, 0)   # seek не выполнялся
        self.assertFalse(ctrl.quit_requested)

    def test_digit_keys_ignored_outside_time_input(self):
        cfg = self._make_config()
        ctrl = self._open(cfg)
        ctrl.on_key(ord("5"))    # вне режима ввода цифры ничего не делают
        self.assertEqual(ctrl.frame_index, 0)


class TestSave(_ControllerBase):
    """[a]: линия из двух кликов → конфиг записан в save_to."""

    def test_save_line_to_config(self):
        cfg = self._make_config()   # счётчиков нет — линия добавится новой (main_line)
        save_to = self.dir / f"saved_{self._id()}.yaml"
        ctrl = self._open(cfg, save_to=save_to)

        ctrl.on_button("line")
        ctrl.on_click(100, 100)
        ctrl.on_click(220, 150)
        self.assertEqual(ctrl.state.line_points,
                         [(round(100 / W, 4), round(100 / H, 4)),
                          (round(220 / W, 4), round(150 / H, 4))])
        out = io.StringIO()
        with redirect_stdout(out):
            ctrl.on_key(ord("a"))
        self.assertTrue(ctrl.saved_once)

        self.assertTrue(save_to.is_file())
        saved = Config.load(save_to)
        line = next(c for c in saved.counters if c.id == "main_line")
        self.assertEqual(list(line.a), [round(100 / W, 4), round(100 / H, 4)])
        self.assertEqual(list(line.b), [round(220 / W, 4), round(150 / H, 4)])
        self.assertIn("сохранено", out.getvalue())

    def test_save_nothing_changes_notifies(self):
        cfg = self._make_config()
        save_to = self.dir / f"saved_none_{self._id()}.yaml"
        ctrl = self._open(cfg, save_to=save_to)
        ctrl.on_key(ord("a"))   # ничего не собрано → не записывать
        self.assertFalse(ctrl.saved_once)
        self.assertFalse(save_to.exists())
        self.assertIn("ничего не менялось", "; ".join(ctrl.messages))


class TestRoi(_ControllerBase):
    """ROI: кнопка + 2 клика + Enter → cfg.processing.roi, источник переоткрыт с кропом."""

    def test_roi_apply_reopens_source_cropped(self):
        cfg = self._make_config()
        ctrl = self._open(cfg)
        self.assertIsNone(cfg.processing.roi)

        out = io.StringIO()
        with redirect_stdout(out):   # переоткрытие Pipeline печатает в stdout
            ctrl.on_button("roi")
            self.assertEqual(ctrl.state.mode, "roi")
            ctrl.on_click(80, 45)          # угол 1 → (0.25, 0.25)
            ctrl.on_click(240, 135)        # угол 2 → (0.75, 0.75)
            ctrl.on_key(13)                # Enter — принять

        self.assertEqual(cfg.processing.roi, [0.25, 0.25, 0.5, 0.5])
        # источник переоткрыт с кропом: кадр стал размером ROI (160x90)
        self.assertEqual((ctrl.width, ctrl.height), (160, 90))
        img = ctrl.step()   # следующий кадр — уже в ROI-размере
        self.assertEqual(img.shape, (90, 160, 3))
        self.assertIsNone(ctrl.state.mode)   # после принятия режим сброшен

    def test_roi_esc_cancels(self):
        cfg = self._make_config()
        ctrl = self._open(cfg)
        ctrl.on_button("roi")
        ctrl.on_click(80, 45)
        ctrl.on_key(27)          # ESC в режиме ROI — отмена (не выход)
        self.assertIsNone(ctrl.state.mode)
        self.assertEqual(ctrl.state.roi_points, [])
        self.assertFalse(ctrl.quit_requested)
        self.assertIsNone(cfg.processing.roi)   # источник не переоткрывался
        self.assertEqual((ctrl.width, ctrl.height), (W, H))


if __name__ == "__main__":
    unittest.main()
