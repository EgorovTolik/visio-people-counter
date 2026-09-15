"""Источники кадров: файл (cv2.VideoCapture) и HLS/URL (ffmpeg + rawvideo-pipe).

Единственный источник кадров конвейера: ``VideoSource.read() -> Frame | None``.
Watchdog переподключения ffmpeg живёт здесь (§4 отчёта pi-research/04); модуль
ничего не знает о детекции/трекинге — лишь опциональный хук ``on_reconnect``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np


@dataclass
class Frame:
    """Один кадр из источника."""
    image: np.ndarray            # BGR, uint8, (h, w, 3)
    index: int                   # порядковый номер кадра в текущем «проходе» (с 0)
    t_wall: float                # time.monotonic() момента получения
    t_video: Optional[float] = None   # время в таймлайне видео, сек (если известно)


class VideoSourceError(Exception):
    """Ошибка открытия/инициализации источника видео."""


def is_url(path_or_url: str) -> bool:
    """True — вход выглядит как URL-поток (HLS/RTSP/http), а не локальный файл."""
    return "://" in path_or_url


def ffprobe_info(path_or_url: str, timeout_s: float = 30.0) -> dict:
    """Запросить ffprobe по первому видео-стриму.

    :return: dict с ключами ``width``, ``height``, ``fps`` (float|None),
        ``codec_name``, ``duration`` (float|None), ``format_name`` (str).
    :raises VideoSourceError: ffprobe не найден / вход недоступен.
    """
    if shutil.which("ffprobe") is None:
        raise VideoSourceError("ffprobe не найден в PATH")
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,codec_name:format=duration,format_name",
        "-of", "json",
        path_or_url,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise VideoSourceError(f"ffprobe не выполнен для {path_or_url!r}: {e}") from e
    if proc.returncode != 0:
        raise VideoSourceError(
            f"ffprobe завершился с кодом {proc.returncode} для {path_or_url!r}: {proc.stderr.strip()[:300]}"
        )
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise VideoSourceError(f"ffprobe: не удалось разобрать JSON-ответ: {e}") from e

    streams = data.get("streams") or []
    if not streams:
        raise VideoSourceError(f"ffprobe: видео-стрим не найден в {path_or_url!r}")
    st = streams[0]
    fmt = data.get("format") or {}

    def _rate(key: str) -> Optional[float]:
        raw = st.get(key) or ""
        try:
            if "/" in raw:
                num, den = raw.split("/", 1)
                den_f = float(den)
                return float(num) / den_f if den_f > 0 else None
            v = float(raw)
            return v if v > 0 else None
        except (ValueError, ZeroDivisionError):
            return None

    fps = _rate("r_frame_rate") or _rate("avg_frame_rate")
    duration: Optional[float] = None
    try:
        duration = float(fmt.get("duration"))
    except (TypeError, ValueError):
        duration = None

    return {
        "width": int(st.get("width", 0)),
        "height": int(st.get("height", 0)),
        "fps": fps,
        "codec_name": st.get("codec_name", ""),
        "duration": duration,
        "format_name": fmt.get("format_name", ""),
    }


class VideoSource(ABC):
    """Абстракция «источник кадров»."""

    @abstractmethod
    def open(self) -> None:
        """Открыть источник. :raises VideoSourceError: если открыть нельзя."""

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """Прочитать следующий кадр; ``None`` = EOF либо «недоступен сейчас» (разрыв)."""

    def seek(self, seconds: float) -> bool:
        """Перемотать позицию чтения к метке ``seconds`` (сек) в таймлайне видео.

        Базовая реализация возвращает ``False`` — случайный доступ недоступен
        (например, ffmpeg-пайп/HLS-поток: только последовательное чтение).
        Подклассы с файловым входом переопределяют и возвращают True при успехе.
        """
        return False

    @abstractmethod
    def close(self) -> None:
        """Закрыть источник и освободить ресурсы."""

    @property
    @abstractmethod
    def width(self) -> int:
        """Ширина обрабатываемого кадра, px."""

    @property
    @abstractmethod
    def height(self) -> int:
        """Высота обрабатываемого кадра, px."""

    @property
    @abstractmethod
    def fps(self) -> float:
        """FPS источника (обрабатываемой последовательности)."""

    @property
    def duration(self) -> float:
        """Длительность видео, с; 0.0 — если неизвестна."""
        return 0.0

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class FileSource(VideoSource):
    """Файл видео через ``cv2.VideoCapture``.

    :param path: путь к файлу (mp4/avi/mkv...).
    :param loop_file: повторять файл после EOF (перемотка в начало; событие
        повторного начала НЕ сигнализируется — по ТЗ достаточно просто
        продолжить кадры с index=0).
    """

    def __init__(self, path: str, loop_file: bool = False) -> None:
        self._path = path
        self._loop = loop_file
        self._cap: Optional[cv2.VideoCapture] = None
        self._index = 0
        self._w = 0
        self._h = 0
        self._fps = 0.0

    def open(self) -> None:
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            raise VideoSourceError(f"cv2.VideoCapture не смог открыть файл: {self._path!r}")
        self._cap = cap
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if w <= 0 or h <= 0:
            raise VideoSourceError(f"{self._path!r}: ffprobe/cv2 не вернули разрешение ({w}x{h})")
        self._w, self._h = w, h
        self._fps = fps if fps > 0 else 0.0
        self._index = 0

    def read(self) -> Optional[Frame]:
        if self._cap is None:
            raise VideoSourceError("FileSource.read() до open()")
        ret, frame = self._cap.read()
        if not ret or frame is None:
            if self._loop:
                # EOF файла в режиме loop: перемотка в начало.
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._index = 0
                ret2, frame2 = self._cap.read()
                if not ret2 or frame2 is None:
                    return None
                frame = frame2
            else:
                return None
        t_video = (self._index / self._fps) if self._fps > 0 else None
        out = Frame(image=frame, index=self._index, t_wall=time.monotonic(), t_video=t_video)
        self._index += 1
        return out

    def seek(self, seconds: float) -> bool:
        """Перемотать к метке ``seconds`` (сек): CAP_PROP_POS_MSEC.

        Точность — «до ключевого кадра» (декодер cv2.VideoCapture встаёт на
        ближайший доступный кадр не позже метки); для калибровки по времени
        это допустимо. Позиция внутреннего счётчика ``_index`` сбрасывается
        в ``int(round(seconds * fps))`` (0, если fps неизвестен).

        :returns: True — перемотка выполнена (источник открыт), False — источник
            закрыт/не открыт.
        """
        if self._cap is None:
            return False
        self._cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.0)
        self._index = int(round(seconds * self._fps)) if self._fps > 0 else 0
        return True

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def width(self) -> int:
        return self._w

    @property
    def height(self) -> int:
        return self._h

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def duration(self) -> float:
        """Длительность файла, с (FRAME_COUNT/fps; фолбэк CAP_PROP_DURATION). 0.0 — неизвестна."""
        if self._cap is None:
            return 0.0
        if self._fps > 0:
            n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if n > 0:
                return float(n) / self._fps
        d = float(self._cap.get(cv2.CAP_PROP_DURATION) or 0.0)
        return d / 1000.0 if d > 0 else 0.0


class FfmpegPipeSource(VideoSource):
    """Видео через внешний ffmpeg (процесс + rawvideo-pipe на stdout).

    Перед стартом вызывается ffprobe (разрешение/fps/кодек). Команда ffmpeg:
    ``-loglevel error``, вход = путь/URL; при ``effective_fps > 0`` добавляется
    ``-r N``; при ``max_width > 0`` — даунскейл до ``max_width`` (высота чётная);
    вывод ``-f rawvideo -pix_fmt bgr24 pipe:1``.

    Watchdog (§4 отчёта 04): после ``bad_read_threshold`` подряд неудачных
    read (короткое/пустое чтение или смерть ffmpeg с ненулевым кодом) процесс
    убивается и запускается снова после паузы ``reconnect_backoff_s``.
    ``reconnect_attempts``: 0 = бесконечно, -1 = не переподключаться.
    EOF файла через pipe (ffmpeg завершился кодом 0) — graceful close.

    :param path_or_url: путь к файлу или URL (HLS/RTSP/http).
    :param effective_fps: целевой fps; 0 = нативный.
    :param max_width: даунскейл до ширины; 0 = как в источнике.
    :param reconnect_attempts: 0 = бесконечно, -1 = не переподключаться.
    :param reconnect_backoff_s: пауза между попытками, сек.
    :param bad_read_threshold: подряд неудачных read() -> считать обрывом.
    :param on_reconnect: callable(), вызывается после каждого успешного
        переподключения (pipeline повесит reset детектора/трекера).
    """

    def __init__(
        self,
        path_or_url: str,
        effective_fps: float = 0,
        max_width: int = 0,
        reconnect_attempts: int = 0,
        reconnect_backoff_s: float = 5.0,
        bad_read_threshold: int = 10,
        on_reconnect: Optional[Callable[[], None]] = None,
    ) -> None:
        self._path_or_url = path_or_url
        self._effective_fps = effective_fps
        self._max_width = max_width
        self._reconnect_attempts = reconnect_attempts
        self._backoff_s = reconnect_backoff_s
        self._bad_read_threshold = bad_read_threshold
        self.on_reconnect = on_reconnect

        self._proc: Optional[subprocess.Popen] = None
        self._fails = 0
        self._restarts = 0
        self._eof = False          # graceful EOF (ffmpeg завершился кодом 0)
        self._stopped = False      # исчерпаны попытки / reconnect_attempts == -1

        self._probe: dict = {}
        self._w = 0
        self._h = 0
        self._fps = 0.0
        self._index = 0

    # --- свойства -----------------------------------------------------------

    @property
    def width(self) -> int:
        return self._w

    @property
    def height(self) -> int:
        return self._h

    @property
    def fps(self) -> float:
        """FPS обрабатываемой последовательности (effective_fps или нативный)."""
        if self._effective_fps > 0:
            return float(self._effective_fps)
        return self._fps

    @property
    def probe_info(self) -> dict:
        """Результат ffprobe: width/height/fps/codec_name/duration/format_name."""
        return dict(self._probe)

    @property
    def exhausted(self) -> bool:
        """True — источник исчерпан (graceful EOF или исчерпаны попытки reconnect);
        дальнейшие ``read()`` вернут None. Пока False, ``None`` из ``read()`` может
        означать временный разрыв/переподключение (pipeline, задача 15)."""
        return self._eof or self._stopped

    # --- управление ---------------------------------------------------------

    def open(self) -> None:
        if shutil.which("ffmpeg") is None:
            raise VideoSourceError("ffmpeg не найден в PATH")
        info = ffprobe_info(self._path_or_url)
        self._probe = info
        src_w, src_h = info["width"], info["height"]
        if src_w <= 0 or src_h <= 0:
            raise VideoSourceError(f"{self._path_or_url!r}: ffprobe не вернул разрешение")
        if self._max_width > 0 and self._max_width < src_w:
            # даунскейл до max_width, высота чётная (bgr24/rawvideo любят чётные)
            self._w = int(self._max_width)
            self._h = max(2, int(round(src_h * self._max_width / src_w / 2)) * 2)
        else:
            self._w, self._h = src_w, src_h
        self._fps = info["fps"] or 0.0
        self._spawn()

    def close(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
            except OSError:
                pass
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            self._close_stdout()
            self._proc = None

    def _close_stdout(self) -> None:
        """Закрыть pipe-буфер stdout (защита от ResourceWarning)."""
        if self._proc is not None and self._proc.stdout is not None:
            try:
                self._proc.stdout.close()
            except (OSError, ValueError):
                pass

    # --- внутреннее ---------------------------------------------------------

    def _build_cmd(self) -> list[str]:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            # сетевые таймауты (~5 c), чтобы read не висел вечно при обрыве:
            "-rw_timeout", "5000000",
            "-i", self._path_or_url,
        ]
        if self._effective_fps > 0:
            cmd += ["-r", str(int(self._effective_fps))]
        if (self._w, self._h) != (self._probe["width"], self._probe["height"]):
            cmd += ["-s", f"{self._w}x{self._h}"]
        cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
        return cmd

    def _spawn(self) -> None:
        try:
            self._proc = subprocess.Popen(
                self._build_cmd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except OSError as e:
            raise VideoSourceError(f"не удалось запустить ffmpeg для {self._path_or_url!r}: {e}") from e
        self._fails = 0

    def _kill_proc(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
            except OSError:
                pass
            self._close_stdout()
            self._proc = None

    def _can_reconnect(self) -> bool:
        if self._reconnect_attempts < 0:
            return False
        if self._reconnect_attempts == 0:
            return True
        return self._restarts < self._reconnect_attempts

    def _handle_bad_read(self) -> None:
        """Одна неудача подряд; при пороге — kill + backoff + respawn."""
        self._fails += 1
        if self._fails < self._bad_read_threshold:
            return
        # обрыв: ffmpeg убиваем и переподключаемся (если разрешено)
        self._kill_proc()
        if not self._can_reconnect():
            self._stopped = True
            return
        time.sleep(self._backoff_s)
        self._restarts += 1
        self._spawn()
        if callable(self.on_reconnect):
            try:
                self.on_reconnect()
            except Exception:
                # хук не должен ронять конвейер, но ошибки видим в логе
                import traceback
                traceback.print_exc()

    # --- VideoSource --------------------------------------------------------

    def read(self) -> Optional[Frame]:
        if self._eof or self._stopped:
            return None
        if self._proc is None:
            # первый кадр (open() уже запустил ffmpeg, но если процесс пропал — respawn)
            if not self._can_reconnect():
                self._stopped = True
                return None
            try:
                self._spawn()
            except VideoSourceError:
                self._stopped = True
                return None

        frame_bytes = self._w * self._h * 3
        raw = b""
        try:
            raw = self._proc.stdout.read(frame_bytes)
        except (OSError, ValueError):
            raw = b""

        rc = None
        if len(raw) < frame_bytes:
            # короткое/пустое чтение: либо ffmpeg умер, либо обрыв
            try:
                rc = self._proc.poll()
            except Exception:
                rc = None
            if len(raw) == 0 and rc == 0:
                # EOF файла через pipe: graceful close
                self._eof = True
                self._kill_proc()
                return None
            self._handle_bad_read()
            return None

        frame = np.frombuffer(raw, dtype=np.uint8).reshape(self._h, self._w, 3)
        self._fails = 0
        t_video = (self._index / self.fps) if self.fps > 0 else None
        out = Frame(image=frame, index=self._index, t_wall=time.monotonic(), t_video=t_video)
        self._index += 1
        return out

    def __del__(self) -> None:  # pragma: no cover — страховка от утечки процесса
        try:
            self.close()
        except Exception:
            pass
