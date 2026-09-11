"""CLI visio-people-counter: ``python -m visio_people_counter <probe|count|calibrate>``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import Config, ConfigError
from .pipeline import Pipeline
from .video_source import VideoSourceError, ffprobe_info, is_url


def _kind_label(path_or_url: str, format_name: str) -> str:
    """Тип источника: hls (m3u8/URL-поток) или file."""
    p = path_or_url.lower()
    if ".m3u8" in p or "hls" in (format_name or "").lower():
        return "hls"
    if is_url(p):
        return "url-поток (обработка как hls)"
    return "file"


def cmd_probe(args: argparse.Namespace) -> int:
    """ffprobe-сводка по входу + для файла — проверка открытия cv2.VideoCapture."""
    video = args.video
    try:
        info = ffprobe_info(video)
    except VideoSourceError as e:
        print(f"probe: ОШИБКА: {e}", file=sys.stderr)
        return 1

    duration = info["duration"]
    fps = info["fps"]
    print("=== visio-people-counter probe ===")
    print(f"вход          : {video}")
    print(f"тип           : {_kind_label(video, info['format_name'])}")
    print(f"контейнер     : {info['format_name'] or 'н/д'}")
    print(f"разрешение    : {info['width']}x{info['height']}")
    print(f"fps           : {('%.3f' % fps) if fps else 'н/д (HLS без стабильного таймкода?)'}")
    print(f"длительность  : {('%.1f c' % duration) if duration is not None else 'н/д (стрим)'}")
    print(f"кодек         : {info['codec_name'] or 'н/д'}")

    rc = 0
    if not is_url(video):
        # для файла дополнительно проверяем, что cv2.VideoCapture открывает его
        import cv2
        cap = cv2.VideoCapture(video)
        if cap.isOpened():
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"cv2.VideoCapture: OK (открывается; кадров по метаданным: {n_frames})")
            ok_ret, _ = cap.read()
            if not ok_ret:
                print("cv2.VideoCapture: ВНИМАНИЕ: первый кадр прочитать не удалось", file=sys.stderr)
                rc = 1
        else:
            print(f"cv2.VideoCapture: ОШИБКА: файл {video!r} не открывается", file=sys.stderr)
            rc = 1
        cap.release()
    return rc


def _speed_value(s: str) -> float:
    """CLI-тип --speed: float в диапазоне 0.25..8."""
    try:
        v = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--speed: ожидалось число, получено {s!r}")
    if not (0.25 <= v <= 8.0):
        raise argparse.ArgumentTypeError("--speed: ожидалось 0.25..8, получено " + repr(v))
    return v


def cmd_count(args: argparse.Namespace) -> int:
    """Подсчёт: headless по умолчанию; --gui [--speed] — GUI-режим (задача 16)."""
    try:
        cfg = Config.load(args.config)
    except ConfigError as e:
        print(f"count: ошибка конфигурации: {e}", file=sys.stderr)
        return 1
    if args.video:
        cfg.video.path = args.video  # переопределение для разовых прогонов

    path = cfg.video.path
    if not path:
        print("count: не задан источник видео — укажите video.path в конфиге или --video", file=sys.stderr)
        return 1
    if cfg.video.type == "file" and not is_url(path) and not Path(path).expanduser().is_file():
        print(f"count: видеофайл не найден: {path!r}", file=sys.stderr)
        return 1

    try:
        pipe = Pipeline(cfg, bench=args.bench)
        if args.gui:
            from .gui import GuiPlayer
            if not GuiPlayer.available():
                print(f"count --gui: ОШИБКА: {GuiPlayer.unavailable_reason()}", file=sys.stderr)
                return 1
            player = GuiPlayer(pipe, speed=args.speed)
            return player.run()
        return pipe.run()
    except (VideoSourceError, ConfigError) as e:
        print(f"count: ошибка: {e}", file=sys.stderr)
        return 1


def cmd_calibrate(args: argparse.Namespace) -> int:
    """GUI-калибровка линии/зоны/size-точек → запись в config.yaml (задача 16)."""
    from .calibrate import run_calibration
    return run_calibration(args.config, video=args.video, counter_id=args.counter_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visio_people_counter",
        description="Подсчёт трафика людей по видеофайлу или HLS-потоку.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    p_probe = sub.add_parser("probe", help="ffprobe-сводка по входу (разрешение/fps/длительность/кодек)")
    p_probe.add_argument("--video", required=True, metavar="PATH|URL",
                         help="путь к файлу ИЛИ URL потока (.m3u8)")
    p_probe.set_defaults(func=cmd_probe)

    p_count = sub.add_parser(
        "count",
        help="подсчёт: headless по умолчанию; --gui [--speed] — GUI-режим с окном")
    p_count.add_argument("--config", default="config.yaml", help="путь к config.yaml")
    p_count.add_argument("--video", default=None, metavar="PATH|URL",
                         help="переопределить video.path из конфига")
    p_count.add_argument("--bench", action="store_true",
                         help="замерять время этапов (detect/track/count) и печатать p50/p95 в финале")
    p_count.add_argument("--gui", action="store_true",
                         help="GUI-режим: окно с overlay по cfg.debug (headless — по умолчанию)")
    p_count.add_argument("--speed", type=_speed_value, default=1.0, metavar="FLOAT",
                         help="скорость воспроизведения в GUI 0.25..8 (по умолчанию 1.0)")
    p_count.set_defaults(func=cmd_count)

    p_cal = sub.add_parser("calibrate", help="GUI-калибровка линии/зоны/size-точек → config.yaml")
    p_cal.add_argument("--config", default="config.yaml", help="путь к config.yaml")
    p_cal.add_argument("--video", default=None, metavar="PATH|URL",
                       help="видео для калибровки (переопределяет video.path)")
    p_cal.add_argument("--counter-id", default="main_line", metavar="ID",
                       help="id счётчика, который рисуем/добавляем (по умолчанию main_line)")
    p_cal.set_defaults(func=cmd_calibrate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
