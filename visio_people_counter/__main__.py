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
            player = GuiPlayer(pipe, speed=args.speed, initial_scale=args.scale)
            return player.run()
        return pipe.run()
    except (VideoSourceError, ConfigError) as e:
        print(f"count: ошибка: {e}", file=sys.stderr)
        return 1


def _cache_frames_value(s: str) -> int:
    """CLI-тип --cache-frames: целое >= 1 (размер кэша кадров calibrate)."""
    try:
        v = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--cache-frames: ожидалось целое, получено {s!r}")
    if v < 1:
        raise argparse.ArgumentTypeError("--cache-frames: ожидалось >= 1")
    return v


def cmd_calibrate(args: argparse.Namespace) -> int:
    """GUI-калибровка линии/зоны/size-точек.

    Без явного --config результат сохраняется рядом с видео (<имя_видео>.config.yaml);
    с явным --config — строго в этот файл (он же используется как входной, если есть).
    """
    from .calibrate import run_calibration
    return run_calibration(args.config or "config.yaml", video=args.video,
                           counter_id=args.counter_id, save_to=args.config,
                           cache_frames=args.cache_frames,
                           initial_scale=args.scale)


def _scale_value(s: str) -> float:
    """--scale: начальный масштаб ОТОБРАЖЕНИЯ окна (любое значение 0.05..8)."""
    v = float(s)
    if not (0.05 <= v <= 8.0):
        raise argparse.ArgumentTypeError(f"--scale: ожидалось 0.05..8, получено {s!r}")
    return v


_FMT = argparse.RawDescriptionHelpFormatter


_MAIN_EPILOG = """команды:
  probe      сводка по входу: разрешение/fps/длительность/кодек (+ что файл открывается в OpenCV)
  count      подсчёт трафика; headless (максимальная скорость) по умолчанию, --gui — с окном
  calibrate  GUI-калибровка линий/зон/size-точек «меркой роста» → конфиг

справка по команде:  visio_people_counter <команда> --help
пример запуска:       visio_people_counter count --config videos/demo.config.yaml --gui"""

_PROBE_EPILOG = """примеры:
  visio_people_counter probe --video videos/demo.mp4
  visio_people_counter probe --video https://cdn.example.com/live/stream.m3u8"""

_COUNT_EPILOG = """примеры значений:
  --config   путь к YAML: config.yaml, videos/demo.config.yaml (обязателен)
  --video    путь или URL: videos/1.mp4, https://…/stream.m3u8 (переопределяет video.path)
  --scale    число 0.05..8 (только --gui): --scale 0.5 | --scale 2 ; пресеты в окне — `,` / `.`
  --speed    число 0.25..8 (только --gui): --speed 0.5 (медленнее), --speed 4 (быстрее)

примеры:
  visio_people_counter count --config videos/demo.config.yaml
  visio_people_counter count --config cfg.yaml --video /tmp/x.mp4 --bench
  visio_people_counter count --config cfg.yaml --gui --speed 1.5 --scale 0.75"""

_CAL_EPILOG = """примеры значений:
  --video          путь к файлу или URL (для seek по времени нужен файл)
  --counter-id     любой строковый id: line_1, zone_2, main_line (по умолчанию main_line)
  --scale          число 0.05..8: --scale 0.5 | --scale 2 ; пресеты в окне — `,` / `.`
  --cache-frames   целое >= 1: --cache-frames 50 | 300 (по умолчанию 100; ~6 МБ/кадр при 1080p)

примеры:
  visio_people_counter calibrate --video videos/demo.mp4
  visio_people_counter calibrate --video /tmp/x.mp4 --config my.yaml --counter-id zone_2 \\
      --scale 0.75 --cache-frames 300"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visio_people_counter",
        description="Подсчёт трафика людей по видеофайлу или HLS-потоку.",
        formatter_class=_FMT, epilog=_MAIN_EPILOG,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    p_probe = sub.add_parser(
        "probe", formatter_class=_FMT, epilog=_PROBE_EPILOG,
        help="ffprobe-сводка по входу (разрешение/fps/длительность/кодек)")
    p_probe.add_argument("--video", required=True, metavar="PATH|URL",
                         help="путь к файлу ИЛИ URL потока (.m3u8), напр. videos/demo.mp4")
    p_probe.set_defaults(func=cmd_probe)

    p_count = sub.add_parser(
        "count", formatter_class=_FMT, epilog=_COUNT_EPILOG,
        help="подсчёт: headless по умолчанию; --gui [--speed] — GUI-режим с окном")
    p_count.add_argument("--config", default="config.yaml", metavar="PATH",
                         help="путь к config.yaml (по умолчанию ./config.yaml)")
    p_count.add_argument("--video", default=None, metavar="PATH|URL",
                         help="переопределить video.path из конфига; файл или HLS/URL")
    p_count.add_argument("--bench", action="store_true",
                         help="замерять время этапов (detect/track/count) и печатать p50/p95 в финале")
    p_count.add_argument("--gui", action="store_true",
                         help="GUI-режим: окно с overlay по cfg.debug (headless — по умолчанию)")
    p_count.add_argument("--scale", type=_scale_value, default=1.0, metavar="FLOAT",
                         help="начальный масштаб отображения окна (--gui), число 0.05..8; напр. --scale 0.75")
    p_count.add_argument("--speed", type=_speed_value, default=1.0, metavar="FLOAT",
                         help="скорость воспроизведения в GUI, число 0.25..8 (по умолчанию 1.0)")
    p_count.set_defaults(func=cmd_count)

    p_cal = sub.add_parser(
        "calibrate", formatter_class=_FMT, epilog=_CAL_EPILOG,
        help="GUI-калибровка линии/зоны/size-точек → config.yaml")
    p_cal.add_argument("--config", default=None, metavar="PATH",
                       help="входной конфиг; если опция задана ЯВНО — результат [a] "
                            "сохраняется строго в этот файл. Без опции результат пишется "
                            "рядом с видео: <имя_видео>.config.yaml. Файла входного конфига "
                            "может не быть (калибровка по дефолтам); для count/probe конфиг обязателен")
    p_cal.add_argument("--video", default=None, metavar="PATH|URL",
                       help="видео для калибровки (переопределяет video.path)")
    p_cal.add_argument("--counter-id", default="main_line", metavar="ID",
                       help="id счётчика, который рисуем/добавляем (по умолчанию main_line)")
    p_cal.add_argument("--scale", type=_scale_value, default=1.0, metavar="FLOAT",
                       help="начальный масштаб отображения окна, число 0.05..8; напр. --scale 0.75")
    p_cal.add_argument("--cache-frames", type=_cache_frames_value, default=100,
                       metavar="N", help="сколько кадров держать в кэше листа [n/p] и "
                                         "загружать после seek ([t]) (по умолчанию 100, >= 1)")
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
