"""Конфигурация visio-people-counter.

Схема 1:1 соответствует ``config.example.yaml`` (эталон всех ключей, типов и
дефолтов). Все координаты — нормализованные доли кадра 0..1; масштабирование
на фактическое w/h выполняется вызывающими модулями, а не здесь.

Использование::

    from visio_people_counter.config import Config
    cfg = Config.load("config.yaml")
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import yaml


class ConfigError(Exception):
    """Ошибка загрузки/валидации конфигурации.

    Сообщение содержит путь к проблемному ключу в формате ``блок.ключ``
    и объяснение: что ожидалось vs что получено.
    """


# ---------------------------------------------------------------------------
# Вспомогательные функции валидации
# ---------------------------------------------------------------------------

def _err(path: str, expected: str, got: Any) -> ConfigError:
    return ConfigError(f"{path}: ожидалось {expected}, получено {type(got).__name__}({got!r})")


def _require(cond: bool, path: str, expected: str, got: Any) -> None:
    if not cond:
        raise _err(path, expected, got)


def _as_dict(value: Any, path: str) -> dict:
    _require(isinstance(value, dict), path, "маппинг (блок конфигурации)", value)
    return value


def _check_unknown_keys(d: dict, allowed: set[str], path: str) -> None:
    unknown = sorted(set(d) - allowed)
    if unknown:
        raise ConfigError(
            f"{path}: неизвестные ключи {unknown}; разрешены: {sorted(allowed)}"
        )


def _get_str(d: dict, key: str, path: str, default: str) -> str:
    if key not in d:
        return default
    v = d[key]
    _require(isinstance(v, str), f"{path}.{key}", "str", v)
    return v


def _get_bool(d: dict, key: str, path: str, default: bool) -> bool:
    if key not in d:
        return default
    v = d[key]
    _require(isinstance(v, bool), f"{path}.{key}", "bool", v)
    return v


def _get_int(d: dict, key: str, path: str, default: int) -> int:
    if key not in d:
        return default
    v = d[key]
    # bool — подкласс int в Python; явно не принимаем.
    _require(isinstance(v, int) and not isinstance(v, bool), f"{path}.{key}", "int", v)
    return v


def _get_optional_int(d: dict, key: str, path: str) -> Optional[int]:
    """Целое >= 0 или null/отсутствует (→ None). bool не принимаем."""
    if key not in d or d[key] is None:
        return None
    v = d[key]
    # bool — подкласс int в Python; явно не принимаем.
    _require(isinstance(v, int) and not isinstance(v, bool), f"{path}.{key}", "int >= 0 или null", v)
    if v < 0:
        raise ConfigError(f"{path}.{key}: ожидалось целое >= 0 (номер кадра, 0-based), получено {v!r}")
    return v


def _get_float(d: dict, key: str, path: str, default: float) -> float:
    if key not in d:
        return default
    v = d[key]
    _require(
        isinstance(v, (int, float)) and not isinstance(v, bool),
        f"{path}.{key}", "число (int/float)", v,
    )
    return float(v)


def _get_enum(d: dict, key: str, path: str, default: str, allowed: set[str]) -> str:
    if key not in d:
        return default
    v = d[key]
    _require(isinstance(v, str), f"{path}.{key}", "str", v)
    _require(v in allowed, f"{path}.{key}", f"одно из {sorted(allowed)}", v)
    return v


def _get_pair(d: dict, key: str, path: str, default: tuple[float, float]) -> tuple[float, float]:
    """Нормализованная координата [x, y], каждая в диапазоне 0..1."""
    if key not in d:
        return default
    v = d[key]
    _require(
        isinstance(v, (list, tuple)) and len(v) == 2,
        f"{path}.{key}", "пара [x, y]", v,
    )
    for i, comp in enumerate(v):
        _require(
            isinstance(comp, (int, float)) and not isinstance(comp, bool),
            f"{path}.{key}[{i}]", "число", comp,
        )
        _require(0.0 <= float(comp) <= 1.0, f"{path}.{key}[{i}]", "доля кадра в диапазоне 0..1", comp)
    return (float(v[0]), float(v[1]))


def _get_pair_list(d: dict, key: str, path: str, default: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if key not in d:
        return copy.deepcopy(default)
    v = d[key]
    _require(isinstance(v, list), f"{path}.{key}", "список пар [x, y]", v)
    out: list[tuple[float, float]] = []
    for i, item in enumerate(v):
        p = f"{path}.{key}[{i}]"
        _require(
            isinstance(item, (list, tuple)) and len(item) == 2,
            p, "пара [x, y]", item,
        )
        for j, comp in enumerate(item):
            _require(
                isinstance(comp, (int, float)) and not isinstance(comp, bool),
                f"{p}[{j}]", "число", comp,
            )
            _require(0.0 <= float(comp) <= 1.0, f"{p}[{j}]", "доля кадра в диапазоне 0..1", comp)
        out.append((float(item[0]), float(item[1])))
    return out


def _get_size_points(d: dict, key: str, path: str,
                     default: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    """Контрольные точки size_profile: тройки ``[x_доля, y_доля, h_доля]``.

    x, y — в диапазоне 0..1 (доли кадра), h — в 0..1 и строго > 0 (доля высоты
    кадра). Назад-совместимость: старые пары ``[x, h]`` мигрируются в
    ``[x, 0.5, h]`` с warning'ом в stdout; при сохранении пишутся ТОЛЬКО тройки.
    """
    if key not in d:
        return copy.deepcopy(default)
    v = d[key]
    _require(isinstance(v, list), f"{path}.{key}", "список точек [x, y, h]", v)
    out: list[tuple[float, float, float]] = []
    legacy = False
    for i, item in enumerate(v):
        p = f"{path}.{key}[{i}]"
        _require(
            isinstance(item, (list, tuple)) and len(item) in (2, 3),
            p, "тройка [x, y, h] (или старую пару [x, h])", item,
        )
        if len(item) == 2:
            # legacy-формат: [x, h] → [x, 0.5, h]; y берётся из миграции
            raws = [("x", item[0]), ("h", item[1])]
            x, _y, h = float(item[0]), 0.5, float(item[1])
            legacy = True
        else:
            raws = [("x", item[0]), ("y", item[1]), ("h", item[2])]
            x, _y, h = (float(item[0]), float(item[1]), float(item[2]))
        for name, raw in raws:
            _require(
                isinstance(raw, (int, float)) and not isinstance(raw, bool),
                f"{p}.{name}", "число", raw,
            )
        _require(0.0 <= x <= 1.0, f"{p}.x", "доля кадра в диапазоне 0..1", item[0])
        if len(item) == 3:
            _require(0.0 <= _y <= 1.0, f"{p}.y", "доля кадра в диапазоне 0..1", item[1])
        if not (h > 0.0):
            raise ConfigError(f"{p}.h: ожидалось доля высоты человека > 0, получено {item[-1]!r}")
        _require(h <= 1.0, f"{p}.h", "доля кадра в диапазоне (0, 1]", item[-1])
        out.append((x, _y, h))
    if legacy:
        print("size_profile: старые точки [x, h] перенесены как y=0.5 — "
              "при возможности перекалибруйте")
    return out


def _get_roi(d: dict, key: str, path: str) -> Optional[list[float]]:
    """ROI — один прямоугольник ``[x, y, w, h]`` в долях ПОЛНОГО кадра.

    Отсутствует/null → None (весь кадр). Иначе — список ровно из 4 чисел,
    каждое в (0..1], причём ``x + w <= 1`` и ``y + h <= 1``; иначе ConfigError.
    Все координаты настроек (линии/зоны/size-точки) и событий считаются
    относительно ROI — кроп применяется на уровне источника.
    """
    if key not in d or d[key] is None:
        return None
    v = d[key]
    _require(
        isinstance(v, (list, tuple)) and len(v) == 4,
        f"{path}.{key}", "список из 4 чисел [x, y, w, h] (доли полного кадра)", v,
    )
    vals: list[float] = []
    for i, comp in enumerate(v):
        _require(
            isinstance(comp, (int, float)) and not isinstance(comp, bool),
            f"{path}.{key}[{i}]", "число", comp,
        )
        c = float(comp)
        if not (0.0 < c <= 1.0):
            raise ConfigError(
                f"{path}.{key}[{i}]: ожидалось число в (0..1] (доля полного кадра), получено {comp!r}"
            )
        vals.append(c)
    x, y, w, h = vals
    if x + w > 1.0:
        raise ConfigError(
            f"{path}.{key}: x + w = {x + w:.4f} > 1 — ROI выходит за правый край кадра"
        )
    if y + h > 1.0:
        raise ConfigError(
            f"{path}.{key}: y + h = {y + h:.4f} > 1 — ROI выходит за нижний край кадра"
        )
    return vals


def _get_range(d: dict, key: str, path: str, default: tuple[float, float], min_bound: float = 0.0) -> tuple[float, float]:
    """Диапазон [min, max] чисел (например aspect_ratio_range); обе части >= min_bound."""
    if key not in d:
        return default
    v = d[key]
    _require(
        isinstance(v, (list, tuple)) and len(v) == 2,
        f"{path}.{key}", "пара [min, max]", v,
    )
    nums: list[float] = []
    for i, comp in enumerate(v):
        _require(
            isinstance(comp, (int, float)) and not isinstance(comp, bool),
            f"{path}.{key}[{i}]", "число", comp,
        )
        if float(comp) < min_bound:
            raise ConfigError(f"{path}.{key}[{i}]: ожидалось число >= {min_bound}, получено {comp!r}")
        nums.append(float(comp))
    _require(nums[0] <= nums[1], f"{path}.{key}", "min <= max", v)
    return (nums[0], nums[1])


def _get_morph_kernel(d: dict, key: str, path: str, default: tuple[int, int]) -> tuple[int, int]:
    """Ядро морфологии: [w, h], целые нечётные > 0."""
    if key not in d:
        return default
    v = d[key]
    _require(
        isinstance(v, (list, tuple)) and len(v) == 2,
        f"{path}.{key}", "пара [w, h]", v,
    )
    for i, comp in enumerate(v):
        _require(
            isinstance(comp, int) and not isinstance(comp, bool),
            f"{path}.{key}[{i}]", "int", comp,
        )
        if comp <= 0 or comp % 2 == 0:
            raise ConfigError(f"{path}.{key}[{i}]: ожидалось нечётное целое > 0 (размер ядра), получено {comp!r}")
    return (v[0], v[1])


# ---------------------------------------------------------------------------
# Dataclass'ы (дефолты = config.example.yaml)
# ---------------------------------------------------------------------------

@dataclass
class HlsConfig:
    """Настройки HLS-источника: watchdog переподключения."""
    reconnect_attempts: int = 0          # 0 = бесконечно, -1 = не переподключаться
    reconnect_backoff_s: float = 5.0     # пауза между попытками, сек
    bad_read_threshold: int = 10         # подряд неудачных read() -> обрыв

    @classmethod
    def from_dict(cls, d: Any) -> "HlsConfig":
        d = _as_dict(d, "video.hls")
        _check_unknown_keys(d, {"reconnect_attempts", "reconnect_backoff_s", "bad_read_threshold"}, "video.hls")
        return cls(
            reconnect_attempts=_get_int(d, "reconnect_attempts", "video.hls", 0),
            reconnect_backoff_s=_get_float(d, "reconnect_backoff_s", "video.hls", 5.0),
            bad_read_threshold=_get_int(d, "bad_read_threshold", "video.hls", 10),
        )


@dataclass
class VideoConfig:
    """Источник видео: файл или HLS-поток."""
    type: str = "file"                   # "file" | "hls"
    path: str = ""                       # путь к файлу ИЛИ URL потока
    hls: HlsConfig = field(default_factory=HlsConfig)
    loop_file: bool = False              # зациклить файл

    @classmethod
    def from_dict(cls, d: Any) -> "VideoConfig":
        d = _as_dict(d, "video")
        _check_unknown_keys(d, {"type", "path", "hls", "loop_file"}, "video")
        return cls(
            type=_get_enum(d, "type", "video", "file", {"file", "hls"}),
            path=_get_str(d, "path", "video", ""),
            hls=HlsConfig.from_dict(d.get("hls", {})),
            loop_file=_get_bool(d, "loop_file", "video", False),
        )


@dataclass
class ProcessingConfig:
    """Параметры обработки кадров.

    ``frame_start``/``frame_end`` — интервал КАДРОВ (0-based номера, совпадают
    с индексом кадра источника/GUI), в котором работают трекер и счётчики;
    детектор при этом обрабатывает ВСЕ кадры (обучение фона MOG2). Семантика:
    обе null — подсчёт на всём видео; только ``frame_start`` — с него до конца;
    только ``frame_end`` — от начала до него включительно; обе —
    ``frame_start <= index <= frame_end`` (обе границы включительно).
    """
    effective_fps: float = 0             # 0 = нативный fps источника
    max_width: int = 0                   # даунскейл до ширины; 0 = как в источнике
    frame_start: Optional[int] = None    # первый кадр подсчёта (0-based, включительно); null = без границы
    frame_end: Optional[int] = None      # последний кадр подсчёта (0-based, включительно); null = до конца
    #: время в секундах (приоритет над frame_start/frame_end): если time_start задан,
    #: frame_start рассчитывается как int(time_start * fps) после открытия источника.
    time_start: Optional[float] = None   # начало обработки (сек); null = без границы
    time_end: Optional[float] = None     # окончание обработки (сек, включительно); null = до конца
    #: ROI — один глобальный прямоугольник [x, y, w, h] в долях ПОЛНОГО кадра;
    #: None/отсутствует = весь кадр. Кроп применяется на уровне источника
    #: (ffmpeg -vf crop / NumPy-срез), поэтому ВСЕ координаты настроек и событий
    #: отсчитываются от ROI.
    roi: Optional[list[float]] = None

    @classmethod
    def from_dict(cls, d: Any) -> "ProcessingConfig":
        d = _as_dict(d, "processing")
        _check_unknown_keys(
            d, {"effective_fps", "max_width", "frame_start", "frame_end",
                "time_start", "time_end", "roi"}, "processing"
        )
        eff = _get_float(d, "effective_fps", "processing", 0.0)
        if eff < 0:
            raise ConfigError(f"processing.effective_fps: ожидалось >= 0, получено {eff!r}")
        mw = _get_int(d, "max_width", "processing", 0)
        if mw < 0:
            raise ConfigError(f"processing.max_width: ожидалось >= 0 (px), получено {mw!r}")
        fs = _get_optional_int(d, "frame_start", "processing")
        fe = _get_optional_int(d, "frame_end", "processing")
        if fs is not None and fe is not None and fs > fe:
            raise ConfigError(
                f"processing.frame_start={fs} > processing.frame_end={fe}: "
                f"ожидалось frame_start <= frame_end (номера кадров 0-based, обе границы включительно)"
            )
        ts = d.get("time_start")
        if ts is not None:
            ts = float(ts)
            if ts < 0:
                raise ConfigError(f"processing.time_start: ожидалось >= 0, получено {ts!r}")
        te = d.get("time_end")
        if te is not None:
            te = float(te)
            if te < 0:
                raise ConfigError(f"processing.time_end: ожидалось >= 0, получено {te!r}")
        roi = _get_roi(d, "roi", "processing")
        return cls(effective_fps=eff, max_width=mw, frame_start=fs, frame_end=fe,
                   time_start=ts, time_end=te, roi=roi)


def describe_frame_range(frame_start: Optional[int], frame_end: Optional[int]) -> str:
    """Человекочитаемый интервал кадров для финальной сводки и markdown-отчёта.

    * обе null → ``весь``;
    * только start → ``100–конец (подсчёт)``;
    * только end → ``начало–250 (подсчёт)``;
    * оба → ``100–250 (подсчёт)``.

    Номера 0-based; ``frame_end`` — включительно.
    """
    if frame_start is None and frame_end is None:
        return "весь"
    s = "начало" if frame_start is None else str(frame_start)
    e = "конец" if frame_end is None else str(frame_end)
    return f"{s}–{e} (подсчёт)"


def describe_roi(roi: Optional[list[float]]) -> str:
    """Человекочитаемая строка ROI для финальной сводки и markdown-отчёта.

    * ``None`` → ``нет``;
    * иначе → ``x–x+w × y–y+h`` (доли полного кадра, 4 знака),
      например ``0.2–0.8 × 0.3–0.7``.
    """
    if roi is None:
        return "нет"
    x, y, w, h = roi

    def f(v: float) -> str:
        s = f"{round(float(v), 4):g}"
        return s

    return f"{f(x)}–{f(round(x + w, 4))} × {f(y)}–{f(round(y + h, 4))}"


@dataclass
class MotionConfig:
    """Выделение движения (background subtraction)."""
    method: str = "mog2"                 # "mog2" | "knn"
    history: int = 500
    var_threshold: int = 32              # (mog2)
    dist2_threshold: float = 4.0         # (knn)
    detect_shadows: bool = True          # (mog2)
    shadow_threshold: int = 200          # (mog2, при detect_shadows=True)
    morph_open: tuple[int, int] = (3, 3)
    morph_close: tuple[int, int] = (9, 15)

    @classmethod
    def from_dict(cls, d: Any) -> "MotionConfig":
        d = _as_dict(d, "motion")
        _check_unknown_keys(
            d,
            {"method", "history", "var_threshold", "dist2_threshold",
             "detect_shadows", "shadow_threshold", "morph_open", "morph_close"},
            "motion",
        )
        return cls(
            method=_get_enum(d, "method", "motion", "mog2", {"mog2", "knn"}),
            history=_get_int(d, "history", "motion", 500),
            var_threshold=_get_int(d, "var_threshold", "motion", 32),
            dist2_threshold=_get_float(d, "dist2_threshold", "motion", 4.0),
            detect_shadows=_get_bool(d, "detect_shadows", "motion", True),
            shadow_threshold=_get_int(d, "shadow_threshold", "motion", 200),
            morph_open=_get_morph_kernel(d, "morph_open", "motion", (3, 3)),
            morph_close=_get_morph_kernel(d, "morph_close", "motion", (9, 15)),
        )


@dataclass
class ObjectsConfig:
    """Фильтры обнаруженных движущихся объектов."""
    min_area_fraction: float = 0.0005    # доля площади кадра
    max_area_fraction: float = 0.25
    min_bbox_side_px: int = 8
    aspect_ratio_range: tuple[float, float] = (0.2, 2.5)
    min_fill: float = 0.25               # area / (w*h) bbox'а
    min_lifetime_frames: int = 3

    @classmethod
    def from_dict(cls, d: Any) -> "ObjectsConfig":
        d = _as_dict(d, "objects")
        _check_unknown_keys(
            d,
            {"min_area_fraction", "max_area_fraction", "min_bbox_side_px",
             "aspect_ratio_range", "min_fill", "min_lifetime_frames"},
            "objects",
        )
        return cls(
            min_area_fraction=_get_float(d, "min_area_fraction", "objects", 0.0005),
            max_area_fraction=_get_float(d, "max_area_fraction", "objects", 0.25),
            min_bbox_side_px=_get_int(d, "min_bbox_side_px", "objects", 8),
            aspect_ratio_range=_get_range(d, "aspect_ratio_range", "objects", (0.2, 2.5)),
            min_fill=_get_float(d, "min_fill", "objects", 0.25),
            min_lifetime_frames=_get_int(d, "min_lifetime_frames", "objects", 3),
        )


@dataclass
class SizeProfileConfig:
    """Адаптация порогов площади к размеру объекта в точке кадра (опционально).

    ``control_points`` — тройки ``[x_доля, y_доля, h_доля]``: точка кадра (x, y)
    и ожидаемая высота человека в ней (доля высоты кадра). Старые пары ``[x, h]``
    при загрузке мигрируются в ``[x, 0.5, h]`` (см. :func:`_get_size_points`).
    """
    enabled: bool = False
    control_points: list[tuple[float, float, float]] = field(default_factory=list)  # [x_frac, y_frac, person_height_fraction]
    k_min: float = 0.2                   # min_area = k_min * h_px^2
    k_max: float = 4.0                   # max_area = k_max * h_px^2

    @classmethod
    def from_dict(cls, d: Any) -> "SizeProfileConfig":
        d = _as_dict(d, "size_profile")
        _check_unknown_keys(
            d, {"enabled", "control_points", "k_min", "k_max"}, "size_profile"
        )
        return cls(
            enabled=_get_bool(d, "enabled", "size_profile", False),
            control_points=_get_size_points(d, "control_points", "size_profile", []),
            k_min=_get_float(d, "k_min", "size_profile", 0.2),
            k_max=_get_float(d, "k_max", "size_profile", 4.0),
        )


@dataclass
class TrackerConfig:
    """Трекер (пакет `trackers`): sort / bytetrack / botsort."""
    type: str = "sort"
    lost_track_buffer: int = 60                     # кадров помнить пропавший объект
    minimum_consecutive_frames: int = 2
    minimum_iou_threshold: float = 0.3

    @classmethod
    def from_dict(cls, d: Any) -> "TrackerConfig":
        d = _as_dict(d, "tracker")
        _check_unknown_keys(
            d,
            {"type", "lost_track_buffer", "minimum_consecutive_frames",
             "minimum_iou_threshold"},
            "tracker",
        )
        return cls(
            type=_get_enum(d, "type", "tracker", "sort", {"sort", "bytetrack", "botsort"}),
            lost_track_buffer=_get_int(d, "lost_track_buffer", "tracker", 60),
            minimum_consecutive_frames=_get_int(d, "minimum_consecutive_frames", "tracker", 2),
            minimum_iou_threshold=_get_float(d, "minimum_iou_threshold", "tracker", 0.3),
        )


@dataclass
class LineCounterConfig:
    """Счётчик-линия: точки A→B задают направление «in»."""
    id: str = ""
    type: str = "line"
    a: tuple[float, float] = (0.0, 0.0)             # нормализованные (x, y)
    b: tuple[float, float] = (1.0, 1.0)
    count_mode: str = "both"                        # "both" | "total"
    cooldown_s: float = 2.0                         # антидубль по track_id
    buffer_width_scale: float = 0.75                # ширина буфера вокруг линии
    min_global_gap_s: float = 0.3                   # мин. интервал между любыми событиями

    @classmethod
    def from_dict(cls, d: Any, cpath: str = "counters[i]") -> "LineCounterConfig":
        d = _as_dict(d, cpath)
        return cls(
            id=_get_str(d, "id", cpath, ""),
            a=_get_pair(d, "a", cpath, (0.0, 0.0)),
            b=_get_pair(d, "b", cpath, (1.0, 1.0)),
            count_mode=_get_enum(d, "count_mode", cpath, "both", {"both", "total"}),
            cooldown_s=_get_float(d, "cooldown_s", cpath, 2.0),
            buffer_width_scale=_get_float(d, "buffer_width_scale", cpath, 0.75),
            min_global_gap_s=_get_float(d, "min_global_gap_s", cpath, 0.3),
        )


@dataclass
class ZoneCounterConfig:
    """Счётчик-зона: полигон, направление по порядку обхода."""
    id: str = ""
    type: str = "zone"
    polygon: list[tuple[float, float]] = field(default_factory=list)
    count_mode: str = "total"                       # "both" | "total"
    cooldown_s: float = 3.0
    min_global_gap_s: float = 0.3

    @classmethod
    def from_dict(cls, d: Any, cpath: str = "counters[i]") -> "ZoneCounterConfig":
        d = _as_dict(d, cpath)
        poly = _get_pair_list(d, "polygon", cpath, [])
        if len(poly) < 3:
            raise ConfigError(f"{cpath}.polygon (id={d.get('id', '?')}): ожидалось >= 3 точек полигона, получено {len(poly)}")
        return cls(
            id=_get_str(d, "id", cpath, ""),
            polygon=poly,
            count_mode=_get_enum(d, "count_mode", cpath, "total", {"both", "total"}),
            cooldown_s=_get_float(d, "cooldown_s", cpath, 3.0),
            min_global_gap_s=_get_float(d, "min_global_gap_s", cpath, 0.3),
        )


CounterConfig = LineCounterConfig | ZoneCounterConfig


@dataclass
class OutputConfig:
    """Где и как писать результаты."""
    events_jsonl: str = ""                          # "" = не писать
    report_path: str = ""                           # "" = автопуть <видео>.report.md (только для файла)
    summary_interval_s: float = 30                  # 0 = только финал
    final_summary: bool = True

    @classmethod
    def from_dict(cls, d: Any) -> "OutputConfig":
        d = _as_dict(d, "output")
        _check_unknown_keys(
            d, {"events_jsonl", "report_path", "summary_interval_s", "final_summary"}, "output"
        )
        return cls(
            events_jsonl=_get_str(d, "events_jsonl", "output", ""),
            report_path=_get_str(d, "report_path", "output", ""),
            summary_interval_s=_get_float(d, "summary_interval_s", "output", 30.0),
            final_summary=_get_bool(d, "final_summary", "output", True),
        )


@dataclass
class DebugConfig:
    """Отладочная визуализация."""
    show_bboxes: bool = True
    show_all_blobs: bool = False
    show_mask: bool = False
    show_counters: bool = True
    save_debug_frames_dir: str = ""
    debug_frame_step: int = 10

    @classmethod
    def from_dict(cls, d: Any) -> "DebugConfig":
        d = _as_dict(d, "debug")
        _check_unknown_keys(
            d,
            {"show_bboxes", "show_all_blobs", "show_mask", "show_counters",
             "save_debug_frames_dir", "debug_frame_step"},
            "debug",
        )
        return cls(
            show_bboxes=_get_bool(d, "show_bboxes", "debug", True),
            show_all_blobs=_get_bool(d, "show_all_blobs", "debug", False),
            show_mask=_get_bool(d, "show_mask", "debug", False),
            show_counters=_get_bool(d, "show_counters", "debug", True),
            save_debug_frames_dir=_get_str(d, "save_debug_frames_dir", "debug", ""),
            debug_frame_step=_get_int(d, "debug_frame_step", "debug", 10),
        )


# ---------------------------------------------------------------------------
# Корневой Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Корневая конфигурация (все блоки config.example.yaml)."""
    video: VideoConfig = field(default_factory=VideoConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    objects: ObjectsConfig = field(default_factory=ObjectsConfig)
    size_profile: SizeProfileConfig = field(default_factory=SizeProfileConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    counters: list[CounterConfig] = field(default_factory=list)
    output: OutputConfig = field(default_factory=OutputConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    # --- загрузка -----------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        """Прочитать YAML-конфиг и валидировать его.

        :param path: путь к файлу конфигурации.
        :raises ConfigError: файл не найден / битый YAML / неверные значения.
        """
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"конфиг {p}: файл не найден")
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            raise ConfigError(f"{p}: битый YAML: {e}") from e
        return cls.from_dict(raw if isinstance(raw, dict) else {}, str(p))

    @classmethod
    def default(cls) -> "Config":
        """Конфиг по умолчанию (значения как в config.example.yaml).

        Используется calibrate при отсутствии файла: оператор калибрует линию/зону,
        а файл создаётся при сохранении. Включает один line-счётчик ``main_line`` —
        без счётчика калибровать нечего.
        """
        return cls(
            counters=[LineCounterConfig(id="main_line", a=(0.25, 0.35), b=(0.75, 0.85))],
        )

    @classmethod
    def from_dict(cls, d: Any, source: str = "<dict>") -> "Config":
        """Собрать Config из словаря с полной валидацией типов/диапазонов."""
        _check_unknown_keys(
            d,
            {"video", "processing", "motion", "objects", "size_profile",
             "tracker", "counters", "output", "debug"},
            source,
        )

        counters: list[CounterConfig] = []
        raw_counters = d.get("counters", [])
        _require(isinstance(raw_counters, list), "counters", "список счётчиков", raw_counters)
        seen_ids: set[str] = set()
        for i, item in enumerate(raw_counters):
            cpath = f"counters[{i}]"
            _as_dict(item, cpath)
            ctype = _get_enum(item, "type", cpath, "", {"line", "zone"})
            if ctype == "line":
                _check_unknown_keys(
                    item,
                    {"id", "type", "a", "b", "count_mode", "cooldown_s",
                     "buffer_width_scale", "min_global_gap_s"},
                    cpath,
                )
                obj = LineCounterConfig.from_dict(item, cpath)
            else:
                _check_unknown_keys(
                    item,
                    {"id", "type", "polygon", "count_mode", "cooldown_s",
                     "min_global_gap_s"},
                    cpath,
                )
                obj = ZoneCounterConfig.from_dict(item, cpath)
            if not obj.id:
                raise ConfigError(f"{cpath}.id: ожидалось непустой str (уникальный id счётчика), получено {obj.id!r}")
            if obj.id in seen_ids:
                raise ConfigError(f"{cpath}.id: дублирующийся id счётчика {obj.id!r}")
            seen_ids.add(obj.id)
            counters.append(obj)

        return cls(
            video=VideoConfig.from_dict(d.get("video", {})),
            processing=ProcessingConfig.from_dict(d.get("processing", {})),
            motion=MotionConfig.from_dict(d.get("motion", {})),
            objects=ObjectsConfig.from_dict(d.get("objects", {})),
            size_profile=SizeProfileConfig.from_dict(d.get("size_profile", {})),
            tracker=TrackerConfig.from_dict(d.get("tracker", {})),
            counters=counters,
            output=OutputConfig.from_dict(d.get("output", {})),
            debug=DebugConfig.from_dict(d.get("debug", {})),
        )

    # --- сохранение ---------------------------------------------------------

    def to_dict(self) -> dict:
        """Обратная сериализация в plain-данные (dataclass → dict)."""
        return asdict(self)

    @staticmethod
    def save(cfg: "Config", path: str | Path) -> None:
        """Сохранить конфиг обратно в YAML (без комментариев)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True, default_flow_style=False),
            encoding="utf-8",
        )
