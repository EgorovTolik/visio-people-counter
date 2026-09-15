"""Адаптивные пороги площади по положению объекта в кадре (2D, задача 05).

``SizeProfile`` строится из блока ``size_profile`` конфига и размеров текущего
кадра:

* контрольные точки — тройки ``[x_доля, y_доля, h_доля]`` (точка кадра +
  ожидаемая высота человека в ней); старые пары ``[x, h]`` мигрируются при
  загрузке конфига (см. :mod:`visio_people_counter.config`);
* по точкам подгоняется поверхность ``h(x, y)`` (наименьшие квадраты, numpy):
  n >= 4 — ``a + b·x + c·y + d·x·y``; n == 3 — плоскость ``a + b·x + c·y``;
  n == 2 — линейная интерполяция вдоль отрезка между двумя точками;
  n == 1 — константа (см. :func:`fit_height_surface`);
* ``person_height_px(x, y)`` — ожидаемая высота человека в px в точке кадра
  (координаты в px исходного кадра; за границами [0..w]/[0..h] — clip к краю);
* ``min_area_at(x, y) = k_min * h_px^2``, ``max_area_at(x, y) = k_max * h_px^2``;
* если профиль отключён/пустой — методы возвращают глобальные пороги из блока
  ``objects`` (``min/max_area_fraction * w*h``);
* ``buffer_width_px(x, y, scale)`` — ширина буфера вокруг линии пересечения
  (= ``scale * person_height_px``), используется счётчиком-линией.

Аргумент ``y`` во всех публичных методах опционален: ``None`` = середина кадра
(назад-совместимый вызов с одним аргументом); все call-sites передают и x, и y.

Пример::

    sp = SizeProfile(cfg.size_profile, cfg.objects, w=1280, h=720)
    if sp.min_area_at(cx, cy) <= area <= sp.max_area_at(cx, cy):
        ...
"""

from __future__ import annotations

import numpy as np

from .config import ObjectsConfig, SizeProfileConfig

#: минимальная высота человека в px (кламп результата подгонки).
_MIN_PERSON_HEIGHT_PX = 1.0


class HeightSurface:
    """Подогнанная поверхность ``h(x, y)`` по контрольным точкам (чистый класс).

    Работает в НОРМАЛИЗОВАННЫХ координатах (доли кадра 0..1); ``height_at``
    сам клампит x, y к [0, 1] перед вычислением и возвращает высоту в тех же
    единицах, что и h в точках (для конфига — доля высоты кадра).
    """

    def __init__(self, points: list[tuple[float, float, float]]) -> None:
        if not points:
            raise ValueError("fit_height_surface: ожидалось >= 1 точки, получено 0")
        self._pts = [(float(x), float(y), float(h)) for x, y, h in points]

    def height_at(self, nx: float, ny: float) -> float:
        """Высота в точке (nx, ny); за границами [0, 1] — clip к краю."""
        nx = min(1.0, max(0.0, float(nx)))
        ny = min(1.0, max(0.0, float(ny)))
        pts = self._pts
        n = len(pts)
        if n == 1:
            return pts[0][2]
        if n == 2:
            (x0, y0, h0), (x1, y1, h1) = pts
            dx, dy = x1 - x0, y1 - y0
            len2 = dx * dx + dy * dy
            if len2 < 1e-12:
                return h0   # вырожденный отрезок — константа
            t = ((nx - x0) * dx + (ny - y0) * dy) / len2
            t = min(1.0, max(0.0, t))   # за краями отрезка — крайнее значение
            return h0 + t * (h1 - h0)
        x = np.array([p[0] for p in pts], dtype=float)
        y = np.array([p[1] for p in pts], dtype=float)
        hv = np.array([p[2] for p in pts], dtype=float)
        if n >= 4:
            A = np.column_stack([np.ones(n), x, y, x * y])
        else:   # n == 3: плоскость a + b*x + c*y
            A = np.column_stack([np.ones(n), x, y])
        coef, *_rest = np.linalg.lstsq(A, hv, rcond=None)
        val = (coef[0] + coef[1] * nx + coef[2] * ny
               + (coef[3] * nx * ny if n >= 4 else 0.0))
        return float(val)


def fit_height_surface(points: list[tuple[float, float, float]]) -> HeightSurface:
    """Подогнать поверхность высоты ``h(x, y)`` по точкам ``(x, y, h)``.

    Чистая функция (numpy, без cv2/окон); координаты — нормализованные доли
    кадра 0..1. Правила подгонки:

    * n >= 4 — биквадрик ``a + b·x + c·y + d·x·y`` (least squares, numpy);
    * n == 3 — плоскость ``a + b·x + c·y`` (least squares; для неколлинеарных
      точек — точное решение);
    * n == 2 — линейная интерполяция вдоль прямой между точками (проекция на
      отрезок, t=0..1; за краями — крайнее значение);
    * n == 1 — константа h.

    :raises ValueError: пустой список точек.
    """
    return HeightSurface(points)


class SizeProfile:
    """Высота человека и пороги площади как функция точки кадра (x, y)."""

    def __init__(self, sp_cfg: SizeProfileConfig, objects_cfg: ObjectsConfig,
                 w: int, h: int) -> None:
        if w <= 0 or h <= 0:
            raise ValueError(f"размеры кадра: ожидалось w>0 и h>0, получено {w}x{h}")
        self.w = int(w)
        self.h = int(h)

        # Глобальные пороги (px²), используются при отключённом/пустом профиле.
        frame_area = float(self.w * self.h)
        self.global_min_area = objects_cfg.min_area_fraction * frame_area
        self.global_max_area = objects_cfg.max_area_fraction * frame_area

        # Адаптивный режим: включён И хотя бы одна контрольная точка с h > 0.
        pts = [p for p in sp_cfg.control_points if p[2] > 0] if sp_cfg.enabled else []
        self.k_min = sp_cfg.k_min
        self.k_max = sp_cfg.k_max
        self._surface: HeightSurface | None = (
            fit_height_surface(pts) if pts else None
        )

    @property
    def adaptive(self) -> bool:
        """True — работают адаптивные пороги, False — глобальные из objects."""
        return self._surface is not None

    def person_height_px(self, x: float, y: float | None = None) -> float:
        """Ожидаемая высота человека в px в точке (x, y) (px исходного кадра).

        :param x: x-координата точки (px); за краями [0..w] — clip к краю.
        :param y: y-координата точки (px); ``None`` — середина кадра по y
            (назад-совместимый вызов).
        """
        if self._surface is None:
            # Без профиля «высота» не определена; возвращаем высоту кадра как
            # нейтральную величину (влияет только на buffer_width_px).
            return float(self.h)
        ny = 0.5 if y is None else float(y) / self.h
        return max(_MIN_PERSON_HEIGHT_PX,
                   self._surface.height_at(float(x) / self.w, ny) * self.h)

    def min_area_at(self, x: float, y: float | None = None) -> float:
        """Минимальная допустимая площадь blob'а (px²) в точке (x, y)."""
        if not self.adaptive:
            return self.global_min_area
        h_px = self.person_height_px(x, y)
        return self.k_min * h_px * h_px

    def max_area_at(self, x: float, y: float | None = None) -> float:
        """Максимальная допустимая площадь blob'а (px²) в точке (x, y)."""
        if not self.adaptive:
            return self.global_max_area
        h_px = self.person_height_px(x, y)
        return self.k_max * h_px * h_px

    def buffer_width_px(self, x: float, y: float | None = None, scale: float = 0.75) -> float:
        """Ширина буфера вокруг линии в px = scale * высота человека в точке (x, y)."""
        if scale < 0:
            raise ValueError(f"scale: ожидалось >= 0, получено {scale!r}")
        return float(scale) * self.person_height_px(x, y)
