"""Адаптивные пороги площади по положению объекта в кадре.

``SizeProfile`` строится из блока ``size_profile`` конфига и размеров текущего
кадра:

* ``person_height_px(x)`` — ожидаемая высота человека в px в точке ``x``
  (линейная интерполяция между контрольными точками; за краями — крайнее
  значение);
* ``min_area_at(x) = k_min * h_px^2``, ``max_area_at(x) = k_max * h_px^2``;
* если профиль отключён/пустой — методы возвращают глобальные пороги из блока
  ``objects`` (``min/max_area_fraction * w*h``);
* ``buffer_width_px(x, scale)`` — ширина буфера вокруг линии пересечения
  (= ``scale * person_height_px``), используется счётчиком-линией (задача 14).

Пример::

    sp = SizeProfile(cfg.size_profile, cfg.objects, w=1280, h=720)
    if sp.min_area_at(cx) <= area <= sp.max_area_at(cx):
        ...
"""

from __future__ import annotations

from .config import ObjectsConfig, SizeProfileConfig


def _interp(xs: list[float], ys: list[float], x: float) -> float:
    """Линейная интерполяция; за краями — крайнее значение."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            t = (x - xs[i - 1]) / (xs[i] - xs[i - 1]) if xs[i] != xs[i - 1] else 0.0
            return ys[i - 1] + t * (ys[i] - ys[i - 1])
    return ys[-1]


class SizeProfile:
    """Высота человека и пороги площади как функция x-координаты в кадре."""

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

        # Адаптивный режим: включён И хотя бы две контрольные точки.
        pts = sorted(sp_cfg.control_points, key=lambda p: p[0]) if sp_cfg.enabled else []
        if len(pts) >= 2 and all(p[1] > 0 for p in pts):
            self._xs = [p[0] * self.w for p in pts]          # x в px
            self._hs = [p[1] * self.h for p in pts]          # высота человека в px
            self.k_min = sp_cfg.k_min
            self.k_max = sp_cfg.k_max
        else:
            self._xs = []
            self._hs = []

    @property
    def adaptive(self) -> bool:
        """True — работают адаптивные пороги, False — глобальные из objects."""
        return len(self._xs) >= 2

    def person_height_px(self, x: float) -> float:
        """Ожидаемая высота человека в px по x-координате (px, исходный кадр)."""
        if not self.adaptive:
            # Без профиля «высота» не определена; возвращаем высоту кадра как
            # нейтральную величину (влияет только на buffer_width_px).
            return float(self.h)
        x = max(0.0, min(float(x), float(self.w)))
        return _interp(self._xs, self._hs, x)

    def min_area_at(self, x: float) -> float:
        """Минимальная допустимая площадь blob'а (px²) при cx = x."""
        if not self.adaptive:
            return self.global_min_area
        h_px = self.person_height_px(x)
        return self.k_min * h_px * h_px

    def max_area_at(self, x: float) -> float:
        """Максимальная допустимая площадь blob'а (px²) при cx = x."""
        if not self.adaptive:
            return self.global_max_area
        h_px = self.person_height_px(x)
        return self.k_max * h_px * h_px

    def buffer_width_px(self, x: float, scale: float) -> float:
        """Ширина буфера вокруг линии в px = scale * высота человека в точке x."""
        if scale < 0:
            raise ValueError(f"scale: ожидалось >= 0, получено {scale!r}")
        return float(scale) * self.person_height_px(x)
