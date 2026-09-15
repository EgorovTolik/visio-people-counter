"""On-screen текст с кириллицей для cv2-окон.

cv2.putText использует встроенные Hershey-шрифты и НЕ поддерживает кириллицу
(рисует нечитаемые глифы). Стандартное решение — отрисовать текст Pillow (PIL)
с TTF-шрифтом на прозрачной подложке и «налепить» её на кадр.

Интерфейс:
    put_text(frame, text, org, size_px=20, color=(B,G,R), shadow=True)
    text_width(text, size_px) -> int

Если Pillow или TTF-шрифт недоступны — тихий фолбэк на cv2.putText с заменой
не-ASCII символов на '?' (интерфейс остаётся рабочим, кириллица теряется).
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional, Tuple

import cv2
import numpy as np

ColorBGR = Tuple[int, int, int]

#: переопределение пути к шрифту (редко нужно)
_FONT_ENV = "VPC_FONT_PATH"

#: кандидаты TTF с кириллицей (Linux-дистрибутивы; первый найденный используется)
_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",
)

_PIL: Optional[object] = None          # модуль PIL (None = ещё не проверяли, False = нет)
_FONT_PATH: Optional[str] = None       # найденный шрифт; '' = проверили, не нашли


def _detect_pil() -> bool:
    """Один раз: есть ли Pillow + TTF-шрифт с кириллицей."""
    global _PIL, _FONT_PATH
    if _PIL is not None or _FONT_PATH is not None:
        return _PIL is True and _FONT_PATH != ""
    ok = False
    path = os.environ.get(_FONT_ENV, "")
    try:
        from PIL import Image, ImageDraw, ImageFont  # noqa: F401
        if path and os.path.isfile(path):
            _FONT_PATH = path
            _PIL = True
            ok = True
        else:
            for cand in _FONT_CANDIDATES:
                if os.path.isfile(cand):
                    _FONT_PATH = cand
                    _PIL = True
                    ok = True
                    break
    except Exception:
        pass
    if not ok:
        _PIL, _FONT_PATH = False, ""
    return ok


def pil_available() -> bool:
    """True — кириллица будет рендериться через PIL/TTF."""
    return _detect_pil()


@lru_cache(maxsize=24)
def _load_font(size_px: int):
    from PIL import ImageFont
    return ImageFont.truetype(_FONT_PATH, int(size_px))


def _draw_with_pil(frame: np.ndarray, text: str, org: Tuple[int, int],
                   size_px: int, color_bgr: ColorBGR, shadow: bool) -> None:
    """Рендер строки на прозрачную RGBA-подложку и композитинг на кадр (in-place)."""
    from PIL import Image, ImageDraw

    font = _load_font(size_px)
    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    bbox = probe.textbbox((0, 0), text, font=font)
    pad = 4                                   # запас на тень/антиалиасинг
    tw = max(1, bbox[2] - bbox[0]) + 2 * pad
    th = max(1, bbox[3] - bbox[1]) + 2 * pad
    layer = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    rgb: Tuple[int, int, int] = (color_bgr[2], color_bgr[1], color_bgr[0])  # BGR→RGB
    ox = pad - bbox[0]
    oy = pad - bbox[1]
    if shadow:
        d.text((ox + 2, oy + 2), text, font=font, fill=(0, 0, 0, 255))
    d.text((ox, oy), text, font=font, fill=rgb + (255,))

    arr = np.asarray(layer)                   # RGBA uint8
    x, y = int(org[0]), int(org[1])
    H, W = frame.shape[:2]
    # пересечение подложки с кадром (обрезка за границами — без ошибок)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + tw), min(H, y + th)
    if x0 >= x1 or y0 >= y1:
        return
    sx0, sy0 = x0 - x, y0 - y                 # сдвиг внутри подложки
    sub = arr[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
    alpha = sub[..., 3:4].astype(np.float32) / 255.0
    region = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (alpha * sub[..., :3] + (1.0 - alpha) * region).astype(np.uint8)


def _fallback_put_text(frame: np.ndarray, text: str, org: Tuple[int, int],
                       size_px: int, color_bgr: ColorBGR, shadow: bool) -> None:
    """cv2.putText: только ASCII; не-ASCII → '?'. Масштаб ~ size_px/25 px."""
    safe = "".join(ch if 32 <= ord(ch) < 127 else "?" for ch in text)
    scale = max(0.4, size_px / 25.0)
    org_cv = (int(org[0]), int(org[1]) + int(size_px * 0.8))   # PIL: top-left → базовая линия
    if shadow:
        cv2.putText(frame, safe, (org_cv[0] + 1, org_cv[1] + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, safe, org_cv, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color_bgr, 2, cv2.LINE_AA)


def put_text(frame: Optional[np.ndarray], text: str, org: Tuple[int, int], *,
             size_px: int = 20, color: ColorBGR = (255, 255, 255),
             shadow: bool = True) -> np.ndarray:
    """Нарисовать текст на BGR-кадре (in-place, возвращает тот же кадр).

    :param org: верхний-левый угол текста в px (как у PIL draw.text).
    :param size_px: высота глифов примерно в пикселях.
    :param color: цвет в BGR.
    :param shadow: чёрная тень со смещением +2px (читаемость над любым фоном).
    """
    if frame is None or not text or frame.size == 0:
        return frame
    if _detect_pil():
        try:
            _draw_with_pil(frame, text, org, size_px, color, shadow)
            return frame
        except Exception:
            pass  # любой сбой рендера не должен ронять конвейер — фолбэк ниже
    _fallback_put_text(frame, text, org, size_px, color, shadow)
    return frame


def text_width(text: str, size_px: int = 20) -> int:
    """Ширина строки в px (для выравнивания по правому краю и т.п.)."""
    if _detect_pil():
        from PIL import Image, ImageDraw
        font = _load_font(size_px)
        probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
        bbox = probe.textbbox((0, 0), text, font=font)
        return max(1, int(bbox[2] - bbox[0]))
    scale = max(0.4, size_px / 25.0)
    return int(cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0])
