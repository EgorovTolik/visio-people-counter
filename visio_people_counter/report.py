"""Markdown-отчёт с результатами подсчёта (задача 10).

После запуска ``count`` на диске остаётся файл-отчёт: итоги по каждому
счётчику (линия/зона) + таблица событий. Данные — из события
:class:`~visio_people_counter.line_counter.CrossingEvent` и конфига; пайплайн
лишь собирает список событий и мета-данные прогона и вызывает :func:`write_report`.

* :func:`choose_report_path` — явный ``output.report_path`` имеет приоритет;
  при пустом значении — автопуть ``<видео>.report.md`` рядом с исходным файлом
  (только для локального файла; HLS/URL без явного пути → отчёт не создаётся);
* :func:`build_report` — строит markdown-текст (чистая функция, тестируется
  без пайплайна);
* :func:`write_report` — записывает файл (родительские каталоги создаются).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from .config import Config, describe_frame_range, describe_roi
from .line_counter import CrossingEvent
from .video_source import is_url


@dataclass
class RunMeta:
    """Мета-данные прогона для шапки отчёта."""

    source: str                       # имя/путь видео или URL
    frames_processed: int             # обработано кадров
    fps: Optional[float] = None       # fps источника (None — неизвестен, напр. HLS)
    duration_s: Optional[float] = None  # длительность видео, с (None — неизвестна)
    reason: str = ""                  # причина остановки: EOF / SIGINT / ошибка


def choose_report_path(cfg: Config) -> Optional[Path]:
    """Путь файла-отчёта или ``None`` (отчёт не создаётся).

    * непустой ``output.report_path`` — всегда используется;
    * пустой + локальный файл видео → автопуть ``<stem>.report.md`` рядом с
      файлом (то же правило, что для конфига в calibrate);
    * пустой + HLS/URL (нет локального пути) → ``None``.
    """
    explicit = cfg.output.report_path
    if explicit:
        return Path(explicit).expanduser()
    # приоритет: имя конфига (если задан config_path)
    if cfg.config_path:
        base = Path(cfg.config_path).expanduser()
        stem = base.stem  # без .yaml/.yml
        return base.with_name(stem + ".report.md")
    # fallback: имя видео (для вживую / default-конфига)
    v = cfg.video
    if v.type == "file" and v.path and not is_url(v.path):
        base = Path(v.path).expanduser()
        if base.suffix:
            return base.with_suffix(".report.md")
        return base.with_name(base.name + ".report.md")
    return None


def _fmt_num(x: float) -> str:
    """Число в отчёте: без хвоста .0 у целых (25 → 25, 13.6 → 13.6)."""
    if x == int(x):
        return str(int(x))
    return f"{x:g}"


def build_report(cfg: Config, events: Sequence[CrossingEvent], meta: RunMeta,
                 *, save_events: bool = False) -> str:
    """Собрать markdown-отчёт прогона.

    :param cfg: конфиг (список счётчиков — разрез «Итогов» по каждому из них).
    :param events: ВСЕ события прогона (в порядке их возникновения).
    :param meta: мета-данные прогона (кадры, fps, длительность, reason).
    :param save_events: True — записать разделы «События: …» с таблицей каждого
        пересечения; False (по умолчанию) — только итоги по счётчикам.
    """
    lines: list[str] = []
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    lines.append(f"# Отчёт подсчёта: {meta.source}")
    lines.append("")
    lines.append(f"- дата запуска: {started_at}")
    dur = f"{_fmt_num(meta.duration_s)} c" if meta.duration_s else "н/д"
    fps = _fmt_num(meta.fps) if meta.fps else "н/д"
    lines.append(f"- длительность видео: {dur}, кадров: {meta.frames_processed}, fps: {fps}")
    lines.append(f"- обработано кадров: {meta.frames_processed}, причина остановки: {meta.reason}")
    # интервал кадров подсчёта (processing.frame_start/frame_end; 0-based, end включительно)
    lines.append(
        f"- интервал кадров: "
        f"{describe_frame_range(cfg.processing.frame_start, cfg.processing.frame_end)}"
    )
    # ROI (processing.roi): кроп на уровне источника; None/отсутствует → «нет»
    roi = cfg.processing.roi
    lines.append(f"- ROI: {describe_roi(roi)}")
    if roi is not None:
        lines.append(
            "- примечание: координаты конфига (линии/зоны/size-точки) и событий "
            "(x_px/y_px, events.jsonl) — в системе ROI, т.е. отсчитываются от кропа"
        )
    lines.append("")

    # --- итоги per-counter (по счётчикам из конфига; 0 событий → нули) ----------
    agg: dict[str, dict[str, int]] = {}
    for ev in events:
        a = agg.setdefault(ev.counter_id, {"in": 0, "out": 0})
        a[ev.direction] += 1

    lines.append("## Итоги по счётчикам")
    lines.append("")
    lines.append("| Счётчик | Тип | in | out | total |")
    lines.append("|---|---|---|---|---|")
    tot_in = tot_out = 0
    for counter in cfg.counters:
        a = agg.get(counter.id, {"in": 0, "out": 0})
        total = a["in"] + a["out"]
        tot_in += a["in"]
        tot_out += a["out"]
        lines.append(f"| {counter.id} | {counter.type} | {a['in']} | {a['out']} | {total} |")
    lines.append(
        f"| **ВСЕГО** | | **{tot_in}** | **{tot_out}** | **{tot_in + tot_out}** |"
    )
    lines.append("")

    # --- события per-counter (только при save_events=True) --------------------
    if save_events:
        for counter in cfg.counters:
            evs = [ev for ev in events if ev.counter_id == counter.id]
            lines.append(f"## События: {counter.id} ({len(evs)})")
            lines.append("")
            if not evs:
                lines.append("(нет)")
                lines.append("")
                continue
            lines.append("| # | время (с) | направление | track_id | x_px | y_px | кадр |")
            lines.append("|---|---|---|---|---|---|---|")
            for i, ev in enumerate(evs, start=1):
                t = f"{ev.t_video:.3f}" if ev.t_video is not None else "н/д"
                lines.append(
                    f"| {i} | {t} | {ev.direction} | {ev.track_id} "
                    f"| {_fmt_num(ev.x_px)} | {_fmt_num(ev.y_px)} | {ev.frame_index} |"
                )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_report(path: str | Path, text: str) -> Path:
    """Записать отчёт в ``path`` (родительские каталоги создаются; существующий файл перезаписывается)."""
    p = Path(path).expanduser()
    if p.parent and str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p
