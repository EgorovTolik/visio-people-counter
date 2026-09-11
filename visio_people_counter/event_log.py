"""JSONL-лог событий пересечения + агрегаты и сводки.

``EventLog`` принимает блок ``output`` конфига (или корневой :class:`Config`):

* события пишутся в ``output.events_jsonl`` — по одной JSON-строке на событие
  (:meth:`CrossingEvent.to_dict <visio_people_counter.line_counter.CrossingEvent.to_dict>`);
  пустая строка/отсутствие пути = не писать файл (агрегаты считаются всё равно);
* агрегаты per-counter ``in``/``out`` + глобальные (свойства :pyattr:`counters`,
  :pyattr:`totals`);
* :meth:`summary` — строка сводки для stdout (pipeline печатает её каждые
  ``output.summary_interval_s`` и при финале).

Thread-безопасность не требуется (один поток pipeline). Файл открывается в режиме
дописывания; каталог создаётся автоматически.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import IO, Optional, Sequence

from .config import Config, OutputConfig
from .line_counter import CrossingEvent


class EventLog:
    """Лог событий и накопитель агрегатов in/out/total."""

    def __init__(self, cfg: "OutputConfig | Config") -> None:
        out_cfg: OutputConfig = cfg.output if isinstance(cfg, Config) else cfg
        self.path: Optional[Path] = Path(out_cfg.events_jsonl).expanduser() if out_cfg.events_jsonl else None
        self._file: Optional[IO[str]] = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.path, "a", encoding="utf-8")

        # агрегаты per-counter: counter_id -> {"in": n, "out": m}
        self._per_counter: dict[str, dict[str, int]] = {}

    # --- запись -------------------------------------------------------------------

    def log(self, event: CrossingEvent) -> dict:
        """Обновить агрегаты и дописать событие в JSONL (если включён).

        :returns: записанный plain-dict события.
        """
        agg = self._per_counter.setdefault(event.counter_id, {"in": 0, "out": 0})
        if event.direction not in ("in", "out"):
            raise ValueError(f"direction: ожидалось 'in' или 'out', получено {event.direction!r}")
        agg[event.direction] += 1
        d = event.to_dict()
        if self._file is not None:
            self._file.write(json.dumps(d, ensure_ascii=False) + "\n")
            self._file.flush()
        return d

    def log_events(self, events: Sequence[CrossingEvent]) -> list[dict]:
        """Записать список событий (обычно результат одного кадра)."""
        return [self.log(ev) for ev in events]

    # --- агрегаты ------------------------------------------------------------------

    @property
    def counters(self) -> dict[str, dict[str, int]]:
        """Агрегаты per-counter: ``{id: {"in", "out", "total"}}`` (копия)."""
        return {cid: {**agg, "total": agg["in"] + agg["out"]}
                for cid, agg in self._per_counter.items()}

    @property
    def totals(self) -> dict[str, int]:
        """Глобальные агрегаты по всем счётчикам: ``{"in", "out", "total"}``."""
        in_sum = sum(a["in"] for a in self._per_counter.values())
        out_sum = sum(a["out"] for a in self._per_counter.values())
        return {"in": in_sum, "out": out_sum, "total": in_sum + out_sum}

    # --- сводка --------------------------------------------------------------------

    def summary(self) -> str:
        """Человекочитаемая строка сводки для stdout."""
        t = self.totals
        if not self._per_counter:
            return f"Сводка: событий нет (in=0 out=0 total=0)"
        parts = []
        for cid in sorted(self._per_counter):
            a = self.counters[cid]
            parts.append(f"{cid}: in={a['in']} out={a['out']} total={a['total']}")
        return "Сводка | " + " | ".join(parts) + \
               f" || ВСЕГО: in={t['in']} out={t['out']} total={t['total']}"

    # --- завершение ------------------------------------------------------------------

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "EventLog":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
