"""Тесты report.py (задача 10): build_report / choose_report_path / write_report.

Чистые функции без пайплайна: события — синтетические CrossingEvent, конфиг —
Config.from_dict с двумя счётчиками (line + zone).
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from visio_people_counter.config import Config  # noqa: E402
from visio_people_counter.line_counter import CrossingEvent  # noqa: E402
from visio_people_counter.report import (  # noqa: E402
    RunMeta,
    build_report,
    choose_report_path,
    write_report,
)


def _make_cfg(report_path: str = "", video_type: str = "file",
              video_path: str = "/a/b/demo.mp4") -> Config:
    return Config.from_dict({
        "video": {"type": video_type, "path": video_path},
        "counters": [
            {"id": "line_2", "type": "line", "a": [0.25, 0.35], "b": [0.75, 0.85]},
            {"id": "zone_1", "type": "zone",
             "polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9]]},
        ],
        "output": {"report_path": report_path},
    }, "<test>")


def _ev(counter_id: str, direction: str, track_id: int, t_video: float | None = 1.5) -> CrossingEvent:
    return CrossingEvent(
        counter_id=counter_id, direction=direction, track_id=track_id,
        t_wall=100.0 + t_video if t_video is not None else 100.0,
        x_px=123.456, y_px=78.0, frame_index=42, t_video=t_video,
    )


META = RunMeta(source="demo.mp4", frames_processed=341, fps=25.0, duration_s=13.6, reason="EOF")


class TestBuildReport(unittest.TestCase):
    def test_two_counters_table_and_totals(self):
        events = (
            [_ev("line_2", "in", i) for i in range(3)] +
            [_ev("line_2", "out", 10 + i) for i in range(2)] +
            [_ev("zone_1", "in", 20 + i) for i in range(4)] +
            [_ev("zone_1", "out", 30)]
        )
        text = build_report(_make_cfg(), events, META, save_events=True)

        self.assertIn("# Отчёт подсчёта: demo.mp4", text)
        self.assertIn("## Итоги по счётчикам", text)
        # line_2: in=3 out=2 total=5; zone_1: in=4 out=1 total=5
        self.assertIn("| line_2 | line | 3 | 2 | 5 |", text)
        self.assertIn("| zone_1 | zone | 4 | 1 | 5 |", text)
        # ВСЕГО считается по всем счётчикам: in=7 out=3 total=10
        self.assertIn("| **ВСЕГО** | | **7** | **3** | **10** |", text)

        # разделы событий с количеством и строками (время — t_video)
        self.assertIn("## События: line_2 (5)", text)
        self.assertIn("## События: zone_1 (5)", text)
        self.assertIn("| 1 | 1.500 | in | 0 | 123.456 | 78 | 42 |", text)

    def test_counter_without_events_zero_row_and_no_section(self):
        # зона в конфиге, но событий нет → нулевая строка + «(нет)»
        cfg = Config.from_dict({
            "video": {"type": "file", "path": "/a/b/demo.mp4"},
            "counters": [
                {"id": "line_2", "type": "line", "a": [0.25, 0.35], "b": [0.75, 0.85]},
                {"id": "zone_empty", "type": "zone",
                 "polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9]]},
            ],
        }, "<test>")
        text = build_report(cfg, [_ev("line_2", "in", 1)], META, save_events=True)

        self.assertIn("| zone_empty | zone | 0 | 0 | 0 |", text)
        self.assertIn("## События: zone_empty (0)", text)
        # «(нет)» — в разделе пустой зоны, а не в разделе line_2 (событие есть)
        zone_section = text.split("## События: zone_empty (0)")[-1]
        self.assertIn("(нет)", zone_section)

    def test_missing_t_video_shown(self):
        text = build_report(_make_cfg(), [_ev("line_2", "out", 5, t_video=None)], META,
                            save_events=True)
        self.assertIn("| н/д | out | 5 |", text)


class TestBuildReportRoi(unittest.TestCase):
    """Задача 13: строка ROI в markdown-отчёте (есть roi → диапазон, нет → «нет»)."""

    def _cfg(self, roi=None) -> Config:
        d = {
            "video": {"type": "file", "path": "/a/b/demo.mp4"},
            "counters": [{"id": "l", "type": "line",
                          "a": [0.25, 0.35], "b": [0.75, 0.85]}],
        }
        if roi is not None:
            d["processing"] = {"roi": roi}
        return Config.from_dict(d, "<test>")

    def test_roi_line_when_set(self):
        text = build_report(self._cfg([0.1, 0.2, 0.8, 0.6]), [_ev("l", "in", 1)], META)
        self.assertIn("- ROI: 0.1–0.9 × 0.2–0.8", text)
        # примечание: координаты конфига/событий — в системе ROI
        self.assertIn("в системе ROI", text)

    def test_roi_none_when_absent(self):
        text = build_report(self._cfg(), [_ev("l", "in", 1)], META)
        self.assertIn("- ROI: нет", text)
        self.assertNotIn("в системе ROI", text)


class TestChooseReportPath(unittest.TestCase):
    def test_auto_path_for_file(self):
        p = choose_report_path(_make_cfg())
        self.assertEqual(p, Path("/a/b/demo.report.md"))

    def test_hls_without_explicit_path_no_report(self):
        cfg = _make_cfg(video_type="hls", video_path="https://cam.example.com/live/stream.m3u8")
        self.assertIsNone(choose_report_path(cfg))

    def test_explicit_path_wins(self):
        cfg = _make_cfg(report_path="/x/run.md", video_type="hls",
                        video_path="https://cam.example.com/live/stream.m3u8")
        self.assertEqual(choose_report_path(cfg), Path("/x/run.md"))
        # явный путь используется и для файла (автопуть не применяется)
        cfg = _make_cfg(report_path="/y/rel/report.md")
        self.assertEqual(choose_report_path(cfg), Path("/y/rel/report.md"))


class TestWriteReport(unittest.TestCase):
    def test_creates_parent_dirs_and_overwrites(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "nested" / "dir" / "report.md"
            write_report(p, "# первый\n")
            self.assertTrue(p.is_file())
            self.assertEqual(p.read_text(encoding="utf-8"), "# первый\n")

            write_report(p, "# короткий\n")  # перезапись существующего
            self.assertEqual(p.read_text(encoding="utf-8"), "# короткий\n")


if __name__ == "__main__":
    unittest.main()
