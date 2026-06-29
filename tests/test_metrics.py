from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bmvm.metrics import markdown_table


class MetricsTest(unittest.TestCase):
    def test_markdown_table_formats_floats(self) -> None:
        table = markdown_table(
            [
                {
                    "policy": "voi",
                    "event_recall": 0.81234,
                    "mean_time_to_detect_s": 12.345,
                    "false_alarms_per_hour": 1.2,
                    "gpu_seconds_per_hour": 120.0,
                    "vlm_calls_per_event": 8.9,
                }
            ]
        )
        self.assertIn("| voi | 0.812 | 12.345 | 1.200 | 120.000 | 8.900 |", table)


if __name__ == "__main__":
    unittest.main()
