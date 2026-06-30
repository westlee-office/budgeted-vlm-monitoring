from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class UpdatePaperResultsTest(unittest.TestCase):
    def test_update_paper_results_from_grid(self) -> None:
        base = ROOT / "data" / "synthetic" / "paper_update_test"
        base.mkdir(parents=True, exist_ok=True)
        aggregate = base / "aggregate.json"
        template = base / "template.json"
        output = base / "assumed_results.json"
        aggregate.write_text(
            json.dumps(
                {
                    "runs": [
                        {
                            "policy": "voi",
                            "query_budget_per_step": 4,
                            "num_streams": 128,
                            "event_recall": 0.8,
                            "mean_time_to_detect_s": 20.0,
                            "false_alarms_per_hour": 2.0,
                            "gpu_seconds_per_hour": 100.0,
                            "vlm_calls_per_event": 10.0,
                        },
                        {
                            "policy": "uniform",
                            "query_budget_per_step": 4,
                            "num_streams": 128,
                            "event_recall": 0.5,
                            "mean_time_to_detect_s": 40.0,
                            "false_alarms_per_hour": 3.0,
                            "gpu_seconds_per_hour": 100.0,
                            "vlm_calls_per_event": 10.0,
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        template.write_text(
            json.dumps(
                {
                    "main_results": [],
                    "ablation": [],
                    "stream_scaling": [],
                    "benchmark_comparison": [],
                    "ttd_cdf": [],
                    "timeline": {"streams": []},
                }
            ),
            encoding="utf-8",
        )
        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "scripts" / "update_paper_results_from_grid.py"),
                "--aggregate",
                str(aggregate),
                "--template",
                str(template),
                "--output",
                str(output),
                "--main-budget",
                "4",
                "--write",
            ],
            cwd=ROOT,
        )
        data = json.loads(output.read_text(encoding="utf-8"))
        methods = {row["method"]: row for row in data["main_results"]}
        self.assertEqual(methods["TriageVLM"]["recall"], 0.8)
        self.assertEqual(data["stream_scaling"][0]["streams"], 128)
        self.assertEqual(data["stream_scaling"][0]["triagevlm"], 0.8)


if __name__ == "__main__":
    unittest.main()
