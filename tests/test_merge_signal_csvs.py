from __future__ import annotations

import csv
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class MergeSignalCsvsTest(unittest.TestCase):
    def write_csv(self, path: Path, fieldnames, rows) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_merge_signal_csvs(self) -> None:
        base = ROOT / "data" / "synthetic" / "merge_test"
        motion = base / "motion.csv"
        clip = base / "clip.csv"
        output = base / "signals.csv"
        self.write_csv(
            motion,
            ["video_id", "t_s", "motion"],
            [{"video_id": "v1", "t_s": "1.1", "motion": "0.5"}],
        )
        self.write_csv(
            clip,
            ["video_id", "t_s", "clip"],
            [{"video_id": "v1", "t_s": "1.0", "clip": "0.7"}],
        )
        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "scripts" / "merge_signal_csvs.py"),
                "--inputs",
                str(motion),
                str(clip),
                "--output-csv",
                str(output),
                "--round-timestep",
                "1.0",
            ],
            cwd=ROOT,
        )
        with output.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["video_id"], "v1")
        self.assertEqual(rows[0]["motion"], "0.500000")
        self.assertEqual(rows[0]["clip"], "0.700000")


if __name__ == "__main__":
    unittest.main()
