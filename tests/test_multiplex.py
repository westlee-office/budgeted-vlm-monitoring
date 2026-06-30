from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bmvm.data.multiplex import build_multiplex_manifest
from bmvm.io import write_json, load_manifest
from bmvm.policies import POLICY_REGISTRY


class MultiplexTest(unittest.TestCase):
    def write_csv(self, path: Path, fieldnames, rows) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_build_multiplex_manifest_with_simulated_signals(self) -> None:
        base = ROOT / "data" / "synthetic" / "multiplex_test"
        videos_csv = base / "videos.csv"
        events_csv = base / "events.csv"
        self.write_csv(
            videos_csv,
            ["video_id", "path", "duration_s", "label"],
            [
                {"video_id": "v_event_1", "path": "event1.mp4", "duration_s": "60", "label": "robbery"},
                {"video_id": "v_event_2", "path": "event2.mp4", "duration_s": "60", "label": "fall"},
                {"video_id": "v_normal_1", "path": "normal1.mp4", "duration_s": "60", "label": "normal"},
                {"video_id": "v_normal_2", "path": "normal2.mp4", "duration_s": "60", "label": "normal"},
            ],
        )
        self.write_csv(
            events_csv,
            ["video_id", "event_id", "start_s", "end_s", "label", "severity", "description"],
            [
                {"video_id": "v_event_1", "event_id": "e1", "start_s": "10", "end_s": "20", "label": "robbery", "severity": "1", "description": "robbery"},
                {"video_id": "v_event_2", "event_id": "e2", "start_s": "25", "end_s": "35", "label": "fall", "severity": "0.8", "description": "fall"},
            ],
        )
        manifest = build_multiplex_manifest(
            videos_csv=videos_csv,
            events_csv=events_csv,
            episodes=2,
            streams=4,
            horizon_s=60,
            step_s=5,
            event_streams_per_episode=2,
            seed=3,
            simulate_missing_signals=True,
        )
        path = base / "manifest.json"
        write_json(path, manifest)
        episodes = load_manifest(path)
        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0].num_streams, 4)
        self.assertGreaterEqual(len(episodes[0].events), 2)
        active_signals = [signal for signal in episodes[0].signals.values() if signal.event_ids]
        self.assertTrue(active_signals)
        self.assertGreater(max(signal.anomaly for signal in active_signals), 0.5)

    def test_dense_policy_registered(self) -> None:
        self.assertIn("dense_vlm", POLICY_REGISTRY)
        self.assertIn("dense", POLICY_REGISTRY)


if __name__ == "__main__":
    unittest.main()
