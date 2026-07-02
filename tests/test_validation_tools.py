from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bmvm.data.synthetic import make_synthetic_manifest
from bmvm.evaluation import run_episode
from bmvm.io import write_json
from bmvm.policies import POLICY_REGISTRY
from bmvm.types import Budget, CostModel, Episode, FrameSignal
from bmvm.vlm_cache import VLMCache


class ValidationToolsTest(unittest.TestCase):
    def write_csv(self, path: Path, fieldnames, rows) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_validate_dataset_csvs_source_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            videos = base / "videos.csv"
            events = base / "events.csv"
            signals = base / "signals.csv"
            report = base / "report.json"
            self.write_csv(
                videos,
                ["video_id", "path", "duration_s", "label", "split"],
                [
                    {"video_id": "v1", "path": "v1.mp4", "duration_s": "60", "label": "fall", "split": "train"},
                    {"video_id": "v2", "path": "v2.mp4", "duration_s": "60", "label": "normal", "split": "train"},
                ],
            )
            self.write_csv(
                events,
                ["video_id", "event_id", "start_s", "end_s", "label"],
                [{"video_id": "v1", "event_id": "e1", "start_s": "10", "end_s": "20", "label": "fall"}],
            )
            self.write_csv(
                signals,
                ["video_id", "t_s", "motion", "anomaly", "clip"],
                [
                    {"video_id": "v1", "t_s": "10", "motion": "0.2", "anomaly": "0.8", "clip": "0.7"},
                    {"video_id": "v2", "t_s": "10", "motion": "0.1", "anomaly": "0.1", "clip": "0.2"},
                ],
            )
            subprocess.check_call(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "validate_dataset_csvs.py"),
                    "--videos",
                    str(videos),
                    "--events",
                    str(events),
                    "--signals",
                    str(signals),
                    "--output",
                    str(report),
                ],
                cwd=ROOT,
            )
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["video_count"], 2)
            self.assertEqual(payload["summary"]["event_count"], 1)
            self.assertEqual(payload["errors"], [])

    def test_validate_dataset_csvs_rejects_bad_event_interval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            videos = base / "videos.csv"
            events = base / "events.csv"
            self.write_csv(
                videos,
                ["video_id", "duration_s"],
                [{"video_id": "v1", "duration_s": "30"}],
            )
            self.write_csv(
                events,
                ["video_id", "start_s", "end_s", "label"],
                [{"video_id": "v1", "start_s": "20", "end_s": "10", "label": "fall"}],
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "validate_dataset_csvs.py"),
                    "--videos",
                    str(videos),
                    "--events",
                    str(events),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("non_positive_event_interval", completed.stdout)

    def test_validate_vlm_cache_reports_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            query_pool = base / "queries.jsonl"
            cache = base / "cache.jsonl"
            report = base / "report.json"
            query_pool.write_text(
                "\n".join(
                    [
                        json.dumps({"episode_id": "e", "stream_id": "s000", "t_s": 0.0}),
                        json.dumps({"episode_id": "e", "stream_id": "s001", "t_s": 0.0}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            cache.write_text(
                json.dumps(
                    {
                        "episode_id": "e",
                        "stream_id": "s000",
                        "t_s": 0.0,
                        "score": 0.9,
                        "model": "test",
                        "prompt_hash": "abc",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "validate_vlm_cache.py"),
                    "--query-pool",
                    str(query_pool),
                    "--cache",
                    str(cache),
                    "--output",
                    str(report),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["coverage"], 0.5)
            self.assertEqual(payload["summary"]["missing_keys"], 1)

    def test_no_simulated_fallback_raises_on_missing_cache_key(self) -> None:
        episode = Episode(
            episode_id="e",
            num_streams=1,
            horizon_s=2.0,
            step_s=2.0,
            events=[],
            signals={
                ("s000", 0.0): FrameSignal(
                    episode_id="e",
                    stream_id="s000",
                    t_s=0.0,
                    motion=0.1,
                    anomaly=0.2,
                    clip=0.3,
                )
            },
        )
        policy = POLICY_REGISTRY["uniform"](query_budget=1, seed=7)
        with self.assertRaises(KeyError):
            run_episode(
                episode=episode,
                policy=policy,
                budget=Budget(query_budget_per_step=1),
                cost_model=CostModel(),
                vlm_cache=VLMCache([]),
                simulated_vlm_fallback=False,
            )

    def test_run_grid_honors_stream_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            manifest_dir = base / "manifests"
            output_dir = base / "grid"
            config = base / "grid.json"
            write_json(
                manifest_dir / "toy_multistream_2.json",
                make_synthetic_manifest(episodes=1, streams=2, horizon_s=4, step_s=2, events_per_episode=1, seed=1),
            )
            write_json(
                manifest_dir / "toy_multistream_3.json",
                make_synthetic_manifest(episodes=1, streams=3, horizon_s=4, step_s=2, events_per_episode=1, seed=2),
            )
            write_json(
                config,
                {
                    "datasets": ["toy_multistream"],
                    "stream_counts": [2, 3],
                    "policies": ["random"],
                    "query_budgets_per_step": [1],
                    "seeds": [7],
                    "dense_vlm_reference": False,
                },
            )
            subprocess.check_call(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_grid.py"),
                    "--config",
                    str(config),
                    "--manifest-dir",
                    str(manifest_dir),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=ROOT,
            )
            aggregate = json.loads((output_dir / "aggregate.json").read_text(encoding="utf-8"))
            self.assertEqual(len(aggregate["runs"]), 2)
            self.assertEqual([row["stream_count_requested"] for row in aggregate["runs"]], [2, 3])
            self.assertEqual([row["num_streams"] for row in aggregate["runs"]], [2, 3])


if __name__ == "__main__":
    unittest.main()
