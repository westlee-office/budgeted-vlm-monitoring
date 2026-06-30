from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bmvm.data.synthetic import make_synthetic_manifest
from bmvm.evaluation import run_benchmark
from bmvm.io import load_manifest, write_json
from bmvm.policies import POLICY_REGISTRY
from bmvm.types import Budget, CostModel
from bmvm.vlm_cache import VLMCache, VLMRecord, write_jsonl


class VLMCacheTest(unittest.TestCase):
    def test_vlm_cache_roundtrip(self) -> None:
        path = ROOT / "data" / "synthetic" / "cache_test.jsonl"
        write_jsonl(
            path,
            [
                VLMRecord(
                    episode_id="ep",
                    stream_id="s001",
                    t_s=10.0,
                    score=0.9,
                    summary="event",
                    model="test-model",
                    prompt_hash="abc",
                )
            ],
        )
        cache = VLMCache.from_jsonl(path)
        record = cache.get("ep", "s001", 10.0)
        self.assertIsNotNone(record)
        self.assertEqual(record.score, 0.9)
        self.assertEqual(record.summary, "event")

    def test_vlm_cache_can_drive_evaluation_without_fallback(self) -> None:
        manifest = make_synthetic_manifest(
            episodes=1,
            streams=4,
            horizon_s=40,
            step_s=5.0,
            events_per_episode=1,
            seed=19,
        )
        manifest_path = ROOT / "data" / "synthetic" / "cache_eval_manifest.json"
        cache_path = ROOT / "data" / "synthetic" / "cache_eval.jsonl"
        write_json(manifest_path, manifest)
        episodes = load_manifest(manifest_path)
        records = []
        for episode in episodes:
            for t_s in episode.timesteps:
                for stream_id in episode.stream_ids:
                    active = episode.active_events(stream_id, t_s)
                    records.append(
                        VLMRecord(
                            episode_id=episode.episode_id,
                            stream_id=stream_id,
                            t_s=t_s,
                            score=0.95 if active else 0.01,
                        )
                    )
        write_jsonl(cache_path, records)
        results = run_benchmark(
            episodes=episodes,
            policies=[POLICY_REGISTRY["dense_vlm"](query_budget=1, seed=1)],
            budget=Budget(query_budget_per_step=1),
            cost_model=CostModel(),
            vlm_cache=VLMCache.from_jsonl(cache_path),
            simulated_vlm_fallback=False,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0].detected_events), 1)


if __name__ == "__main__":
    unittest.main()
