from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bmvm.data.synthetic import make_synthetic_manifest
from bmvm.evaluation import run_benchmark
from bmvm.io import load_manifest, write_json
from bmvm.metrics import summarize_policy_results
from bmvm.policies import POLICY_REGISTRY
from bmvm.types import Budget, CostModel


class EvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        manifest = make_synthetic_manifest(
            episodes=3,
            streams=12,
            horizon_s=80,
            step_s=5.0,
            events_per_episode=3,
            seed=13,
        )
        cls.path = ROOT / "data" / "synthetic" / "test_eval.json"
        write_json(cls.path, manifest)
        cls.episodes = load_manifest(cls.path)

    def test_run_benchmark_returns_policy_summaries(self) -> None:
        policies = [
            POLICY_REGISTRY["uniform"](query_budget=2, seed=3),
            POLICY_REGISTRY["anomaly_topk"](query_budget=2, seed=3),
            POLICY_REGISTRY["voi"](query_budget=2, seed=3),
        ]
        results = run_benchmark(
            episodes=self.episodes,
            policies=policies,
            budget=Budget(query_budget_per_step=2),
            cost_model=CostModel(),
        )
        summary = summarize_policy_results(self.episodes, results)
        self.assertEqual(len(summary), 3)
        for row in summary:
            self.assertGreaterEqual(row["event_recall"], 0.0)
            self.assertLessEqual(row["event_recall"], 1.0)
            self.assertGreater(row["gpu_seconds_per_hour"], 0.0)
            self.assertGreater(row["vlm_calls_per_event"], 0.0)

    def test_voi_is_registered(self) -> None:
        self.assertIn("voi", POLICY_REGISTRY)


if __name__ == "__main__":
    unittest.main()
