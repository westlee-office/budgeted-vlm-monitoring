from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class VLMPredictionAdapterTest(unittest.TestCase):
    def test_build_vlm_cache_from_predictions(self) -> None:
        base = ROOT / "data" / "synthetic" / "prediction_adapter_test"
        base.mkdir(parents=True, exist_ok=True)
        query_pool = base / "queries.jsonl"
        predictions = base / "predictions.jsonl"
        output = base / "cache.jsonl"
        query_pool.write_text(
            json.dumps({"episode_id": "e", "stream_id": "s001", "t_s": 5.0, "path": "video.mp4"}) + "\n",
            encoding="utf-8",
        )
        predictions.write_text(
            json.dumps({"episode_id": "e", "stream_id": "s001", "t_s": 5.0, "score": 0.83, "summary": "fall"}) + "\n",
            encoding="utf-8",
        )
        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "scripts" / "build_vlm_cache_from_predictions.py"),
                "--query-pool",
                str(query_pool),
                "--predictions",
                str(predictions),
                "--output",
                str(output),
                "--model",
                "test-model",
            ],
            cwd=ROOT,
        )
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["score"], 0.83)
        self.assertEqual(rows[0]["summary"], "fall")


if __name__ == "__main__":
    unittest.main()
