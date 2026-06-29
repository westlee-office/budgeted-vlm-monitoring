from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bmvm.data.synthetic import make_synthetic_manifest
from bmvm.io import write_json, load_manifest


class SyntheticManifestTest(unittest.TestCase):
    def test_manifest_roundtrip(self) -> None:
        manifest = make_synthetic_manifest(
            episodes=2,
            streams=4,
            horizon_s=40,
            step_s=5.0,
            events_per_episode=2,
            seed=11,
        )
        path = ROOT / "data" / "synthetic" / "test_roundtrip.json"
        write_json(path, manifest)
        episodes = load_manifest(path)
        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0].num_streams, 4)
        self.assertEqual(len(episodes[0].events), 2)
        self.assertEqual(len(episodes[0].timesteps), 8)
        signal = episodes[0].signal("s000", 0.0)
        self.assertGreaterEqual(signal.motion, 0.0)
        self.assertLessEqual(signal.motion, 1.0)


if __name__ == "__main__":
    unittest.main()
