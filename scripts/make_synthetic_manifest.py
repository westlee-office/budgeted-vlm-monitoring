#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bmvm.data.synthetic import make_synthetic_manifest
from bmvm.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a synthetic BMVM manifest.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--streams", type=int, default=32)
    parser.add_argument("--horizon-s", type=int, default=300)
    parser.add_argument("--step-s", type=float, default=5.0)
    parser.add_argument("--events-per-episode", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = make_synthetic_manifest(
        episodes=args.episodes,
        streams=args.streams,
        horizon_s=args.horizon_s,
        step_s=args.step_s,
        events_per_episode=args.events_per_episode,
        seed=args.seed,
    )
    write_json(args.output, manifest)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
