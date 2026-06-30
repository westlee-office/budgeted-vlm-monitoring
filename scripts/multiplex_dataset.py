#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bmvm.data.multiplex import build_multiplex_manifest
from bmvm.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build multi-stream BMVM episodes from videos/events/signals CSV files.")
    parser.add_argument("--videos-csv", required=True, help="CSV with video_id,path,duration_s,label columns.")
    parser.add_argument("--events-csv", required=True, help="CSV with video_id,start_s,end_s,label columns.")
    parser.add_argument("--signals-csv", default=None, help="Optional CSV with video_id,t_s,motion,anomaly,clip columns.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--streams", type=int, default=128)
    parser.add_argument("--horizon-s", type=float, default=1800.0)
    parser.add_argument("--step-s", type=float, default=2.0)
    parser.add_argument("--event-streams-per-episode", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--simulate-missing-signals", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_multiplex_manifest(
        videos_csv=args.videos_csv,
        events_csv=args.events_csv,
        signals_csv=args.signals_csv,
        episodes=args.episodes,
        streams=args.streams,
        horizon_s=args.horizon_s,
        step_s=args.step_s,
        event_streams_per_episode=args.event_streams_per_episode,
        seed=args.seed,
        simulate_missing_signals=args.simulate_missing_signals,
    )
    write_json(args.output, manifest)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
