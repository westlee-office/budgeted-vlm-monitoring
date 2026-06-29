#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bmvm.data.csv_manifest import build_manifest_from_csv
from bmvm.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a BMVM manifest from event and signal CSV files.")
    parser.add_argument("--events-csv", required=True, help="CSV with episode_id,stream_id,start_s,end_s,label columns.")
    parser.add_argument("--signals-csv", required=True, help="CSV with episode_id,stream_id,t_s,motion,anomaly,clip columns.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--step-s", type=float, default=2.0)
    parser.add_argument("--horizon-s", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest_from_csv(
        events_csv=args.events_csv,
        signals_csv=args.signals_csv,
        default_step_s=args.step_s,
        default_horizon_s=args.horizon_s,
    )
    write_json(args.output, manifest)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
