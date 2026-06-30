#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple


def read_feature_csv(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge source-video feature CSVs into BMVM signals.csv.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Feature CSVs keyed by video_id,t_s.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--round-timestep", type=float, default=1.0)
    parser.add_argument("--default-motion", type=float, default=0.0)
    parser.add_argument("--default-anomaly", type=float, default=0.0)
    parser.add_argument("--default-clip", type=float, default=0.0)
    return parser.parse_args()


def rounded_t(raw_t: str, step: float) -> float:
    t = float(raw_t)
    return round(round(t / step) * step, 3)


def main() -> None:
    args = parse_args()
    rows: Dict[Tuple[str, float], Dict[str, float]] = defaultdict(dict)
    for input_path in args.inputs:
        for row in read_feature_csv(Path(input_path)):
            video_id = row.get("video_id")
            if not video_id:
                raise SystemExit(f"{input_path} must include video_id")
            t_s = rounded_t(row["t_s"], args.round_timestep)
            key = (video_id, t_s)
            for name in ["motion", "anomaly", "clip"]:
                if row.get(name) not in (None, ""):
                    rows[key][name] = float(row[name])

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "t_s", "motion", "anomaly", "clip"])
        writer.writeheader()
        for video_id, t_s in sorted(rows):
            values = rows[(video_id, t_s)]
            writer.writerow(
                {
                    "video_id": video_id,
                    "t_s": f"{t_s:.3f}",
                    "motion": f"{values.get('motion', args.default_motion):.6f}",
                    "anomaly": f"{values.get('anomaly', args.default_anomaly):.6f}",
                    "clip": f"{values.get('clip', args.default_clip):.6f}",
                }
            )
    print(f"Wrote {output} with {len(rows)} rows")


if __name__ == "__main__":
    main()
