#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bmvm.io import load_manifest
from bmvm.vlm_cache import VLMRecord, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an oracle-like VLM cache for debugging the evaluation pipeline.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--positive-score", type=float, default=0.92)
    parser.add_argument("--negative-score", type=float, default=0.08)
    parser.add_argument("--model", default="oracle-debug-cache")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    for episode in load_manifest(args.manifest):
        for t_s in episode.timesteps:
            for stream_id in episode.stream_ids:
                active = episode.active_events(stream_id, t_s)
                records.append(
                    VLMRecord(
                        episode_id=episode.episode_id,
                        stream_id=stream_id,
                        t_s=t_s,
                        score=args.positive_score if active else args.negative_score,
                        summary="; ".join(event.description or event.label for event in active),
                        model=args.model,
                        prompt_hash="oracle",
                    )
                )
    write_jsonl(args.output, records)
    print(f"Wrote {args.output} with {len(records)} records")


if __name__ == "__main__":
    main()
