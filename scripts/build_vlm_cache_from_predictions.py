#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def key(row: Dict[str, object]) -> Tuple[str, str, float]:
    return (str(row["episode_id"]), str(row["stream_id"]), round(float(row["t_s"]), 6))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert external VLM predictions into BMVM VLM cache JSONL.")
    parser.add_argument("--query-pool", required=True)
    parser.add_argument("--predictions", required=True, help="JSONL with episode_id,stream_id,t_s,score and optional summary.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--prompt-hash", default="")
    parser.add_argument("--missing-score", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query_rows = list(read_jsonl(Path(args.query_pool)))
    predictions = {key(row): row for row in read_jsonl(Path(args.predictions))}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    missing = 0
    with output.open("w", encoding="utf-8") as f:
        for query in query_rows:
            pred = predictions.get(key(query))
            if pred is None:
                missing += 1
                if args.missing_score is None:
                    continue
                pred = {"score": args.missing_score, "summary": ""}
            row = {
                "episode_id": query["episode_id"],
                "stream_id": query["stream_id"],
                "t_s": query["t_s"],
                "score": float(pred["score"]),
                "summary": pred.get("summary", ""),
                "model": pred.get("model", args.model),
                "prompt_hash": pred.get("prompt_hash", args.prompt_hash),
            }
            f.write(json.dumps(row, sort_keys=True) + "\n")
            written += 1
    print(f"Wrote {output} with {written} records; missing predictions={missing}")


if __name__ == "__main__":
    main()
