#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a BMVM result JSON.")
    parser.add_argument("result_json")
    parser.add_argument("--format", choices=["md", "json"], default="md")
    return parser.parse_args()


def markdown_table(rows: List[Dict[str, Any]]) -> str:
    headers = [
        "policy",
        "event_recall",
        "mean_time_to_detect_s",
        "false_alarms_per_hour",
        "gpu_seconds_per_hour",
        "vlm_calls_per_event",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        cells = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                cells.append(f"{value:.3f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    with Path(args.result_json).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload.get("summary") or payload.get("runs")
    if rows is None:
        raise SystemExit("Result JSON must include either 'summary' or aggregate 'runs'.")
    if args.format == "json":
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print(markdown_table(rows))


if __name__ == "__main__":
    main()
