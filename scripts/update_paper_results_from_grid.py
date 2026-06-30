#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple


POLICY_TO_METHOD = {
    "random": "Random scheduler",
    "uniform": "Uniform round-robin",
    "motion_topk": "Motion top-k",
    "anomaly_topk": "Anomaly-score top-k",
    "clip_topk": "CLIP prompt top-k",
    "dense_vlm": "Dense VLM every stream",
    "voi": "TriageVLM",
}

ABLATION_TO_VARIANT = {
    "voi": "TriageVLM",
    "voi_no_memory": "w/o event memory",
    "voi_no_uncertainty": "w/o uncertainty bonus",
    "voi_no_clip": "w/o CLIP prompt score",
    "voi_no_anomaly": "w/o anomaly prior",
    "voi_no_cooldown": "greedy no cooldown",
}

MAIN_ORDER = [
    "random",
    "uniform",
    "motion_topk",
    "anomaly_topk",
    "clip_topk",
    "dense_vlm",
    "voi",
]

ABLATION_ORDER = [
    "voi",
    "voi_no_memory",
    "voi_no_uncertainty",
    "voi_no_clip",
    "voi_no_anomaly",
    "voi_no_cooldown",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update paper assumed_results.json from BMVM grid aggregate JSON.")
    parser.add_argument("--aggregate", nargs="+", required=True, help="One or more results/grid/aggregate.json files.")
    parser.add_argument("--template", default="paper/iclr2027/assumed_results.json")
    parser.add_argument("--output", default="paper/iclr2027/assumed_results.json")
    parser.add_argument("--main-budget", type=int, default=4)
    parser.add_argument("--write", action="store_true", help="Write output file. Without this flag, prints JSON to stdout.")
    return parser.parse_args()


def load_rows(paths: Iterable[str]) -> List[Dict[str, Any]]:
    rows = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if "runs" in payload:
            rows.extend(payload["runs"])
        elif "summary" in payload:
            rows.extend(payload["summary"])
        else:
            raise SystemExit(f"{path} must contain 'runs' or 'summary'")
    return rows


def avg(rows: List[Dict[str, Any]], key: str, default: float = 0.0) -> float:
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return mean(values) if values else default


def rows_for(rows: List[Dict[str, Any]], policy: str, budget: int) -> List[Dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("policy") == policy and int(row.get("query_budget_per_step", budget)) == budget
    ]


def convert_main(rows: List[Dict[str, Any]], budget: int, fallback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted = []
    fallback_by_method = {row["method"]: row for row in fallback}
    for policy in MAIN_ORDER:
        policy_rows = rows_for(rows, policy, budget)
        method = POLICY_TO_METHOD[policy]
        if not policy_rows:
            if method in fallback_by_method:
                converted.append(fallback_by_method[method])
            continue
        converted.append(
            {
                "method": method,
                "recall": avg(policy_rows, "event_recall"),
                "ttd_s": avg(policy_rows, "mean_time_to_detect_s"),
                "false_alarms_h": avg(policy_rows, "false_alarms_per_hour"),
                "gpu_s_h": avg(policy_rows, "gpu_seconds_per_hour"),
                "calls_event": avg(policy_rows, "vlm_calls_per_event"),
                **({"ours": True} if policy == "voi" else {}),
            }
        )
    return converted


def convert_ablation(rows: List[Dict[str, Any]], budget: int, fallback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted = []
    fallback_by_variant = {row["variant"]: row for row in fallback}
    for policy in ABLATION_ORDER:
        policy_rows = rows_for(rows, policy, budget)
        variant = ABLATION_TO_VARIANT[policy]
        if not policy_rows:
            if variant in fallback_by_variant:
                converted.append(fallback_by_variant[variant])
            continue
        converted.append(
            {
                "variant": variant,
                "recall": avg(policy_rows, "event_recall"),
                "ttd_s": avg(policy_rows, "mean_time_to_detect_s"),
                "false_alarms_h": avg(policy_rows, "false_alarms_per_hour"),
                "calls_event": avg(policy_rows, "vlm_calls_per_event"),
            }
        )
    return converted


def convert_stream_scaling(rows: List[Dict[str, Any]], budget: int, fallback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_streams: Dict[int, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if int(row.get("query_budget_per_step", budget)) != budget:
            continue
        streams = int(row.get("num_streams") or 0)
        if streams <= 0:
            continue
        by_streams[streams][row.get("policy", "")].append(row)
    if not by_streams:
        return fallback
    converted = []
    for streams in sorted(by_streams):
        group = by_streams[streams]
        converted.append(
            {
                "streams": streams,
                "uniform": avg(group.get("uniform", []), "event_recall"),
                "clip_topk": avg(group.get("clip_topk", []), "event_recall"),
                "triagevlm": avg(group.get("voi", []), "event_recall"),
                "dense_gpu_s_h": avg(group.get("dense_vlm", []), "gpu_seconds_per_hour"),
            }
        )
    return converted


def main() -> None:
    args = parse_args()
    with Path(args.template).open("r", encoding="utf-8") as f:
        template = json.load(f)
    rows = load_rows(args.aggregate)
    updated = dict(template)
    updated["main_results"] = convert_main(rows, args.main_budget, template.get("main_results", []))
    updated["ablation"] = convert_ablation(rows, args.main_budget, template.get("ablation", []))
    updated["stream_scaling"] = convert_stream_scaling(rows, args.main_budget, template.get("stream_scaling", []))

    text = json.dumps(updated, indent=2, sort_keys=False) + "\n"
    if args.write:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
