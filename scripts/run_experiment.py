#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bmvm.evaluation import run_benchmark
from bmvm.io import load_manifest, write_json
from bmvm.metrics import summarize_policy_results
from bmvm.policies import POLICY_REGISTRY
from bmvm.types import Budget, CostModel
from bmvm.vlm_cache import VLMCache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BMVM policy comparisons.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--policies", default="random,uniform,motion_topk,anomaly_topk,clip_topk,voi")
    parser.add_argument("--query-budget", type=int, default=4)
    parser.add_argument("--max-gpu-s-per-episode", type=float, default=None)
    parser.add_argument("--max-vlm-calls-per-episode", type=int, default=None)
    parser.add_argument("--cheap-scan-gpu-s", type=float, default=0.0015)
    parser.add_argument("--vlm-gpu-s", type=float, default=0.65)
    parser.add_argument("--detection-threshold", type=float, default=0.62)
    parser.add_argument("--vlm-cache", default=None, help="Optional JSONL cache with episode_id,stream_id,t_s,score records.")
    parser.add_argument("--no-simulated-vlm-fallback", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes = load_manifest(args.manifest)
    vlm_cache = VLMCache.from_jsonl(args.vlm_cache) if args.vlm_cache else None
    policies = []
    for name in [p.strip() for p in args.policies.split(",") if p.strip()]:
        if name not in POLICY_REGISTRY:
            raise SystemExit(f"Unknown policy '{name}'. Available: {', '.join(sorted(POLICY_REGISTRY))}")
        policies.append(POLICY_REGISTRY[name](query_budget=args.query_budget, seed=args.seed))

    results = run_benchmark(
        episodes=episodes,
        policies=policies,
        budget=Budget(
            query_budget_per_step=args.query_budget,
            max_gpu_s_per_episode=args.max_gpu_s_per_episode,
            max_vlm_calls_per_episode=args.max_vlm_calls_per_episode,
        ),
        cost_model=CostModel(
            cheap_scan_gpu_s_per_stream=args.cheap_scan_gpu_s,
            vlm_gpu_s_per_call=args.vlm_gpu_s,
        ),
        detection_threshold=args.detection_threshold,
        vlm_cache=vlm_cache,
        simulated_vlm_fallback=not args.no_simulated_vlm_fallback,
    )
    summary = summarize_policy_results(episodes, results)
    payload = {
        "manifest": args.manifest,
        "query_budget": args.query_budget,
        "cost_model": {
            "cheap_scan_gpu_s_per_stream": args.cheap_scan_gpu_s,
            "vlm_gpu_s_per_call": args.vlm_gpu_s,
        },
        "summary": summary,
        "vlm_cache": args.vlm_cache,
        "simulated_vlm_fallback": not args.no_simulated_vlm_fallback,
        "episodes": [
            {
                "episode_id": r.episode_id,
                "policy": r.policy,
                "total_events": r.total_events,
                "detected_events": r.detected_events,
                "false_alarms": r.false_alarms,
                "gpu_s": r.gpu_s,
                "vlm_calls": r.vlm_calls,
            }
            for r in results
        ],
    }
    write_json(args.output, payload)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
