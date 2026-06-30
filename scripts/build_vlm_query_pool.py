#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bmvm.io import load_manifest, read_json
from bmvm.policies import POLICY_REGISTRY
from bmvm.types import StepContext


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a JSONL pool of VLM queries from a BMVM manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["all", "topk", "policies"], default="topk")
    parser.add_argument("--topk", type=int, default=8, help="Per-timestep stream count for topk mode.")
    parser.add_argument("--policies", default="anomaly_topk,clip_topk,voi")
    parser.add_argument("--query-budget", type=int, default=4)
    parser.add_argument("--prompt", default="Detect and summarize safety-critical incidents in this video segment.")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def stream_source_index(manifest_path: str | Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    raw = read_json(manifest_path)
    index: Dict[Tuple[str, str], Dict[str, str]] = {}
    for episode in raw.get("episodes", []):
        episode_id = episode["episode_id"]
        for source in episode.get("stream_sources", []):
            index[(episode_id, source["stream_id"])] = source
    return index


def emit_record(episode_id: str, stream_id: str, t_s: float, source: Dict[str, str], prompt: str) -> Dict[str, object]:
    return {
        "episode_id": episode_id,
        "stream_id": stream_id,
        "t_s": t_s,
        "video_id": source.get("video_id", ""),
        "path": source.get("path", ""),
        "prompt": prompt,
    }


def main() -> None:
    args = parse_args()
    episodes = load_manifest(args.manifest)
    sources = stream_source_index(args.manifest)
    records: Dict[Tuple[str, str, float], Dict[str, object]] = {}

    for episode in episodes:
        if args.mode == "policies":
            policies = []
            for name in [p.strip() for p in args.policies.split(",") if p.strip()]:
                if name not in POLICY_REGISTRY:
                    raise SystemExit(f"Unknown policy '{name}'")
                policy = POLICY_REGISTRY[name](query_budget=args.query_budget, seed=args.seed)
                policy.reset()
                policies.append(policy)
        else:
            policies = []

        for t_s in episode.timesteps:
            signals = [episode.signal(stream_id, t_s) for stream_id in episode.stream_ids]
            selected_streams: List[str]
            if args.mode == "all":
                selected_streams = [signal.stream_id for signal in signals]
            elif args.mode == "topk":
                ranked = sorted(signals, key=lambda signal: signal.cheap_score, reverse=True)
                selected_streams = [signal.stream_id for signal in ranked[: args.topk]]
            else:
                selected_streams = []
                context = StepContext(
                    episode=episode,
                    t_s=t_s,
                    signals=signals,
                    remaining_gpu_s=None,
                    remaining_vlm_calls=None,
                )
                for policy in policies:
                    selected_streams.extend(query.stream_id for query in policy.select(context))

            for stream_id in selected_streams:
                key = (episode.episode_id, stream_id, round(t_s, 6))
                records[key] = emit_record(
                    episode_id=episode.episode_id,
                    stream_id=stream_id,
                    t_s=t_s,
                    source=sources.get((episode.episode_id, stream_id), {}),
                    prompt=args.prompt,
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for key in sorted(records):
            f.write(json.dumps(records[key], sort_keys=True) + "\n")
    print(f"Wrote {output} with {len(records)} unique queries")


if __name__ == "__main__":
    main()
