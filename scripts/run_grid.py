#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bmvm.evaluation import run_benchmark
from bmvm.io import load_manifest, read_json, write_json
from bmvm.metrics import summarize_policy_results
from bmvm.policies import POLICY_REGISTRY
from bmvm.types import Budget, CostModel
from bmvm.vlm_cache import VLMCache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a reproducible BMVM experiment grid.")
    parser.add_argument("--config", required=True, help="Grid JSON, e.g. configs/experiments/core_grid.json.")
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument("--output-dir", default="results/grid")
    parser.add_argument("--datasets", default=None, help="Optional comma-separated dataset override.")
    parser.add_argument("--policies", default=None, help="Optional comma-separated policy override.")
    parser.add_argument("--budgets", default=None, help="Optional comma-separated query budget override.")
    parser.add_argument("--seeds", default=None, help="Optional comma-separated seed override.")
    parser.add_argument("--stream-counts", default=None, help="Optional comma-separated stream-count override.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of runs for smoke testing.")
    parser.add_argument("--vlm-cache-dir", default=None, help="Optional directory with <dataset>.jsonl VLM cache files.")
    parser.add_argument("--no-simulated-vlm-fallback", action="store_true")
    parser.add_argument(
        "--source-commit",
        default=None,
        help="Optional provenance commit for ZIP/no-.git runs; overrides automatic git detection.",
    )
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def parse_csv_override(raw: str | None, default: Iterable[Any], cast=str) -> List[Any]:
    if not raw:
        return list(default)
    return [cast(part.strip()) for part in raw.split(",") if part.strip()]


def _unique_paths(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    unique = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _dataset_prefixes(dataset: str) -> List[str]:
    prefixes = [dataset]
    suffix = "_multistream"
    if dataset.endswith(suffix):
        prefixes.append(dataset[: -len(suffix)])
    else:
        prefixes.append(f"{dataset}{suffix}")
    return prefixes


def dataset_manifest_path(manifest_dir: Path, dataset: str, stream_count: Optional[int] = None) -> Path:
    if stream_count is None:
        candidates = [
            manifest_dir / f"{dataset}.json",
            manifest_dir / f"{dataset}_128.json",
            manifest_dir / f"{dataset}_multistream.json",
            manifest_dir / f"{dataset}_multistream_128.json",
        ]
    else:
        candidates = []
        for prefix in _dataset_prefixes(dataset):
            candidates.extend(
                [
                    manifest_dir / f"{prefix}_{stream_count}.json",
                    manifest_dir / f"{prefix}_{stream_count}_streams.json",
                    manifest_dir / f"{prefix}_s{stream_count}.json",
                ]
            )
    candidates = _unique_paths(candidates)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = ", ".join(str(candidate) for candidate in candidates)
    suffix = f" stream_count={stream_count}" if stream_count is not None else ""
    raise FileNotFoundError(f"No manifest found for dataset '{dataset}'{suffix} in {manifest_dir}. Tried: {tried}")


def dataset_cache_path(cache_dir: Path, dataset: str, stream_count: Optional[int] = None) -> Path:
    if stream_count is None:
        candidates = [cache_dir / f"{dataset}.jsonl", cache_dir / f"{dataset}_128.jsonl"]
    else:
        candidates = []
        for prefix in _dataset_prefixes(dataset):
            candidates.extend(
                [
                    cache_dir / f"{prefix}_{stream_count}.jsonl",
                    cache_dir / f"{prefix}_{stream_count}_streams.jsonl",
                    cache_dir / f"{prefix}_s{stream_count}.jsonl",
                ]
            )
        candidates.append(cache_dir / f"{dataset}.jsonl")
    for candidate in _unique_paths(candidates):
        if candidate.exists():
            return candidate
    return candidates[0]


def run_one(
    manifest_path: Path,
    dataset: str,
    policy_name: str,
    budget_value: int,
    seed: int,
    stream_count: Optional[int],
    cost_model: CostModel,
    detection_threshold: float,
    vlm_cache: Optional[VLMCache],
    vlm_cache_path: Optional[Path],
    simulated_vlm_fallback: bool,
    output_dir: Path,
    metadata_base: Dict[str, Any],
) -> Dict[str, Any]:
    episodes = load_manifest(manifest_path)
    num_streams = episodes[0].num_streams if episodes else 0
    policy = POLICY_REGISTRY[policy_name](query_budget=budget_value, seed=seed)
    results = run_benchmark(
        episodes=episodes,
        policies=[policy],
        budget=Budget(query_budget_per_step=budget_value),
        cost_model=cost_model,
        detection_threshold=detection_threshold,
        vlm_cache=vlm_cache,
        simulated_vlm_fallback=simulated_vlm_fallback,
    )
    summary = summarize_policy_results(episodes, results)
    stream_suffix = f"__s{stream_count}" if stream_count is not None else ""
    run_id = f"{dataset}{stream_suffix}__{policy_name}__b{budget_value}__seed{seed}"
    payload = {
        "run_id": run_id,
        "metadata": {
            **metadata_base,
            "dataset": dataset,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "episode_count": len(episodes),
            "num_streams": num_streams,
            "stream_count_requested": stream_count,
            "policy": policy_name,
            "query_budget_per_step": budget_value,
            "seed": seed,
            "detection_threshold": detection_threshold,
            "simulated_vlm_fallback": simulated_vlm_fallback,
            "vlm_cache": str(vlm_cache_path) if vlm_cache_path else None,
            "vlm_cache_sha256": sha256_file(vlm_cache_path) if vlm_cache_path and vlm_cache_path.exists() else None,
            "cost_model": {
                "cheap_scan_gpu_s_per_stream": cost_model.cheap_scan_gpu_s_per_stream,
                "vlm_gpu_s_per_call": cost_model.vlm_gpu_s_per_call,
            },
        },
        "summary": summary,
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
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / f"{run_id}.json", payload)
    return payload


def aggregate(runs: List[Dict[str, Any]], output_dir: Path) -> None:
    rows = []
    for run in runs:
        if not run["summary"]:
            continue
        row = dict(run["summary"][0])
        row.update(
            {
                "run_id": run["run_id"],
                "dataset": run["metadata"]["dataset"],
                "query_budget_per_step": run["metadata"]["query_budget_per_step"],
                "seed": run["metadata"]["seed"],
                "num_streams": run["metadata"].get("num_streams", 0),
                "stream_count_requested": run["metadata"].get("stream_count_requested"),
            }
        )
        rows.append(row)
    write_json(output_dir / "aggregate.json", {"runs": rows})


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    manifest_dir = Path(args.manifest_dir)
    output_dir = Path(args.output_dir)

    datasets = parse_csv_override(args.datasets, config.get("datasets", []), str)
    policies = parse_csv_override(args.policies, config.get("policies", []), str)
    budgets = parse_csv_override(args.budgets, config.get("query_budgets_per_step", [4]), int)
    seeds = parse_csv_override(args.seeds, config.get("seeds", [7]), int)
    stream_counts = parse_csv_override(args.stream_counts, config.get("stream_counts", []), int)
    stream_count_values: List[Optional[int]] = stream_counts if stream_counts else [None]
    if config.get("dense_vlm_reference") and "dense_vlm" not in policies:
        policies.append("dense_vlm")

    for policy_name in policies:
        if policy_name not in POLICY_REGISTRY:
            raise SystemExit(f"Unknown policy '{policy_name}'. Available: {', '.join(sorted(POLICY_REGISTRY))}")

    metadata_base = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "git_commit": args.source_commit or git_commit(),
        "config": args.config,
    }
    cost_model = CostModel(
        cheap_scan_gpu_s_per_stream=float(config.get("cheap_scan_gpu_s_per_stream", 0.0015)),
        vlm_gpu_s_per_call=float(config.get("vlm_gpu_s_per_call", 0.65)),
    )
    detection_threshold = float(config.get("detection_threshold", 0.62))

    runs = []
    planned = [
        (dataset, stream_count, policy_name, budget_value, seed)
        for dataset in datasets
        for stream_count in stream_count_values
        for policy_name in policies
        for budget_value in budgets
        for seed in seeds
    ]
    if args.limit is not None:
        planned = planned[: args.limit]

    for dataset, stream_count, policy_name, budget_value, seed in planned:
        manifest_path = dataset_manifest_path(manifest_dir, dataset, stream_count)
        vlm_cache = None
        if args.vlm_cache_dir:
            cache_path = dataset_cache_path(Path(args.vlm_cache_dir), dataset, stream_count)
            if cache_path.exists():
                vlm_cache = VLMCache.from_jsonl(cache_path)
            elif args.no_simulated_vlm_fallback:
                raise FileNotFoundError(f"Missing VLM cache: {cache_path}")
        else:
            cache_path = None
        stream_part = f" streams={stream_count}" if stream_count is not None else ""
        print(f"Running dataset={dataset}{stream_part} policy={policy_name} budget={budget_value} seed={seed}")
        runs.append(
            run_one(
                manifest_path=manifest_path,
                dataset=dataset,
                policy_name=policy_name,
                budget_value=budget_value,
                seed=seed,
                stream_count=stream_count,
                cost_model=cost_model,
                detection_threshold=detection_threshold,
                vlm_cache=vlm_cache,
                vlm_cache_path=cache_path if vlm_cache is not None else None,
                simulated_vlm_fallback=not args.no_simulated_vlm_fallback,
                output_dir=output_dir,
                metadata_base=metadata_base,
            )
        )
    aggregate(runs, output_dir)
    print(f"Wrote {len(runs)} runs to {output_dir}")


if __name__ == "__main__":
    main()
