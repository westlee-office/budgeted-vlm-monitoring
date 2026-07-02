#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


Key = Tuple[str, str, float]
Issue = Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a BMVM VLM cache against a query-pool JSONL file.")
    parser.add_argument("--query-pool", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--min-coverage", type=float, default=1.0)
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument(
        "--require-provenance",
        action="store_true",
        help="Require non-empty model and prompt_hash fields in cache rows.",
    )
    return parser.parse_args()


def add_issue(issues: List[Issue], code: str, message: str, path: str | Path, line: int | None = None) -> None:
    issue: Issue = {"code": code, "message": message, "path": str(path)}
    if line is not None:
        issue["line"] = line
    issues.append(issue)


def read_jsonl(path: str | Path, errors: List[Issue]) -> List[Tuple[int, Dict[str, Any]]]:
    path = Path(path)
    rows: List[Tuple[int, Dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                add_issue(errors, "invalid_json", f"Invalid JSONL row: {exc}", path, line_no)
                continue
            if not isinstance(row, dict):
                add_issue(errors, "invalid_json_type", "JSONL row must be an object", path, line_no)
                continue
            rows.append((line_no, row))
    return rows


def key_from_row(row: Dict[str, Any], path: str | Path, line: int, errors: List[Issue]) -> Optional[Key]:
    missing = [column for column in ["episode_id", "stream_id", "t_s"] if column not in row or row[column] in (None, "")]
    if missing:
        add_issue(errors, "missing_key_field", f"Missing key fields: {missing}", path, line)
        return None
    try:
        return (str(row["episode_id"]), str(row["stream_id"]), round(float(row["t_s"]), 6))
    except (TypeError, ValueError) as exc:
        add_issue(errors, "invalid_key_field", f"Invalid t_s value: {exc}", path, line)
        return None


def validate_score(row: Dict[str, Any], path: str | Path, line: int, errors: List[Issue]) -> Optional[float]:
    if "score" not in row or row["score"] in (None, ""):
        add_issue(errors, "missing_score", "Cache row is missing score", path, line)
        return None
    try:
        score = float(row["score"])
    except (TypeError, ValueError) as exc:
        add_issue(errors, "invalid_score", f"Invalid score value: {exc}", path, line)
        return None
    if not (0.0 <= score <= 1.0):
        add_issue(errors, "score_out_of_range", f"Score must be in [0, 1], got {score}", path, line)
    return score


def build_key_index(
    rows: Iterable[Tuple[int, Dict[str, Any]]],
    path: str | Path,
    errors: List[Issue],
    *,
    validate_cache_rows: bool,
    require_provenance: bool,
    warnings: List[Issue],
) -> Tuple[Counter[Key], Dict[Key, int], List[float], Counter[str]]:
    counts: Counter[Key] = Counter()
    first_line: Dict[Key, int] = {}
    scores: List[float] = []
    missing_provenance: Counter[str] = Counter()
    for line, row in rows:
        key = key_from_row(row, path, line, errors)
        if key is None:
            continue
        counts[key] += 1
        first_line.setdefault(key, line)
        if validate_cache_rows:
            score = validate_score(row, path, line, errors)
            if score is not None:
                scores.append(score)
            if require_provenance:
                for column in ["model", "prompt_hash"]:
                    if not str(row.get(column, "")).strip():
                        missing_provenance[column] += 1
                        if missing_provenance[column] <= 20:
                            add_issue(errors, "missing_provenance", f"Cache row is missing '{column}'", path, line)
            else:
                for column in ["model", "prompt_hash"]:
                    if not str(row.get(column, "")).strip():
                        missing_provenance[column] += 1
                        if missing_provenance[column] == 1:
                            add_issue(warnings, "missing_provenance", f"Cache row is missing '{column}'", path, line)
    return counts, first_line, scores, missing_provenance


def numeric_summary(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {"min": min(values), "mean": sum(values) / len(values), "max": max(values)}


def validate(
    query_pool: str | Path,
    cache: str | Path,
    *,
    min_coverage: float,
    require_provenance: bool,
) -> Dict[str, Any]:
    errors: List[Issue] = []
    warnings: List[Issue] = []
    query_path = Path(query_pool)
    cache_path = Path(cache)

    query_rows = read_jsonl(query_path, errors)
    cache_rows = read_jsonl(cache_path, errors)
    query_counts, query_first_line, _, _ = build_key_index(
        query_rows,
        query_path,
        errors,
        validate_cache_rows=False,
        require_provenance=False,
        warnings=warnings,
    )
    cache_counts, cache_first_line, scores, missing_provenance = build_key_index(
        cache_rows,
        cache_path,
        errors,
        validate_cache_rows=True,
        require_provenance=require_provenance,
        warnings=warnings,
    )

    for key, count in query_counts.items():
        if count > 1:
            add_issue(warnings, "duplicate_query_key", f"Duplicate query key {key} appears {count} times", query_path, query_first_line[key])
    for key, count in cache_counts.items():
        if count > 1:
            add_issue(errors, "duplicate_cache_key", f"Duplicate cache key {key} appears {count} times", cache_path, cache_first_line[key])

    query_keys = set(query_counts)
    cache_keys = set(cache_counts)
    missing = sorted(query_keys - cache_keys)
    extra = sorted(cache_keys - query_keys)
    for key in missing[:100]:
        add_issue(errors, "missing_cache_key", f"Missing cache key {key}", cache_path)
    for key in extra[:100]:
        add_issue(warnings, "extra_cache_key", f"Cache key is not requested by query pool: {key}", cache_path, cache_first_line[key])

    coverage = (len(query_keys & cache_keys) / len(query_keys)) if query_keys else 1.0
    if coverage + 1e-12 < min_coverage:
        add_issue(
            errors,
            "coverage_below_minimum",
            f"Coverage {coverage:.6f} is below required minimum {min_coverage:.6f}",
            cache_path,
        )

    return {
        "inputs": {
            "query_pool": str(query_path),
            "cache": str(cache_path),
            "min_coverage": min_coverage,
            "require_provenance": require_provenance,
        },
        "summary": {
            "query_rows": len(query_rows),
            "cache_rows": len(cache_rows),
            "query_keys": len(query_keys),
            "cache_keys": len(cache_keys),
            "covered_keys": len(query_keys & cache_keys),
            "missing_keys": len(missing),
            "extra_keys": len(extra),
            "coverage": coverage,
            "score": numeric_summary(scores),
            "missing_provenance": dict(missing_provenance),
        },
        "errors": errors,
        "warnings": warnings,
    }


def print_report(report: Dict[str, Any], max_examples: int) -> None:
    for key, value in report["summary"].items():
        if isinstance(value, dict):
            print(f"{key}={json.dumps(value, sort_keys=True)}")
        else:
            print(f"{key}={value}")
    print(f"errors={len(report['errors'])} warnings={len(report['warnings'])}")
    for level in ["errors", "warnings"]:
        for issue in report[level][:max_examples]:
            where = issue["path"]
            if "line" in issue:
                where = f"{where}:{issue['line']}"
            print(f"{level[:-1].upper()} {issue['code']} {where} - {issue['message']}")
        remaining = len(report[level]) - max_examples
        if remaining > 0:
            print(f"{level[:-1].upper()} ... {remaining} more")


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.min_coverage <= 1.0):
        raise SystemExit("--min-coverage must be in [0, 1]")
    report = validate(
        args.query_pool,
        args.cache,
        min_coverage=args.min_coverage,
        require_provenance=args.require_provenance,
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print_report(report, args.max_examples)
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
