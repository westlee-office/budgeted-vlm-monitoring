#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


Issue = Dict[str, Any]
Row = Dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate normalized dataset CSVs before BMVM manifest generation."
    )
    parser.add_argument("--schema", choices=["source", "episode"], default="source")
    parser.add_argument("--videos", default=None, help="Source-level videos.csv. Required for --schema source.")
    parser.add_argument("--events", required=True, help="events.csv path.")
    parser.add_argument("--signals", default=None, help="Optional signals.csv path.")
    parser.add_argument(
        "--path-root",
        default=None,
        help="Root for relative video paths. Defaults to the videos.csv directory in source mode.",
    )
    parser.add_argument("--check-paths", action="store_true", help="Fail if video paths do not exist.")
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    parser.add_argument("--max-examples", type=int, default=20, help="Max issue examples printed per level.")
    return parser.parse_args()


def read_csv(path: str | Path) -> Tuple[List[str], List[Row]]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows: List[Row] = []
        for line_no, row in enumerate(reader, start=2):
            cleaned = {key: (value if value is not None else "") for key, value in row.items() if key is not None}
            cleaned["_line"] = str(line_no)
            rows.append(cleaned)
    return fieldnames, rows


def add_issue(issues: List[Issue], code: str, message: str, path: str | Path, line: str | int | None = None) -> None:
    issue: Issue = {"code": code, "message": message, "path": str(path)}
    if line not in (None, ""):
        issue["line"] = int(line)
    issues.append(issue)


def missing_columns(fieldnames: Sequence[str], required: Iterable[str]) -> List[str]:
    present = set(fieldnames)
    return [column for column in required if column not in present]


def first_present(fieldnames: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    present = set(fieldnames)
    for candidate in candidates:
        if candidate in present:
            return candidate
    return None


def parse_float(
    raw: str,
    errors: List[Issue],
    path: str | Path,
    line: str | int,
    column: str,
    *,
    required: bool = True,
) -> Optional[float]:
    if raw == "":
        if required:
            add_issue(errors, "missing_numeric", f"Missing numeric value in column '{column}'", path, line)
        return None
    try:
        value = float(raw)
    except ValueError:
        add_issue(errors, "invalid_numeric", f"Invalid numeric value in column '{column}': {raw!r}", path, line)
        return None
    return value


def percentile(values: Sequence[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


def numeric_summary(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"min": None, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "min": min(values),
        "mean": mean(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "max": max(values),
    }


def top_counts(counter: Counter[str], limit: int = 20) -> Dict[str, int]:
    return dict(counter.most_common(limit))


def resolve_video_path(raw_path: str, path_root: Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else path_root / path


def validate_source(
    videos_csv: str | Path,
    events_csv: str | Path,
    signals_csv: str | Path | None,
    *,
    path_root: str | Path | None,
    check_paths: bool,
) -> Dict[str, Any]:
    videos_path = Path(videos_csv)
    events_path = Path(events_csv)
    signals_path = Path(signals_csv) if signals_csv else None
    root = Path(path_root) if path_root else videos_path.parent

    errors: List[Issue] = []
    warnings: List[Issue] = []
    videos_fields, video_rows = read_csv(videos_path)
    events_fields, event_rows = read_csv(events_path)

    video_id_column = first_present(videos_fields, ["video_id", "id"])
    if video_id_column is None:
        add_issue(errors, "missing_column", "videos.csv must include 'video_id' or 'id'", videos_path)
        video_id_column = "video_id"
    if "path" not in videos_fields:
        add_issue(warnings, "missing_recommended_column", "videos.csv should include 'path'", videos_path)
    if not first_present(videos_fields, ["duration_s", "duration", "length_s"]):
        add_issue(warnings, "missing_recommended_column", "videos.csv should include duration_s/duration/length_s", videos_path)

    videos: Dict[str, Dict[str, Any]] = {}
    duplicate_video_ids: Counter[str] = Counter()
    labels = Counter()
    splits_by_path: Dict[str, set[str]] = defaultdict(set)
    durations: List[float] = []
    missing_paths = 0

    for row in video_rows:
        line = row["_line"]
        video_id = row.get(video_id_column, "").strip()
        if not video_id:
            add_issue(errors, "missing_video_id", f"Missing {video_id_column}", videos_path, line)
            continue
        if video_id in videos:
            duplicate_video_ids[video_id] += 1
            add_issue(errors, "duplicate_video_id", f"Duplicate video_id '{video_id}'", videos_path, line)
            continue

        duration = None
        duration_column = first_present(row.keys(), ["duration_s", "duration", "length_s"])
        if duration_column:
            duration = parse_float(row.get(duration_column, ""), errors, videos_path, line, duration_column, required=False)
            if duration is not None:
                if duration <= 0:
                    add_issue(errors, "invalid_duration", f"Video '{video_id}' has non-positive duration {duration}", videos_path, line)
                else:
                    durations.append(duration)

        label = (row.get("label") or row.get("class") or "").strip()
        labels[label or "<missing>"] += 1
        split = (row.get("split") or row.get("subset") or "").strip()
        raw_path = (row.get("path") or "").strip()
        if raw_path:
            if split:
                splits_by_path[raw_path].add(split)
            if check_paths and not resolve_video_path(raw_path, root).exists():
                add_issue(errors, "missing_video_path", f"Video path does not exist: {raw_path}", videos_path, line)
        else:
            missing_paths += 1
            if check_paths:
                add_issue(errors, "missing_video_path", f"Video '{video_id}' has empty path", videos_path, line)

        videos[video_id] = {
            "duration_s": duration,
            "label": label,
            "split": split,
            "path": raw_path,
        }

    for raw_path, splits in sorted(splits_by_path.items()):
        if len(splits) > 1:
            add_issue(
                errors,
                "split_leakage",
                f"Path appears in multiple splits: {raw_path} -> {sorted(splits)}",
                videos_path,
            )

    missing = missing_columns(events_fields, ["video_id", "start_s", "end_s", "label"])
    for column in missing:
        add_issue(errors, "missing_column", f"events.csv is missing required column '{column}'", events_path)

    event_ids = Counter()
    event_labels = Counter()
    event_durations: List[float] = []
    events_by_video = Counter()
    if not missing:
        for idx, row in enumerate(event_rows):
            line = row["_line"]
            video_id = row.get("video_id", "").strip()
            if not video_id:
                add_issue(errors, "missing_event_video_id", "Event has empty video_id", events_path, line)
                continue
            if video_id not in videos:
                add_issue(errors, "unknown_event_video_id", f"Event references unknown video_id '{video_id}'", events_path, line)
            start = parse_float(row.get("start_s", ""), errors, events_path, line, "start_s")
            end = parse_float(row.get("end_s", ""), errors, events_path, line, "end_s")
            if start is None or end is None:
                continue
            if start < 0:
                add_issue(errors, "negative_event_start", f"Event starts before zero: {start}", events_path, line)
            if end <= start:
                add_issue(errors, "non_positive_event_interval", f"Event end_s must be > start_s: {start} -> {end}", events_path, line)
            else:
                event_durations.append(end - start)
            known_duration = videos.get(video_id, {}).get("duration_s")
            if known_duration and end > float(known_duration) + 1e-3:
                add_issue(
                    errors,
                    "event_exceeds_video_duration",
                    f"Event ends at {end}s but video duration is {known_duration}s for '{video_id}'",
                    events_path,
                    line,
                )
            event_id = (row.get("event_id") or "").strip()
            if event_id:
                event_ids[event_id] += 1
                if event_ids[event_id] > 1:
                    add_issue(errors, "duplicate_event_id", f"Duplicate event_id '{event_id}'", events_path, line)
            severity_raw = row.get("severity", "")
            if severity_raw:
                severity = parse_float(severity_raw, errors, events_path, line, "severity", required=False)
                if severity is not None and not (0.0 <= severity <= 1.0):
                    add_issue(warnings, "severity_out_of_range", f"Severity is outside [0, 1]: {severity}", events_path, line)
            event_labels[(row.get("label") or "<missing>").strip() or "<missing>"] += 1
            events_by_video[video_id] += 1

    signal_summary: Dict[str, Any] = {"count": 0, "score_coverage": {}, "t_s": numeric_summary([])}
    if signals_path:
        signals_fields, signal_rows = read_csv(signals_path)
        for column in missing_columns(signals_fields, ["video_id", "t_s"]):
            add_issue(errors, "missing_column", f"signals.csv is missing required column '{column}'", signals_path)
        score_columns = [column for column in ["motion", "anomaly", "clip"] if column in signals_fields]
        score_coverage = Counter()
        signal_times: List[float] = []
        if "video_id" in signals_fields and "t_s" in signals_fields:
            for row in signal_rows:
                line = row["_line"]
                video_id = row.get("video_id", "").strip()
                if not video_id:
                    add_issue(errors, "missing_signal_video_id", "Signal has empty video_id", signals_path, line)
                    continue
                if video_id not in videos:
                    add_issue(errors, "unknown_signal_video_id", f"Signal references unknown video_id '{video_id}'", signals_path, line)
                t_s = parse_float(row.get("t_s", ""), errors, signals_path, line, "t_s")
                if t_s is None:
                    continue
                if t_s < 0:
                    add_issue(errors, "negative_signal_time", f"Signal t_s is negative: {t_s}", signals_path, line)
                known_duration = videos.get(video_id, {}).get("duration_s")
                if known_duration and t_s > float(known_duration) + 1e-3:
                    add_issue(
                        errors,
                        "signal_exceeds_video_duration",
                        f"Signal t_s={t_s}s exceeds duration {known_duration}s for '{video_id}'",
                        signals_path,
                        line,
                    )
                signal_times.append(t_s)
                for column in score_columns:
                    if row.get(column, "") == "":
                        continue
                    score_coverage[column] += 1
                    score = parse_float(row[column], errors, signals_path, line, column, required=False)
                    if score is not None and not (0.0 <= score <= 1.0):
                        add_issue(warnings, "score_out_of_range", f"{column} score is outside [0, 1]: {score}", signals_path, line)
        signal_summary = {
            "count": len(signal_rows),
            "score_coverage": dict(score_coverage),
            "t_s": numeric_summary(signal_times),
        }

    event_video_count = len(events_by_video)
    background_video_count = max(0, len(videos) - event_video_count)
    return {
        "schema": "source",
        "inputs": {
            "videos": str(videos_path),
            "events": str(events_path),
            "signals": str(signals_path) if signals_path else None,
            "path_root": str(root),
            "check_paths": check_paths,
        },
        "summary": {
            "video_count": len(videos),
            "event_count": len(event_rows),
            "event_video_count": event_video_count,
            "background_video_count": background_video_count,
            "missing_path_count": missing_paths,
            "duration_s": numeric_summary(durations),
            "event_duration_s": numeric_summary(event_durations),
            "labels": top_counts(labels),
            "event_labels": top_counts(event_labels),
            "signals": signal_summary,
        },
        "errors": errors,
        "warnings": warnings,
    }


def validate_episode(events_csv: str | Path, signals_csv: str | Path | None) -> Dict[str, Any]:
    events_path = Path(events_csv)
    signals_path = Path(signals_csv) if signals_csv else None
    errors: List[Issue] = []
    warnings: List[Issue] = []
    events_fields, event_rows = read_csv(events_path)
    event_labels = Counter()
    event_durations: List[float] = []
    episodes = set()
    streams_by_episode: Dict[str, set[str]] = defaultdict(set)

    for column in missing_columns(events_fields, ["episode_id", "stream_id", "start_s", "end_s", "label"]):
        add_issue(errors, "missing_column", f"events.csv is missing required column '{column}'", events_path)
    if not errors:
        event_ids = Counter()
        for row in event_rows:
            line = row["_line"]
            episode_id = row.get("episode_id", "").strip()
            stream_id = row.get("stream_id", "").strip()
            if not episode_id:
                add_issue(errors, "missing_episode_id", "Event has empty episode_id", events_path, line)
            if not stream_id:
                add_issue(errors, "missing_stream_id", "Event has empty stream_id", events_path, line)
            start = parse_float(row.get("start_s", ""), errors, events_path, line, "start_s")
            end = parse_float(row.get("end_s", ""), errors, events_path, line, "end_s")
            if start is not None and start < 0:
                add_issue(errors, "negative_event_start", f"Event starts before zero: {start}", events_path, line)
            if start is not None and end is not None:
                if end <= start:
                    add_issue(errors, "non_positive_event_interval", f"Event end_s must be > start_s: {start} -> {end}", events_path, line)
                else:
                    event_durations.append(end - start)
            event_id = (row.get("event_id") or "").strip()
            if event_id:
                event_ids[event_id] += 1
                if event_ids[event_id] > 1:
                    add_issue(errors, "duplicate_event_id", f"Duplicate event_id '{event_id}'", events_path, line)
            event_labels[(row.get("label") or "<missing>").strip() or "<missing>"] += 1
            if episode_id and stream_id:
                episodes.add(episode_id)
                streams_by_episode[episode_id].add(stream_id)

    signal_summary: Dict[str, Any] = {"count": 0, "score_coverage": {}, "t_s": numeric_summary([])}
    if signals_path:
        signals_fields, signal_rows = read_csv(signals_path)
        for column in missing_columns(signals_fields, ["episode_id", "stream_id", "t_s"]):
            add_issue(errors, "missing_column", f"signals.csv is missing required column '{column}'", signals_path)
        score_columns = [column for column in ["motion", "anomaly", "clip"] if column in signals_fields]
        score_coverage = Counter()
        signal_times: List[float] = []
        if not missing_columns(signals_fields, ["episode_id", "stream_id", "t_s"]):
            for row in signal_rows:
                line = row["_line"]
                episode_id = row.get("episode_id", "").strip()
                stream_id = row.get("stream_id", "").strip()
                if not episode_id:
                    add_issue(errors, "missing_episode_id", "Signal has empty episode_id", signals_path, line)
                if not stream_id:
                    add_issue(errors, "missing_stream_id", "Signal has empty stream_id", signals_path, line)
                t_s = parse_float(row.get("t_s", ""), errors, signals_path, line, "t_s")
                if t_s is not None:
                    if t_s < 0:
                        add_issue(errors, "negative_signal_time", f"Signal t_s is negative: {t_s}", signals_path, line)
                    signal_times.append(t_s)
                for column in score_columns:
                    if row.get(column, "") == "":
                        continue
                    score_coverage[column] += 1
                    score = parse_float(row[column], errors, signals_path, line, column, required=False)
                    if score is not None and not (0.0 <= score <= 1.0):
                        add_issue(warnings, "score_out_of_range", f"{column} score is outside [0, 1]: {score}", signals_path, line)
                if episode_id and stream_id:
                    episodes.add(episode_id)
                    streams_by_episode[episode_id].add(stream_id)
        signal_summary = {
            "count": len(signal_rows),
            "score_coverage": dict(score_coverage),
            "t_s": numeric_summary(signal_times),
        }

    return {
        "schema": "episode",
        "inputs": {
            "events": str(events_path),
            "signals": str(signals_path) if signals_path else None,
        },
        "summary": {
            "episode_count": len(episodes),
            "max_streams_per_episode": max((len(v) for v in streams_by_episode.values()), default=0),
            "event_count": len(event_rows),
            "event_duration_s": numeric_summary(event_durations),
            "event_labels": top_counts(event_labels),
            "signals": signal_summary,
        },
        "errors": errors,
        "warnings": warnings,
    }


def print_report(report: Dict[str, Any], max_examples: int) -> None:
    summary = report["summary"]
    print(f"schema={report['schema']}")
    for key, value in summary.items():
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
    if args.schema == "source":
        if not args.videos:
            raise SystemExit("--videos is required for --schema source")
        report = validate_source(
            args.videos,
            args.events,
            args.signals,
            path_root=args.path_root,
            check_paths=args.check_paths,
        )
    else:
        report = validate_episode(args.events, args.signals)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print_report(report, args.max_examples)
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
