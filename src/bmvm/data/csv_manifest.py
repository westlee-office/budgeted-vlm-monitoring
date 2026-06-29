from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _read_csv(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _split_event_ids(raw: str | None) -> List[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]


def build_manifest_from_csv(
    events_csv: str | Path,
    signals_csv: str | Path,
    default_step_s: float = 2.0,
    default_horizon_s: float | None = None,
) -> Dict[str, Any]:
    events = _read_csv(events_csv)
    signals = _read_csv(signals_csv)
    by_episode_events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_episode_signals: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    stream_ids_by_episode: Dict[str, set[str]] = defaultdict(set)

    for idx, row in enumerate(events):
        episode_id = row["episode_id"]
        stream_id = row["stream_id"]
        event_id = row.get("event_id") or f"{episode_id}-event-{idx:06d}"
        stream_ids_by_episode[episode_id].add(stream_id)
        by_episode_events[episode_id].append(
            {
                "event_id": event_id,
                "stream_id": stream_id,
                "start_s": float(row["start_s"]),
                "end_s": float(row["end_s"]),
                "label": row["label"],
                "severity": float(row.get("severity") or 1.0),
                "description": row.get("description", ""),
            }
        )

    for row in signals:
        episode_id = row["episode_id"]
        stream_id = row["stream_id"]
        stream_ids_by_episode[episode_id].add(stream_id)
        by_episode_signals[episode_id].append(
            {
                "stream_id": stream_id,
                "t_s": float(row["t_s"]),
                "motion": float(row.get("motion") or 0.0),
                "anomaly": float(row.get("anomaly") or 0.0),
                "clip": float(row.get("clip") or 0.0),
                "event_ids": _split_event_ids(row.get("event_ids")),
            }
        )

    episodes = []
    for episode_id in sorted(set(by_episode_events) | set(by_episode_signals)):
        episode_signals = by_episode_signals[episode_id]
        if default_horizon_s is not None:
            horizon_s = float(default_horizon_s)
        elif episode_signals:
            horizon_s = max(float(s["t_s"]) for s in episode_signals) + default_step_s
        else:
            horizon_s = max(float(e["end_s"]) for e in by_episode_events[episode_id]) + default_step_s
        episodes.append(
            {
                "episode_id": episode_id,
                "num_streams": len(stream_ids_by_episode[episode_id]),
                "horizon_s": horizon_s,
                "step_s": default_step_s,
                "events": sorted(by_episode_events[episode_id], key=lambda e: (e["stream_id"], e["start_s"])),
                "signals": sorted(episode_signals, key=lambda s: (s["t_s"], s["stream_id"])),
            }
        )

    return {
        "name": "csv-built-budgeted-monitoring",
        "version": "0.1",
        "episodes": episodes,
    }
