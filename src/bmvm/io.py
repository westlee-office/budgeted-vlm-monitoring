from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .types import Episode, Event, FrameSignal


def read_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: Dict[str, Any] | List[Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def load_manifest(path: str | Path) -> List[Episode]:
    payload = read_json(path)
    episodes = []
    for raw_episode in payload["episodes"]:
        events = [
            Event(
                event_id=e["event_id"],
                stream_id=e["stream_id"],
                start_s=float(e["start_s"]),
                end_s=float(e["end_s"]),
                label=e["label"],
                severity=float(e.get("severity", 1.0)),
                description=e.get("description", ""),
            )
            for e in raw_episode.get("events", [])
        ]
        signals = {}
        for raw_signal in raw_episode.get("signals", []):
            signal = FrameSignal(
                episode_id=raw_episode["episode_id"],
                stream_id=raw_signal["stream_id"],
                t_s=float(raw_signal["t_s"]),
                motion=float(raw_signal["motion"]),
                anomaly=float(raw_signal["anomaly"]),
                clip=float(raw_signal["clip"]),
                event_ids=tuple(raw_signal.get("event_ids", [])),
            )
            signals[(signal.stream_id, round(signal.t_s, 6))] = signal
        episodes.append(
            Episode(
                episode_id=raw_episode["episode_id"],
                num_streams=int(raw_episode["num_streams"]),
                horizon_s=float(raw_episode["horizon_s"]),
                step_s=float(raw_episode["step_s"]),
                events=events,
                signals=signals,
            )
        )
    return episodes
