from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List


EVENT_LABELS = [
    "person enters restricted area",
    "vehicle stops in forbidden zone",
    "fight or assault",
    "person falls",
    "abandoned object",
    "fire or smoke",
]


def _stable_noise(seed: int, *parts: object) -> float:
    key = "::".join([str(seed), *[str(p) for p in parts]])
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return value


def make_synthetic_manifest(
    episodes: int = 8,
    streams: int = 32,
    horizon_s: int = 300,
    step_s: float = 5.0,
    events_per_episode: int = 8,
    seed: int = 7,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    manifest = {
        "name": "synthetic-budgeted-monitoring",
        "version": "0.1",
        "description": "Synthetic multi-stream episodes for smoke tests.",
        "episodes": [],
    }

    for episode_idx in range(episodes):
        episode_id = f"synthetic-{episode_idx:04d}"
        stream_ids = [f"s{i:03d}" for i in range(streams)]
        raw_events: List[Dict[str, Any]] = []
        for event_idx in range(events_per_episode):
            stream_id = rng.choice(stream_ids)
            duration = rng.choice([10, 15, 20, 30])
            start = rng.randrange(10, max(20, horizon_s - duration - 10), int(step_s))
            label = rng.choice(EVENT_LABELS)
            raw_events.append(
                {
                    "event_id": f"{episode_id}-e{event_idx:03d}",
                    "stream_id": stream_id,
                    "start_s": float(start),
                    "end_s": float(start + duration),
                    "label": label,
                    "severity": round(rng.uniform(0.6, 1.0), 3),
                    "description": f"{label} in stream {stream_id}.",
                }
            )

        raw_signals: List[Dict[str, Any]] = []
        timesteps = [round(i * step_s, 6) for i in range(int(horizon_s / step_s))]
        for t_s in timesteps:
            for stream_id in stream_ids:
                active = [
                    e
                    for e in raw_events
                    if e["stream_id"] == stream_id and e["start_s"] <= t_s <= e["end_s"]
                ]
                distractor = _stable_noise(seed, episode_id, stream_id, t_s, "distractor")
                base_motion = 0.08 + 0.20 * _stable_noise(seed, episode_id, stream_id, t_s, "motion")
                base_anomaly = 0.04 + 0.12 * _stable_noise(seed, episode_id, stream_id, t_s, "anomaly")
                base_clip = 0.05 + 0.10 * _stable_noise(seed, episode_id, stream_id, t_s, "clip")
                if distractor > 0.985:
                    base_motion += 0.45
                    base_anomaly += 0.28
                if active:
                    severity = max(e["severity"] for e in active)
                    base_motion += 0.42 * severity
                    base_anomaly += 0.62 * severity
                    base_clip += 0.55 * severity
                raw_signals.append(
                    {
                        "stream_id": stream_id,
                        "t_s": t_s,
                        "motion": round(min(base_motion, 1.0), 4),
                        "anomaly": round(min(base_anomaly, 1.0), 4),
                        "clip": round(min(base_clip, 1.0), 4),
                        "event_ids": [e["event_id"] for e in active],
                    }
                )

        manifest["episodes"].append(
            {
                "episode_id": episode_id,
                "num_streams": streams,
                "horizon_s": float(horizon_s),
                "step_s": float(step_s),
                "events": raw_events,
                "signals": raw_signals,
            }
        )
    return manifest
