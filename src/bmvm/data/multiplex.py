from __future__ import annotations

import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _read_csv(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _as_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    raw = row.get(key)
    return float(raw) if raw not in (None, "") else default


def _stable_noise(seed: int, *parts: object) -> float:
    key = "::".join([str(seed), *[str(p) for p in parts]])
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def _video_id(row: Dict[str, str], index: int) -> str:
    return row.get("video_id") or row.get("id") or f"video-{index:06d}"


def _duration(row: Dict[str, str]) -> float:
    for key in ["duration_s", "duration", "length_s"]:
        if row.get(key):
            return float(row[key])
    return 0.0


def _label_is_normal(row: Dict[str, str]) -> bool:
    label = (row.get("label") or row.get("class") or row.get("split_label") or "").lower()
    return label in {"normal", "background", "none", "negative", "0"}


def _build_signal_index(signals_csv: Optional[str | Path]) -> Dict[Tuple[str, float], Dict[str, float]]:
    if not signals_csv:
        return {}
    index: Dict[Tuple[str, float], Dict[str, float]] = {}
    for row in _read_csv(signals_csv):
        video_id = row["video_id"]
        t_s = round(float(row["t_s"]), 3)
        index[(video_id, t_s)] = {
            "motion": _as_float(row, "motion", 0.0),
            "anomaly": _as_float(row, "anomaly", 0.0),
            "clip": _as_float(row, "clip", 0.0),
        }
    return index


def _nearest_signal(
    signal_index: Dict[Tuple[str, float], Dict[str, float]],
    video_id: str,
    t_s: float,
    step_s: float,
) -> Optional[Dict[str, float]]:
    if not signal_index:
        return None
    rounded = round(round(t_s / step_s) * step_s, 3)
    return signal_index.get((video_id, rounded)) or signal_index.get((video_id, round(t_s, 3)))


def _event_payload(row: Dict[str, str], event_id: str, stream_id: str, offset_s: float = 0.0) -> Dict[str, Any]:
    return {
        "event_id": event_id,
        "stream_id": stream_id,
        "start_s": _as_float(row, "start_s") + offset_s,
        "end_s": _as_float(row, "end_s") + offset_s,
        "label": row.get("label") or row.get("event_label") or "event",
        "severity": _as_float(row, "severity", 1.0),
        "description": row.get("description", ""),
    }


def build_multiplex_manifest(
    videos_csv: str | Path,
    events_csv: str | Path,
    signals_csv: Optional[str | Path] = None,
    episodes: int = 16,
    streams: int = 128,
    horizon_s: float = 1800.0,
    step_s: float = 2.0,
    event_streams_per_episode: int = 12,
    seed: int = 7,
    simulate_missing_signals: bool = False,
) -> Dict[str, Any]:
    """Create BMVM multi-stream episodes from source videos and event intervals.

    Expected `videos_csv` columns:
      - required: `video_id` or `id`
      - recommended: `path`, `duration_s`, `label`

    Expected `events_csv` columns:
      - required: `video_id`, `start_s`, `end_s`, `label`
      - optional: `event_id`, `severity`, `description`

    Optional `signals_csv` columns:
      - required: `video_id`, `t_s`
      - optional: `motion`, `anomaly`, `clip`
    """
    rng = random.Random(seed)
    raw_videos = _read_csv(videos_csv)
    raw_events = _read_csv(events_csv)
    signal_index = _build_signal_index(signals_csv)

    videos: Dict[str, Dict[str, Any]] = {}
    for idx, row in enumerate(raw_videos):
        video_id = _video_id(row, idx)
        videos[video_id] = {
            "video_id": video_id,
            "path": row.get("path", ""),
            "duration_s": _duration(row),
            "label": row.get("label") or row.get("class") or "",
            "is_normal": _label_is_normal(row),
        }

    events_by_video: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for idx, row in enumerate(raw_events):
        if not row.get("video_id"):
            raise ValueError("events_csv must include video_id")
        row = dict(row)
        row["event_id"] = row.get("event_id") or f"{row['video_id']}-event-{idx:06d}"
        events_by_video[row["video_id"]].append(row)

    event_video_ids = [video_id for video_id in events_by_video if video_id in videos]
    background_video_ids = [
        video_id
        for video_id, video in videos.items()
        if video_id not in events_by_video or video.get("is_normal")
    ]
    if not event_video_ids:
        raise ValueError("No event videos found. Check videos_csv/events_csv video_id columns.")
    if not background_video_ids:
        background_video_ids = [video_id for video_id in videos if video_id not in event_video_ids]
    if not background_video_ids:
        background_video_ids = event_video_ids

    manifest_episodes = []
    for episode_idx in range(episodes):
        episode_id = f"mux-{episode_idx:05d}"
        stream_sources = []
        used_video_ids = set()
        episode_events: List[Dict[str, Any]] = []
        timesteps = [round(i * step_s, 6) for i in range(int(horizon_s / step_s))]

        event_count = min(event_streams_per_episode, streams, len(event_video_ids))
        chosen_event_videos = rng.sample(event_video_ids, event_count)
        chosen_background_videos = []
        for _ in range(streams - event_count):
            chosen_background_videos.append(rng.choice(background_video_ids))
        chosen_videos = chosen_event_videos + chosen_background_videos
        rng.shuffle(chosen_videos)

        stream_to_video = {}
        for stream_idx, video_id in enumerate(chosen_videos):
            stream_id = f"s{stream_idx:03d}"
            stream_to_video[stream_id] = video_id
            used_video_ids.add(video_id)
            source = videos[video_id]
            stream_sources.append(
                {
                    "stream_id": stream_id,
                    "video_id": video_id,
                    "path": source.get("path", ""),
                    "duration_s": source.get("duration_s", 0.0),
                    "label": source.get("label", ""),
                }
            )
            for row in events_by_video.get(video_id, []):
                event = _event_payload(row, event_id=f"{episode_id}-{row['event_id']}", stream_id=stream_id)
                if event["start_s"] < horizon_s and event["end_s"] >= 0:
                    event["start_s"] = max(0.0, event["start_s"])
                    event["end_s"] = min(horizon_s, event["end_s"])
                    episode_events.append(event)

        event_ids_by_stream_time: Dict[Tuple[str, float], List[str]] = defaultdict(list)
        for event in episode_events:
            for t_s in timesteps:
                if event["start_s"] <= t_s <= event["end_s"]:
                    event_ids_by_stream_time[(event["stream_id"], t_s)].append(event["event_id"])

        episode_signals = []
        for t_s in timesteps:
            for stream_id, video_id in stream_to_video.items():
                source_t = t_s
                cached = _nearest_signal(signal_index, video_id, source_t, step_s)
                active_ids = event_ids_by_stream_time.get((stream_id, t_s), [])
                if cached is not None:
                    scores = dict(cached)
                elif simulate_missing_signals:
                    scores = _simulated_scores(seed, episode_id, stream_id, t_s, bool(active_ids))
                else:
                    scores = {"motion": 0.0, "anomaly": 0.0, "clip": 0.0}
                episode_signals.append(
                    {
                        "stream_id": stream_id,
                        "t_s": t_s,
                        "motion": round(float(scores["motion"]), 6),
                        "anomaly": round(float(scores["anomaly"]), 6),
                        "clip": round(float(scores["clip"]), 6),
                        "event_ids": active_ids,
                    }
                )

        manifest_episodes.append(
            {
                "episode_id": episode_id,
                "num_streams": streams,
                "horizon_s": horizon_s,
                "step_s": step_s,
                "stream_sources": stream_sources,
                "events": sorted(episode_events, key=lambda e: (e["stream_id"], e["start_s"])),
                "signals": episode_signals,
            }
        )

    return {
        "name": "multiplexed-budgeted-monitoring",
        "version": "0.2",
        "source": {
            "videos_csv": str(videos_csv),
            "events_csv": str(events_csv),
            "signals_csv": str(signals_csv) if signals_csv else None,
            "video_count": len(videos),
            "event_video_count": len(event_video_ids),
            "background_video_count": len(background_video_ids),
        },
        "episodes": manifest_episodes,
    }


def _simulated_scores(seed: int, episode_id: str, stream_id: str, t_s: float, active: bool) -> Dict[str, float]:
    motion = 0.06 + 0.18 * _stable_noise(seed, episode_id, stream_id, t_s, "motion")
    anomaly = 0.04 + 0.12 * _stable_noise(seed, episode_id, stream_id, t_s, "anomaly")
    clip = 0.04 + 0.12 * _stable_noise(seed, episode_id, stream_id, t_s, "clip")
    distractor = _stable_noise(seed, episode_id, stream_id, t_s, "distractor")
    if distractor > 0.985:
        motion += 0.45
        anomaly += 0.25
    if active:
        motion += 0.45
        anomaly += 0.60
        clip += 0.55
    return {
        "motion": min(motion, 1.0),
        "anomaly": min(anomaly, 1.0),
        "clip": min(clip, 1.0),
    }
