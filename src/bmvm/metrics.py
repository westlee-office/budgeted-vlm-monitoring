from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Dict, Iterable, List

from .types import Episode, EpisodeResult


def summarize_policy_results(episodes: List[Episode], results: Iterable[EpisodeResult]) -> List[Dict[str, float | str]]:
    episode_by_id = {episode.episode_id: episode for episode in episodes}
    grouped: Dict[str, List[EpisodeResult]] = defaultdict(list)
    for result in results:
        grouped[result.policy].append(result)

    summaries = []
    for policy, policy_results in sorted(grouped.items()):
        total_events = sum(r.total_events for r in policy_results)
        detected = sum(len(r.detected_events) for r in policy_results)
        total_false = sum(r.false_alarms for r in policy_results)
        total_gpu = sum(r.gpu_s for r in policy_results)
        total_calls = sum(r.vlm_calls for r in policy_results)
        total_horizon = sum(episode_by_id[r.episode_id].horizon_s for r in policy_results)
        delays = []
        for result in policy_results:
            episode = episode_by_id[result.episode_id]
            event_by_id = {event.event_id: event for event in episode.events}
            for event_id, detected_t in result.detected_events.items():
                delays.append(max(0.0, detected_t - event_by_id[event_id].start_s))
        summaries.append(
            {
                "policy": policy,
                "event_recall": detected / total_events if total_events else 0.0,
                "mean_time_to_detect_s": mean(delays) if delays else float("inf"),
                "false_alarms_per_hour": total_false / max(total_horizon / 3600.0, 1e-9),
                "gpu_seconds_per_hour": total_gpu / max(total_horizon / 3600.0, 1e-9),
                "vlm_calls_per_event": total_calls / max(total_events, 1),
                "detected_events": detected,
                "total_events": total_events,
                "vlm_calls": total_calls,
            }
        )
    return summaries


def markdown_table(rows: List[Dict[str, float | str]]) -> str:
    headers = [
        "policy",
        "event_recall",
        "mean_time_to_detect_s",
        "false_alarms_per_hour",
        "gpu_seconds_per_hour",
        "vlm_calls_per_event",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        cells = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                if value == float("inf"):
                    cells.append("inf")
                else:
                    cells.append(f"{value:.3f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
