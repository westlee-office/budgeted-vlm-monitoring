from __future__ import annotations

from typing import Iterable, List, Optional

from .policies.base import Policy
from .policies.voi import ValueOfInformationPolicy
from .types import Budget, CostModel, Episode, EpisodeResult, Query, StepContext, StepResult
from .vlm_cache import VLMCache


def _vlm_verifier_score(cheap_score: float, active_event_count: int) -> float:
    if active_event_count:
        return min(1.0, 0.55 + 0.45 * cheap_score)
    return max(0.0, 0.10 + 0.35 * cheap_score)


def run_episode(
    episode: Episode,
    policy: Policy,
    budget: Budget,
    cost_model: CostModel,
    detection_threshold: float = 0.62,
    vlm_cache: Optional[VLMCache] = None,
    simulated_vlm_fallback: bool = True,
) -> EpisodeResult:
    policy.reset()
    detected = {}
    false_alarms = 0
    total_gpu_s = 0.0
    total_calls = 0
    steps: List[StepResult] = []

    for t_s in episode.timesteps:
        signals = [episode.signal(stream_id, t_s) for stream_id in episode.stream_ids]
        cheap_gpu_s = len(signals) * cost_model.cheap_scan_gpu_s_per_stream
        remaining_gpu_s = None
        remaining_calls = None
        if budget.max_gpu_s_per_episode is not None:
            remaining_gpu_s = max(0.0, budget.max_gpu_s_per_episode - total_gpu_s - cheap_gpu_s)
        if budget.max_vlm_calls_per_episode is not None:
            remaining_calls = max(0, budget.max_vlm_calls_per_episode - total_calls)

        context = StepContext(
            episode=episode,
            t_s=t_s,
            signals=signals,
            remaining_gpu_s=remaining_gpu_s,
            remaining_vlm_calls=remaining_calls,
        )
        queries = policy.select(context)
        if remaining_calls is not None:
            queries = queries[:remaining_calls]
        if remaining_gpu_s is not None:
            max_by_gpu = int(remaining_gpu_s // cost_model.vlm_gpu_s_per_call)
            queries = queries[:max_by_gpu]

        step_detected = []
        step_false_alarms = 0
        for query in queries:
            signal = episode.signal(query.stream_id, t_s)
            active_events = episode.active_events(query.stream_id, t_s)
            cached = vlm_cache.get(episode.episode_id, query.stream_id, t_s) if vlm_cache else None
            if cached is not None:
                verifier_score = cached.score
            elif simulated_vlm_fallback:
                verifier_score = _vlm_verifier_score(signal.cheap_score, len(active_events))
            else:
                verifier_score = 0.0
            is_positive = verifier_score >= detection_threshold
            if active_events and is_positive:
                for event in active_events:
                    if event.event_id not in detected:
                        detected[event.event_id] = t_s
                        step_detected.append(event.event_id)
                _observe(policy, query.stream_id, detected=True, false_alarm=False)
            elif is_positive:
                false_alarms += 1
                step_false_alarms += 1
                _observe(policy, query.stream_id, detected=False, false_alarm=True)
            else:
                _observe(policy, query.stream_id, detected=False, false_alarm=False)

        gpu_s = cheap_gpu_s + len(queries) * cost_model.vlm_gpu_s_per_call
        calls = len(queries) * cost_model.vlm_calls_per_query
        total_gpu_s += gpu_s
        total_calls += calls
        steps.append(
            StepResult(
                t_s=t_s,
                queries=queries,
                detected_event_ids=step_detected,
                false_alarms=step_false_alarms,
                gpu_s=gpu_s,
                vlm_calls=calls,
            )
        )

    return EpisodeResult(
        episode_id=episode.episode_id,
        policy=policy.name,
        total_events=len(episode.events),
        detected_events=detected,
        false_alarms=false_alarms,
        gpu_s=total_gpu_s,
        vlm_calls=total_calls,
        steps=steps,
    )


def _observe(policy: Policy, stream_id: str, detected: bool, false_alarm: bool) -> None:
    if isinstance(policy, ValueOfInformationPolicy):
        policy.observe_query_result(stream_id, detected=detected, false_alarm=false_alarm)


def run_benchmark(
    episodes: Iterable[Episode],
    policies: Iterable[Policy],
    budget: Budget,
    cost_model: CostModel,
    detection_threshold: float = 0.62,
    vlm_cache: Optional[VLMCache] = None,
    simulated_vlm_fallback: bool = True,
) -> List[EpisodeResult]:
    results = []
    for policy in policies:
        for episode in episodes:
            results.append(
                run_episode(
                    episode=episode,
                    policy=policy,
                    budget=budget,
                    cost_model=cost_model,
                    detection_threshold=detection_threshold,
                    vlm_cache=vlm_cache,
                    simulated_vlm_fallback=simulated_vlm_fallback,
                )
            )
    return results
