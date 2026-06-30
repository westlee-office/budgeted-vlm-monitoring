from __future__ import annotations

from typing import Dict, List

from ..types import PolicyState, Query, StepContext
from .base import Policy


class ValueOfInformationPolicy(Policy):
    """A lightweight value-of-information scheduler.

    The score estimates the marginal value of spending a VLM call on a stream.
    It combines cheap visual cues, stream-level memory, recency, and a cooldown
    penalty to avoid repeatedly querying the same ambiguous stream.
    """

    name = "voi"
    weight_anomaly = 0.36
    weight_motion = 0.20
    weight_clip = 0.20
    weight_memory = 0.18
    weight_uncertainty = 0.08
    recency_bonus_max = 0.08
    cooldown_penalty = 0.25
    memory_enabled = True

    def __init__(self, query_budget: int, seed: int = 0) -> None:
        super().__init__(query_budget=query_budget, seed=seed)
        self.state = PolicyState()

    def reset(self) -> None:
        self.state = PolicyState()

    def select(self, context: StepContext) -> List[Query]:
        self.state.decay()
        scored = []
        for signal in context.signals:
            memory = self.state.risk_memory.get(signal.stream_id, 0.0)
            last_seen = self.state.last_seen.get(signal.stream_id)
            recency_bonus = self.recency_bonus_max if last_seen is None else min((context.t_s - last_seen) / 120.0, 1.0) * self.recency_bonus_max
            cooldown = 1.0 if self.state.cooldown_until.get(signal.stream_id, -1.0) > context.t_s else 0.0
            uncertainty = 1.0 - abs(signal.cheap_score - 0.5) * 2.0
            score = (
                self.weight_anomaly * signal.anomaly
                + self.weight_motion * signal.motion
                + self.weight_clip * signal.clip
                + self.weight_memory * memory
                + self.weight_uncertainty * uncertainty
                + recency_bonus
                - self.cooldown_penalty * cooldown
            )
            scored.append((score, signal.stream_id))
        scored.sort(reverse=True)
        queries = []
        for score, stream_id in scored[: self.query_budget]:
            self.state.last_seen[stream_id] = context.t_s
            self.state.cooldown_until[stream_id] = context.t_s + context.episode.step_s
            queries.append(Query(stream_id=stream_id, reason="value_of_information", score=round(score, 6)))
        return queries

    def observe_query_result(self, stream_id: str, detected: bool, false_alarm: bool) -> None:
        if not self.memory_enabled:
            return
        current = self.state.risk_memory.get(stream_id, 0.0)
        if detected:
            self.state.risk_memory[stream_id] = min(1.0, current + 0.35)
        elif false_alarm:
            self.state.risk_memory[stream_id] = max(0.0, current - 0.20)
        else:
            self.state.risk_memory[stream_id] = max(0.0, current - 0.05)


class ValueOfInformationNoMemoryPolicy(ValueOfInformationPolicy):
    name = "voi_no_memory"
    weight_memory = 0.0
    memory_enabled = False


class ValueOfInformationNoUncertaintyPolicy(ValueOfInformationPolicy):
    name = "voi_no_uncertainty"
    weight_uncertainty = 0.0


class ValueOfInformationNoClipPolicy(ValueOfInformationPolicy):
    name = "voi_no_clip"
    weight_clip = 0.0


class ValueOfInformationNoAnomalyPolicy(ValueOfInformationPolicy):
    name = "voi_no_anomaly"
    weight_anomaly = 0.0


class ValueOfInformationNoCooldownPolicy(ValueOfInformationPolicy):
    name = "voi_no_cooldown"
    cooldown_penalty = 0.0
