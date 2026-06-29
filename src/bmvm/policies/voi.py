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
            recency_bonus = 0.08 if last_seen is None else min((context.t_s - last_seen) / 120.0, 1.0) * 0.08
            cooldown = 1.0 if self.state.cooldown_until.get(signal.stream_id, -1.0) > context.t_s else 0.0
            uncertainty = 1.0 - abs(signal.cheap_score - 0.5) * 2.0
            score = (
                0.36 * signal.anomaly
                + 0.20 * signal.motion
                + 0.20 * signal.clip
                + 0.18 * memory
                + 0.08 * uncertainty
                + recency_bonus
                - 0.25 * cooldown
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
        current = self.state.risk_memory.get(stream_id, 0.0)
        if detected:
            self.state.risk_memory[stream_id] = min(1.0, current + 0.35)
        elif false_alarm:
            self.state.risk_memory[stream_id] = max(0.0, current - 0.20)
        else:
            self.state.risk_memory[stream_id] = max(0.0, current - 0.05)
