from __future__ import annotations

import random
from typing import Callable, List

from ..types import FrameSignal, Query, StepContext
from .base import Policy


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, query_budget: int, seed: int = 0) -> None:
        super().__init__(query_budget=query_budget, seed=seed)
        self.rng = random.Random(seed)

    def reset(self) -> None:
        self.rng = random.Random(self.seed)

    def select(self, context: StepContext) -> List[Query]:
        signals = list(context.signals)
        self.rng.shuffle(signals)
        return [
            Query(stream_id=s.stream_id, reason="random", score=0.0)
            for s in signals[: self.query_budget]
        ]


class UniformPolicy(Policy):
    name = "uniform"

    def __init__(self, query_budget: int, seed: int = 0) -> None:
        super().__init__(query_budget=query_budget, seed=seed)
        self.offset = 0

    def reset(self) -> None:
        self.offset = 0

    def select(self, context: StepContext) -> List[Query]:
        signals = sorted(context.signals, key=lambda s: s.stream_id)
        if not signals:
            return []
        selected = []
        for i in range(self.query_budget):
            selected.append(signals[(self.offset + i) % len(signals)])
        self.offset = (self.offset + self.query_budget) % len(signals)
        return [
            Query(stream_id=s.stream_id, reason="round_robin", score=1.0)
            for s in selected
        ]


class _TopKPolicy(Policy):
    name = "topk"
    signal_name = "cheap_score"

    def _score(self, signal: FrameSignal) -> float:
        return float(getattr(signal, self.signal_name))

    def select(self, context: StepContext) -> List[Query]:
        ranked = sorted(context.signals, key=self._score, reverse=True)
        return [
            Query(stream_id=s.stream_id, reason=self.signal_name, score=self._score(s))
            for s in ranked[: self.query_budget]
        ]


class MotionTopKPolicy(_TopKPolicy):
    name = "motion_topk"
    signal_name = "motion"


class AnomalyTopKPolicy(_TopKPolicy):
    name = "anomaly_topk"
    signal_name = "anomaly"


class ClipTopKPolicy(_TopKPolicy):
    name = "clip_topk"
    signal_name = "clip"


class DensePolicy(Policy):
    name = "dense_vlm"

    def select(self, context: StepContext) -> List[Query]:
        return [
            Query(stream_id=s.stream_id, reason="dense_vlm_reference", score=1.0)
            for s in sorted(context.signals, key=lambda signal: signal.stream_id)
        ]
