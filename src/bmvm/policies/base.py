from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..types import Query, StepContext


class Policy(ABC):
    name = "policy"

    def __init__(self, query_budget: int, seed: int = 0) -> None:
        self.query_budget = query_budget
        self.seed = seed

    def reset(self) -> None:
        """Reset per-episode policy state."""

    @abstractmethod
    def select(self, context: StepContext) -> List[Query]:
        """Select streams to query with the expensive VLM."""
