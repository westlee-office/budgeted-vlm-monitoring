from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Event:
    event_id: str
    stream_id: str
    start_s: float
    end_s: float
    label: str
    severity: float = 1.0
    description: str = ""

    def is_active(self, t_s: float) -> bool:
        return self.start_s <= t_s <= self.end_s


@dataclass(frozen=True)
class FrameSignal:
    episode_id: str
    stream_id: str
    t_s: float
    motion: float
    anomaly: float
    clip: float
    event_ids: Tuple[str, ...] = ()

    @property
    def cheap_score(self) -> float:
        return 0.35 * self.motion + 0.45 * self.anomaly + 0.20 * self.clip


@dataclass
class Episode:
    episode_id: str
    num_streams: int
    horizon_s: float
    step_s: float
    events: List[Event]
    signals: Dict[Tuple[str, float], FrameSignal]

    @property
    def stream_ids(self) -> List[str]:
        return [f"s{i:03d}" for i in range(self.num_streams)]

    @property
    def timesteps(self) -> List[float]:
        n = int(self.horizon_s / self.step_s)
        return [round(i * self.step_s, 6) for i in range(n)]

    def signal(self, stream_id: str, t_s: float) -> FrameSignal:
        return self.signals[(stream_id, round(t_s, 6))]

    def active_events(self, stream_id: str, t_s: float) -> List[Event]:
        return [e for e in self.events if e.stream_id == stream_id and e.is_active(t_s)]


@dataclass(frozen=True)
class CostModel:
    cheap_scan_gpu_s_per_stream: float = 0.0015
    vlm_gpu_s_per_call: float = 0.65
    vlm_calls_per_query: int = 1


@dataclass(frozen=True)
class Budget:
    query_budget_per_step: int
    max_gpu_s_per_episode: Optional[float] = None
    max_vlm_calls_per_episode: Optional[int] = None


@dataclass
class StepContext:
    episode: Episode
    t_s: float
    signals: Sequence[FrameSignal]
    remaining_gpu_s: Optional[float]
    remaining_vlm_calls: Optional[int]


@dataclass
class Query:
    stream_id: str
    reason: str
    score: float


@dataclass
class PolicyState:
    cooldown_until: Dict[str, float] = field(default_factory=dict)
    risk_memory: Dict[str, float] = field(default_factory=dict)
    last_seen: Dict[str, float] = field(default_factory=dict)

    def decay(self, factor: float = 0.96) -> None:
        for stream_id in list(self.risk_memory):
            self.risk_memory[stream_id] *= factor
            if self.risk_memory[stream_id] < 1e-4:
                del self.risk_memory[stream_id]


@dataclass
class StepResult:
    t_s: float
    queries: List[Query]
    detected_event_ids: List[str]
    false_alarms: int
    gpu_s: float
    vlm_calls: int


@dataclass
class EpisodeResult:
    episode_id: str
    policy: str
    total_events: int
    detected_events: Dict[str, float]
    false_alarms: int
    gpu_s: float
    vlm_calls: int
    steps: List[StepResult] = field(default_factory=list)

    def detected_event_ids(self) -> Iterable[str]:
        return self.detected_events.keys()
