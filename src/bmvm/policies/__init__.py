from .base import Policy
from .baselines import AnomalyTopKPolicy, ClipTopKPolicy, MotionTopKPolicy, RandomPolicy, UniformPolicy
from .voi import ValueOfInformationPolicy

POLICY_REGISTRY = {
    "random": RandomPolicy,
    "uniform": UniformPolicy,
    "motion_topk": MotionTopKPolicy,
    "anomaly_topk": AnomalyTopKPolicy,
    "clip_topk": ClipTopKPolicy,
    "voi": ValueOfInformationPolicy,
}

__all__ = [
    "Policy",
    "POLICY_REGISTRY",
    "RandomPolicy",
    "UniformPolicy",
    "MotionTopKPolicy",
    "AnomalyTopKPolicy",
    "ClipTopKPolicy",
    "ValueOfInformationPolicy",
]
