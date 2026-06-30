from .base import Policy
from .baselines import AnomalyTopKPolicy, ClipTopKPolicy, DensePolicy, MotionTopKPolicy, RandomPolicy, UniformPolicy
from .voi import (
    ValueOfInformationNoAnomalyPolicy,
    ValueOfInformationNoClipPolicy,
    ValueOfInformationNoCooldownPolicy,
    ValueOfInformationNoMemoryPolicy,
    ValueOfInformationNoUncertaintyPolicy,
    ValueOfInformationPolicy,
)

POLICY_REGISTRY = {
    "random": RandomPolicy,
    "uniform": UniformPolicy,
    "motion_topk": MotionTopKPolicy,
    "anomaly_topk": AnomalyTopKPolicy,
    "clip_topk": ClipTopKPolicy,
    "dense": DensePolicy,
    "dense_vlm": DensePolicy,
    "voi": ValueOfInformationPolicy,
    "voi_no_memory": ValueOfInformationNoMemoryPolicy,
    "voi_no_uncertainty": ValueOfInformationNoUncertaintyPolicy,
    "voi_no_clip": ValueOfInformationNoClipPolicy,
    "voi_no_anomaly": ValueOfInformationNoAnomalyPolicy,
    "voi_no_cooldown": ValueOfInformationNoCooldownPolicy,
}

__all__ = [
    "Policy",
    "POLICY_REGISTRY",
    "RandomPolicy",
    "UniformPolicy",
    "MotionTopKPolicy",
    "AnomalyTopKPolicy",
    "ClipTopKPolicy",
    "DensePolicy",
    "ValueOfInformationPolicy",
    "ValueOfInformationNoMemoryPolicy",
    "ValueOfInformationNoUncertaintyPolicy",
    "ValueOfInformationNoClipPolicy",
    "ValueOfInformationNoAnomalyPolicy",
    "ValueOfInformationNoCooldownPolicy",
]
