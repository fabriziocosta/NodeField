"""Configuration and mutable runtime state for the graph generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class LocalityConfig:
    sample_fraction: float
    horizon: int
    negative_sample_factor: int
    sampling_strategy: str
    target_positive_ratio: Optional[float]


@dataclass(frozen=True)
class FeasibilityConfig:
    estimator: Any
    use_filtering: bool
    max_attempts: int
    candidates_per_attempt: int
    failure_mode: str


@dataclass(frozen=True)
class OracleConfig:
    candidates_per_attempt: int
    max_iterations: int
    use_node_label_cuts: bool
    use_edge_label_cuts: bool
    edge_memory_penalty: float
    edge_memory_update: float
    edge_memory_decay: float
    edge_memory_clip: float


@dataclass
class StreamFitStats:
    seen: int = 0
    warmup_count: int = 0
    training_seen: int = 0
    training_accepted: int = 0
    training_skipped: int = 0
    skipped_too_large: int = 0
    skipped_unknown_node_label: int = 0
    skipped_unknown_edge_label: int = 0
    skipped_transform_error: int = 0
    skipped_supervision_error: int = 0
    acceptance_rate: float = 0.0
