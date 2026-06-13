"""Configuration and mutable runtime state for the graph generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


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
    rejection_mode: str
    max_seconds_per_sample: Optional[float]


@dataclass(frozen=True)
class OracleConfig:
    candidates_per_attempt: int
    max_iterations: int
    add_edge_repair_budget: int
    use_node_label_cuts: bool
    use_edge_label_cuts: bool
    edge_label_min_changes_per_violation: int
    edge_memory_penalty: float
    edge_memory_update: float
    edge_memory_decay: float
    edge_memory_clip: float


@dataclass(frozen=True)
class DecodePolicy:
    use_feasibility_filtering: bool
    max_feasibility_attempts: int
    feasibility_candidates_per_attempt: int
    feasibility_failure_mode: str
    feasibility_rejection_mode: str
    max_feasibility_seconds_per_sample: Optional[float]


@dataclass(frozen=True)
class CheckpointPolicy:
    restore_best_checkpoint: bool
    checkpoint_root_dir: str


@dataclass(frozen=True)
class MetricsPolicy:
    plot_on_train_end: bool = True


@dataclass(frozen=True)
class TrainingPolicy:
    maximum_epochs: int
    early_stopping_monitor: str
    early_stopping_mode: str
    enable_early_stopping: bool
    early_stopping_patience: int
    early_stopping_min_delta: float
    suppress_non_batch_output: bool = True


@dataclass
class TrainingProgressSamplingConfig:
    enabled: bool = False
    n_samples: int = 7
    every_n_epochs: int = 1
    output_path: Optional[str] = None
    plot_kwargs: Optional[dict] = None
    plot_fn: Optional[Callable] = None

    def __post_init__(self):
        if int(self.n_samples) < 1:
            raise ValueError("sample_training_progress_n_samples must be >= 1.")
        if int(self.every_n_epochs) < 1:
            raise ValueError("sample_training_progress_every_n_epochs must be >= 1.")


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
