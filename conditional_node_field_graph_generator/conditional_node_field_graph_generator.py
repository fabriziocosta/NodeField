"""Graph encoder/decoder helpers used by the maintained conditional graph-generation pipeline."""

from contextlib import contextmanager
from dataclasses import dataclass
import inspect
import logging
import os
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import random
import threading
import time
import pulp
import dill as pickle
from .runtime_utils import get_runtime_logger, timeit, verbose_log
from typing import List, Tuple, Optional, Any, Sequence, Dict, Union, FrozenSet, Iterable, Callable
from .conditional_node_field_generator import (
    ConditionalNodeGeneratorBase,
    GeneratedNodeBatch,
    GraphConditioningBatch,
    NodeGenerationBatch,
)
from .conditional_node_field_graph_decoder import (
    ConditionalNodeFieldGraphDecoder,
    _assemble_graph_job,
    _assemble_graph_job_star,
    _build_masked_prob_matrix,
    _edge_label_list_to_matrix,
    _edge_label_matrix_to_list,
    _decode_single_adjacency_job,
    _decode_single_adjacency_job_star,
    build_single_generated_node_batch as _build_single_generated_node_batch,
    decode_generated_nodes as _decode_generated_nodes,
    decode_generated_nodes_with_oracle as _decode_generated_nodes_with_oracle,
    resolve_predicted_edge_labels as _resolve_predicted_edge_labels,
    resolve_predicted_node_labels as _resolve_predicted_node_labels,
)
from .oracle_decode import (
    sample_oracle_cuts_for_iteration as _sample_oracle_cuts_for_iteration,
    solve_oracle_relaxed_adjacency as _solve_oracle_relaxed_adjacency,
)
from . import diagnostics as _shared_diagnostics
from .graph_decode_utils import _canonicalize_edge, _normalize_violating_edge_sets
from .input_sources import estimate_source_instance_count, iter_selected_source_graphs
from .naming_utils import sanitize_model_token
from .graph_generator_state import (
    DecodePolicy,
    FeasibilityConfig,
    LocalityConfig,
    OracleConfig,
    StreamFitStats,
    TrainingProgressSamplingConfig,
)
from .decode_service import DecodeService
from .conditioning_sampler import ConditioningSampler
from .encoding_pipeline import EncodingPipeline
from .fit_artifacts import build_fit_artifacts
from .feasibility_effort import (
    DEFAULT_FEASIBILITY_EFFORT_PROFILE,
    feasibility_effort_map as _feasibility_effort_map,
    resolve_feasibility_effort,
)
from .node_batch_builder import NodeBatchBuilder
from .stream_fit import (
    StreamFitService,
    _StreamBatchTimeoutError,
    _StreamSupervisionError,
    _StreamTransformError,
)
from .supervision import SupervisionChannelPlan, SupervisionPlan, SupervisionPlanner
from .decode_pipeline import (
    accept_feasible_candidates_by_slot as _accept_feasible_candidates_by_slot,
    decode_with_feasibility_slots_core as _decode_with_feasibility_slots_core,
    finalize_feasibility_graphs as _finalize_feasibility_graphs,
    log_feasibility_attempt as _log_feasibility_attempt,
    log_feasibility_summary as _log_feasibility_summary,
    score_feasible_rate as _score_feasible_rate,
    should_apply_feasibility_filtering as _should_apply_feasibility_filtering,
)
from .interpolation_utils import (
    interpolate_integer_series as _interpolate_integer_series,
    scaled_slerp,
    scaled_slerp_average,
)
from .oracle_utils import (
    Edge,
    ForbiddenEdgeLabelAssignment,
    ForbiddenNodeLabelAssignment,
    NodeSet,
    _ORACLE_PROBABILITY_EPS,
    apply_oracle_edge_memory_penalty as _apply_oracle_edge_memory_penalty,
    normalize_violating_node_sets as _normalize_violating_node_sets,
    update_oracle_edge_memory as _update_oracle_edge_memory,
)
from .parallel_utils import _normalize_n_jobs, _parallel_map

DEFAULT_DUMMY_NODE_LABEL = "__dummy_node_label__"
logger = get_runtime_logger(__name__)
_ORACLE_EDGE_EXISTENCE_WEIGHT = 1.0
_ORACLE_NODE_LABEL_WEIGHT = 1.0
_ORACLE_EDGE_LABEL_WEIGHT = 1.0
plt = _shared_diagnostics.plt


def _is_molecule_like_graph(graph: nx.Graph) -> bool:
    return _shared_diagnostics._is_molecule_like_graph(graph)


def _coerce_inline_image_array(image: Any) -> Optional[np.ndarray]:
    return _shared_diagnostics._coerce_inline_image_array(image)


def _try_render_molecular_graph_inline(ax: Any, *, decoded_graph: nx.Graph, title: str) -> bool:
    return _shared_diagnostics._try_render_molecular_graph_inline(
        ax,
        decoded_graph=decoded_graph,
        title=title,
    )


def _plot_decoder_diagnostics(**kwargs) -> None:
    return _shared_diagnostics._plot_decoder_diagnostics(
        **kwargs,
        plot_backend=plt,
        inline_renderer=_try_render_molecular_graph_inline,
    )


@dataclass
class GeneratedGuidanceBatch:
    """Generated examples paired with feasibility-derived guidance targets."""

    node_embeddings_list: List[np.ndarray]
    graph_conditioning: GraphConditioningBatch
    decoded_graphs: List[nx.Graph]
    violation_counts: np.ndarray
    guidance_targets: np.ndarray
    feasible_mask: np.ndarray
    sampling_mode: str

    def __len__(self) -> int:
        return int(len(self.decoded_graphs))

# =============================================================================
# ConditionalNodeFieldGraphGenerator Class 
# =============================================================================

class ConditionalNodeFieldGraphGenerator(object):
    """End-to-end manager that vectorises graphs, trains generators, and rebuilds structures."""
    _DEFAULT_FEASIBILITY_ORACLE_CANDIDATES_PER_ATTEMPT = (
        DEFAULT_FEASIBILITY_EFFORT_PROFILE.feasibility_oracle_candidates_per_attempt
    )

    def _ensure_encoding_pipeline(self) -> EncodingPipeline:
        pipeline = getattr(self, "encoding_pipeline_", None)
        if pipeline is None:
            pipeline = EncodingPipeline(self)
            self.encoding_pipeline_ = pipeline
        return pipeline

    def _ensure_supervision_planner(self) -> SupervisionPlanner:
        planner = getattr(self, "supervision_planner_", None)
        if planner is None:
            planner = SupervisionPlanner(self)
            self.supervision_planner_ = planner
        return planner

    def _ensure_node_batch_builder(self) -> NodeBatchBuilder:
        builder = getattr(self, "node_batch_builder_", None)
        if builder is None:
            builder = NodeBatchBuilder(self)
            self.node_batch_builder_ = builder
        return builder

    def _ensure_conditioning_sampler(self) -> ConditioningSampler:
        sampler = getattr(self, "conditioning_sampler_", None)
        if sampler is None:
            sampler = ConditioningSampler(self)
            self.conditioning_sampler_ = sampler
        return sampler

    def _ensure_stream_fit_service(self) -> StreamFitService:
        service = getattr(self, "stream_fit_service_", None)
        if service is None:
            service = StreamFitService(self)
            self.stream_fit_service_ = service
        return service

    def _ensure_stream_fit_stats(self) -> StreamFitStats:
        stats = getattr(self, "stream_fit_stats_", None)
        if stats is None:
            stats = StreamFitStats()
            self.stream_fit_stats_ = stats
        return stats

    def _ensure_stream_runtime_lock(self) -> threading.RLock:
        lock = getattr(self, "_stream_runtime_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._stream_runtime_lock = lock
        return lock

    @staticmethod
    def _edge_label_matrix_to_list(adj_mtx: np.ndarray, edge_label_matrix: np.ndarray) -> np.ndarray:
        return _edge_label_matrix_to_list(adj_mtx, edge_label_matrix)

    @staticmethod
    def _edge_label_list_to_matrix(adj_mtx: np.ndarray, edge_labels: Sequence[Any]) -> np.ndarray:
        return _edge_label_list_to_matrix(adj_mtx, edge_labels)

    @staticmethod
    def _plot_decoder_diagnostics(**kwargs) -> None:
        _plot_decoder_diagnostics(**kwargs)

    @property
    def stream_seen_(self) -> int:
        with self._ensure_stream_runtime_lock():
            return int(self._ensure_stream_fit_stats().seen)

    @stream_seen_.setter
    def stream_seen_(self, value: int) -> None:
        with self._ensure_stream_runtime_lock():
            self._ensure_stream_fit_stats().seen = int(value)

    @property
    def stream_warmup_count_(self) -> int:
        with self._ensure_stream_runtime_lock():
            return int(self._ensure_stream_fit_stats().warmup_count)

    @stream_warmup_count_.setter
    def stream_warmup_count_(self, value: int) -> None:
        with self._ensure_stream_runtime_lock():
            self._ensure_stream_fit_stats().warmup_count = int(value)

    @property
    def stream_training_seen_(self) -> int:
        with self._ensure_stream_runtime_lock():
            return int(self._ensure_stream_fit_stats().training_seen)

    @stream_training_seen_.setter
    def stream_training_seen_(self, value: int) -> None:
        with self._ensure_stream_runtime_lock():
            self._ensure_stream_fit_stats().training_seen = int(value)

    @property
    def stream_training_accepted_(self) -> int:
        with self._ensure_stream_runtime_lock():
            return int(self._ensure_stream_fit_stats().training_accepted)

    @stream_training_accepted_.setter
    def stream_training_accepted_(self, value: int) -> None:
        with self._ensure_stream_runtime_lock():
            self._ensure_stream_fit_stats().training_accepted = int(value)

    @property
    def stream_training_skipped_(self) -> int:
        with self._ensure_stream_runtime_lock():
            return int(self._ensure_stream_fit_stats().training_skipped)

    @stream_training_skipped_.setter
    def stream_training_skipped_(self, value: int) -> None:
        with self._ensure_stream_runtime_lock():
            self._ensure_stream_fit_stats().training_skipped = int(value)

    @property
    def stream_skipped_too_large_(self) -> int:
        with self._ensure_stream_runtime_lock():
            return int(self._ensure_stream_fit_stats().skipped_too_large)

    @stream_skipped_too_large_.setter
    def stream_skipped_too_large_(self, value: int) -> None:
        with self._ensure_stream_runtime_lock():
            self._ensure_stream_fit_stats().skipped_too_large = int(value)

    @property
    def stream_skipped_unknown_node_label_(self) -> int:
        with self._ensure_stream_runtime_lock():
            return int(self._ensure_stream_fit_stats().skipped_unknown_node_label)

    @stream_skipped_unknown_node_label_.setter
    def stream_skipped_unknown_node_label_(self, value: int) -> None:
        with self._ensure_stream_runtime_lock():
            self._ensure_stream_fit_stats().skipped_unknown_node_label = int(value)

    @property
    def stream_skipped_unknown_edge_label_(self) -> int:
        with self._ensure_stream_runtime_lock():
            return int(self._ensure_stream_fit_stats().skipped_unknown_edge_label)

    @stream_skipped_unknown_edge_label_.setter
    def stream_skipped_unknown_edge_label_(self, value: int) -> None:
        with self._ensure_stream_runtime_lock():
            self._ensure_stream_fit_stats().skipped_unknown_edge_label = int(value)

    @property
    def stream_skipped_transform_error_(self) -> int:
        with self._ensure_stream_runtime_lock():
            return int(self._ensure_stream_fit_stats().skipped_transform_error)

    @stream_skipped_transform_error_.setter
    def stream_skipped_transform_error_(self, value: int) -> None:
        with self._ensure_stream_runtime_lock():
            self._ensure_stream_fit_stats().skipped_transform_error = int(value)

    @property
    def stream_skipped_supervision_error_(self) -> int:
        with self._ensure_stream_runtime_lock():
            return int(self._ensure_stream_fit_stats().skipped_supervision_error)

    @stream_skipped_supervision_error_.setter
    def stream_skipped_supervision_error_(self, value: int) -> None:
        with self._ensure_stream_runtime_lock():
            self._ensure_stream_fit_stats().skipped_supervision_error = int(value)

    @property
    def stream_acceptance_rate_(self) -> float:
        with self._ensure_stream_runtime_lock():
            return float(self._ensure_stream_fit_stats().acceptance_rate)

    @stream_acceptance_rate_.setter
    def stream_acceptance_rate_(self, value: float) -> None:
        with self._ensure_stream_runtime_lock():
            self._ensure_stream_fit_stats().acceptance_rate = float(value)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_stream_runtime_lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._stream_runtime_lock = threading.RLock()
        if not hasattr(self, "feasibility_oracle_candidates_per_attempt"):
            self.feasibility_oracle_candidates_per_attempt = int(
                self._DEFAULT_FEASIBILITY_ORACLE_CANDIDATES_PER_ATTEMPT
            )
        if not hasattr(self, "oracle_add_edge_repair_budget"):
            self.oracle_add_edge_repair_budget = int(
                DEFAULT_FEASIBILITY_EFFORT_PROFILE.oracle_add_edge_repair_budget
            )
        if not hasattr(self, "max_oracle_iterations"):
            self.max_oracle_iterations = int(
                DEFAULT_FEASIBILITY_EFFORT_PROFILE.max_oracle_iterations
            )
        if not hasattr(self, "max_feasibility_attempts"):
            self.max_feasibility_attempts = int(
                DEFAULT_FEASIBILITY_EFFORT_PROFILE.max_feasibility_attempts
            )
        if not hasattr(self, "feasibility_candidates_per_attempt"):
            self.feasibility_candidates_per_attempt = int(
                DEFAULT_FEASIBILITY_EFFORT_PROFILE.feasibility_candidates_per_attempt
            )
        if not hasattr(self, "max_feasibility_seconds_per_sample"):
            self.max_feasibility_seconds_per_sample = (
                DEFAULT_FEASIBILITY_EFFORT_PROFILE.max_feasibility_seconds_per_sample
            )
        if not hasattr(self, "max_decode_attempts_per_sample"):
            self.max_decode_attempts_per_sample = int(
                DEFAULT_FEASIBILITY_EFFORT_PROFILE.max_decode_attempts_per_sample
            )
        if not hasattr(self, "decode_service_") or getattr(self.decode_service_, "owner", None) is not self:
            self.decode_service_ = DecodeService(self)

    def __init__(
            self,
            graph_vectorizer: Any = None,
            node_graph_vectorizer: Any = None,
            conditional_node_generator_model: Optional[ConditionalNodeGeneratorBase] = None,
            graph_decoder: Optional[ConditionalNodeFieldGraphDecoder] = None,
            verbose: bool = True,
            locality_sample_fraction: float = 1.0,
            locality_horizon: int = 1,
            negative_sample_factor: int = 1,
            locality_sampling_strategy: str = "stratified_preserve",
            locality_target_positive_ratio: Optional[float] = None,
            feasibility_estimator: Any = None,
            feasibility_oracle_candidates_per_attempt: int = _DEFAULT_FEASIBILITY_ORACLE_CANDIDATES_PER_ATTEMPT,
            use_feasibility_filtering: bool = True,
            max_oracle_iterations: int = DEFAULT_FEASIBILITY_EFFORT_PROFILE.max_oracle_iterations,
            oracle_add_edge_repair_budget: int = DEFAULT_FEASIBILITY_EFFORT_PROFILE.oracle_add_edge_repair_budget,
            oracle_use_node_label_cuts: bool = False,
            oracle_use_edge_label_cuts: bool = False,
            oracle_edge_label_min_changes_per_violation: int = 1,
            oracle_edge_memory_penalty: float = 0.5,
            oracle_edge_memory_update: float = 1.0,
            oracle_edge_memory_decay: float = 1.0,
            oracle_edge_memory_clip: float = 5.0,
            max_feasibility_attempts: int = DEFAULT_FEASIBILITY_EFFORT_PROFILE.max_feasibility_attempts,
            feasibility_candidates_per_attempt: int = DEFAULT_FEASIBILITY_EFFORT_PROFILE.feasibility_candidates_per_attempt,
            feasibility_failure_mode: str = "return_partial",
            feasibility_rejection_mode: str = "fallback_unfiltered",
            max_feasibility_seconds_per_sample: Optional[float] = DEFAULT_FEASIBILITY_EFFORT_PROFILE.max_feasibility_seconds_per_sample,
            max_decode_seconds_per_sample: Optional[float] = None,
            max_decode_attempts_per_sample: int = DEFAULT_FEASIBILITY_EFFORT_PROFILE.max_decode_attempts_per_sample,
            model_name: Optional[str] = None,
            model_dir: Optional[str] = None,
            stream_snapshot_every_n_batches: int = 10,
            stream_batch_timeout_seconds: Optional[float] = 30.0,
            stream_snapshot_timeout_seconds: Optional[float] = 30.0,
            stream_pdf_timeout_seconds: Optional[float] = 60.0,
            stream_max_consecutive_stalls: int = 3,
            use_embedding_svd: bool = True,
            node_embedding_svd_dimension: int = 256,
            graph_embedding_svd_dimension: Optional[int] = None,
            embedding_svd_fit_max_rows: Optional[int] = None,
            embedding_svd_fit_random_state: int = 0,
            embedding_svd_transform_batch_size: Optional[int] = None,
            embedding_svd_n_iter: int = 2,
            embedding_svd_n_oversamples: int = 5,
            ) -> None:
        """Store the collaborating components and configuration used for the pipeline.

        Args:
            graph_vectorizer (Any): Optional input value.
            node_graph_vectorizer (Any): Optional input value.
            conditional_node_generator_model (Optional[ConditionalNodeGeneratorBase]): Optional input value.
            graph_decoder (Optional[ConditionalNodeFieldGraphDecoder]): Optional input value.
            verbose (bool): Optional input value.
            locality_sample_fraction (float): Optional input value.
            locality_horizon (int): Optional input value.
            negative_sample_factor (int): Optional input value.
            locality_sampling_strategy (str): Optional input value.
            locality_target_positive_ratio (Optional[float]): Optional input value.
            feasibility_estimator (Any): Optional input value.
            feasibility_oracle_candidates_per_attempt (int): Optional input value.
            use_feasibility_filtering (bool): Optional input value.
            max_oracle_iterations (int): Optional input value.
            oracle_add_edge_repair_budget (int): Optional input value.
            oracle_use_node_label_cuts (bool): Optional input value.
            oracle_use_edge_label_cuts (bool): Optional input value.
            oracle_edge_label_min_changes_per_violation (int): Optional input value.
            oracle_edge_memory_penalty (float): Optional input value.
            oracle_edge_memory_update (float): Optional input value.
            oracle_edge_memory_decay (float): Optional input value.
            oracle_edge_memory_clip (float): Optional input value.
            max_feasibility_attempts (int): Optional input value.
            feasibility_candidates_per_attempt (int): Optional input value.
            feasibility_failure_mode (str): Optional input value.
            feasibility_rejection_mode (str): Optional input value.
            max_feasibility_seconds_per_sample (Optional[float]): Optional input value.
            max_decode_seconds_per_sample (Optional[float]): Optional input value.
            max_decode_attempts_per_sample (int): Optional input value.
            model_name (Optional[str]): Optional input value.
            model_dir (Optional[str]): Optional input value.
            stream_snapshot_every_n_batches (int): Optional input value.
            stream_batch_timeout_seconds (Optional[float]): Optional input value.
            stream_snapshot_timeout_seconds (Optional[float]): Optional input value.
            stream_pdf_timeout_seconds (Optional[float]): Optional input value.
            stream_max_consecutive_stalls (int): Optional input value.
        """
        self.graph_vectorizer = graph_vectorizer
        self.node_graph_vectorizer = node_graph_vectorizer
        self.use_embedding_svd = bool(use_embedding_svd)
        self.node_embedding_svd_dimension = int(node_embedding_svd_dimension)
        self.graph_embedding_svd_dimension = (
            None
            if graph_embedding_svd_dimension is None
            else int(graph_embedding_svd_dimension)
        )
        self.embedding_svd_fit_max_rows = (
            None
            if embedding_svd_fit_max_rows is None
            else int(embedding_svd_fit_max_rows)
        )
        self.embedding_svd_fit_random_state = int(embedding_svd_fit_random_state)
        self.embedding_svd_transform_batch_size = (
            None
            if embedding_svd_transform_batch_size is None
            else int(embedding_svd_transform_batch_size)
        )
        self.embedding_svd_n_iter = int(embedding_svd_n_iter)
        self.embedding_svd_n_oversamples = int(embedding_svd_n_oversamples)
        self.node_embedding_svd_ = None
        self.graph_embedding_svd_ = None
        self.node_embedding_svd_fitted_ = False
        self.graph_embedding_svd_fitted_ = False
        self.node_embedding_raw_dimension_ = None
        self.graph_embedding_raw_dimension_ = None
        self.node_embedding_effective_dimension_ = None
        self.graph_embedding_effective_dimension_ = None
        self.encoding_pipeline_ = EncodingPipeline(self)
        self.supervision_planner_ = SupervisionPlanner(self)
        self.node_batch_builder_ = NodeBatchBuilder(self)
        self.conditioning_sampler_ = ConditioningSampler(self)
        self.stream_fit_service_ = StreamFitService(self)
        self.conditional_node_generator_model = conditional_node_generator_model
        self.graph_decoder = graph_decoder
        self.verbose = verbose
        self.supervision_plan_: Optional[SupervisionPlan] = None
        self.training_graph_conditioning_: Optional[GraphConditioningBatch] = None
        self.stream_fit_stats_ = StreamFitStats()
        self.stream_prefetch_batches = 2
        self.stream_snapshot_every_n_batches = max(1, int(stream_snapshot_every_n_batches))
        self.stream_batch_timeout_seconds = None if stream_batch_timeout_seconds is None else float(stream_batch_timeout_seconds)
        self.stream_snapshot_timeout_seconds = None if stream_snapshot_timeout_seconds is None else float(stream_snapshot_timeout_seconds)
        self.stream_pdf_timeout_seconds = None if stream_pdf_timeout_seconds is None else float(stream_pdf_timeout_seconds)
        self.stream_max_consecutive_stalls = max(1, int(stream_max_consecutive_stalls))
        self.is_fitted_ = False
        if not 0.0 < locality_sample_fraction <= 1.0:
            raise ValueError("locality_sample_fraction must be between 0.0 (exclusive) and 1.0 (inclusive)")
        self.locality_sample_fraction = locality_sample_fraction
        if locality_horizon < 1:
            raise ValueError("locality_horizon must be >= 1")
        self.locality_horizon = locality_horizon
        self.negative_sample_factor = negative_sample_factor
        self.locality_sampling_strategy = locality_sampling_strategy
        self.locality_target_positive_ratio = locality_target_positive_ratio
        self.feasibility_estimator = feasibility_estimator
        self.feasibility_oracle_candidates_per_attempt = int(feasibility_oracle_candidates_per_attempt)
        self.use_feasibility_filtering = bool(use_feasibility_filtering)
        self.max_oracle_iterations = int(max_oracle_iterations)
        self.oracle_add_edge_repair_budget = int(oracle_add_edge_repair_budget)
        self.oracle_use_node_label_cuts = bool(oracle_use_node_label_cuts)
        self.oracle_use_edge_label_cuts = bool(oracle_use_edge_label_cuts)
        self.oracle_edge_label_min_changes_per_violation = int(
            oracle_edge_label_min_changes_per_violation
        )
        self.oracle_edge_memory_penalty = float(oracle_edge_memory_penalty)
        self.oracle_edge_memory_update = float(oracle_edge_memory_update)
        self.oracle_edge_memory_decay = float(oracle_edge_memory_decay)
        self.oracle_edge_memory_clip = float(oracle_edge_memory_clip)
        self.max_feasibility_attempts = int(max_feasibility_attempts)
        self.feasibility_candidates_per_attempt = int(feasibility_candidates_per_attempt)
        self.feasibility_failure_mode = str(feasibility_failure_mode)
        self.feasibility_rejection_mode = str(feasibility_rejection_mode)
        self.max_feasibility_seconds_per_sample = (
            None if max_feasibility_seconds_per_sample is None else float(max_feasibility_seconds_per_sample)
        )
        self.max_decode_seconds_per_sample = (
            None if max_decode_seconds_per_sample is None else float(max_decode_seconds_per_sample)
        )
        self.max_decode_attempts_per_sample = int(max_decode_attempts_per_sample)
        self.model_name = None if model_name is None else sanitize_model_token(model_name)
        self.model_dir = model_dir
        self._generation_timeout_deadline: Optional[float] = None
        self.warmup_schema_frozen_ = False
        if int(self.verbose) >= 1 and self.model_name is not None:
            verbose_log(
                self,
                f"Configured graph generator model_name={self.model_name} model_dir={self.model_dir}",
            )
        valid_sampling_strategies = {"uniform", "stratified_preserve", "stratified_target"}
        if self.locality_sampling_strategy not in valid_sampling_strategies:
            raise ValueError(
                f"locality_sampling_strategy must be one of {sorted(valid_sampling_strategies)} "
                f"(got {self.locality_sampling_strategy!r})."
            )
        if self.locality_target_positive_ratio is not None and not 0.0 < self.locality_target_positive_ratio < 1.0:
            raise ValueError("locality_target_positive_ratio must be between 0 and 1 when provided.")
        if self.feasibility_oracle_candidates_per_attempt < 0:
            raise ValueError("feasibility_oracle_candidates_per_attempt must be >= 0")
        if self.max_oracle_iterations < 1:
            raise ValueError("max_oracle_iterations must be >= 1")
        if self.oracle_add_edge_repair_budget < 0:
            raise ValueError("oracle_add_edge_repair_budget must be >= 0")
        if self.oracle_edge_label_min_changes_per_violation < 1:
            raise ValueError("oracle_edge_label_min_changes_per_violation must be >= 1")
        if self.oracle_edge_memory_penalty < 0.0:
            raise ValueError("oracle_edge_memory_penalty must be >= 0")
        if self.oracle_edge_memory_update < 0.0:
            raise ValueError("oracle_edge_memory_update must be >= 0")
        if not 0.0 <= self.oracle_edge_memory_decay <= 1.0:
            raise ValueError("oracle_edge_memory_decay must be between 0 and 1")
        if self.oracle_edge_memory_clip < 0.0:
            raise ValueError("oracle_edge_memory_clip must be >= 0")
        if self.max_feasibility_attempts < 1:
            raise ValueError("max_feasibility_attempts must be >= 1")
        if self.feasibility_candidates_per_attempt < 1:
            raise ValueError("feasibility_candidates_per_attempt must be >= 1")
        if (
            self.max_feasibility_seconds_per_sample is not None
            and self.max_feasibility_seconds_per_sample <= 0.0
        ):
            raise ValueError("max_feasibility_seconds_per_sample must be > 0 when provided")
        if self.max_decode_seconds_per_sample is not None and self.max_decode_seconds_per_sample <= 0.0:
            raise ValueError("max_decode_seconds_per_sample must be > 0 when provided")
        if self.max_decode_attempts_per_sample < 1:
            raise ValueError("max_decode_attempts_per_sample must be >= 1")
        if self.stream_batch_timeout_seconds is not None and self.stream_batch_timeout_seconds <= 0.0:
            raise ValueError("stream_batch_timeout_seconds must be > 0 when provided")
        if self.stream_snapshot_timeout_seconds is not None and self.stream_snapshot_timeout_seconds <= 0.0:
            raise ValueError("stream_snapshot_timeout_seconds must be > 0 when provided")
        if self.stream_pdf_timeout_seconds is not None and self.stream_pdf_timeout_seconds <= 0.0:
            raise ValueError("stream_pdf_timeout_seconds must be > 0 when provided")
        valid_feasibility_failure_modes = {"raise", "return_partial"}
        if self.feasibility_failure_mode not in valid_feasibility_failure_modes:
            raise ValueError(
                f"feasibility_failure_mode must be one of {sorted(valid_feasibility_failure_modes)} "
                f"(got {self.feasibility_failure_mode!r})."
            )
        valid_feasibility_rejection_modes = {"fallback_unfiltered", "strict"}
        if self.feasibility_rejection_mode not in valid_feasibility_rejection_modes:
            raise ValueError(
                f"feasibility_rejection_mode must be one of {sorted(valid_feasibility_rejection_modes)} "
                f"(got {self.feasibility_rejection_mode!r})."
            )
        self.locality_config_ = LocalityConfig(
            sample_fraction=float(self.locality_sample_fraction),
            horizon=int(self.locality_horizon),
            negative_sample_factor=int(self.negative_sample_factor),
            sampling_strategy=str(self.locality_sampling_strategy),
            target_positive_ratio=self.locality_target_positive_ratio,
        )
        self.feasibility_config_ = FeasibilityConfig(
            estimator=self.feasibility_estimator,
            use_filtering=bool(self.use_feasibility_filtering),
            max_attempts=int(self.max_feasibility_attempts),
            candidates_per_attempt=int(self.feasibility_candidates_per_attempt),
            failure_mode=str(self.feasibility_failure_mode),
            rejection_mode=str(self.feasibility_rejection_mode),
            max_seconds_per_sample=(
                None
                if self.max_feasibility_seconds_per_sample is None
                else float(self.max_feasibility_seconds_per_sample)
            ),
        )
        self.oracle_config_ = OracleConfig(
            candidates_per_attempt=int(self.feasibility_oracle_candidates_per_attempt),
            max_iterations=int(self.max_oracle_iterations),
            add_edge_repair_budget=int(self.oracle_add_edge_repair_budget),
            use_node_label_cuts=bool(self.oracle_use_node_label_cuts),
            use_edge_label_cuts=bool(self.oracle_use_edge_label_cuts),
            edge_label_min_changes_per_violation=int(
                self.oracle_edge_label_min_changes_per_violation
            ),
            edge_memory_penalty=float(self.oracle_edge_memory_penalty),
            edge_memory_update=float(self.oracle_edge_memory_update),
            edge_memory_decay=float(self.oracle_edge_memory_decay),
            edge_memory_clip=float(self.oracle_edge_memory_clip),
        )
        self.decode_policy_ = DecodePolicy(
            use_feasibility_filtering=bool(self.use_feasibility_filtering),
            max_feasibility_attempts=int(self.max_feasibility_attempts),
            feasibility_candidates_per_attempt=int(self.feasibility_candidates_per_attempt),
            feasibility_failure_mode=str(self.feasibility_failure_mode),
            feasibility_rejection_mode=str(self.feasibility_rejection_mode),
            max_feasibility_seconds_per_sample=(
                None
                if self.max_feasibility_seconds_per_sample is None
                else float(self.max_feasibility_seconds_per_sample)
            ),
        )
        self.decode_service_ = DecodeService(self)

    def _get_generation_timeout_deadline(self) -> Optional[float]:
        deadline = getattr(self, "_generation_timeout_deadline", None)
        if deadline is None:
            return None
        return float(deadline)

    def _remaining_generation_timeout_seconds(self) -> Optional[float]:
        deadline = self._get_generation_timeout_deadline()
        if deadline is None:
            return None
        return max(0.0, float(deadline - time.perf_counter()))

    def _set_generation_timeout_deadline(self, timeout_seconds: Optional[float]) -> Optional[float]:
        previous = getattr(self, "_generation_timeout_deadline", None)
        if timeout_seconds is None:
            self._generation_timeout_deadline = None
        else:
            self._generation_timeout_deadline = time.perf_counter() + max(0.0, float(timeout_seconds))
        return previous

    def _restore_generation_timeout_deadline(self, deadline: Optional[float]) -> None:
        self._generation_timeout_deadline = deadline

    def _resolve_solver_time_limit_seconds(self, default_seconds: Optional[float] = None) -> Optional[float]:
        remaining = self._remaining_generation_timeout_seconds()
        if remaining is None:
            return default_seconds
        if default_seconds is None:
            return remaining
        return min(float(default_seconds), remaining)

    def _resolve_solver_threads(self) -> Optional[int]:
        graph_decoder = getattr(self, "graph_decoder", None)
        solver_threads = None if graph_decoder is None else getattr(graph_decoder, "solver_threads", None)
        return None if solver_threads is None else max(1, int(solver_threads))

    def _restore_label_vocab_metadata_from_node_model(self) -> None:
        node_model = getattr(self, "conditional_node_generator_model", None)
        if node_model is None:
            return

        if getattr(self, "node_label_classes_", None) is None:
            model_node_classes = getattr(node_model, "node_label_classes_", None)
            if model_node_classes is not None:
                self.node_label_classes_ = np.asarray(model_node_classes, dtype=object)
        if getattr(self, "node_label_to_index_", None) is None:
            model_node_to_index = getattr(node_model, "node_label_to_index_", None)
            if model_node_to_index is not None:
                self.node_label_to_index_ = dict(model_node_to_index)
            elif getattr(self, "node_label_classes_", None) is not None:
                self.node_label_to_index_ = {
                    label: idx for idx, label in enumerate(self.node_label_classes_)
                }

        if getattr(self, "edge_label_classes_", None) is None:
            model_edge_classes = getattr(node_model, "edge_label_classes_", None)
            if model_edge_classes is not None:
                self.edge_label_classes_ = np.asarray(model_edge_classes, dtype=object)
        if getattr(self, "edge_label_to_index_", None) is None:
            model_edge_to_index = getattr(node_model, "edge_label_to_index_", None)
            if model_edge_to_index is not None:
                self.edge_label_to_index_ = dict(model_edge_to_index)
            elif getattr(self, "edge_label_classes_", None) is not None:
                self.edge_label_to_index_ = {
                    label: idx for idx, label in enumerate(self.edge_label_classes_)
                }

    def _get_node_label_names(self) -> Optional[np.ndarray]:
        self._restore_label_vocab_metadata_from_node_model()
        node_label_classes = getattr(self, "node_label_classes_", None)
        if node_label_classes is None:
            return None
        return np.asarray(node_label_classes, dtype=object)

    def _get_edge_label_names(self) -> Optional[np.ndarray]:
        self._restore_label_vocab_metadata_from_node_model()
        edge_label_classes = getattr(self, "edge_label_classes_", None)
        if edge_label_classes is None:
            return None
        return np.asarray(edge_label_classes, dtype=object)

    def _fill_unlabeled_active_edges(
        self,
        *,
        adj_mtx: np.ndarray,
        edge_label_matrix: np.ndarray,
        edge_label_probabilities: Optional[np.ndarray],
    ) -> np.ndarray:
        """Fill newly active unlabeled edges from edge-label probabilities when available."""
        repaired = np.asarray(edge_label_matrix, dtype=object).copy()
        edge_label_names = self._get_edge_label_names()
        if edge_label_probabilities is None or edge_label_names is None or len(edge_label_names) == 0:
            return repaired

        adj_mtx = np.asarray(adj_mtx, dtype=float)
        edge_label_probabilities = np.asarray(edge_label_probabilities, dtype=float)
        for i in range(adj_mtx.shape[0]):
            for j in range(i + 1, adj_mtx.shape[1]):
                if adj_mtx[i, j] == 0:
                    continue
                if repaired[i, j] is not None:
                    continue
                label_idx = int(np.argmax(edge_label_probabilities[i, j]))
                selected_label = edge_label_names[label_idx]
                repaired[i, j] = selected_label
                repaired[j, i] = selected_label
        return repaired

    def set_feasibility_filtering(self, enabled: bool) -> None:
        """Enable or disable feasibility filtering during generation without discarding the fitted estimator.

        Args:
            enabled (bool): Input value.
        """
        self.use_feasibility_filtering = bool(enabled)

    @staticmethod
    def feasibility_effort_map() -> Dict[int, dict]:
        """Return the concrete generation settings for feasibility effort levels 0..5."""
        return _feasibility_effort_map()

    @staticmethod
    def _has_explicit_legacy_feasibility_controls(**kwargs) -> bool:
        return any(value is not None for value in kwargs.values())

    @contextmanager
    def _feasibility_effort_context(
        self,
        feasibility_effort: Optional[int],
        *,
        apply_feasibility_filtering: Optional[bool] = None,
        use_feasibility_oracle: Optional[bool] = None,
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
        max_feasibility_attempts: Optional[int] = None,
        feasibility_candidates_per_attempt: Optional[int] = None,
    ):
        legacy_controls = {
            "apply_feasibility_filtering": apply_feasibility_filtering,
            "use_feasibility_oracle": use_feasibility_oracle,
            "feasibility_oracle_candidates_per_attempt": feasibility_oracle_candidates_per_attempt,
            "max_feasibility_attempts": max_feasibility_attempts,
            "feasibility_candidates_per_attempt": feasibility_candidates_per_attempt,
        }
        legacy_controls = {
            key: value
            for key, value in legacy_controls.items()
            if value is not None
        }
        if feasibility_effort is None:
            if legacy_controls:
                warnings.warn(
                    "Direct feasibility/oracle generation kwargs are deprecated; "
                    "use feasibility_effort=0..5 instead.",
                    DeprecationWarning,
                    stacklevel=3,
                )
            yield None
            return
        if legacy_controls:
            names = ", ".join(sorted(legacy_controls))
            raise ValueError(
                "feasibility_effort cannot be combined with deprecated feasibility/oracle "
                f"kwargs: {names}."
            )

        profile = resolve_feasibility_effort(feasibility_effort)
        attrs = {
            "use_feasibility_filtering": profile.apply_feasibility_filtering,
            "feasibility_oracle_candidates_per_attempt": (
                profile.feasibility_oracle_candidates_per_attempt
                if profile.use_feasibility_oracle
                else 0
            ),
            "max_oracle_iterations": profile.max_oracle_iterations,
            "oracle_add_edge_repair_budget": profile.oracle_add_edge_repair_budget,
            "max_feasibility_attempts": profile.max_feasibility_attempts,
            "feasibility_candidates_per_attempt": profile.feasibility_candidates_per_attempt,
            "max_feasibility_seconds_per_sample": profile.max_feasibility_seconds_per_sample,
            "max_decode_attempts_per_sample": profile.max_decode_attempts_per_sample,
        }
        previous = {name: getattr(self, name, None) for name in attrs}
        try:
            for name, value in attrs.items():
                setattr(self, name, value)
            yield profile
        finally:
            for name, value in previous.items():
                setattr(self, name, value)

    @contextmanager
    def _feasibility_filter_context(
        self,
        feasibility_filter: Optional[str],
        *,
        apply_feasibility_filtering: Optional[bool] = None,
    ):
        if feasibility_filter is None:
            yield None
            return
        if apply_feasibility_filtering is not None:
            raise ValueError(
                "feasibility_filter cannot be combined with apply_feasibility_filtering."
            )
        policy = str(feasibility_filter).lower()
        valid_policies = {"none", "fallback", "strict"}
        if policy not in valid_policies:
            raise ValueError(
                f"feasibility_filter must be one of {sorted(valid_policies)} "
                f"(got {feasibility_filter!r})."
            )
        fallback_sentinel = object()
        previous = {
            "feasibility_failure_mode": getattr(self, "feasibility_failure_mode", None),
            "feasibility_rejection_mode": getattr(self, "feasibility_rejection_mode", None),
            "feasibility_fallback_strategy": getattr(
                self,
                "feasibility_fallback_strategy",
                fallback_sentinel,
            ),
        }
        try:
            if policy == "fallback":
                self.feasibility_failure_mode = "return_partial"
                self.feasibility_rejection_mode = "fallback_unfiltered"
                self.feasibility_fallback_strategy = "best_candidate"
                yield True
            elif policy == "strict":
                self.feasibility_failure_mode = "return_partial"
                self.feasibility_rejection_mode = "strict"
                self.feasibility_fallback_strategy = None
                yield True
            else:
                yield False
        finally:
            self.feasibility_failure_mode = previous["feasibility_failure_mode"]
            self.feasibility_rejection_mode = previous["feasibility_rejection_mode"]
            if previous["feasibility_fallback_strategy"] is fallback_sentinel:
                if hasattr(self, "feasibility_fallback_strategy"):
                    delattr(self, "feasibility_fallback_strategy")
            else:
                self.feasibility_fallback_strategy = previous["feasibility_fallback_strategy"]

    def _resolve_feasibility_oracle_candidates_per_attempt(
        self,
        feasibility_oracle_candidates_per_attempt: Optional[int],
    ) -> int:
        if feasibility_oracle_candidates_per_attempt is None:
            return int(self.feasibility_oracle_candidates_per_attempt)
        value = int(feasibility_oracle_candidates_per_attempt)
        if value < 0:
            raise ValueError("feasibility_oracle_candidates_per_attempt must be >= 0")
        return value

    def _resolve_feasibility_oracle_override(
        self,
        *,
        use_feasibility_oracle: Optional[bool],
        feasibility_oracle_candidates_per_attempt: Optional[int],
    ) -> Optional[int]:
        if use_feasibility_oracle is None:
            return feasibility_oracle_candidates_per_attempt
        if not bool(use_feasibility_oracle):
            return 0
        if feasibility_oracle_candidates_per_attempt is not None:
            value = self._resolve_feasibility_oracle_candidates_per_attempt(
                feasibility_oracle_candidates_per_attempt
            )
            if value <= 0:
                raise ValueError(
                    "use_feasibility_oracle=True requires "
                    "feasibility_oracle_candidates_per_attempt > 0."
                )
            return value
        configured_budget = self._resolve_feasibility_oracle_candidates_per_attempt(None)
        if configured_budget > 0:
            return None
        return int(self._DEFAULT_FEASIBILITY_ORACLE_CANDIDATES_PER_ATTEMPT)

    def _can_use_feasibility_oracle(
        self,
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
        attempt_idx: int = 0,
    ) -> bool:
        budget = self._resolve_feasibility_oracle_candidates_per_attempt(
            feasibility_oracle_candidates_per_attempt
        )
        if attempt_idx < 0:
            raise ValueError("attempt_idx must be >= 0")
        if budget <= attempt_idx:
            return False
        if self.feasibility_estimator is None:
            return False
        return hasattr(self.feasibility_estimator, "violating_edge_sets")

    def _build_single_generated_node_batch(self, generated_nodes: GeneratedNodeBatch, graph_idx: int) -> GeneratedNodeBatch:
        return _build_single_generated_node_batch(generated_nodes, graph_idx)

    def _oracle_candidate_score_components(
        self,
        *,
        existence_mask: np.ndarray,
        adj_mtx: np.ndarray,
        node_labels: np.ndarray,
        edge_label_matrix: np.ndarray,
        edge_probability_matrix: np.ndarray,
        node_label_probabilities: Optional[np.ndarray],
        edge_label_probabilities: Optional[np.ndarray],
    ) -> Tuple[float, float, float, float]:
        existence_mask = np.asarray(existence_mask, dtype=bool)
        adj_mtx = np.asarray(adj_mtx, dtype=float)
        edge_probability_matrix = np.asarray(edge_probability_matrix, dtype=float)
        active_indices = np.where(existence_mask[: adj_mtx.shape[0]])[0]
        node_label_to_index = getattr(self, "node_label_to_index_", None)
        edge_label_to_index = getattr(self, "edge_label_to_index_", None)

        edge_terms = []
        for idx_i, i in enumerate(active_indices):
            for j in active_indices[idx_i + 1:]:
                edge_prob = float(np.clip(edge_probability_matrix[i, j], _ORACLE_PROBABILITY_EPS, 1.0 - _ORACLE_PROBABILITY_EPS))
                edge_terms.append(np.log(edge_prob) if adj_mtx[i, j] != 0 else np.log(1.0 - edge_prob))
        edge_score = float(np.mean(edge_terms)) if edge_terms else 0.0

        node_score = 0.0
        if node_label_probabilities is not None and node_label_to_index is not None:
            node_terms = []
            for node_idx in active_indices:
                label = node_labels[node_idx]
                label_idx = node_label_to_index.get(label)
                if label_idx is None:
                    continue
                label_prob = float(np.clip(node_label_probabilities[node_idx, label_idx], _ORACLE_PROBABILITY_EPS, 1.0))
                node_terms.append(np.log(label_prob))
            if node_terms:
                node_score = float(np.mean(node_terms))

        edge_label_score = 0.0
        if edge_label_probabilities is not None and edge_label_to_index is not None:
            edge_label_terms = []
            for idx_i, i in enumerate(active_indices):
                for j in active_indices[idx_i + 1:]:
                    if adj_mtx[i, j] == 0:
                        continue
                    label = edge_label_matrix[i, j]
                    label_idx = edge_label_to_index.get(label)
                    if label_idx is None:
                        continue
                    label_prob = float(np.clip(edge_label_probabilities[i, j, label_idx], _ORACLE_PROBABILITY_EPS, 1.0))
                    edge_label_terms.append(np.log(label_prob))
            if edge_label_terms:
                edge_label_score = float(np.mean(edge_label_terms))

        total_score = (
            _ORACLE_EDGE_EXISTENCE_WEIGHT * edge_score
            + _ORACLE_NODE_LABEL_WEIGHT * node_score
            + _ORACLE_EDGE_LABEL_WEIGHT * edge_label_score
        )
        return total_score, edge_score, node_score, edge_label_score

    def _oracle_candidate_score(
        self,
        *,
        existence_mask: np.ndarray,
        adj_mtx: np.ndarray,
        node_labels: np.ndarray,
        edge_label_matrix: np.ndarray,
        edge_probability_matrix: np.ndarray,
        node_label_probabilities: Optional[np.ndarray],
        edge_label_probabilities: Optional[np.ndarray],
    ) -> float:
        total_score, _, _, _ = self._oracle_candidate_score_components(
            existence_mask=existence_mask,
            adj_mtx=adj_mtx,
            node_labels=node_labels,
            edge_label_matrix=edge_label_matrix,
            edge_probability_matrix=edge_probability_matrix,
            node_label_probabilities=node_label_probabilities,
            edge_label_probabilities=edge_label_probabilities,
        )
        return total_score

    def _get_oracle_node_violation_sets(
        self,
        graph: nx.Graph,
        *,
        n_nodes: int,
    ) -> List[NodeSet]:
        if not getattr(self, "oracle_use_node_label_cuts", True):
            return []
        if self.feasibility_estimator is None or not hasattr(self.feasibility_estimator, "violating_node_labels_sets"):
            return []
        violating_sets_raw = self.feasibility_estimator.violating_node_labels_sets([graph])[0]
        return _normalize_violating_node_sets(violating_sets_raw, n_nodes=n_nodes)

    def _get_oracle_edge_violation_sets(
        self,
        graph: nx.Graph,
        *,
        n_nodes: int,
    ) -> List[FrozenSet[Edge]]:
        if self.feasibility_estimator is None or not hasattr(self.feasibility_estimator, "violating_edge_sets"):
            return []
        violating_sets_raw = self.feasibility_estimator.violating_edge_sets([graph])[0]
        return _normalize_violating_edge_sets(violating_sets_raw, n_nodes=n_nodes)

    @staticmethod
    def _forbidden_node_label_assignment_from_sets(
        node_violation_sets: Sequence[NodeSet],
        node_labels: np.ndarray,
    ) -> List[ForbiddenNodeLabelAssignment]:
        forbidden_assignments: List[ForbiddenNodeLabelAssignment] = []
        for node_set in node_violation_sets:
            labels = tuple(node_labels[node_idx] for node_idx in node_set)
            forbidden_assignments.append((tuple(node_set), labels))
        return forbidden_assignments

    @staticmethod
    def _forbidden_edge_label_assignment_from_sets(
        edge_violation_sets: Sequence[FrozenSet[Edge]],
        edge_label_matrix: np.ndarray,
    ) -> List[ForbiddenEdgeLabelAssignment]:
        forbidden_assignments: List[ForbiddenEdgeLabelAssignment] = []
        for edge_set in edge_violation_sets:
            edges = tuple(sorted(edge_set))
            labels = tuple(edge_label_matrix[i, j] for i, j in edges)
            forbidden_assignments.append((edges, labels))
        return forbidden_assignments

    def _repair_node_labels_with_oracle(
        self,
        *,
        existence_mask: np.ndarray,
        current_node_labels: np.ndarray,
        node_label_probabilities: Optional[np.ndarray],
        forbidden_assignments: Sequence[ForbiddenNodeLabelAssignment],
    ) -> np.ndarray:
        node_label_classes = self._get_node_label_names()
        node_label_to_index = getattr(self, "node_label_to_index_", None)
        if (
            node_label_probabilities is None
            or node_label_classes is None
            or node_label_to_index is None
            or len(node_label_classes) == 0
            or not forbidden_assignments
        ):
            return np.asarray(current_node_labels, dtype=object)

        affected_nodes = sorted({
            int(node_idx)
            for node_set, _ in forbidden_assignments
            for node_idx in node_set
            if int(node_idx) < len(existence_mask) and bool(existence_mask[int(node_idx)])
        })
        if not affected_nodes:
            return np.asarray(current_node_labels, dtype=object)

        prob = pulp.LpProblem("OracleNodeLabelRepair", pulp.LpMaximize)
        y = {
            (node_idx, label_idx): pulp.LpVariable(f"y_node_{node_idx}_{label_idx}", cat="Binary")
            for node_idx in affected_nodes
            for label_idx in range(len(node_label_classes))
        }
        prob += pulp.lpSum(
            np.log(float(np.clip(node_label_probabilities[node_idx, label_idx], _ORACLE_PROBABILITY_EPS, 1.0)))
            * y[(node_idx, label_idx)]
            for node_idx in affected_nodes
            for label_idx in range(len(node_label_classes))
        )
        for node_idx in affected_nodes:
            prob += (
                pulp.lpSum(y[(node_idx, label_idx)] for label_idx in range(len(node_label_classes))) == 1
            ), f"NodeLabelOneHot_{node_idx}"
        for cut_idx, (node_set, label_tuple) in enumerate(forbidden_assignments):
            if any(node_idx not in affected_nodes for node_idx in node_set):
                continue
            label_indices = [node_label_to_index.get(label) for label in label_tuple]
            if any(label_idx is None for label_idx in label_indices):
                continue
            prob += (
                pulp.lpSum(
                    y[(node_idx, int(label_idx))]
                    for node_idx, label_idx in zip(node_set, label_indices)
                ) <= len(node_set) - 1
            ), f"ForbiddenNodeLabels_{cut_idx}"
        solver_kwargs = {"msg": False}
        solver_time_limit = self._resolve_solver_time_limit_seconds()
        if solver_time_limit is not None:
            solver_kwargs["timeLimit"] = max(1.0, float(solver_time_limit))
        solver_threads = self._resolve_solver_threads()
        if solver_threads is not None:
            solver_kwargs["threads"] = solver_threads
        status = prob.solve(pulp.PULP_CBC_CMD(**solver_kwargs))
        if int(status) != pulp.LpStatusOptimal:
            return np.asarray(current_node_labels, dtype=object)
        repaired = np.asarray(current_node_labels, dtype=object).copy()
        for node_idx in affected_nodes:
            for label_idx, label in enumerate(node_label_classes):
                value = pulp.value(y[(node_idx, label_idx)])
                if value is not None and int(round(float(value))) == 1:
                    repaired[node_idx] = label
                    break
        return repaired

    def _repair_edge_labels_with_oracle(
        self,
        *,
        existence_mask: np.ndarray,
        adj_mtx: np.ndarray,
        current_edge_label_matrix: np.ndarray,
        edge_label_probabilities: Optional[np.ndarray],
        forbidden_assignments: Sequence[ForbiddenEdgeLabelAssignment],
    ) -> np.ndarray:
        edge_label_classes = self._get_edge_label_names()
        edge_label_to_index = getattr(self, "edge_label_to_index_", None)
        if (
            edge_label_probabilities is None
            or edge_label_classes is None
            or edge_label_to_index is None
            or len(edge_label_classes) == 0
            or not forbidden_assignments
        ):
            return np.asarray(current_edge_label_matrix, dtype=object)

        active_edges = sorted({
            edge
            for edge_set, _ in forbidden_assignments
            for edge in edge_set
            if edge[0] < len(existence_mask)
            and edge[1] < len(existence_mask)
            and bool(existence_mask[edge[0]])
            and bool(existence_mask[edge[1]])
            and adj_mtx[edge[0], edge[1]] != 0
        })
        if not active_edges:
            return np.asarray(current_edge_label_matrix, dtype=object)

        prob = pulp.LpProblem("OracleEdgeLabelRepair", pulp.LpMaximize)
        z = {
            (edge, label_idx): pulp.LpVariable(f"z_edge_{edge[0]}_{edge[1]}_{label_idx}", cat="Binary")
            for edge in active_edges
            for label_idx in range(len(edge_label_classes))
        }
        prob += pulp.lpSum(
            np.log(float(np.clip(edge_label_probabilities[edge[0], edge[1], label_idx], _ORACLE_PROBABILITY_EPS, 1.0)))
            * z[(edge, label_idx)]
            for edge in active_edges
            for label_idx in range(len(edge_label_classes))
        )
        for edge in active_edges:
            prob += (
                pulp.lpSum(z[(edge, label_idx)] for label_idx in range(len(edge_label_classes))) == 1
            ), f"EdgeLabelOneHot_{edge[0]}_{edge[1]}"
        for cut_idx, (edge_set, label_tuple) in enumerate(forbidden_assignments):
            if any(edge not in active_edges for edge in edge_set):
                continue
            label_indices = [edge_label_to_index.get(label) for label in label_tuple]
            if any(label_idx is None for label_idx in label_indices):
                continue
            min_changes = min(
                len(edge_set),
                max(1, int(getattr(self, "oracle_edge_label_min_changes_per_violation", 1))),
            )
            prob += (
                pulp.lpSum(
                    z[(edge, int(label_idx))]
                    for edge, label_idx in zip(edge_set, label_indices)
                ) <= len(edge_set) - min_changes
            ), f"ForbiddenEdgeLabels_{cut_idx}"
        solver_kwargs = {"msg": False}
        solver_time_limit = self._resolve_solver_time_limit_seconds()
        if solver_time_limit is not None:
            solver_kwargs["timeLimit"] = max(1.0, float(solver_time_limit))
        solver_threads = self._resolve_solver_threads()
        if solver_threads is not None:
            solver_kwargs["threads"] = solver_threads
        status = prob.solve(pulp.PULP_CBC_CMD(**solver_kwargs))
        if int(status) != pulp.LpStatusOptimal:
            return np.asarray(current_edge_label_matrix, dtype=object)
        repaired = np.asarray(current_edge_label_matrix, dtype=object).copy()
        for edge in active_edges:
            selected_label = None
            for label_idx, label in enumerate(edge_label_classes):
                value = pulp.value(z[(edge, label_idx)])
                if value is not None and int(round(float(value))) == 1:
                    selected_label = label
                    break
            repaired[edge[0], edge[1]] = selected_label
            repaired[edge[1], edge[0]] = selected_label
        return repaired

    def _repair_labels_with_oracle(
        self,
        *,
        existence_mask: np.ndarray,
        adj_mtx: np.ndarray,
        current_node_labels: np.ndarray,
        current_edge_label_matrix: np.ndarray,
        node_label_probabilities: Optional[np.ndarray],
        edge_label_probabilities: Optional[np.ndarray],
        forbidden_node_assignments: Sequence[ForbiddenNodeLabelAssignment],
        forbidden_edge_assignments: Sequence[ForbiddenEdgeLabelAssignment],
    ) -> Tuple[np.ndarray, np.ndarray]:
        existence_mask = np.asarray(existence_mask, dtype=bool)
        adj_mtx = np.asarray(adj_mtx, dtype=float)
        repaired_node_labels = np.asarray(current_node_labels, dtype=object).copy()
        repaired_edge_label_matrix = np.asarray(current_edge_label_matrix, dtype=object).copy()

        node_label_classes = self._get_node_label_names()
        node_label_to_index = getattr(self, "node_label_to_index_", None)
        edge_label_classes = self._get_edge_label_names()
        edge_label_to_index = getattr(self, "edge_label_to_index_", None)

        active_nodes = [
            int(node_idx)
            for node_idx in range(min(len(existence_mask), adj_mtx.shape[0]))
            if bool(existence_mask[node_idx])
        ]
        active_edges = [
            (i, j)
            for idx_i, i in enumerate(active_nodes)
            for j in active_nodes[idx_i + 1:]
            if adj_mtx[i, j] != 0
        ]

        use_node_labels = (
            node_label_probabilities is not None
            and node_label_classes is not None
            and node_label_to_index is not None
            and len(node_label_classes) > 0
        )
        use_edge_labels = (
            edge_label_probabilities is not None
            and edge_label_classes is not None
            and edge_label_to_index is not None
            and len(edge_label_classes) > 0
        )
        if (not use_node_labels and not use_edge_labels) or (
            not forbidden_node_assignments and not forbidden_edge_assignments
        ):
            return repaired_node_labels, repaired_edge_label_matrix

        prob = pulp.LpProblem("OracleJointLabelRepair", pulp.LpMaximize)
        y = {}
        z = {}

        node_normalizer = max(1, len(active_nodes))
        edge_normalizer = max(1, len(active_edges))
        objective_terms = []

        if use_node_labels:
            for node_idx in active_nodes:
                for label_idx in range(len(node_label_classes)):
                    y[(node_idx, label_idx)] = pulp.LpVariable(
                        f"y_node_{node_idx}_{label_idx}",
                        cat="Binary",
                    )
                    objective_terms.append(
                        (_ORACLE_NODE_LABEL_WEIGHT / node_normalizer)
                        * np.log(float(np.clip(node_label_probabilities[node_idx, label_idx], _ORACLE_PROBABILITY_EPS, 1.0)))
                        * y[(node_idx, label_idx)]
                    )
            for node_idx in active_nodes:
                prob += (
                    pulp.lpSum(y[(node_idx, label_idx)] for label_idx in range(len(node_label_classes))) == 1
                ), f"JointNodeLabelOneHot_{node_idx}"

        if use_edge_labels:
            for edge in active_edges:
                for label_idx in range(len(edge_label_classes)):
                    z[(edge, label_idx)] = pulp.LpVariable(
                        f"z_edge_{edge[0]}_{edge[1]}_{label_idx}",
                        cat="Binary",
                    )
                    objective_terms.append(
                        (_ORACLE_EDGE_LABEL_WEIGHT / edge_normalizer)
                        * np.log(float(np.clip(edge_label_probabilities[edge[0], edge[1], label_idx], _ORACLE_PROBABILITY_EPS, 1.0)))
                        * z[(edge, label_idx)]
                    )
            for edge in active_edges:
                prob += (
                    pulp.lpSum(z[(edge, label_idx)] for label_idx in range(len(edge_label_classes))) == 1
                ), f"JointEdgeLabelOneHot_{edge[0]}_{edge[1]}"

        if not objective_terms:
            return repaired_node_labels, repaired_edge_label_matrix
        prob += pulp.lpSum(objective_terms)

        if use_node_labels:
            for cut_idx, (node_set, label_tuple) in enumerate(forbidden_node_assignments):
                if any(node_idx not in active_nodes for node_idx in node_set):
                    continue
                label_indices = [node_label_to_index.get(label) for label in label_tuple]
                if any(label_idx is None for label_idx in label_indices):
                    continue
                prob += (
                    pulp.lpSum(
                        y[(node_idx, int(label_idx))]
                        for node_idx, label_idx in zip(node_set, label_indices)
                    ) <= len(node_set) - 1
                ), f"JointForbiddenNodeLabels_{cut_idx}"

        if use_edge_labels:
            active_edge_set = set(active_edges)
            for cut_idx, (edge_set, label_tuple) in enumerate(forbidden_edge_assignments):
                if any(edge not in active_edge_set for edge in edge_set):
                    continue
                label_indices = [edge_label_to_index.get(label) for label in label_tuple]
                if any(label_idx is None for label_idx in label_indices):
                    continue
                min_changes = min(
                    len(edge_set),
                    max(1, int(getattr(self, "oracle_edge_label_min_changes_per_violation", 1))),
                )
                prob += (
                    pulp.lpSum(
                        z[(edge, int(label_idx))]
                        for edge, label_idx in zip(edge_set, label_indices)
                    ) <= len(edge_set) - min_changes
                ), f"JointForbiddenEdgeLabels_{cut_idx}"

        solver_kwargs = {"msg": False}
        solver_time_limit = self._resolve_solver_time_limit_seconds()
        if solver_time_limit is not None:
            solver_kwargs["timeLimit"] = max(1.0, float(solver_time_limit))
        solver_threads = self._resolve_solver_threads()
        if solver_threads is not None:
            solver_kwargs["threads"] = solver_threads
        status = prob.solve(pulp.PULP_CBC_CMD(**solver_kwargs))
        if int(status) != pulp.LpStatusOptimal:
            return repaired_node_labels, repaired_edge_label_matrix

        if use_node_labels:
            for node_idx in active_nodes:
                for label_idx, label in enumerate(node_label_classes):
                    value = pulp.value(y[(node_idx, label_idx)])
                    if value is not None and int(round(float(value))) == 1:
                        repaired_node_labels[node_idx] = label
                        break

        if use_edge_labels:
            for edge in active_edges:
                selected_label = None
                for label_idx, label in enumerate(edge_label_classes):
                    value = pulp.value(z[(edge, label_idx)])
                    if value is not None and int(round(float(value))) == 1:
                        selected_label = label
                        break
                repaired_edge_label_matrix[edge[0], edge[1]] = selected_label
                repaired_edge_label_matrix[edge[1], edge[0]] = selected_label

        return repaired_node_labels, repaired_edge_label_matrix

    def _sample_oracle_cuts_for_iteration(
        self,
        accumulated_cuts: Sequence[FrozenSet[Edge]],
        solve_iteration_idx: int,
    ) -> List[FrozenSet[Edge]]:
        return _sample_oracle_cuts_for_iteration(self, accumulated_cuts, solve_iteration_idx)

    def _solve_oracle_relaxed_adjacency(
        self,
        *,
        masked_prob_matrix: np.ndarray,
        target_degrees: List[int],
        accumulated_cuts: Sequence[FrozenSet[Edge]],
        start_iteration_idx: int,
        edge_violation_prior: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return _solve_oracle_relaxed_adjacency(
            self,
            masked_prob_matrix=masked_prob_matrix,
            target_degrees=target_degrees,
            accumulated_cuts=accumulated_cuts,
            start_iteration_idx=start_iteration_idx,
            edge_violation_prior=edge_violation_prior,
        )

    def _decode_generated_nodes_with_oracle(
        self,
        generated_nodes: GeneratedNodeBatch,
        graph_conditioning: Optional[GraphConditioningBatch] = None,
    ) -> List[Optional[nx.Graph]]:
        return _decode_generated_nodes_with_oracle(self, generated_nodes, graph_conditioning=graph_conditioning)

    def _build_supervision_plan(
        self,
        graphs: List[nx.Graph],
        node_label_targets: List[np.ndarray],
        edge_label_targets: Optional[np.ndarray],
    ) -> SupervisionPlan:
        """Build a single explicit supervision plan for the whole fit() call."""
        return self._ensure_supervision_planner().build_supervision_plan(
            graphs,
            node_label_targets=node_label_targets,
            edge_label_targets=edge_label_targets,
        )

    def _log_supervision_plan(self, supervision_plan: SupervisionPlan) -> None:
        """Print the current supervision plan when verbose logging is enabled."""
        self._ensure_supervision_planner().log_supervision_plan(supervision_plan)

    def _plan_channel(self, channel_name: str) -> Optional[SupervisionChannelPlan]:
        """Return the named supervision channel when a plan is available."""
        return self._ensure_supervision_planner().plan_channel(channel_name)

    def _resolve_predicted_node_labels(
        self,
        generated_nodes: GeneratedNodeBatch,
    ) -> List[np.ndarray]:
        """Resolve node labels from explicit predictions or orchestration policy."""
        return _resolve_predicted_node_labels(self, generated_nodes)

    def _resolve_predicted_edge_labels(
        self,
        generated_nodes: GeneratedNodeBatch,
        predicted_edge_probability_matrices: Optional[List[np.ndarray]],
    ) -> Tuple[Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
        """Resolve edge labels from explicit predictions or orchestration policy."""
        return _resolve_predicted_edge_labels(
            self,
            generated_nodes,
            predicted_edge_probability_matrices=predicted_edge_probability_matrices,
        )

    def toggle_verbose(self) -> None:
        """Flip verbosity for this instance and any nested generators.

        Args:
            None: This callable does not take explicit parameters.
        """
        self.verbose = not self.verbose
        if self.conditional_node_generator_model is not None:
            self.conditional_node_generator_model.verbose = self.verbose
        if self.graph_decoder is not None:
            self.graph_decoder.verbose = self.verbose

    def _require_fitted_for_generation(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(
                "ConditionalNodeFieldGraphGenerator is not fitted. Call fit() before decode(), sample(), or other generation methods."
            )
        if self.conditional_node_generator_model is None:
            raise RuntimeError(
                "ConditionalNodeFieldGraphGenerator cannot generate graphs because conditional_node_generator_model is None."
            )
        if self.graph_decoder is None:
            raise RuntimeError(
                "ConditionalNodeFieldGraphGenerator cannot generate graphs because graph_decoder is None."
            )

    def _require_training_graph_conditioning(self) -> GraphConditioningBatch:
        conditioning = getattr(self, "training_graph_conditioning_", None)
        if conditioning is None:
            raise RuntimeError(
                "ConditionalNodeFieldGraphGenerator cannot sample graph-level conditions "
                "because fit() did not cache training graph conditioning."
            )
        graph_embeddings = np.asarray(conditioning.graph_embeddings)
        if graph_embeddings.ndim == 0 or len(graph_embeddings) == 0:
            raise RuntimeError(
                "ConditionalNodeFieldGraphGenerator cannot sample graph-level conditions "
                "because the cached training conditioning is empty."
            )
        return conditioning

    def _require_fit_components(self, train_node_generator: bool) -> None:
        """Validate that fit-time collaborators are configured before dereferencing them."""
        if self.graph_vectorizer is None:
            raise ValueError(
                "ConditionalNodeFieldGraphGenerator.fit() requires graph_vectorizer to be configured."
            )
        if self.node_graph_vectorizer is None:
            raise ValueError(
                "ConditionalNodeFieldGraphGenerator.fit() requires node_graph_vectorizer to be configured."
            )
        if train_node_generator and self.conditional_node_generator_model is None:
            raise ValueError(
                "ConditionalNodeFieldGraphGenerator.fit() requires "
                "conditional_node_generator_model when train_node_generator=True."
            )
        if train_node_generator and self.graph_decoder is None:
            raise ValueError(
                "ConditionalNodeFieldGraphGenerator.fit() requires "
                "graph_decoder when train_node_generator=True."
            )

    def _sample_conditioning_rows(self, source: GraphConditioningBatch, indices: np.ndarray) -> GraphConditioningBatch:
        """Slice a conditioning batch by row indices."""
        return self._ensure_conditioning_sampler().sample_conditioning_rows(source, indices)

    def _interpolated_conditioning_from_pair(
        self,
        conditioning: GraphConditioningBatch,
        first_idx: int,
        second_idx: int,
        t: float,
    ) -> Tuple[np.ndarray, np.int64, np.int64]:
        """Linearly interpolate one conditioning pair and clamp integer counts."""
        return self._ensure_conditioning_sampler().interpolated_conditioning_from_pair(
            conditioning,
            first_idx,
            second_idx,
            t,
        )

    def _sample_conditions(
        self,
        n_samples: int,
        interpolate_between_n_samples: Optional[int] = None,
    ) -> GraphConditioningBatch:
        """Sample graph-level conditioning from cached training embeddings."""
        return self._ensure_conditioning_sampler().sample_conditions(
            n_samples,
            interpolate_between_n_samples=interpolate_between_n_samples,
        )

    def _reset_stream_fit_stats(self) -> None:
        self.stream_seen_ = 0
        self.stream_warmup_count_ = 0
        self.stream_training_seen_ = 0
        self.stream_training_accepted_ = 0
        self.stream_training_skipped_ = 0
        self.stream_epoch_training_seen_ = 0
        self.stream_epoch_training_accepted_ = 0
        self.stream_epoch_training_skipped_ = 0
        self.stream_skipped_too_large_ = 0
        self.stream_skipped_unknown_node_label_ = 0
        self.stream_skipped_unknown_edge_label_ = 0
        self.stream_skipped_transform_error_ = 0
        self.stream_skipped_supervision_error_ = 0
        self.stream_acceptance_rate_ = 0.0
        self.warmup_schema_frozen_ = False

    def _prepare_fit_artifacts(
        self,
        graphs: List[nx.Graph],
        targets: Optional[Sequence[Any]] = None,
    ) -> Dict[str, Any]:
        return build_fit_artifacts(
            self,
            graphs,
            targets=targets,
        )

    def _build_training_node_batch(
        self,
        graphs: List[nx.Graph],
        *,
        node_embeddings_list: List[np.ndarray],
        node_label_targets: List[np.ndarray],
        edge_label_targets: Optional[np.ndarray],
        edge_label_pairs: Optional[List[Tuple[int, int, int]]],
        supervision_plan,
        log_details: bool = True,
    ) -> NodeGenerationBatch:
        return self._ensure_node_batch_builder().build_training_node_batch(
            graphs,
            node_embeddings_list=node_embeddings_list,
            node_label_targets=node_label_targets,
            edge_label_targets=edge_label_targets,
            edge_label_pairs=edge_label_pairs,
            supervision_plan=supervision_plan,
            log_details=log_details,
        )

    def _stream_rejection_reason(self, graph: nx.Graph) -> Optional[str]:
        return self._ensure_stream_fit_service().stream_rejection_reason(graph)

    def _increment_stream_skip(self, reason: str) -> None:
        self._ensure_stream_fit_service().increment_stream_skip(reason)

    def _log_stream_skip(self, reason: str, graph: nx.Graph) -> None:
        self._ensure_stream_fit_service().log_stream_skip(reason, graph)

    def _finalize_stream_fit_stats(self) -> None:
        self._ensure_stream_fit_service().finalize_stream_fit_stats()

    def _prepare_stream_training_payload(self, graphs: List[nx.Graph]):
        return self._ensure_stream_fit_service().prepare_stream_training_payload(graphs)

    def _prepare_stream_training_batch(self, graphs: List[nx.Graph]):
        return self._ensure_stream_fit_service().prepare_stream_training_batch(graphs)

    def _prepare_stream_training_batch_with_timeout(self, graphs: List[nx.Graph]):
        return self._ensure_stream_fit_service().prepare_stream_training_batch_with_timeout(graphs)

    def fit_from_stream(
        self,
        uri,
        type,
        reader=None,
        warmup_size: int = 2048,
        batch_size: int = 128,
        limit=None,
        random_state=None,
        verbose: bool = False,
        start_after_instance: int = 0,
        train_node_generator: bool = True,
        ckpt_path: Optional[str] = None,
    ) -> 'ConditionalNodeFieldGraphGenerator':
        if int(warmup_size) < 1:
            raise ValueError("warmup_size must be >= 1.")
        self._require_fit_components(train_node_generator=train_node_generator)
        self._reset_stream_fit_stats()
        original_verbose = self.verbose
        original_stream_batch_timeout_seconds = getattr(self, "stream_batch_timeout_seconds", None)
        try:
            if verbose:
                self.verbose = verbose
            if getattr(self, "stream_batch_timeout_seconds", None) is None:
                self.stream_batch_timeout_seconds = 30.0
                verbose_log(
                    self,
                    "Streaming batch timeout is disabled; using the safe default "
                    f"{float(self.stream_batch_timeout_seconds):.1f}s for this fit so stalled "
                    "batch-preparation steps are skipped instead of blocking indefinitely.",
                    level=1,
                )
            source_selected_quota: Optional[int] = None
            if isinstance(limit, float) and 0.0 < float(limit) < 1.0:
                estimated_source_count = estimate_source_instance_count(uri, type)
                if estimated_source_count is not None:
                    source_selected_quota = max(
                        int(warmup_size) + int(batch_size),
                        int(round(float(limit) * float(estimated_source_count))),
                    )
                    verbose_log(
                        self,
                        f"Streaming Bernoulli quota: limit={float(limit):.3f}, "
                        f"source_count={estimated_source_count}, selected_per_epoch={source_selected_quota}.",
                        level=2,
                    )
            def _make_source_iter(*, source_start_after_instance: Optional[int] = None):
                resolved_start_after_instance = start_after_instance
                if source_start_after_instance is not None:
                    resolved_start_after_instance = int(source_start_after_instance)
                return iter_selected_source_graphs(
                    uri,
                    type,
                    reader=reader,
                    limit=limit,
                    random_state=random_state,
                    verbose=bool(verbose),
                    start_after_instance=resolved_start_after_instance,
                    max_selected=source_selected_quota,
                )

            source_iter = _make_source_iter()
            warmup_graphs = []
            for graph in source_iter:
                self.stream_seen_ += 1
                warmup_graphs.append(graph)
                if len(warmup_graphs) >= int(warmup_size):
                    break
            self.stream_warmup_count_ = len(warmup_graphs)
            if len(warmup_graphs) == 0:
                raise ValueError("fit_from_stream() could not load any graphs from the selected source.")
            verbose_log(self, f"Warmup fitting on {len(warmup_graphs)} streamed graphs.")
            artifacts = self._prepare_fit_artifacts(warmup_graphs, targets=None)

            if train_node_generator:
                warmup_node_batch = self._build_training_node_batch(
                    warmup_graphs,
                    node_embeddings_list=artifacts["node_embeddings_list"],
                    node_label_targets=artifacts["node_label_targets"],
                    edge_label_targets=artifacts["edge_label_targets"],
                    edge_label_pairs=artifacts["edge_label_pairs"],
                    supervision_plan=artifacts["supervision_plan"],
                    log_details=True,
                )
                verbose_log(
                    self,
                    f"Warmup schema frozen with up to {warmup_node_batch.node_presence_mask.shape[1]} nodes per graph.",
                )
                self.conditional_node_generator_model.setup(
                    node_batch=warmup_node_batch,
                    graph_conditioning=artifacts["graph_conditioning"],
                    targets=None,
                )
                setattr(self.conditional_node_generator_model, "_graph_generator_snapshot_owner", self)

                warmup_train_graphs = list(warmup_graphs)

                warmup_train_batches = []
                if warmup_train_graphs:
                    for start_idx in range(0, len(warmup_train_graphs), int(batch_size)):
                        warmup_batch_graphs = warmup_train_graphs[start_idx:start_idx + int(batch_size)]
                        warmup_train_batches.append(
                            (
                                int(len(warmup_batch_graphs)),
                                self._prepare_stream_training_batch(warmup_batch_graphs),
                            )
                        )

                def _consume_validation_batch(graph_iter):
                    active_batch = []
                    for graph in graph_iter:
                        self.stream_seen_ += 1
                        rejection_reason = self._stream_rejection_reason(graph)
                        if rejection_reason is not None:
                            continue
                        active_batch.append(graph)
                        if len(active_batch) < int(batch_size):
                            continue
                        try:
                            batch_payload = self._prepare_stream_training_batch_with_timeout(active_batch)
                        except _StreamBatchTimeoutError:
                            active_batch = []
                            continue
                        except _StreamTransformError:
                            active_batch = []
                            continue
                        except _StreamSupervisionError:
                            active_batch = []
                            continue
                        return active_batch
                    if active_batch:
                        try:
                            self._prepare_stream_training_batch_with_timeout(active_batch)
                        except _StreamBatchTimeoutError:
                            return None
                        except _StreamTransformError:
                            return None
                        except _StreamSupervisionError:
                            return None
                        return active_batch
                    return None

                def _iter_training_batches(graph_iter):
                    active_batch = []
                    consecutive_stalls = 0
                    for graph in graph_iter:
                        self.stream_seen_ += 1
                        self.stream_training_seen_ += 1
                        self.stream_epoch_training_seen_ += 1
                        rejection_reason = self._stream_rejection_reason(graph)
                        if rejection_reason is not None:
                            self._increment_stream_skip(rejection_reason)
                            self._log_stream_skip(rejection_reason, graph)
                            continue
                        active_batch.append(graph)
                        if len(active_batch) < int(batch_size):
                            continue
                        try:
                            batch_payload = self._prepare_stream_training_batch_with_timeout(active_batch)
                        except _StreamBatchTimeoutError:
                            consecutive_stalls += 1
                            self.stream_training_skipped_ += len(active_batch)
                            self.stream_epoch_training_skipped_ += len(active_batch)
                            self.stream_skipped_transform_error_ += len(active_batch)
                            verbose_log(
                                self,
                                f"Streamed batch preparation timed out; skipped {len(active_batch)} graphs "
                                f"(consecutive stalls={consecutive_stalls}/{self.stream_max_consecutive_stalls}).",
                                level=1,
                            )
                            if consecutive_stalls >= int(self.stream_max_consecutive_stalls):
                                raise RuntimeError(
                                    "Streaming training aborted after repeated batch-preparation stalls. "
                                    "Resume from the latest checkpoint."
                                )
                            active_batch = []
                            continue
                        except _StreamTransformError:
                            consecutive_stalls = 0
                            self.stream_training_skipped_ += len(active_batch)
                            self.stream_epoch_training_skipped_ += len(active_batch)
                            self.stream_skipped_transform_error_ += len(active_batch)
                            for rejected_graph in active_batch:
                                self._log_stream_skip("transform_error", rejected_graph)
                            active_batch = []
                            continue
                        except _StreamSupervisionError:
                            consecutive_stalls = 0
                            self.stream_training_skipped_ += len(active_batch)
                            self.stream_epoch_training_skipped_ += len(active_batch)
                            self.stream_skipped_supervision_error_ += len(active_batch)
                            for rejected_graph in active_batch:
                                self._log_stream_skip("supervision_error", rejected_graph)
                            active_batch = []
                            continue
                        consecutive_stalls = 0
                        self.stream_training_accepted_ += len(active_batch)
                        self.stream_epoch_training_accepted_ += len(active_batch)
                        yield batch_payload
                        active_batch = []
                    if active_batch:
                        try:
                            batch_payload = self._prepare_stream_training_batch_with_timeout(active_batch)
                        except _StreamBatchTimeoutError:
                            verbose_log(
                                self,
                                f"Trailing streamed batch preparation timed out; skipped {len(active_batch)} graphs.",
                                level=1,
                            )
                        except _StreamTransformError:
                            self.stream_training_skipped_ += len(active_batch)
                            self.stream_epoch_training_skipped_ += len(active_batch)
                            self.stream_skipped_transform_error_ += len(active_batch)
                            for rejected_graph in active_batch:
                                self._log_stream_skip("transform_error", rejected_graph)
                        except _StreamSupervisionError:
                            self.stream_training_skipped_ += len(active_batch)
                            self.stream_epoch_training_skipped_ += len(active_batch)
                            self.stream_skipped_supervision_error_ += len(active_batch)
                            for rejected_graph in active_batch:
                                self._log_stream_skip("supervision_error", rejected_graph)
                        else:
                            self.stream_training_accepted_ += len(active_batch)
                            self.stream_epoch_training_accepted_ += len(active_batch)
                            yield batch_payload

                validation_graphs = _consume_validation_batch(source_iter)
                if validation_graphs is None:
                    verbose_log(self, "No streamed validation batch was available after warmup; skipping node-model training.")
                else:
                    validation_node_embeddings_list, validation_graph_conditioning = self.encode(validation_graphs)
                    validation_node_label_targets = self.graphs_to_node_label_targets(validation_graphs)
                    validation_edge_label_targets, validation_edge_label_pairs = self.graphs_to_edge_label_targets(
                        validation_graphs
                    )
                    validation_node_batch = self._build_training_node_batch(
                        validation_graphs,
                        node_embeddings_list=validation_node_embeddings_list,
                        node_label_targets=validation_node_label_targets,
                        edge_label_targets=validation_edge_label_targets,
                        edge_label_pairs=validation_edge_label_pairs,
                        supervision_plan=artifacts["supervision_plan"],
                        log_details=False,
                    )

                    batch_state = {"epoch_call_count": 0}

                    def _single_pass_batch_factory():
                        batch_state["epoch_call_count"] += 1
                        self.stream_epoch_training_seen_ = 0
                        self.stream_epoch_training_accepted_ = 0
                        self.stream_epoch_training_skipped_ = 0
                        for warmup_batch_size, warmup_batch_payload in warmup_train_batches:
                            self.stream_training_seen_ += int(warmup_batch_size)
                            self.stream_epoch_training_seen_ += int(warmup_batch_size)
                            self.stream_training_accepted_ += int(warmup_batch_size)
                            self.stream_epoch_training_accepted_ += int(warmup_batch_size)
                            yield warmup_batch_payload

                        if batch_state["epoch_call_count"] == 1:
                            replay_tail_iter = source_iter
                        else:
                            replay_tail_iter = _make_source_iter(
                                source_start_after_instance=start_after_instance + int(self.stream_warmup_count_)
                            )
                            reserved_validation = _consume_validation_batch(replay_tail_iter)
                            if reserved_validation is None:
                                return
                        yield from _iter_training_batches(replay_tail_iter)

                    self.conditional_node_generator_model.fit_from_prebuilt_batches(
                        validation_node_batch=validation_node_batch,
                        validation_graph_conditioning=validation_graph_conditioning,
                        batch_iter_factory=_single_pass_batch_factory,
                        ckpt_path=ckpt_path,
                    )
            self.is_fitted_ = True
            self._finalize_stream_fit_stats()
            return self
        finally:
            self.stream_batch_timeout_seconds = original_stream_batch_timeout_seconds
            self.verbose = original_verbose

    @timeit
    def fit(
        self,
        graphs: List[nx.Graph],
        train_node_generator: bool = True,
        targets: Optional[Sequence[Any]] = None,
        ckpt_path: Optional[str] = None,
        sample_training_progress: bool = False,
        sample_training_progress_n_samples: int = 7,
        sample_training_progress_every_n_epochs: int = 1,
        sample_training_progress_pdf_path: Optional[str] = None,
        sample_training_progress_plot_kwargs: Optional[Dict[str, Any]] = None,
        sample_training_progress_plot_fn: Optional[Callable] = None,
    ) -> 'ConditionalNodeFieldGraphGenerator':
        """Fit vectorizers, derive supervision, and optionally train the node generator."""
        if int(sample_training_progress_n_samples) < 1:
            raise ValueError("sample_training_progress_n_samples must be >= 1.")
        if int(sample_training_progress_every_n_epochs) < 1:
            raise ValueError("sample_training_progress_every_n_epochs must be >= 1.")
        if self.model_name is not None:
            verbose_log(
                self,
                f"Fit target model_name={self.model_name} model_dir={self.model_dir}",
            )
        verbose_log(self, f"Fitting model on {len(graphs)} graphs")
        self._require_fit_components(train_node_generator=train_node_generator)
        if targets is not None and len(targets) != len(graphs):
            raise ValueError(
                "targets length must match the number of graphs "
                f"(got {len(targets)} targets for {len(graphs)} graphs)."
            )

        artifacts = self._prepare_fit_artifacts(graphs, targets=targets)
        node_label_targets = artifacts["node_label_targets"]
        edge_label_targets = artifacts["edge_label_targets"]
        edge_label_pairs = artifacts["edge_label_pairs"]
        supervision_plan = artifacts["supervision_plan"]
        node_embeddings_list = artifacts["node_embeddings_list"]
        graph_conditioning = artifacts["graph_conditioning"]

        if train_node_generator:
            edge_pairs_for_cond_gen = None
            edge_targets_for_cond_gen = None
            auxiliary_edge_pairs_for_cond_gen = None
            auxiliary_edge_targets_for_cond_gen = None
            if supervision_plan.direct_edges.enabled:
                if self.graph_decoder is None:
                    raise RuntimeError("Locality supervision requested but graph_decoder is None.")
                self._log_supervision_plan(supervision_plan)

                edge_targets_for_cond_gen, edge_pairs_for_cond_gen = self.graph_decoder.compute_edge_supervision(
                    graphs,
                    node_embeddings_list,
                    locality_sample_fraction=self.locality_sample_fraction,
                    negative_sample_factor=self.negative_sample_factor,
                    locality_sampling_strategy=self.locality_sampling_strategy,
                    locality_target_positive_ratio=self.locality_target_positive_ratio,
                    horizon=1,
                    supervision_name="direct_edge",
                )
                if supervision_plan.auxiliary_locality.enabled:
                    auxiliary_edge_targets_for_cond_gen, auxiliary_edge_pairs_for_cond_gen = (
                        self.graph_decoder.compute_edge_supervision(
                            graphs,
                            node_embeddings_list,
                            locality_sample_fraction=self.locality_sample_fraction,
                            negative_sample_factor=self.negative_sample_factor,
                            locality_sampling_strategy=self.locality_sampling_strategy,
                            locality_target_positive_ratio=self.locality_target_positive_ratio,
                            horizon=supervision_plan.auxiliary_locality.horizon,
                            supervision_name="aux_locality",
                        )
                    )
            else:
                self._log_supervision_plan(supervision_plan)

            node_batch = self._build_node_batch(
                graphs,
                node_embeddings_list,
                node_label_targets=node_label_targets if supervision_plan.node_labels.enabled else None,
                edge_pairs=edge_pairs_for_cond_gen,
                edge_targets=edge_targets_for_cond_gen,
                edge_label_pairs=edge_label_pairs if supervision_plan.edge_labels.enabled else None,
                edge_label_targets=edge_label_targets if supervision_plan.edge_labels.enabled else None,
                auxiliary_edge_pairs=auxiliary_edge_pairs_for_cond_gen,
                auxiliary_edge_targets=auxiliary_edge_targets_for_cond_gen,
            )
            verbose_log(
                self,
                f"Training conditional model on {len(node_batch)} graphs "
                f"with up to {node_batch.node_presence_mask.shape[1]} nodes each.",
            )
            if node_batch.edge_pairs is not None and node_batch.edge_targets is not None:
                verbose_log(self, f"Using direct-edge supervision with {len(node_batch.edge_pairs)} labelled pairs.")
            self.conditional_node_generator_model.setup(
                node_batch=node_batch,
                graph_conditioning=graph_conditioning,
                targets=targets,
            )
            setattr(self.conditional_node_generator_model, "_graph_generator_snapshot_owner", self)
            fit_kwargs = {
                "node_batch": node_batch,
                "graph_conditioning": graph_conditioning,
                "targets": targets,
            }
            fit_signature = inspect.signature(self.conditional_node_generator_model.fit)
            accepts_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in fit_signature.parameters.values()
            )
            if "ckpt_path" in fit_signature.parameters or accepts_kwargs:
                fit_kwargs["ckpt_path"] = ckpt_path
            sample_progress_attr = "_graph_generator_sample_progress_config"
            previous_sample_progress_config = getattr(
                self.conditional_node_generator_model,
                sample_progress_attr,
                None,
            )
            previous_sample_progress_config_present = hasattr(
                self.conditional_node_generator_model,
                sample_progress_attr,
            )
            try:
                setattr(
                    self.conditional_node_generator_model,
                    sample_progress_attr,
                    TrainingProgressSamplingConfig(
                        enabled=bool(sample_training_progress),
                        n_samples=int(sample_training_progress_n_samples),
                        every_n_epochs=int(sample_training_progress_every_n_epochs),
                        output_path=self._resolve_sample_training_progress_pdf_path(
                            sample_training_progress_pdf_path
                        ),
                        plot_kwargs=dict(sample_training_progress_plot_kwargs or {}),
                        plot_fn=sample_training_progress_plot_fn,
                    ),
                )
                self.conditional_node_generator_model.fit(
                    **fit_kwargs,
                )
            finally:
                if previous_sample_progress_config_present:
                    setattr(
                        self.conditional_node_generator_model,
                        sample_progress_attr,
                        previous_sample_progress_config,
                    )
                else:
                    try:
                        delattr(self.conditional_node_generator_model, sample_progress_attr)
                    except AttributeError:
                        pass

        self.is_fitted_ = True
        return self

    def _resolve_sample_training_progress_pdf_path(
        self,
        sample_training_progress_pdf_path: Optional[str],
    ) -> str:
        if sample_training_progress_pdf_path is not None:
            return os.path.expanduser(str(sample_training_progress_pdf_path))
        artifact_root = getattr(
            self.conditional_node_generator_model,
            "artifact_root_dir",
            None,
        )
        if artifact_root is None:
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            artifact_root = os.path.join(repo_root, ".artifacts")
        name = self.model_name or self.__class__.__name__
        sample_dir = os.path.join(str(artifact_root), "samples", sanitize_model_token(str(name)))
        return os.path.join(sample_dir, "training_samples.pdf")

    def set_guidance_predictor(
        self,
        mode: str,
        output_dimension: Optional[int] = None,
        hidden_dimension: Optional[int] = None,
    ) -> None:
        self._require_fitted_for_generation()
        if self.conditional_node_generator_model is None:
            raise RuntimeError("conditional_node_generator_model is None.")
        self.conditional_node_generator_model.set_guidance_predictor(
            mode=mode,
            output_dimension=output_dimension,
            hidden_dimension=hidden_dimension,
        )

    def set_guidance_classifier(self, num_classes: int, hidden_dimension: Optional[int] = None) -> None:
        self.set_guidance_predictor(
            mode="classification",
            output_dimension=int(num_classes),
            hidden_dimension=hidden_dimension,
        )

    def train_guidance_predictor(
        self,
        graphs: List[nx.Graph],
        targets: Sequence[Any],
        mode: Optional[str] = None,
        learning_rate: float = 1e-3,
        maximum_epochs: int = 30,
        batch_size: Optional[int] = None,
        noise_scale: Optional[float] = None,
    ) -> None:
        self._require_fitted_for_generation()
        if self.conditional_node_generator_model is None:
            raise RuntimeError("conditional_node_generator_model is None.")
        node_embeddings_list, graph_conditioning = self.encode(graphs)
        node_batch = self._build_node_batch(graphs, node_embeddings_list)
        self.conditional_node_generator_model.train_guidance_predictor(
            node_batch=node_batch,
            graph_conditioning=graph_conditioning,
            targets=targets,
            mode=mode,
            learning_rate=learning_rate,
            maximum_epochs=maximum_epochs,
            batch_size=batch_size,
            noise_scale=noise_scale,
        )

    def train_guidance_predictor_from_embeddings(
        self,
        node_embeddings_list: List[np.ndarray],
        graph_conditioning: GraphConditioningBatch,
        targets: Sequence[Any],
        mode: Optional[str] = None,
        learning_rate: float = 1e-3,
        maximum_epochs: int = 30,
        batch_size: Optional[int] = None,
        noise_scale: Optional[float] = None,
    ) -> None:
        self._require_fitted_for_generation()
        if self.conditional_node_generator_model is None:
            raise RuntimeError("conditional_node_generator_model is None.")
        self.conditional_node_generator_model.train_guidance_predictor_from_embeddings(
            node_embeddings_list=node_embeddings_list,
            graph_conditioning=graph_conditioning,
            targets=targets,
            mode=mode,
            learning_rate=learning_rate,
            maximum_epochs=maximum_epochs,
            batch_size=batch_size,
            noise_scale=noise_scale,
        )

    def train_guidance_classifier(
        self,
        graphs: List[nx.Graph],
        targets: Sequence[Any],
        learning_rate: float = 1e-3,
        maximum_epochs: int = 30,
        batch_size: Optional[int] = None,
        noise_scale: Optional[float] = None,
    ) -> None:
        self.train_guidance_predictor(
            graphs=graphs,
            targets=targets,
            mode="classification",
            learning_rate=learning_rate,
            maximum_epochs=maximum_epochs,
            batch_size=batch_size,
            noise_scale=noise_scale,
        )

    @staticmethod
    def _to_numpy_2d(matrix) -> np.ndarray:
        return EncodingPipeline.to_numpy_2d(matrix)

    @staticmethod
    def _stack_embedding_rows(embeddings: Sequence[Any]):
        return EncodingPipeline.stack_embedding_rows(embeddings)

    @staticmethod
    def _feature_dimension(matrix) -> int:
        return EncodingPipeline.feature_dimension(matrix)

    def _resolved_graph_embedding_svd_dimension(self) -> int:
        return self._ensure_encoding_pipeline().resolved_graph_embedding_svd_dimension()

    def _raw_node_encode(self, graphs: List[nx.Graph]) -> List[Any]:
        return self._ensure_encoding_pipeline().raw_node_encode(graphs)

    def _raw_graph_encode(self, graphs: List[nx.Graph]):
        return self._ensure_encoding_pipeline().raw_graph_encode(graphs)

    def _fit_single_embedding_svd(self, matrix, requested_dimension: int, label: str):
        return self._ensure_encoding_pipeline().fit_single_embedding_svd(
            matrix,
            requested_dimension,
            label,
        )

    def _fit_embedding_svds(self, raw_node_embeddings_list: List[Any], raw_graph_embeddings) -> None:
        self._ensure_encoding_pipeline().fit_embedding_svds(
            raw_node_embeddings_list,
            raw_graph_embeddings,
        )

    def _compress_node_embeddings(self, raw_node_embeddings_list: List[Any]) -> List[np.ndarray]:
        return self._ensure_encoding_pipeline().compress_node_embeddings(raw_node_embeddings_list)

    def _compress_graph_embeddings(self, raw_graph_embeddings) -> np.ndarray:
        return self._ensure_encoding_pipeline().compress_graph_embeddings(raw_graph_embeddings)

    def _build_graph_conditioning_from_raw(self, graphs: List[nx.Graph], raw_graph_embeddings) -> GraphConditioningBatch:
        return self._ensure_encoding_pipeline().build_graph_conditioning_from_raw(
            graphs,
            raw_graph_embeddings,
        )

    @timeit
    def node_encode(self, graphs: List[nx.Graph]) -> List[np.ndarray]:
        """Encode each input graph into a per-node embedding matrix."""
        return self._ensure_encoding_pipeline().node_encode(graphs)

    @timeit
    def graph_encode(self, graphs: List[nx.Graph]) -> GraphConditioningBatch:
        """Encode graphs into graph-level conditioning vectors plus node and edge counts."""
        return self._ensure_encoding_pipeline().graph_encode(graphs)

    def encode(self, graphs: List[nx.Graph]) -> Tuple[List[np.ndarray], GraphConditioningBatch]:
        """Return both node embeddings and graph-level conditioning for the same graph batch."""
        return self._ensure_encoding_pipeline().encode(graphs)

    def graphs_to_node_label_targets(self, graphs: List[nx.Graph]) -> List[np.ndarray]:
        """Extract node labels in graph iteration order, or emit a shared dummy label when unlabeled."""
        saw_any_node_label = False
        saw_missing_node_label = False
        node_label_targets = []
        for graph in graphs:
            labels = []
            for node in graph.nodes():
                label = graph.nodes[node].get("label")
                if label is None:
                    saw_missing_node_label = True
                else:
                    saw_any_node_label = True
                labels.append(label)
            node_label_targets.append(np.asarray(labels, dtype=object))

        if saw_any_node_label and saw_missing_node_label:
            raise ValueError(
                "Node labels must be either present for every node in every training graph or absent for all nodes."
            )

        if not saw_any_node_label:
            return [
                np.asarray([DEFAULT_DUMMY_NODE_LABEL] * len(labels), dtype=object)
                for labels in node_label_targets
            ]

        return node_label_targets

    def _graphs_have_usable_edge_labels(self, graphs: List[nx.Graph]) -> bool:
        """Return ``True`` only when at least one edge exists and every observed edge is labeled."""
        saw_any_edge = False
        for graph in graphs:
            for u, v, attrs in graph.edges(data=True):
                saw_any_edge = True
                if "label" not in attrs:
                    return False
        return saw_any_edge

    def graphs_to_edge_label_targets(
        self,
        graphs: List[nx.Graph],
    ) -> Tuple[Optional[np.ndarray], Optional[List[Tuple[int, int, int]]]]:
        """Extract edge-label targets and graph-local node-pair indices for decoder supervision."""
        if not self._graphs_have_usable_edge_labels(graphs):
            if self.verbose:
                verbose_log(
                    self,
                    "Edge-label channel disabled at graph inspection time: no usable edge labels were found.",
                    level=1,
                )
            return None, None
        edge_label_targets = []
        edge_label_pairs = []
        for graph_idx, graph in enumerate(graphs):
            node_to_index = {node: idx for idx, node in enumerate(graph.nodes())}
            for u, v, attrs in graph.edges(data=True):
                i = node_to_index[u]
                j = node_to_index[v]
                label = attrs["label"]
                edge_label_pairs.append((graph_idx, i, j))
                edge_label_targets.append(label)
                if not graph.is_directed():
                    edge_label_pairs.append((graph_idx, j, i))
                    edge_label_targets.append(label)
        return np.asarray(edge_label_targets, dtype=object), edge_label_pairs

    def _build_node_batch(
        self,
        graphs: List[nx.Graph],
        node_embeddings_list: List[np.ndarray],
        node_label_targets: Optional[List[np.ndarray]] = None,
        edge_pairs: Optional[List[Tuple[int, int, int]]] = None,
        edge_targets: Optional[np.ndarray] = None,
        edge_label_pairs: Optional[List[Tuple[int, int, int]]] = None,
        edge_label_targets: Optional[np.ndarray] = None,
        auxiliary_edge_pairs: Optional[List[Tuple[int, int, int]]] = None,
        auxiliary_edge_targets: Optional[np.ndarray] = None,
    ) -> NodeGenerationBatch:
        """Assemble a padded node-generation batch from embeddings and supervision signals."""
        return self._ensure_node_batch_builder().build_node_batch(
            graphs,
            node_embeddings_list,
            node_label_targets=node_label_targets,
            edge_pairs=edge_pairs,
            edge_targets=edge_targets,
            edge_label_pairs=edge_label_pairs,
            edge_label_targets=edge_label_targets,
            auxiliary_edge_pairs=auxiliary_edge_pairs,
            auxiliary_edge_targets=auxiliary_edge_targets,
        )

    def _log_generated_batch_info(
        self,
        graph_conditioning: GraphConditioningBatch,
        generated_nodes: GeneratedNodeBatch,
    ) -> None:
        """Print per-graph generation summaries at the highest verbosity level.

        Args:
            graph_conditioning (GraphConditioningBatch): Input value.
            generated_nodes (GeneratedNodeBatch): Input value.
        """
        if int(self.verbose) < 3:
            return
        total_graphs = len(generated_nodes)
        for graph_idx in range(total_graphs):
            node_row_count = (
                int(generated_nodes.node_presence_mask.shape[1])
                if generated_nodes.node_presence_mask is not None
                else int(generated_nodes.node_degree_predictions.shape[1])
            )
            raw_predicted_node_count = (
                int(np.sum(generated_nodes.node_presence_mask[graph_idx][:node_row_count]))
                if generated_nodes.node_presence_mask is not None
                else node_row_count
            )
            conditioning_node_count = int(graph_conditioning.node_counts[graph_idx])
            conditioning_edge_count = int(graph_conditioning.edge_counts[graph_idx])
            decoded_support_node_count = None
            if (
                self.graph_decoder is not None
                and generated_nodes.node_presence_mask is not None
            ):
                decoded_support_mask = self.graph_decoder.resolve_node_presence_mask(
                    np.asarray(generated_nodes.node_presence_mask[graph_idx][:node_row_count], dtype=bool),
                    desired_node_count=conditioning_node_count,
                    node_existence_scores=None if generated_nodes.node_existence_probabilities is None else np.asarray(
                        generated_nodes.node_existence_probabilities[graph_idx][:node_row_count],
                        dtype=float,
                    ),
                )
                decoded_support_node_count = int(np.sum(decoded_support_mask))
            message = (
                f"Generated graph {graph_idx + 1}/{total_graphs}: "
                f"conditioning_nodes={conditioning_node_count}, "
                f"conditioning_edges={conditioning_edge_count}, "
                f"raw_predicted_nodes={raw_predicted_node_count}"
            )
            if decoded_support_node_count is not None:
                message += f", decoded_support_nodes={decoded_support_node_count}"
            if generated_nodes.node_degree_predictions is not None:
                valid_deg = np.asarray(
                    generated_nodes.node_degree_predictions[graph_idx][:node_row_count],
                    dtype=float,
                )
                if generated_nodes.node_presence_mask is not None:
                    valid_mask = np.asarray(
                        generated_nodes.node_presence_mask[graph_idx][:node_row_count],
                        dtype=bool,
                    )
                    valid_deg = valid_deg[valid_mask]
                if valid_deg.size > 0:
                    message += (
                        f", mean_degree={float(np.mean(valid_deg)):.2f}, "
                        f"max_degree={int(np.max(valid_deg))}"
                    )
            if generated_nodes.edge_probability_matrices is not None:
                edge_probs = np.asarray(generated_nodes.edge_probability_matrices[graph_idx], dtype=float)
                off_diag = edge_probs[~np.eye(edge_probs.shape[0], dtype=bool)]
                if off_diag.size > 0:
                    message += (
                        f", mean_edge_prob={float(np.mean(off_diag)):.3f}, "
                        f"max_edge_prob={float(np.max(off_diag)):.3f}"
                    )
            verbose_log(self, message, level=3)
            if generated_nodes.node_labels is not None:
                labels = np.asarray(generated_nodes.node_labels[graph_idx], dtype=object)
                if generated_nodes.node_presence_mask is not None:
                    valid_mask = np.asarray(
                        generated_nodes.node_presence_mask[graph_idx][: labels.shape[0]],
                        dtype=bool,
                    )
                    labels = labels[valid_mask]
                if labels.size > 0:
                    unique_labels, counts = np.unique(labels, return_counts=True)
                    label_summary = {label: int(count) for label, count in zip(unique_labels.tolist(), counts.tolist())}
                    verbose_log(self, f"  node_labels={label_summary}", level=3)

    @staticmethod
    def _slice_graph_conditioning(
        graph_conditioning: GraphConditioningBatch,
        indices: Sequence[int],
    ) -> GraphConditioningBatch:
        """Select a subset of conditioning rows by integer indices.

        Args:
            graph_conditioning (GraphConditioningBatch): Input value.
            indices (Sequence[int]): Input value.

        Returns:
            GraphConditioningBatch: Computed result.
        """
        idx = np.asarray(indices, dtype=np.int64)
        return graph_conditioning.take(idx)

    @staticmethod
    def _repeat_graph_conditioning(
        graph_conditioning: GraphConditioningBatch,
        repeats: int,
    ) -> GraphConditioningBatch:
        """Repeat each conditioning row a fixed number of times.

        Args:
            graph_conditioning (GraphConditioningBatch): Input value.
            repeats (int): Input value.

        Returns:
            GraphConditioningBatch: Computed result.
        """
        if repeats < 1:
            raise ValueError("repeats must be >= 1")
        return graph_conditioning.repeat(repeats)

    @staticmethod
    def _accept_feasible_candidates_by_slot(
        decoded_graphs: Sequence[nx.Graph],
        feasibility_mask: Sequence[bool],
        candidate_slot_indices: Sequence[int],
        accepted_graphs_by_slot: List[Optional[nx.Graph]],
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[int, int]:
        """Count all feasible candidates, then fill each empty slot with one random feasible graph."""
        return _accept_feasible_candidates_by_slot(
            decoded_graphs=decoded_graphs,
            feasibility_mask=feasibility_mask,
            candidate_slot_indices=candidate_slot_indices,
            accepted_graphs_by_slot=accepted_graphs_by_slot,
            rng=rng,
        )

    def _decode_generated_nodes(
        self,
        generated_nodes: GeneratedNodeBatch,
        graph_conditioning: Optional[GraphConditioningBatch] = None,
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
        attempt_idx: int = 0,
        use_ilp_decoder: bool = True,
        edge_probability_threshold: Optional[float] = None,
    ) -> List[nx.Graph]:
        return self.decode_service_.decode_generated_nodes(
            generated_nodes,
            graph_conditioning=graph_conditioning,
            feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
            attempt_idx=attempt_idx,
            use_ilp_decoder=use_ilp_decoder,
            edge_probability_threshold=edge_probability_threshold,
        )

    def _decode_with_feasibility_slots(
        self,
        graph_conditioning: GraphConditioningBatch,
        desired_target: Optional[Union[int, float, Sequence[Any]]] = None,
        guidance_scale: float = 1.0,
        apply_feasibility_filtering: Optional[bool] = None,
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
        feasibility_effort: Optional[int] = None,
        use_ilp_decoder: bool = True,
        edge_probability_threshold: Optional[float] = None,
    ) -> List[Optional[nx.Graph]]:
        with self._feasibility_effort_context(
            feasibility_effort,
            apply_feasibility_filtering=apply_feasibility_filtering,
            feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
        ) as effort_profile:
            return self.decode_service_.decode_with_feasibility_slots(
                graph_conditioning,
                sampling_mode="unguided",
                desired_target=desired_target,
                guidance_scale=guidance_scale,
                apply_feasibility_filtering=(
                    effort_profile.apply_feasibility_filtering
                    if effort_profile is not None
                    else apply_feasibility_filtering
                ),
                feasibility_oracle_candidates_per_attempt=(
                    int(self.feasibility_oracle_candidates_per_attempt)
                    if effort_profile is not None
                    else feasibility_oracle_candidates_per_attempt
                ),
                use_ilp_decoder=use_ilp_decoder,
                edge_probability_threshold=edge_probability_threshold,
            )

    def _predict_generated_nodes(
        self,
        graph_conditioning: GraphConditioningBatch,
        sampling_mode: str = "unguided",
        desired_target: Optional[Union[int, float, Sequence[Any]]] = None,
        guidance_scale: float = 1.0,
        predictor_scale: float = 1.0,
    ) -> GeneratedNodeBatch:
        if sampling_mode == "unguided":
            generated_nodes = self.conditional_node_generator_model.predict(
                graph_conditioning,
                desired_target=desired_target,
                guidance_scale=guidance_scale,
            )
        elif sampling_mode == "regression_guided":
            generated_nodes = self.conditional_node_generator_model.predict_regression_guided(
                graph_conditioning,
                desired_target=1.0 if desired_target is None else desired_target,
                predictor_scale=predictor_scale,
            )
        else:
            raise ValueError(
                "sampling_mode must be 'unguided' or 'regression_guided' "
                f"(got {sampling_mode!r})."
            )
        self._log_generated_batch_info(graph_conditioning, generated_nodes)
        return generated_nodes

    def _sample_training_decode_variants(
        self,
        n_samples: int,
    ) -> Dict[str, List[Optional[nx.Graph]]]:
        """Compatibility alias for older notebooks."""
        return self.sample(n_samples=n_samples, return_decode_stages=True)

    @staticmethod
    def _compute_guidance_targets(violation_counts: Sequence[Any]) -> np.ndarray:
        violations = np.asarray(violation_counts, dtype=float)
        return 1.0 / (1.0 + np.sqrt(violations))

    @staticmethod
    def _empty_generated_guidance_batch() -> GeneratedGuidanceBatch:
        empty_conditioning = GraphConditioningBatch(
            graph_embeddings=np.zeros((0, 0), dtype=float),
            node_counts=np.zeros((0,), dtype=np.int64),
            edge_counts=np.zeros((0,), dtype=np.int64),
        )
        return GeneratedGuidanceBatch(
            node_embeddings_list=[],
            graph_conditioning=empty_conditioning,
            decoded_graphs=[],
            violation_counts=np.zeros((0,), dtype=np.int64),
            guidance_targets=np.zeros((0,), dtype=float),
            feasible_mask=np.zeros((0,), dtype=bool),
            sampling_mode="unguided",
        )

    @staticmethod
    def _slice_generated_guidance_batch(
        batch: GeneratedGuidanceBatch,
        indices: Sequence[int],
    ) -> GeneratedGuidanceBatch:
        indices_array = np.asarray(indices, dtype=np.int64)
        return GeneratedGuidanceBatch(
            node_embeddings_list=[batch.node_embeddings_list[int(idx)] for idx in indices_array],
            graph_conditioning=batch.graph_conditioning.take(indices_array),
            decoded_graphs=[batch.decoded_graphs[int(idx)] for idx in indices_array],
            violation_counts=np.asarray(batch.violation_counts)[indices_array],
            guidance_targets=np.asarray(batch.guidance_targets)[indices_array],
            feasible_mask=np.asarray(batch.feasible_mask)[indices_array],
            sampling_mode=batch.sampling_mode,
        )

    @classmethod
    def _concat_generated_guidance_batches(
        cls,
        batches: Sequence[GeneratedGuidanceBatch],
    ) -> GeneratedGuidanceBatch:
        non_empty_batches = [batch for batch in batches if len(batch) > 0]
        if not non_empty_batches:
            return cls._empty_generated_guidance_batch()
        graph_embeddings = [np.asarray(batch.graph_conditioning.graph_embeddings) for batch in non_empty_batches]
        node_counts = [np.asarray(batch.graph_conditioning.node_counts) for batch in non_empty_batches]
        edge_counts = [np.asarray(batch.graph_conditioning.edge_counts) for batch in non_empty_batches]
        condition_node_embeddings = None
        if all(
            batch.graph_conditioning.condition_node_embeddings is not None
            for batch in non_empty_batches
        ):
            if all(
                isinstance(batch.graph_conditioning.condition_node_embeddings, np.ndarray)
                for batch in non_empty_batches
            ):
                condition_node_embeddings = np.concatenate(
                    [
                        np.asarray(batch.graph_conditioning.condition_node_embeddings)
                        for batch in non_empty_batches
                    ],
                    axis=0,
                )
            else:
                condition_node_embeddings = [
                    np.asarray(embedding, dtype=float)
                    for batch in non_empty_batches
                    for embedding in (
                        list(batch.graph_conditioning.condition_node_embeddings)
                        if not isinstance(batch.graph_conditioning.condition_node_embeddings, np.ndarray)
                        else list(np.asarray(batch.graph_conditioning.condition_node_embeddings))
                    )
                ]
        condition_node_presence_mask = None
        if all(
            batch.graph_conditioning.condition_node_presence_mask is not None
            for batch in non_empty_batches
        ):
            condition_node_presence_mask = np.concatenate(
                [
                    np.asarray(batch.graph_conditioning.condition_node_presence_mask)
                    for batch in non_empty_batches
                ],
                axis=0,
            )
        return GeneratedGuidanceBatch(
            node_embeddings_list=[
                np.asarray(embedding, dtype=float)
                for batch in non_empty_batches
                for embedding in batch.node_embeddings_list
            ],
            graph_conditioning=GraphConditioningBatch(
                graph_embeddings=np.concatenate(graph_embeddings, axis=0),
                node_counts=np.concatenate(node_counts, axis=0),
                edge_counts=np.concatenate(edge_counts, axis=0),
                condition_node_embeddings=condition_node_embeddings,
                condition_node_presence_mask=condition_node_presence_mask,
            ),
            decoded_graphs=[
                graph
                for batch in non_empty_batches
                for graph in batch.decoded_graphs
            ],
            violation_counts=np.concatenate(
                [np.asarray(batch.violation_counts, dtype=np.int64) for batch in non_empty_batches],
                axis=0,
            ),
            guidance_targets=np.concatenate(
                [np.asarray(batch.guidance_targets, dtype=float) for batch in non_empty_batches],
                axis=0,
            ),
            feasible_mask=np.concatenate(
                [np.asarray(batch.feasible_mask, dtype=bool) for batch in non_empty_batches],
                axis=0,
            ),
            sampling_mode="mixed" if len(non_empty_batches) > 1 else non_empty_batches[0].sampling_mode,
        )

    @staticmethod
    def build_guidance_violation_buckets(
        violation_counts: Sequence[Any],
        positive_bucket_count: int = 8,
    ) -> List[Dict[str, Any]]:
        violations = np.asarray(violation_counts, dtype=float).reshape(-1)
        if violations.size == 0:
            return []

        bucket_specs: List[Dict[str, Any]] = []
        zero_indices = np.flatnonzero(violations == 0)
        if zero_indices.size > 0:
            bucket_specs.append(
                {
                    "label": "feasible",
                    "lower": 0.0,
                    "upper": 0.0,
                    "indices": zero_indices.astype(np.int64),
                }
            )

        positive_indices = np.flatnonzero(violations > 0)
        if positive_indices.size == 0:
            return bucket_specs

        positive_values = violations[positive_indices]
        distinct_positive = np.unique(positive_values)
        quantile_bucket_count = max(1, min(int(positive_bucket_count), int(distinct_positive.size)))
        quantile_edges = np.quantile(
            positive_values,
            np.linspace(0.0, 1.0, quantile_bucket_count + 1),
        )
        quantile_edges = np.unique(np.asarray(quantile_edges, dtype=float))
        if quantile_edges.size < 2:
            quantile_edges = np.asarray([positive_values.min(), positive_values.max()], dtype=float)

        for bucket_idx in range(quantile_edges.size - 1):
            lower = float(quantile_edges[bucket_idx])
            upper = float(quantile_edges[bucket_idx + 1])
            if bucket_idx == quantile_edges.size - 2:
                mask = (positive_values >= lower) & (positive_values <= upper)
            else:
                mask = (positive_values >= lower) & (positive_values < upper)
            bucket_indices = positive_indices[np.flatnonzero(mask)]
            if bucket_indices.size == 0:
                continue
            bucket_specs.append(
                {
                    "label": f"q{bucket_idx + 1}: [{lower:.0f}, {upper:.0f}]",
                    "lower": lower,
                    "upper": upper,
                    "indices": bucket_indices.astype(np.int64),
                }
            )
        if len(bucket_specs) == 0:
            bucket_specs.append(
                {
                    "label": "all",
                    "lower": float(np.min(violations)),
                    "upper": float(np.max(violations)),
                    "indices": np.arange(len(violations), dtype=np.int64),
                }
            )
        return bucket_specs

    @classmethod
    def _summarize_violation_buckets(
        cls,
        batch: GeneratedGuidanceBatch,
        positive_bucket_count: int = 8,
    ) -> List[Dict[str, Any]]:
        summaries = []
        for bucket in cls.build_guidance_violation_buckets(
            batch.violation_counts,
            positive_bucket_count=positive_bucket_count,
        ):
            bucket_indices = np.asarray(bucket["indices"], dtype=np.int64)
            bucket_violations = np.asarray(batch.violation_counts, dtype=float)[bucket_indices]
            bucket_targets = np.asarray(batch.guidance_targets, dtype=float)[bucket_indices]
            summaries.append(
                {
                    "label": bucket["label"],
                    "lower": bucket["lower"],
                    "upper": bucket["upper"],
                    "indices": bucket_indices,
                    "count": int(bucket_indices.size),
                    "median_violation": float(np.median(bucket_violations)),
                    "median_target": float(np.median(bucket_targets)),
                }
            )
        return summaries

    @classmethod
    def _sample_bucket_indices(
        cls,
        batch: GeneratedGuidanceBatch,
        sample_size: Optional[int] = None,
        positive_bucket_count: int = 8,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        total_examples = len(batch)
        if total_examples == 0:
            return np.zeros((0,), dtype=np.int64)
        if sample_size is None or int(sample_size) >= total_examples:
            return np.arange(total_examples, dtype=np.int64)
        bucket_summaries = cls._summarize_violation_buckets(
            batch,
            positive_bucket_count=positive_bucket_count,
        )
        if len(bucket_summaries) <= 1:
            return np.arange(total_examples, dtype=np.int64)

        rng = np.random.default_rng(random_state)
        non_empty_buckets = [bucket for bucket in bucket_summaries if int(bucket["count"]) > 0]
        per_bucket = int(sample_size) // len(non_empty_buckets)
        remainder = int(sample_size) % len(non_empty_buckets)
        sampled_indices = []
        for bucket_idx, bucket in enumerate(non_empty_buckets):
            desired = per_bucket + (1 if bucket_idx < remainder else 0)
            if desired <= 0:
                continue
            bucket_indices = np.asarray(bucket["indices"], dtype=np.int64)
            replace = desired > bucket_indices.size
            chosen = rng.choice(bucket_indices, size=desired, replace=replace)
            sampled_indices.extend(np.asarray(chosen, dtype=np.int64).tolist())
        return np.asarray(sampled_indices, dtype=np.int64)

    def collect_generated_guidance_examples(
        self,
        n_samples: int,
        interpolate_between_n_samples: Optional[int] = None,
        sampling_mode: str = "unguided",
        desired_target: Optional[float] = None,
        guidance_scale: float = 1.0,
        predictor_scale: float = 1.0,
    ) -> GeneratedGuidanceBatch:
        self._require_fitted_for_generation()
        if self.feasibility_estimator is None:
            raise RuntimeError(
                "collect_generated_guidance_examples() requires feasibility_estimator to be configured."
            )
        if not hasattr(self.feasibility_estimator, "number_of_violations"):
            raise RuntimeError(
                "collect_generated_guidance_examples() requires feasibility_estimator.number_of_violations()."
            )
        graph_conditioning = self._sample_conditions(
            int(n_samples),
            interpolate_between_n_samples=interpolate_between_n_samples,
        )
        generated_nodes = self._predict_generated_nodes(
            graph_conditioning,
            sampling_mode=sampling_mode,
            desired_target=desired_target,
            guidance_scale=guidance_scale,
            predictor_scale=predictor_scale,
        )
        if generated_nodes.node_embeddings_list is None:
            raise RuntimeError("Generated node embeddings are unavailable for guidance collection.")
        decoded_graphs = self.decode_service_.decode_generated_nodes(generated_nodes)
        try:
            raw_violation_counts = self.feasibility_estimator.number_of_violations(decoded_graphs)
        except AttributeError as exc:
            if "has no attribute 'get'" not in str(exc):
                raise
            raw_violation_counts = self.feasibility_estimator.number_of_violations(
                [graph.graph if hasattr(graph, "graph") else graph for graph in decoded_graphs]
            )
        violation_counts = np.asarray(raw_violation_counts, dtype=np.int64).reshape(-1)
        if violation_counts.shape[0] != len(decoded_graphs):
            raise RuntimeError(
                "Feasibility estimator returned an unexpected number of violation counts "
                f"({violation_counts.shape[0]} for {len(decoded_graphs)} graphs)."
            )
        guidance_targets = self._compute_guidance_targets(violation_counts)
        feasible_mask = np.asarray(violation_counts == 0, dtype=bool)
        return GeneratedGuidanceBatch(
            node_embeddings_list=[
                np.asarray(embedding, dtype=float)
                for embedding in generated_nodes.node_embeddings_list
            ],
            graph_conditioning=graph_conditioning,
            decoded_graphs=decoded_graphs,
            violation_counts=violation_counts,
            guidance_targets=guidance_targets,
            feasible_mask=feasible_mask,
            sampling_mode=str(sampling_mode),
        )

    def bootstrap_guidance_regressor_from_generated(
        self,
        num_cycles: int,
        examples_per_cycle: int,
        interpolate_between_n_samples: Optional[int] = None,
        replay_train_size: Optional[int] = None,
        positive_bucket_count: int = 8,
        guided_fraction_after_first_cycle: float = 0.5,
        guided_target: float = 1.0,
        guidance_learning_rate: float = 1e-3,
        guidance_maximum_epochs: int = 30,
        guidance_batch_size: Optional[int] = None,
        guidance_noise_scale: Optional[float] = None,
        predictor_scale: float = 1.0,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        if int(num_cycles) < 1:
            raise ValueError("num_cycles must be >= 1.")
        if int(examples_per_cycle) < 1:
            raise ValueError("examples_per_cycle must be >= 1.")

        cycle_batches: List[GeneratedGuidanceBatch] = []
        replay_batches: List[GeneratedGuidanceBatch] = []
        history: List[Dict[str, Any]] = []
        guidance_ready = bool(
            getattr(self.conditional_node_generator_model, "guidance_predictor_", None) is not None
            and getattr(self.conditional_node_generator_model, "guidance_predictor_mode_", None) == "regression"
        )

        for cycle_idx in range(int(num_cycles)):
            if cycle_idx == 0:
                unguided_count = int(examples_per_cycle)
                guided_count = 0
            else:
                guided_count = int(round(float(examples_per_cycle) * float(guided_fraction_after_first_cycle)))
                guided_count = max(0, min(int(examples_per_cycle), guided_count))
                if not guidance_ready:
                    guided_count = 0
                unguided_count = int(examples_per_cycle) - guided_count

            new_batches = []
            if unguided_count > 0:
                new_batches.append(
                    self.collect_generated_guidance_examples(
                        n_samples=unguided_count,
                        interpolate_between_n_samples=interpolate_between_n_samples,
                        sampling_mode="unguided",
                    )
                )
            if guided_count > 0:
                new_batches.append(
                    self.collect_generated_guidance_examples(
                        n_samples=guided_count,
                        interpolate_between_n_samples=interpolate_between_n_samples,
                        sampling_mode="regression_guided",
                        desired_target=guided_target,
                        predictor_scale=predictor_scale,
                    )
                )

            cycle_batch = self._concat_generated_guidance_batches(new_batches)
            cycle_batches.append(cycle_batch)
            replay_batches.append(cycle_batch)
            replay_buffer = self._concat_generated_guidance_batches(replay_batches)
            replay_bucket_summaries = self._summarize_violation_buckets(
                replay_buffer,
                positive_bucket_count=positive_bucket_count,
            )
            cycle_bucket_summaries = self._summarize_violation_buckets(
                cycle_batch,
                positive_bucket_count=positive_bucket_count,
            )
            train_indices = self._sample_bucket_indices(
                replay_buffer,
                sample_size=replay_train_size,
                positive_bucket_count=positive_bucket_count,
                random_state=None if random_state is None else int(random_state) + cycle_idx,
            )
            train_batch = self._slice_generated_guidance_batch(replay_buffer, train_indices)
            train_ran = False
            train_skipped_reason = None
            if len(train_batch) == 0:
                train_skipped_reason = "empty_replay_buffer"
            elif np.allclose(train_batch.guidance_targets, train_batch.guidance_targets[0]):
                train_skipped_reason = "constant_targets"
            else:
                self.train_guidance_predictor_from_embeddings(
                    node_embeddings_list=train_batch.node_embeddings_list,
                    graph_conditioning=train_batch.graph_conditioning,
                    targets=train_batch.guidance_targets,
                    mode="regression",
                    learning_rate=guidance_learning_rate,
                    maximum_epochs=guidance_maximum_epochs,
                    batch_size=guidance_batch_size,
                    noise_scale=guidance_noise_scale,
                )
                train_ran = True
                guidance_ready = True

            history.append(
                {
                    "cycle": int(cycle_idx + 1),
                    "unguided_count": int(unguided_count),
                    "guided_count": int(guided_count),
                    "collected_examples": int(len(cycle_batch)),
                    "cycle_feasible_rate": float(np.mean(cycle_batch.feasible_mask)) if len(cycle_batch) else 0.0,
                    "cycle_mean_violation": float(np.mean(cycle_batch.violation_counts)) if len(cycle_batch) else 0.0,
                    "cycle_median_violation": float(np.median(cycle_batch.violation_counts)) if len(cycle_batch) else 0.0,
                    "cycle_mean_target": float(np.mean(cycle_batch.guidance_targets)) if len(cycle_batch) else 0.0,
                    "cycle_median_target": float(np.median(cycle_batch.guidance_targets)) if len(cycle_batch) else 0.0,
                    "cycle_bucket_summaries": cycle_bucket_summaries,
                    "replay_buffer_size": int(len(replay_buffer)),
                    "replay_bucket_summaries": replay_bucket_summaries,
                    "train_sample_size": int(len(train_batch)),
                    "train_ran": bool(train_ran),
                    "train_skipped_reason": train_skipped_reason,
                }
            )

        return {
            "history": history,
            "cycle_batches": cycle_batches,
            "replay_buffer": self._concat_generated_guidance_batches(replay_batches),
        }

    def decode(
        self,
        graph_conditioning: GraphConditioningBatch,
        desired_target: Optional[Union[int, float, Sequence[Any]]] = None,
        guidance_scale: float = 1.0,
        apply_feasibility_filtering: Optional[bool] = None,
        use_feasibility_oracle: Optional[bool] = None,
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
        feasibility_effort: Optional[int] = None,
        use_ilp_decoder: bool = True,
        edge_probability_threshold: Optional[float] = None,
    ) -> List[nx.Graph]:
        """Decode graph conditioning into concrete graphs, optionally using CFG and feasibility filtering."""
        self._require_fitted_for_generation()
        if self.verbose:
            verbose_log(self, f"Decoding {len(graph_conditioning)} conditioning vectors", level=1)
            if desired_target is not None:
                verbose_log(self, f"Using CFG target guidance: {desired_target} (scale={guidance_scale})", level=1)
        with self._feasibility_effort_context(
            feasibility_effort,
            apply_feasibility_filtering=apply_feasibility_filtering,
            use_feasibility_oracle=use_feasibility_oracle,
            feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
        ) as effort_profile:
            return self.decode_service_.decode(
                graph_conditioning,
                sampling_mode="unguided",
                desired_target=desired_target,
                guidance_scale=guidance_scale,
                apply_feasibility_filtering=(
                    effort_profile.apply_feasibility_filtering
                    if effort_profile is not None
                    else apply_feasibility_filtering
                ),
                feasibility_oracle_candidates_per_attempt=(
                    int(self.feasibility_oracle_candidates_per_attempt)
                    if effort_profile is not None
                    else self._resolve_feasibility_oracle_override(
                        use_feasibility_oracle=use_feasibility_oracle,
                        feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
                    )
                ),
                use_ilp_decoder=use_ilp_decoder,
                edge_probability_threshold=edge_probability_threshold,
            )

    def decode_classifier_guided(
        self,
        graph_conditioning: GraphConditioningBatch,
        desired_class: Union[int, Sequence[Any]],
        classifier_scale: float = 1.0,
        apply_feasibility_filtering: Optional[bool] = None,
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
        feasibility_effort: Optional[int] = None,
        use_ilp_decoder: bool = True,
        edge_probability_threshold: Optional[float] = None,
    ) -> List[nx.Graph]:
        self._require_fitted_for_generation()
        if self.verbose:
            verbose_log(self, f"Decoding {len(graph_conditioning)} conditioning vectors", level=1)
            verbose_log(
                self,
                f"Using classifier guidance toward class(es): {desired_class} (scale={classifier_scale})",
                level=1,
            )
        with self._feasibility_effort_context(
            feasibility_effort,
            apply_feasibility_filtering=apply_feasibility_filtering,
            feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
        ) as effort_profile:
            return self.decode_service_.decode(
                graph_conditioning,
                sampling_mode="classifier_guided",
                desired_class=desired_class,
                classifier_scale=classifier_scale,
                apply_feasibility_filtering=(
                    effort_profile.apply_feasibility_filtering
                    if effort_profile is not None
                    else apply_feasibility_filtering
                ),
                feasibility_oracle_candidates_per_attempt=(
                    int(self.feasibility_oracle_candidates_per_attempt)
                    if effort_profile is not None
                    else feasibility_oracle_candidates_per_attempt
                ),
                use_ilp_decoder=use_ilp_decoder,
                edge_probability_threshold=edge_probability_threshold,
            )

    def decode_regression_guided(
        self,
        graph_conditioning: GraphConditioningBatch,
        desired_target: Union[float, Sequence[Any]],
        predictor_scale: float = 1.0,
        apply_feasibility_filtering: Optional[bool] = None,
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
        feasibility_effort: Optional[int] = None,
        use_ilp_decoder: bool = True,
        edge_probability_threshold: Optional[float] = None,
    ) -> List[nx.Graph]:
        self._require_fitted_for_generation()
        if self.verbose:
            verbose_log(self, f"Decoding {len(graph_conditioning)} conditioning vectors", level=1)
            verbose_log(
                self,
                f"Using regression guidance toward target(s): {desired_target} (scale={predictor_scale})",
                level=1,
            )
        with self._feasibility_effort_context(
            feasibility_effort,
            apply_feasibility_filtering=apply_feasibility_filtering,
            feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
        ) as effort_profile:
            return self.decode_service_.decode(
                graph_conditioning,
                sampling_mode="regression_guided",
                desired_target=desired_target,
                predictor_scale=predictor_scale,
                apply_feasibility_filtering=(
                    effort_profile.apply_feasibility_filtering
                    if effort_profile is not None
                    else apply_feasibility_filtering
                ),
                feasibility_oracle_candidates_per_attempt=(
                    int(self.feasibility_oracle_candidates_per_attempt)
                    if effort_profile is not None
                    else feasibility_oracle_candidates_per_attempt
                ),
                use_ilp_decoder=use_ilp_decoder,
                edge_probability_threshold=edge_probability_threshold,
            )

    @timeit
    def sample(
        self,
        n_samples: int = 1,
        interpolate_between_n_samples: Optional[int] = None,
        desired_target: Optional[Union[int, float, Sequence[Any]]] = None,
        guidance_scale: float = 1.0,
        apply_feasibility_filtering: Optional[bool] = None,
        use_feasibility_oracle: Optional[bool] = None,
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
        feasibility_effort: Optional[int] = None,
        feasibility_filter: Optional[str] = None,
        use_ilp_decoder: bool = True,
        edge_probability_threshold: Optional[float] = None,
        return_decode_stages: bool = False,
    ) -> Union[List[nx.Graph], Dict[str, List[Optional[nx.Graph]]]]:
        """Sample random graph-conditioning vectors and decode them into graphs."""
        self._require_fitted_for_generation()
        self._log_sampling_request(n_samples, use_ilp_decoder=use_ilp_decoder)
        if interpolate_between_n_samples is not None:
            verbose_log(
                self,
                "Sampling conditioning via stochastic interpolation over "
                f"{interpolate_between_n_samples} cached training embeddings per output.",
                level=2,
            )
        if desired_target is not None:
            verbose_log(self, f"Using CFG target guidance: {desired_target} (scale={guidance_scale})", level=2)
        if feasibility_filter is not None and apply_feasibility_filtering is not None:
            raise ValueError(
                "feasibility_filter cannot be combined with apply_feasibility_filtering."
            )
        with self._feasibility_effort_context(
            feasibility_effort,
            apply_feasibility_filtering=apply_feasibility_filtering,
            use_feasibility_oracle=use_feasibility_oracle,
            feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
        ) as effort_profile, self._feasibility_filter_context(
            feasibility_filter,
            apply_feasibility_filtering=apply_feasibility_filtering,
        ) as filter_override:
            sampled_conditioning = self._sample_conditions(
                n_samples,
                interpolate_between_n_samples=interpolate_between_n_samples,
            )
            if return_decode_stages:
                return self._sample_decode_stages(
                    sampled_conditioning=sampled_conditioning,
                    n_samples=n_samples,
                    desired_target=desired_target,
                    guidance_scale=guidance_scale,
                    max_effort=(
                        int(effort_profile.effort)
                        if effort_profile is not None
                        else 5
                    ),
                )
            decoded_graphs = self.decode_service_.decode(
                sampled_conditioning,
                sampling_mode="unguided",
                desired_target=desired_target,
                guidance_scale=guidance_scale,
                apply_feasibility_filtering=(
                    filter_override
                    if filter_override is not None
                    else (
                        effort_profile.apply_feasibility_filtering
                        if effort_profile is not None
                        else apply_feasibility_filtering
                    )
                ),
                feasibility_oracle_candidates_per_attempt=(
                    int(self.feasibility_oracle_candidates_per_attempt)
                    if effort_profile is not None
                    else self._resolve_feasibility_oracle_override(
                        use_feasibility_oracle=use_feasibility_oracle,
                        feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
                    )
                ),
                use_ilp_decoder=use_ilp_decoder,
                edge_probability_threshold=edge_probability_threshold,
            )
            if (
                effort_profile is not None
                and int(effort_profile.effort) >= 2
                and len(decoded_graphs) == 0
            ):
                verbose_log(
                    self,
                    "Feasibility effort returned no graphs; retrying same sampled "
                    "conditioning at effort 1 with fallback filtering.",
                    level=1,
                )
                with self._feasibility_effort_context(1) as fallback_effort_profile, self._feasibility_filter_context(
                    "fallback"
                ) as fallback_filter_override:
                    decoded_graphs = self.decode_service_.decode(
                        sampled_conditioning,
                        sampling_mode="unguided",
                        desired_target=desired_target,
                        guidance_scale=guidance_scale,
                        apply_feasibility_filtering=(
                            fallback_filter_override
                            if fallback_filter_override is not None
                            else fallback_effort_profile.apply_feasibility_filtering
                        ),
                        feasibility_oracle_candidates_per_attempt=int(
                            self.feasibility_oracle_candidates_per_attempt
                        ),
                        use_ilp_decoder=use_ilp_decoder,
                        edge_probability_threshold=edge_probability_threshold,
                    )
            return decoded_graphs

    def _sample_decode_stages(
        self,
        *,
        sampled_conditioning: GraphConditioningBatch,
        n_samples: int,
        desired_target: Optional[Union[int, float, Sequence[Any]]],
        guidance_scale: float,
        max_effort: int,
    ) -> Dict[str, List[Optional[nx.Graph]]]:
        initial_generated_nodes = self._predict_generated_nodes(
            sampled_conditioning,
            sampling_mode="unguided",
            desired_target=desired_target,
            guidance_scale=guidance_scale,
        )
        variants: Dict[str, List[Optional[nx.Graph]]] = {
            f"effort_{effort}": [None] * int(n_samples)
            for effort in range(int(max_effort) + 1)
        }
        for slot_idx in range(int(n_samples)):
            slot_conditioning = self._slice_graph_conditioning(sampled_conditioning, [slot_idx])
            slot_generated_nodes = _build_single_generated_node_batch(initial_generated_nodes, slot_idx)
            for effort in range(int(max_effort) + 1):
                profile = resolve_feasibility_effort(effort)
                with self._feasibility_effort_context(effort):
                    try:
                        decoded_graphs = self._decode_generated_nodes(
                            slot_generated_nodes,
                            graph_conditioning=slot_conditioning,
                            feasibility_oracle_candidates_per_attempt=(
                                profile.feasibility_oracle_candidates_per_attempt
                                if profile.use_feasibility_oracle
                                else 0
                            ),
                            use_ilp_decoder=True,
                        )
                    except (RuntimeError, TimeoutError) as exc:
                        verbose_log(
                            self,
                            "Effort-stage decode failed; leaving stage empty "
                            f"(slot={slot_idx}, effort={effort}, error={exc}).",
                            level=1,
                        )
                        decoded_graphs = []
                variants[f"effort_{effort}"][slot_idx] = decoded_graphs[0] if decoded_graphs else None
        return variants

    def score_feasible_rate(
        self,
        n_samples: int = 32,
        max_feasibility_attempts: Optional[int] = None,
        feasibility_candidates_per_attempt: Optional[int] = None,
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
        feasibility_effort: Optional[int] = None,
        interpolate_between_n_samples: Optional[int] = None,
        desired_target: Optional[Union[int, float, Sequence[Any]]] = None,
        guidance_scale: float = 1.0,
        verbose: bool = False,
        ) -> Dict[str, Any]:
        """Score generation quality using the fraction of feasible decoded candidates."""
        with self._feasibility_effort_context(
            feasibility_effort,
            max_feasibility_attempts=max_feasibility_attempts,
            feasibility_candidates_per_attempt=feasibility_candidates_per_attempt,
            feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
        ) as effort_profile:
            return _score_feasible_rate(
                self,
                n_samples=n_samples,
                max_feasibility_attempts=(
                    int(self.max_feasibility_attempts)
                    if effort_profile is not None
                    else max_feasibility_attempts
                ),
                feasibility_candidates_per_attempt=(
                    int(self.feasibility_candidates_per_attempt)
                    if effort_profile is not None
                    else feasibility_candidates_per_attempt
                ),
                feasibility_oracle_candidates_per_attempt=(
                    int(self.feasibility_oracle_candidates_per_attempt)
                    if effort_profile is not None
                    else feasibility_oracle_candidates_per_attempt
                ),
                interpolate_between_n_samples=interpolate_between_n_samples,
                desired_target=desired_target,
                guidance_scale=guidance_scale,
                verbose=verbose,
            )

    def _collect_metric_histories(self) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
        node_model = getattr(self, "conditional_node_generator_model", None)
        collector = getattr(node_model, "_collect_metric_histories", None)
        if callable(collector):
            return collector()
        return {}, {}

    def plot_metrics(self, window: int = 10, alpha: float = 0.3):
        """Visualise the wrapped node model training metrics when available."""
        node_model = getattr(self, "conditional_node_generator_model", None)
        plotter = getattr(node_model, "plot_metrics", None)
        if not callable(plotter):
            logger.info("Node generator does not expose plot_metrics().")
            return None
        return plotter(window=window, alpha=alpha)

    def export_metrics_pdf(self, output_path: str, window: int = 10, alpha: float = 0.3):
        """Write wrapped node model training metrics to a PDF when available."""
        node_model = getattr(self, "conditional_node_generator_model", None)
        exporter = getattr(node_model, "export_metrics_pdf", None)
        if not callable(exporter):
            logger.info("Node generator does not expose export_metrics_pdf().")
            return None
        return exporter(output_path=output_path, window=window, alpha=alpha)

    def _log_sampling_request(self, n_samples: int, *, use_ilp_decoder: bool) -> None:
        decoder_label = "ILP" if use_ilp_decoder else "raw"
        verbose_log(self, f"Sampling {n_samples} {decoder_label} graphs", level=2)

    @timeit
    def conditional_sample(
        self,
        graphs: List[nx.Graph],
        n_samples: int = 1,
        desired_target: Optional[Union[int, float, Sequence[Any]]] = None,
        guidance_scale: float = 1.0,
        apply_feasibility_filtering: Optional[bool] = None,
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
        feasibility_effort: Optional[int] = None,
        use_ilp_decoder: bool = True,
        edge_probability_threshold: Optional[float] = None,
    ) -> List[List[nx.Graph]]:
        """Encode each input graph and sample one or more decoded variations per conditioning vector."""
        self._require_fitted_for_generation()
        _, graph_conditioning = self.encode(graphs)
        repeated_conditioning = self._repeat_graph_conditioning(
            graph_conditioning,
            repeats=n_samples,
        )
        with self._feasibility_effort_context(
            feasibility_effort,
            apply_feasibility_filtering=apply_feasibility_filtering,
            feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
        ) as effort_profile:
            decoded_slots = self.decode_service_.decode_with_feasibility_slots(
                repeated_conditioning,
                sampling_mode="unguided",
                desired_target=desired_target,
                guidance_scale=guidance_scale,
                apply_feasibility_filtering=(
                    effort_profile.apply_feasibility_filtering
                    if effort_profile is not None
                    else apply_feasibility_filtering
                ),
                feasibility_oracle_candidates_per_attempt=(
                    int(self.feasibility_oracle_candidates_per_attempt)
                    if effort_profile is not None
                    else feasibility_oracle_candidates_per_attempt
                ),
                use_ilp_decoder=use_ilp_decoder,
                edge_probability_threshold=edge_probability_threshold,
            )
        return [
            [
                graph
                for graph in decoded_slots[i * n_samples:(i + 1) * n_samples]
                if graph is not None
            ]
            for i in range(len(graphs))
        ]

    @timeit
    def sample_classifier_guided(
        self,
        desired_class: Union[int, Sequence[Any]],
        n_samples: int = 1,
        interpolate_between_n_samples: Optional[int] = None,
        classifier_scale: float = 1.0,
        apply_feasibility_filtering: Optional[bool] = None,
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
        feasibility_effort: Optional[int] = None,
        use_ilp_decoder: bool = True,
        edge_probability_threshold: Optional[float] = None,
    ) -> List[nx.Graph]:
        self._require_fitted_for_generation()
        self._log_sampling_request(n_samples, use_ilp_decoder=use_ilp_decoder)
        if interpolate_between_n_samples is not None:
            verbose_log(
                self,
                "Sampling conditioning via stochastic interpolation over "
                f"{interpolate_between_n_samples} cached training embeddings per output.",
                level=2,
            )
        verbose_log(
            self,
            f"Using classifier guidance toward class(es): {desired_class} (scale={classifier_scale})",
            level=2,
        )
        sampled_conditioning = self._sample_conditions(
            n_samples,
            interpolate_between_n_samples=interpolate_between_n_samples,
        )
        return self.decode_classifier_guided(
            sampled_conditioning,
            desired_class=desired_class,
            classifier_scale=classifier_scale,
            apply_feasibility_filtering=apply_feasibility_filtering,
            feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
            feasibility_effort=feasibility_effort,
            use_ilp_decoder=use_ilp_decoder,
            edge_probability_threshold=edge_probability_threshold,
        )

    @timeit
    def conditional_sample_classifier_guided(
        self,
        graphs: List[nx.Graph],
        desired_class: Union[int, Sequence[Any]],
        n_samples: int = 1,
        classifier_scale: float = 1.0,
        apply_feasibility_filtering: Optional[bool] = None,
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
        feasibility_effort: Optional[int] = None,
        use_ilp_decoder: bool = True,
        edge_probability_threshold: Optional[float] = None,
    ) -> List[List[nx.Graph]]:
        self._require_fitted_for_generation()
        _, graph_conditioning = self.encode(graphs)
        repeated_conditioning = self._repeat_graph_conditioning(
            graph_conditioning,
            repeats=n_samples,
        )
        with self._feasibility_effort_context(
            feasibility_effort,
            apply_feasibility_filtering=apply_feasibility_filtering,
            feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
        ) as effort_profile:
            decoded_slots = self.decode_service_.decode_with_feasibility_slots(
                repeated_conditioning,
                sampling_mode="classifier_guided",
                desired_class=desired_class,
                classifier_scale=classifier_scale,
                apply_feasibility_filtering=(
                    effort_profile.apply_feasibility_filtering
                    if effort_profile is not None
                    else apply_feasibility_filtering
                ),
                feasibility_oracle_candidates_per_attempt=(
                    int(self.feasibility_oracle_candidates_per_attempt)
                    if effort_profile is not None
                    else feasibility_oracle_candidates_per_attempt
                ),
                use_ilp_decoder=use_ilp_decoder,
                edge_probability_threshold=edge_probability_threshold,
            )
        return [
            [
                graph
                for graph in decoded_slots[i * n_samples:(i + 1) * n_samples]
                if graph is not None
            ]
            for i in range(len(graphs))
        ]

    @timeit
    def sample_regression_guided(
        self,
        desired_target: Union[float, Sequence[Any]],
        n_samples: int = 1,
        interpolate_between_n_samples: Optional[int] = None,
        predictor_scale: float = 1.0,
        apply_feasibility_filtering: Optional[bool] = None,
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
        feasibility_effort: Optional[int] = None,
        use_ilp_decoder: bool = True,
        edge_probability_threshold: Optional[float] = None,
    ) -> List[nx.Graph]:
        self._require_fitted_for_generation()
        self._log_sampling_request(n_samples, use_ilp_decoder=use_ilp_decoder)
        if interpolate_between_n_samples is not None:
            verbose_log(
                self,
                "Sampling conditioning via stochastic interpolation over "
                f"{interpolate_between_n_samples} cached training embeddings per output.",
                level=2,
            )
        verbose_log(
            self,
            f"Using regression guidance toward target(s): {desired_target} (scale={predictor_scale})",
            level=2,
        )
        sampled_conditioning = self._sample_conditions(
            n_samples,
            interpolate_between_n_samples=interpolate_between_n_samples,
        )
        return self.decode_regression_guided(
            sampled_conditioning,
            desired_target=desired_target,
            predictor_scale=predictor_scale,
            apply_feasibility_filtering=apply_feasibility_filtering,
            feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
            feasibility_effort=feasibility_effort,
            use_ilp_decoder=use_ilp_decoder,
            edge_probability_threshold=edge_probability_threshold,
        )

    @timeit
    def conditional_sample_regression_guided(
        self,
        graphs: List[nx.Graph],
        desired_target: Union[float, Sequence[Any]],
        n_samples: int = 1,
        predictor_scale: float = 1.0,
        apply_feasibility_filtering: Optional[bool] = None,
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
        feasibility_effort: Optional[int] = None,
        use_ilp_decoder: bool = True,
        edge_probability_threshold: Optional[float] = None,
    ) -> List[List[nx.Graph]]:
        self._require_fitted_for_generation()
        _, graph_conditioning = self.encode(graphs)
        repeated_conditioning = self._repeat_graph_conditioning(
            graph_conditioning,
            repeats=n_samples,
        )
        with self._feasibility_effort_context(
            feasibility_effort,
            apply_feasibility_filtering=apply_feasibility_filtering,
            feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
        ) as effort_profile:
            decoded_slots = self.decode_service_.decode_with_feasibility_slots(
                repeated_conditioning,
                sampling_mode="regression_guided",
                desired_target=desired_target,
                predictor_scale=predictor_scale,
                apply_feasibility_filtering=(
                    effort_profile.apply_feasibility_filtering
                    if effort_profile is not None
                    else apply_feasibility_filtering
                ),
                feasibility_oracle_candidates_per_attempt=(
                    int(self.feasibility_oracle_candidates_per_attempt)
                    if effort_profile is not None
                    else feasibility_oracle_candidates_per_attempt
                ),
                use_ilp_decoder=use_ilp_decoder,
                edge_probability_threshold=edge_probability_threshold,
            )
        return [
            [
                graph
                for graph in decoded_slots[i * n_samples:(i + 1) * n_samples]
                if graph is not None
            ]
            for i in range(len(graphs))
        ]

    def sample_conditioned_on_random(
        self,
        graphs,
        n_samples=1,
        desired_target: Optional[Union[int, float, Sequence[Any]]] = None,
        guidance_scale: float = 1.0,
        apply_feasibility_filtering: Optional[bool] = None,
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
        use_ilp_decoder: bool = True,
        edge_probability_threshold: Optional[float] = None,
    ):
        self._require_fitted_for_generation()
        sampled_seed_graphs = random.choices(graphs, k=n_samples)
        reconstructed_graphs_list = self.conditional_sample(
            sampled_seed_graphs,
            n_samples=1,
            desired_target=desired_target,
            guidance_scale=guidance_scale,
            apply_feasibility_filtering=apply_feasibility_filtering,
            feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
            use_ilp_decoder=use_ilp_decoder,
            edge_probability_threshold=edge_probability_threshold,
        )
        sampled_graphs = [reconstructed_graphs[0] for reconstructed_graphs in reconstructed_graphs_list if reconstructed_graphs]
        return sampled_graphs

    def sample_conditioned_on_random_classifier_guided(
        self,
        graphs,
        desired_class: Union[int, Sequence[Any]],
        n_samples=1,
        classifier_scale: float = 1.0,
        apply_feasibility_filtering: Optional[bool] = None,
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
        feasibility_effort: Optional[int] = None,
        use_ilp_decoder: bool = True,
        edge_probability_threshold: Optional[float] = None,
    ):
        self._require_fitted_for_generation()
        sampled_seed_graphs = random.choices(graphs, k=n_samples)
        reconstructed_graphs_list = self.conditional_sample_classifier_guided(
            sampled_seed_graphs,
            desired_class=desired_class,
            n_samples=1,
            classifier_scale=classifier_scale,
            apply_feasibility_filtering=apply_feasibility_filtering,
            feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
            feasibility_effort=feasibility_effort,
            use_ilp_decoder=use_ilp_decoder,
            edge_probability_threshold=edge_probability_threshold,
        )
        sampled_graphs = [reconstructed_graphs[0] for reconstructed_graphs in reconstructed_graphs_list if reconstructed_graphs]
        return sampled_graphs

    def sample_conditioned_on_random_regression_guided(
        self,
        graphs,
        desired_target: Union[float, Sequence[Any]],
        n_samples=1,
        predictor_scale: float = 1.0,
        apply_feasibility_filtering: Optional[bool] = None,
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
        feasibility_effort: Optional[int] = None,
        use_ilp_decoder: bool = True,
        edge_probability_threshold: Optional[float] = None,
    ):
        self._require_fitted_for_generation()
        sampled_seed_graphs = random.choices(graphs, k=n_samples)
        reconstructed_graphs_list = self.conditional_sample_regression_guided(
            sampled_seed_graphs,
            desired_target=desired_target,
            n_samples=1,
            predictor_scale=predictor_scale,
            apply_feasibility_filtering=apply_feasibility_filtering,
            feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
            feasibility_effort=feasibility_effort,
            use_ilp_decoder=use_ilp_decoder,
            edge_probability_threshold=edge_probability_threshold,
        )
        sampled_graphs = [reconstructed_graphs[0] for reconstructed_graphs in reconstructed_graphs_list if reconstructed_graphs]
        return sampled_graphs

    def interpolate(
        self,
        G1: nx.Graph,
        G2: nx.Graph,
        k: int = 7,
        apply_feasibility_filtering: Optional[bool] = None,
        interpolation_mode: str = "slerp",
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
        feasibility_effort: Optional[int] = None,
        use_ilp_decoder: bool = True,
        edge_probability_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Interpolate between two graph condition vectors and decode intermediate graphs.

        Args:
            G1 (nx.Graph): Input value.
            G2 (nx.Graph): Input value.
            k (int): Optional input value.
            apply_feasibility_filtering (Optional[bool]): Optional input value.
            interpolation_mode (str): Optional input value.
            feasibility_oracle_candidates_per_attempt (Optional[int]): Optional input value.

        Returns:
            Dict[str, Any]: Computed result.
        """
        self._require_fitted_for_generation()
        cond_a = self.graph_encode([G1])
        cond_b = self.graph_encode([G2])
        ts = np.linspace(0.0, 1.0, k + 2)[1:-1]

        interpolation_mode = str(interpolation_mode).lower()
        if interpolation_mode not in {"lerp", "slerp"}:
            raise ValueError(
                f"interpolation_mode must be 'lerp' or 'slerp' (got {interpolation_mode!r})."
            )
        if interpolation_mode == "slerp":
            interpolated_graph_embeddings = np.stack(
                [scaled_slerp(cond_a.graph_embeddings[0], cond_b.graph_embeddings[0], t) for t in ts],
                axis=0,
            )
        else:
            interpolated_graph_embeddings = np.stack(
                [(1.0 - t) * cond_a.graph_embeddings[0] + t * cond_b.graph_embeddings[0] for t in ts],
                axis=0,
            )
        interpolated_node_counts = _interpolate_integer_series(
            cond_a.node_counts[0],
            cond_b.node_counts[0],
            ts,
            minimum=1,
        )
        interpolated_edge_counts = _interpolate_integer_series(
            cond_a.edge_counts[0],
            cond_b.edge_counts[0],
            ts,
            minimum=0,
        )

        interpolated_conditioning = GraphConditioningBatch(
            graph_embeddings=interpolated_graph_embeddings,
            node_counts=interpolated_node_counts,
            edge_counts=interpolated_edge_counts,
        )
        with self._feasibility_effort_context(
            feasibility_effort,
            apply_feasibility_filtering=apply_feasibility_filtering,
            feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
        ) as effort_profile:
            decoded_slots = self.decode_service_.decode_with_feasibility_slots(
                interpolated_conditioning,
                sampling_mode="unguided",
                apply_feasibility_filtering=(
                    effort_profile.apply_feasibility_filtering
                    if effort_profile is not None
                    else apply_feasibility_filtering
                ),
                feasibility_oracle_candidates_per_attempt=(
                    int(self.feasibility_oracle_candidates_per_attempt)
                    if effort_profile is not None
                    else feasibility_oracle_candidates_per_attempt
                ),
                use_ilp_decoder=use_ilp_decoder,
                edge_probability_threshold=edge_probability_threshold,
            )
        step_summary = pd.DataFrame(
            {
                "step": np.arange(1, len(ts) + 1),
                "t": np.round(ts, 3),
                "target_nodes": interpolated_node_counts,
                "target_edges": interpolated_edge_counts,
                "decoded": [graph is not None for graph in decoded_slots],
                "mode": interpolation_mode,
            }
        )
        return {
            "ts": ts,
            "conditioning": interpolated_conditioning,
            "decoded_slots": decoded_slots,
            "generated_graphs": [graph for graph in decoded_slots if graph is not None],
            "summary": step_summary,
        }

    def mean(
        self,
        graphs: List[nx.Graph],
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
        feasibility_effort: Optional[int] = None,
    ) -> nx.Graph:
        """Compute a geometric mean graph via the SLERP barycentre of encodings.

        Args:
            graphs (List[nx.Graph]): Input value.

        Returns:
            nx.Graph: Computed result.
        """
        self._require_fitted_for_generation()
        graph_conditioning = self.graph_encode(graphs)
        Y = np.vstack(graph_conditioning.graph_embeddings)
        centroid = scaled_slerp_average(Y)
        mean_node_count = int(round(np.mean(graph_conditioning.node_counts)))
        mean_edge_count = int(round(np.mean(graph_conditioning.edge_counts)))
        return self.decode(
            GraphConditioningBatch(
                graph_embeddings=np.asarray([centroid]),
                node_counts=np.asarray([mean_node_count], dtype=np.int64),
                edge_counts=np.asarray([mean_edge_count], dtype=np.int64),
            ),
            feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
            feasibility_effort=feasibility_effort,
        )[0]
