"""Decoder helpers for rebuilding labeled graphs from node-field predictions."""

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import dill as pickle
import networkx as nx
import numpy as np

from .conditional_node_field_generator import GeneratedNodeBatch, GraphConditioningBatch
from . import diagnostics as _shared_diagnostics
from .decoder_assembly import (
    assemble_edge_labels_from_matrix as _assemble_edge_labels_from_matrix_impl,
    assemble_graph as _assemble_graph_impl,
    assemble_graph_star as _assemble_graph_star_impl,
    edge_label_list_to_matrix as _edge_label_list_to_matrix_impl,
    edge_label_matrix_to_list as _edge_label_matrix_to_list_impl,
    validate_edge_label_array as _validate_edge_label_array_impl,
    validate_node_label_array as _validate_node_label_array_impl,
)
from .direct_graph_decoder import (
    adjacency_from_edges as _adjacency_from_selected_edges_impl,
    edge_candidates as _direct_edge_candidates_impl,
    select_by_threshold as _select_direct_edges_by_threshold_impl,
    select_degree_aware as _select_direct_edges_degree_aware_impl,
    select_top_k as _select_direct_edges_top_k_impl,
)
from .decode_preparation import (
    build_masked_prob_matrix as _build_masked_prob_matrix_impl,
    build_single_generated_node_batch as _build_single_generated_node_batch_impl,
    resolve_predicted_edge_labels as _resolve_predicted_edge_labels_impl,
    resolve_predicted_node_labels as _resolve_predicted_node_labels_impl,
)
from .graph_decode_utils import _canonicalize_edge, _normalize_violating_edge_sets
from .parallel_utils import _normalize_n_jobs, _parallel_map
from .runtime_utils import verbose_log
from .structural_decoder import AdjacencySolveReport, solve_adjacency

Edge = Tuple[int, int]
_DECODER_ARTIFACT_VERSION = 3
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
    generator_module = sys.modules.get(
        "conditional_node_field_graph_generator.conditional_node_field_graph_generator"
    )
    patched_plotter = None if generator_module is None else getattr(generator_module, "_plot_decoder_diagnostics", None)
    if callable(patched_plotter) and patched_plotter is not _plot_decoder_diagnostics:
        return patched_plotter(**kwargs)
    return _shared_diagnostics._plot_decoder_diagnostics(
        **kwargs,
        plot_backend=plt,
        inline_renderer=_try_render_molecular_graph_inline,
    )




def _build_masked_prob_matrix(
    existence_mask: np.ndarray,
    degree_prediction: np.ndarray,
    prob_matrix: np.ndarray,
) -> np.ndarray:
    return _build_masked_prob_matrix_impl(
        existence_mask,
        degree_prediction,
        prob_matrix,
    )


def _validate_probability_values(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} values must be within [0, 1].")
    return array


def _solver_budget_with_parent_reserve(
    solver_timeout_seconds: Optional[float],
    parent_timeout_seconds: Optional[float],
) -> Optional[float]:
    if parent_timeout_seconds is None:
        return solver_timeout_seconds
    parent_timeout_seconds = float(parent_timeout_seconds)
    reserve = min(0.5, max(0.01, parent_timeout_seconds * 0.05))
    parent_solver_budget = max(0.01, parent_timeout_seconds - reserve)
    if solver_timeout_seconds is None:
        return parent_solver_budget
    return min(float(solver_timeout_seconds), parent_solver_budget)


def _edge_label_matrix_to_list(adj_mtx: np.ndarray, edge_label_matrix: np.ndarray) -> np.ndarray:
    return _edge_label_matrix_to_list_impl(adj_mtx, edge_label_matrix)


def _edge_label_list_to_matrix(
    adj_mtx: np.ndarray,
    edge_labels: Sequence[Any],
) -> np.ndarray:
    return _edge_label_list_to_matrix_impl(adj_mtx, edge_labels)


def _assemble_graph_job(
    node_presence_mask: np.ndarray,
    node_labels: np.ndarray,
    edge_labels: np.ndarray,
    adj_mtx: np.ndarray,
) -> nx.Graph:
    return _assemble_graph_impl(node_presence_mask, node_labels, edge_labels, adj_mtx)


def _assemble_graph_job_star(args) -> nx.Graph:
    return _assemble_graph_star_impl(args)


def _validate_node_label_array(
    node_labels: np.ndarray,
    *,
    graph_idx: int,
    n_slots: int,
) -> np.ndarray:
    return _validate_node_label_array_impl(
        node_labels,
        graph_idx=graph_idx,
        n_slots=n_slots,
    )


def _validate_edge_label_array(
    edge_labels: np.ndarray,
    *,
    graph_idx: int,
    expected_edge_count: int,
) -> np.ndarray:
    return _validate_edge_label_array_impl(
        edge_labels,
        graph_idx=graph_idx,
        expected_edge_count=expected_edge_count,
    )


def _decode_single_adjacency_job(
    prob_list: np.ndarray,
    existence_mask: np.ndarray,
    existence_scores: Optional[np.ndarray],
    degree_prediction: np.ndarray,
    desired_node_count: Optional[int],
    desired_edge_count: Optional[int],
    degree_slack_penalty: float,
    edge_count_slack_penalty: Optional[float],
    enforce_connectivity: bool,
    warm_start_mst: bool,
    verbose: int,
    diagnostic_graph_renderer: Optional[Callable[..., Any]] = None,
    adjacency_time_limit_seconds: Optional[float] = 60.0,
    horizon_probability_matrix: Optional[np.ndarray] = None,
    horizon: Optional[int] = None,
    use_horizon_ilp_constraints: bool = True,
    horizon_constraint_weight: float = 2.0,
    horizon_positive_threshold: float = 0.8,
    horizon_negative_threshold: float = 0.2,
    horizon_pair_budget: int = 24,
    horizon_paths_per_pair: int = 8,
    horizon_path_expansion_budget: int = 4096,
    horizon_max_iterations: int = 1,
    solver_threads: Optional[int] = None,
    deadline_monotonic: Optional[float] = None,
    per_job_timeout_seconds: Optional[float] = None,
) -> Tuple[np.ndarray, AdjacencySolveReport]:
    if deadline_monotonic is None and per_job_timeout_seconds is not None:
        deadline_monotonic = time.monotonic() + float(per_job_timeout_seconds)
    decoder = ConditionalNodeFieldGraphDecoder(
        verbose=bool(verbose),
        degree_slack_penalty=degree_slack_penalty,
        edge_count_slack_penalty=edge_count_slack_penalty,
        enforce_connectivity=enforce_connectivity,
        warm_start_mst=warm_start_mst,
        diagnostic_graph_renderer=diagnostic_graph_renderer,
        adjacency_time_limit_seconds=adjacency_time_limit_seconds,
        use_horizon_ilp_constraints=use_horizon_ilp_constraints,
        horizon_constraint_weight=horizon_constraint_weight,
        horizon_positive_threshold=horizon_positive_threshold,
        horizon_negative_threshold=horizon_negative_threshold,
        horizon_pair_budget=horizon_pair_budget,
        horizon_paths_per_pair=horizon_paths_per_pair,
        horizon_path_expansion_budget=horizon_path_expansion_budget,
        horizon_max_iterations=horizon_max_iterations,
        solver_threads=solver_threads,
    )
    n_nodes = min(len(existence_mask), len(degree_prediction))
    prob_matrix = np.zeros((n_nodes, n_nodes))
    idx = 0
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                prob_matrix[i, j] = prob_list[idx]
                idx += 1
    existent = decoder.resolve_node_presence_mask(
        np.asarray(existence_mask[:n_nodes], dtype=bool),
        desired_node_count=desired_node_count,
        node_existence_scores=None if existence_scores is None else np.asarray(existence_scores[:n_nodes], dtype=float),
    )
    for i in range(n_nodes):
        for j in range(n_nodes):
            if not (existent[i] and existent[j]):
                prob_matrix[i, j] = 0
    prob_matrix = (prob_matrix + prob_matrix.T) / 2
    target_degrees = decoder.get_degree_targets(
        np.asarray(degree_prediction[:n_nodes], dtype=float),
        existent,
        desired_edge_count=desired_edge_count,
    )
    adj_mtx = decoder.optimize_adjacency_matrix(
        prob_matrix,
        target_degrees,
        target_edge_count=desired_edge_count,
        timeLimit=adjacency_time_limit_seconds,
        active_node_mask=existent,
        horizon_probability_matrix=horizon_probability_matrix,
        horizon=horizon,
        _deadline_monotonic=deadline_monotonic,
    )
    if int(verbose) >= 4 and diagnostic_graph_renderer is None:
        _plot_decoder_diagnostics(
            prob_matrix=prob_matrix,
            adj_mtx=adj_mtx,
            target_degrees=target_degrees,
            title="Decoder solve",
            graph_renderer=decoder.diagnostic_graph_renderer,
            existence_mask=existent,
        )
    return adj_mtx, decoder.last_adjacency_solve_report_


def _decode_single_adjacency_job_star(args) -> Tuple[np.ndarray, AdjacencySolveReport]:
    return _decode_single_adjacency_job(*args)


def build_single_generated_node_batch(
    generated_nodes: GeneratedNodeBatch,
    graph_idx: int,
) -> GeneratedNodeBatch:
    return _build_single_generated_node_batch_impl(generated_nodes, graph_idx)


def resolve_predicted_node_labels(
    owner,
    generated_nodes: GeneratedNodeBatch,
) -> List[np.ndarray]:
    return _resolve_predicted_node_labels_impl(owner, generated_nodes)


def resolve_predicted_edge_labels(
    owner,
    generated_nodes: GeneratedNodeBatch,
    predicted_edge_probability_matrices: Optional[List[np.ndarray]],
) -> Tuple[Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
    return _resolve_predicted_edge_labels_impl(
        owner,
        generated_nodes,
        predicted_edge_probability_matrices,
    )


def sample_oracle_cuts_for_iteration(
    owner,
    accumulated_cuts: Sequence[FrozenSet[Edge]],
    solve_iteration_idx: int,
) -> List[FrozenSet[Edge]]:
    from .oracle_decode import sample_oracle_cuts_for_iteration as sample_impl

    return sample_impl(owner, accumulated_cuts, solve_iteration_idx)


def _optimize_adjacency_matrix_worker(graph_decoder, args, kwargs) -> np.ndarray:
    from .oracle_decode import _optimize_adjacency_matrix_worker as worker_impl

    return worker_impl(graph_decoder, args, kwargs)


def _oracle_adjacency_timeout_seconds(owner) -> Optional[float]:
    from .oracle_decode import oracle_adjacency_timeout_seconds

    return oracle_adjacency_timeout_seconds(owner)


def optimize_oracle_adjacency_matrix(owner, *args, **kwargs) -> np.ndarray:
    from .oracle_decode import optimize_oracle_adjacency_matrix as optimize_impl

    return optimize_impl(owner, *args, **kwargs)


def solve_oracle_relaxed_adjacency(
    owner,
    *,
    masked_prob_matrix: np.ndarray,
    target_degrees: List[int],
    accumulated_cuts: Sequence[FrozenSet[Edge]],
    start_iteration_idx: int,
    target_edge_count: Optional[int] = None,
    edge_violation_prior: Optional[np.ndarray] = None,
    active_node_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    from .oracle_decode import solve_oracle_relaxed_adjacency as solve_impl

    return solve_impl(
        owner,
        masked_prob_matrix=masked_prob_matrix,
        target_degrees=target_degrees,
        accumulated_cuts=accumulated_cuts,
        start_iteration_idx=start_iteration_idx,
        target_edge_count=target_edge_count,
        edge_violation_prior=edge_violation_prior,
        active_node_mask=active_node_mask,
    )


class ConditionalNodeFieldGraphDecoder(object):
    """Decode node-field predictions into final ``networkx.Graph`` objects.

    This is the canonical decoder implementation exported by
    ``conditional_node_field_graph_generator``. It owns the structural MILP
    projection, optional connectivity enforcement, locality-supervision helpers,
    and the final node/edge label attachment used by
    ``ConditionalNodeFieldGraphGenerator``.
    """

    def __init__(
        self,
        verbose: bool = True,
        direct_edge_probability_threshold: float = 0.5,
        enforce_connectivity: bool = True,
        degree_slack_penalty: float = 1e6,
        edge_count_slack_penalty: Optional[float] = 2.0,
        warm_start_mst: bool = True,
        n_jobs: int = 1,
        diagnostic_graph_renderer: Optional[Callable[..., Any]] = None,
        adjacency_time_limit_seconds: Optional[float] = 60.0,
        parallel_decode_timeout_seconds: Optional[float] = 30.0,
        use_horizon_ilp_constraints: bool = True,
        horizon_constraint_weight: float = 2.0,
        horizon_positive_threshold: float = 0.8,
        horizon_negative_threshold: float = 0.2,
        horizon_pair_budget: int = 24,
        horizon_paths_per_pair: int = 8,
        horizon_path_expansion_budget: int = 4096,
        horizon_max_iterations: int = 1,
        solver_threads: Optional[int] = None,
    ) -> None:
        self.verbose = verbose
        self.direct_edge_probability_threshold = float(direct_edge_probability_threshold)
        if (
            not np.isfinite(self.direct_edge_probability_threshold)
            or not 0.0 <= self.direct_edge_probability_threshold <= 1.0
        ):
            raise ValueError("direct_edge_probability_threshold must be within [0, 1].")
        self.enforce_connectivity = enforce_connectivity
        self.degree_slack_penalty = float(degree_slack_penalty)
        if not np.isfinite(self.degree_slack_penalty) or self.degree_slack_penalty <= 0.0:
            raise ValueError("degree_slack_penalty must be finite and > 0.")
        self.edge_count_slack_penalty = (
            None if edge_count_slack_penalty is None else float(edge_count_slack_penalty)
        )
        if self.edge_count_slack_penalty is not None and (
            not np.isfinite(self.edge_count_slack_penalty)
            or self.edge_count_slack_penalty <= 0.0
        ):
            raise ValueError("edge_count_slack_penalty must be finite and > 0 when provided.")
        self.warm_start_mst = warm_start_mst
        self.n_jobs = _normalize_n_jobs(n_jobs)
        self.diagnostic_graph_renderer = diagnostic_graph_renderer
        self.adjacency_time_limit_seconds = (
            None if adjacency_time_limit_seconds is None else float(adjacency_time_limit_seconds)
        )
        if self.adjacency_time_limit_seconds is not None and (
            not np.isfinite(self.adjacency_time_limit_seconds)
            or self.adjacency_time_limit_seconds <= 0.0
        ):
            raise ValueError("adjacency_time_limit_seconds must be finite and > 0 when provided.")
        self.parallel_decode_timeout_seconds = (
            None if parallel_decode_timeout_seconds is None else float(parallel_decode_timeout_seconds)
        )
        if self.parallel_decode_timeout_seconds is not None and (
            not np.isfinite(self.parallel_decode_timeout_seconds)
            or self.parallel_decode_timeout_seconds <= 0.0
        ):
            raise ValueError("parallel_decode_timeout_seconds must be finite and > 0 when provided.")
        self.use_horizon_ilp_constraints = bool(use_horizon_ilp_constraints)
        self.horizon_constraint_weight = float(horizon_constraint_weight)
        if not np.isfinite(self.horizon_constraint_weight) or self.horizon_constraint_weight < 0.0:
            raise ValueError("horizon_constraint_weight must be finite and >= 0.")
        self.horizon_positive_threshold = float(horizon_positive_threshold)
        self.horizon_negative_threshold = float(horizon_negative_threshold)
        if (
            not np.isfinite(self.horizon_positive_threshold)
            or not 0.0 <= self.horizon_positive_threshold <= 1.0
        ):
            raise ValueError("horizon_positive_threshold must be within [0, 1].")
        if (
            not np.isfinite(self.horizon_negative_threshold)
            or not 0.0 <= self.horizon_negative_threshold <= 1.0
        ):
            raise ValueError("horizon_negative_threshold must be within [0, 1].")
        if self.horizon_negative_threshold > self.horizon_positive_threshold:
            raise ValueError(
                "horizon_negative_threshold must be <= horizon_positive_threshold."
            )
        self.horizon_pair_budget = int(horizon_pair_budget)
        self.horizon_paths_per_pair = int(horizon_paths_per_pair)
        self.horizon_path_expansion_budget = int(horizon_path_expansion_budget)
        self.horizon_max_iterations = int(horizon_max_iterations)
        if self.horizon_pair_budget < 0:
            raise ValueError("horizon_pair_budget must be >= 0.")
        if self.horizon_paths_per_pair < 1:
            raise ValueError("horizon_paths_per_pair must be >= 1.")
        if self.horizon_path_expansion_budget < 1:
            raise ValueError("horizon_path_expansion_budget must be >= 1.")
        if self.horizon_max_iterations < 0:
            raise ValueError("horizon_max_iterations must be >= 0.")
        if solver_threads is not None and int(solver_threads) < 1:
            raise ValueError("solver_threads must be >= 1 when provided.")
        self.solver_threads = None if solver_threads is None else int(solver_threads)
        self.active_time_limit_seconds: Optional[float] = None
        self.last_adjacency_solve_report_: Optional[AdjacencySolveReport] = None
        self.last_adjacency_solve_reports_: List[AdjacencySolveReport] = []

    def optimize_adjacency_matrix(
        self,
        prob_matrix: np.ndarray,
        target_degrees: List[int],
        target_edge_count: Optional[int] = None,
        timeLimit: Optional[float] = None,
        verbose: bool = False,
        alpha: float = 0.7,
        connectivity: Optional[bool] = None,
        forbidden_edge_sets: Optional[Iterable[Iterable[Sequence[Any]]]] = None,
        active_node_mask: Optional[np.ndarray] = None,
        horizon_probability_matrix: Optional[np.ndarray] = None,
        horizon: Optional[int] = None,
        horizon_node_mask: Optional[np.ndarray] = None,
        _deadline_monotonic: Optional[float] = None,
    ) -> np.ndarray:
        if connectivity is None:
            connectivity = self.enforce_connectivity
        effective_time_limit = timeLimit
        if effective_time_limit is None:
            effective_time_limit = self.active_time_limit_seconds
        if effective_time_limit is None:
            effective_time_limit = self.adjacency_time_limit_seconds
        if active_node_mask is None and horizon_node_mask is not None:
            active_node_mask = horizon_node_mask
        adjacency, report = solve_adjacency(
            prob_matrix,
            target_degrees,
            target_edge_count=target_edge_count,
            time_limit_seconds=effective_time_limit,
            verbose=verbose,
            alpha=alpha,
            connectivity=bool(connectivity),
            forbidden_edge_sets=forbidden_edge_sets,
            active_node_mask=active_node_mask,
            degree_slack_penalty=self.degree_slack_penalty,
            edge_count_slack_penalty=self.edge_count_slack_penalty,
            warm_start_mst=self.warm_start_mst,
            horizon_probability_matrix=horizon_probability_matrix,
            horizon=horizon,
            use_horizon_constraints=self.use_horizon_ilp_constraints,
            horizon_constraint_weight=self.horizon_constraint_weight,
            horizon_positive_threshold=self.horizon_positive_threshold,
            horizon_negative_threshold=self.horizon_negative_threshold,
            horizon_pair_budget=self.horizon_pair_budget,
            horizon_paths_per_pair=self.horizon_paths_per_pair,
            horizon_path_expansion_budget=self.horizon_path_expansion_budget,
            horizon_max_iterations=self.horizon_max_iterations,
            solver_threads=self.solver_threads,
            deadline_monotonic=_deadline_monotonic,
        )
        self.last_adjacency_solve_report_ = report
        return adjacency

    def graphs_to_adjacency_matrices(self, graphs: List[nx.Graph]) -> List[np.ndarray]:
        return [nx.to_numpy_array(graph, dtype=int) for graph in graphs]

    def _target_stats(self, targets: List[int]) -> Tuple[int, int]:
        positive = int(sum(1 for target in targets if int(target) == 1))
        negative = int(len(targets) - positive)
        return positive, negative

    def _sample_pair_indices(
        self,
        targets: List[int],
        sample_count: int,
        locality_sampling_strategy: str,
        locality_target_positive_ratio: Optional[float],
    ) -> np.ndarray:
        num_pairs = len(targets)
        if sample_count <= 0:
            return np.asarray([], dtype=int)
        if sample_count >= num_pairs:
            return np.arange(num_pairs, dtype=int)

        targets_array = np.asarray(targets, dtype=int)
        if locality_sampling_strategy == "uniform":
            return np.random.choice(num_pairs, sample_count, replace=False)

        pos_indices = np.flatnonzero(targets_array == 1)
        neg_indices = np.flatnonzero(targets_array == 0)
        if len(pos_indices) == 0 or len(neg_indices) == 0:
            return np.random.choice(num_pairs, sample_count, replace=False)

        if locality_sampling_strategy == "stratified_target":
            target_positive_ratio = locality_target_positive_ratio
            if target_positive_ratio is None:
                raise ValueError(
                    "locality_sampling_strategy='stratified_target' requires locality_target_positive_ratio."
                )
        else:
            target_positive_ratio = len(pos_indices) / float(num_pairs)

        num_pos = int(round(sample_count * target_positive_ratio))
        num_pos = max(0, min(num_pos, sample_count))
        num_neg = sample_count - num_pos

        num_pos = min(num_pos, len(pos_indices))
        num_neg = min(num_neg, len(neg_indices))

        remaining = sample_count - (num_pos + num_neg)
        if remaining > 0:
            extra_pos = min(remaining, len(pos_indices) - num_pos)
            num_pos += extra_pos
            remaining -= extra_pos
        if remaining > 0:
            extra_neg = min(remaining, len(neg_indices) - num_neg)
            num_neg += extra_neg

        sampled_pos = np.random.choice(pos_indices, num_pos, replace=False) if num_pos > 0 else np.asarray([], dtype=int)
        sampled_neg = np.random.choice(neg_indices, num_neg, replace=False) if num_neg > 0 else np.asarray([], dtype=int)
        sampled = np.concatenate([sampled_pos, sampled_neg])
        np.random.shuffle(sampled)
        return sampled

    def adj_mtx_to_targets(
        self,
        adj_mtx_list: List[np.ndarray],
        node_encodings_list: List[np.ndarray],
        locality_sample_fraction: float,
        negative_sample_factor: int = 1,
        locality_sampling_strategy: str = "stratified_preserve",
        locality_target_positive_ratio: Optional[float] = None,
        force_bi_directional_edges: bool = True,
        is_training: bool = False,
        horizon: int = 1,
        supervision_name: str = "locality",
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        valid_sampling_strategies = {"uniform", "stratified_preserve", "stratified_target"}
        if locality_sampling_strategy not in valid_sampling_strategies:
            raise ValueError(
                f"locality_sampling_strategy must be one of {sorted(valid_sampling_strategies)} "
                f"(got {locality_sampling_strategy!r})."
            )
        if locality_target_positive_ratio is not None and not 0.0 < locality_target_positive_ratio < 1.0:
            raise ValueError("locality_target_positive_ratio must be between 0 and 1 when provided.")

        all_targets = []
        all_pairs = []
        for g_idx, (adj_mtx, encodings) in enumerate(zip(adj_mtx_list, node_encodings_list)):
            n_nodes = adj_mtx.shape[0]
            graph = nx.from_numpy_array(adj_mtx, create_using=nx.Graph)
            shortest_paths = dict(nx.all_pairs_shortest_path_length(graph, cutoff=horizon))
            encodings = np.asarray(encodings, dtype=float)
            for i in range(n_nodes):
                lengths = shortest_paths.get(i, {i: 0})
                pos_neighbors = [j for j, dist in lengths.items() if j != i and dist <= horizon]
                for j in pos_neighbors:
                    all_targets.append(1)
                    all_pairs.append((g_idx, i, j))
                    if force_bi_directional_edges:
                        all_targets.append(1)
                        all_pairs.append((g_idx, j, i))
                num_pos = len(pos_neighbors) * (2 if force_bi_directional_edges else 1)
                num_neg_samples = int(round(negative_sample_factor * num_pos))
                if num_neg_samples <= 0:
                    continue
                candidate_mask = np.ones(n_nodes, dtype=bool)
                candidate_mask[i] = False
                candidate_mask[list(lengths.keys())] = False
                candidate_indices = np.flatnonzero(candidate_mask)
                if candidate_indices.size == 0:
                    continue
                candidate_vectors = encodings[candidate_indices]
                distances = np.linalg.norm(candidate_vectors - encodings[i], axis=1)
                sorted_candidate_indices = np.argsort(distances, kind="stable")
                selected_negatives = candidate_indices[sorted_candidate_indices[:num_neg_samples]]
                for k in selected_negatives:
                    k = int(k)
                    all_targets.append(0)
                    all_pairs.append((g_idx, i, k))
                    if force_bi_directional_edges:
                        all_targets.append(0)
                        all_pairs.append((g_idx, k, i))

        pos_before, neg_before = self._target_stats(all_targets)
        if is_training and locality_sample_fraction < 1.0:
            num_pairs = len(all_pairs)
            num_pairs_to_use = int(round(num_pairs * locality_sample_fraction))
            if self.verbose and num_pairs > 0:
                verbose_log(
                    self,
                    f"adj_mtx_to_targets[{supervision_name}, horizon={horizon}]: "
                    f"sampling {num_pairs_to_use} pairs ({locality_sample_fraction:.2%}) "
                    f"from {num_pairs} total pairs "
                    f"(pos={pos_before}, neg={neg_before}, "
                    f"negative_sample_factor={negative_sample_factor}, "
                    f"sampling_strategy={locality_sampling_strategy}"
                    f"{'' if locality_target_positive_ratio is None else f', target_positive_ratio={locality_target_positive_ratio:.3f}'}).",
                    level=1,
                )
            if 0 < num_pairs_to_use < num_pairs:
                indices = self._sample_pair_indices(
                    all_targets,
                    num_pairs_to_use,
                    locality_sampling_strategy=locality_sampling_strategy,
                    locality_target_positive_ratio=locality_target_positive_ratio,
                )
                all_targets = [all_targets[i] for i in indices]
                all_pairs = [all_pairs[i] for i in indices]
            elif num_pairs_to_use == 0 and num_pairs > 0:
                if self.verbose:
                    verbose_log(
                        self,
                        f"adj_mtx_to_targets[{supervision_name}, horizon={horizon}]: "
                        f"warning - num_pairs_to_use is 0 with locality_sample_fraction="
                        f"{locality_sample_fraction} and num_pairs={num_pairs}. No pairs will be used.",
                        level=1,
                    )
                return np.array([]), []
            elif num_pairs_to_use == 0 and num_pairs == 0:
                return np.array([]), []

        if self.verbose and len(all_targets) > 0:
            pos_after, neg_after = self._target_stats(all_targets)
            ratio_after = pos_after / float(pos_after + neg_after)
            verbose_log(
                self,
                f"adj_mtx_to_targets[{supervision_name}, horizon={horizon}]: "
                f"using pos={pos_after}, neg={neg_after}, positive_ratio={ratio_after:.3f}.",
                level=1,
            )

        return np.array(all_targets), all_pairs

    def compute_edge_supervision(
        self,
        graphs: List[nx.Graph],
        node_encodings_list: List[np.ndarray],
        locality_sample_fraction: float,
        negative_sample_factor: int = 1,
        locality_sampling_strategy: str = "stratified_preserve",
        locality_target_positive_ratio: Optional[float] = None,
        horizon: int = 1,
        supervision_name: str = "locality",
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        adj = self.graphs_to_adjacency_matrices(graphs)
        return self.adj_mtx_to_targets(
            adj,
            node_encodings_list,
            locality_sample_fraction=locality_sample_fraction,
            negative_sample_factor=negative_sample_factor,
            locality_sampling_strategy=locality_sampling_strategy,
            locality_target_positive_ratio=locality_target_positive_ratio,
            is_training=True,
            horizon=horizon,
            supervision_name=supervision_name,
        )

    def encodings_and_adj_mtx_to_dataset(
        self,
        node_encodings_list: List[np.ndarray],
        adj_mtx_list: List[np.ndarray],
        locality_sample_fraction: float,
        horizon: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        y, pair_indices = self.adj_mtx_to_targets(
            adj_mtx_list,
            node_encodings_list,
            locality_sample_fraction=locality_sample_fraction,
            is_training=True,
            horizon=horizon,
        )
        X = self.encodings_to_instances(node_encodings_list, pair_indices)
        return X, y

    def encodings_to_instances(
        self,
        node_encodings_list: List[np.ndarray],
        pair_indices: Optional[List[Tuple[int, int, int]]] = None,
        use_graph_encoding: bool = False,
    ) -> np.ndarray:
        instances = []
        if pair_indices is not None:
            for g_idx, i, j in pair_indices:
                encodings = node_encodings_list[g_idx]
                if use_graph_encoding:
                    graph_encoding = np.sum(encodings, axis=0)
                    instance = np.hstack([graph_encoding, encodings[i], encodings[j]])
                else:
                    instance = np.hstack([encodings[i], encodings[j]])
                instances.append(instance)
        else:
            for _, encodings in enumerate(node_encodings_list):
                if use_graph_encoding:
                    graph_encoding = np.sum(encodings, axis=0)
                n_nodes = encodings.shape[0]
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        if i != j:
                            if use_graph_encoding:
                                instance = np.hstack([graph_encoding, encodings[i], encodings[j]])
                            else:
                                instance = np.hstack([encodings[i], encodings[j]])
                            instances.append(instance)
        return np.vstack(instances)

    def _resolve_target_edge_count(
        self,
        n_nodes: int,
        desired_edge_count: Optional[int],
        *,
        connectivity: Optional[bool] = None,
    ) -> Optional[int]:
        if desired_edge_count is None:
            return None
        if connectivity is None:
            connectivity = self.enforce_connectivity
        n_nodes = int(max(0, n_nodes))
        max_edges = (n_nodes * (n_nodes - 1)) // 2
        min_edges = (n_nodes - 1) if connectivity and n_nodes >= 2 else 0
        return int(np.clip(int(np.rint(desired_edge_count)), min_edges, max_edges))

    def resolve_node_presence_mask(
        self,
        node_presence_mask: np.ndarray,
        *,
        desired_node_count: Optional[int] = None,
        node_existence_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        mask = np.asarray(node_presence_mask, dtype=bool)
        if desired_node_count is None:
            return mask
        n_slots = int(mask.shape[0])
        desired_node_count = int(np.clip(int(np.rint(desired_node_count)), 0, n_slots))
        resolved = np.zeros(n_slots, dtype=bool)
        if desired_node_count == 0:
            return resolved
        if node_existence_scores is None:
            scores = mask.astype(float)
        else:
            scores = np.asarray(node_existence_scores, dtype=float)
            if scores.shape[0] != n_slots:
                raise ValueError(
                    "node_existence_scores must align with node_presence_mask "
                    f"(got {scores.shape[0]} scores for {n_slots} slots)."
                )
        top_indices = np.argsort(scores, kind="stable")[-desired_node_count:]
        resolved[top_indices] = True
        return resolved

    def _project_degrees_to_edge_budget(
        self,
        active_degree_predictions: np.ndarray,
        desired_edge_count: Optional[int],
        *,
        connectivity: Optional[bool] = None,
    ) -> np.ndarray:
        active_degree_predictions = np.asarray(active_degree_predictions, dtype=float)
        n_active = int(active_degree_predictions.shape[0])
        if n_active == 0:
            return np.zeros((0,), dtype=np.int64)
        max_degree = max(0, n_active - 1)
        degrees = np.clip(np.rint(active_degree_predictions), 0, max_degree).astype(np.int64)
        target_edge_count = self._resolve_target_edge_count(
            n_active,
            desired_edge_count,
            connectivity=connectivity,
        )
        if target_edge_count is None:
            return degrees
        target_total_degree = int(2 * target_edge_count)
        current_total_degree = int(degrees.sum())
        while current_total_degree < target_total_degree:
            headroom = max_degree - degrees
            candidate_indices = np.flatnonzero(headroom > 0)
            if candidate_indices.size == 0:
                break
            candidate_scores = active_degree_predictions[candidate_indices] - degrees[candidate_indices]
            best_local_idx = int(np.argmax(candidate_scores + 1e-6 * active_degree_predictions[candidate_indices]))
            degrees[candidate_indices[best_local_idx]] += 1
            current_total_degree += 1
        while current_total_degree > target_total_degree:
            candidate_indices = np.flatnonzero(degrees > 0)
            if candidate_indices.size == 0:
                break
            candidate_scores = degrees[candidate_indices] - active_degree_predictions[candidate_indices]
            best_local_idx = int(np.argmax(candidate_scores + 1e-6 * (max_degree - active_degree_predictions[candidate_indices])))
            degrees[candidate_indices[best_local_idx]] -= 1
            current_total_degree -= 1
        return degrees

    def get_degree_targets(
        self,
        node_degree_predictions: np.ndarray,
        node_presence_mask: np.ndarray,
        *,
        desired_edge_count: Optional[int] = None,
        connectivity: Optional[bool] = None,
    ) -> List[int]:
        predictions = np.asarray(node_degree_predictions, dtype=float)
        mask = np.asarray(node_presence_mask, dtype=bool)
        active_indices = np.flatnonzero(mask)
        projected_active_degrees = self._project_degrees_to_edge_budget(
            predictions[active_indices],
            desired_edge_count,
            connectivity=connectivity,
        )
        target_degrees = np.zeros(mask.shape[0], dtype=np.int64)
        target_degrees[active_indices] = projected_active_degrees
        return [int(value) for value in target_degrees]

    def get_degrees(self, node_degree_predictions: np.ndarray, node_presence_mask: np.ndarray) -> List[int]:
        return self.get_degree_targets(node_degree_predictions, node_presence_mask)

    def decode_adjacency_matrix(
        self,
        generated_nodes: GeneratedNodeBatch,
        predicted_edge_probability_matrices: Optional[List[np.ndarray]] = None,
        horizon_probability_matrices: Optional[List[np.ndarray]] = None,
        horizon: Optional[int] = None,
        desired_node_counts: Optional[Sequence[int]] = None,
        desired_edge_counts: Optional[Sequence[int]] = None,
    ) -> List[np.ndarray]:
        if generated_nodes.node_presence_mask is None:
            raise RuntimeError("decode_adjacency_matrix requires node presence predictions.")
        if generated_nodes.node_degree_predictions is None:
            raise RuntimeError("decode_adjacency_matrix requires node degree predictions.")
        if predicted_edge_probability_matrices is None:
            raise RuntimeError("decode_adjacency_matrix requires explicit edge probability matrices.")

        existence_masks = np.asarray(generated_nodes.node_presence_mask, dtype=bool)
        existence_scores = (
            None
            if generated_nodes.node_existence_probabilities is None
            else np.asarray(generated_nodes.node_existence_probabilities, dtype=float)
        )
        degree_predictions = np.asarray(generated_nodes.node_degree_predictions, dtype=float)
        if desired_node_counts is not None and len(desired_node_counts) != len(existence_masks):
            raise ValueError(
                "desired_node_counts must align with generated_nodes "
                f"(got {len(desired_node_counts)} counts for {len(existence_masks)} graphs)."
            )
        if desired_edge_counts is not None and len(desired_edge_counts) != len(existence_masks):
            raise ValueError(
                "desired_edge_counts must align with generated_nodes "
                f"(got {len(desired_edge_counts)} counts for {len(existence_masks)} graphs)."
            )
        if horizon_probability_matrices is None:
            horizon_probability_matrices = generated_nodes.horizon_probability_matrices
        if horizon is None:
            horizon = generated_nodes.horizon
        if horizon_probability_matrices is not None and len(horizon_probability_matrices) != len(existence_masks):
            raise ValueError(
                "horizon_probability_matrices must align with generated_nodes "
                f"(got {len(horizon_probability_matrices)} matrices for {len(existence_masks)} graphs)."
            )
        predicted_probs_list = []
        horizon_probs_list = []
        for existence_mask, degree_prediction, prob_matrix in zip(
            existence_masks,
            degree_predictions,
            predicted_edge_probability_matrices,
        ):
            n_nodes = min(len(existence_mask), len(degree_prediction))
            prob_matrix = _validate_probability_values(
                prob_matrix,
                name="Edge-probability matrices",
            )
            if prob_matrix.ndim == 2:
                if prob_matrix.shape[0] != n_nodes or prob_matrix.shape[1] != n_nodes:
                    raise ValueError(
                        "Edge-probability matrices must align with node predictions; "
                        f"received {prob_matrix.shape} for n_nodes={n_nodes}."
                    )
                mask = ~np.eye(n_nodes, dtype=bool)
                predicted_probs_list.append(prob_matrix[mask])
            elif prob_matrix.ndim == 1:
                expected_probability_count = n_nodes * (n_nodes - 1)
                if prob_matrix.shape[0] != expected_probability_count:
                    raise ValueError(
                        "Flattened edge-probability vectors must contain exactly "
                        f"n_nodes * (n_nodes - 1)={expected_probability_count} values; "
                        f"received {prob_matrix.shape[0]} for n_nodes={n_nodes}."
                    )
                predicted_probs_list.append(prob_matrix)
            else:
                raise ValueError(
                    "Edge-probability predictions must be square matrices or "
                    f"one-dimensional flattened vectors; received shape {prob_matrix.shape}."
                )
        if horizon_probability_matrices is None:
            horizon_probs_list = [None for _ in predicted_probs_list]
        else:
            for graph_idx, (existence_mask, degree_prediction, horizon_matrix) in enumerate(
                zip(existence_masks, degree_predictions, horizon_probability_matrices)
            ):
                n_nodes = min(len(existence_mask), len(degree_prediction))
                horizon_matrix = _validate_probability_values(
                    horizon_matrix,
                    name="Horizon-probability matrices",
                )
                if horizon_matrix.shape != (n_nodes, n_nodes):
                    raise ValueError(
                        "Horizon-probability matrices must align with node predictions; "
                        f"graph {graph_idx} received {horizon_matrix.shape} for n_nodes={n_nodes}."
                    )
                horizon_probs_list.append(horizon_matrix)

        timeout_seconds = self.parallel_decode_timeout_seconds
        if self.active_time_limit_seconds is not None:
            timeout_seconds = min(
                float(timeout_seconds) if timeout_seconds is not None else float(self.active_time_limit_seconds),
                float(self.active_time_limit_seconds),
            )
        configured_solver_timeout = (
            float(self.active_time_limit_seconds)
            if self.active_time_limit_seconds is not None
            else (
                None
                if self.adjacency_time_limit_seconds is None
                else float(self.adjacency_time_limit_seconds)
            )
        )
        solver_timeout_seconds = _solver_budget_with_parent_reserve(
            configured_solver_timeout,
            timeout_seconds,
        )
        worker_limit = max(1, min(int(self.n_jobs), len(predicted_probs_list)))
        timeout_waves = (
            (len(predicted_probs_list) + worker_limit - 1) // worker_limit
            if predicted_probs_list
            else 1
        )
        parent_timeout_seconds = (
            None
            if timeout_seconds is None
            else float(timeout_seconds) * float(max(1, timeout_waves))
        )
        if (
            parent_timeout_seconds is not None
            and self.active_time_limit_seconds is not None
        ):
            parent_timeout_seconds = min(
                parent_timeout_seconds,
                float(self.active_time_limit_seconds),
            )
        parent_deadline_monotonic = (
            None
            if parent_timeout_seconds is None
            else time.monotonic() + float(parent_timeout_seconds)
        )
        jobs = [
            (
                np.asarray(predicted_probs_list[graph_idx], dtype=float),
                np.asarray(existence_masks[graph_idx], dtype=bool),
                None if existence_scores is None else np.asarray(existence_scores[graph_idx], dtype=float),
                np.asarray(degree_predictions[graph_idx], dtype=float),
                None if desired_node_counts is None else int(desired_node_counts[graph_idx]),
                None if desired_edge_counts is None else int(desired_edge_counts[graph_idx]),
                float(self.degree_slack_penalty),
                self.edge_count_slack_penalty,
                bool(self.enforce_connectivity),
                bool(self.warm_start_mst),
                int(self.verbose),
                self.diagnostic_graph_renderer if self.n_jobs == 1 else None,
                solver_timeout_seconds,
                None if horizon_probs_list[graph_idx] is None else np.asarray(horizon_probs_list[graph_idx], dtype=float),
                None if horizon is None else int(horizon),
                bool(self.use_horizon_ilp_constraints),
                float(self.horizon_constraint_weight),
                float(self.horizon_positive_threshold),
                float(self.horizon_negative_threshold),
                int(self.horizon_pair_budget),
                int(self.horizon_paths_per_pair),
                int(self.horizon_path_expansion_budget),
                int(self.horizon_max_iterations),
                None if self.solver_threads is None else int(self.solver_threads),
                None,
                None if timeout_seconds is None else float(timeout_seconds),
            )
            for graph_idx in range(len(predicted_probs_list))
        ]
        if int(self.verbose) >= 4 and self.n_jobs != 1 and len(jobs) > 1:
            verbose_log(
                self,
                "Decoder plots for verbose>=4 are only shown when n_jobs=1; skipping plots during parallel adjacency decode.",
                level=4,
            )
        if timeout_seconds is not None:
            results = _parallel_map(
                _decode_single_adjacency_job_star,
                jobs,
                self.n_jobs,
                verbose=bool(self.verbose),
                timeout_seconds=parent_timeout_seconds,
                per_job_timeout_seconds=timeout_seconds,
                timeout_fallback_label="parallel adjacency decode",
                fallback_on_timeout=False,
                deadline_monotonic=parent_deadline_monotonic,
            )
        elif self.n_jobs == 1 or len(jobs) <= 1:
            results = [_decode_single_adjacency_job(*job) for job in jobs]
        else:
            results = _parallel_map(
                _decode_single_adjacency_job_star,
                jobs,
                self.n_jobs,
                verbose=bool(self.verbose),
                timeout_seconds=timeout_seconds,
                timeout_fallback_label="parallel adjacency decode",
                fallback_on_timeout=False,
            )
        self.last_adjacency_solve_reports_ = [report for _, report in results]
        self.last_adjacency_solve_report_ = (
            self.last_adjacency_solve_reports_[-1]
            if self.last_adjacency_solve_reports_
            else None
        )
        return [adjacency for adjacency, _ in results]

    def decode_adjacency_matrix_direct(
        self,
        generated_nodes: GeneratedNodeBatch,
        predicted_edge_probability_matrices: Optional[List[np.ndarray]] = None,
        desired_node_counts: Optional[Sequence[int]] = None,
        desired_edge_counts: Optional[Sequence[int]] = None,
        edge_probability_threshold: Optional[float] = None,
    ) -> List[np.ndarray]:
        if generated_nodes.node_presence_mask is None:
            raise RuntimeError("decode_adjacency_matrix_direct requires node presence predictions.")
        if predicted_edge_probability_matrices is None:
            raise RuntimeError(
                "decode_adjacency_matrix_direct requires explicit edge probability matrices."
            )

        existence_masks = np.asarray(generated_nodes.node_presence_mask, dtype=bool)
        existence_scores = None if generated_nodes.node_existence_probabilities is None else np.asarray(
            generated_nodes.node_existence_probabilities,
            dtype=float,
        )
        degree_predictions = (
            None
            if generated_nodes.node_degree_predictions is None
            else np.asarray(generated_nodes.node_degree_predictions, dtype=float)
        )
        if desired_node_counts is not None and len(desired_node_counts) != len(existence_masks):
            raise ValueError(
                "desired_node_counts must align with generated_nodes "
                f"(got {len(desired_node_counts)} counts for {len(existence_masks)} graphs)."
            )
        if desired_edge_counts is not None and len(desired_edge_counts) != len(existence_masks):
            raise ValueError(
                "desired_edge_counts must align with generated_nodes "
                f"(got {len(desired_edge_counts)} counts for {len(existence_masks)} graphs)."
            )
        if degree_predictions is not None and len(degree_predictions) != len(existence_masks):
            raise ValueError(
                "node_degree_predictions must align with generated_nodes "
                f"(got {len(degree_predictions)} degree rows for {len(existence_masks)} graphs)."
            )

        threshold = (
            float(self.direct_edge_probability_threshold)
            if edge_probability_threshold is None
            else float(edge_probability_threshold)
        )
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("edge_probability_threshold must be within [0, 1].")
        return [
            self._decode_single_adjacency_matrix_direct(
                existence_mask=existence_mask,
                existence_scores=None if existence_scores is None else existence_scores[graph_idx],
                prob_matrix=prob_matrix,
                node_degree_predictions=(
                    None if degree_predictions is None else degree_predictions[graph_idx]
                ),
                desired_node_count=(
                    None if desired_node_counts is None else int(desired_node_counts[graph_idx])
                ),
                desired_edge_count=(
                    None if desired_edge_counts is None else int(desired_edge_counts[graph_idx])
                ),
                threshold=threshold,
            )
            for graph_idx, (existence_mask, prob_matrix) in enumerate(
                zip(existence_masks, predicted_edge_probability_matrices)
            )
        ]

    @staticmethod
    def _direct_edge_candidates(active_indices: np.ndarray, prob_matrix: np.ndarray):
        return _direct_edge_candidates_impl(active_indices, prob_matrix)

    @staticmethod
    def _select_direct_edges_top_k(edge_candidates, desired_edge_count: int):
        return _select_direct_edges_top_k_impl(edge_candidates, desired_edge_count)

    @staticmethod
    def _select_direct_edges_degree_aware(
        edge_candidates,
        active_indices: np.ndarray,
        target_degrees: Sequence[int],
        desired_edge_count: int,
    ):
        return _select_direct_edges_degree_aware_impl(
            edge_candidates,
            active_indices,
            target_degrees,
            desired_edge_count,
        )

    @staticmethod
    def _select_direct_edges_by_threshold(edge_candidates, threshold: float):
        return _select_direct_edges_by_threshold_impl(edge_candidates, threshold)

    @staticmethod
    def _adjacency_from_selected_edges(n_nodes: int, selected_edges) -> np.ndarray:
        return _adjacency_from_selected_edges_impl(n_nodes, selected_edges)

    def _decode_single_adjacency_matrix_direct(
        self,
        *,
        existence_mask: np.ndarray,
        existence_scores: Optional[np.ndarray],
        prob_matrix: np.ndarray,
        node_degree_predictions: Optional[np.ndarray],
        desired_node_count: Optional[int],
        desired_edge_count: Optional[int],
        threshold: float,
    ) -> np.ndarray:
        prob_matrix = _validate_probability_values(
            prob_matrix,
            name="Direct edge-probability matrix",
        )
        n_nodes = int(len(existence_mask))
        if (
            prob_matrix.ndim != 2
            or prob_matrix.shape[0] != n_nodes
            or prob_matrix.shape[1] != n_nodes
        ):
            raise ValueError(
                "Direct edge decoding requires square edge-probability matrices aligned "
                f"with node predictions; received {prob_matrix.shape} for n_nodes={n_nodes}."
            )
        resolved_mask = self.resolve_node_presence_mask(
            np.asarray(existence_mask, dtype=bool),
            desired_node_count=desired_node_count,
            node_existence_scores=None if existence_scores is None else np.asarray(
                existence_scores,
                dtype=float,
            ),
        )
        active_indices = np.flatnonzero(resolved_mask)
        edge_candidates = self._direct_edge_candidates(active_indices, prob_matrix)
        if desired_edge_count is not None and node_degree_predictions is not None:
            degree_predictions = np.asarray(node_degree_predictions, dtype=float)
            if degree_predictions.shape[0] != n_nodes:
                raise ValueError(
                    "Direct degree-aware decoding requires node-degree predictions aligned "
                    f"with node predictions; received {degree_predictions.shape[0]} degree "
                    f"values for n_nodes={n_nodes}."
                )
            target_degrees = self.get_degree_targets(
                degree_predictions,
                resolved_mask,
                desired_edge_count=desired_edge_count,
                connectivity=False,
            )
            selected_edges = self._select_direct_edges_degree_aware(
                edge_candidates,
                active_indices,
                target_degrees,
                desired_edge_count,
            )
        elif desired_edge_count is not None:
            selected_edges = self._select_direct_edges_top_k(edge_candidates, desired_edge_count)
        else:
            selected_edges = self._select_direct_edges_by_threshold(edge_candidates, threshold)
        return self._adjacency_from_selected_edges(n_nodes, selected_edges)

    def decode_node_labels(
        self,
        generated_nodes: GeneratedNodeBatch,
        predicted_node_labels_list: Optional[List[np.ndarray]] = None,
    ) -> List[np.ndarray]:
        if predicted_node_labels_list is None:
            raise RuntimeError("decode_node_labels requires explicit node labels.")
        expected_graph_count = len(np.asarray(generated_nodes.node_presence_mask, dtype=bool))
        if len(predicted_node_labels_list) != expected_graph_count:
            raise ValueError(
                "predicted_node_labels_list must align with generated_nodes "
                f"(got {len(predicted_node_labels_list)} label arrays for {expected_graph_count} graphs)."
            )
        return [
            _validate_node_label_array(
                node_labels,
                graph_idx=graph_idx,
                n_slots=int(np.asarray(generated_nodes.node_presence_mask[graph_idx], dtype=bool).shape[0]),
            )
            for graph_idx, node_labels in enumerate(predicted_node_labels_list)
        ]

    def decode_edge_labels(
        self,
        generated_nodes: GeneratedNodeBatch,
        adj_mtx_list: List[np.ndarray],
        predicted_edge_label_matrices: Optional[List[np.ndarray]] = None,
        predicted_edge_labels_list: Optional[List[np.ndarray]] = None,
    ) -> List[np.ndarray]:
        if predicted_edge_labels_list is not None:
            if len(predicted_edge_labels_list) != len(adj_mtx_list):
                raise ValueError(
                    "predicted_edge_labels_list must align with adj_mtx_list "
                    f"(got {len(predicted_edge_labels_list)} label arrays for {len(adj_mtx_list)} graphs)."
                )
            return [
                _validate_edge_label_array(
                    edge_labels,
                    graph_idx=graph_idx,
                    expected_edge_count=int(np.sum(np.asarray(adj_mtx, dtype=float)) // 2),
                )
                for graph_idx, (adj_mtx, edge_labels) in enumerate(zip(adj_mtx_list, predicted_edge_labels_list))
            ]

        if predicted_edge_label_matrices is not None:
            if len(predicted_edge_label_matrices) != len(adj_mtx_list):
                raise ValueError(
                    "predicted_edge_label_matrices must align with adj_mtx_list "
                    f"(got {len(predicted_edge_label_matrices)} matrices for {len(adj_mtx_list)} graphs)."
                )
            return [
                _assemble_edge_labels_from_matrix(adj_mtx, np.asarray(edge_label_matrix, dtype=object))
                for adj_mtx, edge_label_matrix in zip(adj_mtx_list, predicted_edge_label_matrices)
            ]

        raise RuntimeError("decode_edge_labels requires explicit edge labels or edge-label matrices.")

    def decode(
        self,
        generated_nodes: GeneratedNodeBatch,
        predicted_node_labels_list: Optional[List[np.ndarray]] = None,
        predicted_edge_probability_matrices: Optional[List[np.ndarray]] = None,
        predicted_edge_labels_list: Optional[List[np.ndarray]] = None,
        predicted_edge_label_matrices: Optional[List[np.ndarray]] = None,
        horizon_probability_matrices: Optional[List[np.ndarray]] = None,
        horizon: Optional[int] = None,
        desired_node_counts: Optional[Sequence[int]] = None,
        desired_edge_counts: Optional[Sequence[int]] = None,
        use_ilp_decoder: bool = True,
        edge_probability_threshold: Optional[float] = None,
    ) -> List[nx.Graph]:
        if use_ilp_decoder:
            adj_mtx_list = self.decode_adjacency_matrix(
                generated_nodes,
                predicted_edge_probability_matrices=predicted_edge_probability_matrices,
                horizon_probability_matrices=horizon_probability_matrices,
                horizon=horizon,
                desired_node_counts=desired_node_counts,
                desired_edge_counts=desired_edge_counts,
            )
        else:
            adj_mtx_list = self.decode_adjacency_matrix_direct(
                generated_nodes,
                predicted_edge_probability_matrices=predicted_edge_probability_matrices,
                desired_node_counts=desired_node_counts,
                desired_edge_counts=desired_edge_counts,
                edge_probability_threshold=edge_probability_threshold,
            )
        predicted_node_labels_list = self.decode_node_labels(
            generated_nodes,
            predicted_node_labels_list=predicted_node_labels_list,
        )
        predicted_edge_labels_list = self.decode_edge_labels(
            generated_nodes,
            adj_mtx_list,
            predicted_edge_labels_list=predicted_edge_labels_list,
            predicted_edge_label_matrices=predicted_edge_label_matrices,
        )

        resolved_presence_masks = [
            self.resolve_node_presence_mask(
                np.asarray(generated_nodes.node_presence_mask[graph_idx], dtype=bool),
                desired_node_count=None if desired_node_counts is None else int(desired_node_counts[graph_idx]),
                node_existence_scores=None if generated_nodes.node_existence_probabilities is None else np.asarray(
                    generated_nodes.node_existence_probabilities[graph_idx],
                    dtype=float,
                ),
            )
            for graph_idx in range(len(adj_mtx_list))
        ]
        predicted_node_labels_list = [
            _validate_node_label_array(
                node_labels,
                graph_idx=graph_idx,
                n_slots=int(np.asarray(node_presence_mask, dtype=bool).shape[0]),
            )
            for graph_idx, (node_labels, node_presence_mask) in enumerate(
                zip(predicted_node_labels_list, resolved_presence_masks)
            )
        ]
        predicted_edge_labels_list = [
            _validate_edge_label_array(
                edge_labels,
                graph_idx=graph_idx,
                expected_edge_count=int(np.sum(np.asarray(adj_mtx, dtype=float)) // 2),
            )
            for graph_idx, (edge_labels, adj_mtx) in enumerate(zip(predicted_edge_labels_list, adj_mtx_list))
        ]
        jobs = [
            (
                np.asarray(node_presence_mask, dtype=bool),
                np.asarray(node_labels, dtype=object),
                np.asarray(edge_labels, dtype=object),
                np.asarray(adj_mtx, dtype=float),
            )
            for node_presence_mask, node_labels, edge_labels, adj_mtx in zip(
                resolved_presence_masks,
                predicted_node_labels_list,
                predicted_edge_labels_list,
                adj_mtx_list,
            )
        ]
        if self.n_jobs == 1 or len(jobs) <= 1:
            decoded_graphs = [_assemble_graph_job(*job) for job in jobs]
        else:
            decoded_graphs = _parallel_map(
                _assemble_graph_job_star,
                jobs,
                self.n_jobs,
                verbose=bool(self.verbose),
                timeout_seconds=self.parallel_decode_timeout_seconds,
                timeout_fallback_label="parallel graph assembly",
            )

        if int(self.verbose) >= 4:
            for graph_idx, (adj_mtx, decoded_graph) in enumerate(zip(adj_mtx_list, decoded_graphs)):
                existence_mask = np.asarray(resolved_presence_masks[graph_idx], dtype=bool)
                degree_prediction = np.asarray(generated_nodes.node_degree_predictions[graph_idx], dtype=float)
                prob_matrix = np.asarray(predicted_edge_probability_matrices[graph_idx], dtype=float)
                masked_prob_matrix = _build_masked_prob_matrix(
                    existence_mask=existence_mask,
                    degree_prediction=degree_prediction,
                    prob_matrix=prob_matrix,
                )
                target_degrees = self.get_degree_targets(
                    degree_prediction,
                    existence_mask,
                    desired_edge_count=None if desired_edge_counts is None else int(desired_edge_counts[graph_idx]),
                )
                _plot_decoder_diagnostics(
                    prob_matrix=masked_prob_matrix,
                    adj_mtx=np.asarray(adj_mtx, dtype=float),
                    target_degrees=target_degrees,
                    title=f"Decoder solve graph={graph_idx}",
                    decoded_graph=decoded_graph,
                    graph_renderer=self.diagnostic_graph_renderer,
                    existence_mask=existence_mask,
                )
        return decoded_graphs

    def decode_generated_nodes(
        self,
        owner,
        generated_nodes: GeneratedNodeBatch,
        graph_conditioning: Optional[GraphConditioningBatch] = None,
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
        attempt_idx: int = 0,
    ) -> List[nx.Graph]:
        return decode_generated_nodes(
            owner,
            generated_nodes,
            graph_conditioning=graph_conditioning,
            feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
            attempt_idx=attempt_idx,
        )

    def decode_generated_nodes_with_oracle(
        self,
        owner,
        generated_nodes: GeneratedNodeBatch,
        graph_conditioning: Optional[GraphConditioningBatch] = None,
    ) -> List[Optional[nx.Graph]]:
        return decode_generated_nodes_with_oracle(
            owner,
            generated_nodes,
            graph_conditioning=graph_conditioning,
        )

    def save(self, filename: str = "generative_model.obj") -> None:
        path = Path(filename)
        artifact = {
            "artifact_type": "ConditionalNodeFieldGraphDecoder",
            "artifact_version": _DECODER_ARTIFACT_VERSION,
            "config": {
                "verbose": self.verbose,
                "direct_edge_probability_threshold": self.direct_edge_probability_threshold,
                "enforce_connectivity": self.enforce_connectivity,
                "degree_slack_penalty": self.degree_slack_penalty,
                "edge_count_slack_penalty": self.edge_count_slack_penalty,
                "warm_start_mst": self.warm_start_mst,
                "n_jobs": self.n_jobs,
                "adjacency_time_limit_seconds": self.adjacency_time_limit_seconds,
                "parallel_decode_timeout_seconds": self.parallel_decode_timeout_seconds,
                "use_horizon_ilp_constraints": self.use_horizon_ilp_constraints,
                "horizon_constraint_weight": self.horizon_constraint_weight,
                "horizon_positive_threshold": self.horizon_positive_threshold,
                "horizon_negative_threshold": self.horizon_negative_threshold,
                "horizon_pair_budget": self.horizon_pair_budget,
                "horizon_paths_per_pair": self.horizon_paths_per_pair,
                "horizon_path_expansion_budget": self.horizon_path_expansion_budget,
                "horizon_max_iterations": self.horizon_max_iterations,
                "solver_threads": self.solver_threads,
            },
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(artifact, handle, indent=2, sort_keys=True)

    def load(self, filename: str = "generative_model.obj") -> "ConditionalNodeFieldGraphDecoder":
        path = Path(filename)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                artifact = json.load(handle)
        except (UnicodeDecodeError, json.JSONDecodeError):
            with open(path, "rb") as handle:
                restored = pickle.load(handle)
            if not hasattr(restored, "direct_edge_probability_threshold"):
                restored.direct_edge_probability_threshold = float(
                    getattr(restored, "existence_threshold", 0.5)
                )
            if hasattr(restored, "existence_threshold"):
                delattr(restored, "existence_threshold")
            if not hasattr(restored, "horizon_path_expansion_budget"):
                restored.horizon_path_expansion_budget = 4096
            restored.last_adjacency_solve_report_ = None
            restored.last_adjacency_solve_reports_ = []
            return restored
        if artifact.get("artifact_type") != "ConditionalNodeFieldGraphDecoder":
            raise RuntimeError(
                f"Unsupported decoder artifact type {artifact.get('artifact_type')!r} in {path}."
            )
        artifact_version = int(artifact.get("artifact_version", 0))
        if artifact_version not in {1, 2, _DECODER_ARTIFACT_VERSION}:
            raise RuntimeError(
                "Saved decoder artifact version is incompatible with this NodeField version. "
                f"Expected v1, v2, or v{_DECODER_ARTIFACT_VERSION}, found "
                f"v{artifact_version}: {path}"
            )
        config = artifact.get("config", {})
        direct_edge_probability_threshold = config.get(
            "direct_edge_probability_threshold",
            config.get("existence_threshold", 0.5),
        )
        return self.__class__(
            verbose=config.get("verbose", True),
            direct_edge_probability_threshold=direct_edge_probability_threshold,
            enforce_connectivity=config.get("enforce_connectivity", True),
            degree_slack_penalty=config.get("degree_slack_penalty", 1e6),
            edge_count_slack_penalty=config.get("edge_count_slack_penalty", 2.0),
            warm_start_mst=config.get("warm_start_mst", True),
            n_jobs=config.get("n_jobs", 1),
            adjacency_time_limit_seconds=config.get("adjacency_time_limit_seconds", 60.0),
            parallel_decode_timeout_seconds=config.get("parallel_decode_timeout_seconds", 30.0),
            use_horizon_ilp_constraints=config.get("use_horizon_ilp_constraints", True),
            horizon_constraint_weight=config.get("horizon_constraint_weight", 2.0),
            horizon_positive_threshold=config.get("horizon_positive_threshold", 0.8),
            horizon_negative_threshold=config.get("horizon_negative_threshold", 0.2),
            horizon_pair_budget=config.get("horizon_pair_budget", 24),
            horizon_paths_per_pair=config.get("horizon_paths_per_pair", 8),
            horizon_path_expansion_budget=config.get(
                "horizon_path_expansion_budget",
                4096,
            ),
            horizon_max_iterations=config.get("horizon_max_iterations", 1),
            solver_threads=config.get("solver_threads", None),
        )


def _assemble_edge_labels_from_matrix(adj_mtx: np.ndarray, edge_label_matrix: np.ndarray) -> np.ndarray:
    return _assemble_edge_labels_from_matrix_impl(adj_mtx, edge_label_matrix)


def decode_generated_nodes(
    owner,
    generated_nodes: GeneratedNodeBatch,
    graph_conditioning: Optional[GraphConditioningBatch] = None,
    feasibility_oracle_candidates_per_attempt: Optional[int] = None,
    attempt_idx: int = 0,
    use_ilp_decoder: bool = True,
    edge_probability_threshold: Optional[float] = None,
) -> List[nx.Graph]:
    """Dispatch generated-node decoding through either the oracle path or the plain decoder."""
    if use_ilp_decoder and owner._can_use_feasibility_oracle(
        feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
        attempt_idx=attempt_idx,
    ):
        oracle_decode = globals().get("decode_generated_nodes_with_oracle")
        if oracle_decode is None:
            # Be tolerant of partial reload/autoreload states in notebooks.
            from .conditional_node_field_graph_decoder import (
                decode_generated_nodes_with_oracle as oracle_decode,
            )
        return oracle_decode(
            owner,
            generated_nodes,
            graph_conditioning=graph_conditioning,
        )
    predicted_edge_probability_matrices = generated_nodes.edge_probability_matrices
    if predicted_edge_probability_matrices is None:
        raise RuntimeError(
            "Graph decoding requires explicit edge-probability matrices from the conditional node generator."
        )
    predicted_node_labels_list = resolve_predicted_node_labels(owner, generated_nodes)
    predicted_edge_labels_list, predicted_edge_label_matrices = resolve_predicted_edge_labels(
        owner,
        generated_nodes,
        predicted_edge_probability_matrices=predicted_edge_probability_matrices,
    )
    return owner.graph_decoder.decode(
        generated_nodes,
        predicted_node_labels_list=predicted_node_labels_list,
        predicted_edge_probability_matrices=predicted_edge_probability_matrices,
        predicted_edge_labels_list=predicted_edge_labels_list,
        predicted_edge_label_matrices=predicted_edge_label_matrices,
        desired_node_counts=(
            None if graph_conditioning is None else np.asarray(graph_conditioning.node_counts, dtype=int)
        ),
        desired_edge_counts=(
            None if graph_conditioning is None else np.asarray(graph_conditioning.edge_counts, dtype=int)
        ),
        use_ilp_decoder=use_ilp_decoder,
        edge_probability_threshold=edge_probability_threshold,
    )


def decode_generated_nodes_with_oracle(
    owner,
    generated_nodes: GeneratedNodeBatch,
    graph_conditioning: Optional[GraphConditioningBatch] = None,
) -> List[Optional[nx.Graph]]:
    from .oracle_decode import decode_generated_nodes_with_oracle as decode_impl

    if owner.graph_decoder is None:
        owner.graph_decoder = ConditionalNodeFieldGraphDecoder(
            verbose=bool(owner.verbose)
        )
    return decode_impl(
        owner,
        generated_nodes,
        graph_conditioning=graph_conditioning,
    )
