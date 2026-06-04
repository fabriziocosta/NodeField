"""Decoder helpers for rebuilding labeled graphs from node-field predictions."""

import json
import random
import sys
from pathlib import Path
from typing import Any, Callable, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import dill as pickle
import networkx as nx
import numpy as np
import pulp

from .conditional_node_field_generator import GeneratedNodeBatch, GraphConditioningBatch
from . import diagnostics as _shared_diagnostics
from .graph_decode_utils import _canonicalize_edge, _normalize_violating_edge_sets
from .parallel_utils import _normalize_n_jobs, _parallel_map
from .runtime_utils import verbose_log

Edge = Tuple[int, int]
_DECODER_PROBABILITY_EPS = 1e-6
_DECODER_ARTIFACT_VERSION = 1
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
    n_nodes = min(len(existence_mask), len(degree_prediction))
    masked_prob_matrix = np.asarray(prob_matrix, dtype=float)[:n_nodes, :n_nodes].copy()
    existent = np.asarray(existence_mask[:n_nodes], dtype=bool)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j or not (existent[i] and existent[j]):
                masked_prob_matrix[i, j] = 0.0
    return (masked_prob_matrix + masked_prob_matrix.T) / 2.0


def _edge_label_matrix_to_list(adj_mtx: np.ndarray, edge_label_matrix: np.ndarray) -> np.ndarray:
    edge_labels = []
    for i in range(adj_mtx.shape[0]):
        for j in range(i + 1, adj_mtx.shape[1]):
            if adj_mtx[i, j] != 0:
                edge_labels.append(edge_label_matrix[i, j])
    return np.asarray(edge_labels, dtype=object)


def _edge_label_list_to_matrix(
    adj_mtx: np.ndarray,
    edge_labels: Sequence[Any],
) -> np.ndarray:
    edge_label_matrix = np.full(adj_mtx.shape, None, dtype=object)
    edge_idx = 0
    for i in range(adj_mtx.shape[0]):
        for j in range(i + 1, adj_mtx.shape[1]):
            if adj_mtx[i, j] != 0:
                edge_label = edge_labels[edge_idx] if edge_idx < len(edge_labels) else None
                edge_label_matrix[i, j] = edge_label
                edge_label_matrix[j, i] = edge_label
                edge_idx += 1
    return edge_label_matrix


def _assemble_graph_job(
    node_presence_mask: np.ndarray,
    node_labels: np.ndarray,
    edge_labels: np.ndarray,
    adj_mtx: np.ndarray,
) -> nx.Graph:
    graph = nx.from_numpy_array(adj_mtx)

    if len(node_labels) > 0 and not all(label is None for label in node_labels):
        node_label_map = {i: label for i, label in enumerate(node_labels)}
        nx.set_node_attributes(graph, node_label_map, "label")

    if np.sum(adj_mtx) > 0 and len(edge_labels) > 0:
        n_nodes = graph.number_of_nodes()
        edge_idx = 0
        edge_attr = {}
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if adj_mtx[i, j] != 0:
                    edge_attr[(i, j)] = edge_labels[edge_idx]
                    edge_idx += 1
        nx.set_edge_attributes(graph, edge_attr, "label")

    existent_indices = np.where(np.asarray(node_presence_mask[: adj_mtx.shape[0]], dtype=bool))[0]
    return graph.subgraph(existent_indices).copy()


def _assemble_graph_job_star(args) -> nx.Graph:
    return _assemble_graph_job(*args)


def _validate_node_label_array(
    node_labels: np.ndarray,
    *,
    graph_idx: int,
    n_slots: int,
) -> np.ndarray:
    node_labels = np.asarray(node_labels, dtype=object)
    if node_labels.ndim != 1:
        raise ValueError(
            "Each predicted node-label array must be one-dimensional; "
            f"graph {graph_idx} received shape {node_labels.shape}."
        )
    if node_labels.shape[0] != n_slots:
        raise ValueError(
            "Each predicted node-label array must align with the decoder node slots; "
            f"graph {graph_idx} received {node_labels.shape[0]} labels for {n_slots} slots."
        )
    return node_labels


def _validate_edge_label_array(
    edge_labels: np.ndarray,
    *,
    graph_idx: int,
    expected_edge_count: int,
) -> np.ndarray:
    edge_labels = np.asarray(edge_labels, dtype=object)
    if edge_labels.ndim != 1:
        raise ValueError(
            "Each predicted edge-label array must be one-dimensional; "
            f"graph {graph_idx} received shape {edge_labels.shape}."
        )
    if edge_labels.shape[0] != expected_edge_count:
        raise ValueError(
            "Each predicted edge-label array must align with the decoded adjacency edge count; "
            f"graph {graph_idx} received {edge_labels.shape[0]} labels for {expected_edge_count} edges."
        )
    return edge_labels


def _decode_single_adjacency_job(
    prob_list: np.ndarray,
    existence_mask: np.ndarray,
    existence_scores: Optional[np.ndarray],
    degree_prediction: np.ndarray,
    desired_node_count: Optional[int],
    desired_edge_count: Optional[int],
    degree_slack_penalty: float,
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
    horizon_max_iterations: int = 1,
) -> np.ndarray:
    decoder = ConditionalNodeFieldGraphDecoder(
        verbose=bool(verbose),
        degree_slack_penalty=degree_slack_penalty,
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
        horizon_max_iterations=horizon_max_iterations,
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
        horizon_probability_matrix=horizon_probability_matrix,
        horizon=horizon,
        horizon_node_mask=existent,
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
    return adj_mtx


def _decode_single_adjacency_job_star(args) -> np.ndarray:
    return _decode_single_adjacency_job(*args)


def build_single_generated_node_batch(
    generated_nodes: GeneratedNodeBatch,
    graph_idx: int,
) -> GeneratedNodeBatch:
    """Slice a multi-graph generated batch down to a single-example batch."""
    node_embeddings_list = None if generated_nodes.node_embeddings_list is None else [
        np.asarray(generated_nodes.node_embeddings_list[graph_idx])
    ]
    node_presence_mask = None if generated_nodes.node_presence_mask is None else np.asarray(
        [generated_nodes.node_presence_mask[graph_idx]]
    )
    node_existence_probabilities = None if generated_nodes.node_existence_probabilities is None else np.asarray(
        [generated_nodes.node_existence_probabilities[graph_idx]],
        dtype=float,
    )
    node_degree_predictions = None if generated_nodes.node_degree_predictions is None else np.asarray(
        [generated_nodes.node_degree_predictions[graph_idx]]
    )
    node_labels = None if generated_nodes.node_labels is None else [
        np.asarray(generated_nodes.node_labels[graph_idx], dtype=object)
    ]
    edge_probability_matrices = None if generated_nodes.edge_probability_matrices is None else [
        np.asarray(generated_nodes.edge_probability_matrices[graph_idx], dtype=float)
    ]
    edge_label_matrices = None if generated_nodes.edge_label_matrices is None else [
        np.asarray(generated_nodes.edge_label_matrices[graph_idx], dtype=object)
    ]
    node_label_logits = None if generated_nodes.node_label_logits is None else [
        np.asarray(generated_nodes.node_label_logits[graph_idx], dtype=float)
    ]
    node_label_probabilities = None if generated_nodes.node_label_probabilities is None else [
        np.asarray(generated_nodes.node_label_probabilities[graph_idx], dtype=float)
    ]
    edge_existence_probabilities = None if generated_nodes.edge_existence_probabilities is None else [
        np.asarray(generated_nodes.edge_existence_probabilities[graph_idx], dtype=float)
    ]
    edge_label_logits = None if generated_nodes.edge_label_logits is None else [
        np.asarray(generated_nodes.edge_label_logits[graph_idx], dtype=float)
    ]
    edge_label_probabilities = None if generated_nodes.edge_label_probabilities is None else [
        np.asarray(generated_nodes.edge_label_probabilities[graph_idx], dtype=float)
    ]
    horizon_probability_matrices = None if generated_nodes.horizon_probability_matrices is None else [
        np.asarray(generated_nodes.horizon_probability_matrices[graph_idx], dtype=float)
    ]
    return GeneratedNodeBatch(
        node_embeddings_list=node_embeddings_list,
        node_presence_mask=node_presence_mask,
        node_degree_predictions=node_degree_predictions,
        node_labels=node_labels,
        node_existence_probabilities=node_existence_probabilities,
        edge_probability_matrices=edge_probability_matrices,
        edge_label_matrices=edge_label_matrices,
        node_label_logits=node_label_logits,
        node_label_probabilities=node_label_probabilities,
        edge_existence_probabilities=edge_existence_probabilities,
        edge_label_logits=edge_label_logits,
        edge_label_probabilities=edge_label_probabilities,
        horizon_probability_matrices=horizon_probability_matrices,
        horizon=generated_nodes.horizon,
    )


def resolve_predicted_node_labels(
    owner,
    generated_nodes: GeneratedNodeBatch,
) -> List[np.ndarray]:
    """Resolve node labels from explicit predictions or the configured supervision policy."""
    node_label_plan = owner._plan_channel("node_labels")
    if generated_nodes.node_labels is not None:
        return [np.asarray(node_labels, dtype=object) for node_labels in generated_nodes.node_labels]
    if generated_nodes.node_presence_mask is None:
        raise RuntimeError("Node-label resolution requires node_presence_mask predictions.")
    if node_label_plan is None:
        return [
            np.asarray([None] * len(node_presence_mask), dtype=object)
            for node_presence_mask in generated_nodes.node_presence_mask
        ]
    if node_label_plan.mode == "constant":
        return [
            np.asarray([node_label_plan.constant_value] * len(node_presence_mask), dtype=object)
            for node_presence_mask in generated_nodes.node_presence_mask
        ]
    if node_label_plan.mode == "disabled":
        return [
            np.asarray([None] * len(node_presence_mask), dtype=object)
            for node_presence_mask in generated_nodes.node_presence_mask
        ]
    raise RuntimeError("Node-label channel is configured as learned, but the generator returned no node labels.")


def resolve_predicted_edge_labels(
    owner,
    generated_nodes: GeneratedNodeBatch,
    predicted_edge_probability_matrices: Optional[List[np.ndarray]],
) -> Tuple[Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
    """Resolve edge labels from explicit predictions or the configured supervision policy."""
    edge_label_plan = owner._plan_channel("edge_labels")
    if generated_nodes.edge_label_matrices is not None:
        return None, [np.asarray(edge_label_matrix, dtype=object) for edge_label_matrix in generated_nodes.edge_label_matrices]
    if predicted_edge_probability_matrices is None:
        raise RuntimeError("Edge-label resolution requires edge probabilities to determine decoded edge counts.")
    if edge_label_plan is None:
        return [np.asarray([], dtype=object) for _ in predicted_edge_probability_matrices], None
    if edge_label_plan.mode == "constant":
        predicted_edge_label_matrices = []
        for prob_matrix in predicted_edge_probability_matrices:
            prob_matrix = np.asarray(prob_matrix)
            if prob_matrix.ndim != 2 or prob_matrix.shape[0] != prob_matrix.shape[1]:
                raise ValueError(
                    "Constant edge-label resolution expects square edge-probability matrices "
                    f"(got shape={prob_matrix.shape})."
                )
            edge_label_matrix = np.full(prob_matrix.shape, edge_label_plan.constant_value, dtype=object)
            np.fill_diagonal(edge_label_matrix, None)
            predicted_edge_label_matrices.append(edge_label_matrix)
        return None, predicted_edge_label_matrices
    if edge_label_plan.mode == "disabled":
        return [np.asarray([], dtype=object) for _ in predicted_edge_probability_matrices], None
    raise RuntimeError("Edge-label channel is configured as learned, but the generator returned no edge labels.")


def sample_oracle_cuts_for_iteration(
    owner,
    accumulated_cuts: Sequence[FrozenSet[Edge]],
    solve_iteration_idx: int,
) -> List[FrozenSet[Edge]]:
    """Subsample accumulated structural cuts so later retries relax more aggressively."""
    if not accumulated_cuts:
        return []
    if owner.max_oracle_iterations <= 1:
        return []
    if solve_iteration_idx >= owner.max_oracle_iterations - 1:
        return []
    keep_fraction = 1.0 - (float(solve_iteration_idx) / float(owner.max_oracle_iterations - 1))
    keep_count = int(np.ceil(len(accumulated_cuts) * keep_fraction))
    keep_count = max(0, min(len(accumulated_cuts), keep_count))
    if keep_count <= 0:
        return []
    if keep_count >= len(accumulated_cuts):
        return list(accumulated_cuts)
    selected_indices = sorted(random.sample(range(len(accumulated_cuts)), keep_count))
    return [accumulated_cuts[idx] for idx in selected_indices]


def solve_oracle_relaxed_adjacency(
    owner,
    *,
    masked_prob_matrix: np.ndarray,
    target_degrees: List[int],
    accumulated_cuts: Sequence[FrozenSet[Edge]],
    start_iteration_idx: int,
    edge_violation_prior: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Retry adjacency optimization while progressively relaxing accumulated oracle cuts."""
    from .oracle_utils import apply_oracle_edge_memory_penalty

    solve_prob_matrix = np.asarray(masked_prob_matrix, dtype=float)
    if edge_violation_prior is not None and owner.oracle_edge_memory_penalty > 0.0:
        solve_prob_matrix = apply_oracle_edge_memory_penalty(
            solve_prob_matrix,
            edge_violation_prior,
            owner.oracle_edge_memory_penalty,
        )
    last_error: Optional[Exception] = None
    for solve_iteration_idx in range(start_iteration_idx, owner.max_oracle_iterations):
        active_cuts = sample_oracle_cuts_for_iteration(owner, accumulated_cuts, solve_iteration_idx)
        try:
            return owner.graph_decoder.optimize_adjacency_matrix(
                solve_prob_matrix,
                target_degrees,
                forbidden_edge_sets=active_cuts,
            )
        except Exception as exc:
            last_error = exc
            if int(owner.verbose) >= 1:
                verbose_log(
                    owner,
                    "Oracle-guided adjacency solve failed with "
                    f"{len(active_cuts)} active cuts at iteration {solve_iteration_idx + 1}/"
                    f"{owner.max_oracle_iterations}; retrying with fewer cuts.",
                )
    if last_error is not None:
        raise RuntimeError("Oracle-guided adjacency solve failed even after relaxing all oracle cuts.") from last_error
    raise RuntimeError("Oracle-guided adjacency solve could not be attempted.")


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
        existence_threshold: float = 0.5,
        enforce_connectivity: bool = True,
        degree_slack_penalty: float = 1e6,
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
        horizon_max_iterations: int = 1,
    ) -> None:
        self.verbose = verbose
        self.existence_threshold = existence_threshold
        self.enforce_connectivity = enforce_connectivity
        self.degree_slack_penalty = degree_slack_penalty
        self.warm_start_mst = warm_start_mst
        self.n_jobs = _normalize_n_jobs(n_jobs)
        self.diagnostic_graph_renderer = diagnostic_graph_renderer
        self.adjacency_time_limit_seconds = (
            None if adjacency_time_limit_seconds is None else float(adjacency_time_limit_seconds)
        )
        self.parallel_decode_timeout_seconds = (
            None if parallel_decode_timeout_seconds is None else float(parallel_decode_timeout_seconds)
        )
        self.use_horizon_ilp_constraints = bool(use_horizon_ilp_constraints)
        self.horizon_constraint_weight = float(horizon_constraint_weight)
        self.horizon_positive_threshold = float(horizon_positive_threshold)
        self.horizon_negative_threshold = float(horizon_negative_threshold)
        self.horizon_pair_budget = int(horizon_pair_budget)
        self.horizon_paths_per_pair = int(horizon_paths_per_pair)
        self.horizon_max_iterations = int(horizon_max_iterations)
        self.active_time_limit_seconds: Optional[float] = None

    @staticmethod
    def _edge_key(i: int, j: int) -> Edge:
        return (int(i), int(j)) if int(i) < int(j) else (int(j), int(i))

    @classmethod
    def _path_edges(cls, path: Sequence[int]) -> List[Edge]:
        return [cls._edge_key(path[idx], path[idx + 1]) for idx in range(len(path) - 1)]

    @staticmethod
    def _edge_logit(probability: float) -> float:
        edge_prob = float(np.clip(probability, _DECODER_PROBABILITY_EPS, 1.0 - _DECODER_PROBABILITY_EPS))
        return float(np.log(edge_prob) - np.log1p(-edge_prob))

    def _select_horizon_pairs(
        self,
        horizon_probability_matrix: np.ndarray,
        *,
        active_mask: np.ndarray,
    ) -> Tuple[List[Tuple[int, int, float, float]], List[Tuple[int, int, float, float]]]:
        horizon_probs = np.asarray(horizon_probability_matrix, dtype=float)
        active_indices = np.flatnonzero(np.asarray(active_mask, dtype=bool))
        positive_pairs = []
        negative_pairs = []
        for local_i, i in enumerate(active_indices):
            for j in active_indices[local_i + 1:]:
                q_ij = float(np.clip((horizon_probs[i, j] + horizon_probs[j, i]) / 2.0, 0.0, 1.0))
                confidence = abs(q_ij - 0.5) * 2.0
                if q_ij >= self.horizon_positive_threshold:
                    positive_pairs.append((int(i), int(j), q_ij, confidence))
                elif q_ij <= self.horizon_negative_threshold:
                    negative_pairs.append((int(i), int(j), q_ij, confidence))

        positive_pairs.sort(key=lambda item: (-item[3], item[0], item[1]))
        negative_pairs.sort(key=lambda item: (-item[3], item[0], item[1]))
        budget = max(0, int(self.horizon_pair_budget))
        if budget == 0:
            return [], []
        if positive_pairs and negative_pairs:
            positive_budget = max(1, budget // 2)
            negative_budget = max(0, budget - positive_budget)
        elif positive_pairs:
            positive_budget = budget
            negative_budget = 0
        else:
            positive_budget = 0
            negative_budget = budget
        return positive_pairs[:positive_budget], negative_pairs[:negative_budget]

    def _enumerate_horizon_paths(
        self,
        source: int,
        target: int,
        *,
        horizon: int,
        active_mask: np.ndarray,
        edge_logit_matrix: np.ndarray,
    ) -> List[Tuple[List[int], float]]:
        max_paths = max(1, int(self.horizon_paths_per_pair))
        max_candidates = max(max_paths * 16, max_paths)
        active_nodes = set(int(idx) for idx in np.flatnonzero(np.asarray(active_mask, dtype=bool)))
        if source not in active_nodes or target not in active_nodes:
            return []
        ordered_neighbors = {
            node: sorted(
                (candidate for candidate in active_nodes if candidate != node),
                key=lambda candidate: (-float(edge_logit_matrix[node, candidate]), candidate),
            )
            for node in active_nodes
        }
        candidates: List[Tuple[List[int], float]] = []

        def _walk(path: List[int], visited: set[int]) -> None:
            if len(candidates) >= max_candidates:
                return
            current = path[-1]
            if current == target:
                score = sum(
                    float(edge_logit_matrix[path[idx], path[idx + 1]])
                    for idx in range(len(path) - 1)
                )
                candidates.append((list(path), score))
                return
            if len(path) - 1 >= int(horizon):
                return
            for neighbor in ordered_neighbors[current]:
                if neighbor in visited:
                    continue
                _walk(path + [neighbor], visited | {neighbor})
                if len(candidates) >= max_candidates:
                    return

        _walk([int(source)], {int(source)})
        candidates.sort(key=lambda item: (-item[1], len(item[0]), item[0]))
        return candidates[:max_paths]

    def _find_negative_horizon_cuts(
        self,
        adj: np.ndarray,
        negative_pairs: Sequence[Tuple[int, int, float, float]],
        *,
        horizon: int,
    ) -> List[Tuple[List[Edge], float]]:
        graph = nx.from_numpy_array(np.asarray(adj, dtype=int))
        cuts = []
        seen = set()
        for i, j, _probability, confidence in negative_pairs:
            try:
                path = nx.shortest_path(graph, int(i), int(j))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            if len(path) - 1 > int(horizon):
                continue
            edge_set = self._path_edges(path)
            frozen = frozenset(edge_set)
            if not frozen or frozen in seen:
                continue
            seen.add(frozen)
            cuts.append((edge_set, float(confidence)))
        return cuts

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
        horizon_probability_matrix: Optional[np.ndarray] = None,
        horizon: Optional[int] = None,
        horizon_node_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        n = prob_matrix.shape[0]
        if alpha != 1.0:
            prob_matrix = np.power(prob_matrix, alpha)
        if connectivity is None:
            connectivity = self.enforce_connectivity

        edge_logit_matrix = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                edge_logit = self._edge_logit(float(prob_matrix[i, j]))
                edge_logit_matrix[i, j] = edge_logit_matrix[j, i] = edge_logit

        normalized_forbidden_edge_sets = _normalize_violating_edge_sets(
            [] if forbidden_edge_sets is None else forbidden_edge_sets,
            n_nodes=n,
        )

        horizon_enabled = (
            bool(self.use_horizon_ilp_constraints)
            and horizon_probability_matrix is not None
            and horizon is not None
            and int(horizon) > 1
            and float(self.horizon_constraint_weight) > 0.0
        )
        positive_pairs: List[Tuple[int, int, float, float]] = []
        negative_pairs: List[Tuple[int, int, float, float]] = []
        if horizon_enabled:
            horizon_probs = np.asarray(horizon_probability_matrix, dtype=float)
            if horizon_probs.shape != (n, n):
                raise ValueError(
                    "horizon_probability_matrix must align with prob_matrix; "
                    f"received {horizon_probs.shape} for n={n}."
                )
            active_mask = (
                np.ones(n, dtype=bool)
                if horizon_node_mask is None
                else np.asarray(horizon_node_mask, dtype=bool)[:n]
            )
            if active_mask.shape != (n,):
                raise ValueError("horizon_node_mask must align with prob_matrix.")
            positive_pairs, negative_pairs = self._select_horizon_pairs(
                horizon_probs,
                active_mask=active_mask,
            )
        else:
            horizon_probs = None
            active_mask = np.ones(n, dtype=bool)

        effective_time_limit = timeLimit
        if effective_time_limit is None:
            effective_time_limit = self.active_time_limit_seconds
        if effective_time_limit is None:
            effective_time_limit = self.adjacency_time_limit_seconds

        def _solve(negative_cuts: Sequence[Tuple[List[Edge], float]], *, solve_name: str) -> np.ndarray:
            prob = pulp.LpProblem(solve_name, pulp.LpMaximize)
            x = {
                (i, j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary")
                for i in range(n)
                for j in range(i + 1, n)
            }
            u = {i: pulp.LpVariable(f"u_{i}", lowBound=0, cat="Integer") for i in range(n)}
            v = {i: pulp.LpVariable(f"v_{i}", lowBound=0, cat="Integer") for i in range(n)}

            objective_terms = [
                edge_logit_matrix[i, j] * x[(i, j)]
                for i in range(n)
                for j in range(i + 1, n)
            ]
            objective_terms.extend(
                -self.degree_slack_penalty * (u[i] + v[i])
                for i in range(n)
            )

            for i in range(n):
                incident = [x[(i, j)] for j in range(i + 1, n)] + [
                    x[(j, i)] for j in range(i) if (j, i) in x
                ]
                prob += (pulp.lpSum(incident) + u[i] - v[i] == target_degrees[i]), f"Degree_{i}"
            if target_edge_count is not None:
                resolved_edge_count = self._resolve_target_edge_count(
                    n,
                    target_edge_count,
                    connectivity=connectivity,
                )
                prob += (pulp.lpSum(var for var in x.values()) == resolved_edge_count), "EdgeCount"

            if connectivity:
                directed_edges = [(i, j) for (i, j) in x] + [(j, i) for (i, j) in x]
                f_vars = {
                    (u_, v_): pulp.LpVariable(f"f_{u_}_{v_}", lowBound=0, cat="Continuous")
                    for u_, v_ in directed_edges
                }
                M = n - 1
                root = 0
                for v_idx in range(n):
                    inflow = pulp.lpSum(f_vars[(u_, v2)] for (u_, v2) in directed_edges if v2 == v_idx)
                    outflow = pulp.lpSum(f_vars[(v2, w)] for (v2, w) in directed_edges if v2 == v_idx)
                    prob += ((outflow - inflow) == M if v_idx == root else (inflow - outflow) == 1), f"Flow_{v_idx}"
                for u_, v_ in directed_edges:
                    i, j = min(u_, v_), max(u_, v_)
                    prob += (f_vars[(u_, v_)] <= M * x[(i, j)]), f"FlowCouple_{u_}_{v_}"

            for cut_idx, edge_set in enumerate(normalized_forbidden_edge_sets):
                prob += (pulp.lpSum(x[edge] for edge in edge_set) <= len(edge_set) - 1), f"ForbiddenMotif_{cut_idx}"

            if positive_pairs:
                for pair_idx, (i, j, _probability, confidence) in enumerate(positive_pairs):
                    paths = self._enumerate_horizon_paths(
                        i,
                        j,
                        horizon=int(horizon),
                        active_mask=active_mask,
                        edge_logit_matrix=edge_logit_matrix,
                    )
                    if not paths:
                        continue
                    slack = pulp.LpVariable(f"hpos_slack_{pair_idx}_{i}_{j}", lowBound=0, upBound=1, cat="Continuous")
                    objective_terms.append(
                        -float(self.horizon_constraint_weight) * float(confidence) * slack
                    )
                    path_vars = []
                    for path_idx, (path, _path_score) in enumerate(paths):
                        path_edges = self._path_edges(path)
                        z_path = pulp.LpVariable(f"hpos_path_{pair_idx}_{path_idx}_{i}_{j}", cat="Binary")
                        path_vars.append(z_path)
                        for edge in path_edges:
                            prob += z_path <= x[edge], f"HPosPathUpper_{pair_idx}_{path_idx}_{edge[0]}_{edge[1]}"
                        prob += (
                            z_path >= pulp.lpSum(x[edge] for edge in path_edges) - len(path_edges) + 1
                        ), f"HPosPathLower_{pair_idx}_{path_idx}"
                    prob += (pulp.lpSum(path_vars) + slack >= 1), f"HPosPair_{pair_idx}_{i}_{j}"

            for cut_idx, (edge_set, confidence) in enumerate(negative_cuts):
                slack = pulp.LpVariable(f"hneg_slack_{cut_idx}", lowBound=0, upBound=1, cat="Continuous")
                objective_terms.append(
                    -float(self.horizon_constraint_weight) * float(confidence) * slack
                )
                prob += (
                    pulp.lpSum(x[edge] for edge in edge_set) <= len(edge_set) - 1 + slack
                ), f"HNegPathCut_{cut_idx}"

            prob += pulp.lpSum(objective_terms)

            if self.warm_start_mst:
                graph = nx.Graph()
                graph.add_nodes_from(range(n))
                for i in range(n):
                    for j in range(i + 1, n):
                        graph.add_edge(i, j, weight=prob_matrix[i, j])
                tree = nx.maximum_spanning_tree(graph)
                for (i, j), var in x.items():
                    var.start = 1 if tree.has_edge(i, j) else 0

            solver_kwargs = {"msg": verbose}
            if effective_time_limit is not None:
                solver_kwargs["timeLimit"] = max(1.0, float(effective_time_limit))
            solver = pulp.PULP_CBC_CMD(**solver_kwargs)
            prob.solve(solver)
            status_code = int(getattr(prob, "status", 0))
            status_label = pulp.LpStatus.get(status_code, f"Unknown({status_code})")
            if status_code != pulp.LpStatusOptimal:
                raise RuntimeError(
                    "Adjacency ILP did not produce an optimal solution "
                    f"(status={status_label}, code={status_code}, n={n}, "
                    f"target_degree_sum={int(sum(target_degrees))}, connectivity={bool(connectivity)})."
                )

            adj = np.zeros((n, n), dtype=int)
            for (i, j), var in x.items():
                value = pulp.value(var)
                if value is None:
                    raise RuntimeError(
                        "Adjacency ILP finished without assigning all decision variables "
                        f"(status={status_label}, missing_edge=({i}, {j}))."
                    )
                adj[i, j] = adj[j, i] = int(round(float(value)))
            return adj

        adj = _solve([], solve_name="AdjacencyMatrixOptimization")
        if negative_pairs and int(self.horizon_max_iterations) > 0:
            negative_cuts = self._find_negative_horizon_cuts(
                adj,
                negative_pairs,
                horizon=int(horizon),
            )
            if negative_cuts:
                adj = _solve(negative_cuts, solve_name="AdjacencyMatrixOptimizationHorizonRepair")
        return adj

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
            prob_matrix = np.asarray(prob_matrix, dtype=float)
            if prob_matrix.ndim == 2:
                if prob_matrix.shape[0] != n_nodes or prob_matrix.shape[1] != n_nodes:
                    raise ValueError(
                        "Edge-probability matrices must align with node predictions; "
                        f"received {prob_matrix.shape} for n_nodes={n_nodes}."
                    )
                mask = ~np.eye(n_nodes, dtype=bool)
                predicted_probs_list.append(prob_matrix[mask])
            else:
                predicted_probs_list.append(prob_matrix)
        if horizon_probability_matrices is None:
            horizon_probs_list = [None for _ in predicted_probs_list]
        else:
            for graph_idx, (existence_mask, degree_prediction, horizon_matrix) in enumerate(
                zip(existence_masks, degree_predictions, horizon_probability_matrices)
            ):
                n_nodes = min(len(existence_mask), len(degree_prediction))
                horizon_matrix = np.asarray(horizon_matrix, dtype=float)
                if horizon_matrix.shape != (n_nodes, n_nodes):
                    raise ValueError(
                        "Horizon-probability matrices must align with node predictions; "
                        f"graph {graph_idx} received {horizon_matrix.shape} for n_nodes={n_nodes}."
                    )
                horizon_probs_list.append(horizon_matrix)

        jobs = [
            (
                np.asarray(predicted_probs_list[graph_idx], dtype=float),
                np.asarray(existence_masks[graph_idx], dtype=bool),
                None if existence_scores is None else np.asarray(existence_scores[graph_idx], dtype=float),
                np.asarray(degree_predictions[graph_idx], dtype=float),
                None if desired_node_counts is None else int(desired_node_counts[graph_idx]),
                None if desired_edge_counts is None else int(desired_edge_counts[graph_idx]),
                float(self.degree_slack_penalty),
                bool(self.enforce_connectivity),
                bool(self.warm_start_mst),
                int(self.verbose),
                self.diagnostic_graph_renderer if self.n_jobs == 1 else None,
                (
                    float(self.active_time_limit_seconds)
                    if self.active_time_limit_seconds is not None
                    else (
                        None
                        if self.adjacency_time_limit_seconds is None
                        else float(self.adjacency_time_limit_seconds)
                    )
                ),
                None if horizon_probs_list[graph_idx] is None else np.asarray(horizon_probs_list[graph_idx], dtype=float),
                None if horizon is None else int(horizon),
                bool(self.use_horizon_ilp_constraints),
                float(self.horizon_constraint_weight),
                float(self.horizon_positive_threshold),
                float(self.horizon_negative_threshold),
                int(self.horizon_pair_budget),
                int(self.horizon_paths_per_pair),
                int(self.horizon_max_iterations),
            )
            for graph_idx in range(len(predicted_probs_list))
        ]
        if int(self.verbose) >= 4 and self.n_jobs != 1 and len(jobs) > 1:
            verbose_log(
                self,
                "Decoder plots for verbose>=4 are only shown when n_jobs=1; skipping plots during parallel adjacency decode.",
                level=4,
            )
        if self.n_jobs == 1 or len(jobs) <= 1:
            return [_decode_single_adjacency_job(*job) for job in jobs]
        timeout_seconds = self.parallel_decode_timeout_seconds
        if self.active_time_limit_seconds is not None:
            timeout_seconds = max(
                float(timeout_seconds) if timeout_seconds is not None else 0.0,
                float(self.active_time_limit_seconds) + 5.0,
            )
        return _parallel_map(
            _decode_single_adjacency_job_star,
            jobs,
            self.n_jobs,
            verbose=bool(self.verbose),
            timeout_seconds=timeout_seconds,
            timeout_fallback_label="parallel adjacency decode",
        )

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
            float(self.existence_threshold)
            if edge_probability_threshold is None
            else float(edge_probability_threshold)
        )
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
        edge_candidates = []
        for local_i, i in enumerate(active_indices):
            for j in active_indices[local_i + 1:]:
                probability = float((prob_matrix[i, j] + prob_matrix[j, i]) / 2.0)
                edge_candidates.append((probability, int(i), int(j)))
        return edge_candidates

    @staticmethod
    def _select_direct_edges_top_k(edge_candidates, desired_edge_count: int):
        target_edge_count = max(0, int(desired_edge_count))
        target_edge_count = min(target_edge_count, len(edge_candidates))
        return sorted(edge_candidates, key=lambda item: (-item[0], item[1], item[2]))[:target_edge_count]

    @staticmethod
    def _select_direct_edges_degree_aware(
        edge_candidates,
        active_indices: np.ndarray,
        target_degrees: Sequence[int],
        desired_edge_count: int,
    ):
        target_edge_count = max(0, int(desired_edge_count))
        target_edge_count = min(target_edge_count, len(edge_candidates))
        if target_edge_count == 0:
            return []

        target_degrees = np.asarray(target_degrees, dtype=int)
        edge_by_key = {}
        candidates_by_node = {int(node): [] for node in active_indices}
        for probability, i, j in edge_candidates:
            i = int(i)
            j = int(j)
            edge = (float(probability), min(i, j), max(i, j))
            edge_by_key[(edge[1], edge[2])] = edge
            if i in candidates_by_node:
                candidates_by_node[i].append(edge)
            if j in candidates_by_node:
                candidates_by_node[j].append(edge)

        selected_by_key = {}
        positive_target_nodes = {
            int(node)
            for node in active_indices
            if int(node) < target_degrees.shape[0] and int(target_degrees[int(node)]) > 0
        }
        for node in active_indices:
            node = int(node)
            if node >= target_degrees.shape[0]:
                continue
            quota = max(0, int(target_degrees[node]))
            if quota == 0:
                continue
            ranked_edges = sorted(
                (
                    edge
                    for edge in candidates_by_node.get(node, [])
                    if (edge[1] if edge[2] == node else edge[2]) in positive_target_nodes
                ),
                key=lambda item: (-item[0], item[1], item[2]),
            )
            for edge in ranked_edges[:quota]:
                selected_by_key[(edge[1], edge[2])] = edge

        selected_edges = list(selected_by_key.values())
        selected_degrees = np.zeros(target_degrees.shape[0], dtype=int)
        for _probability, i, j in selected_edges:
            if i < selected_degrees.shape[0]:
                selected_degrees[i] += 1
            if j < selected_degrees.shape[0]:
                selected_degrees[j] += 1

        while len(selected_edges) > target_edge_count:
            removable_idx = None
            for edge_idx in sorted(
                range(len(selected_edges)),
                key=lambda idx: (selected_edges[idx][0], -selected_edges[idx][1], -selected_edges[idx][2]),
            ):
                _probability, i, j = selected_edges[edge_idx]
                if (
                    i < selected_degrees.shape[0]
                    and j < selected_degrees.shape[0]
                    and selected_degrees[i] - 1 >= target_degrees[i]
                    and selected_degrees[j] - 1 >= target_degrees[j]
                ):
                    removable_idx = edge_idx
                    break
            if removable_idx is None:
                break
            _probability, i, j = selected_edges.pop(removable_idx)
            selected_degrees[i] -= 1
            selected_degrees[j] -= 1

        selected_edges = sorted(
            selected_edges,
            key=lambda item: (-item[0], item[1], item[2]),
        )
        if len(selected_edges) >= target_edge_count:
            return selected_edges

        selected_keys = {(i, j) for _, i, j in selected_edges}
        for edge in sorted(edge_by_key.values(), key=lambda item: (-item[0], item[1], item[2])):
            key = (edge[1], edge[2])
            if key in selected_keys:
                continue
            selected_edges.append(edge)
            selected_keys.add(key)
            if len(selected_edges) >= target_edge_count:
                break
        return selected_edges

    @staticmethod
    def _select_direct_edges_by_threshold(edge_candidates, threshold: float):
        threshold = float(threshold)
        return [
            (probability, i, j)
            for probability, i, j in edge_candidates
            if probability >= threshold
        ]

    @staticmethod
    def _adjacency_from_selected_edges(n_nodes: int, selected_edges) -> np.ndarray:
        adj_mtx = np.zeros((int(n_nodes), int(n_nodes)), dtype=float)
        for _, i, j in selected_edges:
            adj_mtx[i, j] = 1.0
            adj_mtx[j, i] = 1.0
        return adj_mtx

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
        prob_matrix = np.asarray(prob_matrix, dtype=float)
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
    ) -> List[nx.Graph]:
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
                "existence_threshold": self.existence_threshold,
                "enforce_connectivity": self.enforce_connectivity,
                "degree_slack_penalty": self.degree_slack_penalty,
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
                "horizon_max_iterations": self.horizon_max_iterations,
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
                return pickle.load(handle)
        if artifact.get("artifact_type") != "ConditionalNodeFieldGraphDecoder":
            raise RuntimeError(
                f"Unsupported decoder artifact type {artifact.get('artifact_type')!r} in {path}."
            )
        artifact_version = int(artifact.get("artifact_version", 0))
        if artifact_version != _DECODER_ARTIFACT_VERSION:
            raise RuntimeError(
                "Saved decoder artifact version is incompatible with this NodeField version. "
                f"Expected v{_DECODER_ARTIFACT_VERSION}, found v{artifact_version}: {path}"
            )
        config = artifact.get("config", {})
        return self.__class__(
            verbose=config.get("verbose", True),
            existence_threshold=config.get("existence_threshold", 0.5),
            enforce_connectivity=config.get("enforce_connectivity", True),
            degree_slack_penalty=config.get("degree_slack_penalty", 1e6),
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
            horizon_max_iterations=config.get("horizon_max_iterations", 1),
        )


def _assemble_edge_labels_from_matrix(adj_mtx: np.ndarray, edge_label_matrix: np.ndarray) -> np.ndarray:
    if edge_label_matrix.shape != adj_mtx.shape:
        raise ValueError(
            "Each predicted edge-label matrix must have the same shape as its adjacency matrix; "
            f"received {edge_label_matrix.shape} and {adj_mtx.shape}."
        )
    edge_labels = []
    n_nodes = adj_mtx.shape[0]
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if adj_mtx[i, j] != 0:
                edge_labels.append(edge_label_matrix[i, j])
    return np.asarray(edge_labels, dtype=object)


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
) -> List[nx.Graph]:
    """Decode one generated batch using the feasibility oracle, label repairs, and structural cuts."""
    from .oracle_utils import update_oracle_edge_memory

    if owner.graph_decoder is None:
        owner.graph_decoder = ConditionalNodeFieldGraphDecoder(verbose=bool(owner.verbose))
    predicted_edge_probability_matrices = generated_nodes.edge_probability_matrices
    if predicted_edge_probability_matrices is None:
        raise RuntimeError(
            "Graph decoding requires explicit edge-probability matrices from the conditional node generator."
        )
    edge_existence_probabilities = generated_nodes.edge_existence_probabilities
    node_label_probabilities = generated_nodes.node_label_probabilities
    edge_label_probabilities = generated_nodes.edge_label_probabilities
    predicted_node_labels_list = resolve_predicted_node_labels(owner, generated_nodes)
    predicted_edge_labels_list, predicted_edge_label_matrices = resolve_predicted_edge_labels(
        owner,
        generated_nodes,
        predicted_edge_probability_matrices=predicted_edge_probability_matrices,
    )

    decoded_graphs: List[nx.Graph] = []
    for graph_idx in range(len(predicted_edge_probability_matrices)):
        single_generated_nodes = build_single_generated_node_batch(generated_nodes, graph_idx)
        desired_node_count = None if graph_conditioning is None else int(np.asarray(graph_conditioning.node_counts)[graph_idx])
        desired_edge_count = None if graph_conditioning is None else int(np.asarray(graph_conditioning.edge_counts)[graph_idx])
        existence_mask = owner.graph_decoder.resolve_node_presence_mask(
            np.asarray(single_generated_nodes.node_presence_mask[0], dtype=bool),
            desired_node_count=desired_node_count,
            node_existence_scores=None if single_generated_nodes.node_existence_probabilities is None else np.asarray(
                single_generated_nodes.node_existence_probabilities[0],
                dtype=float,
            ),
        )
        degree_predictions = np.asarray(single_generated_nodes.node_degree_predictions[0], dtype=float)
        prob_matrix = np.asarray(predicted_edge_probability_matrices[graph_idx], dtype=float)
        masked_prob_matrix = _build_masked_prob_matrix(
            existence_mask=existence_mask,
            degree_prediction=degree_predictions,
            prob_matrix=prob_matrix,
        )
        target_degrees = owner.graph_decoder.get_degree_targets(
            degree_predictions,
            existence_mask,
            desired_edge_count=desired_edge_count,
        )
        current_node_labels = np.asarray(predicted_node_labels_list[graph_idx], dtype=object).copy()
        current_edge_label_matrix = (
            np.asarray(predicted_edge_label_matrices[graph_idx], dtype=object).copy()
            if predicted_edge_label_matrices is not None
            else _edge_label_list_to_matrix(
                np.asarray(predicted_edge_probability_matrices[graph_idx] > 0, dtype=int),
                np.asarray(predicted_edge_labels_list[graph_idx], dtype=object)
                if predicted_edge_labels_list is not None
                else np.asarray([], dtype=object),
            )
        )
        try:
            single_adj_mtx = owner.graph_decoder.decode_adjacency_matrix(
                single_generated_nodes,
                predicted_edge_probability_matrices=[predicted_edge_probability_matrices[graph_idx]],
                desired_node_counts=None if desired_node_count is None else [desired_node_count],
                desired_edge_counts=None if desired_edge_count is None else [desired_edge_count],
            )[0]
        except RuntimeError:
            if int(owner.verbose) >= 1:
                verbose_log(
                    owner,
                    "Oracle initial adjacency decode failed under connectivity constraints; "
                    "retrying with connectivity disabled for the seed solve.",
                )
            try:
                single_adj_mtx = owner.graph_decoder.optimize_adjacency_matrix(
                    masked_prob_matrix,
                    target_degrees,
                    target_edge_count=desired_edge_count,
                    connectivity=False,
                )
            except Exception as fallback_exc:
                raise RuntimeError(
                    "Oracle initial adjacency decode failed before any oracle cuts could be applied."
                ) from fallback_exc
        accumulated_structural_cuts: List[FrozenSet[Edge]] = []
        accumulated_node_label_forbidden = []
        accumulated_edge_label_forbidden = []
        local_edge_violation_prior = np.zeros_like(masked_prob_matrix, dtype=float)
        best_graph: Optional[nx.Graph] = None
        best_score = float("-inf")
        best_feasible_graph: Optional[nx.Graph] = None
        best_feasible_score = float("-inf")
        current_total_log_score = float("nan")
        current_edge_log_score = float("nan")
        current_node_log_score = float("nan")
        current_edge_label_log_score = float("nan")
        use_node_label_cuts = bool(getattr(owner, "oracle_use_node_label_cuts", False))
        use_edge_label_cuts = bool(getattr(owner, "oracle_use_edge_label_cuts", False))

        def plot_oracle_phase(
            phase_name: str,
            *,
            adj_mtx: np.ndarray,
            decoded_graph: nx.Graph,
            node_violation_sets: Sequence[Any],
            edge_violation_sets: Sequence[FrozenSet[Edge]],
            new_node_cut_count: int = 0,
            new_edge_label_cut_count: int = 0,
            joint_label_changed: bool = False,
            new_structural_cut_count: int = 0,
            detail: Optional[str] = None,
        ) -> None:
            if int(owner.verbose) < 4:
                return
            phase_title = f"Oracle {phase_name}"
            if detail:
                phase_title += f" [{detail}]"
            _plot_decoder_diagnostics(
                prob_matrix=masked_prob_matrix,
                adj_mtx=adj_mtx,
                target_degrees=target_degrees,
                title=(
                    f"{phase_title} graph={graph_idx} iteration={_iteration_idx + 1} "
                    f"| violating_node_sets={len(node_violation_sets)} "
                    f"| violating_edge_sets={len(edge_violation_sets)} "
                    f"| new_node_cuts={int(new_node_cut_count)} "
                    f"| new_edge_label_cuts={int(new_edge_label_cut_count)} "
                    f"| joint_label_changed={int(bool(joint_label_changed))} "
                    f"| log_total={current_total_log_score:.3f} "
                    f"| log_edge={current_edge_log_score:.3f} "
                    f"| log_node={current_node_log_score:.3f} "
                    f"| log_edge_label={current_edge_label_log_score:.3f} "
                    f"| best_log_total={best_score:.3f} "
                    f"| best_feasible_log_total={best_feasible_score:.3f} "
                    f"| new_structural_cuts={int(new_structural_cut_count)} "
                    f"| accepted_structural_cuts={len(accumulated_structural_cuts) + int(new_structural_cut_count)}"
                ),
                violating_edge_sets=edge_violation_sets,
                decoded_graph=decoded_graph,
                graph_renderer=owner.graph_decoder.diagnostic_graph_renderer,
                node_label_probabilities=None if node_label_probabilities is None else np.asarray(
                    node_label_probabilities[graph_idx],
                    dtype=float,
                ),
                node_label_names=owner._get_node_label_names(),
                node_labels=np.asarray(current_node_labels, dtype=object),
                existence_mask=existence_mask,
            )

        def evaluate_oracle_state(
            node_labels: np.ndarray,
            edge_label_matrix: np.ndarray,
        ) -> tuple[nx.Graph, List[Any], List[FrozenSet[Edge]], float, float, float, float]:
            candidate_graph = _assemble_graph_job(
                existence_mask,
                np.asarray(node_labels, dtype=object),
                np.asarray(_edge_label_matrix_to_list(single_adj_mtx, edge_label_matrix), dtype=object),
                np.asarray(single_adj_mtx, dtype=float),
            )
            node_violation_sets = (
                owner._get_oracle_node_violation_sets(candidate_graph, n_nodes=single_adj_mtx.shape[0])
                if use_node_label_cuts
                else []
            )
            edge_violation_sets = owner._get_oracle_edge_violation_sets(
                candidate_graph,
                n_nodes=single_adj_mtx.shape[0],
            )
            score, edge_score, node_score, edge_label_score = owner._oracle_candidate_score_components(
                existence_mask=existence_mask,
                adj_mtx=single_adj_mtx,
                node_labels=np.asarray(node_labels, dtype=object),
                edge_label_matrix=np.asarray(edge_label_matrix, dtype=object),
                edge_probability_matrix=np.asarray(
                    edge_existence_probabilities[graph_idx]
                    if edge_existence_probabilities is not None
                    else predicted_edge_probability_matrices[graph_idx],
                    dtype=float,
                ),
                node_label_probabilities=None if node_label_probabilities is None else np.asarray(
                    node_label_probabilities[graph_idx],
                    dtype=float,
                ),
                edge_label_probabilities=None if edge_label_probabilities is None else np.asarray(
                    edge_label_probabilities[graph_idx],
                    dtype=float,
                ),
            )
            return (
                candidate_graph,
                node_violation_sets,
                edge_violation_sets,
                float(score),
                float(edge_score),
                float(node_score),
                float(edge_label_score),
            )

        def oracle_rank(
            node_violation_sets: Sequence[Any],
            edge_violation_sets: Sequence[FrozenSet[Edge]],
            score: float,
        ) -> tuple[float, float, float, float]:
            node_count = len(node_violation_sets) if use_node_label_cuts else 0
            edge_count = len(edge_violation_sets)
            return (edge_count + node_count, edge_count, node_count, -float(score))

        for _iteration_idx in range(owner.max_oracle_iterations):
            edge_labels = owner.graph_decoder.decode_edge_labels(
                single_generated_nodes,
                [single_adj_mtx],
                predicted_edge_labels_list=None if predicted_edge_labels_list is None else [
                    predicted_edge_labels_list[graph_idx]
                ],
                predicted_edge_label_matrices=None if predicted_edge_label_matrices is None else [
                    current_edge_label_matrix
                ],
            )[0]
            current_edge_label_matrix = _edge_label_list_to_matrix(single_adj_mtx, edge_labels)
            (
                graph,
                current_node_violation_sets,
                current_edge_violation_sets,
                score,
                edge_score,
                node_score,
                edge_label_score,
            ) = evaluate_oracle_state(current_node_labels, current_edge_label_matrix)

            new_node_forbidden = []
            if use_node_label_cuts and node_label_probabilities is not None:
                new_node_forbidden = [
                    assignment
                    for assignment in owner._forbidden_node_label_assignment_from_sets(
                        current_node_violation_sets,
                        current_node_labels,
                    )
                    if assignment not in accumulated_node_label_forbidden
                ]
            new_edge_label_forbidden = []
            if use_edge_label_cuts and edge_label_probabilities is not None:
                new_edge_label_forbidden = [
                    assignment
                    for assignment in owner._forbidden_edge_label_assignment_from_sets(
                        current_edge_violation_sets,
                        current_edge_label_matrix,
                    )
                    if assignment not in accumulated_edge_label_forbidden
                ]
            if new_node_forbidden:
                accumulated_node_label_forbidden.extend(new_node_forbidden)
            if new_edge_label_forbidden:
                accumulated_edge_label_forbidden.extend(new_edge_label_forbidden)

            current_rank = oracle_rank(current_node_violation_sets, current_edge_violation_sets, score)
            joint_label_changed = False
            joint_label_detail = "label follow-up disabled"
            can_repair_joint_labels = (
                (
                    use_node_label_cuts
                    and accumulated_node_label_forbidden
                    and node_label_probabilities is not None
                )
                or (
                    use_edge_label_cuts
                    and accumulated_edge_label_forbidden
                    and edge_label_probabilities is not None
                )
            )
            if can_repair_joint_labels:
                repaired_node_labels, repaired_edge_label_matrix = owner._repair_labels_with_oracle(
                    existence_mask=existence_mask,
                    adj_mtx=single_adj_mtx,
                    current_node_labels=current_node_labels,
                    current_edge_label_matrix=current_edge_label_matrix,
                    node_label_probabilities=(
                        None
                        if node_label_probabilities is None
                        else np.asarray(node_label_probabilities[graph_idx], dtype=float)
                    ),
                    edge_label_probabilities=(
                        None
                        if edge_label_probabilities is None
                        else np.asarray(edge_label_probabilities[graph_idx], dtype=float)
                    ),
                    forbidden_node_assignments=accumulated_node_label_forbidden,
                    forbidden_edge_assignments=accumulated_edge_label_forbidden,
                )
                repaired_edge_label_matrix = owner._fill_unlabeled_active_edges(
                    adj_mtx=single_adj_mtx,
                    edge_label_matrix=repaired_edge_label_matrix,
                    edge_label_probabilities=(
                        None
                        if edge_label_probabilities is None
                        else np.asarray(edge_label_probabilities[graph_idx], dtype=float)
                    ),
                )
                labels_changed = not np.array_equal(repaired_node_labels, current_node_labels)
                edge_labels_changed = not np.array_equal(repaired_edge_label_matrix, current_edge_label_matrix)
                if labels_changed or edge_labels_changed:
                    candidate = evaluate_oracle_state(repaired_node_labels, repaired_edge_label_matrix)
                    candidate_rank = oracle_rank(candidate[1], candidate[2], candidate[3])
                    if candidate_rank < current_rank:
                        current_node_labels = repaired_node_labels
                        current_edge_label_matrix = repaired_edge_label_matrix
                        (
                            graph,
                            current_node_violation_sets,
                            current_edge_violation_sets,
                            score,
                            edge_score,
                            node_score,
                            edge_label_score,
                        ) = candidate
                        current_rank = candidate_rank
                        joint_label_changed = True
                        if labels_changed and edge_labels_changed:
                            joint_label_detail = "node+edge labels changed"
                        elif labels_changed:
                            joint_label_detail = "node labels changed"
                        else:
                            joint_label_detail = "edge labels changed"
                    else:
                        joint_label_detail = "joint label follow-up rejected"
                else:
                    joint_label_detail = "joint label follow-up unchanged"

            if not joint_label_changed and node_label_probabilities is None and edge_label_probabilities is None:
                joint_label_detail = "label probabilities unavailable"
            elif not joint_label_changed and (
                (use_node_label_cuts and accumulated_node_label_forbidden)
                or (use_edge_label_cuts and accumulated_edge_label_forbidden)
            ) and "rejected" not in joint_label_detail and "unchanged" not in joint_label_detail:
                joint_label_detail = "label follow-up attempted but unchanged"

            current_total_log_score = float(score)
            current_edge_log_score = float(edge_score)
            current_node_log_score = float(node_score)
            current_edge_label_log_score = float(edge_label_score)
            plot_oracle_phase(
                "Soft Label Follow-Up",
                adj_mtx=single_adj_mtx,
                decoded_graph=graph,
                node_violation_sets=current_node_violation_sets,
                edge_violation_sets=current_edge_violation_sets,
                new_node_cut_count=len(new_node_forbidden),
                new_edge_label_cut_count=len(new_edge_label_forbidden),
                joint_label_changed=joint_label_changed,
                detail=joint_label_detail,
            )
            if score > best_score:
                best_score = score
                best_graph = graph

            is_feasible = not current_node_violation_sets and not current_edge_violation_sets
            if is_feasible and score > best_feasible_score:
                best_feasible_score = score
                best_feasible_graph = graph
                plot_oracle_phase(
                    "Feasibility Check",
                    adj_mtx=single_adj_mtx,
                    decoded_graph=graph,
                    node_violation_sets=current_node_violation_sets,
                    edge_violation_sets=current_edge_violation_sets,
                    joint_label_changed=joint_label_changed,
                )
                break

            persistent_structural_cuts = [
                edge_set for edge_set in current_edge_violation_sets if edge_set not in accumulated_structural_cuts
            ]
            if not new_node_forbidden and not new_edge_label_forbidden and not persistent_structural_cuts and not joint_label_changed:
                plot_oracle_phase(
                    "Feasibility Check",
                    adj_mtx=single_adj_mtx,
                    decoded_graph=graph,
                    node_violation_sets=current_node_violation_sets,
                    edge_violation_sets=current_edge_violation_sets,
                    joint_label_changed=joint_label_changed,
                )
                break

            local_edge_violation_prior = update_oracle_edge_memory(
                local_edge_violation_prior,
                current_edge_violation_sets,
                update_weight=owner.oracle_edge_memory_update,
                decay=owner.oracle_edge_memory_decay,
                clip_value=owner.oracle_edge_memory_clip,
            )
            accumulated_structural_cuts.extend(persistent_structural_cuts)
            if _iteration_idx + 1 >= owner.max_oracle_iterations:
                plot_oracle_phase(
                    "Structural Edge-Set Phase",
                    adj_mtx=single_adj_mtx,
                    decoded_graph=graph,
                    node_violation_sets=current_node_violation_sets,
                    edge_violation_sets=current_edge_violation_sets,
                    joint_label_changed=joint_label_changed,
                    new_structural_cut_count=len(persistent_structural_cuts),
                )
                break
            single_adj_mtx = solve_oracle_relaxed_adjacency(
                owner,
                masked_prob_matrix=masked_prob_matrix,
                target_degrees=target_degrees,
                accumulated_cuts=accumulated_structural_cuts,
                start_iteration_idx=_iteration_idx + 1,
                edge_violation_prior=local_edge_violation_prior,
            )
            current_edge_label_matrix = owner._fill_unlabeled_active_edges(
                adj_mtx=single_adj_mtx,
                edge_label_matrix=current_edge_label_matrix,
                edge_label_probabilities=None if edge_label_probabilities is None else np.asarray(
                    edge_label_probabilities[graph_idx],
                    dtype=float,
                ),
            )
            if (
                (
                    use_node_label_cuts
                    and accumulated_node_label_forbidden
                    and node_label_probabilities is not None
                )
                or (
                    use_edge_label_cuts
                    and accumulated_edge_label_forbidden
                    and edge_label_probabilities is not None
                )
            ):
                current_node_labels, current_edge_label_matrix = owner._repair_labels_with_oracle(
                    existence_mask=existence_mask,
                    adj_mtx=single_adj_mtx,
                    current_node_labels=current_node_labels,
                    current_edge_label_matrix=current_edge_label_matrix,
                    node_label_probabilities=(
                        None
                        if node_label_probabilities is None
                        else np.asarray(node_label_probabilities[graph_idx], dtype=float)
                    ),
                    edge_label_probabilities=(
                        None
                        if edge_label_probabilities is None
                        else np.asarray(edge_label_probabilities[graph_idx], dtype=float)
                    ),
                    forbidden_node_assignments=accumulated_node_label_forbidden,
                    forbidden_edge_assignments=accumulated_edge_label_forbidden,
                )
            structural_graph = _assemble_graph_job(
                existence_mask,
                current_node_labels,
                np.asarray(_edge_label_matrix_to_list(single_adj_mtx, current_edge_label_matrix), dtype=object),
                np.asarray(single_adj_mtx, dtype=float),
            )
            structural_node_violation_sets = (
                owner._get_oracle_node_violation_sets(structural_graph, n_nodes=single_adj_mtx.shape[0])
                if use_node_label_cuts
                else []
            )
            structural_edge_violation_sets = owner._get_oracle_edge_violation_sets(
                structural_graph,
                n_nodes=single_adj_mtx.shape[0],
            )
            plot_oracle_phase(
                "Structural Edge-Set Phase",
                adj_mtx=single_adj_mtx,
                decoded_graph=structural_graph,
                node_violation_sets=structural_node_violation_sets,
                edge_violation_sets=structural_edge_violation_sets,
                joint_label_changed=False,
                new_structural_cut_count=len(persistent_structural_cuts),
                detail="post-structural solve",
            )
        final_graph = best_feasible_graph if best_feasible_graph is not None else best_graph
        if final_graph is None:
            raise RuntimeError("Oracle-guided decoding failed to assemble a graph.")
        decoded_graphs.append(final_graph)
    return decoded_graphs
