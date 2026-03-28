"""Decoder helpers for rebuilding labeled graphs from node-field predictions."""

import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import dill as pickle
import networkx as nx
import numpy as np
import pulp

from .conditional_node_field_generator import GeneratedNodeBatch
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
) -> np.ndarray:
    decoder = ConditionalNodeFieldGraphDecoder(
        verbose=bool(verbose),
        degree_slack_penalty=degree_slack_penalty,
        enforce_connectivity=enforce_connectivity,
        warm_start_mst=warm_start_mst,
        diagnostic_graph_renderer=diagnostic_graph_renderer,
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
    ) -> None:
        self.verbose = verbose
        self.existence_threshold = existence_threshold
        self.enforce_connectivity = enforce_connectivity
        self.degree_slack_penalty = degree_slack_penalty
        self.warm_start_mst = warm_start_mst
        self.n_jobs = _normalize_n_jobs(n_jobs)
        self.diagnostic_graph_renderer = diagnostic_graph_renderer

    def optimize_adjacency_matrix(
        self,
        prob_matrix: np.ndarray,
        target_degrees: List[int],
        target_edge_count: Optional[int] = None,
        timeLimit: int = 60,
        verbose: bool = False,
        alpha: float = 0.7,
        connectivity: Optional[bool] = None,
        forbidden_edge_sets: Optional[Iterable[Iterable[Sequence[Any]]]] = None,
    ) -> np.ndarray:
        n = prob_matrix.shape[0]
        if alpha != 1.0:
            prob_matrix = np.power(prob_matrix, alpha)
        if connectivity is None:
            connectivity = self.enforce_connectivity

        prob = pulp.LpProblem("AdjacencyMatrixOptimization", pulp.LpMaximize)
        x = {(i, j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary") for i in range(n) for j in range(i + 1, n)}
        u = {i: pulp.LpVariable(f"u_{i}", lowBound=0, cat="Integer") for i in range(n)}
        v = {i: pulp.LpVariable(f"v_{i}", lowBound=0, cat="Integer") for i in range(n)}

        edge_log_likelihood_terms = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_prob = float(np.clip(prob_matrix[i, j], _DECODER_PROBABILITY_EPS, 1.0 - _DECODER_PROBABILITY_EPS))
                edge_log_likelihood_terms.append((np.log(edge_prob) - np.log(1.0 - edge_prob)) * x[(i, j)])
        prob += (
            pulp.lpSum(edge_log_likelihood_terms)
            - self.degree_slack_penalty * pulp.lpSum(u[i] + v[i] for i in range(n))
        )

        for i in range(n):
            incident = [x[(i, j)] for j in range(i + 1, n)] + [x[(j, i)] for j in range(i) if (j, i) in x]
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
            f_vars = {(u_, v_): pulp.LpVariable(f"f_{u_}_{v_}", lowBound=0, cat="Continuous") for u_, v_ in directed_edges}
            M = n - 1
            root = 0
            for v_idx in range(n):
                inflow = pulp.lpSum(f_vars[(u_, v2)] for (u_, v2) in directed_edges if v2 == v_idx)
                outflow = pulp.lpSum(f_vars[(v2, w)] for (v2, w) in directed_edges if v2 == v_idx)
                prob += ((outflow - inflow) == M if v_idx == root else (inflow - outflow) == 1), f"Flow_{v_idx}"
            for u_, v_ in directed_edges:
                i, j = min(u_, v_), max(u_, v_)
                prob += (f_vars[(u_, v_)] <= M * x[(i, j)]), f"FlowCouple_{u_}_{v_}"

        normalized_forbidden_edge_sets = _normalize_violating_edge_sets(
            [] if forbidden_edge_sets is None else forbidden_edge_sets,
            n_nodes=n,
        )
        for cut_idx, edge_set in enumerate(normalized_forbidden_edge_sets):
            prob += (pulp.lpSum(x[edge] for edge in edge_set) <= len(edge_set) - 1), f"ForbiddenMotif_{cut_idx}"

        if self.warm_start_mst:
            graph = nx.Graph()
            graph.add_nodes_from(range(n))
            for i in range(n):
                for j in range(i + 1, n):
                    graph.add_edge(i, j, weight=prob_matrix[i, j])
            tree = nx.maximum_spanning_tree(graph)
            for (i, j), var in x.items():
                var.start = 1 if tree.has_edge(i, j) else 0

        solver = pulp.PULP_CBC_CMD(timeLimit=timeLimit, msg=verbose)
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
            for i in range(n_nodes):
                lengths = nx.single_source_shortest_path_length(graph, i, cutoff=horizon)
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
                candidate_indices = [k for k in range(n_nodes) if k != i and k not in lengths]
                if not candidate_indices:
                    continue
                distances = np.array([np.linalg.norm(encodings[i] - encodings[k]) for k in candidate_indices])
                sorted_candidate_indices = np.argsort(distances)
                selected_negatives = [candidate_indices[idx] for idx in sorted_candidate_indices[:num_neg_samples]]
                for k in selected_negatives:
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
        existence_scores = None if generated_nodes.node_existence_probabilities is None else np.asarray(
            generated_nodes.node_existence_probabilities,
            dtype=float,
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
        predicted_probs_list = []
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
        return _parallel_map(_decode_single_adjacency_job_star, jobs, self.n_jobs, verbose=bool(self.verbose))

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
        desired_node_counts: Optional[Sequence[int]] = None,
        desired_edge_counts: Optional[Sequence[int]] = None,
    ) -> List[nx.Graph]:
        adj_mtx_list = self.decode_adjacency_matrix(
            generated_nodes,
            predicted_edge_probability_matrices=predicted_edge_probability_matrices,
            desired_node_counts=desired_node_counts,
            desired_edge_counts=desired_edge_counts,
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
            decoded_graphs = _parallel_map(_assemble_graph_job_star, jobs, self.n_jobs, verbose=bool(self.verbose))

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
