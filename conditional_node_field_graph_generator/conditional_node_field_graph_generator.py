"""Graph encoder/decoder helpers used by the maintained conditional graph-generation pipeline."""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
import io
import inspect
import logging
import os
import time
import numpy as np
import pandas as pd
import networkx as nx
import random
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

DEFAULT_DUMMY_NODE_LABEL = "__dummy_node_label__"
logger = get_runtime_logger(__name__)
Edge = Tuple[int, int]
NodeSet = Tuple[int, ...]
ForbiddenNodeLabelAssignment = Tuple[NodeSet, Tuple[Any, ...]]
ForbiddenEdgeLabelAssignment = Tuple[Tuple[Edge, ...], Tuple[Any, ...]]
_ORACLE_PROBABILITY_EPS = 1e-6
_ORACLE_EDGE_EXISTENCE_WEIGHT = 1.0
_ORACLE_NODE_LABEL_WEIGHT = 1.0
_ORACLE_EDGE_LABEL_WEIGHT = 1.0


def _format_elapsed_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m {seconds:04.1f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:d}h {minutes:02d}m {seconds:04.1f}s"


def _format_feasibility_attempt_status(
    *,
    attempt: int,
    max_attempts: int,
    attempted_total: int,
    feasible_now: int,
    filled_now: int,
    pending_now: int,
    acceptance_rate: float,
    filled_total: int,
    missing_total: int,
    attempt_elapsed_seconds: float,
    total_elapsed_seconds: float,
) -> str:
    remaining_attempts = max(0, int(max_attempts) - int(attempt))
    eta_seconds = 0.0 if pending_now <= 0 else (total_elapsed_seconds / max(1, attempt)) * remaining_attempts
    return (
        f"Feasibility attempt {attempt:>2}/{max_attempts:<2} | "
        f"generated={attempted_total:>4} | "
        f"feasible_candidates={feasible_now:>2} | "
        f"fulfilled_slots={filled_now:>2} | "
        f"pending_slots={pending_now:>2} | "
        f"feasible_rate={acceptance_rate:>6.1%} | "
        f"fulfilled_total={filled_total:>2} | "
        f"missing_total={missing_total:>2} | "
        f"attempt_time={_format_elapsed_seconds(attempt_elapsed_seconds):>8} | "
        f"eta={_format_elapsed_seconds(eta_seconds):>8}"
    )


def _canonicalize_edge(edge: Sequence[Any]) -> Optional[Edge]:
    if len(edge) != 2:
        return None
    try:
        u = int(edge[0])
        v = int(edge[1])
    except (TypeError, ValueError):
        return None
    if u == v:
        return None
    return (u, v) if u < v else (v, u)


def _normalize_violating_edge_sets(
    edge_sets: Iterable[Iterable[Sequence[Any]]],
    *,
    n_nodes: Optional[int] = None,
) -> List[FrozenSet[Edge]]:
    normalized: List[FrozenSet[Edge]] = []
    seen: set[FrozenSet[Edge]] = set()
    for edge_set in edge_sets:
        canonical_edges = []
        for edge in edge_set:
            normalized_edge = _canonicalize_edge(edge)
            if normalized_edge is None:
                continue
            if n_nodes is not None and (
                normalized_edge[0] < 0
                or normalized_edge[1] < 0
                or normalized_edge[0] >= int(n_nodes)
                or normalized_edge[1] >= int(n_nodes)
            ):
                continue
            canonical_edges.append(normalized_edge)
        frozen = frozenset(canonical_edges)
        if not frozen or frozen in seen:
            continue
        seen.add(frozen)
        normalized.append(frozen)
    return normalized


def _normalize_violating_node_sets(
    node_sets: Iterable[Iterable[Any]],
    *,
    n_nodes: Optional[int] = None,
) -> List[NodeSet]:
    normalized: List[NodeSet] = []
    seen: set[NodeSet] = set()
    for node_set in node_sets:
        canonical_nodes = []
        for node in node_set:
            try:
                node_idx = int(node)
            except (TypeError, ValueError):
                continue
            if n_nodes is not None and (node_idx < 0 or node_idx >= int(n_nodes)):
                continue
            canonical_nodes.append(node_idx)
        canonical = tuple(sorted(set(canonical_nodes)))
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        normalized.append(canonical)
    return normalized


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


@dataclass(frozen=True)
class SupervisionChannelPlan:
    """Description of how one prediction channel should be handled during training."""

    name: str
    mode: str
    reason: str
    constant_value: Optional[Any] = None
    horizon: Optional[int] = None
    enabled: bool = False


@dataclass(frozen=True)
class SupervisionPlan:
    """Single source of truth for supervision decisions in the orchestration layer."""

    node_labels: SupervisionChannelPlan
    edge_labels: SupervisionChannelPlan
    direct_edges: SupervisionChannelPlan
    auxiliary_locality: SupervisionChannelPlan

    def as_dict(self) -> Dict[str, SupervisionChannelPlan]:
        return {
            "node_labels": self.node_labels,
            "edge_labels": self.edge_labels,
            "direct_edges": self.direct_edges,
            "auxiliary_locality": self.auxiliary_locality,
        }


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

def _interpolate_integer_series(start, end, ts, minimum):
    values = np.rint([(1.0 - t) * start + t * end for t in ts]).astype(np.int64)
    return np.maximum(values, np.int64(minimum))

def scaled_slerp(v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
    """Interpolate between vectors on the hypersphere while blending magnitudes linearly.

    Args:
        v0 (np.ndarray): Input value.
        v1 (np.ndarray): Input value.
        t (float): Input value.

    Returns:
        np.ndarray: Computed result.
    """
    # Compute magnitudes
    mag0 = np.linalg.norm(v0)
    mag1 = np.linalg.norm(v1)

    # Normalize directions (guard against zero)
    v0_unit = v0 / mag0 if mag0 != 0 else v0
    v1_unit = v1 / mag1 if mag1 != 0 else v1

    # Compute angle between
    dot = np.clip(np.dot(v0_unit, v1_unit), -1.0, 1.0)
    theta = np.arccos(dot)

    # Slerp the direction
    if theta < 1e-6:
        # Nearly colinear: fall back to linear interpolation + renormalization
        direction = (1 - t) * v0_unit + t * v1_unit
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm != 0 else direction
    else:
        sin_theta = np.sin(theta)
        direction = (
            np.sin((1 - t) * theta) * v0_unit +
            np.sin(t * theta) * v1_unit
        ) / sin_theta

    # Linearly interpolate magnitudes
    mag = (1 - t) * mag0 + t * mag1
    return direction * mag


def scaled_slerp_average(vectors: np.ndarray) -> np.ndarray:
    """Compute a magnitude-aware mean direction for a batch of vectors.

    Args:
        vectors (np.ndarray): Input value.

    Returns:
        np.ndarray: Computed result.
    """
    vs = np.asarray(vectors, dtype=float)             # (B, D)
    mags = np.linalg.norm(vs, axis=1)                # (B,)
    unit_vs = np.zeros_like(vs)                      # (B, D)
    nonzero = mags > 0
    unit_vs[nonzero] = vs[nonzero] / mags[nonzero, None]

    avg_dir = unit_vs.sum(axis=0)                    # (D,)
    norm = np.linalg.norm(avg_dir)
    if norm > 0:
        avg_dir /= norm

    avg_mag = mags.mean()
    return avg_dir * avg_mag                         # (D,)


def _normalize_n_jobs(n_jobs: Optional[int]) -> int:
    if n_jobs is None:
        return 1
    n_jobs = int(n_jobs)
    if n_jobs == 0:
        raise ValueError("n_jobs must be != 0.")
    if n_jobs < 0:
        cpu_count = os.cpu_count() or 1
        return max(1, cpu_count + 1 + n_jobs)
    return max(1, n_jobs)


def _parallel_map(func, jobs, max_workers: int, verbose: bool = False):
    if max_workers <= 1 or len(jobs) <= 1:
        return [func(job) for job in jobs]
    try:
        with ProcessPoolExecutor(max_workers=min(max_workers, len(jobs))) as executor:
            return list(executor.map(func, jobs))
    except (OSError, PermissionError):
        if verbose:
            print("Process-based decode parallelism unavailable; falling back to threads.")
        with ThreadPoolExecutor(max_workers=min(max_workers, len(jobs))) as executor:
            return list(executor.map(func, jobs))


def _plot_decoder_diagnostics(
    *,
    prob_matrix: np.ndarray,
    adj_mtx: np.ndarray,
    target_degrees: Sequence[int],
    title: str,
    violating_edge_sets: Optional[Iterable[Iterable[Sequence[Any]]]] = None,
    decoded_graph: Optional[nx.Graph] = None,
    graph_renderer: Optional[Callable[..., Any]] = None,
) -> None:
    if plt is None:
        return
    prob_matrix = np.asarray(prob_matrix, dtype=float)
    adj_mtx = np.asarray(adj_mtx, dtype=float)
    target_degrees = np.asarray(target_degrees, dtype=float)
    realized_degrees = adj_mtx.sum(axis=1)
    normalized_violations = _normalize_violating_edge_sets(
        [] if violating_edge_sets is None else violating_edge_sets,
        n_nodes=adj_mtx.shape[0],
    )

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    im0 = axes[0].imshow(prob_matrix, vmin=0.0, vmax=max(1.0, float(np.max(prob_matrix)) if prob_matrix.size else 1.0), cmap="viridis")
    axes[0].set_title("Edge probabilities")
    axes[0].set_xlabel("node j")
    axes[0].set_ylabel("node i")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].imshow(adj_mtx, vmin=0.0, vmax=1.0, cmap="gray")
    axes[1].set_title("Decoded adjacency")
    axes[1].set_xlabel("node j")
    axes[1].set_ylabel("node i")
    if normalized_violations:
        for edge_set in normalized_violations:
            for i, j in edge_set:
                axes[1].plot([j, i], [i, j], marker="s", color="tab:red", markersize=6, linewidth=1.5)
                axes[1].plot([i, j], [j, i], marker="s", color="tab:red", markersize=6, linewidth=1.5)

    node_idx = np.arange(len(target_degrees))
    axes[2].bar(node_idx - 0.18, target_degrees, width=0.36, label="target")
    axes[2].bar(node_idx + 0.18, realized_degrees, width=0.36, label="realized")
    axes[2].set_title("Degree targets vs realized")
    axes[2].set_xlabel("node")
    axes[2].set_ylabel("degree")
    axes[2].legend()

    graph = nx.from_numpy_array(adj_mtx.astype(int))
    violating_edges = {
        (min(i, j), max(i, j))
        for edge_set in normalized_violations
        for i, j in edge_set
    }
    rendered_inline = False
    if decoded_graph is not None:
        rendered_inline = _try_render_molecular_graph_inline(
            axes[3],
            decoded_graph=decoded_graph,
            title=title,
        )
    if not rendered_inline and (graph_renderer is None or decoded_graph is None):
        layout = nx.circular_layout(graph)
        edge_colors = [
            "tab:red" if (min(u, v), max(u, v)) in violating_edges else "black"
            for u, v in graph.edges()
        ]
        edge_widths = [
            2.5 if (min(u, v), max(u, v)) in violating_edges else 1.5
            for u, v in graph.edges()
        ]
        nx.draw_networkx(
            graph,
            pos=layout,
            ax=axes[3],
            node_color="white",
            edge_color=edge_colors,
            width=edge_widths,
            with_labels=True,
            font_size=9,
            node_size=500,
            linewidths=1.5,
        )
        axes[3].set_title("Decoded graph")
    elif not rendered_inline:
        axes[3].text(
            0.5,
            0.5,
            "Custom graph renderer\nshown below",
            ha="center",
            va="center",
            fontsize=11,
        )
        axes[3].set_title("Decoded graph")
    axes[3].set_axis_off()

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    if graph_renderer is not None and decoded_graph is not None and not rendered_inline:
        try:
            graph_renderer([decoded_graph], legends=[title])
        except TypeError:
            graph_renderer([decoded_graph])


def _is_molecule_like_graph(graph: nx.Graph) -> bool:
    atom_symbols = {
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
        "Br", "I",
    }
    graph_meta = getattr(graph, "graph", {})
    if any(key in graph_meta for key in ("smiles", "mol", "molecule", "inchi")):
        return True
    for _, attrs in graph.nodes(data=True):
        if "symbol" in attrs or "atomic_num" in attrs or "atom" in attrs:
            return True
        label = attrs.get("label")
        if isinstance(label, str) and label in atom_symbols:
            return True
    return False


def _try_render_molecular_graph_inline(ax: Any, *, decoded_graph: nx.Graph, title: str) -> bool:
    if not _is_molecule_like_graph(decoded_graph):
        return False
    try:
        from .extensions.molecular import molecule_graphs_to_grid_image
    except Exception:
        return False
    image = molecule_graphs_to_grid_image(
        [decoded_graph],
        legends=[title],
        mols_per_row=1,
        sub_img_size=(500, 350),
    )
    if image is None:
        return False
    image_array = _coerce_inline_image_array(image)
    if image_array is None:
        return False
    ax.imshow(image_array)
    ax.set_title("Decoded graph")
    ax.set_axis_off()
    return True


def _coerce_inline_image_array(image: Any) -> Optional[np.ndarray]:
    try:
        image_array = np.asarray(image)
        if image_array.dtype != object:
            return image_array
    except Exception:
        image_array = None

    image_bytes = getattr(image, "data", None)
    if image_bytes is None:
        return None
    if isinstance(image_bytes, memoryview):
        image_bytes = image_bytes.tobytes()
    if isinstance(image_bytes, str):
        image_bytes = image_bytes.encode("utf-8")
    try:
        from PIL import Image
    except Exception:
        return None
    try:
        with Image.open(io.BytesIO(image_bytes)) as pil_image:
            return np.asarray(pil_image)
    except Exception:
        return None


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


def _decode_single_adjacency_job(
    prob_list: np.ndarray,
    existence_mask: np.ndarray,
    degree_prediction: np.ndarray,
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
    existent = np.asarray(existence_mask[:n_nodes], dtype=bool)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if not (existent[i] and existent[j]):
                prob_matrix[i, j] = 0
    prob_matrix = (prob_matrix + prob_matrix.T) / 2
    target_degrees = decoder.get_degrees(
        np.asarray(degree_prediction[:n_nodes], dtype=float),
        existent,
    )
    adj_mtx = decoder.optimize_adjacency_matrix(prob_matrix, target_degrees)
    if int(verbose) >= 4 and diagnostic_graph_renderer is None:
        _plot_decoder_diagnostics(
            prob_matrix=prob_matrix,
            adj_mtx=adj_mtx,
            target_degrees=target_degrees,
            title="Decoder solve",
            graph_renderer=decoder.diagnostic_graph_renderer,
        )
    return adj_mtx


def _decode_single_adjacency_job_star(args) -> np.ndarray:
    return _decode_single_adjacency_job(*args)


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

    existent_indices = np.where(np.asarray(node_presence_mask[:adj_mtx.shape[0]], dtype=bool))[0]
    return graph.subgraph(existent_indices).copy()


def _assemble_graph_job_star(args) -> nx.Graph:
    return _assemble_graph_job(*args)


# =============================================================================
# ConditionalNodeFieldGraphDecoder Class
# =============================================================================

class ConditionalNodeFieldGraphDecoder(object):
    """Graph decoder that turns generator outputs into final NetworkX graphs."""
    
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
        """Store graph decoding hyper-parameters.

        Args:
            verbose (bool): Optional input value.
            existence_threshold (float): Optional input value.
            enforce_connectivity (bool): Optional input value.
            degree_slack_penalty (float): Optional input value.
            warm_start_mst (bool): Optional input value.
            n_jobs (int): Number of worker processes used for per-graph decode. Use `1`
                to disable parallelism and `-1` to use all available CPUs.
            diagnostic_graph_renderer (Optional[Callable[..., Any]]): Optional callable
                used when `verbose >= 4` to render decoded graphs in notebook workflows.
        """
        self.verbose                    = verbose
        self.existence_threshold        = existence_threshold
        self.enforce_connectivity       = enforce_connectivity
        self.degree_slack_penalty       = degree_slack_penalty
        self.warm_start_mst             = warm_start_mst
        self.n_jobs                     = _normalize_n_jobs(n_jobs)
        self.diagnostic_graph_renderer  = diagnostic_graph_renderer

    def optimize_adjacency_matrix(
        self,
        prob_matrix: np.ndarray,
        target_degrees: List[int],
        timeLimit: int = 60,
        verbose: bool = False,
        alpha: float = 0.7,
        connectivity: Optional[bool] = None,
        forbidden_edge_sets: Optional[Iterable[Iterable[Sequence[Any]]]] = None,
    ) -> np.ndarray:
        """Optimise a binary adjacency matrix subject to degree and connectivity targets.

        Args:
            prob_matrix (np.ndarray): Input value.
            target_degrees (List[int]): Input value.
            timeLimit (int): Optional input value.
            verbose (bool): Optional input value.
            alpha (float): Optional input value.
            connectivity (Optional[bool]): Optional input value.

        Returns:
            np.ndarray: Computed result.
        """
        n = prob_matrix.shape[0]
        # Smooth probabilities
        if alpha != 1.0:
            prob_matrix = np.power(prob_matrix, alpha)

        # Connectivity setting
        if connectivity is None:
            connectivity = self.enforce_connectivity

        # Build LP
        prob = pulp.LpProblem("AdjacencyMatrixOptimization", pulp.LpMaximize)

        # Decision vars
        x = {(i, j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary")
             for i in range(n) for j in range(i+1, n)}
        u = {i: pulp.LpVariable(f"u_{i}", lowBound=0, cat="Integer") for i in range(n)}
        v = {i: pulp.LpVariable(f"v_{i}", lowBound=0, cat="Integer") for i in range(n)}

        # Objective: maximize binary log-likelihood of edge existence plus degree slack penalty.
        edge_log_likelihood_terms = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_prob = float(np.clip(prob_matrix[i, j], _ORACLE_PROBABILITY_EPS, 1.0 - _ORACLE_PROBABILITY_EPS))
                edge_log_likelihood_terms.append(
                    (np.log(edge_prob) - np.log(1.0 - edge_prob)) * x[(i, j)]
                )
        prob += (
            pulp.lpSum(edge_log_likelihood_terms)
            - self.degree_slack_penalty * pulp.lpSum(u[i] + v[i] for i in range(n))
        )

        # Degree constraints
        for i in range(n):
            incident = [x[(i,j)] for j in range(i+1, n)] + [x[(j,i)] for j in range(i) if (j,i) in x]
            prob += (pulp.lpSum(incident) + u[i] - v[i] == target_degrees[i]), f"Degree_{i}"

        # Connectivity via flow
        if connectivity:
            directed_edges = [(i,j) for (i,j) in x] + [(j,i) for (i,j) in x]
            f_vars = {(u_, v_): pulp.LpVariable(f"f_{u_}_{v_}", lowBound=0, cat="Continuous")
                      for u_,v_ in directed_edges}
            M = n-1
            root = 0
            for v_idx in range(n):
                inflow  = pulp.lpSum(f_vars[(u_,v2)] for (u_,v2) in directed_edges if v2==v_idx)
                outflow = pulp.lpSum(f_vars[(v2,w)] for (v2,w) in directed_edges if v2==v_idx)
                prob += ((outflow-inflow)==M if v_idx==root else (inflow-outflow)==1), f"Flow_{v_idx}"
            for u_,v_ in directed_edges:
                i,j = min(u_,v_), max(u_,v_)
                prob += (f_vars[(u_,v_)] <= M * x[(i,j)]), f"FlowCouple_{u_}_{v_}"

        normalized_forbidden_edge_sets = _normalize_violating_edge_sets(
            [] if forbidden_edge_sets is None else forbidden_edge_sets,
            n_nodes=n,
        )
        for cut_idx, edge_set in enumerate(normalized_forbidden_edge_sets):
            prob += (
                pulp.lpSum(x[edge] for edge in edge_set) <= len(edge_set) - 1
            ), f"ForbiddenMotif_{cut_idx}"

        # Warm-start with MST
        if self.warm_start_mst:
            G = nx.Graph()
            G.add_nodes_from(range(n))
            for i in range(n):
                for j in range(i+1, n):
                    G.add_edge(i, j, weight=prob_matrix[i,j])
            T = nx.maximum_spanning_tree(G)
            # Initialize x
            for (i,j), var in x.items():
                var.start = 1 if T.has_edge(i,j) else 0

        # Solve
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

        # Build adjacency
        adj = np.zeros((n,n), dtype=int)
        for (i,j), var in x.items():
            value = pulp.value(var)
            if value is None:
                raise RuntimeError(
                    "Adjacency ILP finished without assigning all decision variables "
                    f"(status={status_label}, missing_edge=({i}, {j}))."
                )
            adj[i,j] = adj[j,i] = int(round(float(value)))
        return adj

    def graphs_to_adjacency_matrices(self, graphs: List[nx.Graph]) -> List[np.ndarray]:
        """Convert each NetworkX graph into a dense adjacency array.

        Args:
            graphs (List[nx.Graph]): Input value.

        Returns:
            List[np.ndarray]: Computed result.
        """
        adj_mtx_list = []
        for graph in graphs:
            # Convert graph to a numpy array with integer type.
            adj_mtx = nx.to_numpy_array(graph, dtype=int)
            adj_mtx_list.append(adj_mtx)
        return adj_mtx_list

    def _target_stats(self, targets: List[int]) -> Tuple[int, int]:
        """Count positive and negative supervision pairs.

        Args:
            targets (List[int]): Input value.

        Returns:
            Tuple[int, int]: Computed result.
        """
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
        """Sample supervision indices using the configured class-balancing strategy.

        Args:
            targets (List[int]): Input value.
            sample_count (int): Input value.
            locality_sampling_strategy (str): Input value.
            locality_target_positive_ratio (Optional[float]): Input value.

        Returns:
            np.ndarray: Computed result.
        """
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
        """Label node pairs as local or non-local using shortest-path distance within each graph.

        Args:
            adj_mtx_list (List[np.ndarray]): Input value.
            node_encodings_list (List[np.ndarray]): Input value.
            locality_sample_fraction (float): Input value.
            negative_sample_factor (int): Optional input value.
            locality_sampling_strategy (str): Optional input value.
            locality_target_positive_ratio (Optional[float]): Optional input value.
            force_bi_directional_edges (bool): Optional input value.
            is_training (bool): Optional input value.
            horizon (int): Optional input value.
            supervision_name (str): Optional input value.

        Returns:
            Tuple[np.ndarray, List[Tuple[int, int, int]]]: Computed result.
        """
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

        # Collect all targets and pairs first
        all_targets = []
        all_pairs = []
        
        for g_idx, (adj_mtx, encodings) in enumerate(zip(adj_mtx_list, node_encodings_list)):
            n_nodes = adj_mtx.shape[0]
            # Treat graphs as simple, unweighted, undirected structures
            G = nx.from_numpy_array(adj_mtx, create_using=nx.Graph)
            for i in range(n_nodes):
                # Collect positive neighbours for node i up to the specified horizon
                lengths = nx.single_source_shortest_path_length(G, i, cutoff=horizon)
                pos_neighbors = [j for j, dist in lengths.items() if j != i and dist <= horizon]
                
                # Add positive examples (both directions if force_bi_directional_edges)
                for j in pos_neighbors:
                    all_targets.append(1)
                    all_pairs.append((g_idx, i, j))
                    if force_bi_directional_edges:
                        all_targets.append(1)  
                        all_pairs.append((g_idx, j, i))
                
                # Determine number of negative samples
                num_pos = len(pos_neighbors) * (2 if force_bi_directional_edges else 1)
                num_neg_samples = int(round(negative_sample_factor * num_pos))
                if num_neg_samples <= 0:
                    continue
                
                # Build candidate list for negative examples
                candidate_indices = [k for k in range(n_nodes) if k != i and k not in lengths]
                if not candidate_indices:
                    continue
                
                # Get distances and sort candidates
                distances = np.array([np.linalg.norm(encodings[i] - encodings[k]) for k in candidate_indices])
                sorted_candidate_indices = np.argsort(distances)
                selected_negatives = [candidate_indices[idx] for idx in sorted_candidate_indices[:num_neg_samples]]
                
                # Add negative examples (both directions if force_bi_directional_edges)
                for k in selected_negatives:
                    all_targets.append(0)
                    all_pairs.append((g_idx, i, k))
                    if force_bi_directional_edges:
                        all_targets.append(0)
                        all_pairs.append((g_idx, k, i))
        
        # Apply locality-pair sampling during training when requested
        pos_before, neg_before = self._target_stats(all_targets)
        if is_training and locality_sample_fraction < 1.0:
            num_pairs = len(all_pairs)
            num_pairs_to_use = int(round(num_pairs * locality_sample_fraction))

            if self.verbose and num_pairs > 0:
                print(
                    f"adj_mtx_to_targets[{supervision_name}, horizon={horizon}]: "
                    f"sampling {num_pairs_to_use} pairs ({locality_sample_fraction:.2%}) "
                    f"from {num_pairs} total pairs "
                    f"(pos={pos_before}, neg={neg_before}, "
                    f"negative_sample_factor={negative_sample_factor}, "
                    f"sampling_strategy={locality_sampling_strategy}"
                    f"{'' if locality_target_positive_ratio is None else f', target_positive_ratio={locality_target_positive_ratio:.3f}'})."
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
                    print(
                        f"adj_mtx_to_targets[{supervision_name}, horizon={horizon}]: "
                        f"warning - num_pairs_to_use is 0 with locality_sample_fraction="
                        f"{locality_sample_fraction} and num_pairs={num_pairs}. No pairs will be used."
                    )
                return np.array([]), []
            elif num_pairs_to_use == 0 and num_pairs == 0:
                return np.array([]), []

        if self.verbose and len(all_targets) > 0:
            pos_after, neg_after = self._target_stats(all_targets)
            ratio_after = pos_after / float(pos_after + neg_after)
            print(
                f"adj_mtx_to_targets[{supervision_name}, horizon={horizon}]: "
                f"using pos={pos_after}, neg={neg_after}, positive_ratio={ratio_after:.3f}."
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
        """Compute locality supervision pairs for training.

        Args:
            graphs (List[nx.Graph]): Input value.
            node_encodings_list (List[np.ndarray]): Input value.
            locality_sample_fraction (float): Input value.
            negative_sample_factor (int): Optional input value.
            locality_sampling_strategy (str): Optional input value.
            locality_target_positive_ratio (Optional[float]): Optional input value.
            horizon (int): Optional input value.
            supervision_name (str): Optional input value.

        Returns:
            Tuple[np.ndarray, List[Tuple[int, int, int]]]: Computed result.
        """
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
        horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Assemble feature/label pairs for the locality classifier derived from adjacencies.

        Args:
            node_encodings_list (List[np.ndarray]): Input value.
            adj_mtx_list (List[np.ndarray]): Input value.
            locality_sample_fraction (float): Input value.
            horizon (int): Optional input value.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Computed result.
        """
        y, pair_indices = self.adj_mtx_to_targets(
            adj_mtx_list,
            node_encodings_list,
            locality_sample_fraction=locality_sample_fraction,
            is_training=True,
            horizon=horizon
        )
        X = self.encodings_to_instances(node_encodings_list, pair_indices)
        return X, y

    def encodings_to_instances(
        self,
        node_encodings_list: List[np.ndarray],
        pair_indices: Optional[List[Tuple[int, int, int]]] = None,
        use_graph_encoding: bool = False
    ) -> np.ndarray:
        """Stack node-pair feature vectors, optionally augmented with a graph-level summary.

        Args:
            node_encodings_list (List[np.ndarray]): Input value.
            pair_indices (Optional[List[Tuple[int, int, int]]]): Optional input value.
            use_graph_encoding (bool): Optional input value.

        Returns:
            np.ndarray: Computed result.
        """
        instances = []
        if pair_indices is not None:
            # Use provided pair indices.
            for g_idx, i, j in pair_indices:
                encodings = node_encodings_list[g_idx]
                if use_graph_encoding: 
                    graph_encoding = np.sum(encodings, axis=0)
                    instance = np.hstack([graph_encoding, encodings[i], encodings[j]])
                else:
                    instance = np.hstack([encodings[i], encodings[j]])
                instances.append(instance)
        else:
            # Evaluate all pairs (i, j) with i != j for every graph.
            for g_idx, encodings in enumerate(node_encodings_list):
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
        
    def get_degrees(
        self,
        degree_predictions: np.ndarray,
        existence_mask: np.ndarray,
    ) -> List[int]:
        """Derive integer degree targets from explicit degree and existence predictions.

        Args:
            degree_predictions (np.ndarray): Input value.
            existence_mask (np.ndarray): Input value.

        Returns:
            List[int]: Computed result.
        """
        degs = np.rint(np.asarray(degree_predictions, dtype=float))
        existent = np.asarray(existence_mask, dtype=bool)
        degs = np.where(existent, np.maximum(degs, 1), 0)
        return degs.astype(int).tolist()

    def decode_adjacency_matrix(
        self,
        generated_nodes: GeneratedNodeBatch,
        predicted_edge_probability_matrices: Optional[List[np.ndarray]] = None,
    ) -> List[np.ndarray]:
        """Project predicted adjacency probabilities onto valid binary graphs.

        Args:
            generated_nodes (GeneratedNodeBatch): Input value.
            predicted_edge_probability_matrices (Optional[List[np.ndarray]]): Optional input value.

        Returns:
            List[np.ndarray]: Computed result.
        """
        existence_masks = generated_nodes.node_presence_mask
        degree_predictions = generated_nodes.node_degree_predictions

        if existence_masks is None:
            raise RuntimeError("decode_adjacency_matrix requires explicit node_presence_mask predictions.")
        if degree_predictions is None:
            raise RuntimeError("decode_adjacency_matrix requires explicit node_degree_predictions.")
        if len(degree_predictions) != len(existence_masks):
            raise ValueError(
                "node_degree_predictions must align with node_presence_mask "
                f"(got {len(degree_predictions)} degree rows for {len(existence_masks)} existence rows)."
            )

        if predicted_edge_probability_matrices is None:
            raise RuntimeError(
                "decode_adjacency_matrix requires generator-provided edge probabilities."
            )
        else:
            if len(predicted_edge_probability_matrices) != len(existence_masks):
                raise ValueError(
                    "predicted_edge_probability_matrices must align with generated node batches "
                    f"(got {len(predicted_edge_probability_matrices)} matrices for {len(existence_masks)} graphs)."
                )
            predicted_probs_list = []
            for existence_mask, degree_prediction, prob_matrix in zip(
                existence_masks,
                degree_predictions,
                predicted_edge_probability_matrices,
            ):
                n_nodes = min(len(existence_mask), len(degree_prediction))
                prob_matrix = np.asarray(prob_matrix, dtype=float)
                if prob_matrix.shape != (n_nodes, n_nodes):
                    raise ValueError(
                        "Each predicted edge-probability matrix must have shape (n_nodes, n_nodes); "
                        f"received {prob_matrix.shape} for n_nodes={n_nodes}."
                    )
                mask = ~np.eye(n_nodes, dtype=bool)
                predicted_probs_list.append(prob_matrix[mask])
        
        jobs = [
            (
                np.asarray(predicted_probs_list[graph_idx], dtype=float),
                np.asarray(existence_masks[graph_idx], dtype=bool),
                np.asarray(degree_predictions[graph_idx], dtype=float),
                float(self.degree_slack_penalty),
                bool(self.enforce_connectivity),
                bool(self.warm_start_mst),
                int(self.verbose),
                self.diagnostic_graph_renderer if self.n_jobs == 1 else None,
            )
            for graph_idx in range(len(predicted_probs_list))
        ]
        if int(self.verbose) >= 4 and self.n_jobs != 1 and len(jobs) > 1:
            print("Decoder plots for verbose>=4 are only shown when n_jobs=1; skipping plots during parallel adjacency decode.")
        if self.n_jobs == 1 or len(jobs) <= 1:
            return [_decode_single_adjacency_job(*job) for job in jobs]
        return _parallel_map(_decode_single_adjacency_job_star, jobs, self.n_jobs, verbose=bool(self.verbose))

    def decode_node_labels(
        self,
        generated_nodes: GeneratedNodeBatch,
        predicted_node_labels_list: Optional[List[np.ndarray]] = None,
    ) -> List[np.ndarray]:
        """Predict node-level labels for each encoding matrix.

        Args:
            generated_nodes (GeneratedNodeBatch): Input value.

        Returns:
            List[np.ndarray]: Computed result.
        """
        if predicted_node_labels_list is None:
            raise RuntimeError("decode_node_labels requires explicit node labels.")
        return [np.asarray(node_labels, dtype=object) for node_labels in predicted_node_labels_list]

    def decode_edge_labels(
        self,
        generated_nodes: GeneratedNodeBatch,
        adj_mtx_list: List[np.ndarray],
        predicted_edge_label_matrices: Optional[List[np.ndarray]] = None,
        predicted_edge_labels_list: Optional[List[np.ndarray]] = None,
    ) -> List[np.ndarray]:
        """Predict edge labels for every edge present in the supplied adjacency matrices.

        Args:
            generated_nodes (GeneratedNodeBatch): Input value.
            adj_mtx_list (List[np.ndarray]): Input value.
            predicted_edge_label_matrices (Optional[List[np.ndarray]]): Optional input value.

        Returns:
            List[np.ndarray]: Computed result.
        """
        if predicted_edge_labels_list is not None:
            if len(predicted_edge_labels_list) != len(adj_mtx_list):
                raise ValueError(
                    "predicted_edge_labels_list must align with adj_mtx_list "
                    f"(got {len(predicted_edge_labels_list)} label arrays for {len(adj_mtx_list)} graphs)."
                )
            return [np.asarray(edge_labels, dtype=object) for edge_labels in predicted_edge_labels_list]

        if predicted_edge_label_matrices is not None:
            if len(predicted_edge_label_matrices) != len(adj_mtx_list):
                raise ValueError(
                    "predicted_edge_label_matrices must align with adj_mtx_list "
                    f"(got {len(predicted_edge_label_matrices)} matrices for {len(adj_mtx_list)} graphs)."
                )
            predicted_edge_labels_list = []
            for adj_mtx, edge_label_matrix in zip(adj_mtx_list, predicted_edge_label_matrices):
                edge_label_matrix = np.asarray(edge_label_matrix, dtype=object)
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
                predicted_edge_labels_list.append(np.asarray(edge_labels, dtype=object))
            return predicted_edge_labels_list

        raise RuntimeError("decode_edge_labels requires explicit edge labels or edge-label matrices.")

    @timeit
    def decode(
        self,
        generated_nodes: GeneratedNodeBatch,
        predicted_node_labels_list: Optional[List[np.ndarray]] = None,
        predicted_edge_probability_matrices: Optional[List[np.ndarray]] = None,
        predicted_edge_labels_list: Optional[List[np.ndarray]] = None,
        predicted_edge_label_matrices: Optional[List[np.ndarray]] = None,
    ) -> List[nx.Graph]:
        """Reconstruct labelled graphs from predicted structural and semantic channels.

        Args:
            generated_nodes (GeneratedNodeBatch): Input value.
            predicted_node_labels_list (Optional[List[np.ndarray]]): Optional input value.
            predicted_edge_probability_matrices (Optional[List[np.ndarray]]): Optional input value.
            predicted_edge_label_matrices (Optional[List[np.ndarray]]): Optional input value.

        Returns:
            List[nx.Graph]: Computed result.
        """
        adj_mtx_list = self.decode_adjacency_matrix(
            generated_nodes,
            predicted_edge_probability_matrices=predicted_edge_probability_matrices,
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
        
        jobs = [
            (
                np.asarray(node_presence_mask, dtype=bool),
                np.asarray(node_labels, dtype=object),
                np.asarray(edge_labels, dtype=object),
                np.asarray(adj_mtx, dtype=float),
            )
            for node_presence_mask, node_labels, edge_labels, adj_mtx in zip(
                generated_nodes.node_presence_mask,
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
                existence_mask = np.asarray(generated_nodes.node_presence_mask[graph_idx], dtype=bool)
                degree_prediction = np.asarray(generated_nodes.node_degree_predictions[graph_idx], dtype=float)
                prob_matrix = np.asarray(predicted_edge_probability_matrices[graph_idx], dtype=float)
                masked_prob_matrix = _build_masked_prob_matrix(
                    existence_mask=existence_mask,
                    degree_prediction=degree_prediction,
                    prob_matrix=prob_matrix,
                )
                target_degrees = self.get_degrees(degree_prediction, existence_mask)
                _plot_decoder_diagnostics(
                    prob_matrix=masked_prob_matrix,
                    adj_mtx=np.asarray(adj_mtx, dtype=float),
                    target_degrees=target_degrees,
                    title=f"Decoder solve graph={graph_idx}",
                    decoded_graph=decoded_graph,
                    graph_renderer=self.diagnostic_graph_renderer,
                )

        return decoded_graphs

    def save(self, filename: str = 'generative_model.obj') -> None:
        """Serialise the current object to `filename` using pickle.

        Args:
            filename (str): Optional input value.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    def load(self, filename: str = 'generative_model.obj') -> 'ConditionalNodeFieldGraphDecoder':
        """Load a previously saved instance from disk and return it.

        Args:
            filename (str): Optional input value.

        Returns:
            'ConditionalNodeFieldGraphDecoder': Computed result.
        """
        with open(filename, 'rb') as f:
            self = pickle.load(f)
        return self

# =============================================================================
# ConditionalNodeFieldGraphGenerator Class 
# =============================================================================

class ConditionalNodeFieldGraphGenerator(object):
    """End-to-end manager that vectorises graphs, trains generators, and rebuilds structures."""
    _DEFAULT_FEASIBILITY_ORACLE_CANDIDATES_PER_ATTEMPT = 2

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
            max_oracle_iterations: int = 8,
            max_feasibility_attempts: int = 10,
            feasibility_candidates_per_attempt: int = 4,
            feasibility_failure_mode: str = "return_partial",
            model_name: Optional[str] = None,
            model_dir: Optional[str] = None,
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
            max_feasibility_attempts (int): Optional input value.
            feasibility_candidates_per_attempt (int): Optional input value.
            feasibility_failure_mode (str): Optional input value.
            model_name (Optional[str]): Optional input value.
            model_dir (Optional[str]): Optional input value.
        """
        self.graph_vectorizer = graph_vectorizer
        self.node_graph_vectorizer = node_graph_vectorizer
        self.conditional_node_generator_model = conditional_node_generator_model
        self.graph_decoder = graph_decoder
        self.verbose = verbose
        self.supervision_plan_: Optional[SupervisionPlan] = None
        self.training_graph_conditioning_: Optional[GraphConditioningBatch] = None
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
        self.max_feasibility_attempts = int(max_feasibility_attempts)
        self.feasibility_candidates_per_attempt = int(feasibility_candidates_per_attempt)
        self.feasibility_failure_mode = str(feasibility_failure_mode)
        self.model_name = model_name
        self.model_dir = model_dir
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
        if self.max_feasibility_attempts < 1:
            raise ValueError("max_feasibility_attempts must be >= 1")
        if self.feasibility_candidates_per_attempt < 1:
            raise ValueError("feasibility_candidates_per_attempt must be >= 1")
        valid_feasibility_failure_modes = {"raise", "return_partial"}
        if self.feasibility_failure_mode not in valid_feasibility_failure_modes:
            raise ValueError(
                f"feasibility_failure_mode must be one of {sorted(valid_feasibility_failure_modes)} "
                f"(got {self.feasibility_failure_mode!r})."
            )

    def _apply_legacy_feasibility_config(self) -> None:
        if not hasattr(self, "feasibility_oracle_candidates_per_attempt"):
            legacy_use_oracle = getattr(self, "use_feasibility_oracle", True)
            self.feasibility_oracle_candidates_per_attempt = (
                int(self._DEFAULT_FEASIBILITY_ORACLE_CANDIDATES_PER_ATTEMPT)
                if bool(legacy_use_oracle)
                else 0
            )
        self.feasibility_oracle_candidates_per_attempt = int(
            self.feasibility_oracle_candidates_per_attempt
        )
        if self.feasibility_oracle_candidates_per_attempt < 0:
            self.feasibility_oracle_candidates_per_attempt = 0

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._apply_legacy_feasibility_config()

    def set_feasibility_filtering(self, enabled: bool) -> None:
        """Enable or disable feasibility filtering during generation without discarding the fitted estimator.

        Args:
            enabled (bool): Input value.
        """
        self.use_feasibility_filtering = bool(enabled)

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
        node_embeddings_list = None if generated_nodes.node_embeddings_list is None else [
            np.asarray(generated_nodes.node_embeddings_list[graph_idx])
        ]
        node_presence_mask = None if generated_nodes.node_presence_mask is None else np.asarray(
            [generated_nodes.node_presence_mask[graph_idx]]
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
        return GeneratedNodeBatch(
            node_embeddings_list=node_embeddings_list,
            node_presence_mask=node_presence_mask,
            node_degree_predictions=node_degree_predictions,
            node_labels=node_labels,
            edge_probability_matrices=edge_probability_matrices,
            edge_label_matrices=edge_label_matrices,
            node_label_logits=node_label_logits,
            node_label_probabilities=node_label_probabilities,
            edge_existence_probabilities=edge_existence_probabilities,
            edge_label_logits=edge_label_logits,
            edge_label_probabilities=edge_label_probabilities,
        )

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
        existence_mask = np.asarray(existence_mask, dtype=bool)
        adj_mtx = np.asarray(adj_mtx, dtype=float)
        edge_probability_matrix = np.asarray(edge_probability_matrix, dtype=float)
        active_indices = np.where(existence_mask[: adj_mtx.shape[0]])[0]

        edge_terms = []
        for idx_i, i in enumerate(active_indices):
            for j in active_indices[idx_i + 1:]:
                edge_prob = float(np.clip(edge_probability_matrix[i, j], _ORACLE_PROBABILITY_EPS, 1.0 - _ORACLE_PROBABILITY_EPS))
                edge_terms.append(np.log(edge_prob) if adj_mtx[i, j] != 0 else np.log(1.0 - edge_prob))
        edge_score = float(np.mean(edge_terms)) if edge_terms else 0.0

        node_score = 0.0
        if node_label_probabilities is not None and self.node_label_to_index_ is not None:
            node_terms = []
            for node_idx in active_indices:
                label = node_labels[node_idx]
                label_idx = self.node_label_to_index_.get(label)
                if label_idx is None:
                    continue
                label_prob = float(np.clip(node_label_probabilities[node_idx, label_idx], _ORACLE_PROBABILITY_EPS, 1.0))
                node_terms.append(np.log(label_prob))
            if node_terms:
                node_score = float(np.mean(node_terms))

        edge_label_score = 0.0
        if edge_label_probabilities is not None and self.edge_label_to_index_ is not None:
            edge_label_terms = []
            for idx_i, i in enumerate(active_indices):
                for j in active_indices[idx_i + 1:]:
                    if adj_mtx[i, j] == 0:
                        continue
                    label = edge_label_matrix[i, j]
                    label_idx = self.edge_label_to_index_.get(label)
                    if label_idx is None:
                        continue
                    label_prob = float(np.clip(edge_label_probabilities[i, j, label_idx], _ORACLE_PROBABILITY_EPS, 1.0))
                    edge_label_terms.append(np.log(label_prob))
            if edge_label_terms:
                edge_label_score = float(np.mean(edge_label_terms))

        return (
            _ORACLE_EDGE_EXISTENCE_WEIGHT * edge_score
            + _ORACLE_NODE_LABEL_WEIGHT * node_score
            + _ORACLE_EDGE_LABEL_WEIGHT * edge_label_score
        )

    def _get_oracle_node_violation_sets(
        self,
        graph: nx.Graph,
        *,
        n_nodes: int,
    ) -> List[NodeSet]:
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
        if (
            node_label_probabilities is None
            or self.node_label_classes_ is None
            or self.node_label_to_index_ is None
            or len(self.node_label_classes_) == 0
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
            for label_idx in range(len(self.node_label_classes_))
        }
        prob += pulp.lpSum(
            np.log(float(np.clip(node_label_probabilities[node_idx, label_idx], _ORACLE_PROBABILITY_EPS, 1.0)))
            * y[(node_idx, label_idx)]
            for node_idx in affected_nodes
            for label_idx in range(len(self.node_label_classes_))
        )
        for node_idx in affected_nodes:
            prob += (
                pulp.lpSum(y[(node_idx, label_idx)] for label_idx in range(len(self.node_label_classes_))) == 1
            ), f"NodeLabelOneHot_{node_idx}"
        for cut_idx, (node_set, label_tuple) in enumerate(forbidden_assignments):
            if any(node_idx not in affected_nodes for node_idx in node_set):
                continue
            label_indices = [self.node_label_to_index_.get(label) for label in label_tuple]
            if any(label_idx is None for label_idx in label_indices):
                continue
            prob += (
                pulp.lpSum(
                    y[(node_idx, int(label_idx))]
                    for node_idx, label_idx in zip(node_set, label_indices)
                ) <= len(node_set) - 1
            ), f"ForbiddenNodeLabels_{cut_idx}"
        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        if int(status) != pulp.LpStatusOptimal:
            return np.asarray(current_node_labels, dtype=object)
        repaired = np.asarray(current_node_labels, dtype=object).copy()
        for node_idx in affected_nodes:
            for label_idx, label in enumerate(self.node_label_classes_):
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
        if (
            edge_label_probabilities is None
            or self.edge_label_classes_ is None
            or self.edge_label_to_index_ is None
            or len(self.edge_label_classes_) == 0
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
            for label_idx in range(len(self.edge_label_classes_))
        }
        prob += pulp.lpSum(
            np.log(float(np.clip(edge_label_probabilities[edge[0], edge[1], label_idx], _ORACLE_PROBABILITY_EPS, 1.0)))
            * z[(edge, label_idx)]
            for edge in active_edges
            for label_idx in range(len(self.edge_label_classes_))
        )
        for edge in active_edges:
            prob += (
                pulp.lpSum(z[(edge, label_idx)] for label_idx in range(len(self.edge_label_classes_))) == 1
            ), f"EdgeLabelOneHot_{edge[0]}_{edge[1]}"
        for cut_idx, (edge_set, label_tuple) in enumerate(forbidden_assignments):
            if any(edge not in active_edges for edge in edge_set):
                continue
            label_indices = [self.edge_label_to_index_.get(label) for label in label_tuple]
            if any(label_idx is None for label_idx in label_indices):
                continue
            prob += (
                pulp.lpSum(
                    z[(edge, int(label_idx))]
                    for edge, label_idx in zip(edge_set, label_indices)
                ) <= len(edge_set) - 1
            ), f"ForbiddenEdgeLabels_{cut_idx}"
        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        if int(status) != pulp.LpStatusOptimal:
            return np.asarray(current_edge_label_matrix, dtype=object)
        repaired = np.asarray(current_edge_label_matrix, dtype=object).copy()
        for edge in active_edges:
            selected_label = None
            for label_idx, label in enumerate(self.edge_label_classes_):
                value = pulp.value(z[(edge, label_idx)])
                if value is not None and int(round(float(value))) == 1:
                    selected_label = label
                    break
            repaired[edge[0], edge[1]] = selected_label
            repaired[edge[1], edge[0]] = selected_label
        return repaired

    def _sample_oracle_cuts_for_iteration(
        self,
        accumulated_cuts: Sequence[FrozenSet[Edge]],
        solve_iteration_idx: int,
    ) -> List[FrozenSet[Edge]]:
        if not accumulated_cuts:
            return []
        if self.max_oracle_iterations <= 1:
            return []
        if solve_iteration_idx >= self.max_oracle_iterations - 1:
            return []
        keep_fraction = 1.0 - (float(solve_iteration_idx) / float(self.max_oracle_iterations - 1))
        keep_count = int(np.ceil(len(accumulated_cuts) * keep_fraction))
        keep_count = max(0, min(len(accumulated_cuts), keep_count))
        if keep_count <= 0:
            return []
        if keep_count >= len(accumulated_cuts):
            return list(accumulated_cuts)
        selected_indices = sorted(random.sample(range(len(accumulated_cuts)), keep_count))
        return [accumulated_cuts[idx] for idx in selected_indices]

    def _solve_oracle_relaxed_adjacency(
        self,
        *,
        masked_prob_matrix: np.ndarray,
        target_degrees: List[int],
        accumulated_cuts: Sequence[FrozenSet[Edge]],
        start_iteration_idx: int,
    ) -> np.ndarray:
        last_error: Optional[Exception] = None
        for solve_iteration_idx in range(start_iteration_idx, self.max_oracle_iterations):
            active_cuts = self._sample_oracle_cuts_for_iteration(accumulated_cuts, solve_iteration_idx)
            try:
                return self.graph_decoder.optimize_adjacency_matrix(
                    masked_prob_matrix,
                    target_degrees,
                    forbidden_edge_sets=active_cuts,
                )
            except Exception as exc:
                last_error = exc
                if int(self.verbose) >= 1:
                    verbose_log(
                        self,
                        "Oracle-guided adjacency solve failed with "
                        f"{len(active_cuts)} active cuts at iteration {solve_iteration_idx + 1}/"
                        f"{self.max_oracle_iterations}; retrying with fewer cuts."
                    )
        if last_error is not None:
            raise RuntimeError("Oracle-guided adjacency solve failed even after relaxing all oracle cuts.") from last_error
        raise RuntimeError("Oracle-guided adjacency solve could not be attempted.")

    def _decode_generated_nodes_with_oracle(self, generated_nodes: GeneratedNodeBatch) -> List[nx.Graph]:
        predicted_edge_probability_matrices = generated_nodes.edge_probability_matrices
        if predicted_edge_probability_matrices is None:
            raise RuntimeError(
                "Graph decoding requires explicit edge-probability matrices from the conditional node generator."
            )
        edge_existence_probabilities = generated_nodes.edge_existence_probabilities
        node_label_probabilities = generated_nodes.node_label_probabilities
        edge_label_probabilities = generated_nodes.edge_label_probabilities
        predicted_node_labels_list = self._resolve_predicted_node_labels(generated_nodes)
        predicted_edge_labels_list, predicted_edge_label_matrices = self._resolve_predicted_edge_labels(
            generated_nodes,
            predicted_edge_probability_matrices=predicted_edge_probability_matrices,
        )

        decoded_graphs: List[nx.Graph] = []
        for graph_idx in range(len(predicted_edge_probability_matrices)):
            single_generated_nodes = self._build_single_generated_node_batch(generated_nodes, graph_idx)
            existence_mask = np.asarray(single_generated_nodes.node_presence_mask[0], dtype=bool)
            degree_predictions = np.asarray(single_generated_nodes.node_degree_predictions[0], dtype=float)
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
            single_adj_mtx = self.graph_decoder.decode_adjacency_matrix(
                single_generated_nodes,
                predicted_edge_probability_matrices=[predicted_edge_probability_matrices[graph_idx]],
            )[0]
            accumulated_structural_cuts: List[FrozenSet[Edge]] = []
            accumulated_node_label_forbidden: List[ForbiddenNodeLabelAssignment] = []
            accumulated_edge_label_forbidden: List[ForbiddenEdgeLabelAssignment] = []
            best_graph: Optional[nx.Graph] = None
            best_score = float("-inf")
            best_feasible_graph: Optional[nx.Graph] = None
            best_feasible_score = float("-inf")
            for _iteration_idx in range(self.max_oracle_iterations):
                if predicted_edge_label_matrices is None:
                    edge_labels = self.graph_decoder.decode_edge_labels(
                        single_generated_nodes,
                        [single_adj_mtx],
                        predicted_edge_labels_list=None if predicted_edge_labels_list is None else [
                            predicted_edge_labels_list[graph_idx]
                        ],
                        predicted_edge_label_matrices=None,
                    )[0]
                    current_edge_label_matrix = _edge_label_list_to_matrix(single_adj_mtx, edge_labels)
                edge_labels = _edge_label_matrix_to_list(single_adj_mtx, current_edge_label_matrix)
                graph = _assemble_graph_job(
                    existence_mask,
                    current_node_labels,
                    np.asarray(edge_labels, dtype=object),
                    np.asarray(single_adj_mtx, dtype=float),
                )
                current_node_violation_sets = self._get_oracle_node_violation_sets(graph, n_nodes=single_adj_mtx.shape[0])
                current_edge_violation_sets = self._get_oracle_edge_violation_sets(graph, n_nodes=single_adj_mtx.shape[0])

                new_node_forbidden = [
                    assignment
                    for assignment in self._forbidden_node_label_assignment_from_sets(
                        current_node_violation_sets,
                        current_node_labels,
                    )
                    if assignment not in accumulated_node_label_forbidden
                ]
                if new_node_forbidden:
                    accumulated_node_label_forbidden.extend(new_node_forbidden)
                    repaired_node_labels = self._repair_node_labels_with_oracle(
                        existence_mask=existence_mask,
                        current_node_labels=current_node_labels,
                        node_label_probabilities=None if node_label_probabilities is None else np.asarray(
                            node_label_probabilities[graph_idx],
                            dtype=float,
                        ),
                        forbidden_assignments=accumulated_node_label_forbidden,
                    )
                    if not np.array_equal(repaired_node_labels, current_node_labels):
                        current_node_labels = repaired_node_labels
                        graph = _assemble_graph_job(
                            existence_mask,
                            current_node_labels,
                            np.asarray(_edge_label_matrix_to_list(single_adj_mtx, current_edge_label_matrix), dtype=object),
                            np.asarray(single_adj_mtx, dtype=float),
                        )
                        current_node_violation_sets = self._get_oracle_node_violation_sets(graph, n_nodes=single_adj_mtx.shape[0])
                        current_edge_violation_sets = self._get_oracle_edge_violation_sets(graph, n_nodes=single_adj_mtx.shape[0])

                new_edge_label_forbidden = [
                    assignment
                    for assignment in self._forbidden_edge_label_assignment_from_sets(
                        current_edge_violation_sets,
                        current_edge_label_matrix,
                    )
                    if assignment not in accumulated_edge_label_forbidden
                ]
                if new_edge_label_forbidden:
                    accumulated_edge_label_forbidden.extend(new_edge_label_forbidden)
                    repaired_edge_label_matrix = self._repair_edge_labels_with_oracle(
                        existence_mask=existence_mask,
                        adj_mtx=single_adj_mtx,
                        current_edge_label_matrix=current_edge_label_matrix,
                        edge_label_probabilities=None if edge_label_probabilities is None else np.asarray(
                            edge_label_probabilities[graph_idx],
                            dtype=float,
                        ),
                        forbidden_assignments=accumulated_edge_label_forbidden,
                    )
                    if not np.array_equal(repaired_edge_label_matrix, current_edge_label_matrix):
                        current_edge_label_matrix = repaired_edge_label_matrix
                        graph = _assemble_graph_job(
                            existence_mask,
                            current_node_labels,
                            np.asarray(_edge_label_matrix_to_list(single_adj_mtx, current_edge_label_matrix), dtype=object),
                            np.asarray(single_adj_mtx, dtype=float),
                        )
                        current_node_violation_sets = self._get_oracle_node_violation_sets(graph, n_nodes=single_adj_mtx.shape[0])
                        current_edge_violation_sets = self._get_oracle_edge_violation_sets(graph, n_nodes=single_adj_mtx.shape[0])

                score = self._oracle_candidate_score(
                    existence_mask=existence_mask,
                    adj_mtx=single_adj_mtx,
                    node_labels=current_node_labels,
                    edge_label_matrix=current_edge_label_matrix,
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
                if score > best_score:
                    best_score = score
                    best_graph = graph

                is_feasible = not current_node_violation_sets and not current_edge_violation_sets
                if is_feasible and score > best_feasible_score:
                    best_feasible_score = score
                    best_feasible_graph = graph

                persistent_structural_cuts = [
                    edge_set for edge_set in current_edge_violation_sets if edge_set not in accumulated_structural_cuts
                ]
                if int(self.verbose) >= 4:
                    prob_matrix = np.asarray(predicted_edge_probability_matrices[graph_idx], dtype=float)
                    masked_prob_matrix = _build_masked_prob_matrix(
                        existence_mask=existence_mask,
                        degree_prediction=degree_predictions,
                        prob_matrix=prob_matrix,
                    )
                    target_degrees = self.graph_decoder.get_degrees(degree_predictions, existence_mask)
                    _plot_decoder_diagnostics(
                        prob_matrix=masked_prob_matrix,
                        adj_mtx=single_adj_mtx,
                        target_degrees=target_degrees,
                        title=(
                            f"Oracle decode graph={graph_idx} iteration={_iteration_idx + 1} "
                            f"| violating_node_sets={len(current_node_violation_sets)} "
                            f"| violating_edge_sets={len(current_edge_violation_sets)} "
                            f"| new_node_cuts={len(new_node_forbidden)} "
                            f"| new_edge_label_cuts={len(new_edge_label_forbidden)} "
                            f"| new_structural_cuts={len(persistent_structural_cuts)} "
                            f"| accepted_structural_cuts={len(accumulated_structural_cuts) + len(persistent_structural_cuts)}"
                        ),
                        violating_edge_sets=current_edge_violation_sets,
                        decoded_graph=graph,
                        graph_renderer=self.graph_decoder.diagnostic_graph_renderer,
                    )
                if is_feasible:
                    break
                if not new_node_forbidden and not new_edge_label_forbidden and not persistent_structural_cuts:
                    break
                accumulated_structural_cuts.extend(persistent_structural_cuts)
                if _iteration_idx + 1 >= self.max_oracle_iterations:
                    break
                prob_matrix = np.asarray(predicted_edge_probability_matrices[graph_idx], dtype=float)
                masked_prob_matrix = _build_masked_prob_matrix(
                    existence_mask=existence_mask,
                    degree_prediction=degree_predictions,
                    prob_matrix=prob_matrix,
                )
                target_degrees = self.graph_decoder.get_degrees(degree_predictions, existence_mask)
                single_adj_mtx = self._solve_oracle_relaxed_adjacency(
                    masked_prob_matrix=masked_prob_matrix,
                    target_degrees=target_degrees,
                    accumulated_cuts=accumulated_structural_cuts,
                    start_iteration_idx=_iteration_idx + 1,
                )
            final_graph = best_feasible_graph if best_feasible_graph is not None else best_graph
            if final_graph is None:
                raise RuntimeError("Oracle-guided decoding failed to assemble a graph.")
            decoded_graphs.append(final_graph)
        return decoded_graphs

    def _build_supervision_plan(
        self,
        graphs: List[nx.Graph],
        node_label_targets: List[np.ndarray],
        edge_label_targets: Optional[np.ndarray],
    ) -> SupervisionPlan:
        """Build a single explicit supervision plan for the whole fit() call.

        Args:
            graphs (List[nx.Graph]): Input value.
            node_label_targets (List[np.ndarray]): Input value.
            edge_label_targets (Optional[np.ndarray]): Input value.

        Returns:
            SupervisionPlan: Computed result.
        """
        del graphs

        flat_node_labels = [
            label
            for labels in node_label_targets
            for label in np.asarray(labels, dtype=object).tolist()
        ]
        if len(flat_node_labels) == 0:
            node_label_mode = "disabled"
            node_label_reason = "No node labels were provided."
            node_label_constant = None
        else:
            unique_node_labels = np.unique(np.asarray(flat_node_labels, dtype=object))
            if len(unique_node_labels) == 1:
                node_label_mode = "constant"
                node_label_reason = "All training nodes share one label."
                node_label_constant = unique_node_labels[0]
            else:
                node_label_mode = "learned"
                node_label_reason = f"{len(unique_node_labels)} node labels detected."
                node_label_constant = None

        if edge_label_targets is None:
            edge_label_mode = "disabled"
            edge_label_reason = "No usable edge labels were provided."
            edge_label_constant = None
        else:
            unique_edge_labels = np.unique(np.asarray(edge_label_targets, dtype=object))
            if len(unique_edge_labels) == 1:
                edge_label_mode = "constant"
                edge_label_reason = "All labelled edges share one label."
                edge_label_constant = unique_edge_labels[0]
            else:
                edge_label_mode = "learned"
                edge_label_reason = f"{len(unique_edge_labels)} edge labels detected."
                edge_label_constant = None

        direct_edges_enabled = True
        direct_edges_mode = "learned"
        direct_edges_reason = "Generator should learn horizon-1 edge presence for the decoder."

        aux_enabled = bool(self.locality_horizon > 1)
        if aux_enabled:
            auxiliary_mode = "learned"
            auxiliary_reason = f"Use horizon-{self.locality_horizon} locality as auxiliary regularization."
        else:
            auxiliary_mode = "disabled"
            auxiliary_reason = "No auxiliary locality is needed when locality_horizon=1."

        return SupervisionPlan(
            node_labels=SupervisionChannelPlan(
                name="node_labels",
                mode=node_label_mode,
                reason=node_label_reason,
                constant_value=node_label_constant,
                enabled=node_label_mode != "disabled",
            ),
            edge_labels=SupervisionChannelPlan(
                name="edge_labels",
                mode=edge_label_mode,
                reason=edge_label_reason,
                constant_value=edge_label_constant,
                enabled=edge_label_mode != "disabled",
            ),
            direct_edges=SupervisionChannelPlan(
                name="direct_edges",
                mode=direct_edges_mode,
                reason=direct_edges_reason,
                horizon=1,
                enabled=direct_edges_enabled,
            ),
            auxiliary_locality=SupervisionChannelPlan(
                name="auxiliary_locality",
                mode=auxiliary_mode,
                reason=auxiliary_reason,
                horizon=self.locality_horizon if aux_enabled else None,
                enabled=aux_enabled,
            ),
        )

    def _log_supervision_plan(self, supervision_plan: SupervisionPlan) -> None:
        """Print the current supervision plan when verbose logging is enabled.

        Args:
            supervision_plan (SupervisionPlan): Input value.
        """
        if not self.verbose:
            return
        verbose_log(self, "Supervision plan:")
        for channel in supervision_plan.as_dict().values():
            enabled_text = "enabled" if channel.enabled else "disabled"
            horizon_text = f", horizon={channel.horizon}" if channel.horizon is not None else ""
            verbose_log(self, f"  {channel.name}: mode={channel.mode}, {enabled_text}{horizon_text}. {channel.reason}")

    def _plan_channel(self, channel_name: str) -> Optional[SupervisionChannelPlan]:
        """Return the named supervision channel when a plan is available.

        Args:
            channel_name (str): Input value.

        Returns:
            Optional[SupervisionChannelPlan]: Computed result.
        """
        plan = getattr(self, "supervision_plan_", None)
        if plan is None:
            return None
        return getattr(plan, channel_name, None)

    def _resolve_predicted_node_labels(
        self,
        generated_nodes: GeneratedNodeBatch,
    ) -> List[np.ndarray]:
        """Resolve node labels from explicit predictions or orchestration policy."""
        node_label_plan = self._plan_channel("node_labels")
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

    def _resolve_predicted_edge_labels(
        self,
        generated_nodes: GeneratedNodeBatch,
        predicted_edge_probability_matrices: Optional[List[np.ndarray]],
    ) -> Tuple[Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
        """Resolve edge labels from explicit predictions or orchestration policy."""
        edge_label_plan = self._plan_channel("edge_labels")
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
        """Slice a conditioning batch by row indices.

        Args:
            source (GraphConditioningBatch): Input value.
            indices (np.ndarray): Input value.

        Returns:
            GraphConditioningBatch: Computed result.
        """
        idx = np.asarray(indices, dtype=np.int64)
        return GraphConditioningBatch(
            graph_embeddings=np.asarray(source.graph_embeddings)[idx],
            node_counts=np.asarray(source.node_counts)[idx],
            edge_counts=np.asarray(source.edge_counts)[idx],
        )

    def _interpolated_conditioning_from_pair(
        self,
        conditioning: GraphConditioningBatch,
        first_idx: int,
        second_idx: int,
        t: float,
    ) -> Tuple[np.ndarray, np.int64, np.int64]:
        """Linearly interpolate one conditioning pair and clamp integer counts.

        Args:
            conditioning (GraphConditioningBatch): Input value.
            first_idx (int): Input value.
            second_idx (int): Input value.
            t (float): Input value.

        Returns:
            Tuple[np.ndarray, np.int64, np.int64]: Computed result.
        """
        graph_embeddings = np.asarray(conditioning.graph_embeddings, dtype=float)
        node_counts = np.asarray(conditioning.node_counts, dtype=float)
        edge_counts = np.asarray(conditioning.edge_counts, dtype=float)

        interpolated_embedding = (1.0 - t) * graph_embeddings[first_idx] + t * graph_embeddings[second_idx]
        interpolated_node_count = np.int64(max(1, int(np.rint((1.0 - t) * node_counts[first_idx] + t * node_counts[second_idx]))))
        interpolated_edge_count = np.int64(max(0, int(np.rint((1.0 - t) * edge_counts[first_idx] + t * edge_counts[second_idx]))))
        return interpolated_embedding, interpolated_node_count, interpolated_edge_count

    def _sample_conditions(
        self,
        n_samples: int,
        interpolate_between_n_samples: Optional[int] = None,
    ) -> GraphConditioningBatch:
        """Sample graph-level conditioning from cached training embeddings.

        Args:
            n_samples (int): Input value.
            interpolate_between_n_samples (Optional[int]): Optional input value.

        Returns:
            GraphConditioningBatch: Computed result.
        """
        conditioning = self._require_training_graph_conditioning()
        if interpolate_between_n_samples is not None and int(interpolate_between_n_samples) < 2:
            raise ValueError("interpolate_between_n_samples must be >= 2 when provided.")

        n_training = len(conditioning)
        if interpolate_between_n_samples is None or n_training == 1:
            sample_indices = np.random.choice(n_training, size=int(n_samples), replace=True)
            return self._sample_conditioning_rows(conditioning, sample_indices)

        subset_size = min(int(interpolate_between_n_samples), n_training)
        if subset_size < 2:
            sample_indices = np.random.choice(n_training, size=int(n_samples), replace=True)
            return self._sample_conditioning_rows(conditioning, sample_indices)

        graph_embeddings = np.asarray(conditioning.graph_embeddings, dtype=float)
        sampled_embeddings = []
        sampled_node_counts = []
        sampled_edge_counts = []

        for _ in range(int(n_samples)):
            candidate_indices = np.random.choice(n_training, size=subset_size, replace=False)
            if len(candidate_indices) < 2:
                fallback_idx = int(np.random.choice(n_training))
                direct_conditioning = self._sample_conditioning_rows(conditioning, np.asarray([fallback_idx], dtype=np.int64))
                sampled_embeddings.append(np.asarray(direct_conditioning.graph_embeddings[0], dtype=float))
                sampled_node_counts.append(np.int64(direct_conditioning.node_counts[0]))
                sampled_edge_counts.append(np.int64(direct_conditioning.edge_counts[0]))
                continue

            pair_indices = []
            pair_weights = []
            raw_pair_cosines = []
            for local_i in range(len(candidate_indices)):
                for local_j in range(local_i + 1, len(candidate_indices)):
                    first_idx = int(candidate_indices[local_i])
                    second_idx = int(candidate_indices[local_j])
                    first_embedding = graph_embeddings[first_idx]
                    second_embedding = graph_embeddings[second_idx]
                    denom = float(np.linalg.norm(first_embedding) * np.linalg.norm(second_embedding))
                    cosine = 0.0 if denom == 0.0 else float(np.dot(first_embedding, second_embedding) / denom)
                    pair_indices.append((first_idx, second_idx))
                    raw_pair_cosines.append(cosine)
                    pair_weights.append(max(cosine, 0.0))

            pair_weights_array = np.asarray(pair_weights, dtype=float)
            if np.all(pair_weights_array == 0.0):
                raw_pair_cosines_array = np.asarray(raw_pair_cosines, dtype=float)
                max_cosine = float(np.max(raw_pair_cosines_array))
                candidate_pair_choices = np.flatnonzero(np.isclose(raw_pair_cosines_array, max_cosine))
                pair_choice = int(np.random.choice(candidate_pair_choices))
            else:
                pair_probabilities = pair_weights_array / pair_weights_array.sum()
                pair_choice = int(np.random.choice(len(pair_indices), p=pair_probabilities))
            first_idx, second_idx = pair_indices[pair_choice]
            t = float(np.random.uniform(0.0, 1.0))
            interpolated_embedding, interpolated_node_count, interpolated_edge_count = (
                self._interpolated_conditioning_from_pair(
                    conditioning,
                    first_idx,
                    second_idx,
                    t,
                )
            )
            sampled_embeddings.append(interpolated_embedding)
            sampled_node_counts.append(interpolated_node_count)
            sampled_edge_counts.append(interpolated_edge_count)

        return GraphConditioningBatch(
            graph_embeddings=np.asarray(sampled_embeddings, dtype=float),
            node_counts=np.asarray(sampled_node_counts, dtype=np.int64),
            edge_counts=np.asarray(sampled_edge_counts, dtype=np.int64),
        )

    @timeit
    def fit(
        self,
        graphs: List[nx.Graph],
        train_node_generator: bool = True,
        targets: Optional[Sequence[Any]] = None,
        ckpt_path: Optional[str] = None,
    ) -> 'ConditionalNodeFieldGraphGenerator':
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

        # Fit vectorizers
        self.graph_vectorizer.fit(graphs)
        self.node_graph_vectorizer.fit(graphs)
        if self.feasibility_estimator is not None:
            verbose_log(self, f"Fitting feasibility estimator on {len(graphs)} graphs")
            self.feasibility_estimator.fit(graphs)
        node_label_targets = self.graphs_to_node_label_targets(graphs)
        edge_label_targets, edge_label_pairs = self.graphs_to_edge_label_targets(graphs)
        supervision_plan = self._build_supervision_plan(
            graphs,
            node_label_targets=node_label_targets,
            edge_label_targets=edge_label_targets,
        )
        self.supervision_plan_ = supervision_plan
        if self.conditional_node_generator_model is not None:
            setattr(self.conditional_node_generator_model, "supervision_plan_", supervision_plan)

        node_embeddings_list, graph_conditioning = self.encode(graphs)
        self.training_graph_conditioning_ = GraphConditioningBatch(
            graph_embeddings=np.asarray(graph_conditioning.graph_embeddings),
            node_counts=np.asarray(graph_conditioning.node_counts, dtype=np.int64),
            edge_counts=np.asarray(graph_conditioning.edge_counts, dtype=np.int64),
        )

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
            self.conditional_node_generator_model.fit(
                **fit_kwargs,
            )

        self.is_fitted_ = True
        return self

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

    @timeit
    def node_encode(self, graphs: List[nx.Graph]) -> List[np.ndarray]:
        """Transform graphs into per-node embedding matrices.

        Args:
            graphs (List[nx.Graph]): Input value.

        Returns:
            List[np.ndarray]: Computed result.
        """
        if int(self.verbose) >= 3:
            print(f"Node encoding {len(graphs)} graphs")
        return self.node_graph_vectorizer.transform(graphs)

    @timeit
    def graph_encode(self, graphs: List[nx.Graph]) -> GraphConditioningBatch:
        """Transform graphs into explicit graph-level conditioning signals.

        Args:
            graphs (List[nx.Graph]): Input value.

        Returns:
            GraphConditioningBatch: Computed result.
        """
        if int(self.verbose) >= 3:
            print(f"Encoding {len(graphs)} graphs")
        graph_embeddings = np.asarray(self.graph_vectorizer.transform(graphs))
        node_counts = np.asarray([graph.number_of_nodes() for graph in graphs], dtype=np.int64)
        edge_counts = np.asarray([graph.number_of_edges() for graph in graphs], dtype=np.int64)
        return GraphConditioningBatch(
            graph_embeddings=graph_embeddings,
            node_counts=node_counts,
            edge_counts=edge_counts,
        )

    def encode(self, graphs: List[nx.Graph]) -> Tuple[List[np.ndarray], GraphConditioningBatch]:
        """Produce both node-level embeddings and explicit graph-level conditioning.

        Args:
            graphs (List[nx.Graph]): Input value.

        Returns:
            Tuple[List[np.ndarray], GraphConditioningBatch]: Computed result.
        """
        return self.node_encode(graphs), self.graph_encode(graphs)

    def graphs_to_node_label_targets(self, graphs: List[nx.Graph]) -> List[np.ndarray]:
        """Extract per-node categorical labels in the node ordering used elsewhere.

        Args:
            graphs (List[nx.Graph]): Input value.

        Returns:
            List[np.ndarray]: Computed result.
        """
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
        """Return True only when every observed edge carries a label and at least one labelled edge exists.

        Args:
            graphs (List[nx.Graph]): Input value.

        Returns:
            bool: Computed result.
        """
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
        """Extract per-edge categorical labels in the ordered node-pair convention used elsewhere.

        Args:
            graphs (List[nx.Graph]): Input value.

        Returns:
            Tuple[Optional[np.ndarray], Optional[List[Tuple[int, int, int]]]]: Computed result.
        """
        if not self._graphs_have_usable_edge_labels(graphs):
            if self.verbose:
                print("Edge-label channel disabled at graph inspection time: no usable edge labels were found.")
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
        """Assemble explicit node-level supervision tensors from graphs.

        Args:
            graphs (List[nx.Graph]): Input value.
            node_embeddings_list (List[np.ndarray]): Input value.
            node_label_targets (Optional[List[np.ndarray]]): Optional input value.
            edge_pairs (Optional[List[Tuple[int, int, int]]]): Optional input value.
            edge_targets (Optional[np.ndarray]): Optional input value.
            edge_label_pairs (Optional[List[Tuple[int, int, int]]]): Optional input value.
            edge_label_targets (Optional[np.ndarray]): Optional input value.
            auxiliary_edge_pairs (Optional[List[Tuple[int, int, int]]]): Optional input value.
            auxiliary_edge_targets (Optional[np.ndarray]): Optional input value.

        Returns:
            NodeGenerationBatch: Computed result.
        """
        max_num_rows = max(emb.shape[0] for emb in node_embeddings_list)
        node_presence_mask = np.zeros((len(graphs), max_num_rows), dtype=bool)
        node_degree_targets = np.zeros((len(graphs), max_num_rows), dtype=np.int64)
        for graph_idx, graph in enumerate(graphs):
            nodes = list(graph.nodes())
            node_presence_mask[graph_idx, :len(nodes)] = True
            node_degree_targets[graph_idx, :len(nodes)] = np.asarray(
                [graph.degree(node) for node in nodes],
                dtype=np.int64,
            )
        return NodeGenerationBatch(
            node_embeddings_list=node_embeddings_list,
            node_presence_mask=node_presence_mask,
            node_degree_targets=node_degree_targets,
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
            predicted_node_count = (
                int(np.sum(generated_nodes.node_presence_mask[graph_idx][:node_row_count]))
                if generated_nodes.node_presence_mask is not None
                else node_row_count
            )
            conditioning_node_count = int(graph_conditioning.node_counts[graph_idx])
            conditioning_edge_count = int(graph_conditioning.edge_counts[graph_idx])
            message = (
                f"Generated graph {graph_idx + 1}/{total_graphs}: "
                f"conditioning_nodes={conditioning_node_count}, "
                f"conditioning_edges={conditioning_edge_count}, "
                f"predicted_nodes={predicted_node_count}"
            )
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
            print(message)
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
                    print(f"  node_labels={label_summary}")

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
        return GraphConditioningBatch(
            graph_embeddings=np.asarray(graph_conditioning.graph_embeddings)[idx],
            node_counts=np.asarray(graph_conditioning.node_counts)[idx],
            edge_counts=np.asarray(graph_conditioning.edge_counts)[idx],
        )

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
        return GraphConditioningBatch(
            graph_embeddings=np.repeat(
                np.asarray(graph_conditioning.graph_embeddings),
                repeats,
                axis=0,
            ),
            node_counts=np.repeat(np.asarray(graph_conditioning.node_counts), repeats, axis=0),
            edge_counts=np.repeat(np.asarray(graph_conditioning.edge_counts), repeats, axis=0),
        )

    @staticmethod
    def _accept_feasible_candidates_by_slot(
        decoded_graphs: Sequence[nx.Graph],
        feasibility_mask: Sequence[bool],
        candidate_slot_indices: Sequence[int],
        accepted_graphs_by_slot: List[Optional[nx.Graph]],
    ) -> Tuple[int, int]:
        """Count all feasible candidates, then fill each empty slot with one random feasible graph."""
        feasible_candidates_by_slot: Dict[int, List[nx.Graph]] = {}
        feasible_candidate_count = 0
        for graph, is_feasible, slot_idx in zip(decoded_graphs, feasibility_mask, candidate_slot_indices):
            if not is_feasible:
                continue
            feasible_candidate_count += 1
            if accepted_graphs_by_slot[slot_idx] is None:
                feasible_candidates_by_slot.setdefault(slot_idx, []).append(graph)

        filled_now = 0
        for slot_idx, graphs_for_slot in feasible_candidates_by_slot.items():
            if not graphs_for_slot or accepted_graphs_by_slot[slot_idx] is not None:
                continue
            selected_idx = int(np.random.randint(len(graphs_for_slot)))
            accepted_graphs_by_slot[slot_idx] = graphs_for_slot[selected_idx]
            filled_now += 1
        return feasible_candidate_count, filled_now

    def _decode_generated_nodes(
        self,
        generated_nodes: GeneratedNodeBatch,
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
        attempt_idx: int = 0,
    ) -> List[nx.Graph]:
        if self._can_use_feasibility_oracle(
            feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
            attempt_idx=attempt_idx,
        ):
            return self._decode_generated_nodes_with_oracle(generated_nodes)
        predicted_edge_probability_matrices = generated_nodes.edge_probability_matrices
        if predicted_edge_probability_matrices is None:
            raise RuntimeError(
                "Graph decoding requires explicit edge-probability matrices from the conditional node generator."
            )
        predicted_node_labels_list = self._resolve_predicted_node_labels(generated_nodes)
        predicted_edge_labels_list, predicted_edge_label_matrices = self._resolve_predicted_edge_labels(
            generated_nodes,
            predicted_edge_probability_matrices=predicted_edge_probability_matrices,
        )
        return self.graph_decoder.decode(
            generated_nodes,
            predicted_node_labels_list=predicted_node_labels_list,
            predicted_edge_probability_matrices=predicted_edge_probability_matrices,
            predicted_edge_labels_list=predicted_edge_labels_list,
            predicted_edge_label_matrices=predicted_edge_label_matrices,
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
            graph_conditioning=GraphConditioningBatch(
                graph_embeddings=np.asarray(batch.graph_conditioning.graph_embeddings)[indices_array],
                node_counts=np.asarray(batch.graph_conditioning.node_counts)[indices_array],
                edge_counts=np.asarray(batch.graph_conditioning.edge_counts)[indices_array],
            ),
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
        decoded_graphs = self._decode_generated_nodes(generated_nodes)
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

    def _decode_conditioning_batch(
        self,
        graph_conditioning: GraphConditioningBatch,
        desired_target: Optional[Union[int, float, Sequence[Any]]] = None,
        guidance_scale: float = 1.0,
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
        attempt_idx: int = 0,
    ) -> List[nx.Graph]:
        """Run a single generator pass and decode graphs without feasibility retries.

        Args:
            graph_conditioning (GraphConditioningBatch): Input value.
            desired_target (Optional[Union[int, float, Sequence[Any]]]): Optional input value.
            guidance_scale (float): Optional input value.
        Returns:
            List[nx.Graph]: Computed result.
        """
        if int(self.verbose) >= 3:
            print(f"Predicting node matrices for {len(graph_conditioning)} graphs...")
        generated_nodes = self._predict_generated_nodes(
            graph_conditioning,
            sampling_mode="unguided",
            desired_target=desired_target,
            guidance_scale=guidance_scale,
        )
        return self._decode_generated_nodes(
            generated_nodes,
            feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
            attempt_idx=attempt_idx,
        )

    def _decode_with_feasibility_slots(
        self,
        graph_conditioning: GraphConditioningBatch,
        desired_target: Optional[Union[int, float, Sequence[Any]]] = None,
        guidance_scale: float = 1.0,
        apply_feasibility_filtering: Optional[bool] = None,
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
    ) -> List[Optional[nx.Graph]]:
        """Decode graphs and optionally reject infeasible outputs until the batch is filled.

        Args:
            graph_conditioning (GraphConditioningBatch): Input value.
            desired_target (Optional[Union[int, float, Sequence[Any]]]): Optional input value.
            guidance_scale (float): Optional input value.
            apply_feasibility_filtering (Optional[bool]): Optional input value.

        Returns:
            List[Optional[nx.Graph]]: Computed result.
        """
        use_filtering = (
            self.use_feasibility_filtering
            if apply_feasibility_filtering is None
            else bool(apply_feasibility_filtering)
        )
        if self.feasibility_estimator is None or not use_filtering:
            return list(
                self._decode_conditioning_batch(
                    graph_conditioning,
                    desired_target=desired_target,
                    guidance_scale=guidance_scale,
                    feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
                )
            )

        accepted_graphs_by_slot: List[Optional[nx.Graph]] = [None] * len(graph_conditioning)
        pending_conditioning = graph_conditioning
        pending_slot_indices = list(range(len(graph_conditioning)))
        attempt = 0
        total_generated = 0
        total_feasible = 0
        feasibility_started_at = time.perf_counter()
        while len(pending_conditioning) > 0 and attempt < self.max_feasibility_attempts:
            attempt += 1
            attempt_started_at = time.perf_counter()
            candidate_conditioning = self._repeat_graph_conditioning(
                pending_conditioning,
                repeats=self.feasibility_candidates_per_attempt,
            )
            candidate_slot_indices = [
                slot_idx
                for slot_idx in pending_slot_indices
                for _ in range(self.feasibility_candidates_per_attempt)
            ]
            decoded_graphs = self._decode_conditioning_batch(
                candidate_conditioning,
                desired_target=desired_target,
                guidance_scale=guidance_scale,
                feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
                attempt_idx=attempt - 1,
            )
            total_generated += len(decoded_graphs)
            feasibility_mask = np.asarray(
                self.feasibility_estimator.predict(decoded_graphs),
                dtype=bool,
            )
            if feasibility_mask.shape[0] != len(decoded_graphs):
                raise RuntimeError(
                    "Feasibility estimator returned a mask of unexpected length "
                    f"({feasibility_mask.shape[0]} for {len(decoded_graphs)} graphs)."
                )
            feasible_now, filled_now = self._accept_feasible_candidates_by_slot(
                decoded_graphs=decoded_graphs,
                feasibility_mask=feasibility_mask.tolist(),
                candidate_slot_indices=candidate_slot_indices,
                accepted_graphs_by_slot=accepted_graphs_by_slot,
            )
            total_feasible += feasible_now
            rejected_slot_indices = [
                slot_idx for slot_idx in pending_slot_indices if accepted_graphs_by_slot[slot_idx] is None
            ]
            if int(self.verbose) >= 1:
                pending_now = len(rejected_slot_indices)
                filled_total = sum(graph is not None for graph in accepted_graphs_by_slot)
                missing_total = len(graph_conditioning) - filled_total
                attempted_total = len(decoded_graphs)
                acceptance_rate = (feasible_now / attempted_total) if attempted_total > 0 else 0.0
                attempt_elapsed_seconds = time.perf_counter() - attempt_started_at
                total_elapsed_seconds = time.perf_counter() - feasibility_started_at
                print(
                    _format_feasibility_attempt_status(
                        attempt=attempt,
                        max_attempts=self.max_feasibility_attempts,
                        attempted_total=attempted_total,
                        feasible_now=feasible_now,
                        filled_now=filled_now,
                        pending_now=pending_now,
                        acceptance_rate=acceptance_rate,
                        filled_total=filled_total,
                        missing_total=missing_total,
                        attempt_elapsed_seconds=attempt_elapsed_seconds,
                        total_elapsed_seconds=total_elapsed_seconds,
                    )
                )
            if not rejected_slot_indices:
                break
            pending_slot_indices = rejected_slot_indices
            pending_conditioning = self._slice_graph_conditioning(
                graph_conditioning,
                pending_slot_indices,
            )

        accepted_count = sum(graph is not None for graph in accepted_graphs_by_slot)
        if int(self.verbose) >= 1:
            overall_rate = (total_feasible / total_generated) if total_generated > 0 else 0.0
            print(
                "Feasibility filtering summary: "
                f"generated={total_generated}, feasible_candidates={total_feasible}, "
                f"feasible_rate={overall_rate:.1%}, "
                f"fulfilled_slots={accepted_count}/{len(graph_conditioning)}."
            )
        return accepted_graphs_by_slot

    def _decode_with_feasibility(
        self,
        graph_conditioning: GraphConditioningBatch,
        desired_target: Optional[Union[int, float, Sequence[Any]]] = None,
        guidance_scale: float = 1.0,
        apply_feasibility_filtering: Optional[bool] = None,
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
    ) -> List[nx.Graph]:
        """Decode graphs and optionally reject infeasible outputs until the batch is filled.

        Args:
            graph_conditioning (GraphConditioningBatch): Input value.
            desired_target (Optional[Union[int, float, Sequence[Any]]]): Optional input value.
            guidance_scale (float): Optional input value.
            apply_feasibility_filtering (Optional[bool]): Optional input value.

        Returns:
            List[nx.Graph]: Computed result.
        """
        accepted_graphs_by_slot = self._decode_with_feasibility_slots(
            graph_conditioning,
            desired_target=desired_target,
            guidance_scale=guidance_scale,
            apply_feasibility_filtering=apply_feasibility_filtering,
            feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
        )
        accepted_count = sum(graph is not None for graph in accepted_graphs_by_slot)
        if accepted_count != len(graph_conditioning):
            if self.feasibility_failure_mode == "raise":
                raise RuntimeError(
                    "Feasibility filtering did not recover enough graphs: "
                    f"accepted {accepted_count} of {len(graph_conditioning)} after "
                    f"{self.max_feasibility_attempts} attempts."
                )
            if int(self.verbose) >= 1:
                print(
                    "Feasibility filtering exhausted retries; returning only feasible graphs: "
                    f"accepted {accepted_count} of {len(graph_conditioning)}."
                )
        return [graph for graph in accepted_graphs_by_slot if graph is not None]

    def decode(
        self,
        graph_conditioning: GraphConditioningBatch,
        desired_target: Optional[Union[int, float, Sequence[Any]]] = None,
        guidance_scale: float = 1.0,
        apply_feasibility_filtering: Optional[bool] = None,
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
    ) -> List[nx.Graph]:
        """Decode conditioning vectors into reconstructed graphs.

        Args:
            graph_conditioning (GraphConditioningBatch): Input value.
            desired_target (Optional[Union[int, float, Sequence[Any]]]): Optional input value.
            guidance_scale (float): Optional input value.
            apply_feasibility_filtering (Optional[bool]): Optional input value.

        Returns:
            List[nx.Graph]: Computed result.
        """
        self._require_fitted_for_generation()
        if self.verbose:
            print(f"Decoding {len(graph_conditioning)} conditioning vectors")
            if desired_target is not None:
                print(f"Using CFG target guidance: {desired_target} (scale={guidance_scale})")
        return self._decode_with_feasibility(
            graph_conditioning,
            desired_target=desired_target,
            guidance_scale=guidance_scale,
            apply_feasibility_filtering=apply_feasibility_filtering,
            feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
        )

    def _decode_conditioning_batch_classifier_guided(
        self,
        graph_conditioning: GraphConditioningBatch,
        desired_class: Union[int, Sequence[Any]],
        classifier_scale: float = 1.0,
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
        attempt_idx: int = 0,
    ) -> List[nx.Graph]:
        if int(self.verbose) >= 3:
            print(f"Predicting classifier-guided node matrices for {len(graph_conditioning)} graphs...")
        generated_nodes = self.conditional_node_generator_model.predict_classifier_guided(
            graph_conditioning,
            desired_class=desired_class,
            classifier_scale=classifier_scale,
        )
        self._log_generated_batch_info(graph_conditioning, generated_nodes)
        return self._decode_generated_nodes(
            generated_nodes,
            feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
            attempt_idx=attempt_idx,
        )

    def _decode_with_feasibility_slots_classifier_guided(
        self,
        graph_conditioning: GraphConditioningBatch,
        desired_class: Union[int, Sequence[Any]],
        classifier_scale: float = 1.0,
        apply_feasibility_filtering: Optional[bool] = None,
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
    ) -> List[Optional[nx.Graph]]:
        use_filtering = (
            self.use_feasibility_filtering
            if apply_feasibility_filtering is None
            else bool(apply_feasibility_filtering)
        )
        if self.feasibility_estimator is None or not use_filtering:
            return list(
                self._decode_conditioning_batch_classifier_guided(
                    graph_conditioning,
                    desired_class=desired_class,
                    classifier_scale=classifier_scale,
                    feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
                )
            )

        accepted_graphs_by_slot: List[Optional[nx.Graph]] = [None] * len(graph_conditioning)
        pending_conditioning = graph_conditioning
        pending_slot_indices = list(range(len(graph_conditioning)))
        attempt = 0
        total_generated = 0
        total_feasible = 0
        feasibility_started_at = time.perf_counter()
        while len(pending_conditioning) > 0 and attempt < self.max_feasibility_attempts:
            attempt += 1
            attempt_started_at = time.perf_counter()
            candidate_conditioning = self._repeat_graph_conditioning(
                pending_conditioning,
                repeats=self.feasibility_candidates_per_attempt,
            )
            candidate_slot_indices = [
                slot_idx
                for slot_idx in pending_slot_indices
                for _ in range(self.feasibility_candidates_per_attempt)
            ]
            decoded_graphs = self._decode_conditioning_batch_classifier_guided(
                candidate_conditioning,
                desired_class=desired_class,
                classifier_scale=classifier_scale,
                feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
                attempt_idx=attempt - 1,
            )
            total_generated += len(decoded_graphs)
            feasibility_mask = np.asarray(
                self.feasibility_estimator.predict(decoded_graphs),
                dtype=bool,
            )
            if feasibility_mask.shape[0] != len(decoded_graphs):
                raise RuntimeError(
                    "Feasibility estimator returned a mask of unexpected length "
                    f"({feasibility_mask.shape[0]} for {len(decoded_graphs)} graphs)."
                )
            feasible_now, filled_now = self._accept_feasible_candidates_by_slot(
                decoded_graphs=decoded_graphs,
                feasibility_mask=feasibility_mask.tolist(),
                candidate_slot_indices=candidate_slot_indices,
                accepted_graphs_by_slot=accepted_graphs_by_slot,
            )
            total_feasible += feasible_now
            rejected_slot_indices = [
                slot_idx for slot_idx in pending_slot_indices if accepted_graphs_by_slot[slot_idx] is None
            ]
            if int(self.verbose) >= 1:
                pending_now = len(rejected_slot_indices)
                filled_total = sum(graph is not None for graph in accepted_graphs_by_slot)
                missing_total = len(graph_conditioning) - filled_total
                attempted_total = len(decoded_graphs)
                acceptance_rate = (feasible_now / attempted_total) if attempted_total > 0 else 0.0
                attempt_elapsed_seconds = time.perf_counter() - attempt_started_at
                total_elapsed_seconds = time.perf_counter() - feasibility_started_at
                print(
                    _format_feasibility_attempt_status(
                        attempt=attempt,
                        max_attempts=self.max_feasibility_attempts,
                        attempted_total=attempted_total,
                        feasible_now=feasible_now,
                        filled_now=filled_now,
                        pending_now=pending_now,
                        acceptance_rate=acceptance_rate,
                        filled_total=filled_total,
                        missing_total=missing_total,
                        attempt_elapsed_seconds=attempt_elapsed_seconds,
                        total_elapsed_seconds=total_elapsed_seconds,
                    )
                )
            if not rejected_slot_indices:
                break
            pending_slot_indices = rejected_slot_indices
            pending_conditioning = self._slice_graph_conditioning(
                graph_conditioning,
                pending_slot_indices,
            )

        accepted_count = sum(graph is not None for graph in accepted_graphs_by_slot)
        if int(self.verbose) >= 1:
            overall_rate = (total_feasible / total_generated) if total_generated > 0 else 0.0
            print(
                "Feasibility filtering summary: "
                f"generated={total_generated}, feasible_candidates={total_feasible}, "
                f"feasible_rate={overall_rate:.1%}, "
                f"fulfilled_slots={accepted_count}/{len(graph_conditioning)}."
            )
        return accepted_graphs_by_slot

    def decode_classifier_guided(
        self,
        graph_conditioning: GraphConditioningBatch,
        desired_class: Union[int, Sequence[Any]],
        classifier_scale: float = 1.0,
        apply_feasibility_filtering: Optional[bool] = None,
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
    ) -> List[nx.Graph]:
        self._require_fitted_for_generation()
        if self.verbose:
            print(f"Decoding {len(graph_conditioning)} conditioning vectors")
            print(f"Using classifier guidance toward class(es): {desired_class} (scale={classifier_scale})")
        accepted_graphs_by_slot = self._decode_with_feasibility_slots_classifier_guided(
            graph_conditioning,
            desired_class=desired_class,
            classifier_scale=classifier_scale,
            apply_feasibility_filtering=apply_feasibility_filtering,
            feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
        )
        accepted_count = sum(graph is not None for graph in accepted_graphs_by_slot)
        if accepted_count != len(graph_conditioning):
            if self.feasibility_failure_mode == "raise":
                raise RuntimeError(
                    "Feasibility filtering did not recover enough graphs: "
                    f"accepted {accepted_count} of {len(graph_conditioning)} after "
                    f"{self.max_feasibility_attempts} attempts."
                )
            if int(self.verbose) >= 1:
                print(
                    "Feasibility filtering exhausted retries; returning only feasible graphs: "
                    f"accepted {accepted_count} of {len(graph_conditioning)}."
                )
        return [graph for graph in accepted_graphs_by_slot if graph is not None]

    def _decode_conditioning_batch_regression_guided(
        self,
        graph_conditioning: GraphConditioningBatch,
        desired_target: Union[float, Sequence[Any]],
        predictor_scale: float = 1.0,
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
        attempt_idx: int = 0,
    ) -> List[nx.Graph]:
        if int(self.verbose) >= 3:
            print(f"Predicting regression-guided node matrices for {len(graph_conditioning)} graphs...")
        generated_nodes = self._predict_generated_nodes(
            graph_conditioning,
            sampling_mode="regression_guided",
            desired_target=desired_target,
            predictor_scale=predictor_scale,
        )
        return self._decode_generated_nodes(
            generated_nodes,
            feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
            attempt_idx=attempt_idx,
        )

    def _decode_with_feasibility_slots_regression_guided(
        self,
        graph_conditioning: GraphConditioningBatch,
        desired_target: Union[float, Sequence[Any]],
        predictor_scale: float = 1.0,
        apply_feasibility_filtering: Optional[bool] = None,
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
    ) -> List[Optional[nx.Graph]]:
        use_filtering = (
            self.use_feasibility_filtering
            if apply_feasibility_filtering is None
            else bool(apply_feasibility_filtering)
        )
        if self.feasibility_estimator is None or not use_filtering:
            return list(
                self._decode_conditioning_batch_regression_guided(
                    graph_conditioning,
                    desired_target=desired_target,
                    predictor_scale=predictor_scale,
                    feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
                )
            )

        accepted_graphs_by_slot: List[Optional[nx.Graph]] = [None] * len(graph_conditioning)
        pending_conditioning = graph_conditioning
        pending_slot_indices = list(range(len(graph_conditioning)))
        attempt = 0
        total_generated = 0
        total_feasible = 0
        feasibility_started_at = time.perf_counter()
        while len(pending_conditioning) > 0 and attempt < self.max_feasibility_attempts:
            attempt += 1
            attempt_started_at = time.perf_counter()
            candidate_conditioning = self._repeat_graph_conditioning(
                pending_conditioning,
                repeats=self.feasibility_candidates_per_attempt,
            )
            candidate_slot_indices = [
                slot_idx
                for slot_idx in pending_slot_indices
                for _ in range(self.feasibility_candidates_per_attempt)
            ]
            decoded_graphs = self._decode_conditioning_batch_regression_guided(
                candidate_conditioning,
                desired_target=desired_target,
                predictor_scale=predictor_scale,
                feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
                attempt_idx=attempt - 1,
            )
            total_generated += len(decoded_graphs)
            feasibility_mask = np.asarray(
                self.feasibility_estimator.predict(decoded_graphs),
                dtype=bool,
            )
            if feasibility_mask.shape[0] != len(decoded_graphs):
                raise RuntimeError(
                    "Feasibility estimator returned a mask of unexpected length "
                    f"({feasibility_mask.shape[0]} for {len(decoded_graphs)} graphs)."
                )
            feasible_now, filled_now = self._accept_feasible_candidates_by_slot(
                decoded_graphs=decoded_graphs,
                feasibility_mask=feasibility_mask.tolist(),
                candidate_slot_indices=candidate_slot_indices,
                accepted_graphs_by_slot=accepted_graphs_by_slot,
            )
            total_feasible += feasible_now
            rejected_slot_indices = [
                slot_idx for slot_idx in pending_slot_indices if accepted_graphs_by_slot[slot_idx] is None
            ]
            if int(self.verbose) >= 1:
                pending_now = len(rejected_slot_indices)
                filled_total = sum(graph is not None for graph in accepted_graphs_by_slot)
                missing_total = len(graph_conditioning) - filled_total
                attempted_total = len(decoded_graphs)
                acceptance_rate = (feasible_now / attempted_total) if attempted_total > 0 else 0.0
                attempt_elapsed_seconds = time.perf_counter() - attempt_started_at
                total_elapsed_seconds = time.perf_counter() - feasibility_started_at
                print(
                    _format_feasibility_attempt_status(
                        attempt=attempt,
                        max_attempts=self.max_feasibility_attempts,
                        attempted_total=attempted_total,
                        feasible_now=feasible_now,
                        filled_now=filled_now,
                        pending_now=pending_now,
                        acceptance_rate=acceptance_rate,
                        filled_total=filled_total,
                        missing_total=missing_total,
                        attempt_elapsed_seconds=attempt_elapsed_seconds,
                        total_elapsed_seconds=total_elapsed_seconds,
                    )
                )
            if not rejected_slot_indices:
                break
            pending_slot_indices = rejected_slot_indices
            pending_conditioning = self._slice_graph_conditioning(
                graph_conditioning,
                pending_slot_indices,
            )

        accepted_count = sum(graph is not None for graph in accepted_graphs_by_slot)
        if int(self.verbose) >= 1:
            overall_rate = (total_feasible / total_generated) if total_generated > 0 else 0.0
            print(
                "Feasibility filtering summary: "
                f"generated={total_generated}, feasible_candidates={total_feasible}, "
                f"feasible_rate={overall_rate:.1%}, "
                f"fulfilled_slots={accepted_count}/{len(graph_conditioning)}."
            )
        return accepted_graphs_by_slot

    def decode_regression_guided(
        self,
        graph_conditioning: GraphConditioningBatch,
        desired_target: Union[float, Sequence[Any]],
        predictor_scale: float = 1.0,
        apply_feasibility_filtering: Optional[bool] = None,
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
    ) -> List[nx.Graph]:
        self._require_fitted_for_generation()
        if self.verbose:
            print(f"Decoding {len(graph_conditioning)} conditioning vectors")
            print(f"Using regression guidance toward target(s): {desired_target} (scale={predictor_scale})")
        accepted_graphs_by_slot = self._decode_with_feasibility_slots_regression_guided(
            graph_conditioning,
            desired_target=desired_target,
            predictor_scale=predictor_scale,
            apply_feasibility_filtering=apply_feasibility_filtering,
            feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
        )
        accepted_count = sum(graph is not None for graph in accepted_graphs_by_slot)
        if accepted_count != len(graph_conditioning):
            if self.feasibility_failure_mode == "raise":
                raise RuntimeError(
                    "Feasibility filtering did not recover enough graphs: "
                    f"accepted {accepted_count} of {len(graph_conditioning)} after "
                    f"{self.max_feasibility_attempts} attempts."
                )
            if int(self.verbose) >= 1:
                print(
                    "Feasibility filtering exhausted retries; returning only feasible graphs: "
                    f"accepted {accepted_count} of {len(graph_conditioning)}."
                )
        return [graph for graph in accepted_graphs_by_slot if graph is not None]

    @timeit
    def sample(
        self,
        n_samples: int = 1,
        interpolate_between_n_samples: Optional[int] = None,
        desired_target: Optional[Union[int, float, Sequence[Any]]] = None,
        guidance_scale: float = 1.0,
        apply_feasibility_filtering: Optional[bool] = None,
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
    ) -> List[nx.Graph]:
        """Generate random graphs by sampling conditioning vectors from the prior.

        Args:
            n_samples (int): Optional input value.
            interpolate_between_n_samples (Optional[int]): Optional input value.
            desired_target (Optional[Union[int, float, Sequence[Any]]]): Optional input value.
            guidance_scale (float): Optional input value.
            apply_feasibility_filtering (Optional[bool]): Optional input value.

        Returns:
            List[nx.Graph]: Computed result.
        """
        self._require_fitted_for_generation()
        if self.verbose:
            print(f"Sampling {n_samples} graphs")
            if interpolate_between_n_samples is not None:
                print(
                    "Sampling conditioning via stochastic interpolation over "
                    f"{interpolate_between_n_samples} cached training embeddings per output."
                )
            if desired_target is not None:
                print(f"Using CFG target guidance: {desired_target} (scale={guidance_scale})")
        sampled_conditioning = self._sample_conditions(
            n_samples,
            interpolate_between_n_samples=interpolate_between_n_samples,
        )
        return self._decode_with_feasibility(
            sampled_conditioning,
            desired_target=desired_target,
            guidance_scale=guidance_scale,
            apply_feasibility_filtering=apply_feasibility_filtering,
            feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
        )

    def score_feasible_rate(
        self,
        n_samples: int = 32,
        max_feasibility_attempts: Optional[int] = None,
        feasibility_candidates_per_attempt: Optional[int] = None,
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
        interpolate_between_n_samples: Optional[int] = None,
        desired_target: Optional[Union[int, float, Sequence[Any]]] = None,
        guidance_scale: float = 1.0,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Score generation quality using the fraction of feasible decoded candidates."""
        if int(n_samples) < 1:
            raise ValueError("n_samples must be >= 1")

        self._require_fitted_for_generation()

        original_attempts = self.max_feasibility_attempts
        original_candidates = self.feasibility_candidates_per_attempt
        original_oracle_candidates = self.feasibility_oracle_candidates_per_attempt
        original_verbose = self.verbose

        if max_feasibility_attempts is not None:
            self.max_feasibility_attempts = int(max_feasibility_attempts)
        if feasibility_candidates_per_attempt is not None:
            self.feasibility_candidates_per_attempt = int(feasibility_candidates_per_attempt)
        if feasibility_oracle_candidates_per_attempt is not None:
            self.feasibility_oracle_candidates_per_attempt = int(feasibility_oracle_candidates_per_attempt)
        self.verbose = original_verbose if verbose else 0

        try:
            graph_conditioning = self._sample_conditions(
                int(n_samples),
                interpolate_between_n_samples=interpolate_between_n_samples,
            )

            if self.feasibility_estimator is None:
                decoded_graphs = self._decode_conditioning_batch(
                    graph_conditioning,
                    desired_target=desired_target,
                    guidance_scale=guidance_scale,
                )
                accepted_slots = len(decoded_graphs)
                total_generated = len(decoded_graphs)
                total_feasible = len(decoded_graphs)
            elif not self.use_feasibility_filtering:
                decoded_graphs = self._decode_conditioning_batch(
                    graph_conditioning,
                    desired_target=desired_target,
                    guidance_scale=guidance_scale,
                )
                feasibility_mask = np.asarray(self.feasibility_estimator.predict(decoded_graphs), dtype=bool)
                if feasibility_mask.shape[0] != len(decoded_graphs):
                    raise RuntimeError(
                        "Feasibility estimator returned a mask of unexpected length "
                        f"({feasibility_mask.shape[0]} for {len(decoded_graphs)} graphs)."
                    )
                accepted_slots = len(decoded_graphs)
                total_generated = len(decoded_graphs)
                total_feasible = int(feasibility_mask.sum())
            else:
                accepted_graphs_by_slot: List[Optional[nx.Graph]] = [None] * len(graph_conditioning)
                pending_conditioning = graph_conditioning
                pending_slot_indices = list(range(len(graph_conditioning)))
                total_generated = 0
                total_feasible = 0
                attempt = 0

                while len(pending_conditioning) > 0 and attempt < self.max_feasibility_attempts:
                    attempt += 1
                    candidate_conditioning = self._repeat_graph_conditioning(
                        pending_conditioning,
                        repeats=self.feasibility_candidates_per_attempt,
                    )
                    candidate_slot_indices = [
                        slot_idx
                        for slot_idx in pending_slot_indices
                        for _ in range(self.feasibility_candidates_per_attempt)
                    ]
                    decoded_graphs = self._decode_conditioning_batch(
                        candidate_conditioning,
                        desired_target=desired_target,
                        guidance_scale=guidance_scale,
                        attempt_idx=attempt - 1,
                    )
                    feasibility_mask = np.asarray(self.feasibility_estimator.predict(decoded_graphs), dtype=bool)
                    if feasibility_mask.shape[0] != len(decoded_graphs):
                        raise RuntimeError(
                            "Feasibility estimator returned a mask of unexpected length "
                            f"({feasibility_mask.shape[0]} for {len(decoded_graphs)} graphs)."
                        )
                    total_generated += len(decoded_graphs)
                    feasible_now, _ = self._accept_feasible_candidates_by_slot(
                        decoded_graphs=decoded_graphs,
                        feasibility_mask=feasibility_mask.tolist(),
                        candidate_slot_indices=candidate_slot_indices,
                        accepted_graphs_by_slot=accepted_graphs_by_slot,
                    )
                    total_feasible += feasible_now
                    pending_slot_indices = [
                        slot_idx for slot_idx in pending_slot_indices if accepted_graphs_by_slot[slot_idx] is None
                    ]
                    if pending_slot_indices:
                        pending_conditioning = self._slice_graph_conditioning(
                            graph_conditioning,
                            pending_slot_indices,
                        )
                accepted_slots = sum(graph is not None for graph in accepted_graphs_by_slot)

            feasible_rate = (total_feasible / total_generated) if total_generated > 0 else 0.0
            fulfilled_rate = accepted_slots / int(n_samples)
            return {
                "score": feasible_rate,
                "feasible_rate": feasible_rate,
                "fulfilled_rate": fulfilled_rate,
                "accepted_slots": accepted_slots,
                "n_samples": int(n_samples),
                "generated_candidates": total_generated,
                "feasible_candidates": total_feasible,
                "max_feasibility_attempts": int(self.max_feasibility_attempts),
                "feasibility_candidates_per_attempt": int(self.feasibility_candidates_per_attempt),
                "feasibility_oracle_candidates_per_attempt": int(self.feasibility_oracle_candidates_per_attempt),
            }
        finally:
            self.max_feasibility_attempts = original_attempts
            self.feasibility_candidates_per_attempt = original_candidates
            self.feasibility_oracle_candidates_per_attempt = original_oracle_candidates
            self.verbose = original_verbose

    @timeit
    def conditional_sample(
        self,
        graphs: List[nx.Graph],
        n_samples: int = 1,
        desired_target: Optional[Union[int, float, Sequence[Any]]] = None,
        guidance_scale: float = 1.0,
        apply_feasibility_filtering: Optional[bool] = None,
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
    ) -> List[List[nx.Graph]]:
        """Sample multiple graphs per input by conditioning on each graph's encoding.

        Args:
            graphs (List[nx.Graph]): Input value.
            n_samples (int): Optional input value.
            desired_target (Optional[Union[int, float, Sequence[Any]]]): Optional input value.
            guidance_scale (float): Optional input value.
            apply_feasibility_filtering (Optional[bool]): Optional input value.

        Returns:
            List[List[nx.Graph]]: Computed result.
        """
        self._require_fitted_for_generation()
        _, graph_conditioning = self.encode(graphs)
        repeated_conditioning = self._repeat_graph_conditioning(
            graph_conditioning,
            repeats=n_samples,
        )
        decoded_slots = self._decode_with_feasibility_slots(
            repeated_conditioning,
            desired_target=desired_target,
            guidance_scale=guidance_scale,
            apply_feasibility_filtering=apply_feasibility_filtering,
            feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
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
    ) -> List[nx.Graph]:
        self._require_fitted_for_generation()
        if self.verbose:
            print(f"Sampling {n_samples} graphs")
            if interpolate_between_n_samples is not None:
                print(
                    "Sampling conditioning via stochastic interpolation over "
                    f"{interpolate_between_n_samples} cached training embeddings per output."
                )
            print(f"Using classifier guidance toward class(es): {desired_class} (scale={classifier_scale})")
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
    ) -> List[List[nx.Graph]]:
        self._require_fitted_for_generation()
        _, graph_conditioning = self.encode(graphs)
        repeated_conditioning = self._repeat_graph_conditioning(
            graph_conditioning,
            repeats=n_samples,
        )
        decoded_slots = self._decode_with_feasibility_slots_classifier_guided(
            repeated_conditioning,
            desired_class=desired_class,
            classifier_scale=classifier_scale,
            apply_feasibility_filtering=apply_feasibility_filtering,
            feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
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
    ) -> List[nx.Graph]:
        self._require_fitted_for_generation()
        if self.verbose:
            print(f"Sampling {n_samples} graphs")
            if interpolate_between_n_samples is not None:
                print(
                    "Sampling conditioning via stochastic interpolation over "
                    f"{interpolate_between_n_samples} cached training embeddings per output."
                )
            print(f"Using regression guidance toward target(s): {desired_target} (scale={predictor_scale})")
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
    ) -> List[List[nx.Graph]]:
        self._require_fitted_for_generation()
        _, graph_conditioning = self.encode(graphs)
        repeated_conditioning = self._repeat_graph_conditioning(
            graph_conditioning,
            repeats=n_samples,
        )
        decoded_slots = self._decode_with_feasibility_slots_regression_guided(
            repeated_conditioning,
            desired_target=desired_target,
            predictor_scale=predictor_scale,
            apply_feasibility_filtering=apply_feasibility_filtering,
            feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
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
        decoded_slots = self._decode_with_feasibility_slots(
            interpolated_conditioning,
            apply_feasibility_filtering=apply_feasibility_filtering,
            feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
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
        )[0]
