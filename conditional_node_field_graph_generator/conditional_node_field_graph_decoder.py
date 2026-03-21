"""Decoder helpers for rebuilding labeled graphs from node-field predictions."""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import io
import os
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import dill as pickle
import networkx as nx
import numpy as np
import pulp

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

from .conditional_node_field_generator import GeneratedNodeBatch

Edge = Tuple[int, int]
_DECODER_PROBABILITY_EPS = 1e-6


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
) -> List[frozenset[Edge]]:
    normalized: List[frozenset[Edge]] = []
    seen: set[frozenset[Edge]] = set()
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

    im0 = axes[0].imshow(
        prob_matrix,
        vmin=0.0,
        vmax=max(1.0, float(np.max(prob_matrix)) if prob_matrix.size else 1.0),
        cmap="viridis",
    )
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

    def get_degrees(self, node_degree_predictions: np.ndarray, node_presence_mask: np.ndarray) -> List[int]:
        degrees = np.rint(np.asarray(node_degree_predictions, dtype=float)).astype(np.int64)
        mask = np.asarray(node_presence_mask, dtype=bool)
        return [int(degrees[idx]) if mask[idx] else 0 for idx in range(len(mask))]

    def decode_adjacency_matrix(
        self,
        generated_nodes: GeneratedNodeBatch,
        predicted_edge_probability_matrices: Optional[List[np.ndarray]] = None,
    ) -> List[np.ndarray]:
        if generated_nodes.node_presence_mask is None:
            raise RuntimeError("decode_adjacency_matrix requires node presence predictions.")
        if generated_nodes.node_degree_predictions is None:
            raise RuntimeError("decode_adjacency_matrix requires node degree predictions.")
        if predicted_edge_probability_matrices is None:
            raise RuntimeError("decode_adjacency_matrix requires explicit edge probability matrices.")

        existence_masks = np.asarray(generated_nodes.node_presence_mask, dtype=bool)
        degree_predictions = np.asarray(generated_nodes.node_degree_predictions, dtype=float)
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
    ) -> List[nx.Graph]:
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

    def save(self, filename: str = "generative_model.obj") -> None:
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def load(self, filename: str = "generative_model.obj") -> "ConditionalNodeFieldGraphDecoder":
        with open(filename, "rb") as f:
            self = pickle.load(f)
        return self


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
