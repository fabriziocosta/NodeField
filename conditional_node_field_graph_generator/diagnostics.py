"""Shared graph-decoder diagnostics helpers."""

import io
from typing import Any, Callable, Iterable, Optional, Sequence

import networkx as nx
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

from .graph_decode_utils import _normalize_violating_edge_sets


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
        from abstractgraph_graphicalizer.chem import draw_molecule
    except Exception:
        return False
    try:
        image = draw_molecule(decoded_graph, size=(500, 350))
    except Exception:
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
    node_label_probabilities: Optional[np.ndarray] = None,
    node_label_names: Optional[Sequence[Any]] = None,
    node_labels: Optional[Sequence[Any]] = None,
    existence_mask: Optional[Sequence[bool]] = None,
    plot_backend: Optional[Any] = None,
    inline_renderer: Optional[Callable[..., bool]] = None,
) -> None:
    plot_backend = plt if plot_backend is None else plot_backend
    inline_renderer = _try_render_molecular_graph_inline if inline_renderer is None else inline_renderer
    if plot_backend is None:
        return

    def _format_plot_title(value: str) -> str:
        if " | " not in value or not value.startswith("Oracle "):
            return value
        parts = value.split(" | ")
        head = parts[0]
        groups = [[], [], []]
        for metric in parts[1:]:
            key = metric.split("=", 1)[0].strip()
            if key in {
                "violating_node_sets",
                "violating_edge_sets",
                "new_node_cuts",
                "new_edge_label_cuts",
                "joint_label_changed",
            }:
                groups[0].append(metric)
            elif key in {
                "log_total",
                "log_edge",
                "log_node",
                "log_edge_label",
            }:
                groups[1].append(metric)
            else:
                groups[2].append(metric)
        return "\n".join([head] + [" | ".join(group) for group in groups if group])

    formatted_title = _format_plot_title(title)
    prob_matrix = np.asarray(prob_matrix, dtype=float)
    adj_mtx = np.asarray(adj_mtx, dtype=float)
    target_degrees = np.asarray(target_degrees, dtype=float)
    active_mask = None if existence_mask is None else np.asarray(existence_mask, dtype=bool)
    if active_mask is not None and len(active_mask) == adj_mtx.shape[0]:
        active_indices = np.flatnonzero(active_mask)
    else:
        active_mask = None
        active_indices = np.arange(adj_mtx.shape[0], dtype=int)
    prob_display = prob_matrix[np.ix_(active_indices, active_indices)]
    adj_display = adj_mtx[np.ix_(active_indices, active_indices)]
    target_degrees_display = target_degrees[active_indices]
    realized_degrees = adj_display.sum(axis=1)
    normalized_violations = _normalize_violating_edge_sets(
        [] if violating_edge_sets is None else violating_edge_sets,
        n_nodes=adj_mtx.shape[0],
    )

    has_node_label_panel = node_label_probabilities is not None
    n_panels = 5 if has_node_label_panel else 4
    fig, axes = plot_backend.subplots(1, n_panels, figsize=(24 if has_node_label_panel else 20, 4.8))

    im0 = axes[0].imshow(
        prob_display,
        vmin=0.0,
        vmax=max(1.0, float(np.max(prob_display)) if prob_display.size else 1.0),
        cmap="viridis",
    )
    axes[0].set_title("Edge probabilities")
    axes[0].set_xlabel("node j")
    axes[0].set_ylabel("node i")
    if active_indices.size > 0:
        axes[0].set_xticks(np.arange(len(active_indices)))
        axes[0].set_xticklabels(active_indices.astype(int).tolist())
        axes[0].set_yticks(np.arange(len(active_indices)))
        axes[0].set_yticklabels(active_indices.astype(int).tolist())
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].imshow(adj_display, vmin=0.0, vmax=1.0, cmap="gray")
    axes[1].set_title("Decoded adjacency")
    axes[1].set_xlabel("node j")
    axes[1].set_ylabel("node i")
    if active_indices.size > 0:
        axes[1].set_xticks(np.arange(len(active_indices)))
        axes[1].set_xticklabels(active_indices.astype(int).tolist())
        axes[1].set_yticks(np.arange(len(active_indices)))
        axes[1].set_yticklabels(active_indices.astype(int).tolist())
    active_index_lookup = {int(node_idx): pos for pos, node_idx in enumerate(active_indices.tolist())}
    if normalized_violations:
        for edge_set in normalized_violations:
            for i, j in edge_set:
                if i not in active_index_lookup or j not in active_index_lookup:
                    continue
                ii = active_index_lookup[i]
                jj = active_index_lookup[j]
                axes[1].plot([jj, ii], [ii, jj], marker="s", color="tab:red", markersize=6, linewidth=1.5)
                axes[1].plot([ii, jj], [jj, ii], marker="s", color="tab:red", markersize=6, linewidth=1.5)

    node_idx = active_indices.astype(int)
    axes[2].bar(node_idx - 0.18, target_degrees_display, width=0.36, label="target")
    axes[2].bar(node_idx + 0.18, realized_degrees, width=0.36, label="realized")
    axes[2].set_title("Degree targets vs realized")
    axes[2].set_xlabel("node")
    axes[2].set_ylabel("degree")
    if node_idx.size > 0:
        axes[2].set_xticks(node_idx)
        y_max = int(max(np.max(target_degrees_display), np.max(realized_degrees)))
        axes[2].set_yticks(np.arange(0, y_max + 1, 1))
    axes[2].grid(axis="y", alpha=0.3)
    axes[2].legend()

    graph_axis = axes[4] if has_node_label_panel else axes[3]
    graph = nx.from_numpy_array(adj_display.astype(int))
    violating_edges = {
        (min(active_index_lookup[i], active_index_lookup[j]), max(active_index_lookup[i], active_index_lookup[j]))
        for edge_set in normalized_violations
        for i, j in edge_set
        if i in active_index_lookup and j in active_index_lookup
    }
    rendered_inline = False
    if decoded_graph is not None:
        rendered_inline = inline_renderer(
            graph_axis,
            decoded_graph=decoded_graph,
            title=formatted_title,
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
            ax=graph_axis,
            node_color="white",
            edge_color=edge_colors,
            width=edge_widths,
            with_labels=True,
            font_size=9,
            node_size=500,
            linewidths=1.5,
        )
        graph_axis.set_title("Decoded graph")
    elif not rendered_inline:
        graph_axis.text(
            0.5,
            0.5,
            "Custom graph renderer\nshown below",
            ha="center",
            va="center",
            fontsize=11,
        )
        graph_axis.set_title("Decoded graph")
    graph_axis.set_axis_off()

    if has_node_label_panel:
        label_probs = np.asarray(node_label_probabilities, dtype=float)
        if active_mask is not None and len(active_mask) == label_probs.shape[0]:
            label_probs = label_probs.copy()
            label_probs[~active_mask, :] = 0.0
        heatmap_axis = axes[3]
        im3 = heatmap_axis.imshow(label_probs, vmin=0.0, vmax=1.0, cmap="magma", aspect="auto")
        heatmap_axis.set_title("Node-label probabilities\nrows=node ids, cols=labels")
        heatmap_axis.set_xlabel("decoded node label")
        heatmap_axis.set_ylabel("node id")
        label_names: Optional[list[str]] = None
        if node_label_names is not None:
            label_names = [str(label) for label in node_label_names]
            if len(label_names) == label_probs.shape[1]:
                heatmap_axis.set_xticks(np.arange(len(label_names)))
                heatmap_axis.set_xticklabels(label_names, rotation=0, ha="center")
            else:
                label_names = None
        if label_names is None:
            heatmap_axis.set_xticks(np.arange(label_probs.shape[1]))
            heatmap_axis.set_xticklabels([str(idx) for idx in range(label_probs.shape[1])], rotation=0, ha="center")
        node_ids = np.arange(label_probs.shape[0])
        heatmap_axis.set_yticks(node_ids)
        heatmap_axis.set_yticklabels([str(idx) for idx in node_ids])
        chosen_cols: list[Optional[int]] = [None] * label_probs.shape[0]
        if node_labels is not None and len(node_labels) == label_probs.shape[0]:
            label_to_col = None
            if node_label_names is not None and len(node_label_names) == label_probs.shape[1]:
                label_to_col = {label: idx for idx, label in enumerate(node_label_names)}
            for row_idx, label in enumerate(node_labels):
                if active_mask is not None and row_idx < len(active_mask) and not bool(active_mask[row_idx]):
                    continue
                col_idx: Optional[int] = None
                if label_to_col is not None:
                    col_idx = label_to_col.get(label)
                if col_idx is None:
                    try:
                        numeric_label = int(label)
                    except (TypeError, ValueError):
                        numeric_label = None
                    if numeric_label is not None and 0 <= numeric_label < label_probs.shape[1]:
                        col_idx = numeric_label
                if col_idx is None and row_idx < label_probs.shape[0] and label_probs.shape[1] > 0:
                    col_idx = int(np.argmax(label_probs[row_idx]))
                chosen_cols[row_idx] = col_idx
        else:
            for row_idx in range(label_probs.shape[0]):
                if active_mask is not None and row_idx < len(active_mask) and not bool(active_mask[row_idx]):
                    continue
                if label_probs.shape[1] > 0:
                    chosen_cols[row_idx] = int(np.argmax(label_probs[row_idx]))
        for row_idx, col_idx in enumerate(chosen_cols):
            if col_idx is None:
                continue
            heatmap_axis.scatter(
                [col_idx],
                [row_idx],
                marker="o",
                s=64,
                facecolors="white",
                color="black",
                linewidths=1.5,
                zorder=3,
            )
        heatmap_axis.set_xticks(np.arange(-0.5, label_probs.shape[1], 1), minor=True)
        heatmap_axis.set_yticks(np.arange(-0.5, label_probs.shape[0], 1), minor=True)
        heatmap_axis.grid(which="minor", color="white", linestyle="-", linewidth=0.3, alpha=0.2)
        heatmap_axis.tick_params(which="minor", bottom=False, left=False)
        fig.colorbar(im3, ax=heatmap_axis, fraction=0.046, pad=0.04)

    fig.suptitle(formatted_title, fontsize=10)
    plot_backend.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    plot_backend.show()
    plot_backend.close(fig)
    if graph_renderer is not None and decoded_graph is not None and not rendered_inline:
        try:
            graph_renderer([decoded_graph], titles=[formatted_title])
        except TypeError:
            graph_renderer([decoded_graph])
