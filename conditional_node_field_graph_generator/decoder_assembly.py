"""Validation and graph assembly helpers for decoder outputs."""

from typing import Any

import networkx as nx
import numpy as np


def validate_node_label_array(
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


def validate_edge_label_array(
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
            f"graph {graph_idx} received {edge_labels.shape[0]} labels for "
            f"{expected_edge_count} edges."
        )
    return edge_labels


def assemble_graph(
    node_presence_mask: np.ndarray,
    node_labels: np.ndarray,
    edge_labels: np.ndarray,
    adj_mtx: np.ndarray,
) -> nx.Graph:
    graph = nx.from_numpy_array(adj_mtx)

    if len(node_labels) > 0 and not all(label is None for label in node_labels):
        nx.set_node_attributes(
            graph,
            {i: label for i, label in enumerate(node_labels)},
            "label",
        )

    if np.sum(adj_mtx) > 0 and len(edge_labels) > 0:
        edge_idx = 0
        edge_attr: dict[tuple[int, int], Any] = {}
        for i in range(graph.number_of_nodes()):
            for j in range(i + 1, graph.number_of_nodes()):
                if adj_mtx[i, j] != 0:
                    edge_attr[(i, j)] = edge_labels[edge_idx]
                    edge_idx += 1
        nx.set_edge_attributes(graph, edge_attr, "label")

    existent_indices = np.where(
        np.asarray(node_presence_mask[: adj_mtx.shape[0]], dtype=bool)
    )[0]
    return graph.subgraph(existent_indices).copy()


def assemble_graph_star(args) -> nx.Graph:
    return assemble_graph(*args)


def edge_label_matrix_to_list(
    adj_mtx: np.ndarray,
    edge_label_matrix: np.ndarray,
) -> np.ndarray:
    edge_labels = []
    for i in range(adj_mtx.shape[0]):
        for j in range(i + 1, adj_mtx.shape[1]):
            if adj_mtx[i, j] != 0:
                edge_labels.append(edge_label_matrix[i, j])
    return np.asarray(edge_labels, dtype=object)


def edge_label_list_to_matrix(
    adj_mtx: np.ndarray,
    edge_labels,
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


def assemble_edge_labels_from_matrix(
    adj_mtx: np.ndarray,
    edge_label_matrix: np.ndarray,
) -> np.ndarray:
    if edge_label_matrix.shape != adj_mtx.shape:
        raise ValueError(
            "Each predicted edge-label matrix must have the same shape as its adjacency matrix; "
            f"received {edge_label_matrix.shape} and {adj_mtx.shape}."
        )
    return edge_label_matrix_to_list(adj_mtx, edge_label_matrix)
