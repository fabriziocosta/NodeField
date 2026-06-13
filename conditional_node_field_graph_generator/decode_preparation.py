"""Preparation helpers shared by plain and oracle-guided graph decoding."""

from typing import List, Optional, Tuple

import numpy as np

from .conditional_node_field_generator import GeneratedNodeBatch


def build_masked_prob_matrix(
    existence_mask: np.ndarray,
    degree_prediction: np.ndarray,
    prob_matrix: np.ndarray,
) -> np.ndarray:
    n_nodes = min(len(existence_mask), len(degree_prediction))
    masked = np.asarray(prob_matrix, dtype=float)[:n_nodes, :n_nodes].copy()
    existent = np.asarray(existence_mask[:n_nodes], dtype=bool)
    inactive_edges = ~(existent[:, None] & existent[None, :])
    masked[inactive_edges] = 0.0
    np.fill_diagonal(masked, 0.0)
    return (masked + masked.T) / 2.0


def build_single_generated_node_batch(
    generated_nodes: GeneratedNodeBatch,
    graph_idx: int,
) -> GeneratedNodeBatch:
    """Slice a multi-graph generated batch down to a single-example batch."""

    def list_item(value, dtype=None):
        if value is None:
            return None
        return [np.asarray(value[graph_idx], dtype=dtype)]

    def array_item(value, dtype=None):
        if value is None:
            return None
        return np.asarray([value[graph_idx]], dtype=dtype)

    return GeneratedNodeBatch(
        node_embeddings_list=list_item(generated_nodes.node_embeddings_list),
        node_presence_mask=array_item(generated_nodes.node_presence_mask),
        node_degree_predictions=array_item(generated_nodes.node_degree_predictions),
        node_labels=list_item(generated_nodes.node_labels, object),
        node_existence_probabilities=array_item(
            generated_nodes.node_existence_probabilities,
            float,
        ),
        edge_probability_matrices=list_item(
            generated_nodes.edge_probability_matrices,
            float,
        ),
        edge_label_matrices=list_item(generated_nodes.edge_label_matrices, object),
        node_label_logits=list_item(generated_nodes.node_label_logits, float),
        node_label_probabilities=list_item(
            generated_nodes.node_label_probabilities,
            float,
        ),
        edge_existence_probabilities=list_item(
            generated_nodes.edge_existence_probabilities,
            float,
        ),
        edge_label_logits=list_item(generated_nodes.edge_label_logits, float),
        edge_label_probabilities=list_item(
            generated_nodes.edge_label_probabilities,
            float,
        ),
        horizon_probability_matrices=list_item(
            generated_nodes.horizon_probability_matrices,
            float,
        ),
        horizon=generated_nodes.horizon,
    )


def resolve_predicted_node_labels(
    owner,
    generated_nodes: GeneratedNodeBatch,
) -> List[np.ndarray]:
    node_label_plan = owner._plan_channel("node_labels")
    if generated_nodes.node_labels is not None:
        return [
            np.asarray(node_labels, dtype=object)
            for node_labels in generated_nodes.node_labels
        ]
    if generated_nodes.node_presence_mask is None:
        raise RuntimeError("Node-label resolution requires node_presence_mask predictions.")
    if node_label_plan is None or node_label_plan.mode == "disabled":
        value = None
    elif node_label_plan.mode == "constant":
        value = node_label_plan.constant_value
    else:
        raise RuntimeError(
            "Node-label channel is configured as learned, but the generator returned no node labels."
        )
    return [
        np.asarray([value] * len(node_presence_mask), dtype=object)
        for node_presence_mask in generated_nodes.node_presence_mask
    ]


def resolve_predicted_edge_labels(
    owner,
    generated_nodes: GeneratedNodeBatch,
    predicted_edge_probability_matrices: Optional[List[np.ndarray]],
) -> Tuple[Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
    edge_label_plan = owner._plan_channel("edge_labels")
    if generated_nodes.edge_label_matrices is not None:
        return None, [
            np.asarray(edge_label_matrix, dtype=object)
            for edge_label_matrix in generated_nodes.edge_label_matrices
        ]
    if predicted_edge_probability_matrices is None:
        raise RuntimeError(
            "Edge-label resolution requires edge probabilities to determine decoded edge counts."
        )
    if edge_label_plan is None or edge_label_plan.mode == "disabled":
        return [
            np.asarray([], dtype=object)
            for _ in predicted_edge_probability_matrices
        ], None
    if edge_label_plan.mode != "constant":
        raise RuntimeError(
            "Edge-label channel is configured as learned, but the generator returned no edge labels."
        )
    matrices = []
    for prob_matrix in predicted_edge_probability_matrices:
        prob_matrix = np.asarray(prob_matrix)
        if prob_matrix.ndim != 2 or prob_matrix.shape[0] != prob_matrix.shape[1]:
            raise ValueError(
                "Constant edge-label resolution expects square edge-probability matrices "
                f"(got shape={prob_matrix.shape})."
            )
        matrix = np.full(
            prob_matrix.shape,
            edge_label_plan.constant_value,
            dtype=object,
        )
        np.fill_diagonal(matrix, None)
        matrices.append(matrix)
    return None, matrices


def decode_generated_nodes(*args, **kwargs):
    """Compatibility adapter for the public decode dispatcher."""
    from .conditional_node_field_graph_decoder import decode_generated_nodes as decode_impl

    return decode_impl(*args, **kwargs)


__all__ = [
    "build_masked_prob_matrix",
    "build_single_generated_node_batch",
    "decode_generated_nodes",
    "resolve_predicted_edge_labels",
    "resolve_predicted_node_labels",
]
