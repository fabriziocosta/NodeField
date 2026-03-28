"""Prediction-resolution and decode-preparation helpers."""

from __future__ import annotations

from typing import List, Optional, Tuple

import networkx as nx
import numpy as np

from .conditional_node_field_generator import GeneratedNodeBatch, GraphConditioningBatch


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


def decode_generated_nodes(
    owner,
    generated_nodes: GeneratedNodeBatch,
    graph_conditioning: Optional[GraphConditioningBatch] = None,
    feasibility_oracle_candidates_per_attempt: Optional[int] = None,
    attempt_idx: int = 0,
) -> List[nx.Graph]:
    """Dispatch generated-node decoding through either the oracle path or the plain decoder."""
    if owner._can_use_feasibility_oracle(
        feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
        attempt_idx=attempt_idx,
    ):
        return owner._decode_generated_nodes_with_oracle(
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
        desired_node_counts=None if graph_conditioning is None else np.asarray(graph_conditioning.node_counts, dtype=int),
        desired_edge_counts=None if graph_conditioning is None else np.asarray(graph_conditioning.edge_counts, dtype=int),
    )
