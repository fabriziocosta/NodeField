"""Shared summaries and decoder inspection for molecular notebooks."""

import numpy as np
import pandas as pd
from IPython.display import display

from .visualization import show_molecules


def label_counter(graphs, kind="node"):
    counts = {}
    if kind == "node":
        for graph in graphs:
            for _, attrs in graph.nodes(data=True):
                label = attrs.get("label")
                counts[label] = counts.get(label, 0) + 1
    elif kind == "edge":
        for graph in graphs:
            for _, _, attrs in graph.edges(data=True):
                label = attrs.get("label")
                counts[label] = counts.get(label, 0) + 1
    else:
        raise ValueError("kind must be 'node' or 'edge'")
    return dict(sorted(counts.items(), key=lambda item: str(item[0])))


def summarize_graphs(graphs, targets=None, prefix="dataset"):
    graphs = list(graphs)
    if not graphs:
        print(f"{prefix}: 0 graphs")
        return
    node_counts = np.array([graph.number_of_nodes() for graph in graphs], dtype=int)
    edge_counts = np.array([graph.number_of_edges() for graph in graphs], dtype=int)
    print(f"{prefix}: {len(graphs)} graphs")
    if targets is not None:
        print(f"{prefix}: class counts = {dict(zip(*np.unique(targets, return_counts=True)))}")
    print(
        f"{prefix}: node count min/median/max = "
        f"{node_counts.min()}/{int(np.median(node_counts))}/{node_counts.max()}"
    )
    print(
        f"{prefix}: edge count min/median/max = "
        f"{edge_counts.min()}/{int(np.median(edge_counts))}/{edge_counts.max()}"
    )
    print(f"{prefix}: node labels = {label_counter(graphs, 'node')}")
    print(f"{prefix}: edge labels = {label_counter(graphs, 'edge')}")


def inspect_predicted_masks_and_edge_labels(graph_generator, graphs, n_graphs=6):
    graph_conditioning = graph_generator.graph_encode(list(graphs)[:n_graphs])
    generated_nodes = graph_generator.conditional_node_generator_model.predict(graph_conditioning)

    decoded = graph_generator.graph_decoder.decode(
        generated_nodes,
        predicted_node_labels_list=generated_nodes.node_labels,
        predicted_edge_probability_matrices=generated_nodes.edge_probability_matrices,
        predicted_edge_label_matrices=generated_nodes.edge_label_matrices,
        desired_node_counts=graph_conditioning.node_counts[:n_graphs],
        desired_edge_counts=graph_conditioning.edge_counts[:n_graphs],
    )

    observed_node_counts = graph_conditioning.node_counts[:n_graphs]
    raw_predicted_node_counts = generated_nodes.node_presence_mask[:n_graphs].sum(axis=1)
    decoded_support_node_counts = []
    raw_expected_edge_counts = []
    decoded_edge_counts = []
    for idx in range(min(n_graphs, len(decoded))):
        raw_mask = np.asarray(generated_nodes.node_presence_mask[idx], dtype=bool)
        existence_scores = None if generated_nodes.node_existence_probabilities is None else np.asarray(
            generated_nodes.node_existence_probabilities[idx],
            dtype=float,
        )
        decoded_support_mask = graph_generator.graph_decoder.resolve_node_presence_mask(
            raw_mask,
            desired_node_count=int(observed_node_counts[idx]),
            node_existence_scores=existence_scores,
        )
        decoded_support_node_counts.append(int(decoded_support_mask.sum()))
        if generated_nodes.edge_probability_matrices is None:
            raw_expected_edge_counts.append(np.nan)
        else:
            prob_matrix = np.asarray(generated_nodes.edge_probability_matrices[idx], dtype=float)
            active_prob_matrix = prob_matrix[: raw_mask.shape[0], : raw_mask.shape[0]].copy()
            active_prob_matrix[~decoded_support_mask, :] = 0.0
            active_prob_matrix[:, ~decoded_support_mask] = 0.0
            raw_expected_edge_counts.append(float(np.triu(active_prob_matrix, k=1).sum()))
        decoded_edge_counts.append(int(decoded[idx].number_of_edges()))
    mask_frame = pd.DataFrame(
        {
            "conditioning_nodes": observed_node_counts,
            "raw_predicted_nodes": raw_predicted_node_counts,
            "decoded_support_nodes": decoded_support_node_counts,
            "conditioning_edges": graph_conditioning.edge_counts[:n_graphs],
            "raw_expected_edges": raw_expected_edge_counts,
            "decoded_edges": decoded_edge_counts,
        }
    )
    display(mask_frame)

    edge_label_matrices = generated_nodes.edge_label_matrices
    if edge_label_matrices is None:
        print("No edge-label matrices were predicted.")
    else:
        summaries = []
        for idx, matrix in enumerate(edge_label_matrices[:n_graphs]):
            labels, counts = np.unique(np.asarray(matrix, dtype=object), return_counts=True)
            summaries.append({"graph_idx": idx, **{str(label): int(count) for label, count in zip(labels, counts)}})
        display(pd.DataFrame(summaries).fillna(0).astype({"graph_idx": int}))

    show_molecules(decoded, n=n_graphs, title="Decoded molecules from the inspected latent batch")
    return generated_nodes, decoded
