"""Direct probability-based adjacency decoding helpers."""

from collections.abc import Sequence

import numpy as np


def edge_candidates(active_indices: np.ndarray, prob_matrix: np.ndarray):
    candidates = []
    for local_i, i in enumerate(active_indices):
        for j in active_indices[local_i + 1 :]:
            probability = float((prob_matrix[i, j] + prob_matrix[j, i]) / 2.0)
            candidates.append((probability, int(i), int(j)))
    return candidates


def select_top_k(candidates, desired_edge_count: int):
    target = min(max(0, int(desired_edge_count)), len(candidates))
    return sorted(candidates, key=lambda item: (-item[0], item[1], item[2]))[:target]


def select_by_threshold(candidates, threshold: float):
    return [edge for edge in candidates if edge[0] >= float(threshold)]


def select_degree_aware(
    candidates,
    active_indices: np.ndarray,
    target_degrees: Sequence[int],
    desired_edge_count: int,
):
    target_edge_count = min(max(0, int(desired_edge_count)), len(candidates))
    if target_edge_count == 0:
        return []

    target_degrees = np.asarray(target_degrees, dtype=int)
    edge_by_key = {}
    candidates_by_node = {int(node): [] for node in active_indices}
    for probability, i, j in candidates:
        i, j = int(i), int(j)
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
        if int(node) < target_degrees.shape[0] and target_degrees[int(node)] > 0
    }
    for node in active_indices:
        node = int(node)
        if node >= target_degrees.shape[0]:
            continue
        quota = max(0, int(target_degrees[node]))
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
        selected_degrees[i] += 1
        selected_degrees[j] += 1

    while len(selected_edges) > target_edge_count:
        removable_idx = None
        for edge_idx in sorted(
            range(len(selected_edges)),
            key=lambda idx: (
                selected_edges[idx][0],
                -selected_edges[idx][1],
                -selected_edges[idx][2],
            ),
        ):
            _probability, i, j = selected_edges[edge_idx]
            if (
                selected_degrees[i] - 1 >= target_degrees[i]
                and selected_degrees[j] - 1 >= target_degrees[j]
            ):
                removable_idx = edge_idx
                break
        if removable_idx is None:
            break
        _probability, i, j = selected_edges.pop(removable_idx)
        selected_degrees[i] -= 1
        selected_degrees[j] -= 1

    selected_edges.sort(key=lambda item: (-item[0], item[1], item[2]))
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


def adjacency_from_edges(n_nodes: int, selected_edges) -> np.ndarray:
    adjacency = np.zeros((int(n_nodes), int(n_nodes)), dtype=float)
    for _, i, j in selected_edges:
        adjacency[i, j] = adjacency[j, i] = 1.0
    return adjacency
