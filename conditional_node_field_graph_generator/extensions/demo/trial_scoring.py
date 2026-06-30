"""Trial scoring helpers shared by molecule and artificial-graph demos."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def number_of_violations(graph_generator: Any, generated_graphs: list[Any]) -> np.ndarray:
    feasibility_estimator = getattr(graph_generator, "feasibility_estimator", None)
    if feasibility_estimator is None or not hasattr(feasibility_estimator, "number_of_violations"):
        raise RuntimeError(
            "average_num_violations scoring requires graph_generator.feasibility_estimator."
        )
    return np.asarray(feasibility_estimator.number_of_violations(generated_graphs), dtype=float)


def graph_embeddings_from_node_embeddings(node_embeddings_list: list[Any]) -> np.ndarray:
    graph_embeddings = []
    for node_embeddings in node_embeddings_list:
        node_matrix = np.asarray(node_embeddings, dtype=float)
        if node_matrix.ndim == 1:
            node_matrix = node_matrix.reshape(1, -1)
        if node_matrix.ndim != 2:
            raise ValueError("Each node embedding array must be 1D or 2D.")
        graph_embeddings.append(np.sum(node_matrix, axis=0))
    if not graph_embeddings:
        return np.empty((0, 0), dtype=float)
    return np.vstack(graph_embeddings)


def mean_graph_embedding(graph_generator: Any, graphs: list[Any]) -> np.ndarray | None:
    if not graphs:
        return None
    node_embeddings_list = graph_generator.node_encode(graphs)
    graph_embeddings = graph_embeddings_from_node_embeddings(list(node_embeddings_list))
    if graph_embeddings.shape[0] == 0:
        return None
    return np.mean(graph_embeddings, axis=0)


def bounded_cosine_distance(first: np.ndarray | None, second: np.ndarray | None) -> float:
    if first is None or second is None:
        return math.inf
    first_vector = np.asarray(first, dtype=float).reshape(-1)
    second_vector = np.asarray(second, dtype=float).reshape(-1)
    if first_vector.shape != second_vector.shape:
        raise ValueError(
            "Cosine distance requires matching embedding dimensions "
            f"({first_vector.shape} != {second_vector.shape})."
        )
    first_norm = float(np.linalg.norm(first_vector))
    second_norm = float(np.linalg.norm(second_vector))
    if first_norm == 0.0 or second_norm == 0.0:
        return math.inf
    similarity = float(np.dot(first_vector, second_vector) / (first_norm * second_norm))
    bounded_similarity = float(np.clip(similarity, 0.0, 1.0))
    return float(1.0 - bounded_similarity)


def optimization_score(average_num_violations: float, embedding_distance: float) -> float:
    if not np.isfinite(average_num_violations) or not np.isfinite(embedding_distance):
        return math.inf
    return float(average_num_violations * embedding_distance)


def score_generated_graphs(
    graph_generator: Any,
    generated_graphs: list[Any],
    *,
    training_mean_graph_embedding: np.ndarray | None,
) -> dict[str, Any]:
    if len(generated_graphs) == 0:
        return {
            "returned_samples": 0,
            "average_num_violations": math.inf,
            "median_num_violations": math.inf,
            "feasible_count": 0,
            "feasible_rate": 0.0,
            "violation_counts": np.asarray([], dtype=float),
            "average_training_embedding_cosine_distance": math.inf,
            "optimization_score": math.inf,
        }
    violation_counts = number_of_violations(graph_generator, generated_graphs)
    if violation_counts.shape[0] != len(generated_graphs):
        raise RuntimeError(
            "Feasibility estimator returned an unexpected number of violation counts "
            f"({violation_counts.shape[0]} for {len(generated_graphs)} graphs)."
        )
    generated_mean = mean_graph_embedding(graph_generator, generated_graphs)
    embedding_distance = bounded_cosine_distance(training_mean_graph_embedding, generated_mean)
    average_num_violations = float(np.mean(violation_counts))
    feasible_count = int(np.sum(violation_counts == 0))
    return {
        "returned_samples": len(generated_graphs),
        "average_num_violations": average_num_violations,
        "median_num_violations": float(np.median(violation_counts)),
        "feasible_count": feasible_count,
        "feasible_rate": float(feasible_count / len(generated_graphs)),
        "violation_counts": violation_counts,
        "average_training_embedding_cosine_distance": embedding_distance,
        "optimization_score": optimization_score(average_num_violations, embedding_distance),
    }


def trial_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, int]:
    optimization = row.get("optimization_score")
    if optimization is None:
        optimization = row.get("average_num_violations", math.inf)
    return (
        float(optimization),
        float(row.get("average_num_violations", math.inf)),
        float(row.get("average_training_embedding_cosine_distance", math.inf)),
        -float(row.get("feasible_rate", 0.0) or 0.0),
        int(row.get("trial_id", row.get("campaign_trial_id", 0)) or 0),
    )


TRIAL_SORT_COLUMNS = [
    "optimization_score",
    "average_num_violations",
    "average_training_embedding_cosine_distance",
    "feasible_rate",
    "trial_id",
]
TRIAL_SORT_ASCENDING = [True, True, True, False, True]


__all__ = [
    "TRIAL_SORT_ASCENDING",
    "TRIAL_SORT_COLUMNS",
    "bounded_cosine_distance",
    "graph_embeddings_from_node_embeddings",
    "mean_graph_embedding",
    "number_of_violations",
    "optimization_score",
    "score_generated_graphs",
    "trial_sort_key",
]
