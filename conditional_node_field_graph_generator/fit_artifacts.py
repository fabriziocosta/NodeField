"""Fit-time artifact assembly for the graph generator orchestrator."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

import networkx as nx
import numpy as np

from .conditional_node_field_generator import GraphConditioningBatch
from .runtime_utils import verbose_log


def _format_minutes_seconds(elapsed_seconds: float) -> str:
    total_seconds = max(0.0, float(elapsed_seconds))
    minutes = int(total_seconds // 60)
    seconds = total_seconds - (minutes * 60)
    return f"{minutes}m {seconds:.1f}s"


def build_fit_artifacts(
    owner: Any,
    graphs: List[nx.Graph],
    targets: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Fit vectorizers and assemble compressed embeddings, labels, and supervision state."""
    owner.graph_vectorizer.fit(graphs)
    owner.node_graph_vectorizer.fit(graphs)
    raw_node_embeddings_list = owner._raw_node_encode(graphs)
    raw_graph_embeddings = owner._raw_graph_encode(graphs)
    owner._fit_embedding_svds(raw_node_embeddings_list, raw_graph_embeddings)
    if owner.feasibility_estimator is not None:
        verbose_log(owner, f"Fitting feasibility estimator on {len(graphs)} graphs")
        feasibility_started_at = time.time()
        owner.feasibility_estimator.fit(graphs)
        verbose_log(
            owner,
            "Finished fitting feasibility estimator in "
            f"{_format_minutes_seconds(time.time() - feasibility_started_at)}",
        )
    node_label_targets = owner.graphs_to_node_label_targets(graphs)
    edge_label_targets, edge_label_pairs = owner.graphs_to_edge_label_targets(graphs)
    supervision_plan = owner._build_supervision_plan(
        graphs,
        node_label_targets=node_label_targets,
        edge_label_targets=edge_label_targets,
    )
    owner.supervision_plan_ = supervision_plan
    if owner.conditional_node_generator_model is not None:
        setattr(owner.conditional_node_generator_model, "supervision_plan_", supervision_plan)

    node_embeddings_list = owner._compress_node_embeddings(raw_node_embeddings_list)
    graph_conditioning = owner._build_graph_conditioning_from_raw(
        graphs,
        raw_graph_embeddings,
    )
    owner.training_graph_conditioning_ = GraphConditioningBatch(
        graph_embeddings=np.asarray(graph_conditioning.graph_embeddings),
        node_counts=np.asarray(graph_conditioning.node_counts, dtype=np.int64),
        edge_counts=np.asarray(graph_conditioning.edge_counts, dtype=np.int64),
    )
    return {
        "node_label_targets": node_label_targets,
        "edge_label_targets": edge_label_targets,
        "edge_label_pairs": edge_label_pairs,
        "supervision_plan": supervision_plan,
        "node_embeddings_list": node_embeddings_list,
        "graph_conditioning": graph_conditioning,
        "targets": targets,
    }
