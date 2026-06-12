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


def _run_timed_step(owner: Any, label: str, callback):
    verbose_log(owner, label)
    started_at = time.time()
    result = callback()
    verbose_log(
        owner,
        f"Finished {label[0].lower()}{label[1:]} in "
        f"{_format_minutes_seconds(time.time() - started_at)}",
    )
    return result


def build_fit_artifacts(
    owner: Any,
    graphs: List[nx.Graph],
    targets: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Fit vectorizers and assemble compressed embeddings, labels, and supervision state."""
    _run_timed_step(
        owner,
        f"Fitting graph vectorizer on {len(graphs)} graphs",
        lambda: owner.graph_vectorizer.fit(graphs),
    )
    _run_timed_step(
        owner,
        f"Fitting node graph vectorizer on {len(graphs)} graphs",
        lambda: owner.node_graph_vectorizer.fit(graphs),
    )
    raw_node_embeddings_list = _run_timed_step(
        owner,
        f"Encoding node embeddings for {len(graphs)} graphs",
        lambda: owner._raw_node_encode(graphs),
    )
    raw_graph_embeddings = _run_timed_step(
        owner,
        f"Encoding graph embeddings for {len(graphs)} graphs",
        lambda: owner._raw_graph_encode(graphs),
    )
    _run_timed_step(
        owner,
        "Fitting embedding SVDs",
        lambda: owner._fit_embedding_svds(raw_node_embeddings_list, raw_graph_embeddings),
    )
    if owner.feasibility_estimator is not None:
        _run_timed_step(
            owner,
            f"Fitting feasibility estimator on {len(graphs)} graphs",
            lambda: owner.feasibility_estimator.fit(graphs),
        )
    node_label_targets = _run_timed_step(
        owner,
        f"Building node-label targets for {len(graphs)} graphs",
        lambda: owner.graphs_to_node_label_targets(graphs),
    )
    edge_label_targets, edge_label_pairs = _run_timed_step(
        owner,
        f"Building edge-label targets for {len(graphs)} graphs",
        lambda: owner.graphs_to_edge_label_targets(graphs),
    )
    supervision_plan = _run_timed_step(
        owner,
        "Building supervision plan",
        lambda: owner._build_supervision_plan(
            graphs,
            node_label_targets=node_label_targets,
            edge_label_targets=edge_label_targets,
        ),
    )
    owner.supervision_plan_ = supervision_plan
    if owner.conditional_node_generator_model is not None:
        setattr(owner.conditional_node_generator_model, "supervision_plan_", supervision_plan)

    node_embeddings_list = _run_timed_step(
        owner,
        f"Compressing node embeddings for {len(graphs)} graphs",
        lambda: owner._compress_node_embeddings(raw_node_embeddings_list),
    )
    graph_conditioning = _run_timed_step(
        owner,
        f"Building graph conditioning for {len(graphs)} graphs",
        lambda: owner._build_graph_conditioning_from_raw(
            graphs,
            raw_graph_embeddings,
        ),
    )
    owner.training_graph_conditioning_ = GraphConditioningBatch(
        graph_embeddings=np.asarray(graph_conditioning.graph_embeddings),
        node_counts=np.asarray(graph_conditioning.node_counts, dtype=np.int64),
        edge_counts=np.asarray(graph_conditioning.edge_counts, dtype=np.int64),
        condition_node_embeddings=(
            None
            if graph_conditioning.condition_node_embeddings is None
            else (
                np.asarray(graph_conditioning.condition_node_embeddings)
                if isinstance(graph_conditioning.condition_node_embeddings, np.ndarray)
                else [
                    np.asarray(embedding, dtype=float)
                    for embedding in graph_conditioning.condition_node_embeddings
                ]
            )
        ),
        condition_node_presence_mask=(
            None
            if graph_conditioning.condition_node_presence_mask is None
            else np.asarray(graph_conditioning.condition_node_presence_mask, dtype=bool)
        ),
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
