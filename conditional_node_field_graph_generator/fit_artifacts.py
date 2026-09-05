"""Fit-time artifact assembly for the graph generator orchestrator."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import networkx as nx
import numpy as np

from .conditional_node_field_generator import GraphConditioningBatch
from .runtime_utils import verbose_log


def _format_minutes_seconds(elapsed_seconds: float) -> str:
    total_seconds = max(0.0, float(elapsed_seconds))
    minutes = int(total_seconds // 60)
    seconds = total_seconds - (minutes * 60)
    return f"{minutes}m {seconds:.1f}s"


def _run_timed_step(
    owner: Any,
    *,
    step: int,
    total_steps: int,
    next_action: str,
    purpose: str,
    callback: Callable,
    result_message: Callable[[Any], str],
):
    """Run one fit step with a user-facing explanation and elapsed time."""
    step_prefix = f"Step {step} of {total_steps}"
    verbose_log(owner, "_" * 80)
    verbose_log(owner, f"{step_prefix} | Next: {next_action} Purpose: {purpose}")
    started_at = time.perf_counter()
    try:
        result = callback()
    except Exception:
        verbose_log(
            owner,
            f"{step_prefix} | This step failed after "
            f"{_format_minutes_seconds(time.perf_counter() - started_at)}.",
        )
        raise
    verbose_log(
        owner,
        f"{step_prefix} | Done: {result_message(result)} "
        f"Time: {_format_minutes_seconds(time.perf_counter() - started_at)}.",
    )
    return result


def build_fit_artifacts(
    owner: Any,
    graphs: List[nx.Graph],
    targets: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Fit vectorizers and assemble compressed embeddings, labels, and supervision state."""
    graph_count = len(graphs)
    total_steps = 11 if owner.feasibility_estimator is not None else 10
    _run_timed_step(
        owner,
        step=1,
        total_steps=total_steps,
        next_action="learn how to describe each complete graph with numeric features.",
        purpose="The generator will use this representation to understand each graph as a whole.",
        callback=lambda: owner.graph_vectorizer.fit(graphs),
        result_message=lambda _: f"Learned complete-graph features from {graph_count:,} graphs.",
    )
    _run_timed_step(
        owner,
        step=2,
        total_steps=total_steps,
        next_action="learn how to describe each node and its local graph context.",
        purpose="This gives the generator a numeric input for every node it must create.",
        callback=lambda: owner.node_graph_vectorizer.fit(graphs),
        result_message=lambda _: f"Learned node-level features from {graph_count:,} graphs.",
    )
    raw_node_embeddings_list = _run_timed_step(
        owner,
        step=3,
        total_steps=total_steps,
        next_action="convert every node into its raw numeric embedding.",
        purpose="These embeddings are the starting point for the smaller inputs used during training.",
        callback=lambda: owner._raw_node_encode(graphs),
        result_message=lambda embeddings: (
            f"Created raw node embeddings for {len(embeddings):,} graphs, "
            f"covering {sum(int(embedding.shape[0]) for embedding in embeddings):,} nodes."
        ),
    )
    raw_graph_embeddings = _run_timed_step(
        owner,
        step=4,
        total_steps=total_steps,
        next_action="convert every complete graph into a raw numeric embedding.",
        purpose="These embeddings provide graph-level information to condition the generator.",
        callback=lambda: owner._raw_graph_encode(graphs),
        result_message=lambda embeddings: (
            f"Created one raw graph embedding for each of {int(embeddings.shape[0]):,} graphs."
        ),
    )
    _run_timed_step(
        owner,
        step=5,
        total_steps=total_steps,
        next_action="learn compact versions of the node and graph embeddings.",
        purpose="Fewer features make later training faster and use less memory while keeping the main signal.",
        callback=lambda: owner._fit_embedding_svds(raw_node_embeddings_list, raw_graph_embeddings),
        result_message=lambda _: (
            f"Learned compact embeddings with up to {int(owner.node_embedding_effective_dimension_):,} "
            f"node features and {int(owner.graph_embedding_effective_dimension_):,} graph features."
        ),
    )
    if owner.feasibility_estimator is not None:
        _run_timed_step(
            owner,
            step=6,
            total_steps=total_steps,
            next_action="learn the graph-feasibility checks.",
            purpose="These checks will screen generated graphs against the configured constraints.",
            callback=lambda: owner.feasibility_estimator.fit(graphs),
            result_message=lambda _: "The feasibility estimator is ready to screen generated graphs.",
        )
    node_label_targets = _run_timed_step(
        owner,
        step=7 if owner.feasibility_estimator is not None else 6,
        total_steps=total_steps,
        next_action="extract the node properties the generator should learn.",
        purpose="These targets tell the model which labels belong on generated nodes.",
        callback=lambda: owner.graphs_to_node_label_targets(graphs),
        result_message=lambda targets_: f"Prepared node-label targets for {len(targets_):,} graphs.",
    )
    edge_label_targets, edge_label_pairs = _run_timed_step(
        owner,
        step=8 if owner.feasibility_estimator is not None else 7,
        total_steps=total_steps,
        next_action="look for edge labels the generator should learn.",
        purpose="These targets teach edge types when the training graphs contain usable edge labels.",
        callback=lambda: owner.graphs_to_edge_label_targets(graphs),
        result_message=lambda result: (
            f"Prepared {len(result[1]):,} directed edge-label examples."
            if result[0] is not None
            else "No usable edge labels were found; edge-label supervision will be skipped."
        ),
    )
    supervision_plan = _run_timed_step(
        owner,
        step=9 if owner.feasibility_estimator is not None else 8,
        total_steps=total_steps,
        next_action="decide which training signals are available and should be used.",
        purpose="This keeps training aligned with both the data and the configured supervision options.",
        callback=lambda: owner._build_supervision_plan(
            graphs,
            node_label_targets=node_label_targets,
            edge_label_targets=edge_label_targets,
        ),
        result_message=lambda _: "Built the training plan for the available graph, node, and edge signals.",
    )
    owner.supervision_plan_ = supervision_plan
    if owner.conditional_node_generator_model is not None:
        setattr(owner.conditional_node_generator_model, "supervision_plan_", supervision_plan)

    node_embeddings_list = _run_timed_step(
        owner,
        step=10 if owner.feasibility_estimator is not None else 9,
        total_steps=total_steps,
        next_action="convert raw node embeddings into the compact training inputs.",
        purpose="The neural model will train on these smaller per-node inputs.",
        callback=lambda: owner._compress_node_embeddings(raw_node_embeddings_list),
        result_message=lambda embeddings: (
            f"Compressed node embeddings for {len(embeddings):,} graphs."
        ),
    )
    graph_conditioning = _run_timed_step(
        owner,
        step=11 if owner.feasibility_estimator is not None else 10,
        total_steps=total_steps,
        next_action="assemble the graph-level information used to guide generation.",
        purpose="This pairs each graph embedding with its node and edge counts.",
        callback=lambda: owner._build_graph_conditioning_from_raw(
            graphs,
            raw_graph_embeddings,
        ),
        result_message=lambda conditioning: (
            f"Built graph-conditioning data for {len(conditioning.graph_embeddings):,} graphs."
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
