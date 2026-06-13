"""Helper types and utilities for feasibility-oracle guided decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .graph_decode_utils import _canonicalize_edge, _normalize_violating_edge_sets

Edge = Tuple[int, int]
NodeSet = Tuple[int, ...]
ForbiddenNodeLabelAssignment = Tuple[NodeSet, Tuple[Any, ...]]
ForbiddenEdgeLabelAssignment = Tuple[Tuple[Edge, ...], Tuple[Any, ...]]

_ORACLE_PROBABILITY_EPS = 1e-6


@dataclass(frozen=True)
class OracleEdgeAdditionProposal:
    edge: Edge
    label: Any
    priority: float


def enumerate_localized_edge_addition_proposals(
    *,
    adjacency_matrix: np.ndarray,
    violating_edge_sets: Sequence[FrozenSet[Edge]],
    active_node_mask: np.ndarray,
    edge_probability_matrix: np.ndarray,
    edge_label_classes: Optional[Sequence[Any]],
    edge_label_probabilities: Optional[np.ndarray],
    predicted_edge_label_matrix: Optional[np.ndarray],
    budget: int,
) -> List[OracleEdgeAdditionProposal]:
    """Rank missing labelled edges whose endpoints occur in one violation set."""
    budget = max(0, int(budget))
    if budget == 0:
        return []

    adjacency = np.asarray(adjacency_matrix, dtype=float)
    n_nodes = int(adjacency.shape[0])
    active_mask = np.asarray(active_node_mask, dtype=bool)[:n_nodes]
    edge_probabilities = np.asarray(edge_probability_matrix, dtype=float)
    label_probabilities = (
        None
        if edge_label_probabilities is None
        else np.asarray(edge_label_probabilities, dtype=float)
    )
    predicted_labels = (
        None
        if predicted_edge_label_matrix is None
        else np.asarray(predicted_edge_label_matrix, dtype=object)
    )
    label_classes = (
        []
        if edge_label_classes is None
        else list(np.asarray(edge_label_classes, dtype=object).reshape(-1))
    )

    candidate_edges = set()
    for edge_set in _normalize_violating_edge_sets(
        violating_edge_sets,
        n_nodes=n_nodes,
    ):
        nodes = sorted({node_idx for edge in edge_set for node_idx in edge})
        for idx, i in enumerate(nodes):
            if not active_mask[i]:
                continue
            for j in nodes[idx + 1 :]:
                if active_mask[j] and adjacency[i, j] == 0:
                    candidate_edges.add((i, j))

    proposals = []
    for i, j in sorted(candidate_edges):
        edge_probability = float(edge_probabilities[i, j])
        if label_probabilities is not None and label_classes:
            for label_idx, label in enumerate(label_classes):
                if label_idx >= label_probabilities.shape[-1]:
                    continue
                proposals.append(
                    OracleEdgeAdditionProposal(
                        edge=(i, j),
                        label=label,
                        priority=edge_probability + float(label_probabilities[i, j, label_idx]),
                    )
                )
            continue

        predicted_label = None if predicted_labels is None else predicted_labels[i, j]
        if predicted_label is not None:
            proposals.append(
                OracleEdgeAdditionProposal(
                    edge=(i, j),
                    label=predicted_label,
                    priority=edge_probability,
                )
            )

    proposals.sort(
        key=lambda proposal: (
            -float(proposal.priority),
            proposal.edge[0],
            proposal.edge[1],
            repr(proposal.label),
        )
    )
    return proposals[:budget]


def normalize_violating_node_sets(
    node_sets: Iterable[Iterable[Any]],
    *,
    n_nodes: Optional[int] = None,
) -> List[NodeSet]:
    normalized: List[NodeSet] = []
    seen: set[NodeSet] = set()
    for node_set in node_sets:
        canonical_nodes = []
        for node in node_set:
            try:
                node_idx = int(node)
            except (TypeError, ValueError):
                continue
            if n_nodes is not None and (node_idx < 0 or node_idx >= int(n_nodes)):
                continue
            canonical_nodes.append(node_idx)
        canonical = tuple(sorted(set(canonical_nodes)))
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        normalized.append(canonical)
    return normalized


def apply_oracle_edge_memory_penalty(
    prob_matrix: np.ndarray,
    edge_violation_prior: np.ndarray,
    penalty_weight: float,
) -> np.ndarray:
    """Penalize repeatedly violating edges in logit space for one decode trace."""
    base_prob = np.clip(
        np.asarray(prob_matrix, dtype=float),
        _ORACLE_PROBABILITY_EPS,
        1.0 - _ORACLE_PROBABILITY_EPS,
    )
    prior = np.maximum(np.asarray(edge_violation_prior, dtype=float), 0.0)
    adjusted_logit = np.log(base_prob) - np.log1p(-base_prob) - float(penalty_weight) * prior
    adjusted_prob = 1.0 / (1.0 + np.exp(-adjusted_logit))
    adjusted_prob = np.asarray(adjusted_prob, dtype=float)
    np.fill_diagonal(adjusted_prob, 0.0)
    return np.clip(adjusted_prob, 0.0, 1.0)


def update_oracle_edge_memory(
    edge_violation_prior: np.ndarray,
    violating_edge_sets: Sequence[FrozenSet[Edge]],
    *,
    update_weight: float,
    decay: float,
    clip_value: float,
) -> np.ndarray:
    """Update one graph's temporary violation memory from newly observed bad edges."""
    updated_prior = np.asarray(edge_violation_prior, dtype=float).copy()
    updated_prior *= float(decay)
    for edge_set in violating_edge_sets:
        for edge in edge_set:
            canonical_edge = _canonicalize_edge(edge)
            if canonical_edge is None:
                continue
            i, j = canonical_edge
            if i >= updated_prior.shape[0] or j >= updated_prior.shape[1]:
                continue
            updated_prior[i, j] += float(update_weight)
            updated_prior[j, i] += float(update_weight)
    np.fill_diagonal(updated_prior, 0.0)
    if np.isfinite(float(clip_value)):
        np.clip(updated_prior, 0.0, float(clip_value), out=updated_prior)
    return updated_prior


__all__ = [
    "Edge",
    "NodeSet",
    "ForbiddenNodeLabelAssignment",
    "ForbiddenEdgeLabelAssignment",
    "OracleEdgeAdditionProposal",
    "_ORACLE_PROBABILITY_EPS",
    "enumerate_localized_edge_addition_proposals",
    "normalize_violating_node_sets",
    "apply_oracle_edge_memory_penalty",
    "update_oracle_edge_memory",
]
