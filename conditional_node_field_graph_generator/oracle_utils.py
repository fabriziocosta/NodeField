"""Helper types and utilities for feasibility-oracle guided decoding."""

from __future__ import annotations

from typing import Any, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .graph_decode_utils import _canonicalize_edge

Edge = Tuple[int, int]
NodeSet = Tuple[int, ...]
ForbiddenNodeLabelAssignment = Tuple[NodeSet, Tuple[Any, ...]]
ForbiddenEdgeLabelAssignment = Tuple[Tuple[Edge, ...], Tuple[Any, ...]]

_ORACLE_PROBABILITY_EPS = 1e-6


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
