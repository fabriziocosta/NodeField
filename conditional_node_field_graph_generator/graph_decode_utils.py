"""Shared graph-decoding normalization helpers."""

from typing import Any, Iterable, Optional, Sequence, Tuple


Edge = Tuple[int, int]


def _canonicalize_edge(edge: Sequence[Any]) -> Optional[Edge]:
    if len(edge) != 2:
        return None
    try:
        u = int(edge[0])
        v = int(edge[1])
    except (TypeError, ValueError):
        return None
    if u == v:
        return None
    return (u, v) if u < v else (v, u)


def _normalize_violating_edge_sets(
    edge_sets: Iterable[Iterable[Sequence[Any]]],
    *,
    n_nodes: Optional[int] = None,
) -> list[frozenset[Edge]]:
    normalized: list[frozenset[Edge]] = []
    seen: set[frozenset[Edge]] = set()
    for edge_set in edge_sets:
        canonical_edges = []
        for edge in edge_set:
            normalized_edge = _canonicalize_edge(edge)
            if normalized_edge is None:
                continue
            if n_nodes is not None and (
                normalized_edge[0] < 0
                or normalized_edge[1] < 0
                or normalized_edge[0] >= int(n_nodes)
                or normalized_edge[1] >= int(n_nodes)
            ):
                continue
            canonical_edges.append(normalized_edge)
        frozen = frozenset(canonical_edges)
        if not frozen or frozen in seen:
            continue
        seen.add(frozen)
        normalized.append(frozen)
    return normalized
