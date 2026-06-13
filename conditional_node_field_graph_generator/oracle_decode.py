"""Oracle-guided adjacency solve orchestration."""

import random
from typing import FrozenSet, List, Optional, Sequence

import numpy as np

from .runtime_utils import run_with_fork_timeout, verbose_log

Edge = tuple[int, int]


def sample_oracle_cuts_for_iteration(
    owner,
    accumulated_cuts: Sequence[FrozenSet[Edge]],
    solve_iteration_idx: int,
) -> List[FrozenSet[Edge]]:
    if not accumulated_cuts:
        return []
    max_iterations = max(1, int(owner.max_oracle_iterations))
    if solve_iteration_idx >= max_iterations - 1:
        return []
    keep_fraction = 1.0 - (float(solve_iteration_idx) / float(max_iterations - 1))
    keep_count = int(np.ceil(len(accumulated_cuts) * keep_fraction))
    keep_count = min(len(accumulated_cuts), max(0, keep_count))
    if keep_count >= len(accumulated_cuts):
        return list(accumulated_cuts)
    if keep_count <= 0:
        return []
    selected_indices = sorted(random.sample(range(len(accumulated_cuts)), keep_count))
    return [accumulated_cuts[idx] for idx in selected_indices]


def _optimize_adjacency_matrix_worker(graph_decoder, args, kwargs) -> np.ndarray:
    return graph_decoder.optimize_adjacency_matrix(*args, **kwargs)


def oracle_adjacency_timeout_seconds(owner) -> Optional[float]:
    timeout_seconds = getattr(owner, "max_decode_seconds_per_sample", None)
    if timeout_seconds is None:
        timeout_seconds = getattr(owner, "max_feasibility_seconds_per_sample", None)
    if timeout_seconds is None:
        graph_decoder = getattr(owner, "graph_decoder", None)
        if graph_decoder is not None:
            timeout_seconds = getattr(graph_decoder, "active_time_limit_seconds", None)
            if timeout_seconds is None:
                timeout_seconds = getattr(graph_decoder, "adjacency_time_limit_seconds", None)
    if timeout_seconds is None:
        return None
    return max(1.0, float(timeout_seconds))


def optimize_oracle_adjacency_matrix(owner, *args, **kwargs) -> np.ndarray:
    timeout_seconds = oracle_adjacency_timeout_seconds(owner)
    if timeout_seconds is None:
        return owner.graph_decoder.optimize_adjacency_matrix(*args, **kwargs)
    return run_with_fork_timeout(
        _optimize_adjacency_matrix_worker,
        owner.graph_decoder,
        args,
        kwargs,
        timeout_seconds=timeout_seconds,
    )


def solve_oracle_relaxed_adjacency(
    owner,
    *,
    masked_prob_matrix: np.ndarray,
    target_degrees: List[int],
    accumulated_cuts: Sequence[FrozenSet[Edge]],
    start_iteration_idx: int,
    target_edge_count: Optional[int] = None,
    edge_violation_prior: Optional[np.ndarray] = None,
    active_node_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    from .oracle_utils import apply_oracle_edge_memory_penalty

    solve_prob_matrix = np.asarray(masked_prob_matrix, dtype=float)
    if edge_violation_prior is not None and owner.oracle_edge_memory_penalty > 0.0:
        solve_prob_matrix = apply_oracle_edge_memory_penalty(
            solve_prob_matrix,
            edge_violation_prior,
            owner.oracle_edge_memory_penalty,
        )
    last_error: Optional[Exception] = None
    for solve_iteration_idx in range(start_iteration_idx, owner.max_oracle_iterations):
        active_cuts = sample_oracle_cuts_for_iteration(
            owner,
            accumulated_cuts,
            solve_iteration_idx,
        )
        try:
            optimize_kwargs = {
                "target_edge_count": target_edge_count,
                "forbidden_edge_sets": active_cuts,
            }
            if active_node_mask is not None:
                optimize_kwargs["active_node_mask"] = active_node_mask
            return optimize_oracle_adjacency_matrix(
                owner,
                solve_prob_matrix,
                target_degrees,
                **optimize_kwargs,
            )
        except TimeoutError:
            raise
        except Exception as exc:
            last_error = exc
            if int(owner.verbose) >= 1:
                verbose_log(
                    owner,
                    "Oracle-guided adjacency solve failed with "
                    f"{len(active_cuts)} active cuts at iteration "
                    f"{solve_iteration_idx + 1}/{owner.max_oracle_iterations}; "
                    "retrying with fewer cuts.",
                )
    if last_error is not None:
        raise RuntimeError(
            "Oracle-guided adjacency solve failed even after relaxing all oracle cuts."
        ) from last_error
    raise RuntimeError("Oracle-guided adjacency solve could not be attempted.")


def decode_generated_nodes_with_oracle(owner, generated_nodes, graph_conditioning=None):
    # The high-level loop remains lazily imported to avoid a module cycle while
    # the graph-generator compatibility adapters continue to support notebooks.
    from .conditional_node_field_graph_decoder import (
        decode_generated_nodes_with_oracle as decode_impl,
    )

    return decode_impl(
        owner,
        generated_nodes,
        graph_conditioning=graph_conditioning,
    )


__all__ = [
    "decode_generated_nodes_with_oracle",
    "optimize_oracle_adjacency_matrix",
    "oracle_adjacency_timeout_seconds",
    "sample_oracle_cuts_for_iteration",
    "solve_oracle_relaxed_adjacency",
]
