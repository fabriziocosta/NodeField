"""Shared feasibility retry and decode-pipeline helpers."""

from __future__ import annotations

import time
from typing import Any, Callable, Optional, Sequence

import networkx as nx
import numpy as np

from .feasibility_utils import format_feasibility_attempt_status
from .runtime_utils import verbose_log


def should_apply_feasibility_filtering(
    owner: Any,
    apply_feasibility_filtering: Optional[bool],
) -> bool:
    """Resolve the effective feasibility-filtering flag for a decode call."""
    return (
        owner.use_feasibility_filtering
        if apply_feasibility_filtering is None
        else bool(apply_feasibility_filtering)
    )


def accept_feasible_candidates_by_slot(
    *,
    decoded_graphs: Sequence[nx.Graph],
    feasibility_mask: Sequence[bool],
    candidate_slot_indices: Sequence[int],
    accepted_graphs_by_slot: list[Optional[nx.Graph]],
    rng: Optional[np.random.Generator] = None,
) -> tuple[int, int]:
    """Count feasible candidates and fill each open slot with one random feasible graph."""
    feasible_candidates_by_slot: dict[int, list[nx.Graph]] = {}
    feasible_candidate_count = 0
    for graph, is_feasible, slot_idx in zip(decoded_graphs, feasibility_mask, candidate_slot_indices):
        if not is_feasible:
            continue
        feasible_candidate_count += 1
        if accepted_graphs_by_slot[slot_idx] is None:
            feasible_candidates_by_slot.setdefault(slot_idx, []).append(graph)

    filled_now = 0
    for slot_idx, graphs_for_slot in feasible_candidates_by_slot.items():
        if not graphs_for_slot or accepted_graphs_by_slot[slot_idx] is not None:
            continue
        selected_idx = (
            int(rng.integers(len(graphs_for_slot)))
            if rng is not None
            else int(np.random.randint(len(graphs_for_slot)))
        )
        accepted_graphs_by_slot[slot_idx] = graphs_for_slot[selected_idx]
        filled_now += 1
    return feasible_candidate_count, filled_now


def _validate_feasibility_mask(decoded_graphs: Sequence[nx.Graph], feasibility_mask: np.ndarray) -> None:
    if feasibility_mask.shape[0] != len(decoded_graphs):
        raise RuntimeError(
            "Feasibility estimator returned a mask of unexpected length "
            f"({feasibility_mask.shape[0]} for {len(decoded_graphs)} graphs)."
        )


def log_feasibility_attempt(
    owner: Any,
    *,
    graph_conditioning: Sequence[Any],
    accepted_graphs_by_slot: Sequence[Optional[nx.Graph]],
    rejected_slot_indices: Sequence[int],
    decoded_graphs: Sequence[nx.Graph],
    feasible_now: int,
    filled_now: int,
    attempt: int,
    attempt_started_at: float,
    feasibility_started_at: float,
) -> None:
    """Emit one feasibility-attempt status line when verbose logging is enabled."""
    if int(getattr(owner, "verbose", 0)) < 1:
        return
    pending_now = len(rejected_slot_indices)
    filled_total = sum(graph is not None for graph in accepted_graphs_by_slot)
    missing_total = len(graph_conditioning) - filled_total
    attempted_total = len(decoded_graphs)
    acceptance_rate = (feasible_now / attempted_total) if attempted_total > 0 else 0.0
    attempt_elapsed_seconds = time.perf_counter() - attempt_started_at
    total_elapsed_seconds = time.perf_counter() - feasibility_started_at
    verbose_log(
        owner,
        format_feasibility_attempt_status(
            attempt=attempt,
            max_attempts=owner.max_feasibility_attempts,
            attempted_total=attempted_total,
            feasible_now=feasible_now,
            filled_now=filled_now,
            pending_now=pending_now,
            acceptance_rate=acceptance_rate,
            filled_total=filled_total,
            missing_total=missing_total,
            attempt_elapsed_seconds=attempt_elapsed_seconds,
            total_elapsed_seconds=total_elapsed_seconds,
        ),
        level=1,
    )


def log_feasibility_summary(
    owner: Any,
    *,
    graph_conditioning: Sequence[Any],
    accepted_graphs_by_slot: Sequence[Optional[nx.Graph]],
    total_generated: int,
    total_feasible: int,
) -> None:
    """Emit the final feasibility-filtering summary when verbose logging is enabled."""
    if int(getattr(owner, "verbose", 0)) < 1:
        return
    accepted_count = sum(graph is not None for graph in accepted_graphs_by_slot)
    overall_rate = (total_feasible / total_generated) if total_generated > 0 else 0.0
    verbose_log(
        owner,
        "Feasibility filtering summary: "
        f"generated={total_generated}, feasible_candidates={total_feasible}, "
        f"feasible_rate={overall_rate:.1%}, "
        f"fulfilled_slots={accepted_count}/{len(graph_conditioning)}.",
        level=1,
    )


def decode_with_feasibility_slots_core(
    owner: Any,
    graph_conditioning: Sequence[Any],
    *,
    decode_attempt_fn: Callable[[Sequence[Any], int], list[nx.Graph]],
    return_stats: bool = False,
) -> Any:
    """Retry decoding until each slot has a feasible graph or attempts are exhausted."""
    accepted_graphs_by_slot: list[Optional[nx.Graph]] = [None] * len(graph_conditioning)
    pending_conditioning = graph_conditioning
    pending_slot_indices = list(range(len(graph_conditioning)))
    attempt = 0
    total_generated = 0
    total_feasible = 0
    feasibility_started_at = time.perf_counter()
    while len(pending_conditioning) > 0 and attempt < owner.max_feasibility_attempts:
        attempt += 1
        attempt_started_at = time.perf_counter()
        candidate_conditioning = owner._repeat_graph_conditioning(
            pending_conditioning,
            repeats=owner.feasibility_candidates_per_attempt,
        )
        candidate_slot_indices = [
            slot_idx
            for slot_idx in pending_slot_indices
            for _ in range(owner.feasibility_candidates_per_attempt)
        ]
        decoded_graphs = decode_attempt_fn(candidate_conditioning, attempt - 1)
        total_generated += len(decoded_graphs)
        feasibility_mask = np.asarray(owner.feasibility_estimator.predict(decoded_graphs), dtype=bool)
        _validate_feasibility_mask(decoded_graphs, feasibility_mask)
        feasible_now, filled_now = accept_feasible_candidates_by_slot(
            decoded_graphs=decoded_graphs,
            feasibility_mask=feasibility_mask.tolist(),
            candidate_slot_indices=candidate_slot_indices,
            accepted_graphs_by_slot=accepted_graphs_by_slot,
        )
        total_feasible += feasible_now
        rejected_slot_indices = [
            slot_idx for slot_idx in pending_slot_indices if accepted_graphs_by_slot[slot_idx] is None
        ]
        log_feasibility_attempt(
            owner,
            graph_conditioning=graph_conditioning,
            accepted_graphs_by_slot=accepted_graphs_by_slot,
            rejected_slot_indices=rejected_slot_indices,
            decoded_graphs=decoded_graphs,
            feasible_now=feasible_now,
            filled_now=filled_now,
            attempt=attempt,
            attempt_started_at=attempt_started_at,
            feasibility_started_at=feasibility_started_at,
        )
        if not rejected_slot_indices:
            break
        pending_slot_indices = rejected_slot_indices
        pending_conditioning = owner._slice_graph_conditioning(
            graph_conditioning,
            pending_slot_indices,
        )
    log_feasibility_summary(
        owner,
        graph_conditioning=graph_conditioning,
        accepted_graphs_by_slot=accepted_graphs_by_slot,
        total_generated=total_generated,
        total_feasible=total_feasible,
    )
    if return_stats:
        return accepted_graphs_by_slot, total_generated, total_feasible
    return accepted_graphs_by_slot


def finalize_feasibility_graphs(
    owner: Any,
    accepted_graphs_by_slot: Sequence[Optional[nx.Graph]],
    expected_count: int,
) -> list[nx.Graph]:
    """Apply the configured failure mode and return the recovered feasible graphs."""
    accepted_count = sum(graph is not None for graph in accepted_graphs_by_slot)
    if accepted_count != expected_count:
        if owner.feasibility_failure_mode == "raise":
            raise RuntimeError(
                "Feasibility filtering did not recover enough graphs: "
                f"accepted {accepted_count} of {expected_count} after "
                f"{owner.max_feasibility_attempts} attempts."
            )
        if int(getattr(owner, "verbose", 0)) >= 1:
            verbose_log(
                owner,
                "Feasibility filtering exhausted retries; returning only feasible graphs: "
                f"accepted {accepted_count} of {expected_count}.",
                level=1,
            )
    return [graph for graph in accepted_graphs_by_slot if graph is not None]


def score_feasible_rate(owner: Any, **kwargs: Any) -> dict[str, Any]:
    """Run feasible-rate scoring while temporarily overriding retry parameters."""
    n_samples = int(kwargs.pop("n_samples", 32))
    max_feasibility_attempts = kwargs.pop("max_feasibility_attempts", None)
    feasibility_candidates_per_attempt = kwargs.pop("feasibility_candidates_per_attempt", None)
    feasibility_oracle_candidates_per_attempt = kwargs.pop("feasibility_oracle_candidates_per_attempt", None)
    interpolate_between_n_samples = kwargs.pop("interpolate_between_n_samples", None)
    desired_target = kwargs.pop("desired_target", None)
    guidance_scale = float(kwargs.pop("guidance_scale", 1.0))
    verbose = bool(kwargs.pop("verbose", False))
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword arguments: {unknown}")
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")

    owner._require_fitted_for_generation()

    original_attempts = owner.max_feasibility_attempts
    original_candidates = owner.feasibility_candidates_per_attempt
    original_oracle_candidates = getattr(owner, "feasibility_oracle_candidates_per_attempt", 0)
    original_verbose = owner.verbose

    if max_feasibility_attempts is not None:
        owner.max_feasibility_attempts = int(max_feasibility_attempts)
    if feasibility_candidates_per_attempt is not None:
        owner.feasibility_candidates_per_attempt = int(feasibility_candidates_per_attempt)
    if feasibility_oracle_candidates_per_attempt is not None:
        owner.feasibility_oracle_candidates_per_attempt = int(feasibility_oracle_candidates_per_attempt)
    owner.verbose = original_verbose if verbose else 0

    try:
        graph_conditioning = owner._sample_conditions(
            n_samples,
            interpolate_between_n_samples=interpolate_between_n_samples,
        )

        if owner.feasibility_estimator is None:
            decoded_graphs = owner._decode_conditioning_batch(
                graph_conditioning,
                desired_target=desired_target,
                guidance_scale=guidance_scale,
            )
            accepted_slots = len(decoded_graphs)
            total_generated = len(decoded_graphs)
            total_feasible = len(decoded_graphs)
        elif not owner.use_feasibility_filtering:
            decoded_graphs = owner._decode_conditioning_batch(
                graph_conditioning,
                desired_target=desired_target,
                guidance_scale=guidance_scale,
            )
            feasibility_mask = np.asarray(owner.feasibility_estimator.predict(decoded_graphs), dtype=bool)
            _validate_feasibility_mask(decoded_graphs, feasibility_mask)
            accepted_slots = len(decoded_graphs)
            total_generated = len(decoded_graphs)
            total_feasible = int(feasibility_mask.sum())
        else:
            def _decode_attempt(candidate_conditioning: Sequence[Any], attempt_idx: int) -> list[nx.Graph]:
                try:
                    return owner._decode_conditioning_batch(
                        candidate_conditioning,
                        desired_target=desired_target,
                        guidance_scale=guidance_scale,
                        attempt_idx=attempt_idx,
                    )
                except TypeError:
                    return owner._decode_conditioning_batch(
                        candidate_conditioning,
                        desired_target=desired_target,
                        guidance_scale=guidance_scale,
                    )

            accepted_graphs_by_slot, total_generated, total_feasible = decode_with_feasibility_slots_core(
                owner,
                graph_conditioning,
                decode_attempt_fn=_decode_attempt,
                return_stats=True,
            )
            accepted_slots = sum(graph is not None for graph in accepted_graphs_by_slot)

        feasible_rate = (total_feasible / total_generated) if total_generated > 0 else 0.0
        fulfilled_rate = accepted_slots / n_samples
        return {
            "score": feasible_rate,
            "feasible_rate": feasible_rate,
            "fulfilled_rate": fulfilled_rate,
            "accepted_slots": accepted_slots,
            "n_samples": n_samples,
            "generated_candidates": total_generated,
            "feasible_candidates": total_feasible,
            "max_feasibility_attempts": int(owner.max_feasibility_attempts),
            "feasibility_candidates_per_attempt": int(owner.feasibility_candidates_per_attempt),
            "feasibility_oracle_candidates_per_attempt": int(
                getattr(owner, "feasibility_oracle_candidates_per_attempt", original_oracle_candidates)
            ),
        }
    finally:
        owner.max_feasibility_attempts = original_attempts
        owner.feasibility_candidates_per_attempt = original_candidates
        owner.feasibility_oracle_candidates_per_attempt = original_oracle_candidates
        owner.verbose = original_verbose
