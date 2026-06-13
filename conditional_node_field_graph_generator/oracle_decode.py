"""Oracle-guided adjacency solve orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import random
import time
from typing import Any, FrozenSet, List, Optional, Sequence

import numpy as np

from .decode_preparation import (
    build_masked_prob_matrix,
    build_single_generated_node_batch,
    resolve_predicted_edge_labels,
    resolve_predicted_node_labels,
)
from .decoder_assembly import (
    assemble_graph,
    edge_label_list_to_matrix,
    edge_label_matrix_to_list,
)
from .diagnostics import _plot_decoder_diagnostics
from .runtime_utils import run_with_fork_timeout, verbose_log

Edge = tuple[int, int]


@dataclass
class _OracleGraphState:
    graph_idx: int
    generated_nodes: Any
    desired_node_count: Optional[int]
    desired_edge_count: Optional[int]
    existence_mask: np.ndarray
    degree_predictions: np.ndarray
    masked_prob_matrix: np.ndarray
    target_degrees: List[int]
    current_node_labels: np.ndarray
    current_edge_label_matrix: np.ndarray
    predicted_edge_label_matrix: np.ndarray


def _prepare_oracle_graph_state(
    owner,
    generated_nodes,
    graph_conditioning,
    graph_idx: int,
    predicted_edge_probability_matrices,
    predicted_node_labels_list,
    predicted_edge_labels_list,
    predicted_edge_label_matrices,
) -> _OracleGraphState:
    single_generated_nodes = build_single_generated_node_batch(generated_nodes, graph_idx)
    desired_node_count = (
        None
        if graph_conditioning is None
        else int(np.asarray(graph_conditioning.node_counts)[graph_idx])
    )
    desired_edge_count = (
        None
        if graph_conditioning is None
        else int(np.asarray(graph_conditioning.edge_counts)[graph_idx])
    )
    existence_mask = owner.graph_decoder.resolve_node_presence_mask(
        np.asarray(single_generated_nodes.node_presence_mask[0], dtype=bool),
        desired_node_count=desired_node_count,
        node_existence_scores=(
            None
            if single_generated_nodes.node_existence_probabilities is None
            else np.asarray(
                single_generated_nodes.node_existence_probabilities[0],
                dtype=float,
            )
        ),
    )
    degree_predictions = np.asarray(
        single_generated_nodes.node_degree_predictions[0],
        dtype=float,
    )
    masked_prob_matrix = build_masked_prob_matrix(
        existence_mask,
        degree_predictions,
        np.asarray(predicted_edge_probability_matrices[graph_idx], dtype=float),
    )
    target_degrees = owner.graph_decoder.get_degree_targets(
        degree_predictions,
        existence_mask,
        desired_edge_count=desired_edge_count,
    )
    current_node_labels = np.asarray(
        predicted_node_labels_list[graph_idx],
        dtype=object,
    ).copy()
    current_edge_label_matrix = (
        np.asarray(predicted_edge_label_matrices[graph_idx], dtype=object).copy()
        if predicted_edge_label_matrices is not None
        else edge_label_list_to_matrix(
            np.asarray(predicted_edge_probability_matrices[graph_idx] > 0, dtype=int),
            (
                np.asarray(predicted_edge_labels_list[graph_idx], dtype=object)
                if predicted_edge_labels_list is not None
                else np.asarray([], dtype=object)
            ),
        )
    )
    return _OracleGraphState(
        graph_idx=graph_idx,
        generated_nodes=single_generated_nodes,
        desired_node_count=desired_node_count,
        desired_edge_count=desired_edge_count,
        existence_mask=existence_mask,
        degree_predictions=degree_predictions,
        masked_prob_matrix=masked_prob_matrix,
        target_degrees=target_degrees,
        current_node_labels=current_node_labels,
        current_edge_label_matrix=current_edge_label_matrix,
        predicted_edge_label_matrix=np.asarray(
            current_edge_label_matrix,
            dtype=object,
        ).copy(),
    )


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
    remaining = getattr(owner, "_remaining_generation_timeout_seconds", None)
    if callable(remaining):
        remaining_seconds = remaining()
        if remaining_seconds is not None:
            return float(remaining_seconds)
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
    return float(timeout_seconds)


def optimize_oracle_adjacency_matrix(
    owner,
    *args,
    deadline_monotonic: Optional[float] = None,
    **kwargs,
) -> np.ndarray:
    timeout_seconds = oracle_adjacency_timeout_seconds(owner)
    if deadline_monotonic is None and timeout_seconds is not None:
        deadline_monotonic = time.monotonic() + float(timeout_seconds)
    if deadline_monotonic is not None:
        timeout_seconds = float(deadline_monotonic) - time.monotonic()
        if timeout_seconds <= 0.0:
            raise TimeoutError("Oracle adjacency solve exhausted its shared time budget.")
        kwargs["_deadline_monotonic"] = deadline_monotonic
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
    timeout_seconds = oracle_adjacency_timeout_seconds(owner)
    deadline_monotonic = (
        None
        if timeout_seconds is None
        else time.monotonic() + float(timeout_seconds)
    )
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
                deadline_monotonic=deadline_monotonic,
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


def decode_generated_nodes_with_oracle(
    owner,
    generated_nodes: GeneratedNodeBatch,
    graph_conditioning: Optional[GraphConditioningBatch] = None,
) -> List[Optional[nx.Graph]]:
    """Decode one generated batch using the feasibility oracle, label repairs, and structural cuts."""
    from .oracle_utils import (
        enumerate_localized_edge_addition_proposals,
        update_oracle_edge_memory,
    )

    if owner.graph_decoder is None:
        raise RuntimeError(
            "Oracle-guided decoding requires owner.graph_decoder to be configured."
        )
    predicted_edge_probability_matrices = generated_nodes.edge_probability_matrices
    if predicted_edge_probability_matrices is None:
        raise RuntimeError(
            "Graph decoding requires explicit edge-probability matrices from the conditional node generator."
        )
    edge_existence_probabilities = generated_nodes.edge_existence_probabilities
    node_label_probabilities = generated_nodes.node_label_probabilities
    edge_label_probabilities = generated_nodes.edge_label_probabilities
    predicted_node_labels_list = resolve_predicted_node_labels(owner, generated_nodes)
    predicted_edge_labels_list, predicted_edge_label_matrices = resolve_predicted_edge_labels(
        owner,
        generated_nodes,
        predicted_edge_probability_matrices=predicted_edge_probability_matrices,
    )

    decoded_graphs: List[Optional[nx.Graph]] = []
    for graph_idx in range(len(predicted_edge_probability_matrices)):
        state = _prepare_oracle_graph_state(
            owner,
            generated_nodes,
            graph_conditioning,
            graph_idx,
            predicted_edge_probability_matrices,
            predicted_node_labels_list,
            predicted_edge_labels_list,
            predicted_edge_label_matrices,
        )
        single_generated_nodes = state.generated_nodes
        desired_node_count = state.desired_node_count
        desired_edge_count = state.desired_edge_count
        existence_mask = state.existence_mask
        degree_predictions = state.degree_predictions
        masked_prob_matrix = state.masked_prob_matrix
        target_degrees = state.target_degrees
        current_node_labels = state.current_node_labels
        current_edge_label_matrix = state.current_edge_label_matrix
        predicted_edge_label_matrix = state.predicted_edge_label_matrix
        try:
            single_adj_mtx = owner.graph_decoder.decode_adjacency_matrix(
                single_generated_nodes,
                predicted_edge_probability_matrices=[predicted_edge_probability_matrices[graph_idx]],
                desired_node_counts=None if desired_node_count is None else [desired_node_count],
                desired_edge_counts=None if desired_edge_count is None else [desired_edge_count],
            )[0]
        except TimeoutError:
            verbose_log(
                owner,
                "Oracle initial adjacency decode timed out; skipping oracle graph for this sample.",
                level=2,
            )
            decoded_graphs.append(None)
            continue
        except RuntimeError:
            verbose_log(
                owner,
                "Oracle initial adjacency decode failed under connectivity constraints; "
                "retrying with connectivity disabled for the seed solve.",
            )
            try:
                single_adj_mtx = optimize_oracle_adjacency_matrix(
                    owner,
                    masked_prob_matrix,
                    target_degrees,
                    target_edge_count=desired_edge_count,
                    connectivity=False,
                    active_node_mask=existence_mask,
                )
            except TimeoutError:
                verbose_log(
                    owner,
                    "Oracle fallback seed adjacency decode timed out; skipping oracle graph for this sample.",
                    level=2,
                )
                decoded_graphs.append(None)
                continue
            except Exception as fallback_exc:
                raise RuntimeError(
                    "Oracle initial adjacency decode failed before any oracle cuts could be applied."
                ) from fallback_exc
        accumulated_structural_cuts: List[FrozenSet[Edge]] = []
        accumulated_node_label_forbidden = []
        accumulated_edge_label_forbidden = []
        local_edge_violation_prior = np.zeros_like(masked_prob_matrix, dtype=float)
        best_graph: Optional[nx.Graph] = None
        best_score = float("-inf")
        best_feasible_graph: Optional[nx.Graph] = None
        best_feasible_score = float("-inf")
        current_total_log_score = float("nan")
        current_edge_log_score = float("nan")
        current_node_log_score = float("nan")
        current_edge_label_log_score = float("nan")
        use_node_label_cuts = bool(getattr(owner, "oracle_use_node_label_cuts", False))
        use_edge_label_cuts = bool(getattr(owner, "oracle_use_edge_label_cuts", False))
        add_edge_repair_budget = max(
            0,
            int(getattr(owner, "oracle_add_edge_repair_budget", 32)),
        )

        def plot_oracle_phase(
            phase_name: str,
            *,
            adj_mtx: np.ndarray,
            decoded_graph: nx.Graph,
            node_violation_sets: Sequence[Any],
            edge_violation_sets: Sequence[FrozenSet[Edge]],
            new_node_cut_count: int = 0,
            new_edge_label_cut_count: int = 0,
            joint_label_changed: bool = False,
            new_structural_cut_count: int = 0,
            detail: Optional[str] = None,
        ) -> None:
            if int(owner.verbose) < 4:
                return
            phase_title = f"Oracle {phase_name}"
            if detail:
                phase_title += f" [{detail}]"
            _plot_decoder_diagnostics(
                prob_matrix=masked_prob_matrix,
                adj_mtx=adj_mtx,
                target_degrees=target_degrees,
                title=(
                    f"{phase_title} graph={graph_idx} iteration={_iteration_idx + 1} "
                    f"| violating_node_sets={len(node_violation_sets)} "
                    f"| violating_edge_sets={len(edge_violation_sets)} "
                    f"| new_node_cuts={int(new_node_cut_count)} "
                    f"| new_edge_label_cuts={int(new_edge_label_cut_count)} "
                    f"| joint_label_changed={int(bool(joint_label_changed))} "
                    f"| log_total={current_total_log_score:.3f} "
                    f"| log_edge={current_edge_log_score:.3f} "
                    f"| log_node={current_node_log_score:.3f} "
                    f"| log_edge_label={current_edge_label_log_score:.3f} "
                    f"| best_log_total={best_score:.3f} "
                    f"| best_feasible_log_total={best_feasible_score:.3f} "
                    f"| new_structural_cuts={int(new_structural_cut_count)} "
                    f"| accepted_structural_cuts={len(accumulated_structural_cuts) + int(new_structural_cut_count)}"
                ),
                violating_edge_sets=edge_violation_sets,
                decoded_graph=decoded_graph,
                graph_renderer=owner.graph_decoder.diagnostic_graph_renderer,
                node_label_probabilities=None if node_label_probabilities is None else np.asarray(
                    node_label_probabilities[graph_idx],
                    dtype=float,
                ),
                node_label_names=owner._get_node_label_names(),
                node_labels=np.asarray(current_node_labels, dtype=object),
                existence_mask=existence_mask,
            )

        def evaluate_oracle_state(
            node_labels: np.ndarray,
            edge_label_matrix: np.ndarray,
        ) -> tuple[nx.Graph, List[Any], List[FrozenSet[Edge]], float, float, float, float]:
            candidate_graph = assemble_graph(
                existence_mask,
                np.asarray(node_labels, dtype=object),
                np.asarray(edge_label_matrix_to_list(single_adj_mtx, edge_label_matrix), dtype=object),
                np.asarray(single_adj_mtx, dtype=float),
            )
            node_violation_sets = (
                owner._get_oracle_node_violation_sets(candidate_graph, n_nodes=single_adj_mtx.shape[0])
                if use_node_label_cuts
                else []
            )
            edge_violation_sets = owner._get_oracle_edge_violation_sets(
                candidate_graph,
                n_nodes=single_adj_mtx.shape[0],
            )
            score, edge_score, node_score, edge_label_score = owner._oracle_candidate_score_components(
                existence_mask=existence_mask,
                adj_mtx=single_adj_mtx,
                node_labels=np.asarray(node_labels, dtype=object),
                edge_label_matrix=np.asarray(edge_label_matrix, dtype=object),
                edge_probability_matrix=np.asarray(
                    edge_existence_probabilities[graph_idx]
                    if edge_existence_probabilities is not None
                    else predicted_edge_probability_matrices[graph_idx],
                    dtype=float,
                ),
                node_label_probabilities=None if node_label_probabilities is None else np.asarray(
                    node_label_probabilities[graph_idx],
                    dtype=float,
                ),
                edge_label_probabilities=None if edge_label_probabilities is None else np.asarray(
                    edge_label_probabilities[graph_idx],
                    dtype=float,
                ),
            )
            return (
                candidate_graph,
                node_violation_sets,
                edge_violation_sets,
                float(score),
                float(edge_score),
                float(node_score),
                float(edge_label_score),
            )

        def oracle_rank(
            node_violation_sets: Sequence[Any],
            edge_violation_sets: Sequence[FrozenSet[Edge]],
            score: float,
        ) -> tuple[float, float, float, float]:
            node_count = len(node_violation_sets) if use_node_label_cuts else 0
            edge_count = len(edge_violation_sets)
            return (edge_count + node_count, edge_count, node_count, -float(score))

        for _iteration_idx in range(owner.max_oracle_iterations):
            edge_labels = owner.graph_decoder.decode_edge_labels(
                single_generated_nodes,
                [single_adj_mtx],
                predicted_edge_labels_list=None if predicted_edge_labels_list is None else [
                    predicted_edge_labels_list[graph_idx]
                ],
                predicted_edge_label_matrices=None if predicted_edge_label_matrices is None else [
                    current_edge_label_matrix
                ],
            )[0]
            current_edge_label_matrix = edge_label_list_to_matrix(single_adj_mtx, edge_labels)
            (
                graph,
                current_node_violation_sets,
                current_edge_violation_sets,
                score,
                edge_score,
                node_score,
                edge_label_score,
            ) = evaluate_oracle_state(current_node_labels, current_edge_label_matrix)

            can_try_edge_additions = (
                add_edge_repair_budget > 0
                and bool(current_edge_violation_sets)
                and owner.feasibility_estimator is not None
                and hasattr(owner.feasibility_estimator, "number_of_violations")
                and (
                    desired_edge_count is None
                    or owner.graph_decoder.edge_count_slack_penalty is not None
                )
            )
            if can_try_edge_additions:
                edge_probability_matrix = np.asarray(
                    edge_existence_probabilities[graph_idx]
                    if edge_existence_probabilities is not None
                    else predicted_edge_probability_matrices[graph_idx],
                    dtype=float,
                )
                graph_edge_label_probabilities = (
                    None
                    if edge_label_probabilities is None
                    else np.asarray(edge_label_probabilities[graph_idx], dtype=float)
                )
                proposals = enumerate_localized_edge_addition_proposals(
                    adjacency_matrix=single_adj_mtx,
                    violating_edge_sets=current_edge_violation_sets,
                    active_node_mask=existence_mask,
                    edge_probability_matrix=edge_probability_matrix,
                    edge_label_classes=owner._get_edge_label_names(),
                    edge_label_probabilities=graph_edge_label_probabilities,
                    predicted_edge_label_matrix=predicted_edge_label_matrix,
                    budget=add_edge_repair_budget,
                )
                if proposals:
                    current_violation_counts = np.asarray(
                        owner.feasibility_estimator.number_of_violations([graph]),
                        dtype=int,
                    ).reshape(-1)
                    if current_violation_counts.size != 1:
                        raise ValueError(
                            "feasibility_estimator.number_of_violations() must return "
                            "one count per input graph."
                        )
                    current_violation_count = int(current_violation_counts[0])
                    candidate_states = []
                    candidate_graphs = []
                    for proposal in proposals:
                        candidate_adj_mtx = np.asarray(single_adj_mtx, dtype=int).copy()
                        i, j = proposal.edge
                        candidate_adj_mtx[i, j] = candidate_adj_mtx[j, i] = 1
                        candidate_edge_label_matrix = np.asarray(
                            current_edge_label_matrix,
                            dtype=object,
                        ).copy()
                        candidate_edge_label_matrix[i, j] = proposal.label
                        candidate_edge_label_matrix[j, i] = proposal.label
                        candidate_graph = assemble_graph(
                            existence_mask,
                            current_node_labels,
                            np.asarray(
                                edge_label_matrix_to_list(
                                    candidate_adj_mtx,
                                    candidate_edge_label_matrix,
                                ),
                                dtype=object,
                            ),
                            candidate_adj_mtx,
                        )
                        candidate_states.append(
                            (
                                proposal,
                                candidate_adj_mtx,
                                candidate_edge_label_matrix,
                                candidate_graph,
                            )
                        )
                        candidate_graphs.append(candidate_graph)

                    candidate_violation_counts = np.asarray(
                        owner.feasibility_estimator.number_of_violations(candidate_graphs),
                        dtype=int,
                    ).reshape(-1)
                    if candidate_violation_counts.size != len(candidate_graphs):
                        raise ValueError(
                            "feasibility_estimator.number_of_violations() must return "
                            "one count per input graph."
                        )
                    improving_candidates = []
                    for candidate_state, violation_count in zip(
                        candidate_states,
                        candidate_violation_counts,
                    ):
                        if int(violation_count) >= current_violation_count:
                            continue
                        proposal, candidate_adj_mtx, candidate_edge_label_matrix, candidate_graph = candidate_state
                        candidate_scores = owner._oracle_candidate_score_components(
                            existence_mask=existence_mask,
                            adj_mtx=candidate_adj_mtx,
                            node_labels=current_node_labels,
                            edge_label_matrix=candidate_edge_label_matrix,
                            edge_probability_matrix=edge_probability_matrix,
                            node_label_probabilities=(
                                None
                                if node_label_probabilities is None
                                else np.asarray(node_label_probabilities[graph_idx], dtype=float)
                            ),
                            edge_label_probabilities=graph_edge_label_probabilities,
                        )
                        edge_count_deviation = (
                            0
                            if desired_edge_count is None
                            else abs(
                                int(np.sum(candidate_adj_mtx) // 2)
                                - int(desired_edge_count)
                            )
                        )
                        edge_count_penalty = (
                            0.0
                            if owner.graph_decoder.edge_count_slack_penalty is None
                            else float(owner.graph_decoder.edge_count_slack_penalty)
                            * float(edge_count_deviation)
                        )
                        improving_candidates.append(
                            (
                                (
                                    int(violation_count),
                                    -float(candidate_scores[0]),
                                    edge_count_penalty,
                                    proposal.edge[0],
                                    proposal.edge[1],
                                    repr(proposal.label),
                                ),
                                proposal,
                                candidate_adj_mtx,
                                candidate_edge_label_matrix,
                                candidate_graph,
                                candidate_scores,
                                int(violation_count),
                                edge_count_deviation,
                            )
                        )

                    if improving_candidates:
                        (
                            _,
                            selected_proposal,
                            single_adj_mtx,
                            current_edge_label_matrix,
                            graph,
                            selected_scores,
                            selected_violation_count,
                            selected_edge_count_deviation,
                        ) = min(improving_candidates, key=lambda item: item[0])
                        (
                            score,
                            edge_score,
                            node_score,
                            edge_label_score,
                        ) = selected_scores
                        current_total_log_score = float(score)
                        current_edge_log_score = float(edge_score)
                        current_node_log_score = float(node_score)
                        current_edge_label_log_score = float(edge_label_score)
                        current_node_violation_sets = (
                            owner._get_oracle_node_violation_sets(
                                graph,
                                n_nodes=single_adj_mtx.shape[0],
                            )
                            if use_node_label_cuts
                            else []
                        )
                        current_edge_violation_sets = owner._get_oracle_edge_violation_sets(
                            graph,
                            n_nodes=single_adj_mtx.shape[0],
                        )
                        if score > best_score:
                            best_score = float(score)
                            best_graph = graph
                        if selected_violation_count == 0 and score > best_feasible_score:
                            best_feasible_score = float(score)
                            best_feasible_graph = graph
                        plot_oracle_phase(
                            "Constructive Edge Addition",
                            adj_mtx=single_adj_mtx,
                            decoded_graph=graph,
                            node_violation_sets=current_node_violation_sets,
                            edge_violation_sets=current_edge_violation_sets,
                            detail=(
                                f"proposals={len(proposals)} "
                                f"improving={len(improving_candidates)} "
                                f"selected={selected_proposal.edge!r}:{selected_proposal.label!r} "
                                f"violations={current_violation_count}->{selected_violation_count} "
                                f"edge_count_deviation={selected_edge_count_deviation}"
                            ),
                        )
                        if selected_violation_count == 0:
                            break
                        continue

            new_node_forbidden = []
            if use_node_label_cuts and node_label_probabilities is not None:
                new_node_forbidden = [
                    assignment
                    for assignment in owner._forbidden_node_label_assignment_from_sets(
                        current_node_violation_sets,
                        current_node_labels,
                    )
                    if assignment not in accumulated_node_label_forbidden
                ]
            new_edge_label_forbidden = []
            if use_edge_label_cuts and edge_label_probabilities is not None:
                new_edge_label_forbidden = [
                    assignment
                    for assignment in owner._forbidden_edge_label_assignment_from_sets(
                        current_edge_violation_sets,
                        current_edge_label_matrix,
                    )
                    if assignment not in accumulated_edge_label_forbidden
                ]
            if new_node_forbidden:
                accumulated_node_label_forbidden.extend(new_node_forbidden)
            if new_edge_label_forbidden:
                accumulated_edge_label_forbidden.extend(new_edge_label_forbidden)

            current_rank = oracle_rank(current_node_violation_sets, current_edge_violation_sets, score)
            joint_label_changed = False
            joint_label_detail = "label follow-up disabled"
            can_repair_joint_labels = (
                (
                    use_node_label_cuts
                    and accumulated_node_label_forbidden
                    and node_label_probabilities is not None
                )
                or (
                    use_edge_label_cuts
                    and accumulated_edge_label_forbidden
                    and edge_label_probabilities is not None
                )
            )
            if can_repair_joint_labels:
                repaired_node_labels, repaired_edge_label_matrix = owner._repair_labels_with_oracle(
                    existence_mask=existence_mask,
                    adj_mtx=single_adj_mtx,
                    current_node_labels=current_node_labels,
                    current_edge_label_matrix=current_edge_label_matrix,
                    node_label_probabilities=(
                        None
                        if node_label_probabilities is None
                        else np.asarray(node_label_probabilities[graph_idx], dtype=float)
                    ),
                    edge_label_probabilities=(
                        None
                        if edge_label_probabilities is None
                        else np.asarray(edge_label_probabilities[graph_idx], dtype=float)
                    ),
                    forbidden_node_assignments=accumulated_node_label_forbidden,
                    forbidden_edge_assignments=accumulated_edge_label_forbidden,
                )
                repaired_edge_label_matrix = owner._fill_unlabeled_active_edges(
                    adj_mtx=single_adj_mtx,
                    edge_label_matrix=repaired_edge_label_matrix,
                    edge_label_probabilities=(
                        None
                        if edge_label_probabilities is None
                        else np.asarray(edge_label_probabilities[graph_idx], dtype=float)
                    ),
                )
                labels_changed = not np.array_equal(repaired_node_labels, current_node_labels)
                edge_labels_changed = not np.array_equal(repaired_edge_label_matrix, current_edge_label_matrix)
                if labels_changed or edge_labels_changed:
                    candidate = evaluate_oracle_state(repaired_node_labels, repaired_edge_label_matrix)
                    candidate_rank = oracle_rank(candidate[1], candidate[2], candidate[3])
                    if candidate_rank < current_rank:
                        current_node_labels = repaired_node_labels
                        current_edge_label_matrix = repaired_edge_label_matrix
                        (
                            graph,
                            current_node_violation_sets,
                            current_edge_violation_sets,
                            score,
                            edge_score,
                            node_score,
                            edge_label_score,
                        ) = candidate
                        current_rank = candidate_rank
                        joint_label_changed = True
                        if labels_changed and edge_labels_changed:
                            joint_label_detail = "node+edge labels changed"
                        elif labels_changed:
                            joint_label_detail = "node labels changed"
                        else:
                            joint_label_detail = "edge labels changed"
                    else:
                        joint_label_detail = "joint label follow-up rejected"
                else:
                    joint_label_detail = "joint label follow-up unchanged"

            if not joint_label_changed and node_label_probabilities is None and edge_label_probabilities is None:
                joint_label_detail = "label probabilities unavailable"
            elif not joint_label_changed and (
                (use_node_label_cuts and accumulated_node_label_forbidden)
                or (use_edge_label_cuts and accumulated_edge_label_forbidden)
            ) and "rejected" not in joint_label_detail and "unchanged" not in joint_label_detail:
                joint_label_detail = "label follow-up attempted but unchanged"

            current_total_log_score = float(score)
            current_edge_log_score = float(edge_score)
            current_node_log_score = float(node_score)
            current_edge_label_log_score = float(edge_label_score)
            plot_oracle_phase(
                "Soft Label Follow-Up",
                adj_mtx=single_adj_mtx,
                decoded_graph=graph,
                node_violation_sets=current_node_violation_sets,
                edge_violation_sets=current_edge_violation_sets,
                new_node_cut_count=len(new_node_forbidden),
                new_edge_label_cut_count=len(new_edge_label_forbidden),
                joint_label_changed=joint_label_changed,
                detail=joint_label_detail,
            )
            if score > best_score:
                best_score = score
                best_graph = graph

            is_feasible = not current_node_violation_sets and not current_edge_violation_sets
            if is_feasible and score > best_feasible_score:
                best_feasible_score = score
                best_feasible_graph = graph
                plot_oracle_phase(
                    "Feasibility Check",
                    adj_mtx=single_adj_mtx,
                    decoded_graph=graph,
                    node_violation_sets=current_node_violation_sets,
                    edge_violation_sets=current_edge_violation_sets,
                    joint_label_changed=joint_label_changed,
                )
                break

            persistent_structural_cuts = [
                edge_set for edge_set in current_edge_violation_sets if edge_set not in accumulated_structural_cuts
            ]
            if not new_node_forbidden and not new_edge_label_forbidden and not persistent_structural_cuts and not joint_label_changed:
                plot_oracle_phase(
                    "Feasibility Check",
                    adj_mtx=single_adj_mtx,
                    decoded_graph=graph,
                    node_violation_sets=current_node_violation_sets,
                    edge_violation_sets=current_edge_violation_sets,
                    joint_label_changed=joint_label_changed,
                )
                break

            local_edge_violation_prior = update_oracle_edge_memory(
                local_edge_violation_prior,
                current_edge_violation_sets,
                update_weight=owner.oracle_edge_memory_update,
                decay=owner.oracle_edge_memory_decay,
                clip_value=owner.oracle_edge_memory_clip,
            )
            accumulated_structural_cuts.extend(persistent_structural_cuts)
            if _iteration_idx + 1 >= owner.max_oracle_iterations:
                plot_oracle_phase(
                    "Structural Edge-Set Phase",
                    adj_mtx=single_adj_mtx,
                    decoded_graph=graph,
                    node_violation_sets=current_node_violation_sets,
                    edge_violation_sets=current_edge_violation_sets,
                    joint_label_changed=joint_label_changed,
                    new_structural_cut_count=len(persistent_structural_cuts),
                )
                break
            try:
                single_adj_mtx = solve_oracle_relaxed_adjacency(
                    owner,
                    masked_prob_matrix=masked_prob_matrix,
                    target_degrees=target_degrees,
                    accumulated_cuts=accumulated_structural_cuts,
                    start_iteration_idx=_iteration_idx + 1,
                    target_edge_count=desired_edge_count,
                    edge_violation_prior=local_edge_violation_prior,
                    active_node_mask=existence_mask,
                )
            except TimeoutError:
                verbose_log(
                    owner,
                    "Oracle-guided adjacency refinement timed out; "
                    "returning the best graph seen before the slow refinement.",
                    level=2,
                )
                break
            current_edge_label_matrix = owner._fill_unlabeled_active_edges(
                adj_mtx=single_adj_mtx,
                edge_label_matrix=current_edge_label_matrix,
                edge_label_probabilities=None if edge_label_probabilities is None else np.asarray(
                    edge_label_probabilities[graph_idx],
                    dtype=float,
                ),
            )
            if (
                (
                    use_node_label_cuts
                    and accumulated_node_label_forbidden
                    and node_label_probabilities is not None
                )
                or (
                    use_edge_label_cuts
                    and accumulated_edge_label_forbidden
                    and edge_label_probabilities is not None
                )
            ):
                current_node_labels, current_edge_label_matrix = owner._repair_labels_with_oracle(
                    existence_mask=existence_mask,
                    adj_mtx=single_adj_mtx,
                    current_node_labels=current_node_labels,
                    current_edge_label_matrix=current_edge_label_matrix,
                    node_label_probabilities=(
                        None
                        if node_label_probabilities is None
                        else np.asarray(node_label_probabilities[graph_idx], dtype=float)
                    ),
                    edge_label_probabilities=(
                        None
                        if edge_label_probabilities is None
                        else np.asarray(edge_label_probabilities[graph_idx], dtype=float)
                    ),
                    forbidden_node_assignments=accumulated_node_label_forbidden,
                    forbidden_edge_assignments=accumulated_edge_label_forbidden,
                )
            structural_graph = assemble_graph(
                existence_mask,
                current_node_labels,
                np.asarray(edge_label_matrix_to_list(single_adj_mtx, current_edge_label_matrix), dtype=object),
                np.asarray(single_adj_mtx, dtype=float),
            )
            structural_node_violation_sets = (
                owner._get_oracle_node_violation_sets(structural_graph, n_nodes=single_adj_mtx.shape[0])
                if use_node_label_cuts
                else []
            )
            structural_edge_violation_sets = owner._get_oracle_edge_violation_sets(
                structural_graph,
                n_nodes=single_adj_mtx.shape[0],
            )
            plot_oracle_phase(
                "Structural Edge-Set Phase",
                adj_mtx=single_adj_mtx,
                decoded_graph=structural_graph,
                node_violation_sets=structural_node_violation_sets,
                edge_violation_sets=structural_edge_violation_sets,
                joint_label_changed=False,
                new_structural_cut_count=len(persistent_structural_cuts),
                detail="post-structural solve",
            )
        final_graph = best_feasible_graph if best_feasible_graph is not None else best_graph
        if final_graph is None:
            raise RuntimeError("Oracle-guided decoding failed to assemble a graph.")
        decoded_graphs.append(final_graph)
    return decoded_graphs

__all__ = [
    "decode_generated_nodes_with_oracle",
    "optimize_oracle_adjacency_matrix",
    "oracle_adjacency_timeout_seconds",
    "sample_oracle_cuts_for_iteration",
    "solve_oracle_relaxed_adjacency",
]
