"""Decode orchestration service for graph generation."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Optional, Sequence, Union

import networkx as nx
import numpy as np

from .conditional_node_field_generator import GeneratedNodeBatch, GraphConditioningBatch
from .conditional_node_field_graph_decoder import decode_generated_nodes
from .decode_pipeline import (
    decode_with_feasibility_slots_core,
    finalize_feasibility_graphs,
    should_apply_feasibility_filtering,
)
from .runtime_utils import verbose_log


@dataclass
class DecodeService:
    owner: Any

    def _record_decode_summary(
        self,
        *,
        requested_count: int,
        generated_count: int,
        candidate_feasible_count: int,
        feasible_count: int,
        unfiltered_count: int,
        rejected_count: int,
    ) -> dict[str, Union[int, float]]:
        returned_count = int(feasible_count) + int(unfiltered_count)
        denominator = max(1, int(requested_count))
        generated_denominator = max(1, int(generated_count))
        summary = {
            "requested": int(requested_count),
            "returned": int(returned_count),
            "generated": int(generated_count),
            "candidate_feasible": int(candidate_feasible_count),
            "candidate_feasible_fraction": float(candidate_feasible_count) / float(generated_denominator),
            "feasible": int(feasible_count),
            "feasible_fraction": float(feasible_count) / float(denominator),
            "unfiltered": int(unfiltered_count),
            "unfiltered_fraction": float(unfiltered_count) / float(denominator),
            "rejected": int(rejected_count),
            "rejected_fraction": float(rejected_count) / float(denominator),
        }
        self.owner.last_decode_summary_ = summary
        if int(getattr(self.owner, "verbose", 0)) >= 1:
            verbose_log(
                self.owner,
                "Generation summary: "
                f"requested={summary['requested']}, returned={summary['returned']}, "
                f"generated={summary['generated']}, "
                f"candidate_feasible={summary['candidate_feasible']} "
                f"({summary['candidate_feasible_fraction']:.1%}), "
                f"feasible={summary['feasible']} ({summary['feasible_fraction']:.1%}), "
                f"unfiltered={summary['unfiltered']} ({summary['unfiltered_fraction']:.1%}), "
                f"rejected={summary['rejected']} ({summary['rejected_fraction']:.1%}).",
                level=1,
            )
        return summary

    def _should_fallback_to_unfiltered(self) -> bool:
        return getattr(self.owner, "feasibility_rejection_mode", "fallback_unfiltered") == "fallback_unfiltered"

    def _should_use_candidate_fallback(self) -> bool:
        return getattr(self.owner, "feasibility_fallback_strategy", None) == "best_candidate"

    def _decode_unfiltered_backup_conditioning(
        self,
        graph_conditioning: GraphConditioningBatch,
        *,
        sampling_mode: str,
        desired_target: Optional[Union[int, float, Sequence[Any]]] = None,
        guidance_scale: float = 1.0,
        predictor_scale: float = 1.0,
        desired_class: Optional[Union[int, Sequence[Any]]] = None,
        classifier_scale: float = 1.0,
        use_ilp_decoder: bool = True,
        edge_probability_threshold: Optional[float] = None,
        attempt_idx: int = 0,
    ) -> list[nx.Graph]:
        owner = self.owner
        previous_deadline = owner._get_generation_timeout_deadline()
        previous_max_feasibility_seconds = getattr(owner, "max_feasibility_seconds_per_sample", None)
        graph_decoder = getattr(owner, "graph_decoder", None)
        previous_active_time_limit = None if graph_decoder is None else getattr(
            graph_decoder,
            "active_time_limit_seconds",
            None,
        )
        try:
            owner._restore_generation_timeout_deadline(None)
            owner.max_feasibility_seconds_per_sample = None
            if graph_decoder is not None:
                graph_decoder.active_time_limit_seconds = previous_active_time_limit
            return self.decode_conditioning_batch(
                graph_conditioning,
                sampling_mode=sampling_mode,
                desired_target=desired_target,
                guidance_scale=guidance_scale,
                predictor_scale=predictor_scale,
                desired_class=desired_class,
                classifier_scale=classifier_scale,
                feasibility_oracle_candidates_per_attempt=0,
                use_ilp_decoder=use_ilp_decoder,
                edge_probability_threshold=edge_probability_threshold,
                attempt_idx=attempt_idx,
            )
        finally:
            owner.max_feasibility_seconds_per_sample = previous_max_feasibility_seconds
            if graph_decoder is not None:
                graph_decoder.active_time_limit_seconds = previous_active_time_limit
            owner._restore_generation_timeout_deadline(previous_deadline)

    def _decode_single_conditioning_with_timeout(
        self,
        graph_conditioning: GraphConditioningBatch,
        *,
        sampling_mode: str,
        desired_target: Optional[Union[int, float, Sequence[Any]]] = None,
        guidance_scale: float = 1.0,
        predictor_scale: float = 1.0,
        desired_class: Optional[Union[int, Sequence[Any]]] = None,
        classifier_scale: float = 1.0,
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
        use_ilp_decoder: bool = True,
        edge_probability_threshold: Optional[float] = None,
        timeout_seconds: float,
    ) -> tuple[Optional[nx.Graph], str, int, int]:
        owner = self.owner
        previous_deadline = owner._set_generation_timeout_deadline(timeout_seconds)
        previous_active_time_limit = None if owner.graph_decoder is None else getattr(
            owner.graph_decoder,
            "active_time_limit_seconds",
            None,
        )
        if owner.graph_decoder is not None:
            owner.graph_decoder.active_time_limit_seconds = owner._resolve_solver_time_limit_seconds(
                getattr(owner.graph_decoder, "adjacency_time_limit_seconds", None)
        )
        try:
            core_result = decode_with_feasibility_slots_core(
                owner,
                graph_conditioning,
                decode_attempt_fn=lambda candidate_conditioning, attempt_idx: self.decode_conditioning_batch(
                    candidate_conditioning,
                    sampling_mode=sampling_mode,
                    desired_target=desired_target,
                    guidance_scale=guidance_scale,
                    predictor_scale=predictor_scale,
                    desired_class=desired_class,
                    classifier_scale=classifier_scale,
                    feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
                    use_ilp_decoder=use_ilp_decoder,
                    edge_probability_threshold=edge_probability_threshold,
                    attempt_idx=attempt_idx,
                ),
                return_stats=True,
                return_fallbacks=self._should_use_candidate_fallback(),
            )
            if self._should_use_candidate_fallback():
                accepted, total_generated, total_feasible, fallback_graphs = core_result
            else:
                accepted, total_generated, total_feasible = core_result
                fallback_graphs = [None]
        except RuntimeError:
            accepted = [None]
            fallback_graphs = [None]
            total_generated = 0
            total_feasible = 0
        finally:
            if owner.graph_decoder is not None:
                owner.graph_decoder.active_time_limit_seconds = previous_active_time_limit
            owner._restore_generation_timeout_deadline(previous_deadline)

        if accepted and accepted[0] is not None:
            return accepted[0], "feasible", int(total_generated), int(total_feasible)
        if not self._should_fallback_to_unfiltered():
            return None, "rejected", int(total_generated), int(total_feasible)
        if self._should_use_candidate_fallback() and fallback_graphs and fallback_graphs[0] is not None:
            return fallback_graphs[0], "unfiltered", int(total_generated), int(total_feasible)
        fallback_attempt_idx = max(
            0,
            int(getattr(owner, "feasibility_oracle_candidates_per_attempt", 0)),
        )
        fallback_graphs = self._decode_unfiltered_backup_conditioning(
            graph_conditioning,
            sampling_mode=sampling_mode,
            desired_target=desired_target,
            guidance_scale=guidance_scale,
            predictor_scale=predictor_scale,
            desired_class=desired_class,
            classifier_scale=classifier_scale,
            use_ilp_decoder=use_ilp_decoder,
            edge_probability_threshold=edge_probability_threshold,
            attempt_idx=fallback_attempt_idx,
        )
        if not fallback_graphs:
            return None, "rejected", int(total_generated), int(total_feasible)
        return fallback_graphs[0], "unfiltered", int(total_generated), int(total_feasible)

    def _decode_single_conditioning_unfiltered_with_timeout(
        self,
        graph_conditioning: GraphConditioningBatch,
        *,
        sampling_mode: str,
        desired_target: Optional[Union[int, float, Sequence[Any]]] = None,
        guidance_scale: float = 1.0,
        predictor_scale: float = 1.0,
        desired_class: Optional[Union[int, Sequence[Any]]] = None,
        classifier_scale: float = 1.0,
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
        use_ilp_decoder: bool = True,
        edge_probability_threshold: Optional[float] = None,
        timeout_seconds: float,
        attempt_idx: int = 0,
    ) -> Optional[nx.Graph]:
        owner = self.owner
        previous_deadline = owner._set_generation_timeout_deadline(timeout_seconds)
        graph_decoder = getattr(owner, "graph_decoder", None)
        previous_active_time_limit = None if graph_decoder is None else getattr(
            graph_decoder,
            "active_time_limit_seconds",
            None,
        )
        if graph_decoder is not None:
            graph_decoder.active_time_limit_seconds = owner._resolve_solver_time_limit_seconds(
                getattr(graph_decoder, "adjacency_time_limit_seconds", None)
            )
        started_at = time.perf_counter()
        try:
            decoded_graphs = self.decode_conditioning_batch(
                graph_conditioning,
                sampling_mode=sampling_mode,
                desired_target=desired_target,
                guidance_scale=guidance_scale,
                predictor_scale=predictor_scale,
                desired_class=desired_class,
                classifier_scale=classifier_scale,
                feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
                use_ilp_decoder=use_ilp_decoder,
                edge_probability_threshold=edge_probability_threshold,
                attempt_idx=attempt_idx,
            )
        finally:
            if graph_decoder is not None:
                graph_decoder.active_time_limit_seconds = previous_active_time_limit
            owner._restore_generation_timeout_deadline(previous_deadline)

        elapsed_seconds = time.perf_counter() - started_at
        if elapsed_seconds > float(timeout_seconds):
            raise TimeoutError(
                f"Decode attempt exceeded {float(timeout_seconds):.1f}s "
                f"(elapsed={elapsed_seconds:.1f}s)."
            )
        if not decoded_graphs:
            return None
        return decoded_graphs[0]

    def _decode_unfiltered_slots_with_timeout(
        self,
        graph_conditioning: GraphConditioningBatch,
        *,
        sampling_mode: str,
        desired_target: Optional[Union[int, float, Sequence[Any]]] = None,
        guidance_scale: float = 1.0,
        predictor_scale: float = 1.0,
        desired_class: Optional[Union[int, Sequence[Any]]] = None,
        classifier_scale: float = 1.0,
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
        use_ilp_decoder: bool = True,
        edge_probability_threshold: Optional[float] = None,
        timeout_seconds: float,
    ) -> list[Optional[nx.Graph]]:
        owner = self.owner
        max_attempts = max(1, int(getattr(owner, "max_decode_attempts_per_sample", 1)))
        decoded_slots: list[Optional[nx.Graph]] = []
        returned_count = 0
        rejected_count = 0
        total_attempts = 0
        for slot_idx in range(len(graph_conditioning)):
            slot_conditioning = owner._slice_graph_conditioning(graph_conditioning, [slot_idx])
            accepted_graph = None
            for attempt_idx in range(max_attempts):
                total_attempts += 1
                try:
                    accepted_graph = self._decode_single_conditioning_unfiltered_with_timeout(
                        slot_conditioning,
                        sampling_mode=sampling_mode,
                        desired_target=desired_target,
                        guidance_scale=guidance_scale,
                        predictor_scale=predictor_scale,
                        desired_class=desired_class,
                        classifier_scale=classifier_scale,
                        feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
                        use_ilp_decoder=use_ilp_decoder,
                        edge_probability_threshold=edge_probability_threshold,
                        timeout_seconds=timeout_seconds,
                        attempt_idx=attempt_idx,
                    )
                except (RuntimeError, TimeoutError) as exc:
                    if int(getattr(owner, "verbose", 0)) >= 2:
                        verbose_log(
                            owner,
                            "Decode attempt failed; retrying with a fresh sample "
                            f"(slot={slot_idx}, attempt={attempt_idx + 1}/{max_attempts}, error={exc}).",
                            level=2,
                        )
                    accepted_graph = None
                if accepted_graph is not None:
                    break
            decoded_slots.append(accepted_graph)
            if accepted_graph is None:
                rejected_count += 1
            else:
                returned_count += 1
        self._record_decode_summary(
            requested_count=len(graph_conditioning),
            generated_count=total_attempts,
            candidate_feasible_count=returned_count,
            feasible_count=0,
            unfiltered_count=returned_count,
            rejected_count=rejected_count,
        )
        return decoded_slots

    def decode_generated_nodes(
        self,
        generated_nodes: GeneratedNodeBatch,
        graph_conditioning: Optional[GraphConditioningBatch] = None,
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
        attempt_idx: int = 0,
        use_ilp_decoder: bool = True,
        edge_probability_threshold: Optional[float] = None,
    ) -> list[nx.Graph]:
        return decode_generated_nodes(
            self.owner,
            generated_nodes,
            graph_conditioning=graph_conditioning,
            feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
            attempt_idx=attempt_idx,
            use_ilp_decoder=use_ilp_decoder,
            edge_probability_threshold=edge_probability_threshold,
        )

    def decode_conditioning_batch(
        self,
        graph_conditioning: GraphConditioningBatch,
        *,
        sampling_mode: str,
        desired_target: Optional[Union[int, float, Sequence[Any]]] = None,
        guidance_scale: float = 1.0,
        predictor_scale: float = 1.0,
        desired_class: Optional[Union[int, Sequence[Any]]] = None,
        classifier_scale: float = 1.0,
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
        attempt_idx: int = 0,
        use_ilp_decoder: bool = True,
        edge_probability_threshold: Optional[float] = None,
    ) -> list[nx.Graph]:
        owner = self.owner
        if int(getattr(owner, "verbose", 0)) >= 3:
            mode_label = {
                "unguided": "Predicting node matrices",
                "classifier_guided": "Predicting classifier-guided node matrices",
                "regression_guided": "Predicting regression-guided node matrices",
            }.get(sampling_mode, "Predicting node matrices")
            verbose_log(owner, f"{mode_label} for {len(graph_conditioning)} graphs...", level=3)
        if sampling_mode == "unguided":
            generated_nodes = owner._predict_generated_nodes(
                graph_conditioning,
                sampling_mode="unguided",
                desired_target=desired_target,
                guidance_scale=guidance_scale,
            )
        elif sampling_mode == "classifier_guided":
            generated_nodes = owner.conditional_node_generator_model.predict_classifier_guided(
                graph_conditioning,
                desired_class=desired_class,
                classifier_scale=classifier_scale,
            )
            owner._log_generated_batch_info(graph_conditioning, generated_nodes)
        elif sampling_mode == "regression_guided":
            generated_nodes = owner._predict_generated_nodes(
                graph_conditioning,
                sampling_mode="regression_guided",
                desired_target=desired_target,
                predictor_scale=predictor_scale,
            )
        else:
            raise ValueError(f"Unsupported sampling_mode: {sampling_mode!r}")
        return self.decode_generated_nodes(
            generated_nodes,
            graph_conditioning=graph_conditioning,
            feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
            attempt_idx=attempt_idx,
            use_ilp_decoder=use_ilp_decoder,
            edge_probability_threshold=edge_probability_threshold,
        )

    def decode_with_feasibility_slots(
        self,
        graph_conditioning: GraphConditioningBatch,
        *,
        sampling_mode: str,
        desired_target: Optional[Union[int, float, Sequence[Any]]] = None,
        guidance_scale: float = 1.0,
        predictor_scale: float = 1.0,
        desired_class: Optional[Union[int, Sequence[Any]]] = None,
        classifier_scale: float = 1.0,
        apply_feasibility_filtering: Optional[bool] = None,
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
        use_ilp_decoder: bool = True,
        edge_probability_threshold: Optional[float] = None,
    ) -> list[Optional[nx.Graph]]:
        owner = self.owner
        use_filtering = should_apply_feasibility_filtering(owner, apply_feasibility_filtering)
        if owner.feasibility_estimator is None or not use_filtering:
            unfiltered_oracle_candidates = (
                0
                if feasibility_oracle_candidates_per_attempt is None
                else int(feasibility_oracle_candidates_per_attempt)
            )
            timeout_seconds = getattr(owner, "max_decode_seconds_per_sample", None)
            if timeout_seconds is not None and use_ilp_decoder:
                return self._decode_unfiltered_slots_with_timeout(
                    graph_conditioning,
                    sampling_mode=sampling_mode,
                    desired_target=desired_target,
                    guidance_scale=guidance_scale,
                    predictor_scale=predictor_scale,
                    desired_class=desired_class,
                    classifier_scale=classifier_scale,
                    feasibility_oracle_candidates_per_attempt=unfiltered_oracle_candidates,
                    use_ilp_decoder=use_ilp_decoder,
                    edge_probability_threshold=edge_probability_threshold,
                    timeout_seconds=float(timeout_seconds),
                )
            return list(
                self.decode_conditioning_batch(
                    graph_conditioning,
                    sampling_mode=sampling_mode,
                    desired_target=desired_target,
                    guidance_scale=guidance_scale,
                    predictor_scale=predictor_scale,
                    desired_class=desired_class,
                    classifier_scale=classifier_scale,
                    feasibility_oracle_candidates_per_attempt=unfiltered_oracle_candidates,
                    use_ilp_decoder=use_ilp_decoder,
                    edge_probability_threshold=edge_probability_threshold,
                )
            )
        timeout_seconds = getattr(owner, "max_feasibility_seconds_per_sample", None)
        if timeout_seconds is not None:
            accepted_graphs: list[Optional[nx.Graph]] = []
            total_generated = 0
            total_candidate_feasible = 0
            feasible_count = 0
            unfiltered_count = 0
            rejected_count = 0
            for slot_idx in range(len(graph_conditioning)):
                graph, status, generated_now, feasible_now = self._decode_single_conditioning_with_timeout(
                    owner._slice_graph_conditioning(graph_conditioning, [slot_idx]),
                    sampling_mode=sampling_mode,
                    desired_target=desired_target,
                    guidance_scale=guidance_scale,
                    predictor_scale=predictor_scale,
                    desired_class=desired_class,
                    classifier_scale=classifier_scale,
                    feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
                    use_ilp_decoder=use_ilp_decoder,
                    edge_probability_threshold=edge_probability_threshold,
                    timeout_seconds=float(timeout_seconds),
                )
                accepted_graphs.append(graph)
                total_generated += int(generated_now)
                total_candidate_feasible += int(feasible_now)
                if status == "feasible":
                    feasible_count += 1
                elif status == "unfiltered":
                    unfiltered_count += 1
                else:
                    rejected_count += 1
            self._record_decode_summary(
                requested_count=len(graph_conditioning),
                generated_count=total_generated,
                candidate_feasible_count=total_candidate_feasible,
                feasible_count=feasible_count,
                unfiltered_count=unfiltered_count,
                rejected_count=rejected_count,
            )
            return accepted_graphs
        core_result = decode_with_feasibility_slots_core(
            owner,
            graph_conditioning,
            decode_attempt_fn=lambda candidate_conditioning, attempt_idx: self.decode_conditioning_batch(
                candidate_conditioning,
                sampling_mode=sampling_mode,
                desired_target=desired_target,
                guidance_scale=guidance_scale,
                predictor_scale=predictor_scale,
                desired_class=desired_class,
                classifier_scale=classifier_scale,
                feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
                use_ilp_decoder=use_ilp_decoder,
                edge_probability_threshold=edge_probability_threshold,
                attempt_idx=attempt_idx,
            ),
            return_stats=True,
            return_fallbacks=self._should_use_candidate_fallback(),
        )
        if self._should_use_candidate_fallback():
            accepted_graphs, total_generated, total_candidate_feasible, fallback_graphs = core_result
        else:
            accepted_graphs, total_generated, total_candidate_feasible = core_result
            fallback_graphs = [None] * len(graph_conditioning)
        feasible_count = sum(graph is not None for graph in accepted_graphs)
        unfiltered_count = 0
        if self._should_fallback_to_unfiltered():
            missing_slot_indices = [
                slot_idx for slot_idx, graph in enumerate(accepted_graphs) if graph is None
            ]
            if missing_slot_indices:
                if self._should_use_candidate_fallback():
                    fallback_graphs_for_missing = [
                        fallback_graphs[slot_idx]
                        for slot_idx in missing_slot_indices
                    ]
                else:
                    fallback_graphs_for_missing = list(
                        self._decode_unfiltered_backup_conditioning(
                            owner._slice_graph_conditioning(graph_conditioning, missing_slot_indices),
                            sampling_mode=sampling_mode,
                            desired_target=desired_target,
                            guidance_scale=guidance_scale,
                            predictor_scale=predictor_scale,
                            desired_class=desired_class,
                            classifier_scale=classifier_scale,
                            use_ilp_decoder=use_ilp_decoder,
                            edge_probability_threshold=edge_probability_threshold,
                            attempt_idx=max(0, int(getattr(owner, "feasibility_oracle_candidates_per_attempt", 0))),
                        )
                    )
                for slot_idx, graph in zip(missing_slot_indices, fallback_graphs_for_missing):
                    accepted_graphs[slot_idx] = graph
                    if graph is not None:
                        unfiltered_count += 1
        rejected_count = sum(graph is None for graph in accepted_graphs)
        self._record_decode_summary(
            requested_count=len(graph_conditioning),
            generated_count=int(total_generated),
            candidate_feasible_count=int(total_candidate_feasible),
            feasible_count=feasible_count,
            unfiltered_count=unfiltered_count,
            rejected_count=rejected_count,
        )
        return accepted_graphs

    def decode(
        self,
        graph_conditioning: GraphConditioningBatch,
        *,
        sampling_mode: str,
        desired_target: Optional[Union[int, float, Sequence[Any]]] = None,
        guidance_scale: float = 1.0,
        predictor_scale: float = 1.0,
        desired_class: Optional[Union[int, Sequence[Any]]] = None,
        classifier_scale: float = 1.0,
        apply_feasibility_filtering: Optional[bool] = None,
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
        use_ilp_decoder: bool = True,
        edge_probability_threshold: Optional[float] = None,
    ) -> list[nx.Graph]:
        use_filtering = should_apply_feasibility_filtering(self.owner, apply_feasibility_filtering)
        accepted = self.decode_with_feasibility_slots(
            graph_conditioning,
            sampling_mode=sampling_mode,
            desired_target=desired_target,
            guidance_scale=guidance_scale,
            predictor_scale=predictor_scale,
            desired_class=desired_class,
            classifier_scale=classifier_scale,
            apply_feasibility_filtering=apply_feasibility_filtering,
            feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
            use_ilp_decoder=use_ilp_decoder,
            edge_probability_threshold=edge_probability_threshold,
        )
        if self.owner.feasibility_estimator is None or not use_filtering:
            returned_graphs = [graph for graph in accepted if graph is not None]
            missing_count = len(graph_conditioning) - len(returned_graphs)
            if missing_count > 0:
                verbose_log(
                    self.owner,
                    "Unfiltered decode did not return all requested graphs: "
                    f"returned {len(returned_graphs)} of {len(graph_conditioning)}.",
                    level=1,
                )
            return returned_graphs
        return finalize_feasibility_graphs(self.owner, accepted, len(graph_conditioning))
