"""Decode orchestration service for graph generation."""

from __future__ import annotations

from dataclasses import dataclass
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
        timeout_seconds: float,
    ) -> Optional[nx.Graph]:
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
            accepted = decode_with_feasibility_slots_core(
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
                    attempt_idx=attempt_idx,
                ),
            )
        except RuntimeError:
            accepted = [None]
        finally:
            if owner.graph_decoder is not None:
                owner.graph_decoder.active_time_limit_seconds = previous_active_time_limit
            owner._restore_generation_timeout_deadline(previous_deadline)

        if accepted and accepted[0] is not None:
            return accepted[0]
        if int(getattr(owner, "verbose", 0)) >= 1:
            verbose_log(
                owner,
                "Feasibility filtering timed out or exhausted for one sample; falling back to unfiltered decode.",
                level=1,
            )
        fallback_attempt_idx = max(
            0,
            int(getattr(owner, "feasibility_oracle_candidates_per_attempt", 0)),
        )
        fallback_graphs = self.decode_conditioning_batch(
            graph_conditioning,
            sampling_mode=sampling_mode,
            desired_target=desired_target,
            guidance_scale=guidance_scale,
            predictor_scale=predictor_scale,
            desired_class=desired_class,
            classifier_scale=classifier_scale,
            feasibility_oracle_candidates_per_attempt=0,
            attempt_idx=fallback_attempt_idx,
        )
        return None if not fallback_graphs else fallback_graphs[0]

    def decode_generated_nodes(
        self,
        generated_nodes: GeneratedNodeBatch,
        graph_conditioning: Optional[GraphConditioningBatch] = None,
        feasibility_oracle_candidates_per_attempt: Optional[int] = None,
        attempt_idx: int = 0,
    ) -> list[nx.Graph]:
        return decode_generated_nodes(
            self.owner,
            generated_nodes,
            graph_conditioning=graph_conditioning,
            feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
            attempt_idx=attempt_idx,
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
    ) -> list[Optional[nx.Graph]]:
        owner = self.owner
        use_filtering = should_apply_feasibility_filtering(owner, apply_feasibility_filtering)
        if owner.feasibility_estimator is None or not use_filtering:
            return list(
                self.decode_conditioning_batch(
                    graph_conditioning,
                    sampling_mode=sampling_mode,
                    desired_target=desired_target,
                    guidance_scale=guidance_scale,
                    predictor_scale=predictor_scale,
                    desired_class=desired_class,
                    classifier_scale=classifier_scale,
                    feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
                )
            )
        timeout_seconds = getattr(owner, "max_feasibility_seconds_per_sample", None)
        if timeout_seconds is not None:
            return [
                self._decode_single_conditioning_with_timeout(
                    owner._slice_graph_conditioning(graph_conditioning, [slot_idx]),
                    sampling_mode=sampling_mode,
                    desired_target=desired_target,
                    guidance_scale=guidance_scale,
                    predictor_scale=predictor_scale,
                    desired_class=desired_class,
                    classifier_scale=classifier_scale,
                    feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
                    timeout_seconds=float(timeout_seconds),
                )
                for slot_idx in range(len(graph_conditioning))
            ]
        return decode_with_feasibility_slots_core(
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
                attempt_idx=attempt_idx,
            ),
        )

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
    ) -> list[nx.Graph]:
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
        )
        return finalize_feasibility_graphs(self.owner, accepted, len(graph_conditioning))
