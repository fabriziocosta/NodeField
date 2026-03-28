import types

import networkx as nx
import numpy as np
import pytest
import torch

from conditional_node_field_graph_generator.decode_pipeline import (
    accept_feasible_candidates_by_slot,
    decode_with_feasibility_slots_core,
    score_feasible_rate,
    should_apply_feasibility_filtering,
)
import conditional_node_field_graph_generator.metrics_collection as metrics_collection
from conditional_node_field_graph_generator.metrics_collection import MetricsLogger


def test_decode_pipeline_accepts_feasible_candidates_by_slot(monkeypatch):
    accepted_graphs_by_slot = [None, None]
    decoded_graphs = ["s0_a", "s0_b", "s1_a", "s1_b"]
    feasibility_mask = [True, True, False, True]
    candidate_slot_indices = [0, 0, 1, 1]

    monkeypatch.setattr(np.random, "randint", lambda high: 1 if high == 2 else 0)

    feasible_count, filled_now = accept_feasible_candidates_by_slot(
        decoded_graphs=decoded_graphs,
        feasibility_mask=feasibility_mask,
        candidate_slot_indices=candidate_slot_indices,
        accepted_graphs_by_slot=accepted_graphs_by_slot,
    )

    assert feasible_count == 3
    assert filled_now == 2
    assert accepted_graphs_by_slot == ["s0_b", "s1_b"]


def test_should_apply_feasibility_filtering_uses_override():
    owner = types.SimpleNamespace(use_feasibility_filtering=True)

    assert should_apply_feasibility_filtering(owner, None) is True
    assert should_apply_feasibility_filtering(owner, False) is False


class _FakeFeasibilityEstimator:
    def predict(self, decoded_graphs):
        return [bool(graph.get("feasible", False)) for graph in decoded_graphs]


class _FakePipelineOwner:
    def __init__(self):
        self.max_feasibility_attempts = 3
        self.feasibility_candidates_per_attempt = 2
        self.feasibility_failure_mode = "filter"
        self.use_feasibility_filtering = True
        self.feasibility_estimator = _FakeFeasibilityEstimator()
        self.verbose = 0
        self._attempt = 0

    def _repeat_graph_conditioning(self, conditioning, repeats):
        return [item for item in conditioning for _ in range(repeats)]

    def _slice_graph_conditioning(self, conditioning, slot_indices):
        return [conditioning[idx] for idx in slot_indices]

    def _decode_conditioning_batch(self, conditioning, desired_target=None, guidance_scale=1.0, attempt_idx=0):
        del conditioning, desired_target, guidance_scale, attempt_idx
        self._attempt += 1
        if self._attempt == 1:
            return [
                {"slot": "slot-0", "feasible": True},
                {"slot": "slot-0", "feasible": False},
                {"slot": "slot-1", "feasible": False},
                {"slot": "slot-1", "feasible": False},
            ]
        return [
            {"slot": "slot-1", "feasible": True},
            {"slot": "slot-1", "feasible": False},
        ]

    def _require_fitted_for_generation(self):
        return None

    def _sample_conditions(self, n_samples, interpolate_between_n_samples=None):
        del interpolate_between_n_samples
        return [f"slot-{idx}" for idx in range(n_samples)]


def test_decode_with_feasibility_slots_core_returns_stats():
    owner = _FakePipelineOwner()

    accepted, total_generated, total_feasible = decode_with_feasibility_slots_core(
        owner,
        ["slot-0", "slot-1"],
        decode_attempt_fn=lambda candidate_conditioning, attempt_idx: owner._decode_conditioning_batch(
            candidate_conditioning,
            attempt_idx=attempt_idx,
        ),
        return_stats=True,
    )

    assert accepted == [
        {"slot": "slot-0", "feasible": True},
        {"slot": "slot-1", "feasible": True},
    ]
    assert total_generated == 6
    assert total_feasible == 2


def test_score_feasible_rate_uses_pipeline_helper_and_restores_state():
    owner = _FakePipelineOwner()
    owner.verbose = 5

    result = score_feasible_rate(
        owner,
        n_samples=2,
        max_feasibility_attempts=2,
        feasibility_candidates_per_attempt=2,
    )

    assert result["score"] == pytest.approx(2 / 6)
    assert result["fulfilled_rate"] == pytest.approx(1.0)
    assert result["generated_candidates"] == 6
    assert result["feasible_candidates"] == 2
    assert owner.max_feasibility_attempts == 3
    assert owner.feasibility_candidates_per_attempt == 2
    assert owner.verbose == 5


def test_metrics_logger_validation_summary_uses_logger(monkeypatch):
    logged_messages = []
    monkeypatch.setattr(metrics_collection.logger, "info", lambda msg, *args: logged_messages.append(msg % args if args else msg))

    callback = MetricsLogger()
    trainer = types.SimpleNamespace(
        callback_metrics={
            "train_total": torch.tensor(1.0),
            "train_node_field": torch.tensor(0.5),
            "train_deg_ce": torch.tensor(0.1),
            "val_total": torch.tensor(0.9),
            "val_node_field": torch.tensor(0.4),
            "val_deg_ce": torch.tensor(0.2),
        },
        current_epoch=0,
        max_epochs=5,
        logged_metrics={},
    )
    pl_module = types.SimpleNamespace(
        val_losses=[],
        val_deg_ce=[],
        val_node_field=[],
        verbose=2,
        verbose_epoch_interval=1,
        _fit_start_time=0.0,
        _ema_metrics={},
        lambda_degree_importance=1.0,
        use_locality_supervision=False,
        use_auxiliary_locality_supervision=False,
        train_losses=[1.0],
        train_deg_ce=[0.1],
        train_node_field=[0.5],
    )

    callback.on_validation_epoch_end(trainer, pl_module)

    assert logged_messages
    assert logged_messages[0].startswith("Epoch 1/5")
    assert any("train" in message for message in logged_messages[1:])
    assert any("val" in message for message in logged_messages[1:])
