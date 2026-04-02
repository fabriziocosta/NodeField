import types

import numpy as np

from conditional_node_field_graph_generator.conditional_node_field_generator import GraphConditioningBatch
from conditional_node_field_graph_generator.decode_service import DecodeService


class _Estimator:
    def predict(self, decoded_graphs):
        return [bool(graph["feasible"]) for graph in decoded_graphs]


class _Owner:
    def __init__(self):
        self.verbose = 0
        self.use_feasibility_filtering = True
        self.feasibility_estimator = _Estimator()
        self.feasibility_candidates_per_attempt = 2
        self.max_feasibility_attempts = 3
        self.feasibility_failure_mode = "return_partial"
        self.feasibility_rejection_mode = "fallback_unfiltered"
        self._calls = 0
        self.conditional_node_generator_model = types.SimpleNamespace(
            predict_classifier_guided=lambda *args, **kwargs: None,
        )

    @staticmethod
    def _repeat_graph_conditioning(graph_conditioning, repeats):
        return GraphConditioningBatch(
            graph_embeddings=np.repeat(np.asarray(graph_conditioning.graph_embeddings), repeats, axis=0),
            node_counts=np.repeat(np.asarray(graph_conditioning.node_counts), repeats, axis=0),
            edge_counts=np.repeat(np.asarray(graph_conditioning.edge_counts), repeats, axis=0),
        )

    @staticmethod
    def _slice_graph_conditioning(graph_conditioning, slot_indices):
        indices = np.asarray(slot_indices, dtype=int)
        return GraphConditioningBatch(
            graph_embeddings=np.asarray(graph_conditioning.graph_embeddings)[indices],
            node_counts=np.asarray(graph_conditioning.node_counts)[indices],
            edge_counts=np.asarray(graph_conditioning.edge_counts)[indices],
        )

    def _predict_generated_nodes(self, *args, **kwargs):
        raise AssertionError("predict path should not run in this unit test")

    def _log_generated_batch_info(self, *args, **kwargs):
        return None

    def _set_generation_timeout_deadline(self, timeout_seconds):
        del timeout_seconds
        return None

    def _restore_generation_timeout_deadline(self, previous_deadline):
        del previous_deadline
        return None


def test_decode_service_retries_until_slots_are_filled():
    owner = _Owner()
    service = DecodeService(owner)
    conditioning = GraphConditioningBatch(
        graph_embeddings=np.asarray([[0.0], [1.0]], dtype=float),
        node_counts=np.asarray([2, 2], dtype=int),
        edge_counts=np.asarray([1, 1], dtype=int),
    )

    def _fake_decode(candidate_conditioning, sampling_mode, **kwargs):
        del candidate_conditioning, sampling_mode, kwargs
        owner._calls += 1
        if owner._calls == 1:
            return [
                {"slot": 0, "feasible": True},
                {"slot": 0, "feasible": False},
                {"slot": 1, "feasible": False},
                {"slot": 1, "feasible": False},
            ]
        return [
            {"slot": 1, "feasible": True},
            {"slot": 1, "feasible": False},
        ]

    owner.decode_service_ = service
    service.decode_conditioning_batch = _fake_decode

    decoded = service.decode(conditioning, sampling_mode="unguided")

    assert decoded == [
        {"slot": 0, "feasible": True},
        {"slot": 1, "feasible": True},
    ]


def test_decode_service_bypasses_filtering_when_disabled():
    owner = _Owner()
    service = DecodeService(owner)
    conditioning = GraphConditioningBatch(
        graph_embeddings=np.asarray([[0.0]], dtype=float),
        node_counts=np.asarray([2], dtype=int),
        edge_counts=np.asarray([1], dtype=int),
    )

    service.decode_conditioning_batch = lambda *args, **kwargs: [{"slot": 0, "feasible": False}]

    decoded = service.decode_with_feasibility_slots(
        conditioning,
        sampling_mode="unguided",
        apply_feasibility_filtering=False,
    )

    assert decoded == [{"slot": 0, "feasible": False}]


def test_decode_service_timeout_mode_records_final_summary_with_fallback(monkeypatch):
    owner = _Owner()
    owner.max_feasibility_seconds_per_sample = 5.0
    owner.graph_decoder = None
    service = DecodeService(owner)
    conditioning = GraphConditioningBatch(
        graph_embeddings=np.asarray([[0.0], [1.0]], dtype=float),
        node_counts=np.asarray([2, 2], dtype=int),
        edge_counts=np.asarray([1, 1], dtype=int),
    )

    def _fake_single(*args, **kwargs):
        del args, kwargs
        if not hasattr(owner, "_single_calls"):
            owner._single_calls = 0
        owner._single_calls += 1
        return ("fallback-graph" if owner._single_calls == 1 else {"slot": 1, "feasible": True},
                "unfiltered" if owner._single_calls == 1 else "feasible")

    monkeypatch.setattr(service, "_decode_single_conditioning_with_timeout", _fake_single)

    decoded = service.decode_with_feasibility_slots(conditioning, sampling_mode="unguided")

    assert decoded == ["fallback-graph", {"slot": 1, "feasible": True}]
    assert owner.last_decode_summary_ == {
        "requested": 2,
        "returned": 2,
        "generated": 0,
        "candidate_feasible": 0,
        "candidate_feasible_fraction": 0.0,
        "feasible": 1,
        "feasible_fraction": 0.5,
        "unfiltered": 1,
        "unfiltered_fraction": 0.5,
        "rejected": 0,
        "rejected_fraction": 0.0,
    }


def test_decode_service_strict_mode_skips_unfiltered_fallback(monkeypatch):
    owner = _Owner()
    owner.max_feasibility_seconds_per_sample = 5.0
    owner.feasibility_rejection_mode = "strict"
    owner.graph_decoder = None
    service = DecodeService(owner)
    conditioning = GraphConditioningBatch(
        graph_embeddings=np.asarray([[0.0]], dtype=float),
        node_counts=np.asarray([2], dtype=int),
        edge_counts=np.asarray([1], dtype=int),
    )

    monkeypatch.setattr(
        "conditional_node_field_graph_generator.decode_service.decode_with_feasibility_slots_core",
        lambda *args, **kwargs: ([None], 0, 0) if kwargs.get("return_stats") else [None],
    )
    fallback_called = {"value": False}

    def _unexpected_fallback(*args, **kwargs):
        del args, kwargs
        fallback_called["value"] = True
        return [{"slot": 0, "feasible": False}]

    service.decode_conditioning_batch = _unexpected_fallback

    decoded = service.decode_with_feasibility_slots(conditioning, sampling_mode="unguided")

    assert decoded == [None]
    assert fallback_called["value"] is False
    assert owner.last_decode_summary_["rejected"] == 1
    assert owner.last_decode_summary_["generated"] == 0
    assert owner.last_decode_summary_["candidate_feasible"] == 0
    assert owner.last_decode_summary_["candidate_feasible_fraction"] == 0.0
