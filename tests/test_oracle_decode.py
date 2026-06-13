import numpy as np
import pytest
from pathlib import Path

import conditional_node_field_graph_generator.conditional_node_field_graph_generator as cngg_module
import conditional_node_field_graph_generator.oracle_decode as oracle_decode_module
from conditional_node_field_graph_generator.conditional_node_field_generator import (
    ConditionalNodeFieldGraphGenerator,
)
from conditional_node_field_graph_generator.oracle_decode import (
    sample_oracle_cuts_for_iteration,
    solve_oracle_relaxed_adjacency,
)


def test_sample_oracle_cuts_for_iteration_relaxes_to_zero_on_final_attempt(monkeypatch):
    owner = type("Owner", (), {"max_oracle_iterations": 4})()
    accumulated = [
        frozenset({(0, 1)}),
        frozenset({(1, 2)}),
        frozenset({(2, 3)}),
    ]

    monkeypatch.setattr(
        oracle_decode_module.random,
        "sample",
        lambda population, k: list(population)[:k],
    )

    assert sample_oracle_cuts_for_iteration(owner, accumulated, 0) == accumulated
    assert len(sample_oracle_cuts_for_iteration(owner, accumulated, 1)) == 2
    assert len(sample_oracle_cuts_for_iteration(owner, accumulated, 2)) == 1
    assert sample_oracle_cuts_for_iteration(owner, accumulated, 3) == []


def test_solve_oracle_relaxed_adjacency_retries_with_fewer_cuts():
    active_cut_counts = []

    class Decoder:
        def optimize_adjacency_matrix(
            self,
            prob_matrix,
            target_degrees,
            target_edge_count=None,
            forbidden_edge_sets=None,
        ):
            del prob_matrix, target_degrees, target_edge_count
            active_cut_counts.append(len(list(forbidden_edge_sets or [])))
            if forbidden_edge_sets:
                raise RuntimeError("forced failure while cuts remain")
            return np.asarray([[0, 1], [1, 0]], dtype=int)

    owner = type(
        "Owner",
        (),
        {
            "graph_decoder": Decoder(),
            "oracle_edge_memory_penalty": 0.0,
            "max_oracle_iterations": 3,
            "verbose": False,
        },
    )()

    adj = solve_oracle_relaxed_adjacency(
        owner,
        masked_prob_matrix=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float),
        target_degrees=[1, 1],
        accumulated_cuts=[frozenset({(0, 1)})],
        start_iteration_idx=0,
    )

    assert np.array_equal(adj, np.asarray([[0, 1], [1, 0]], dtype=int))
    assert active_cut_counts[-2:] == [1, 0]


def test_graph_generator_oracle_adapter_dispatches_to_extracted_helper(monkeypatch):
    calls = []

    def fake_decode(owner, generated_nodes, graph_conditioning=None):
        calls.append((owner, generated_nodes, graph_conditioning))
        return ["decoded"]

    monkeypatch.setattr(cngg_module, "_decode_generated_nodes_with_oracle", fake_decode)

    generator = ConditionalNodeFieldGraphGenerator(verbose=False)

    result = generator._decode_generated_nodes_with_oracle("generated", graph_conditioning="conditioning")

    assert result == ["decoded"]
    assert calls == [(generator, "generated", "conditioning")]


def test_oracle_module_does_not_import_decoder_facade():
    source = Path(oracle_decode_module.__file__).read_text(encoding="utf-8")
    assert "conditional_node_field_graph_decoder import" not in source
