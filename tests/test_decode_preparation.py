import numpy as np
import pytest

from conditional_node_field_graph_generator.conditional_node_field_generator import (
    GeneratedNodeBatch,
    GraphConditioningBatch,
)
from conditional_node_field_graph_generator.conditional_node_field_graph_generator import (
    ConditionalNodeFieldGraphGenerator,
    DEFAULT_DUMMY_NODE_LABEL,
)
from conditional_node_field_graph_generator.decode_preparation import (
    build_single_generated_node_batch,
    decode_generated_nodes,
    resolve_predicted_edge_labels,
    resolve_predicted_node_labels,
)


def _owner_with_plan(node_mode=None, node_constant=None, edge_mode=None, edge_constant=None):
    generator = ConditionalNodeFieldGraphGenerator(verbose=False)
    generator.supervision_plan_ = type(
        "_Plan",
        (),
        {
            "node_labels": type(
                "_NodeChannel",
                (),
                {"mode": node_mode, "constant_value": node_constant},
            )(),
            "edge_labels": type(
                "_EdgeChannel",
                (),
                {"mode": edge_mode, "constant_value": edge_constant},
            )(),
        },
    )()
    return generator


def test_resolve_predicted_node_labels_supports_constant_dummy_fallback():
    owner = _owner_with_plan(
        node_mode="constant",
        node_constant=DEFAULT_DUMMY_NODE_LABEL,
        edge_mode="disabled",
        edge_constant=None,
    )

    labels = resolve_predicted_node_labels(
        owner,
        GeneratedNodeBatch(node_presence_mask=np.asarray([[True, True]], dtype=bool)),
    )

    assert labels[0].tolist() == [DEFAULT_DUMMY_NODE_LABEL, DEFAULT_DUMMY_NODE_LABEL]


def test_resolve_predicted_edge_labels_builds_constant_label_matrix():
    owner = _owner_with_plan(
        node_mode="disabled",
        node_constant=None,
        edge_mode="constant",
        edge_constant="-",
    )

    edge_labels, edge_label_matrices = resolve_predicted_edge_labels(
        owner,
        GeneratedNodeBatch(),
        predicted_edge_probability_matrices=[np.asarray([[0.0, 0.8], [0.8, 0.0]], dtype=float)],
    )

    assert edge_labels is None
    assert edge_label_matrices is not None
    assert edge_label_matrices[0][0, 0] is None
    assert edge_label_matrices[0][0, 1] == "-"
    assert edge_label_matrices[0][1, 0] == "-"


def test_build_single_generated_node_batch_slices_all_available_fields():
    batch = GeneratedNodeBatch(
        node_embeddings_list=[
            np.asarray([[1.0, 2.0]], dtype=float),
            np.asarray([[3.0, 4.0]], dtype=float),
        ],
        node_presence_mask=np.asarray([[True], [False]], dtype=bool),
        node_degree_predictions=np.asarray([[1.0], [2.0]], dtype=float),
        node_labels=[np.asarray(["A"], dtype=object), np.asarray(["B"], dtype=object)],
        edge_probability_matrices=[
            np.asarray([[0.0]], dtype=float),
            np.asarray([[1.0]], dtype=float),
        ],
    )

    single = build_single_generated_node_batch(batch, 1)

    assert len(single.node_embeddings_list) == 1
    np.testing.assert_array_equal(single.node_presence_mask, np.asarray([[False]], dtype=bool))
    np.testing.assert_array_equal(single.node_degree_predictions, np.asarray([[2.0]], dtype=float))
    assert single.node_labels[0].tolist() == ["B"]
    np.testing.assert_array_equal(single.edge_probability_matrices[0], np.asarray([[1.0]], dtype=float))


def test_decode_generated_nodes_dispatches_plain_decoder_when_oracle_disabled():
    class Decoder:
        def __init__(self):
            self.calls = []

        def decode(self, generated_nodes, **kwargs):
            self.calls.append((generated_nodes, kwargs))
            return ["decoded"]

    owner = _owner_with_plan(
        node_mode="constant",
        node_constant="C",
        edge_mode="disabled",
        edge_constant=None,
    )
    owner.graph_decoder = Decoder()
    owner._can_use_feasibility_oracle = lambda **kwargs: False
    generated_nodes = GeneratedNodeBatch(
        node_presence_mask=np.asarray([[True, True]], dtype=bool),
        edge_probability_matrices=[np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float)],
    )
    conditioning = GraphConditioningBatch(
        graph_embeddings=np.asarray([[1.0]], dtype=float),
        node_counts=np.asarray([2], dtype=int),
        edge_counts=np.asarray([1], dtype=int),
    )

    result = decode_generated_nodes(owner, generated_nodes, graph_conditioning=conditioning)

    assert result == ["decoded"]
    assert owner.graph_decoder.calls
    _, kwargs = owner.graph_decoder.calls[0]
    assert kwargs["predicted_node_labels_list"][0].tolist() == ["C", "C"]
    assert kwargs["predicted_edge_labels_list"][0].tolist() == []
    np.testing.assert_array_equal(kwargs["desired_node_counts"], np.asarray([2], dtype=int))
    np.testing.assert_array_equal(kwargs["desired_edge_counts"], np.asarray([1], dtype=int))


def test_decode_generated_nodes_requires_edge_probabilities():
    owner = _owner_with_plan(
        node_mode="disabled",
        node_constant=None,
        edge_mode="disabled",
        edge_constant=None,
    )
    owner._can_use_feasibility_oracle = lambda **kwargs: False
    owner.graph_decoder = object()

    with pytest.raises(RuntimeError, match="edge-probability matrices"):
        decode_generated_nodes(owner, GeneratedNodeBatch(node_presence_mask=np.asarray([[True]], dtype=bool)))
