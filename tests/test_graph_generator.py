import numpy as np
import networkx as nx
import pytest
import pulp

from conditional_node_field_graph_generator.conditional_node_field_graph_generator import (
    DEFAULT_DUMMY_NODE_LABEL,
    ConditionalNodeFieldGraphDecoder,
    ConditionalNodeFieldGraphGenerator,
)
from conditional_node_field_graph_generator.conditional_node_field_generator import ConditionalNodeFieldGenerator, GeneratedNodeBatch


class _GraphVectorizer:
    def __init__(self):
        self.fitted_graph_count = None

    def fit(self, graphs):
        self.fitted_graph_count = len(graphs)
        return self

    def transform(self, graphs):
        rows = []
        for graph in graphs:
            rows.append([graph.number_of_nodes(), graph.number_of_edges()])
        return np.asarray(rows, dtype=float)


class _NodeVectorizer:
    def fit(self, graphs):
        return self

    def transform(self, graphs):
        output = []
        for graph in graphs:
            emb = np.asarray(
                [[float(graph.degree(node)), float(node)] for node in graph.nodes()],
                dtype=float,
            )
            output.append(emb)
        return output


class _Component:
    def __init__(self, verbose=False):
        self.verbose = verbose


class _TrainableNodeModel(_Component):
    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)
        self.setup_calls = []
        self.fit_calls = []

    def setup(self, **kwargs):
        self.setup_calls.append(kwargs)

    def fit(self, **kwargs):
        self.fit_calls.append(kwargs)


def _labeled_graph():
    graph = nx.Graph()
    graph.add_node(0, label="C")
    graph.add_node(1, label="O")
    graph.add_edge(0, 1, label="-")
    return graph


def _unlabeled_edge_graph():
    graph = nx.Graph()
    graph.add_node(0, label="C")
    graph.add_node(1, label="N")
    graph.add_edge(0, 1)
    return graph


def _unlabeled_node_graph():
    graph = nx.Graph()
    graph.add_node(0)
    graph.add_node(1)
    graph.add_edge(0, 1, label="-")
    return graph


def _sampling_graphs():
    graphs = []
    for node_count, edge_count in [(2, 1), (3, 2), (4, 3), (5, 4)]:
        graph = nx.path_graph(node_count)
        if edge_count > graph.number_of_edges():
            next_node = node_count
            while graph.number_of_edges() < edge_count:
                graph.add_edge(0, next_node)
                next_node += 1
        graphs.append(graph)
    return graphs


def _make_fitted_sampling_generator():
    generator = ConditionalNodeFieldGraphGenerator(
        graph_vectorizer=_GraphVectorizer(),
        node_graph_vectorizer=_NodeVectorizer(),
        conditional_node_generator_model=_Component(verbose=False),
        graph_decoder=_Component(verbose=False),
        verbose=False,
    )
    generator.fit(_sampling_graphs(), train_node_generator=False)
    return generator


def test_graph_generator_init_validates_inputs():
    with pytest.raises(ValueError, match="locality_sample_fraction"):
        ConditionalNodeFieldGraphGenerator(locality_sample_fraction=0.0)
    with pytest.raises(ValueError, match="locality_horizon must be >= 1"):
        ConditionalNodeFieldGraphGenerator(locality_horizon=0)
    with pytest.raises(ValueError, match="locality_sampling_strategy"):
        ConditionalNodeFieldGraphGenerator(locality_sampling_strategy="bad")
    with pytest.raises(ValueError, match="locality_target_positive_ratio"):
        ConditionalNodeFieldGraphGenerator(locality_target_positive_ratio=1.1)
    with pytest.raises(ValueError, match="max_feasibility_attempts"):
        ConditionalNodeFieldGraphGenerator(max_feasibility_attempts=0)
    with pytest.raises(ValueError, match="feasibility_candidates_per_attempt"):
        ConditionalNodeFieldGraphGenerator(feasibility_candidates_per_attempt=0)
    with pytest.raises(ValueError, match="feasibility_failure_mode"):
        ConditionalNodeFieldGraphGenerator(feasibility_failure_mode="drop")


def test_fit_requires_graph_vectorizer():
    generator = ConditionalNodeFieldGraphGenerator(
        graph_vectorizer=None,
        node_graph_vectorizer=_NodeVectorizer(),
        conditional_node_generator_model=_Component(verbose=False),
        graph_decoder=_Component(verbose=False),
        verbose=False,
    )

    with pytest.raises(ValueError, match="requires graph_vectorizer"):
        generator.fit([_labeled_graph()], train_node_generator=False)


def test_fit_requires_node_graph_vectorizer():
    generator = ConditionalNodeFieldGraphGenerator(
        graph_vectorizer=_GraphVectorizer(),
        node_graph_vectorizer=None,
        conditional_node_generator_model=_Component(verbose=False),
        graph_decoder=_Component(verbose=False),
        verbose=False,
    )

    with pytest.raises(ValueError, match="requires node_graph_vectorizer"):
        generator.fit([_labeled_graph()], train_node_generator=False)


def test_fit_requires_conditional_node_generator_when_training_enabled():
    generator = ConditionalNodeFieldGraphGenerator(
        graph_vectorizer=_GraphVectorizer(),
        node_graph_vectorizer=_NodeVectorizer(),
        conditional_node_generator_model=None,
        graph_decoder=ConditionalNodeFieldGraphDecoder(verbose=False),
        verbose=False,
    )

    with pytest.raises(ValueError, match="requires conditional_node_generator_model"):
        generator.fit([_labeled_graph()], train_node_generator=True)


def test_fit_requires_graph_decoder_when_training_enabled():
    generator = ConditionalNodeFieldGraphGenerator(
        graph_vectorizer=_GraphVectorizer(),
        node_graph_vectorizer=_NodeVectorizer(),
        conditional_node_generator_model=_Component(verbose=False),
        graph_decoder=None,
        verbose=False,
    )

    with pytest.raises(ValueError, match="requires graph_decoder"):
        generator.fit([_labeled_graph()], train_node_generator=True)


def test_fit_forwards_resume_checkpoint_path_to_node_generator():
    node_model = _TrainableNodeModel(verbose=False)
    generator = ConditionalNodeFieldGraphGenerator(
        graph_vectorizer=_GraphVectorizer(),
        node_graph_vectorizer=_NodeVectorizer(),
        conditional_node_generator_model=node_model,
        graph_decoder=ConditionalNodeFieldGraphDecoder(verbose=False),
        verbose=False,
    )

    generator.fit([_labeled_graph()], train_node_generator=True, ckpt_path="/tmp/resume.ckpt")

    assert len(node_model.setup_calls) == 1
    assert len(node_model.fit_calls) == 1
    assert node_model.fit_calls[0]["ckpt_path"] == "/tmp/resume.ckpt"


def test_toggle_verbose_updates_nested_components():
    node_model = _Component(verbose=False)
    decoder = _Component(verbose=False)
    generator = ConditionalNodeFieldGraphGenerator(
        conditional_node_generator_model=node_model,
        graph_decoder=decoder,
        verbose=False,
    )

    generator.toggle_verbose()

    assert generator.verbose is True
    assert node_model.verbose is True
    assert decoder.verbose is True


def test_build_supervision_plan_modes_depend_on_labels_and_horizon():
    generator = ConditionalNodeFieldGraphGenerator(locality_horizon=2, verbose=False)
    node_label_targets = [np.asarray(["C", "C"], dtype=object), np.asarray(["C"], dtype=object)]
    edge_label_targets = np.asarray(["-"], dtype=object)

    plan = generator._build_supervision_plan(
        graphs=[],
        node_label_targets=node_label_targets,
        edge_label_targets=edge_label_targets,
    )

    assert plan.node_labels.mode == "constant"
    assert plan.node_labels.constant_value == "C"
    assert plan.edge_labels.mode == "constant"
    assert plan.direct_edges.enabled is True
    assert plan.auxiliary_locality.enabled is True
    assert plan.auxiliary_locality.horizon == 2


def test_graphs_to_edge_label_targets_disables_channel_if_any_edge_missing_label():
    generator = ConditionalNodeFieldGraphGenerator(verbose=False)
    graph = _unlabeled_edge_graph()

    edge_targets, edge_pairs = generator.graphs_to_edge_label_targets([graph])

    assert edge_targets is None
    assert edge_pairs is None


def test_graphs_to_edge_label_targets_returns_ordered_pairs_for_labeled_edges():
    generator = ConditionalNodeFieldGraphGenerator(verbose=False)
    graph = _labeled_graph()

    edge_targets, edge_pairs = generator.graphs_to_edge_label_targets([graph])

    assert edge_targets is not None
    assert edge_pairs is not None
    # Undirected edge appears twice because pairs are enumerated over ordered (i, j).
    assert edge_pairs == [(0, 0, 1), (0, 1, 0)]
    assert edge_targets.tolist() == ["-", "-"]


def test_graphs_to_node_label_targets_uses_dummy_label_when_all_nodes_are_unlabelled():
    generator = ConditionalNodeFieldGraphGenerator(verbose=False)

    node_targets = generator.graphs_to_node_label_targets([_unlabeled_node_graph()])

    assert len(node_targets) == 1
    assert node_targets[0].tolist() == [DEFAULT_DUMMY_NODE_LABEL, DEFAULT_DUMMY_NODE_LABEL]


def test_graphs_to_node_label_targets_rejects_mixed_labelled_and_unlabelled_nodes():
    generator = ConditionalNodeFieldGraphGenerator(verbose=False)
    graph = _labeled_graph()
    del graph.nodes[1]["label"]

    with pytest.raises(ValueError, match="either present for every node"):
        generator.graphs_to_node_label_targets([graph])


def test_build_supervision_plan_uses_dummy_label_as_constant_when_nodes_are_unlabelled():
    generator = ConditionalNodeFieldGraphGenerator(verbose=False)
    node_label_targets = generator.graphs_to_node_label_targets([_unlabeled_node_graph()])

    plan = generator._build_supervision_plan(
        graphs=[],
        node_label_targets=node_label_targets,
        edge_label_targets=np.asarray(["-"], dtype=object),
    )

    assert plan.node_labels.mode == "constant"
    assert plan.node_labels.constant_value == DEFAULT_DUMMY_NODE_LABEL


def test_generator_resolves_dummy_constant_node_labels_for_unlabelled_training_setup():
    generator = ConditionalNodeFieldGraphGenerator(verbose=False)
    generator.supervision_plan_ = type(
        "_Plan",
        (),
        {
            "node_labels": type(
                "_Channel",
                (),
                {"mode": "constant", "constant_value": DEFAULT_DUMMY_NODE_LABEL},
            )()
        },
    )()

    labels = generator._resolve_predicted_node_labels(
        GeneratedNodeBatch(
            node_presence_mask=np.asarray([[True, True]], dtype=bool),
        )
    )

    assert labels[0].tolist() == [DEFAULT_DUMMY_NODE_LABEL, DEFAULT_DUMMY_NODE_LABEL]


def test_decoder_decode_node_labels_requires_explicit_labels():
    decoder = ConditionalNodeFieldGraphDecoder(verbose=False)

    with pytest.raises(RuntimeError, match="requires explicit node labels"):
        decoder.decode_node_labels(
            GeneratedNodeBatch(
                node_presence_mask=np.asarray([[True, True]], dtype=bool),
            )
        )


def test_decode_adjacency_matrix_does_not_use_node_embedding_shapes():
    decoder = ConditionalNodeFieldGraphDecoder(verbose=False)

    generated_nodes = GeneratedNodeBatch(
        node_presence_mask=np.asarray([[True, True]], dtype=bool),
        node_degree_predictions=np.asarray([[1, 1]], dtype=float),
        edge_probability_matrices=[np.asarray([[0.0, 0.9], [0.9, 0.0]], dtype=float)],
    )

    adj_mtx_list = decoder.decode_adjacency_matrix(
        generated_nodes,
        predicted_edge_probability_matrices=generated_nodes.edge_probability_matrices,
    )

    assert len(adj_mtx_list) == 1
    assert adj_mtx_list[0].shape == (2, 2)


def test_parallel_decode_matches_serial_decode():
    generated_nodes = GeneratedNodeBatch(
        node_presence_mask=np.asarray([[True, True], [True, True]], dtype=bool),
        node_degree_predictions=np.asarray([[1, 1], [1, 1]], dtype=float),
        edge_probability_matrices=[
            np.asarray([[0.0, 0.9], [0.9, 0.0]], dtype=float),
            np.asarray([[0.0, 0.8], [0.8, 0.0]], dtype=float),
        ],
    )
    predicted_node_labels = [
        np.asarray(["C", "O"], dtype=object),
        np.asarray(["N", "C"], dtype=object),
    ]
    predicted_edge_label_matrices = [
        np.asarray([[None, "-"], ["-", None]], dtype=object),
        np.asarray([[None, "="], ["=", None]], dtype=object),
    ]

    serial_decoder = ConditionalNodeFieldGraphDecoder(verbose=False, n_jobs=1)
    parallel_decoder = ConditionalNodeFieldGraphDecoder(verbose=False, n_jobs=2)

    serial_graphs = serial_decoder.decode(
        generated_nodes,
        predicted_node_labels_list=predicted_node_labels,
        predicted_edge_probability_matrices=generated_nodes.edge_probability_matrices,
        predicted_edge_label_matrices=predicted_edge_label_matrices,
    )
    parallel_graphs = parallel_decoder.decode(
        generated_nodes,
        predicted_node_labels_list=predicted_node_labels,
        predicted_edge_probability_matrices=generated_nodes.edge_probability_matrices,
        predicted_edge_label_matrices=predicted_edge_label_matrices,
    )

    assert len(serial_graphs) == len(parallel_graphs) == 2
    for serial_graph, parallel_graph in zip(serial_graphs, parallel_graphs):
        assert sorted(serial_graph.nodes(data=True)) == sorted(parallel_graph.nodes(data=True))
        assert sorted(serial_graph.edges(data=True)) == sorted(parallel_graph.edges(data=True))


def test_edge_importance_parameters_are_exposed_on_model():
    model = ConditionalNodeFieldGenerator(
        lambda_direct_edge_importance=12.0,
        lambda_auxiliary_edge_importance=7.0,
        verbose=False,
    )

    assert model.lambda_direct_edge_importance == 12.0
    assert model.lambda_auxiliary_edge_importance == 7.0


def test_encode_paths_return_expected_shapes_and_counts():
    g1 = _labeled_graph()
    g2 = _labeled_graph()
    g2.add_node(2, label="N")
    g2.add_edge(1, 2, label="=")
    graphs = [g1, g2]

    generator = ConditionalNodeFieldGraphGenerator(
        graph_vectorizer=_GraphVectorizer(),
        node_graph_vectorizer=_NodeVectorizer(),
        verbose=False,
    )

    node_embeddings = generator.node_encode(graphs)
    conditioning = generator.graph_encode(graphs)
    node_embeddings_2, conditioning_2 = generator.encode(graphs)

    assert len(node_embeddings) == 2
    assert node_embeddings[0].shape == (2, 2)
    assert node_embeddings[1].shape == (3, 2)
    np.testing.assert_array_equal(conditioning.node_counts, np.array([2, 3], dtype=np.int64))
    np.testing.assert_array_equal(conditioning.edge_counts, np.array([1, 2], dtype=np.int64))
    assert conditioning.graph_embeddings.shape == (2, 2)
    assert len(node_embeddings_2) == 2
    np.testing.assert_array_equal(conditioning_2.graph_embeddings, conditioning.graph_embeddings)


def test_build_node_batch_masks_presence_and_degrees():
    graph = _labeled_graph()
    node_embeddings = [np.zeros((2, 3), dtype=float)]
    generator = ConditionalNodeFieldGraphGenerator(verbose=False)

    batch = generator._build_node_batch(
        graphs=[graph],
        node_embeddings_list=node_embeddings,
        node_label_targets=[np.asarray(["C", "O"], dtype=object)],
    )

    assert batch.node_presence_mask.shape == (1, 2)
    assert batch.node_presence_mask.tolist() == [[True, True]]
    assert batch.node_degree_targets.tolist() == [[1, 1]]


def test_fit_stores_training_graph_conditioning_without_training_graph_copy():
    generator = _make_fitted_sampling_generator()

    assert generator.training_graph_conditioning_ is not None
    assert not hasattr(generator, "training_graphs_")
    np.testing.assert_array_equal(
        generator.training_graph_conditioning_.node_counts,
        np.asarray([2, 3, 4, 5], dtype=np.int64),
    )


def test_sample_conditions_direct_mode_returns_cached_rows(monkeypatch):
    generator = _make_fitted_sampling_generator()

    monkeypatch.setattr(np.random, "choice", lambda *args, **kwargs: np.asarray([3, 1], dtype=np.int64))

    conditioning = generator._sample_conditions(2)

    np.testing.assert_array_equal(
        conditioning.graph_embeddings,
        generator.training_graph_conditioning_.graph_embeddings[[3, 1]],
    )
    np.testing.assert_array_equal(conditioning.node_counts, np.asarray([5, 3], dtype=np.int64))
    np.testing.assert_array_equal(conditioning.edge_counts, np.asarray([4, 2], dtype=np.int64))


def test_sample_conditions_interpolation_clamps_negative_cosine_and_avoids_worse_pairs(monkeypatch):
    generator = _make_fitted_sampling_generator()
    generator.training_graph_conditioning_ = type(generator.training_graph_conditioning_)(
        graph_embeddings=np.asarray(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=float,
        ),
        node_counts=np.asarray([2, 8, 4], dtype=np.int64),
        edge_counts=np.asarray([1, 7, 3], dtype=np.int64),
    )

    choice_calls = iter(
        [
            np.asarray([0, 1, 2], dtype=np.int64),
            1,
        ]
    )

    def _fake_choice(*args, **kwargs):
        del args, kwargs
        return next(choice_calls)

    monkeypatch.setattr(np.random, "choice", _fake_choice)
    monkeypatch.setattr(np.random, "uniform", lambda *args, **kwargs: 0.25)

    conditioning = generator._sample_conditions(1, interpolate_between_n_samples=3)

    np.testing.assert_allclose(conditioning.graph_embeddings[0], np.asarray([0.75, 0.25], dtype=float))
    np.testing.assert_array_equal(conditioning.node_counts, np.asarray([2], dtype=np.int64))
    np.testing.assert_array_equal(conditioning.edge_counts, np.asarray([2], dtype=np.int64))


def test_sample_conditions_interpolation_falls_back_to_uniform_best_pair_sampling_when_all_weights_zero(monkeypatch):
    generator = _make_fitted_sampling_generator()
    generator.training_graph_conditioning_ = type(generator.training_graph_conditioning_)(
        graph_embeddings=np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
            ],
            dtype=float,
        ),
        node_counts=np.asarray([2, 4, 6], dtype=np.int64),
        edge_counts=np.asarray([1, 3, 5], dtype=np.int64),
    )

    sampled_probabilities = []
    sampled_choice_args = []
    choice_calls = iter([np.asarray([0, 1, 2], dtype=np.int64), 1])

    def _fake_choice(a, size=None, replace=None, p=None):
        del size, replace
        sampled_probabilities.append(p)
        sampled_choice_args.append(a)
        if isinstance(a, int):
            return next(choice_calls)
        return next(choice_calls)

    monkeypatch.setattr(np.random, "choice", _fake_choice)
    monkeypatch.setattr(np.random, "uniform", lambda *args, **kwargs: 0.5)

    conditioning = generator._sample_conditions(1, interpolate_between_n_samples=3)

    assert sampled_probabilities[-1] is None
    np.testing.assert_array_equal(np.asarray(sampled_choice_args[-1]), np.asarray([0, 1], dtype=np.int64))
    np.testing.assert_array_equal(conditioning.node_counts, np.asarray([4], dtype=np.int64))
    np.testing.assert_array_equal(conditioning.edge_counts, np.asarray([3], dtype=np.int64))


def test_sample_conditions_rejects_invalid_interpolation_subset_size():
    generator = _make_fitted_sampling_generator()

    with pytest.raises(ValueError, match="interpolate_between_n_samples must be >= 2"):
        generator._sample_conditions(1, interpolate_between_n_samples=1)


def test_sample_conditions_single_cached_row_falls_back_to_direct_sampling(monkeypatch):
    generator = _make_fitted_sampling_generator()
    generator.training_graph_conditioning_ = type(generator.training_graph_conditioning_)(
        graph_embeddings=np.asarray([[9.0, 4.0]], dtype=float),
        node_counts=np.asarray([7], dtype=np.int64),
        edge_counts=np.asarray([6], dtype=np.int64),
    )

    monkeypatch.setattr(np.random, "choice", lambda *args, **kwargs: np.asarray([0, 0], dtype=np.int64))

    conditioning = generator._sample_conditions(2, interpolate_between_n_samples=10)

    np.testing.assert_array_equal(conditioning.graph_embeddings, np.asarray([[9.0, 4.0], [9.0, 4.0]], dtype=float))
    np.testing.assert_array_equal(conditioning.node_counts, np.asarray([7, 7], dtype=np.int64))
    np.testing.assert_array_equal(conditioning.edge_counts, np.asarray([6, 6], dtype=np.int64))


def test_sample_passes_direct_conditioning_to_decode(monkeypatch):
    generator = _make_fitted_sampling_generator()
    captured = {}

    monkeypatch.setattr(
        generator,
        "_sample_conditions",
        lambda n_samples, interpolate_between_n_samples=None: type(generator.training_graph_conditioning_)(
            graph_embeddings=np.asarray([[1.0, 2.0]], dtype=float),
            node_counts=np.asarray([3], dtype=np.int64),
            edge_counts=np.asarray([2], dtype=np.int64),
        ),
    )

    def _fake_decode(graph_conditioning, **kwargs):
        del kwargs
        captured["conditioning"] = graph_conditioning
        return ["decoded"]

    monkeypatch.setattr(generator, "_decode_with_feasibility", _fake_decode)

    result = generator.sample(1)

    assert result == ["decoded"]
    np.testing.assert_array_equal(captured["conditioning"].node_counts, np.asarray([3], dtype=np.int64))


def test_sample_passes_interpolation_parameter_to_condition_sampler(monkeypatch):
    generator = _make_fitted_sampling_generator()
    captured = {}

    def _fake_sample_conditions(n_samples, interpolate_between_n_samples=None):
        captured["n_samples"] = n_samples
        captured["interpolate_between_n_samples"] = interpolate_between_n_samples
        return type(generator.training_graph_conditioning_)(
            graph_embeddings=np.asarray([[1.0, 2.0]], dtype=float),
            node_counts=np.asarray([3], dtype=np.int64),
            edge_counts=np.asarray([2], dtype=np.int64),
        )

    monkeypatch.setattr(generator, "_sample_conditions", _fake_sample_conditions)
    monkeypatch.setattr(generator, "_decode_with_feasibility", lambda graph_conditioning, **kwargs: [graph_conditioning])

    result = generator.sample(1, interpolate_between_n_samples=10)

    assert len(result) == 1
    assert captured == {"n_samples": 1, "interpolate_between_n_samples": 10}


def test_optimize_adjacency_matrix_raises_when_solver_status_is_not_optimal(monkeypatch):
    decoder = ConditionalNodeFieldGraphDecoder(verbose=False)

    def _fake_solve(self, solver):
        del solver
        self.status = pulp.LpStatusInfeasible
        return self.status

    monkeypatch.setattr(pulp.LpProblem, "solve", _fake_solve)

    with pytest.raises(RuntimeError, match="did not produce an optimal solution"):
        decoder.optimize_adjacency_matrix(
            prob_matrix=np.array([[0.0, 0.9], [0.9, 0.0]], dtype=float),
            target_degrees=[1, 1],
        )


def test_optimize_adjacency_matrix_raises_when_variable_value_is_missing(monkeypatch):
    decoder = ConditionalNodeFieldGraphDecoder(verbose=False)

    def _fake_solve(self, solver):
        del solver
        self.status = pulp.LpStatusOptimal
        return self.status

    monkeypatch.setattr(pulp.LpProblem, "solve", _fake_solve)

    with pytest.raises(RuntimeError, match="without assigning all decision variables"):
        decoder.optimize_adjacency_matrix(
            prob_matrix=np.array([[0.0, 0.9], [0.9, 0.0]], dtype=float),
            target_degrees=[1, 1],
        )
