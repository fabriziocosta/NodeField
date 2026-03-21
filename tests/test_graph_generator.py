import io
import types

import numpy as np
import networkx as nx
import pytest
import pulp
import torch
from sklearn.preprocessing import MinMaxScaler

import conditional_node_field_graph_generator.conditional_node_field_graph_generator as cngg_module
from conditional_node_field_graph_generator.conditional_node_field_graph_generator import (
    DEFAULT_DUMMY_NODE_LABEL,
    ConditionalNodeFieldGraphDecoder,
    ConditionalNodeFieldGraphGenerator,
    GeneratedGuidanceBatch,
)
from conditional_node_field_graph_generator.conditional_node_field_generator import (
    ConditionalNodeFieldGenerator,
    GeneratedNodeBatch,
    GraphConditioningBatch,
)


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


class _PredictiveStubModel(torch.nn.Module):
    def __init__(self, generated, cached_outputs):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.generated = torch.as_tensor(generated, dtype=torch.float32)
        self.cached_outputs = cached_outputs
        self.generate_calls = []

    def generate(self, cond_tensor, **kwargs):
        self.generate_calls.append(
            {
                "cond_shape": tuple(cond_tensor.shape),
                **kwargs,
            }
        )
        for name, value in self.cached_outputs.items():
            setattr(self, name, None if value is None else value.detach().cpu().clone())
        return self.generated.to(cond_tensor.device)


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


def _sample_graph_conditioning(batch_size=2):
    return GraphConditioningBatch(
        graph_embeddings=np.asarray([[1.0], [2.0]], dtype=float)[:batch_size],
        node_counts=np.asarray([2, 3], dtype=np.int64)[:batch_size],
        edge_counts=np.asarray([1, 2], dtype=np.int64)[:batch_size],
    )


def _make_stubbed_node_field_generator(model, *, guidance_enabled=False):
    generator = ConditionalNodeFieldGenerator(verbose=False)
    generator.model = model
    generator.x_scaler = MinMaxScaler().fit(np.asarray([[0.0, 0.0]], dtype=float))
    generator.y_scaler = MinMaxScaler().fit(np.asarray([[0.0, 0.0, 0.0]], dtype=float))
    generator.base_condition_scaler_ = MinMaxScaler().fit(np.asarray([[0.0, 0.0, 0.0]], dtype=float))
    generator.is_setup_ = True
    generator.guidance_enabled_ = guidance_enabled
    generator.target_condition_dim_ = 0
    generator.target_condition_start_ = 3
    generator.node_label_classes_ = np.asarray(["C", "N", "O"], dtype=object)
    generator.edge_label_classes_ = np.asarray(["-", "="], dtype=object)
    generator._inverse_transform_input = lambda values: values
    return generator


def _rich_cached_outputs():
    node_label_logits = torch.tensor(
        [
            [[3.0, 1.0, -1.0], [0.1, 0.2, 4.0], [0.2, 5.0, 0.1]],
            [[1.0, 2.0, 0.5], [4.0, 0.1, 0.0], [0.3, 0.2, 3.0]],
        ],
        dtype=torch.float32,
    )
    edge_label_logits = torch.tensor(
        [
            [
                [[0.0, 0.0], [4.0, 1.0], [1.0, 3.0]],
                [[2.0, 1.0], [0.0, 0.0], [3.0, 2.0]],
                [[1.0, 3.0], [0.5, 4.0], [0.0, 0.0]],
            ],
            [
                [[0.0, 0.0], [1.0, 2.5], [2.0, 0.5]],
                [[3.0, 1.0], [0.0, 0.0], [0.5, 2.0]],
                [[4.0, 0.1], [1.5, 2.5], [0.0, 0.0]],
            ],
        ],
        dtype=torch.float32,
    )
    edge_probs = torch.tensor(
        [
            [[0.0, 0.9, 0.2], [0.9, 0.0, 0.7], [0.2, 0.7, 0.0]],
            [[0.0, 0.6, 0.4], [0.6, 0.0, 0.8], [0.4, 0.8, 0.0]],
        ],
        dtype=torch.float32,
    )
    return {
        "_last_node_presence_mask": torch.tensor(
            [[True, True, False], [False, False, False]],
            dtype=torch.bool,
        ),
        "_last_deg_classes": torch.tensor([[1, 2, 0], [0, 1, 2]], dtype=torch.int64),
        "_last_node_label_classes": torch.argmax(node_label_logits, dim=-1),
        "_last_node_label_logits": node_label_logits,
        "_last_node_label_probabilities": torch.softmax(node_label_logits, dim=-1),
        "_last_edge_probability_matrices": edge_probs,
        "_last_edge_existence_probabilities": edge_probs,
        "_last_edge_label_matrices": torch.argmax(edge_label_logits, dim=-1),
        "_last_edge_label_logits": edge_label_logits,
        "_last_edge_label_probabilities": torch.softmax(edge_label_logits, dim=-1),
    }


def _assert_rich_generated_batch(batch):
    assert batch.node_label_logits is not None
    assert batch.node_label_probabilities is not None
    assert batch.edge_existence_probabilities is not None
    assert batch.edge_label_logits is not None
    assert batch.edge_label_probabilities is not None
    assert batch.node_label_logits[0].shape == (3, 3)
    assert batch.node_label_probabilities[0].shape == (3, 3)
    assert batch.edge_existence_probabilities[0].shape == (3, 3)
    assert batch.edge_label_logits[0].shape == (3, 3, 2)
    assert batch.edge_label_probabilities[0].shape == (3, 3, 2)
    assert batch.node_labels[0].tolist() == ["C", "O", "N"]
    assert batch.edge_label_matrices[0].shape == (3, 3)


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
    with pytest.raises(ValueError, match="feasibility_oracle_candidates_per_attempt"):
        ConditionalNodeFieldGraphGenerator(feasibility_oracle_candidates_per_attempt=-1)
    with pytest.raises(ValueError, match="feasibility_candidates_per_attempt"):
        ConditionalNodeFieldGraphGenerator(feasibility_candidates_per_attempt=0)
    with pytest.raises(ValueError, match="max_oracle_iterations"):
        ConditionalNodeFieldGraphGenerator(max_oracle_iterations=0)
    with pytest.raises(ValueError, match="feasibility_failure_mode"):
        ConditionalNodeFieldGraphGenerator(feasibility_failure_mode="drop")


def test_graph_generator_logs_model_name_when_verbose(caplog):
    with caplog.at_level("INFO", logger="conditional_node_field_graph_generator"):
        ConditionalNodeFieldGraphGenerator(
            verbose=1,
            model_name="demo-artificial-n100-size8",
            model_dir="/tmp/models",
        )

    assert "Configured graph generator model_name=demo-artificial-n100-size8 model_dir=/tmp/models" in caplog.text


def test_fit_logs_model_name_when_configured(caplog):
    generator = ConditionalNodeFieldGraphGenerator(
        graph_vectorizer=_GraphVectorizer(),
        node_graph_vectorizer=_NodeVectorizer(),
        conditional_node_generator_model=_Component(verbose=False),
        graph_decoder=_Component(verbose=False),
        verbose=1,
        model_name="demo-artificial-n100-size8",
        model_dir="/tmp/models",
    )

    caplog.clear()
    with caplog.at_level("INFO", logger="conditional_node_field_graph_generator"):
        generator.fit([_labeled_graph()], train_node_generator=False)

    assert "Fit target model_name=demo-artificial-n100-size8 model_dir=/tmp/models" in caplog.text


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


def test_accept_feasible_candidates_counts_all_feasible_and_fills_one_per_slot(monkeypatch):
    accepted_graphs_by_slot = [None, None]
    decoded_graphs = ["s0_a", "s0_b", "s1_a", "s1_b"]
    feasibility_mask = [True, True, False, True]
    candidate_slot_indices = [0, 0, 1, 1]

    monkeypatch.setattr(np.random, "randint", lambda high: 1 if high == 2 else 0)

    feasible_count, filled_now = ConditionalNodeFieldGraphGenerator._accept_feasible_candidates_by_slot(
        decoded_graphs=decoded_graphs,
        feasibility_mask=feasibility_mask,
        candidate_slot_indices=candidate_slot_indices,
        accepted_graphs_by_slot=accepted_graphs_by_slot,
    )

    assert feasible_count == 3
    assert filled_now == 2
    assert accepted_graphs_by_slot == ["s0_b", "s1_b"]


class _FakeFeasibilityEstimator:
    def predict(self, decoded_graphs):
        return [bool(graph.get("feasible", False)) for graph in decoded_graphs]

    def number_of_violations(self, decoded_graphs):
        return [int(graph.get("violations", 0)) for graph in decoded_graphs]


class _FakeScoreGenerator:
    def __init__(self):
        self.max_feasibility_attempts = 3
        self.feasibility_candidates_per_attempt = 2
        self.verbose = 5
        self.use_feasibility_filtering = True
        self.feasibility_estimator = _FakeFeasibilityEstimator()
        self._attempt = 0

    def _require_fitted_for_generation(self):
        return None

    def _sample_conditions(self, n_samples, interpolate_between_n_samples=None):
        del interpolate_between_n_samples
        return [f"slot-{idx}" for idx in range(n_samples)]

    def _repeat_graph_conditioning(self, conditioning, repeats):
        return [item for item in conditioning for _ in range(repeats)]

    def _slice_graph_conditioning(self, conditioning, slot_indices):
        return [conditioning[idx] for idx in slot_indices]

    def _decode_conditioning_batch(self, conditioning, desired_target=None, guidance_scale=1.0):
        del conditioning, desired_target, guidance_scale
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

    _accept_feasible_candidates_by_slot = staticmethod(
        ConditionalNodeFieldGraphGenerator._accept_feasible_candidates_by_slot
    )


def test_score_feasible_rate_counts_candidate_feasibility():
    generator = _FakeScoreGenerator()

    result = ConditionalNodeFieldGraphGenerator.score_feasible_rate(
        generator,
        n_samples=2,
        max_feasibility_attempts=2,
        feasibility_candidates_per_attempt=2,
    )

    assert result["score"] == pytest.approx(2 / 6)
    assert result["feasible_rate"] == pytest.approx(2 / 6)
    assert result["fulfilled_rate"] == pytest.approx(1.0)
    assert result["accepted_slots"] == 2
    assert result["generated_candidates"] == 6
    assert result["feasible_candidates"] == 2
    assert generator.max_feasibility_attempts == 3
    assert generator.feasibility_candidates_per_attempt == 2
    assert generator.verbose == 5


def test_compute_guidance_targets_is_bounded_and_monotone():
    scores = ConditionalNodeFieldGraphGenerator._compute_guidance_targets([0, 1, 3, 15])

    assert scores[0] == pytest.approx(1.0)
    assert np.all(scores > 0.0)
    assert np.all(scores <= 1.0)
    assert np.all(np.diff(scores) < 0.0)


def test_build_guidance_violation_buckets_keeps_feasible_bucket_and_quantiles():
    buckets = ConditionalNodeFieldGraphGenerator.build_guidance_violation_buckets(
        [0, 0, 1, 1, 3, 7, 12, 20],
        positive_bucket_count=3,
    )

    assert buckets[0]["label"] == "feasible"
    assert np.array_equal(buckets[0]["indices"], np.asarray([0, 1], dtype=np.int64))
    assert sum(int(len(bucket["indices"])) for bucket in buckets) == 8
    assert len(buckets) >= 3


def test_build_guidance_violation_buckets_collapses_duplicate_quantiles():
    buckets = ConditionalNodeFieldGraphGenerator.build_guidance_violation_buckets(
        [0, 2, 2, 2, 9, 9, 9],
        positive_bucket_count=8,
    )

    assert buckets[0]["label"] == "feasible"
    positive_counts = [int(len(bucket["indices"])) for bucket in buckets[1:]]
    assert positive_counts == [3, 3]


class _CollectConditionalStub:
    def __init__(self):
        self.regression_calls = []

    def predict(self, graph_conditioning, desired_target=None, guidance_scale=1.0):
        del desired_target, guidance_scale
        n = len(graph_conditioning)
        embeddings = [
            np.asarray([[float(i + 1)], [float(i + 2)]], dtype=float)
            for i in range(n)
        ]
        return GeneratedNodeBatch(
            node_embeddings_list=embeddings,
            node_presence_mask=np.ones((n, 2), dtype=bool),
            edge_probability_matrices=[np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float) for _ in range(n)],
        )

    def predict_regression_guided(self, graph_conditioning, desired_target, predictor_scale=1.0):
        self.regression_calls.append(
            {"desired_target": desired_target, "predictor_scale": predictor_scale}
        )
        return self.predict(graph_conditioning)


class _CollectDecoderStub:
    def decode(self, generated_nodes, **kwargs):
        del kwargs
        graphs = []
        for idx, _ in enumerate(generated_nodes.node_embeddings_list):
            graph = nx.Graph()
            graph.graph["violations"] = idx
            graph.graph["feasible"] = (idx == 0)
            graph.add_node(0, label="C")
            if idx > 0:
                graph.add_node(1, label="O")
                graph.add_edge(0, 1, label="-")
            graphs.append(graph)
        return graphs


class _OracleOnceEstimator:
    def __init__(self, edge_sets_per_call):
        self.edge_sets_per_call = list(edge_sets_per_call)
        self.calls = []

    def violating_edge_sets(self, graphs):
        graph = graphs[0]
        self.calls.append(frozenset(tuple(sorted(edge)) for edge in graph.edges()))
        if self.edge_sets_per_call:
            return [self.edge_sets_per_call.pop(0)]
        return [[]]


class _LabelAwareOracleEstimator:
    def __init__(self, edge_sets_per_call=None, node_sets_per_call=None):
        self.edge_sets_per_call = list(edge_sets_per_call or [])
        self.node_sets_per_call = list(node_sets_per_call or [])
        self.edge_calls = 0
        self.node_calls = 0

    def violating_edge_sets(self, graphs):
        del graphs
        self.edge_calls += 1
        if self.edge_sets_per_call:
            return [self.edge_sets_per_call.pop(0)]
        return [[]]

    def violating_node_labels_sets(self, graphs):
        del graphs
        self.node_calls += 1
        if self.node_sets_per_call:
            return [self.node_sets_per_call.pop(0)]
        return [[]]


def _oracle_generated_batch():
    return GeneratedNodeBatch(
        node_presence_mask=np.asarray([[True, True, True, True]], dtype=bool),
        node_degree_predictions=np.asarray([[2.0, 2.0, 2.0, 2.0]], dtype=float),
        node_labels=[np.asarray(["C", "C", "C", "C"], dtype=object)],
        edge_probability_matrices=[
            np.asarray(
                [
                    [0.0, 0.95, 0.10, 0.95],
                    [0.95, 0.0, 0.95, 0.10],
                    [0.10, 0.95, 0.0, 0.95],
                    [0.95, 0.10, 0.95, 0.0],
                ],
                dtype=float,
            )
        ],
        edge_label_matrices=[
            np.asarray(
                [
                    [None, "-", "-", "-"],
                    ["-", None, "-", "-"],
                    ["-", "-", None, "-"],
                    ["-", "-", "-", None],
                ],
                dtype=object,
            )
        ],
    )


def _oracle_label_generated_batch(
    *,
    node_labels,
    edge_label_matrix,
    node_label_probabilities=None,
    edge_label_probabilities=None,
):
    return GeneratedNodeBatch(
        node_presence_mask=np.asarray([[True, True]], dtype=bool),
        node_degree_predictions=np.asarray([[1.0, 1.0]], dtype=float),
        node_labels=[np.asarray(node_labels, dtype=object)],
        edge_probability_matrices=[
            np.asarray(
                [
                    [0.0, 0.95],
                    [0.95, 0.0],
                ],
                dtype=float,
            )
        ],
        edge_existence_probabilities=[
            np.asarray(
                [
                    [0.0, 0.95],
                    [0.95, 0.0],
                ],
                dtype=float,
            )
        ],
        edge_label_matrices=[np.asarray(edge_label_matrix, dtype=object)],
        node_label_probabilities=None
        if node_label_probabilities is None
        else [np.asarray(node_label_probabilities, dtype=float)],
        edge_label_probabilities=None
        if edge_label_probabilities is None
        else [np.asarray(edge_label_probabilities, dtype=float)],
    )


def test_collect_generated_guidance_examples_uses_generated_embeddings():
    generator = ConditionalNodeFieldGraphGenerator(verbose=False)
    generator.is_fitted_ = True
    generator.feasibility_estimator = _FakeFeasibilityEstimator()
    generator.conditional_node_generator_model = _CollectConditionalStub()
    generator.graph_decoder = _CollectDecoderStub()
    conditioning = GraphConditioningBatch(
        graph_embeddings=np.asarray([[1.0], [2.0], [3.0]], dtype=float),
        node_counts=np.asarray([2, 2, 2], dtype=np.int64),
        edge_counts=np.asarray([1, 1, 1], dtype=np.int64),
    )
    generator.training_graph_conditioning_ = conditioning

    batch = generator.collect_generated_guidance_examples(
        n_samples=3,
        interpolate_between_n_samples=None,
    )

    assert len(batch.node_embeddings_list) == 3
    assert batch.violation_counts.tolist() == [0, 1, 2]
    assert batch.feasible_mask.tolist() == [True, False, False]
    assert batch.guidance_targets[0] == pytest.approx(1.0)
    assert batch.guidance_targets[1] < 1.0


class _BootstrapGenerator:
    build_guidance_violation_buckets = staticmethod(
        ConditionalNodeFieldGraphGenerator.build_guidance_violation_buckets
    )
    _compute_guidance_targets = staticmethod(
        ConditionalNodeFieldGraphGenerator._compute_guidance_targets
    )
    _sample_bucket_indices = classmethod(ConditionalNodeFieldGraphGenerator._sample_bucket_indices.__func__)
    _slice_generated_guidance_batch = staticmethod(
        ConditionalNodeFieldGraphGenerator._slice_generated_guidance_batch
    )
    _summarize_violation_buckets = classmethod(
        ConditionalNodeFieldGraphGenerator._summarize_violation_buckets.__func__
    )
    _concat_generated_guidance_batches = classmethod(
        ConditionalNodeFieldGraphGenerator._concat_generated_guidance_batches.__func__
    )
    _empty_generated_guidance_batch = staticmethod(
        ConditionalNodeFieldGraphGenerator._empty_generated_guidance_batch
    )
    train_guidance_predictor_from_embeddings = ConditionalNodeFieldGraphGenerator.train_guidance_predictor_from_embeddings
    bootstrap_guidance_regressor_from_generated = ConditionalNodeFieldGraphGenerator.bootstrap_guidance_regressor_from_generated

    def __init__(self):
        self.is_fitted_ = True
        self.conditional_node_generator_model = types.SimpleNamespace(
            train_guidance_predictor_from_embeddings=self._train
        )
        self.train_calls = []
        self.collect_calls = []

    def _require_fitted_for_generation(self):
        return None

    def _train(self, **kwargs):
        self.train_calls.append(kwargs)

    def collect_generated_guidance_examples(
        self,
        n_samples,
        interpolate_between_n_samples=None,
        sampling_mode="unguided",
        desired_target=None,
        guidance_scale=1.0,
        predictor_scale=1.0,
    ):
        del interpolate_between_n_samples, guidance_scale
        self.collect_calls.append(
            {
                "n_samples": n_samples,
                "sampling_mode": sampling_mode,
                "desired_target": desired_target,
                "predictor_scale": predictor_scale,
            }
        )
        if sampling_mode == "unguided":
            violations = np.asarray([0, 2, 4, 8][:n_samples], dtype=np.int64)
        else:
            violations = np.asarray([0, 1, 1, 3][:n_samples], dtype=np.int64)
        node_embeddings_list = [np.asarray([[float(i)]], dtype=float) for i in range(len(violations))]
        conditioning = GraphConditioningBatch(
            graph_embeddings=np.arange(len(violations), dtype=float).reshape(-1, 1),
            node_counts=np.ones((len(violations),), dtype=np.int64),
            edge_counts=np.zeros((len(violations),), dtype=np.int64),
        )
        return GeneratedGuidanceBatch(
            node_embeddings_list=node_embeddings_list,
            graph_conditioning=conditioning,
            decoded_graphs=[{"violations": int(v)} for v in violations],
            violation_counts=violations,
            guidance_targets=ConditionalNodeFieldGraphGenerator._compute_guidance_targets(violations),
            feasible_mask=(violations == 0),
            sampling_mode=sampling_mode,
        )


def test_bootstrap_guidance_regressor_uses_unguided_then_mixed_cycles():
    generator = _BootstrapGenerator()

    result = generator.bootstrap_guidance_regressor_from_generated(
        num_cycles=3,
        examples_per_cycle=4,
        replay_train_size=4,
        guidance_maximum_epochs=2,
        random_state=7,
    )

    assert [row["cycle"] for row in result["history"]] == [1, 2, 3]
    assert generator.collect_calls[0]["sampling_mode"] == "unguided"
    assert generator.collect_calls[0]["desired_target"] is None
    assert generator.collect_calls[1]["sampling_mode"] == "unguided"
    assert generator.collect_calls[1]["desired_target"] is None
    assert generator.collect_calls[2]["sampling_mode"] == "regression_guided"
    assert generator.collect_calls[2]["desired_target"] == 1.0
    assert result["history"][0]["guided_count"] == 0
    assert result["history"][1]["guided_count"] == 2
    assert result["history"][1]["unguided_count"] == 2
    assert len(generator.train_calls) == 3


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


def test_generated_node_batch_len_supports_rich_distribution_fields():
    batch = GeneratedNodeBatch(
        node_label_logits=[
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=float),
            np.asarray([[0.2, 0.8]], dtype=float),
        ]
    )

    assert len(batch) == 2


def test_predict_returns_rich_distribution_tensors_and_legacy_outputs():
    generated = np.asarray(
        [
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
            [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]],
        ],
        dtype=float,
    )
    model = _PredictiveStubModel(generated=generated, cached_outputs=_rich_cached_outputs())
    generator = _make_stubbed_node_field_generator(model)

    batch = generator.predict(_sample_graph_conditioning())

    _assert_rich_generated_batch(batch)
    assert batch.edge_probability_matrices is not None
    assert batch.node_embeddings_list[0].shape == (2, 2)
    assert batch.node_embeddings_list[1].shape == (1, 2)
    np.testing.assert_array_equal(batch.node_presence_mask[0], np.asarray([True, True, False], dtype=bool))
    np.testing.assert_allclose(batch.edge_probability_matrices[0], batch.edge_existence_probabilities[0])
    assert generator.last_predicted_node_label_logits_ is batch.node_label_logits
    assert generator.last_predicted_node_label_probabilities_ is batch.node_label_probabilities
    assert generator.last_predicted_edge_existence_probabilities_ is batch.edge_existence_probabilities
    assert generator.last_predicted_edge_label_logits_ is batch.edge_label_logits
    assert generator.last_predicted_edge_label_probabilities_ is batch.edge_label_probabilities


def test_predict_leaves_rich_distribution_fields_none_when_heads_are_disabled():
    generated = np.asarray([[[1.0, 2.0], [3.0, 4.0]]], dtype=float)
    model = _PredictiveStubModel(
        generated=generated,
        cached_outputs={
            "_last_node_presence_mask": torch.tensor([[True, False]], dtype=torch.bool),
            "_last_deg_classes": torch.tensor([[1, 0]], dtype=torch.int64),
            "_last_node_label_classes": None,
            "_last_node_label_logits": None,
            "_last_node_label_probabilities": None,
            "_last_edge_probability_matrices": None,
            "_last_edge_existence_probabilities": None,
            "_last_edge_label_matrices": None,
            "_last_edge_label_logits": None,
            "_last_edge_label_probabilities": None,
        },
    )
    generator = _make_stubbed_node_field_generator(model)

    batch = generator.predict(_sample_graph_conditioning(batch_size=1))

    assert batch.node_label_logits is None
    assert batch.node_label_probabilities is None
    assert batch.edge_existence_probabilities is None
    assert batch.edge_label_logits is None
    assert batch.edge_label_probabilities is None
    assert batch.node_labels is None
    assert batch.edge_probability_matrices is None
    assert batch.edge_label_matrices is None


def test_predict_classifier_guided_returns_rich_distribution_tensors():
    generated = np.asarray(
        [
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            [[4.0, 0.0], [5.0, 0.0], [6.0, 0.0]],
        ],
        dtype=float,
    )
    model = _PredictiveStubModel(generated=generated, cached_outputs=_rich_cached_outputs())
    generator = _make_stubbed_node_field_generator(model)
    generator.guidance_predictor_ = object()
    generator.guidance_predictor_mode_ = "classification"
    generator.guidance_predictor_label_to_index_ = {"low": 0, "high": 1}
    generator._classification_guidance_gradient = lambda x, base_condition, desired: torch.zeros_like(x)

    batch = generator.predict_classifier_guided(
        _sample_graph_conditioning(),
        desired_class=["low", "high"],
        classifier_scale=1.5,
    )

    _assert_rich_generated_batch(batch)
    assert model.generate_calls[-1]["classifier_scale"] == pytest.approx(1.5)
    assert callable(model.generate_calls[-1]["classifier_guidance_fn"])


def test_predict_regression_guided_returns_rich_distribution_tensors():
    generated = np.asarray(
        [
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            [[4.0, 0.0], [5.0, 0.0], [6.0, 0.0]],
        ],
        dtype=float,
    )
    model = _PredictiveStubModel(generated=generated, cached_outputs=_rich_cached_outputs())
    generator = _make_stubbed_node_field_generator(model)
    generator.guidance_predictor_ = object()
    generator.guidance_predictor_mode_ = "regression"
    generator.guidance_predictor_target_scaler_ = MinMaxScaler().fit(
        np.asarray([[0.0], [1.0]], dtype=float)
    )
    generator._regression_guidance_gradient = lambda x, base_condition, desired: torch.zeros_like(x)

    batch = generator.predict_regression_guided(
        _sample_graph_conditioning(),
        desired_target=[0.25, 0.75],
        predictor_scale=2.0,
    )

    _assert_rich_generated_batch(batch)
    assert model.generate_calls[-1]["classifier_scale"] == pytest.approx(2.0)
    assert callable(model.generate_calls[-1]["classifier_guidance_fn"])


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


def test_decode_forwards_diagnostic_graph_renderer_with_decoded_graph(monkeypatch):
    plot_calls = []

    def fake_plot_decoder_diagnostics(**kwargs):
        plot_calls.append(kwargs)

    monkeypatch.setattr(cngg_module, "_plot_decoder_diagnostics", fake_plot_decoder_diagnostics)

    def fake_renderer(graphs, legends=None):
        return None

    decoder = ConditionalNodeFieldGraphDecoder(
        verbose=4,
        n_jobs=1,
        diagnostic_graph_renderer=fake_renderer,
    )
    generated_nodes = GeneratedNodeBatch(
        node_presence_mask=np.asarray([[True, True]], dtype=bool),
        node_degree_predictions=np.asarray([[1, 1]], dtype=float),
        edge_probability_matrices=[np.asarray([[0.0, 0.9], [0.9, 0.0]], dtype=float)],
    )
    predicted_node_labels = [np.asarray(["C", "O"], dtype=object)]
    predicted_edge_label_matrices = [np.asarray([[None, "-"], ["-", None]], dtype=object)]

    decoded_graphs = decoder.decode(
        generated_nodes,
        predicted_node_labels_list=predicted_node_labels,
        predicted_edge_probability_matrices=generated_nodes.edge_probability_matrices,
        predicted_edge_label_matrices=predicted_edge_label_matrices,
    )

    assert len(plot_calls) == 1
    assert plot_calls[0]["graph_renderer"] is fake_renderer
    assert plot_calls[0]["decoded_graph"] is decoded_graphs[0]
    assert decoded_graphs[0].nodes[0]["label"] == "C"
    assert decoded_graphs[0].edges[(0, 1)]["label"] == "-"


def test_try_render_molecular_graph_inline_renders_image(monkeypatch):
    rendered = {}

    class _Axis:
        def imshow(self, image):
            rendered["image"] = image

        def set_title(self, value):
            rendered["title"] = value

        def set_axis_off(self):
            rendered["axis_off"] = True

    def fake_molecule_graphs_to_grid_image(graphs, legends, mols_per_row, sub_img_size):
        rendered["graphs"] = graphs
        rendered["legends"] = legends
        rendered["mols_per_row"] = mols_per_row
        rendered["sub_img_size"] = sub_img_size
        return np.ones((4, 4, 3), dtype=np.uint8) * 255

    monkeypatch.setattr(
        "conditional_node_field_graph_generator.extensions.molecular.molecule_graphs_to_grid_image",
        fake_molecule_graphs_to_grid_image,
    )

    graph = nx.Graph()
    graph.add_node(0, label="C")
    graph.add_node(1, label="O")
    graph.add_edge(0, 1, label="-")

    result = cngg_module._try_render_molecular_graph_inline(
        _Axis(),
        decoded_graph=graph,
        title="Decoder solve graph=0",
    )

    assert result is True
    assert rendered["legends"] == ["Decoder solve graph=0"]
    assert rendered["title"] == "Decoded graph"
    assert rendered["axis_off"] is True


def test_coerce_inline_image_array_reads_display_wrapper_bytes():
    from PIL import Image

    class _DisplayImage:
        def __init__(self, data):
            self.data = data

    buffer = io.BytesIO()
    Image.fromarray(np.ones((3, 4, 3), dtype=np.uint8) * 255).save(buffer, format="PNG")
    image_array = cngg_module._coerce_inline_image_array(_DisplayImage(buffer.getvalue()))

    assert image_array is not None
    assert image_array.shape == (3, 4, 3)


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


def test_optimize_adjacency_matrix_applies_forbidden_edge_set_cuts():
    decoder = ConditionalNodeFieldGraphDecoder(verbose=False, enforce_connectivity=True)
    prob_matrix = np.asarray(
        [
            [0.0, 0.95, 0.10, 0.95],
            [0.95, 0.0, 0.95, 0.10],
            [0.10, 0.95, 0.0, 0.95],
            [0.95, 0.10, 0.95, 0.0],
        ],
        dtype=float,
    )

    adj = decoder.optimize_adjacency_matrix(
        prob_matrix,
        [2, 2, 2, 2],
        forbidden_edge_sets=[
            [(1, 0), (2, 1), (3, 2), (0, 3)],
            [(0, 1), (1, 2), (2, 3), (3, 0)],
            [],
        ],
    )

    edge_set = frozenset((min(u, v), max(u, v)) for u, v in nx.from_numpy_array(adj).edges())
    assert edge_set != frozenset({(0, 1), (1, 2), (2, 3), (0, 3)})


def test_decode_generated_nodes_uses_oracle_cuts_when_available():
    first_cycle = frozenset({(0, 1), (1, 2), (2, 3), (0, 3)})
    second_cycle = frozenset({(0, 1), (1, 3), (2, 3), (0, 2)})
    estimator = _OracleOnceEstimator([
        [[(1, 0), (2, 1), (3, 2), (3, 0)], [(0, 1), (3, 1), (3, 2), (2, 0)], []],
        [],
    ])
    generator = ConditionalNodeFieldGraphGenerator(
        graph_decoder=ConditionalNodeFieldGraphDecoder(verbose=False, enforce_connectivity=True),
        feasibility_estimator=estimator,
        verbose=False,
    )

    decoded = generator._decode_generated_nodes(_oracle_generated_batch())

    assert len(decoded) == 1
    decoded_edge_set = frozenset((min(u, v), max(u, v)) for u, v in decoded[0].edges())
    assert decoded_edge_set not in {first_cycle, second_cycle}
    assert len(estimator.calls) >= 2
    assert estimator.calls[0] == first_cycle


def test_decode_generated_nodes_relaxes_oracle_cuts_to_zero_on_last_attempt(monkeypatch):
    estimator = _OracleOnceEstimator([
        [[(0, 1), (1, 2), (2, 3), (0, 3)]],
        [[(0, 1), (1, 2), (2, 3), (0, 3)]],
    ])
    decoder = ConditionalNodeFieldGraphDecoder(verbose=False, enforce_connectivity=True)
    generator = ConditionalNodeFieldGraphGenerator(
        graph_decoder=decoder,
        feasibility_estimator=estimator,
        verbose=False,
        max_oracle_iterations=3,
    )

    original_optimize = decoder.optimize_adjacency_matrix
    active_cut_counts = []

    def wrapped_optimize(prob_matrix, target_degrees, *args, forbidden_edge_sets=None, **kwargs):
        active_cut_counts.append(len(list(forbidden_edge_sets or [])))
        if forbidden_edge_sets:
            raise RuntimeError("forced infeasible under oracle cuts")
        return original_optimize(
            prob_matrix,
            target_degrees,
            *args,
            forbidden_edge_sets=forbidden_edge_sets,
            **kwargs,
        )

    monkeypatch.setattr(decoder, "optimize_adjacency_matrix", wrapped_optimize)

    decoded = generator._decode_generated_nodes(_oracle_generated_batch())

    assert len(decoded) == 1
    assert active_cut_counts[-2:] == [1, 0]


def test_decode_generated_nodes_repairs_node_labels_before_structural_cuts():
    estimator = _LabelAwareOracleEstimator(
        edge_sets_per_call=[[], []],
        node_sets_per_call=[[[0]], []],
    )
    generator = ConditionalNodeFieldGraphGenerator(verbose=False)
    generator.feasibility_estimator = estimator
    generator.node_label_classes_ = np.asarray(["C", "O"], dtype=object)
    generator.node_label_to_index_ = {"C": 0, "O": 1}
    generator.edge_label_classes_ = np.asarray(["-"], dtype=object)
    generator.edge_label_to_index_ = {"-": 0}

    decoded = generator._decode_generated_nodes(
        _oracle_label_generated_batch(
            node_labels=["C", "C"],
            edge_label_matrix=[[None, "-"], ["-", None]],
            node_label_probabilities=[
                [0.05, 0.95],
                [0.90, 0.10],
            ],
            edge_label_probabilities=[
                [[1.0], [1.0]],
                [[1.0], [1.0]],
            ],
        )
    )

    assert len(decoded) == 1
    assert decoded[0].nodes[0]["label"] == "O"
    assert sorted(decoded[0].edges()) == [(0, 1)]
    assert estimator.node_calls >= 2


def test_decode_generated_nodes_repairs_edge_labels_before_structural_cuts():
    estimator = _LabelAwareOracleEstimator(
        edge_sets_per_call=[[[(0, 1)]], []],
        node_sets_per_call=[[], []],
    )
    generator = ConditionalNodeFieldGraphGenerator(verbose=False)
    generator.feasibility_estimator = estimator
    generator.node_label_classes_ = np.asarray(["C"], dtype=object)
    generator.node_label_to_index_ = {"C": 0}
    generator.edge_label_classes_ = np.asarray(["-", "="], dtype=object)
    generator.edge_label_to_index_ = {"-": 0, "=": 1}

    decoded = generator._decode_generated_nodes(
        _oracle_label_generated_batch(
            node_labels=["C", "C"],
            edge_label_matrix=[[None, "-"], ["-", None]],
            node_label_probabilities=[
                [1.0],
                [1.0],
            ],
            edge_label_probabilities=[
                [[1.0, 0.0], [0.05, 0.95]],
                [[0.05, 0.95], [1.0, 0.0]],
            ],
        )
    )

    assert len(decoded) == 1
    assert decoded[0].edges[(0, 1)]["label"] == "="
    assert sorted(decoded[0].edges()) == [(0, 1)]
    assert estimator.edge_calls >= 2


def test_decode_generated_nodes_skips_node_label_repair_without_probabilities():
    estimator = _LabelAwareOracleEstimator(
        edge_sets_per_call=[[], []],
        node_sets_per_call=[[[0]], [[0]]],
    )
    generator = ConditionalNodeFieldGraphGenerator(verbose=False, max_oracle_iterations=2)
    generator.feasibility_estimator = estimator
    generator.node_label_classes_ = np.asarray(["C", "O"], dtype=object)
    generator.node_label_to_index_ = {"C": 0, "O": 1}

    decoded = generator._decode_generated_nodes(
        _oracle_label_generated_batch(
            node_labels=["C", "C"],
            edge_label_matrix=[[None, "-"], ["-", None]],
            node_label_probabilities=None,
            edge_label_probabilities=None,
        )
    )

    assert len(decoded) == 1
    assert decoded[0].nodes[0]["label"] == "C"


def test_oracle_candidate_score_prefers_higher_probability_feasible_labels():
    generator = ConditionalNodeFieldGraphGenerator(verbose=False)
    generator.node_label_to_index_ = {"C": 0, "O": 1}
    generator.edge_label_to_index_ = {"-": 0, "=": 1}

    existence_mask = np.asarray([True, True], dtype=bool)
    adj_mtx = np.asarray([[0, 1], [1, 0]], dtype=float)
    edge_probs = np.asarray([[0.0, 0.9], [0.9, 0.0]], dtype=float)
    node_label_probs = np.asarray([[0.1, 0.9], [0.8, 0.2]], dtype=float)
    edge_label_probs = np.asarray(
        [
            [[1.0, 0.0], [0.8, 0.2]],
            [[0.8, 0.2], [1.0, 0.0]],
        ],
        dtype=float,
    )

    score_high = generator._oracle_candidate_score(
        existence_mask=existence_mask,
        adj_mtx=adj_mtx,
        node_labels=np.asarray(["O", "C"], dtype=object),
        edge_label_matrix=np.asarray([[None, "-"], ["-", None]], dtype=object),
        edge_probability_matrix=edge_probs,
        node_label_probabilities=node_label_probs,
        edge_label_probabilities=edge_label_probs,
    )
    score_low = generator._oracle_candidate_score(
        existence_mask=existence_mask,
        adj_mtx=adj_mtx,
        node_labels=np.asarray(["C", "C"], dtype=object),
        edge_label_matrix=np.asarray([[None, "="], ["=", None]], dtype=object),
        edge_probability_matrix=edge_probs,
        node_label_probabilities=node_label_probs,
        edge_label_probabilities=edge_label_probs,
    )

    assert score_high > score_low


def test_decode_generated_nodes_falls_back_when_oracle_method_missing():
    generator = ConditionalNodeFieldGraphGenerator(verbose=False)
    generator.feasibility_estimator = _FakeFeasibilityEstimator()
    generator.graph_decoder = _CollectDecoderStub()

    decoded = generator._decode_generated_nodes(
        GeneratedNodeBatch(
            node_embeddings_list=[np.asarray([[0.0], [1.0]], dtype=float)],
            node_presence_mask=np.asarray([[True, True]], dtype=bool),
            edge_probability_matrices=[np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float)],
        )
    )

    assert len(decoded) == 1
    assert decoded[0].graph["feasible"] is True


def test_can_use_feasibility_oracle_respects_attempt_budget():
    generator = ConditionalNodeFieldGraphGenerator(
        feasibility_oracle_candidates_per_attempt=2,
        verbose=False,
    )
    generator.feasibility_estimator = _OracleOnceEstimator([[]])

    assert generator._can_use_feasibility_oracle() is True
    assert generator._can_use_feasibility_oracle(attempt_idx=0) is True
    assert generator._can_use_feasibility_oracle(attempt_idx=1) is True
    assert generator._can_use_feasibility_oracle(attempt_idx=2) is False
    assert generator._can_use_feasibility_oracle(
        feasibility_oracle_candidates_per_attempt=0,
        attempt_idx=0,
    ) is False


def test_apply_legacy_feasibility_config_maps_old_boolean_flag():
    generator = ConditionalNodeFieldGraphGenerator(verbose=False)
    del generator.feasibility_oracle_candidates_per_attempt
    generator.use_feasibility_oracle = False

    generator._apply_legacy_feasibility_config()

    assert generator.feasibility_oracle_candidates_per_attempt == 0

    del generator.feasibility_oracle_candidates_per_attempt
    generator.use_feasibility_oracle = True

    generator._apply_legacy_feasibility_config()

    assert generator.feasibility_oracle_candidates_per_attempt == 2


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
