import io
import json
import os
import multiprocessing as mp
import time
import types
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import networkx as nx
import pandas as pd
import pytest
import pulp
import torch
from scipy import sparse
from sklearn.preprocessing import MinMaxScaler

import conditional_node_field_graph_generator.conditional_node_field_graph_generator as cngg_module
import conditional_node_field_graph_generator.conditional_node_field_graph_decoder as decoder_module
import conditional_node_field_graph_generator.oracle_decode as oracle_decode_module
import conditional_node_field_graph_generator.parallel_utils as parallel_utils
import conditional_node_field_graph_generator.structural_decoder as structural_decoder
from conditional_node_field_graph_generator.persistence import (
    load_graph_generator,
    save_graph_generator,
)
from conditional_node_field_graph_generator.encoding_pipeline import EncodingPipeline
from conditional_node_field_graph_generator.conditioning_sampler import ConditioningSampler
from conditional_node_field_graph_generator.node_batch_builder import NodeBatchBuilder
from conditional_node_field_graph_generator.stream_fit import StreamFitService
from conditional_node_field_graph_generator.supervision import SupervisionPlanner
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
from conditional_node_field_graph_generator.conditional_node_field_graph_decoder import (
    build_single_generated_node_batch,
)
from conditional_node_field_graph_generator.oracle_utils import (
    enumerate_localized_edge_addition_proposals,
)


def _sleep_and_return(job):
    delay, value = job
    time.sleep(delay)
    return value


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
    def __init__(self):
        self.fitted_graph_count = None

    def fit(self, graphs):
        self.fitted_graph_count = len(graphs)
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


class _SparseGraphVectorizer:
    def __init__(self, dimension=10):
        self.dimension = int(dimension)
        self.fitted_graph_count = None

    def fit(self, graphs):
        self.fitted_graph_count = len(graphs)
        return self

    def transform(self, graphs):
        rows = []
        cols = []
        data = []
        for graph_idx, graph in enumerate(graphs):
            for col_idx, value in (
                (0, graph.number_of_nodes()),
                (1, graph.number_of_edges()),
                (2 + graph.number_of_nodes() % max(1, self.dimension - 2), 1.0),
                (2 + graph.number_of_edges() % max(1, self.dimension - 2), 0.5),
            ):
                rows.append(graph_idx)
                cols.append(int(col_idx % self.dimension))
                data.append(float(value))
        return sparse.csr_matrix((data, (rows, cols)), shape=(len(graphs), self.dimension))


class _SparseNodeVectorizer:
    def __init__(self, dimension=12):
        self.dimension = int(dimension)
        self.fitted_graph_count = None

    def fit(self, graphs):
        self.fitted_graph_count = len(graphs)
        return self

    def transform(self, graphs):
        matrices = []
        for graph in graphs:
            rows = []
            cols = []
            data = []
            for row_idx, node in enumerate(graph.nodes()):
                degree = graph.degree(node)
                for col_idx, value in (
                    (0, degree + 1.0),
                    (1 + int(node) % max(1, self.dimension - 1), 1.0),
                    (1 + degree % max(1, self.dimension - 1), 0.5),
                ):
                    rows.append(row_idx)
                    cols.append(int(col_idx % self.dimension))
                    data.append(float(value))
            matrices.append(
                sparse.csr_matrix(
                    (data, (rows, cols)),
                    shape=(graph.number_of_nodes(), self.dimension),
                )
            )
        return matrices


class _Component:
    def __init__(self, verbose=False):
        self.verbose = verbose


class _TrainableNodeModel(_Component):
    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)
        self.setup_calls = []
        self.fit_calls = []
        self.fit_from_prebuilt_batches_calls = []

    def setup(self, **kwargs):
        self.setup_calls.append(kwargs)

    def fit(self, **kwargs):
        self.fit_calls.append(kwargs)


class _StreamTrainableNodeModel(_TrainableNodeModel):
    def __init__(self, verbose=False, maximum_epochs=1):
        super().__init__(verbose=verbose)
        self.maximum_epochs = int(maximum_epochs)

    def setup(self, **kwargs):
        super().setup(**kwargs)
        node_batch = kwargs["node_batch"]
        self.number_of_rows_per_example = int(node_batch.node_presence_mask.shape[1])
        node_label_targets = node_batch.node_label_targets or []
        flat_node_labels = [
            label
            for labels in node_label_targets
            for label in np.asarray(labels, dtype=object).tolist()
        ]
        self.node_label_to_index_ = {
            label: idx for idx, label in enumerate(np.unique(np.asarray(flat_node_labels, dtype=object)))
        } if flat_node_labels else {}
        edge_label_targets = node_batch.edge_label_targets
        if edge_label_targets is None:
            self.edge_label_to_index_ = {}
        else:
            unique_edge_labels = np.unique(np.asarray(edge_label_targets, dtype=object))
            self.edge_label_to_index_ = {label: idx for idx, label in enumerate(unique_edge_labels.tolist())}
        self.guidance_enabled_ = False
        self.target_condition_dim_ = 0

    def _build_processed_training_payload(self, node_batch, graph_conditioning, targets=None):
        del targets
        return {
            "graphs": len(node_batch),
            "conditioning_rows": len(graph_conditioning),
            "max_rows": int(node_batch.node_presence_mask.shape[1]),
        }

    def _collate_processed_payload(self, payload):
        return payload

    def fit_from_prebuilt_batches(self, validation_node_batch, validation_graph_conditioning, batch_iter_factory, ckpt_path=None):
        epoch_batches = [list(batch_iter_factory()) for _ in range(max(1, int(self.maximum_epochs)))]
        self.fit_from_prebuilt_batches_calls.append(
            {
                "validation_graphs": len(validation_node_batch),
                "validation_conditioning": len(validation_graph_conditioning),
                "batches": epoch_batches,
                "ckpt_path": ckpt_path,
            }
        )


class _EdgeSupervisionDecoder:
    def __init__(self, raise_on_flag=None):
        self.raise_on_flag = raise_on_flag

    def compute_edge_supervision(self, graphs, node_embeddings_list, **kwargs):
        del node_embeddings_list, kwargs
        if self.raise_on_flag is not None and any(graph.graph.get(self.raise_on_flag) for graph in graphs):
            raise RuntimeError(f"decoder blocked on {self.raise_on_flag}")
        return np.zeros((0,), dtype=float), []


class _FitRecorderEstimator:
    def __init__(self):
        self.fit_graph_count = None

    def fit(self, graphs):
        self.fit_graph_count = len(graphs)
        return self


class _ExplodingNodeVectorizer(_NodeVectorizer):
    def transform(self, graphs):
        if any(graph.graph.get("explode_transform") for graph in graphs):
            raise RuntimeError("transform failed")
        return super().transform(graphs)


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


def _stream_reader(graphs):
    def _reader(_uri):
        return iter(graphs)
    return _reader


def _labelled_path(node_count, node_label="C", edge_label="-"):
    graph = nx.path_graph(node_count)
    for node in graph.nodes():
        graph.nodes[node]["label"] = node_label
    for u, v in graph.edges():
        graph.edges[u, v]["label"] = edge_label
    return graph


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
        "_last_node_existence_probabilities": torch.tensor(
            [[0.9, 0.8, 0.1], [0.4, 0.3, 0.2]],
            dtype=torch.float32,
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
    assert batch.node_existence_probabilities is not None
    assert batch.edge_existence_probabilities is not None
    assert batch.edge_label_logits is not None
    assert batch.edge_label_probabilities is not None
    assert batch.node_label_logits[0].shape == (3, 3)
    assert batch.node_label_probabilities[0].shape == (3, 3)
    assert batch.node_existence_probabilities.shape == (2, 3)
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
    with pytest.raises(ValueError, match="oracle_add_edge_repair_budget"):
        ConditionalNodeFieldGraphGenerator(oracle_add_edge_repair_budget=-1)
    with pytest.raises(ValueError, match="oracle_edge_label_min_changes_per_violation"):
        ConditionalNodeFieldGraphGenerator(oracle_edge_label_min_changes_per_violation=0)
    with pytest.raises(ValueError, match="oracle_edge_memory_penalty"):
        ConditionalNodeFieldGraphGenerator(oracle_edge_memory_penalty=-0.1)
    with pytest.raises(ValueError, match="oracle_edge_memory_update"):
        ConditionalNodeFieldGraphGenerator(oracle_edge_memory_update=-0.1)
    with pytest.raises(ValueError, match="oracle_edge_memory_decay"):
        ConditionalNodeFieldGraphGenerator(oracle_edge_memory_decay=1.1)
    with pytest.raises(ValueError, match="oracle_edge_memory_clip"):
        ConditionalNodeFieldGraphGenerator(oracle_edge_memory_clip=-0.1)
    with pytest.raises(ValueError, match="feasibility_failure_mode"):
        ConditionalNodeFieldGraphGenerator(feasibility_failure_mode="drop")


def test_graph_generator_defaults_to_ten_oracle_iterations():
    generator = ConditionalNodeFieldGraphGenerator()

    assert generator.max_oracle_iterations == 10
    assert generator.oracle_add_edge_repair_budget == 32


def test_fit_from_stream_reuses_cached_warmup_batches_during_training():
    graph_vectorizer = _GraphVectorizer()
    node_vectorizer = _NodeVectorizer()
    node_model = _StreamTrainableNodeModel()
    estimator = _FitRecorderEstimator()
    graphs = [
        _labelled_path(2, node_label="C"),
        _labelled_path(2, node_label="C"),
        _labelled_path(2, node_label="C"),
        _labelled_path(2, node_label="C"),
        _labelled_path(2, node_label="C"),
    ]
    generator = ConditionalNodeFieldGraphGenerator(
        graph_vectorizer=graph_vectorizer,
        node_graph_vectorizer=node_vectorizer,
        conditional_node_generator_model=node_model,
        graph_decoder=_EdgeSupervisionDecoder(),
        feasibility_estimator=estimator,
        verbose=False,
    )

    generator.fit_from_stream(
        "ignored",
        "custom",
        reader=_stream_reader(graphs),
        warmup_size=2,
        batch_size=2,
    )

    assert graph_vectorizer.fitted_graph_count == 2
    assert node_vectorizer.fitted_graph_count == 2
    assert estimator.fit_graph_count == 2
    assert len(generator.training_graph_conditioning_) == 2
    assert node_model.setup_calls[0]["targets"] is None
    assert node_model.fit_from_prebuilt_batches_calls[0]["validation_graphs"] == 2
    assert [
        batch["graphs"] for batch in node_model.fit_from_prebuilt_batches_calls[0]["batches"][0]
    ] == [2, 1]
    assert generator.stream_training_seen_ == 3
    assert generator.stream_training_accepted_ == 3
    assert generator.stream_training_skipped_ == 0
    assert generator.warmup_schema_frozen_ is True


def test_fit_from_stream_reuses_warmup_batches_and_restarts_post_warmup_stream_for_multiple_epochs():
    node_model = _StreamTrainableNodeModel(maximum_epochs=2)
    graphs = [
        _labelled_path(2, node_label="C"),
        _labelled_path(2, node_label="C"),
        _labelled_path(2, node_label="C"),
        _labelled_path(2, node_label="C"),
    ]
    generator = ConditionalNodeFieldGraphGenerator(
        graph_vectorizer=_GraphVectorizer(),
        node_graph_vectorizer=_NodeVectorizer(),
        conditional_node_generator_model=node_model,
        graph_decoder=_EdgeSupervisionDecoder(),
        verbose=False,
    )

    generator.fit_from_stream(
        "ignored",
        "custom",
        reader=_stream_reader(graphs),
        warmup_size=2,
        batch_size=1,
    )

    epoch_batches = node_model.fit_from_prebuilt_batches_calls[0]["batches"]
    assert len(epoch_batches) == 2
    assert [[batch["graphs"] for batch in epoch] for epoch in epoch_batches] == [[1, 1, 1], [1, 1, 1]]
    assert generator.stream_warmup_count_ == 2
    assert generator.stream_training_seen_ == 6
    assert generator.stream_training_accepted_ == 6


def test_fit_from_stream_uses_first_post_warmup_batch_for_validation():
    node_model = _StreamTrainableNodeModel(maximum_epochs=1)
    graphs = [
        _labelled_path(2, node_label="C"),
        _labelled_path(2, node_label="C"),
        _labelled_path(2, node_label="C"),
        _labelled_path(2, node_label="C"),
        _labelled_path(2, node_label="C"),
    ]
    generator = ConditionalNodeFieldGraphGenerator(
        graph_vectorizer=_GraphVectorizer(),
        node_graph_vectorizer=_NodeVectorizer(),
        conditional_node_generator_model=node_model,
        graph_decoder=_EdgeSupervisionDecoder(),
        verbose=False,
    )

    generator.fit_from_stream(
        "ignored",
        "custom",
        reader=_stream_reader(graphs),
        warmup_size=3,
        batch_size=2,
    )

    fit_call = node_model.fit_from_prebuilt_batches_calls[0]
    assert fit_call["validation_graphs"] == 2
    assert [batch["graphs"] for batch in fit_call["batches"][0]] == []
    assert generator.stream_warmup_count_ == 3
    assert generator.stream_training_seen_ == 0
    assert generator.stream_training_accepted_ == 0


def test_fit_from_stream_skips_unknown_node_labels():
    node_model = _StreamTrainableNodeModel()
    graphs = [
        _labelled_path(2, node_label="C"),
        _labelled_path(2, node_label="C"),
        _labelled_path(2, node_label="X"),
        _labelled_path(2, node_label="C"),
    ]
    generator = ConditionalNodeFieldGraphGenerator(
        graph_vectorizer=_GraphVectorizer(),
        node_graph_vectorizer=_NodeVectorizer(),
        conditional_node_generator_model=node_model,
        graph_decoder=_EdgeSupervisionDecoder(),
        verbose=False,
    )

    generator.fit_from_stream(
        "ignored",
        "custom",
        reader=_stream_reader(graphs),
        warmup_size=2,
        batch_size=1,
    )

    assert generator.stream_training_seen_ == 2
    assert generator.stream_training_accepted_ == 2
    assert generator.stream_training_skipped_ == 0
    assert generator.stream_skipped_unknown_node_label_ == 0


def test_fit_from_stream_skips_unknown_edge_labels_when_edge_labels_are_learned():
    node_model = _StreamTrainableNodeModel()
    warmup_a = _labelled_path(2, edge_label="-")
    warmup_b = _labelled_path(2, edge_label="=")
    bad_graph = _labelled_path(2, edge_label="#")
    generator = ConditionalNodeFieldGraphGenerator(
        graph_vectorizer=_GraphVectorizer(),
        node_graph_vectorizer=_NodeVectorizer(),
        conditional_node_generator_model=node_model,
        graph_decoder=_EdgeSupervisionDecoder(),
        verbose=False,
    )

    generator.fit_from_stream(
        "ignored",
        "custom",
        reader=_stream_reader([warmup_a, warmup_b, bad_graph]),
        warmup_size=2,
        batch_size=1,
    )

    assert generator.stream_training_seen_ == 0
    assert generator.stream_training_accepted_ == 0
    assert generator.stream_skipped_unknown_edge_label_ == 0
    assert node_model.fit_from_prebuilt_batches_calls == []


def test_fit_from_stream_skips_graphs_larger_than_warmup_schema():
    node_model = _StreamTrainableNodeModel()
    graphs = [
        _labelled_path(2),
        _labelled_path(2),
        _labelled_path(3),
    ]
    generator = ConditionalNodeFieldGraphGenerator(
        graph_vectorizer=_GraphVectorizer(),
        node_graph_vectorizer=_NodeVectorizer(),
        conditional_node_generator_model=node_model,
        graph_decoder=_EdgeSupervisionDecoder(),
        verbose=False,
    )

    generator.fit_from_stream(
        "ignored",
        "custom",
        reader=_stream_reader(graphs),
        warmup_size=2,
        batch_size=1,
    )

    assert generator.stream_training_seen_ == 0
    assert generator.stream_training_accepted_ == 0
    assert generator.stream_skipped_too_large_ == 0
    assert node_model.fit_from_prebuilt_batches_calls == []


def test_fit_from_stream_counts_transform_errors():
    node_model = _StreamTrainableNodeModel()
    graphs = [
        _labelled_path(2),
        _labelled_path(2),
        _labelled_path(2),
    ]
    graphs[2].graph["explode_transform"] = True
    generator = ConditionalNodeFieldGraphGenerator(
        graph_vectorizer=_GraphVectorizer(),
        node_graph_vectorizer=_ExplodingNodeVectorizer(),
        conditional_node_generator_model=node_model,
        graph_decoder=_EdgeSupervisionDecoder(),
        verbose=False,
    )

    generator.fit_from_stream(
        "ignored",
        "custom",
        reader=_stream_reader(graphs),
        warmup_size=2,
        batch_size=1,
    )

    assert generator.stream_training_seen_ == 0
    assert generator.stream_training_accepted_ == 0
    assert generator.stream_skipped_transform_error_ == 0
    assert node_model.fit_from_prebuilt_batches_calls == []


def test_fit_from_stream_counts_supervision_errors():
    node_model = _StreamTrainableNodeModel()
    graphs = [
        _labelled_path(2),
        _labelled_path(2),
        _labelled_path(2),
    ]
    graphs[2].graph["explode_supervision"] = True
    generator = ConditionalNodeFieldGraphGenerator(
        graph_vectorizer=_GraphVectorizer(),
        node_graph_vectorizer=_NodeVectorizer(),
        conditional_node_generator_model=node_model,
        graph_decoder=_EdgeSupervisionDecoder(raise_on_flag="explode_supervision"),
        verbose=False,
    )

    generator.fit_from_stream(
        "ignored",
        "custom",
        reader=_stream_reader(graphs),
        warmup_size=2,
        batch_size=1,
    )

    assert generator.stream_training_seen_ == 0
    assert generator.stream_training_accepted_ == 0
    assert generator.stream_skipped_supervision_error_ == 0
    assert node_model.fit_from_prebuilt_batches_calls == []


def test_fit_from_stream_rejects_empty_source():
    generator = ConditionalNodeFieldGraphGenerator(
        graph_vectorizer=_GraphVectorizer(),
        node_graph_vectorizer=_NodeVectorizer(),
        conditional_node_generator_model=_StreamTrainableNodeModel(),
        graph_decoder=_EdgeSupervisionDecoder(),
        verbose=False,
    )

    with pytest.raises(ValueError, match="could not load any graphs"):
        generator.fit_from_stream(
            "ignored",
            "custom",
            reader=_stream_reader([]),
            warmup_size=2,
        )


def test_fit_from_stream_keeps_cfg_target_state_disabled():
    node_model = ConditionalNodeFieldGenerator(
        verbose=False,
        maximum_epochs=1,
        batch_size=2,
    )
    graphs = [_labelled_path(2), _labelled_path(2)]
    generator = ConditionalNodeFieldGraphGenerator(
        graph_vectorizer=_GraphVectorizer(),
        node_graph_vectorizer=_NodeVectorizer(),
        conditional_node_generator_model=node_model,
        graph_decoder=_EdgeSupervisionDecoder(),
        verbose=False,
    )

    generator.fit_from_stream(
        "ignored",
        "custom",
        reader=_stream_reader(graphs),
        warmup_size=2,
        batch_size=1,
    )

    assert node_model.guidance_enabled_ is False
    assert node_model.target_condition_dim_ == 0


def test_fit_from_stream_accepts_smiles_csv_source(tmp_path):
    pytest.importorskip("abstractgraph_graphicalizer.chem")
    csv_path = tmp_path / "tiny_zinc.csv"
    pd.DataFrame(
        {
            "smiles": ["CC", "CCC", "CCCC"],
            "logP": [1.0, 2.0, 3.0],
        }
    ).to_csv(csv_path, index=False)
    node_model = _StreamTrainableNodeModel()
    generator = ConditionalNodeFieldGraphGenerator(
        graph_vectorizer=_GraphVectorizer(),
        node_graph_vectorizer=_NodeVectorizer(),
        conditional_node_generator_model=node_model,
        graph_decoder=_EdgeSupervisionDecoder(),
        verbose=False,
    )

    generator.fit_from_stream(
        csv_path,
        "smiles_csv",
        warmup_size=1,
        batch_size=1,
    )

    assert generator.stream_warmup_count_ == 1
    assert generator.stream_training_seen_ == 0
    assert generator.stream_training_accepted_ == 0
    assert generator.stream_training_skipped_ == 0
    assert generator.stream_skipped_too_large_ == 0
    assert node_model.fit_from_prebuilt_batches_calls == []


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


def test_fit_configures_training_sample_progress_on_node_generator(tmp_path):
    class _ProgressNodeModel(_TrainableNodeModel):
        def fit(self, **kwargs):
            config = self._graph_generator_sample_progress_config
            self.progress_state = {
                "enabled": config.enabled,
                "n_samples": config.n_samples,
                "every": config.every_n_epochs,
                "pdf_path": config.output_path,
                "plot_kwargs": config.plot_kwargs,
                "plot_fn": config.plot_fn,
            }
            super().fit(**kwargs)

    node_model = _ProgressNodeModel(verbose=False)
    generator = ConditionalNodeFieldGraphGenerator(
        graph_vectorizer=_GraphVectorizer(),
        node_graph_vectorizer=_NodeVectorizer(),
        conditional_node_generator_model=node_model,
        graph_decoder=ConditionalNodeFieldGraphDecoder(verbose=False),
        verbose=False,
    )

    def _plot_fn(ax, graph, title=None):
        del ax, graph, title

    generator.fit(
        [_labeled_graph()],
        train_node_generator=True,
        sample_training_progress=True,
        sample_training_progress_n_samples=5,
        sample_training_progress_every_n_epochs=2,
        sample_training_progress_pdf_path=tmp_path / "samples.pdf",
        sample_training_progress_plot_kwargs={
            "node_label_colors": {0: "#ffffff"},
            "size": 2.5,
        },
        sample_training_progress_plot_fn=_plot_fn,
    )

    assert node_model.progress_state == {
        "enabled": True,
        "n_samples": 5,
        "every": 2,
        "pdf_path": os.path.expanduser(str(tmp_path / "samples.pdf")),
        "plot_kwargs": {
            "node_label_colors": {0: "#ffffff"},
            "size": 2.5,
        },
        "plot_fn": _plot_fn,
    }
    assert not hasattr(node_model, "_graph_generator_sample_progress_config")


def test_fit_rejects_invalid_training_sample_progress_options():
    generator = ConditionalNodeFieldGraphGenerator(
        graph_vectorizer=_GraphVectorizer(),
        node_graph_vectorizer=_NodeVectorizer(),
        conditional_node_generator_model=_TrainableNodeModel(verbose=False),
        graph_decoder=ConditionalNodeFieldGraphDecoder(verbose=False),
        verbose=False,
    )

    with pytest.raises(ValueError, match="sample_training_progress_n_samples"):
        generator.fit(
            [_labeled_graph()],
            train_node_generator=False,
            sample_training_progress_n_samples=0,
        )
    with pytest.raises(ValueError, match="sample_training_progress_every_n_epochs"):
        generator.fit(
            [_labeled_graph()],
            train_node_generator=False,
            sample_training_progress_every_n_epochs=0,
        )


def test_sample_return_decode_stages_reuses_single_generated_batch():
    generator = ConditionalNodeFieldGraphGenerator(verbose=False)
    generator.is_fitted_ = True
    generator.conditional_node_generator_model = object()
    generator.graph_decoder = object()
    conditioning = GraphConditioningBatch(
        graph_embeddings=np.zeros((2, 3), dtype=float),
        node_counts=np.array([2, 2], dtype=np.int64),
        edge_counts=np.array([1, 1], dtype=np.int64),
    )
    generated = GeneratedNodeBatch(
        node_presence_mask=np.ones((2, 1), dtype=bool),
    )
    prediction_calls = []
    decode_calls = []

    def _graph(label):
        graph = nx.Graph()
        graph.add_node(0, label=label)
        return graph

    generator.feasibility_estimator = object()
    generator._sample_conditions = lambda n_samples, **kwargs: conditioning

    def _predict_generated_nodes(graph_conditioning, **kwargs):
        prediction_calls.append((graph_conditioning, kwargs))
        return generated

    def _decode_generated_nodes(generated_nodes, **kwargs):
        decode_calls.append(("decode", generated_nodes, kwargs))
        label = "raw" if kwargs["use_ilp_decoder"] is False else "ilp"
        return [_graph(label), _graph(label)]

    def _decode_generated_nodes_with_oracle(generated_nodes, **kwargs):
        decode_calls.append(("oracle", generated_nodes, kwargs))
        return [_graph("oracle"), _graph("oracle")]

    generator._predict_generated_nodes = _predict_generated_nodes
    generator._decode_generated_nodes = _decode_generated_nodes
    generator._decode_generated_nodes_with_oracle = _decode_generated_nodes_with_oracle

    variants = generator.sample(2, return_decode_stages=True)

    assert prediction_calls == [
        (
            conditioning,
            {
                "sampling_mode": "unguided",
                "desired_target": None,
                "guidance_scale": 1.0,
            },
        )
    ]
    assert sorted(variants) == ["ilp", "oracle", "raw"]
    assert [graph.nodes[0]["label"] for graph in variants["raw"]] == ["raw", "raw"]
    assert [graph.nodes[0]["label"] for graph in variants["ilp"]] == ["ilp", "ilp"]
    assert [graph.nodes[0]["label"] for graph in variants["oracle"]] == ["oracle", "oracle"]
    assert len(decode_calls) == 6
    assert decode_calls[0][2]["use_ilp_decoder"] is False
    assert decode_calls[1][2]["use_ilp_decoder"] is True
    assert decode_calls[1][2]["feasibility_oracle_candidates_per_attempt"] == 0
    assert decode_calls[2][0] == "oracle"
    assert decode_calls[3][2]["use_ilp_decoder"] is False
    assert decode_calls[4][2]["use_ilp_decoder"] is True
    assert decode_calls[5][0] == "oracle"
    assert all(len(call[2]["graph_conditioning"]) == 1 for call in decode_calls)


def test_sample_return_decode_stages_marks_oracle_missing_without_estimator():
    generator = ConditionalNodeFieldGraphGenerator(verbose=False)
    generator.is_fitted_ = True
    generator.conditional_node_generator_model = object()
    generator.graph_decoder = object()
    conditioning = GraphConditioningBatch(
        graph_embeddings=np.zeros((2, 3), dtype=float),
        node_counts=np.array([2, 2], dtype=np.int64),
        edge_counts=np.array([1, 1], dtype=np.int64),
    )
    generator.feasibility_estimator = None
    generator._sample_conditions = lambda n_samples, **kwargs: conditioning
    generator._predict_generated_nodes = lambda graph_conditioning, **kwargs: GeneratedNodeBatch(
        node_presence_mask=np.ones((len(graph_conditioning), 1), dtype=bool),
    )
    generator._decode_generated_nodes = lambda generated_nodes, **kwargs: [nx.Graph(), nx.Graph()]
    generator._decode_generated_nodes_with_oracle = lambda *args, **kwargs: pytest.fail(
        "oracle decode should not be called without a feasibility estimator"
    )

    variants = generator.sample(2, return_decode_stages=True)

    assert variants["oracle"] == [None, None]


def test_sample_return_decode_stages_retries_after_timeout():
    generator = ConditionalNodeFieldGraphGenerator(
        verbose=False,
        max_decode_attempts_per_sample=2,
    )
    generator.is_fitted_ = True
    generator.conditional_node_generator_model = object()
    generator.graph_decoder = object()
    generator.feasibility_estimator = None
    conditioning = GraphConditioningBatch(
        graph_embeddings=np.zeros((1, 3), dtype=float),
        node_counts=np.array([2], dtype=np.int64),
        edge_counts=np.array([1], dtype=np.int64),
    )
    generated_batches = [
        GeneratedNodeBatch(node_presence_mask=np.ones((1, 1), dtype=bool)),
        GeneratedNodeBatch(node_presence_mask=np.ones((1, 1), dtype=bool)),
    ]
    prediction_calls = []
    ilp_calls = []
    generator._sample_conditions = lambda n_samples, **kwargs: conditioning

    def _predict_generated_nodes(graph_conditioning, **kwargs):
        del graph_conditioning, kwargs
        generated = generated_batches[len(prediction_calls)]
        prediction_calls.append(generated)
        return generated

    def _decode_generated_nodes(generated_nodes, **kwargs):
        if kwargs["use_ilp_decoder"]:
            ilp_calls.append(generated_nodes)
            if len(ilp_calls) == 1:
                raise TimeoutError("slow ILP")
        return [nx.Graph()]

    generator._predict_generated_nodes = _predict_generated_nodes
    generator._decode_generated_nodes = _decode_generated_nodes

    variants = generator.sample(1, return_decode_stages=True)

    assert prediction_calls == generated_batches
    assert ilp_calls == generated_batches
    assert len(variants["raw"]) == 1
    assert len(variants["ilp"]) == 1
    assert variants["oracle"] == [None]


def test_sample_return_decode_stages_skips_only_slot_after_retries_exhausted():
    generator = ConditionalNodeFieldGraphGenerator(
        verbose=False,
        max_decode_attempts_per_sample=2,
    )
    generator.is_fitted_ = True
    generator.conditional_node_generator_model = object()
    generator.graph_decoder = object()
    generator.feasibility_estimator = None
    conditioning = GraphConditioningBatch(
        graph_embeddings=np.zeros((2, 3), dtype=float),
        node_counts=np.array([2, 2], dtype=np.int64),
        edge_counts=np.array([1, 1], dtype=np.int64),
    )
    generator._sample_conditions = lambda n_samples, **kwargs: conditioning
    generator._predict_generated_nodes = lambda graph_conditioning, **kwargs: GeneratedNodeBatch(
        node_presence_mask=np.ones((len(graph_conditioning), 1), dtype=bool),
    )

    def _decode_generated_nodes(generated_nodes, **kwargs):
        slot_marker = float(kwargs["graph_conditioning"].graph_embeddings[0, 0])
        if kwargs["use_ilp_decoder"] and slot_marker == 0.0:
            raise TimeoutError("slow ILP")
        return [nx.Graph()]

    conditioning.graph_embeddings[1, 0] = 1.0
    generator._decode_generated_nodes = _decode_generated_nodes

    variants = generator.sample(2, return_decode_stages=True)

    assert variants["raw"][0] is None
    assert variants["ilp"][0] is None
    assert variants["oracle"][0] is None
    assert variants["raw"][1] is not None
    assert variants["ilp"][1] is not None
    assert variants["oracle"][1] is None


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


class _ConstructiveEdgeEstimator:
    def __init__(self, required_edges):
        self.required_edges = {
            (min(int(i), int(j)), max(int(i), int(j)), label)
            for i, j, label in required_edges
        }
        self.violation_batch_sizes = []

    def _missing_required_edges(self, graph):
        missing = []
        for i, j, label in sorted(self.required_edges, key=lambda item: (item[0], item[1], repr(item[2]))):
            if not graph.has_edge(i, j) or graph.edges[i, j].get("label") != label:
                missing.append((i, j, label))
        return missing

    def number_of_violations(self, graphs):
        self.violation_batch_sizes.append(len(graphs))
        return [len(self._missing_required_edges(graph)) for graph in graphs]

    def violating_edge_sets(self, graphs):
        output = []
        for graph in graphs:
            if not self._missing_required_edges(graph):
                output.append([])
                continue
            path_edges = frozenset(
                (min(int(i), int(j)), max(int(i), int(j)))
                for i, j in graph.edges()
            )
            output.append([path_edges] if path_edges else [])
        return output


def _constructive_oracle_generated_batch(n_nodes=4):
    edge_probabilities = np.full((n_nodes, n_nodes), 0.05, dtype=float)
    np.fill_diagonal(edge_probabilities, 0.0)
    for node_idx in range(n_nodes - 1):
        edge_probabilities[node_idx, node_idx + 1] = 0.95
        edge_probabilities[node_idx + 1, node_idx] = 0.95

    edge_label_matrix = np.full((n_nodes, n_nodes), None, dtype=object)
    edge_label_probabilities = np.zeros((n_nodes, n_nodes, 2), dtype=float)
    edge_label_probabilities[..., 0] = 0.8
    edge_label_probabilities[..., 1] = 0.2
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            edge_label_matrix[i, j] = edge_label_matrix[j, i] = "bad"

    return GeneratedNodeBatch(
        node_presence_mask=np.ones((1, n_nodes), dtype=bool),
        node_degree_predictions=np.asarray(
            [[1.0] + [2.0] * (n_nodes - 2) + [1.0]],
            dtype=float,
        ),
        node_labels=[np.asarray([f"node-{idx}" for idx in range(n_nodes)], dtype=object)],
        edge_probability_matrices=[edge_probabilities],
        edge_existence_probabilities=[edge_probabilities],
        edge_label_matrices=[edge_label_matrix],
        edge_label_probabilities=[edge_label_probabilities],
    )


def test_constructive_edge_proposals_are_localized_ranked_and_budgeted():
    proposals = enumerate_localized_edge_addition_proposals(
        adjacency_matrix=np.asarray(
            [
                [0, 1, 0, 0, 0],
                [1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=int,
        ),
        violating_edge_sets=[frozenset({(0, 1), (1, 2), (2, 3)})],
        active_node_mask=np.ones(5, dtype=bool),
        edge_probability_matrix=np.asarray(
            [
                [0.0, 0.9, 0.4, 0.8, 0.99],
                [0.9, 0.0, 0.9, 0.3, 0.99],
                [0.4, 0.9, 0.0, 0.9, 0.99],
                [0.8, 0.3, 0.9, 0.0, 0.99],
                [0.99, 0.99, 0.99, 0.99, 0.0],
            ],
            dtype=float,
        ),
        edge_label_classes=["low", "high"],
        edge_label_probabilities=np.asarray(
            [
                [[0.0, 0.0], [0.5, 0.5], [0.9, 0.1], [0.1, 0.9], [0.0, 1.0]],
                [[0.5, 0.5], [0.0, 0.0], [0.5, 0.5], [0.8, 0.2], [0.0, 1.0]],
                [[0.9, 0.1], [0.5, 0.5], [0.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
                [[0.1, 0.9], [0.8, 0.2], [0.5, 0.5], [0.0, 0.0], [0.0, 1.0]],
                [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
            ],
            dtype=float,
        ),
        predicted_edge_label_matrix=None,
        budget=2,
    )

    assert [(proposal.edge, proposal.label) for proposal in proposals] == [
        ((0, 3), "high"),
        ((0, 2), "low"),
    ]
    assert all(4 not in proposal.edge for proposal in proposals)


def test_constructive_edge_proposals_use_predicted_label_without_probabilities():
    proposals = enumerate_localized_edge_addition_proposals(
        adjacency_matrix=np.asarray(
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ],
            dtype=int,
        ),
        violating_edge_sets=[frozenset({(0, 1), (1, 2)})],
        active_node_mask=np.ones(3, dtype=bool),
        edge_probability_matrix=np.asarray(
            [
                [0.0, 0.9, 0.4],
                [0.9, 0.0, 0.9],
                [0.4, 0.9, 0.0],
            ],
            dtype=float,
        ),
        edge_label_classes=["unused-a", "unused-b"],
        edge_label_probabilities=None,
        predicted_edge_label_matrix=np.asarray(
            [
                [None, "path", "closure"],
                ["path", None, "path"],
                ["closure", "path", None],
            ],
            dtype=object,
        ),
        budget=32,
    )

    assert [(proposal.edge, proposal.label) for proposal in proposals] == [
        ((0, 2), "closure"),
    ]


def test_constructive_oracle_adds_only_the_improving_label():
    estimator = _ConstructiveEdgeEstimator([(0, 3, "good")])
    generator = ConditionalNodeFieldGraphGenerator(
        feasibility_estimator=estimator,
        verbose=False,
        max_oracle_iterations=3,
    )
    generator.edge_label_classes_ = np.asarray(["bad", "good"], dtype=object)
    generator.edge_label_to_index_ = {"bad": 0, "good": 1}
    conditioning = GraphConditioningBatch(
        graph_embeddings=np.zeros((1, 1), dtype=float),
        node_counts=np.asarray([4], dtype=int),
        edge_counts=np.asarray([3], dtype=int),
    )

    decoded = generator._decode_generated_nodes(
        _constructive_oracle_generated_batch(),
        graph_conditioning=conditioning,
    )

    assert len(decoded) == 1
    assert decoded[0].edges[0, 3]["label"] == "good"
    assert estimator.violation_batch_sizes[0] == 1
    assert 1 < estimator.violation_batch_sizes[1] <= 32


def test_constructive_oracle_can_apply_multiple_partial_improvements():
    estimator = _ConstructiveEdgeEstimator(
        [
            (0, 2, "good"),
            (1, 3, "good"),
        ]
    )
    generator = ConditionalNodeFieldGraphGenerator(
        feasibility_estimator=estimator,
        verbose=False,
        max_oracle_iterations=4,
    )
    generator.edge_label_classes_ = np.asarray(["bad", "good"], dtype=object)
    generator.edge_label_to_index_ = {"bad": 0, "good": 1}
    conditioning = GraphConditioningBatch(
        graph_embeddings=np.zeros((1, 1), dtype=float),
        node_counts=np.asarray([4], dtype=int),
        edge_counts=np.asarray([3], dtype=int),
    )

    decoded = generator._decode_generated_nodes(
        _constructive_oracle_generated_batch(),
        graph_conditioning=conditioning,
    )

    assert decoded[0].edges[0, 2]["label"] == "good"
    assert decoded[0].edges[1, 3]["label"] == "good"
    assert decoded[0].number_of_edges() == 5


def test_constructive_oracle_rejects_non_improving_additions():
    class _NeverImproves(_ConstructiveEdgeEstimator):
        def number_of_violations(self, graphs):
            self.violation_batch_sizes.append(len(graphs))
            return [1 for _ in graphs]

    estimator = _NeverImproves([(0, 3, "good")])
    generator = ConditionalNodeFieldGraphGenerator(
        feasibility_estimator=estimator,
        verbose=False,
        max_oracle_iterations=1,
    )
    generator.edge_label_classes_ = np.asarray(["bad", "good"], dtype=object)
    generator.edge_label_to_index_ = {"bad": 0, "good": 1}

    decoded = generator._decode_generated_nodes(_constructive_oracle_generated_batch())

    assert not decoded[0].has_edge(0, 3)


def test_constructive_oracle_budget_zero_disables_additions():
    estimator = _ConstructiveEdgeEstimator([(0, 3, "good")])
    generator = ConditionalNodeFieldGraphGenerator(
        feasibility_estimator=estimator,
        oracle_add_edge_repair_budget=0,
        verbose=False,
        max_oracle_iterations=1,
    )
    generator.edge_label_classes_ = np.asarray(["bad", "good"], dtype=object)
    generator.edge_label_to_index_ = {"bad": 0, "good": 1}

    decoded = generator._decode_generated_nodes(_constructive_oracle_generated_batch())

    assert not decoded[0].has_edge(0, 3)
    assert estimator.violation_batch_sizes == []


def test_constructive_oracle_respects_exact_conditioned_edge_count():
    estimator = _ConstructiveEdgeEstimator([(0, 3, "good")])
    decoder = ConditionalNodeFieldGraphDecoder(
        verbose=False,
        edge_count_slack_penalty=None,
    )
    generator = ConditionalNodeFieldGraphGenerator(
        graph_decoder=decoder,
        feasibility_estimator=estimator,
        verbose=False,
        max_oracle_iterations=1,
    )
    generator.edge_label_classes_ = np.asarray(["bad", "good"], dtype=object)
    generator.edge_label_to_index_ = {"bad": 0, "good": 1}
    conditioning = GraphConditioningBatch(
        graph_embeddings=np.zeros((1, 1), dtype=float),
        node_counts=np.asarray([4], dtype=int),
        edge_counts=np.asarray([3], dtype=int),
    )

    decoded = generator._decode_generated_nodes(
        _constructive_oracle_generated_batch(),
        graph_conditioning=conditioning,
    )

    assert decoded[0].number_of_edges() == 3
    assert not decoded[0].has_edge(0, 3)


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
    assert batch.horizon_probability_matrices is None
    assert batch.horizon is None


def test_predict_returns_horizon_predictions_when_auxiliary_head_outputs_exist():
    generated = np.asarray([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]], dtype=float)
    horizon_probs = torch.tensor(
        [[[0.0, 0.9, 0.7], [0.9, 0.0, 0.8], [0.7, 0.8, 0.0]]],
        dtype=torch.float32,
    )
    cached_outputs = {
        "_last_node_presence_mask": torch.tensor([[True, True, True]], dtype=torch.bool),
        "_last_node_existence_probabilities": torch.tensor([[0.9, 0.8, 0.7]], dtype=torch.float32),
        "_last_deg_classes": torch.tensor([[1, 2, 1]], dtype=torch.int64),
        "_last_node_label_classes": None,
        "_last_node_label_logits": None,
        "_last_node_label_probabilities": None,
        "_last_edge_probability_matrices": None,
        "_last_edge_existence_probabilities": None,
        "_last_horizon_probability_matrices": horizon_probs,
        "_last_edge_label_matrices": None,
        "_last_edge_label_logits": None,
        "_last_edge_label_probabilities": None,
    }
    model = _PredictiveStubModel(generated=generated, cached_outputs=cached_outputs)
    generator = _make_stubbed_node_field_generator(model)
    generator.locality_horizon_ = 3

    batch = generator.predict(_sample_graph_conditioning(batch_size=1))

    assert batch.horizon == 3
    assert generator.last_predicted_horizon_probability_matrices_ is batch.horizon_probability_matrices
    np.testing.assert_allclose(batch.horizon_probability_matrices[0], horizon_probs[0].numpy())


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
    assert batch.horizon_probability_matrices is None
    assert batch.horizon is None


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


def test_decoder_decode_node_labels_validates_batch_alignment():
    decoder = ConditionalNodeFieldGraphDecoder(verbose=False)

    with pytest.raises(ValueError, match="predicted_node_labels_list must align with generated_nodes"):
        decoder.decode_node_labels(
            GeneratedNodeBatch(
                node_presence_mask=np.asarray([[True, True], [True, False]], dtype=bool),
            ),
            predicted_node_labels_list=[np.asarray(["C", "O"], dtype=object)],
        )


def test_decoder_decode_node_labels_validates_slot_count():
    decoder = ConditionalNodeFieldGraphDecoder(verbose=False)

    with pytest.raises(ValueError, match="received 1 labels for 2 slots"):
        decoder.decode_node_labels(
            GeneratedNodeBatch(
                node_presence_mask=np.asarray([[True, True]], dtype=bool),
            ),
            predicted_node_labels_list=[np.asarray(["C"], dtype=object)],
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


def test_build_single_generated_node_batch_preserves_horizon_predictions():
    generated_nodes = GeneratedNodeBatch(
        node_presence_mask=np.asarray([[True, True], [True, False]], dtype=bool),
        horizon_probability_matrices=[
            np.asarray([[0.0, 0.8], [0.8, 0.0]], dtype=float),
            np.asarray([[0.0, 0.2], [0.2, 0.0]], dtype=float),
        ],
        horizon=3,
    )

    single = build_single_generated_node_batch(generated_nodes, 1)

    assert single.horizon == 3
    assert len(single.horizon_probability_matrices) == 1
    np.testing.assert_array_equal(
        single.horizon_probability_matrices[0],
        np.asarray([[0.0, 0.2], [0.2, 0.0]], dtype=float),
    )


def test_decoder_resolve_node_presence_mask_uses_top_existence_scores_for_desired_count():
    decoder = ConditionalNodeFieldGraphDecoder(verbose=False)

    resolved = decoder.resolve_node_presence_mask(
        np.asarray([False, False, False, False], dtype=bool),
        desired_node_count=2,
        node_existence_scores=np.asarray([0.1, 0.9, 0.3, 0.8], dtype=float),
    )

    np.testing.assert_array_equal(resolved, np.asarray([False, True, False, True], dtype=bool))


def test_decoder_degree_targets_match_desired_edge_budget():
    decoder = ConditionalNodeFieldGraphDecoder(verbose=False, enforce_connectivity=False)

    target_degrees = decoder.get_degree_targets(
        np.asarray([3.9, 0.2, 0.2, 0.2], dtype=float),
        np.asarray([True, True, True, True], dtype=bool),
        desired_edge_count=1,
    )

    assert sum(target_degrees) == 2
    assert max(target_degrees) <= 3


def test_decode_adjacency_matrix_enforces_desired_edge_count():
    decoder = ConditionalNodeFieldGraphDecoder(verbose=False, enforce_connectivity=False)
    generated_nodes = GeneratedNodeBatch(
        node_presence_mask=np.asarray([[True, True, True, True]], dtype=bool),
        node_existence_probabilities=np.asarray([[0.2, 0.9, 0.8, 0.7]], dtype=float),
        node_degree_predictions=np.asarray([[3, 3, 3, 3]], dtype=float),
        edge_probability_matrices=[
            np.asarray(
                [
                    [0.0, 0.95, 0.90, 0.85],
                    [0.95, 0.0, 0.80, 0.75],
                    [0.90, 0.80, 0.0, 0.70],
                    [0.85, 0.75, 0.70, 0.0],
                ],
                dtype=float,
            )
        ],
    )

    adj_mtx = decoder.decode_adjacency_matrix(
        generated_nodes,
        predicted_edge_probability_matrices=generated_nodes.edge_probability_matrices,
        desired_node_counts=[3],
        desired_edge_counts=[1],
    )[0]

    assert adj_mtx.shape == (4, 4)
    assert int(np.sum(adj_mtx) // 2) == 1


def test_decode_adjacency_matrix_excludes_inactive_padded_slots_from_constraints():
    decoder = ConditionalNodeFieldGraphDecoder(
        verbose=False,
        enforce_connectivity=True,
        degree_slack_penalty=1.0,
        edge_count_slack_penalty=None,
        n_jobs=1,
        parallel_decode_timeout_seconds=None,
    )
    generated_nodes = GeneratedNodeBatch(
        node_presence_mask=np.asarray([[True, True, False, False]], dtype=bool),
        node_degree_predictions=np.asarray([[1.0, 1.0, 0.0, 0.0]], dtype=float),
        edge_probability_matrices=[
            np.asarray(
                [
                    [0.0, 0.99, 0.0, 0.0],
                    [0.99, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype=float,
            )
        ],
    )

    adjacency = decoder.decode_adjacency_matrix(
        generated_nodes,
        predicted_edge_probability_matrices=generated_nodes.edge_probability_matrices,
        desired_edge_counts=[1],
    )[0]

    assert int(adjacency.sum() // 2) == 1
    assert adjacency[0, 1] == 1
    assert not adjacency[2:, :].any()
    assert not adjacency[:, 2:].any()
    assert decoder.last_adjacency_solve_report_.active_node_count == 2


@pytest.mark.parametrize(
    ("presence_mask", "expected_edges", "expected_active_count"),
    [
        ([False, False, False], [], 0),
        ([False, True, False], [], 1),
        ([True, False, True], [(0, 2)], 2),
    ],
)
def test_decode_adjacency_matrix_handles_sparse_active_slot_layouts(
    presence_mask,
    expected_edges,
    expected_active_count,
):
    decoder = ConditionalNodeFieldGraphDecoder(
        verbose=False,
        enforce_connectivity=True,
        n_jobs=1,
        parallel_decode_timeout_seconds=None,
    )
    generated_nodes = GeneratedNodeBatch(
        node_presence_mask=np.asarray([presence_mask], dtype=bool),
        node_degree_predictions=np.asarray([[1.0, 0.0, 1.0]], dtype=float),
        edge_probability_matrices=[
            np.asarray(
                [
                    [0.0, 0.1, 0.9],
                    [0.1, 0.0, 0.1],
                    [0.9, 0.1, 0.0],
                ],
                dtype=float,
            )
        ],
    )

    adjacency = decoder.decode_adjacency_matrix(
        generated_nodes,
        predicted_edge_probability_matrices=generated_nodes.edge_probability_matrices,
        desired_edge_counts=[0],
    )[0]

    assert sorted(nx.from_numpy_array(adjacency).edges()) == expected_edges
    assert decoder.last_adjacency_solve_report_.active_node_count == expected_active_count


def test_connected_edge_count_is_resolved_against_active_node_count():
    decoder = ConditionalNodeFieldGraphDecoder(
        verbose=False,
        enforce_connectivity=True,
        edge_count_slack_penalty=None,
        n_jobs=1,
        parallel_decode_timeout_seconds=None,
    )
    generated_nodes = GeneratedNodeBatch(
        node_presence_mask=np.asarray([[True, False, True, False, True]], dtype=bool),
        node_degree_predictions=np.asarray([[2.0, 0.0, 2.0, 0.0, 2.0]], dtype=float),
        edge_probability_matrices=[np.ones((5, 5), dtype=float) - np.eye(5)],
    )

    adjacency = decoder.decode_adjacency_matrix(
        generated_nodes,
        predicted_edge_probability_matrices=generated_nodes.edge_probability_matrices,
        desired_edge_counts=[0],
    )[0]

    assert int(adjacency.sum() // 2) == 2
    assert nx.is_connected(nx.from_numpy_array(adjacency).subgraph([0, 2, 4]))
    assert not adjacency[[1, 3], :].any()


def test_soft_edge_count_allows_nearby_solution_when_exact_count_is_cut_off():
    prob_matrix = np.asarray(
        [
            [0.0, 0.95, 0.90],
            [0.95, 0.0, 0.85],
            [0.90, 0.85, 0.0],
        ],
        dtype=float,
    )
    forbidden_two_edge_sets = [
        [(0, 1), (0, 2)],
        [(0, 1), (1, 2)],
        [(0, 2), (1, 2)],
    ]
    hard_decoder = ConditionalNodeFieldGraphDecoder(
        verbose=False,
        enforce_connectivity=False,
        degree_slack_penalty=1.0,
        edge_count_slack_penalty=None,
    )
    with pytest.raises(RuntimeError, match="Adjacency ILP did not produce a feasible solution"):
        hard_decoder.optimize_adjacency_matrix(
            prob_matrix,
            target_degrees=[1, 2, 1],
            target_edge_count=2,
            forbidden_edge_sets=forbidden_two_edge_sets,
        )

    soft_decoder = ConditionalNodeFieldGraphDecoder(
        verbose=False,
        enforce_connectivity=False,
        degree_slack_penalty=1.0,
        edge_count_slack_penalty=2.0,
    )
    adjacency = soft_decoder.optimize_adjacency_matrix(
        prob_matrix,
        target_degrees=[1, 2, 1],
        target_edge_count=2,
        forbidden_edge_sets=forbidden_two_edge_sets,
    )

    assert int(np.sum(adjacency) // 2) == 1


def test_default_edge_count_slack_allows_deviation_greater_than_one():
    decoder = ConditionalNodeFieldGraphDecoder(
        verbose=False,
        enforce_connectivity=False,
        degree_slack_penalty=1.0,
    )
    prob_matrix = np.ones((4, 4), dtype=float) - np.eye(4, dtype=float)
    forbidden_two_edge_sets = [
        [first_edge, second_edge]
        for edge_idx, first_edge in enumerate(
            [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        )
        for second_edge in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)][
            edge_idx + 1 :
        ]
    ]

    adjacency = decoder.optimize_adjacency_matrix(
        prob_matrix,
        target_degrees=[1, 2, 2, 1],
        target_edge_count=3,
        forbidden_edge_sets=forbidden_two_edge_sets,
    )

    assert int(np.sum(adjacency) // 2) == 1


@pytest.mark.parametrize("penalty", [-0.1, 0.0])
def test_decoder_rejects_non_positive_edge_count_slack_penalty(penalty):
    with pytest.raises(ValueError, match="edge_count_slack_penalty"):
        ConditionalNodeFieldGraphDecoder(edge_count_slack_penalty=penalty)


def test_oracle_relaxed_adjacency_forwards_target_edge_count(monkeypatch):
    captured = {}

    def _fake_optimize(owner, prob_matrix, target_degrees, **kwargs):
        del owner, target_degrees
        captured.update(kwargs)
        return np.zeros_like(prob_matrix, dtype=int)

    monkeypatch.setattr(
        oracle_decode_module,
        "optimize_oracle_adjacency_matrix",
        _fake_optimize,
    )
    owner = types.SimpleNamespace(
        max_oracle_iterations=2,
        oracle_edge_memory_penalty=0.0,
        verbose=False,
    )

    decoder_module.solve_oracle_relaxed_adjacency(
        owner,
        masked_prob_matrix=np.zeros((3, 3), dtype=float),
        target_degrees=[1, 1, 0],
        accumulated_cuts=[],
        start_iteration_idx=0,
        target_edge_count=1,
    )

    assert captured["target_edge_count"] == 1


def test_decode_adjacency_matrix_direct_selects_top_edges_by_desired_count():
    decoder = ConditionalNodeFieldGraphDecoder(verbose=False, enforce_connectivity=False)
    generated_nodes = GeneratedNodeBatch(
        node_presence_mask=np.asarray([[True, True, True, True]], dtype=bool),
        edge_probability_matrices=[
            np.asarray(
                [
                    [0.0, 0.70, 0.95, 0.10],
                    [0.70, 0.0, 0.30, 0.80],
                    [0.95, 0.30, 0.0, 0.20],
                    [0.10, 0.80, 0.20, 0.0],
                ],
                dtype=float,
            )
        ],
    )

    adj_mtx = decoder.decode_adjacency_matrix_direct(
        generated_nodes,
        predicted_edge_probability_matrices=generated_nodes.edge_probability_matrices,
        desired_edge_counts=[2],
    )[0]

    assert sorted(nx.from_numpy_array(adj_mtx).edges()) == [(0, 2), (1, 3)]


def test_decode_adjacency_matrix_direct_uses_degree_predictions_for_raw_edge_budget():
    decoder = ConditionalNodeFieldGraphDecoder(verbose=False, enforce_connectivity=False)
    generated_nodes = GeneratedNodeBatch(
        node_presence_mask=np.asarray([[True, True, True, True, True]], dtype=bool),
        node_degree_predictions=np.asarray([[1, 1, 1, 1, 0]], dtype=float),
        edge_probability_matrices=[
            np.asarray(
                [
                    [0.0, 0.60, 0.10, 0.10, 0.99],
                    [0.60, 0.0, 0.10, 0.10, 0.98],
                    [0.10, 0.10, 0.0, 0.59, 0.97],
                    [0.10, 0.10, 0.59, 0.0, 0.96],
                    [0.99, 0.98, 0.97, 0.96, 0.0],
                ],
                dtype=float,
            )
        ],
    )

    adj_mtx = decoder.decode_adjacency_matrix_direct(
        generated_nodes,
        predicted_edge_probability_matrices=generated_nodes.edge_probability_matrices,
        desired_edge_counts=[2],
    )[0]

    assert sorted(nx.from_numpy_array(adj_mtx).edges()) == [(0, 1), (2, 3)]


def test_decode_adjacency_matrix_direct_trims_without_violating_degree_targets():
    decoder = ConditionalNodeFieldGraphDecoder(verbose=False, enforce_connectivity=False)
    generated_nodes = GeneratedNodeBatch(
        node_presence_mask=np.asarray([[True, True, True, True]], dtype=bool),
        node_degree_predictions=np.asarray([[1, 1, 1, 1]], dtype=float),
        edge_probability_matrices=[
            np.asarray(
                [
                    [0.0, 0.90, 0.80, 0.05],
                    [0.90, 0.0, 0.10, 0.05],
                    [0.80, 0.10, 0.0, 0.20],
                    [0.05, 0.05, 0.20, 0.0],
                ],
                dtype=float,
            )
        ],
    )

    adj_mtx = decoder.decode_adjacency_matrix_direct(
        generated_nodes,
        predicted_edge_probability_matrices=generated_nodes.edge_probability_matrices,
        desired_edge_counts=[2],
    )[0]

    assert sorted(nx.from_numpy_array(adj_mtx).edges()) == [(0, 1), (2, 3)]


def test_select_direct_edges_degree_aware_fills_under_budget_from_global_candidates():
    selected_edges = ConditionalNodeFieldGraphDecoder._select_direct_edges_degree_aware(
        [
            (0.90, 0, 1),
            (0.80, 0, 2),
            (0.70, 1, 2),
        ],
        np.asarray([0, 1, 2], dtype=int),
        [1, 0, 0],
        desired_edge_count=2,
    )

    assert [(i, j) for _, i, j in selected_edges] == [(0, 1), (0, 2)]


def test_decode_adjacency_matrix_direct_uses_threshold_without_desired_count():
    decoder = ConditionalNodeFieldGraphDecoder(verbose=False, enforce_connectivity=False)
    generated_nodes = GeneratedNodeBatch(
        node_presence_mask=np.asarray([[True, True, True, True]], dtype=bool),
        edge_probability_matrices=[
            np.asarray(
                [
                    [0.0, 0.90, 0.20, 0.10],
                    [0.90, 0.0, 0.30, 0.20],
                    [0.20, 0.30, 0.0, 0.85],
                    [0.10, 0.20, 0.85, 0.0],
                ],
                dtype=float,
            )
        ],
    )

    adj_mtx = decoder.decode_adjacency_matrix_direct(
        generated_nodes,
        predicted_edge_probability_matrices=generated_nodes.edge_probability_matrices,
        edge_probability_threshold=0.8,
    )[0]

    graph = nx.from_numpy_array(adj_mtx)
    assert sorted(graph.edges()) == [(0, 1), (2, 3)]
    assert not nx.is_connected(graph)


def test_decode_adjacency_matrix_direct_respects_desired_node_count():
    decoder = ConditionalNodeFieldGraphDecoder(verbose=False, enforce_connectivity=False)
    generated_nodes = GeneratedNodeBatch(
        node_presence_mask=np.asarray([[False, False, False, False]], dtype=bool),
        node_existence_probabilities=np.asarray([[0.10, 0.95, 0.20, 0.90]], dtype=float),
        edge_probability_matrices=[
            np.asarray(
                [
                    [0.0, 0.80, 0.70, 0.60],
                    [0.80, 0.0, 0.40, 0.99],
                    [0.70, 0.40, 0.0, 0.50],
                    [0.60, 0.99, 0.50, 0.0],
                ],
                dtype=float,
            )
        ],
    )

    adj_mtx = decoder.decode_adjacency_matrix_direct(
        generated_nodes,
        predicted_edge_probability_matrices=generated_nodes.edge_probability_matrices,
        desired_node_counts=[2],
        desired_edge_counts=[1],
    )[0]

    assert sorted(nx.from_numpy_array(adj_mtx).edges()) == [(1, 3)]


def test_degree_aware_direct_decode_honors_exact_edge_budget():
    decoder = ConditionalNodeFieldGraphDecoder(verbose=False, enforce_connectivity=False)
    candidates = [
        (0.99, 0, 1),
        (0.98, 0, 2),
        (0.97, 0, 3),
        (0.96, 0, 4),
        (0.95, 1, 2),
        (0.94, 3, 4),
    ]

    selected = decoder._select_direct_edges_degree_aware(
        candidates,
        np.arange(5),
        [0, 1, 1, 1, 1],
        desired_edge_count=2,
    )

    assert len(selected) == 2
    assert selected == decoder._select_direct_edges_degree_aware(
        candidates,
        np.arange(5),
        [0, 1, 1, 1, 1],
        desired_edge_count=2,
    )


@pytest.mark.parametrize("edge_budget", range(7))
def test_degree_aware_direct_decode_honors_all_feasible_small_budgets(edge_budget):
    decoder = ConditionalNodeFieldGraphDecoder(verbose=False, enforce_connectivity=False)
    candidates = [
        (1.0 - 0.05 * idx, i, j)
        for idx, (i, j) in enumerate(
            (pair for i in range(4) for pair in ((i, j) for j in range(i + 1, 4)))
        )
    ]

    selected = decoder._select_direct_edges_degree_aware(
        candidates,
        np.arange(4),
        [1, 2, 2, 1],
        desired_edge_count=edge_budget,
    )

    assert len(selected) == min(edge_budget, len(candidates))


def test_horizon_positive_constraint_can_add_short_path():
    decoder = ConditionalNodeFieldGraphDecoder(
        verbose=False,
        enforce_connectivity=False,
        degree_slack_penalty=0.1,
        horizon_constraint_weight=100.0,
        horizon_positive_threshold=0.8,
        horizon_pair_budget=2,
        horizon_paths_per_pair=4,
        horizon_max_iterations=0,
    )
    horizon_probs = np.full((3, 3), 0.5, dtype=float)
    np.fill_diagonal(horizon_probs, 0.0)
    horizon_probs[0, 2] = horizon_probs[2, 0] = 0.99

    adj = decoder.optimize_adjacency_matrix(
        prob_matrix=np.asarray(
            [
                [0.0, 0.95, 0.01],
                [0.95, 0.0, 0.95],
                [0.01, 0.95, 0.0],
            ],
            dtype=float,
        ),
        target_degrees=[1, 1, 0],
        connectivity=False,
        horizon_probability_matrix=horizon_probs,
        horizon=2,
        horizon_node_mask=np.asarray([True, True, True], dtype=bool),
    )

    graph = nx.from_numpy_array(adj)
    assert nx.shortest_path_length(graph, 0, 2) <= 2
    assert decoder.last_adjacency_solve_report_.horizon_termination_reason == "positive_only"
    assert decoder.last_adjacency_solve_report_.horizon_path_expansion_count > 0
    assert decoder.last_adjacency_solve_report_.horizon_path_search_truncated is False


def test_horizon_negative_constraint_can_break_short_path():
    decoder = ConditionalNodeFieldGraphDecoder(
        verbose=False,
        enforce_connectivity=False,
        degree_slack_penalty=0.1,
        horizon_constraint_weight=100.0,
        horizon_negative_threshold=0.2,
        horizon_pair_budget=2,
        horizon_paths_per_pair=4,
        horizon_max_iterations=1,
    )
    horizon_probs = np.ones((3, 3), dtype=float)
    horizon_probs[0, 2] = horizon_probs[2, 0] = 0.01

    adj = decoder.optimize_adjacency_matrix(
        prob_matrix=np.asarray(
            [
                [0.0, 0.99, 0.01],
                [0.99, 0.0, 0.99],
                [0.01, 0.99, 0.0],
            ],
            dtype=float,
        ),
        target_degrees=[1, 2, 1],
        connectivity=False,
        horizon_probability_matrix=horizon_probs,
        horizon=2,
        horizon_node_mask=np.asarray([True, True, True], dtype=bool),
    )

    graph = nx.from_numpy_array(adj)
    assert not nx.has_path(graph, 0, 2) or nx.shortest_path_length(graph, 0, 2) > 2


def test_horizon_negative_separation_runs_until_no_new_cuts(monkeypatch):
    calls = []

    def fake_find(adjacency, negative_pairs, *, horizon):
        del adjacency, negative_pairs, horizon
        calls.append(len(calls))
        if len(calls) == 1:
            return [([(0, 1)], 0.0)]
        if len(calls) == 2:
            return [([(1, 2)], 0.0)]
        return []

    monkeypatch.setattr(structural_decoder, "find_negative_horizon_cuts", fake_find)
    decoder = ConditionalNodeFieldGraphDecoder(
        verbose=False,
        enforce_connectivity=False,
        horizon_negative_threshold=0.2,
        horizon_pair_budget=2,
        horizon_max_iterations=5,
        n_jobs=1,
        parallel_decode_timeout_seconds=None,
    )
    horizon_probs = np.ones((3, 3), dtype=float)
    horizon_probs[0, 2] = horizon_probs[2, 0] = 0.01

    decoder.optimize_adjacency_matrix(
        np.ones((3, 3), dtype=float) - np.eye(3),
        [1, 2, 1],
        connectivity=False,
        horizon_probability_matrix=horizon_probs,
        horizon=2,
    )

    assert len(calls) == 3
    assert decoder.last_adjacency_solve_report_.solve_count == 3
    assert decoder.last_adjacency_solve_report_.horizon_iterations == 2


def test_horizon_report_exposes_unresolved_soft_violation(monkeypatch):
    monkeypatch.setattr(
        structural_decoder,
        "find_negative_horizon_cuts",
        lambda adjacency, negative_pairs, *, horizon: [([(0, 1), (1, 2)], 1.0)],
    )
    decoder = ConditionalNodeFieldGraphDecoder(
        verbose=False,
        enforce_connectivity=False,
        degree_slack_penalty=0.1,
        horizon_constraint_weight=0.0,
        horizon_negative_threshold=0.2,
        horizon_pair_budget=2,
        horizon_max_iterations=2,
    )
    decoder.horizon_constraint_weight = 1.0
    horizon_probs = np.ones((3, 3), dtype=float)
    horizon_probs[0, 2] = horizon_probs[2, 0] = 0.01

    decoder.optimize_adjacency_matrix(
        np.ones((3, 3), dtype=float) - np.eye(3),
        [1, 2, 1],
        connectivity=False,
        horizon_probability_matrix=horizon_probs,
        horizon=2,
    )

    report = decoder.last_adjacency_solve_report_
    assert report.horizon_termination_reason == "no_new_cuts"
    assert report.unresolved_horizon_pair_count == 1
    assert report.objective_value is not None
    assert report.degree_slack_total >= 0.0
    assert report.edge_count_slack >= 0.0


def test_positive_horizon_paths_are_ranked_by_path_probability():
    probabilities = np.asarray(
        [
            [0.0, 0.9, 0.6],
            [0.9, 0.0, 0.9],
            [0.6, 0.9, 0.0],
        ]
    )
    logits = np.zeros_like(probabilities)
    for i in range(3):
        for j in range(i + 1, 3):
            logits[i, j] = logits[j, i] = structural_decoder.edge_logit(
                probabilities[i, j]
            )

    paths = structural_decoder.enumerate_horizon_paths(
        0,
        2,
        horizon=2,
        edge_logit_matrix=logits,
        paths_per_pair=2,
    )

    assert [path for path, _score in paths] == [[0, 1, 2], [0, 2]]


def test_positive_horizon_path_search_reports_expansion_truncation():
    logits = np.zeros((5, 5), dtype=float)
    paths, expansion_count, truncated = structural_decoder.enumerate_horizon_paths(
        0,
        4,
        horizon=4,
        edge_logit_matrix=logits,
        paths_per_pair=3,
        expansion_budget=1,
        return_stats=True,
    )

    assert paths == []
    assert expansion_count == 1
    assert truncated is True


def test_adjacency_report_records_horizon_path_search_truncation():
    decoder = ConditionalNodeFieldGraphDecoder(
        verbose=False,
        enforce_connectivity=False,
        horizon_positive_threshold=0.8,
        horizon_negative_threshold=0.1,
        horizon_pair_budget=1,
        horizon_paths_per_pair=2,
        horizon_path_expansion_budget=1,
        horizon_max_iterations=0,
    )
    horizon_probs = np.full((4, 4), 0.5, dtype=float)
    np.fill_diagonal(horizon_probs, 0.0)
    horizon_probs[0, 3] = horizon_probs[3, 0] = 0.99

    decoder.optimize_adjacency_matrix(
        np.full((4, 4), 0.5, dtype=float) - np.eye(4) * 0.5,
        [0, 0, 0, 0],
        connectivity=False,
        horizon_probability_matrix=horizon_probs,
        horizon=3,
    )

    report = decoder.last_adjacency_solve_report_
    assert report.horizon_path_search_truncated is True
    assert report.horizon_path_expansion_count == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"direct_edge_probability_threshold": -0.1}, "direct_edge_probability_threshold"),
        ({"degree_slack_penalty": 0.0}, "degree_slack_penalty"),
        ({"adjacency_time_limit_seconds": 0.0}, "adjacency_time_limit_seconds"),
        ({"parallel_decode_timeout_seconds": float("nan")}, "parallel_decode_timeout_seconds"),
        ({"horizon_constraint_weight": -1.0}, "horizon_constraint_weight"),
        ({"horizon_negative_threshold": 0.9, "horizon_positive_threshold": 0.8}, "threshold"),
        ({"horizon_pair_budget": -1}, "horizon_pair_budget"),
        ({"horizon_paths_per_pair": 0}, "horizon_paths_per_pair"),
        ({"horizon_max_iterations": -1}, "horizon_max_iterations"),
        ({"solver_threads": 0}, "solver_threads"),
    ],
)
def test_decoder_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ConditionalNodeFieldGraphDecoder(**kwargs)


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), -0.1, 1.1])
def test_optimize_adjacency_matrix_rejects_invalid_probabilities(invalid_value):
    decoder = ConditionalNodeFieldGraphDecoder(verbose=False)
    probabilities = np.asarray([[0.0, invalid_value], [invalid_value, 0.0]])
    with pytest.raises(ValueError, match="prob_matrix"):
        decoder.optimize_adjacency_matrix(probabilities, [1, 1])


@pytest.mark.parametrize("invalid_value", [-1.0, 2.0, float("nan")])
def test_optimize_adjacency_matrix_rejects_non_binary_integer_incumbents(
    monkeypatch,
    invalid_value,
):
    decoder = ConditionalNodeFieldGraphDecoder(verbose=False, enforce_connectivity=False)

    def _fake_solve(self, solver):
        del solver
        self.status = pulp.LpStatusOptimal
        self.sol_status = pulp.LpSolutionIntegerFeasible
        for variable in self.variables():
            if variable.name == "x_0_1":
                variable.varValue = invalid_value
            else:
                variable.varValue = 0.0
        return self.status

    monkeypatch.setattr(pulp.LpProblem, "solve", _fake_solve)
    with pytest.raises(RuntimeError):
        decoder.optimize_adjacency_matrix(
            np.asarray([[0.0, 0.9], [0.9, 0.0]]),
            [0, 0],
            connectivity=False,
        )


def test_optimize_adjacency_matrix_rejects_missing_slack_values(monkeypatch):
    decoder = ConditionalNodeFieldGraphDecoder(
        verbose=False,
        enforce_connectivity=False,
    )

    def _fake_solve(self, solver):
        del solver
        self.status = pulp.LpStatusOptimal
        self.sol_status = pulp.LpSolutionIntegerFeasible
        for variable in self.variables():
            if variable.name == "x_0_1":
                variable.varValue = 0.0
        return self.status

    monkeypatch.setattr(pulp.LpProblem, "solve", _fake_solve)
    with pytest.raises(RuntimeError, match="slack"):
        decoder.optimize_adjacency_matrix(
            np.asarray([[0.0, 0.9], [0.9, 0.0]]),
            [0, 0],
            connectivity=False,
        )


def test_adjacency_report_omits_non_finite_objective(monkeypatch):
    decoder = ConditionalNodeFieldGraphDecoder(
        verbose=False,
        enforce_connectivity=False,
    )
    original_value = pulp.value

    def _fake_solve(self, solver):
        del solver
        self.status = pulp.LpStatusOptimal
        self.sol_status = pulp.LpSolutionOptimal
        for variable in self.variables():
            variable.varValue = 0.0
        return self.status

    def _fake_value(value):
        if isinstance(value, pulp.LpAffineExpression):
            return float("nan")
        return original_value(value)

    monkeypatch.setattr(pulp.LpProblem, "solve", _fake_solve)
    monkeypatch.setattr(pulp, "value", _fake_value)

    decoder.optimize_adjacency_matrix(
        np.asarray([[0.0, 0.9], [0.9, 0.0]]),
        [0, 0],
        connectivity=False,
    )

    assert decoder.last_adjacency_solve_report_.objective_value is None


@pytest.mark.parametrize("shape", [(1,), (3,), (2, 2, 1)])
def test_decode_adjacency_matrix_rejects_malformed_flattened_probabilities(shape):
    decoder = ConditionalNodeFieldGraphDecoder(
        verbose=False,
        parallel_decode_timeout_seconds=None,
    )
    generated_nodes = GeneratedNodeBatch(
        node_presence_mask=np.asarray([[True, True]], dtype=bool),
        node_degree_predictions=np.asarray([[1.0, 1.0]], dtype=float),
    )
    values = np.full(shape, 0.5, dtype=float)
    with pytest.raises(ValueError, match="probability"):
        decoder.decode_adjacency_matrix(
            generated_nodes,
            predicted_edge_probability_matrices=[values],
        )


def test_parent_timeout_reserves_solver_shutdown_time():
    solver_budget = decoder_module._solver_budget_with_parent_reserve(60.0, 2.0)
    assert 0.0 < solver_budget < 2.0


def test_timed_parallel_map_terminates_running_workers():
    before = {process.pid for process in mp.active_children()}
    with pytest.raises(TimeoutError):
        parallel_utils._parallel_map(
            _sleep_and_return,
            [(2.0, 1), (2.0, 2)],
            max_workers=2,
            timeout_seconds=0.1,
            timeout_fallback_label="test work",
            fallback_on_timeout=False,
        )
    time.sleep(0.1)
    leaked = [
        process
        for process in mp.active_children()
        if process.pid not in before and process.is_alive()
    ]
    assert leaked == []


def test_timed_serial_map_uses_one_batch_deadline():
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        parallel_utils._parallel_map(
            _sleep_and_return,
            [(0.08, 1), (0.08, 2)],
            max_workers=1,
            timeout_seconds=0.12,
            timeout_fallback_label="serial test work",
            fallback_on_timeout=False,
        )
    assert time.monotonic() - started < 0.5


def test_decoder_direct_mode_attaches_labels_and_bypasses_optimizer(monkeypatch):
    decoder = ConditionalNodeFieldGraphDecoder(verbose=False, enforce_connectivity=False)
    generated_nodes = GeneratedNodeBatch(
        node_presence_mask=np.asarray([[True, True, True]], dtype=bool),
        node_labels=[np.asarray(["C", "O", "N"], dtype=object)],
        edge_probability_matrices=[
            np.asarray(
                [
                    [0.0, 0.20, 0.95],
                    [0.20, 0.0, 0.90],
                    [0.95, 0.90, 0.0],
                ],
                dtype=float,
            )
        ],
        edge_label_matrices=[
            np.asarray(
                [
                    [None, "low", "single"],
                    ["low", None, "double"],
                    ["single", "double", None],
                ],
                dtype=object,
            )
        ],
    )

    def _raise_if_ilp_used(*args, **kwargs):
        raise AssertionError("ILP optimizer should not be called in direct mode")

    monkeypatch.setattr(decoder, "optimize_adjacency_matrix", _raise_if_ilp_used)

    graph = decoder.decode(
        generated_nodes,
        predicted_node_labels_list=generated_nodes.node_labels,
        predicted_edge_probability_matrices=generated_nodes.edge_probability_matrices,
        predicted_edge_label_matrices=generated_nodes.edge_label_matrices,
        desired_edge_counts=[1],
        use_ilp_decoder=False,
    )[0]

    assert sorted(graph.nodes(data="label")) == [(0, "C"), (1, "O"), (2, "N")]
    assert sorted(graph.edges(data="label")) == [(0, 2, "single")]


def test_decoder_decode_edge_labels_validates_edge_label_count():
    decoder = ConditionalNodeFieldGraphDecoder(verbose=False, enforce_connectivity=False)

    with pytest.raises(ValueError, match="received 0 labels for 1 edges"):
        decoder.decode_edge_labels(
            GeneratedNodeBatch(
                node_presence_mask=np.asarray([[True, True]], dtype=bool),
            ),
            adj_mtx_list=[np.asarray([[0, 1], [1, 0]], dtype=int)],
            predicted_edge_labels_list=[np.asarray([], dtype=object)],
        )


def test_decoder_decode_validates_node_label_count_during_decode():
    decoder = ConditionalNodeFieldGraphDecoder(verbose=False, enforce_connectivity=False)
    generated_nodes = GeneratedNodeBatch(
        node_presence_mask=np.asarray([[True, True, True]], dtype=bool),
        node_existence_probabilities=np.asarray([[0.9, 0.8, 0.1]], dtype=float),
        node_degree_predictions=np.asarray([[1, 1, 0]], dtype=float),
        edge_probability_matrices=[
            np.asarray(
                [
                    [0.0, 0.95, 0.10],
                    [0.95, 0.0, 0.05],
                    [0.10, 0.05, 0.0],
                ],
                dtype=float,
            )
        ],
    )

    with pytest.raises(ValueError, match="received 4 labels for 3 slots"):
        decoder.decode(
            generated_nodes,
            predicted_node_labels_list=[np.asarray(["C", "O", "N", "F"], dtype=object)],
            predicted_edge_probability_matrices=generated_nodes.edge_probability_matrices,
            predicted_edge_label_matrices=[
                np.asarray(
                    [
                        [None, "-", None],
                        ["-", None, None],
                        [None, None, None],
                    ],
                    dtype=object,
                )
            ],
        )


def test_decode_forwards_diagnostic_graph_renderer_with_decoded_graph(monkeypatch):
    plot_calls = []

    def fake_plot_decoder_diagnostics(**kwargs):
        plot_calls.append(kwargs)

    monkeypatch.setattr(cngg_module, "_plot_decoder_diagnostics", fake_plot_decoder_diagnostics)

    def fake_renderer(graphs, titles=None):
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

    def fake_draw_molecule(graph, size):
        rendered["graph"] = graph
        rendered["size"] = size
        return np.ones((4, 4, 3), dtype=np.uint8) * 255

    monkeypatch.setattr(
        "abstractgraph_graphicalizer.chem.draw_molecule",
        fake_draw_molecule,
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
    assert rendered["size"] == (500, 350)
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


def test_plot_decoder_diagnostics_uses_integer_node_ticks(monkeypatch):
    calls = {"axes": None}

    class _Axis:
        def __init__(self):
            self.xticks = None
            self.xticklabels = None
            self.yticks = None
            self.yticklabels = None
            self.images = []

        def imshow(self, *args, **kwargs):
            if args:
                self.images.append(np.asarray(args[0]))
            return object()

        def set_title(self, *args, **kwargs):
            return None

        def set_xlabel(self, *args, **kwargs):
            return None

        def set_ylabel(self, *args, **kwargs):
            return None

        def set_xticks(self, values, *args, **kwargs):
            self.xticks = list(values)

        def set_xticklabels(self, values, *args, **kwargs):
            self.xticklabels = list(values)

        def set_yticks(self, values, *args, **kwargs):
            self.yticks = list(values)

        def set_yticklabels(self, values, *args, **kwargs):
            self.yticklabels = list(values)

        def plot(self, *args, **kwargs):
            return None

        def bar(self, *args, **kwargs):
            return None

        def grid(self, *args, **kwargs):
            return None

        def tick_params(self, *args, **kwargs):
            return None

        def legend(self, *args, **kwargs):
            return None

        def scatter(self, *args, **kwargs):
            return None

        def text(self, *args, **kwargs):
            return None

        def set_axis_off(self, *args, **kwargs):
            return None

    class _Figure:
        def suptitle(self, *args, **kwargs):
            return None

        def colorbar(self, *args, **kwargs):
            return None

    def fake_subplots(*args, **kwargs):
        axes = [_Axis() for _ in range(5)]
        calls["axes"] = axes
        return _Figure(), np.asarray(axes, dtype=object)

    monkeypatch.setattr(cngg_module.plt, "subplots", fake_subplots)
    monkeypatch.setattr(cngg_module.plt, "tight_layout", lambda *args, **kwargs: None)
    monkeypatch.setattr(cngg_module.plt, "show", lambda *args, **kwargs: None)
    monkeypatch.setattr(cngg_module.plt, "close", lambda *args, **kwargs: None)
    monkeypatch.setattr(cngg_module, "_try_render_molecular_graph_inline", lambda *args, **kwargs: True)

    cngg_module._plot_decoder_diagnostics(
        prob_matrix=np.asarray(
            [
                [0.0, 0.1, 0.2],
                [0.1, 0.0, 0.3],
                [0.2, 0.3, 0.0],
            ],
            dtype=float,
        ),
        adj_mtx=np.asarray(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        target_degrees=[1, 2, 1],
        title="Oracle demo",
        decoded_graph=nx.path_graph(3),
        existence_mask=[False, True, True],
        node_label_probabilities=np.asarray([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3]], dtype=float),
        node_label_names=["C", "N"],
        node_labels=["C", "N", "C"],
    )

    edge_axis = calls["axes"][0]
    adjacency_axis = calls["axes"][1]
    label_axis = calls["axes"][3]
    assert edge_axis.images[0].tolist() == [[0.0, 0.3], [0.3, 0.0]]
    assert edge_axis.xticks == [0, 1]
    assert edge_axis.xticklabels == [1, 2]
    assert edge_axis.yticks == [0, 1]
    assert edge_axis.yticklabels == [1, 2]
    assert adjacency_axis.xticks == [0, 1]
    assert adjacency_axis.yticks == [0, 1]
    assert adjacency_axis.yticklabels == [1, 2]
    assert label_axis.yticklabels == ["0", "1", "2"]


def test_parallel_map_falls_back_to_threads_on_broken_process_pool(monkeypatch):
    calls = {"thread_used": False}

    class _BrokenProcessExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, func, jobs):
            raise BrokenProcessPool("process pool terminated")

    class _ThreadExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            calls["thread_used"] = True
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, func, jobs):
            return [func(job) for job in jobs]

    monkeypatch.setattr(parallel_utils, "ProcessPoolExecutor", _BrokenProcessExecutor)
    monkeypatch.setattr(parallel_utils, "ThreadPoolExecutor", _ThreadExecutor)

    result = parallel_utils._parallel_map(lambda x: x + 1, [1, 2, 3], max_workers=2, verbose=False)

    assert result == [2, 3, 4]
    assert calls["thread_used"] is True


def test_parallel_map_falls_back_to_threads_on_pickle_type_error(monkeypatch):
    calls = {"thread_used": False}

    class _PickleFailingProcessExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, func, jobs):
            raise TypeError("cannot pickle local object")

    class _ThreadExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            calls["thread_used"] = True
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, func, jobs):
            return [func(job) for job in jobs]

    monkeypatch.setattr(parallel_utils, "ProcessPoolExecutor", _PickleFailingProcessExecutor)
    monkeypatch.setattr(parallel_utils, "ThreadPoolExecutor", _ThreadExecutor)

    result = parallel_utils._parallel_map(lambda x: x * 2, [1, 2], max_workers=2, verbose=False)

    assert result == [2, 4]
    assert calls["thread_used"] is True


def test_parallel_map_does_not_mask_non_parallel_worker_errors(monkeypatch):
    calls = {"thread_used": False}

    class _FailingProcessExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, func, jobs):
            raise ValueError("worker failed")

    class _ThreadExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            calls["thread_used"] = True
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, func, jobs):
            return [func(job) for job in jobs]

    monkeypatch.setattr(parallel_utils, "ProcessPoolExecutor", _FailingProcessExecutor)
    monkeypatch.setattr(parallel_utils, "ThreadPoolExecutor", _ThreadExecutor)

    with pytest.raises(ValueError, match="worker failed"):
        parallel_utils._parallel_map(lambda x: x, [1, 2], max_workers=2, verbose=False)

    assert calls["thread_used"] is False


def test_fill_unlabeled_active_edges_uses_edge_label_probabilities():
    generator = ConditionalNodeFieldGraphGenerator(verbose=False)
    generator.edge_label_classes_ = np.asarray(["single", "double", "triple"], dtype=object)

    repaired = generator._fill_unlabeled_active_edges(
        adj_mtx=np.asarray(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        edge_label_matrix=np.asarray(
            [
                [None, "single", None],
                ["single", None, None],
                [None, None, None],
            ],
            dtype=object,
        ),
        edge_label_probabilities=np.asarray(
            [
                [[0.0, 0.0, 0.0], [0.8, 0.1, 0.1], [0.0, 0.0, 0.0]],
                [[0.8, 0.1, 0.1], [0.0, 0.0, 0.0], [0.1, 0.7, 0.2]],
                [[0.0, 0.0, 0.0], [0.1, 0.7, 0.2], [0.0, 0.0, 0.0]],
            ],
            dtype=float,
        ),
    )

    assert repaired[0, 1] == "single"
    assert repaired[1, 2] == "double"
    assert repaired[2, 1] == "double"


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
    assert len(serial_decoder.last_adjacency_solve_reports_) == 2
    assert len(parallel_decoder.last_adjacency_solve_reports_) == 2
    assert all(report.active_node_count == 2 for report in parallel_decoder.last_adjacency_solve_reports_)


def test_decoder_save_and_load_round_trip_json_artifact(tmp_path):
    decoder = ConditionalNodeFieldGraphDecoder(
        verbose=False,
        direct_edge_probability_threshold=0.7,
        enforce_connectivity=False,
        degree_slack_penalty=123.0,
        edge_count_slack_penalty=4.0,
        warm_start_mst=False,
        n_jobs=3,
        use_horizon_ilp_constraints=False,
        horizon_constraint_weight=4.0,
        horizon_positive_threshold=0.75,
        horizon_negative_threshold=0.15,
        horizon_pair_budget=12,
        horizon_paths_per_pair=5,
        horizon_path_expansion_budget=123,
        horizon_max_iterations=2,
    )
    path = tmp_path / "decoder.json"

    decoder.save(str(path))

    restored = ConditionalNodeFieldGraphDecoder().load(str(path))

    assert restored.verbose is False
    assert restored.direct_edge_probability_threshold == pytest.approx(0.7)
    assert restored.enforce_connectivity is False
    assert restored.degree_slack_penalty == pytest.approx(123.0)
    assert restored.edge_count_slack_penalty == pytest.approx(4.0)
    assert restored.warm_start_mst is False
    assert restored.n_jobs == 3
    assert restored.use_horizon_ilp_constraints is False
    assert restored.horizon_constraint_weight == pytest.approx(4.0)
    assert restored.horizon_positive_threshold == pytest.approx(0.75)
    assert restored.horizon_negative_threshold == pytest.approx(0.15)
    assert restored.horizon_pair_budget == 12
    assert restored.horizon_paths_per_pair == 5
    assert restored.horizon_path_expansion_budget == 123
    assert restored.horizon_max_iterations == 2

    artifact = json.loads(path.read_text(encoding="utf-8"))
    assert artifact["artifact_version"] == 3


def test_decoder_load_migrates_v1_threshold_name(tmp_path):
    path = tmp_path / "decoder-v1.json"
    path.write_text(
        json.dumps(
            {
                "artifact_type": "ConditionalNodeFieldGraphDecoder",
                "artifact_version": 1,
                "config": {
                    "verbose": False,
                    "existence_threshold": 0.73,
                },
            }
        ),
        encoding="utf-8",
    )

    restored = ConditionalNodeFieldGraphDecoder().load(str(path))

    assert restored.direct_edge_probability_threshold == pytest.approx(0.73)
    assert not hasattr(restored, "existence_threshold")
    assert restored.horizon_path_expansion_budget == 4096


def test_decoder_load_migrates_v2_path_expansion_budget(tmp_path):
    path = tmp_path / "decoder-v2.json"
    path.write_text(
        json.dumps(
            {
                "artifact_type": "ConditionalNodeFieldGraphDecoder",
                "artifact_version": 2,
                "config": {"verbose": False},
            }
        ),
        encoding="utf-8",
    )

    restored = ConditionalNodeFieldGraphDecoder().load(str(path))

    assert restored.horizon_path_expansion_budget == 4096


def test_decoder_load_supports_legacy_dill_artifact(tmp_path):
    decoder = ConditionalNodeFieldGraphDecoder(
        verbose=False,
        enforce_connectivity=False,
        degree_slack_penalty=321.0,
    )
    decoder.existence_threshold = 0.61
    delattr(decoder, "direct_edge_probability_threshold")
    path = tmp_path / "decoder_legacy.pkl"
    import dill as pickle

    with open(path, "wb") as handle:
        pickle.dump(decoder, handle)

    restored = ConditionalNodeFieldGraphDecoder().load(str(path))

    assert isinstance(restored, ConditionalNodeFieldGraphDecoder)
    assert restored.enforce_connectivity is False
    assert restored.degree_slack_penalty == pytest.approx(321.0)
    assert restored.direct_edge_probability_threshold == pytest.approx(0.61)
    assert not hasattr(restored, "existence_threshold")


def test_load_graph_generator_accepts_unsanitized_model_name(tmp_path):
    generator = _make_fitted_sampling_generator()
    generator.model_name = "zinc-streaming-n64-s0.05-w2048-b256-e5"

    filename = save_graph_generator(generator, model_dir=tmp_path, log=False)
    assert filename == "zinc-streaming-n64-s0-05-w2048-b256-e5.pkl"

    restored = load_graph_generator(generator.model_name, model_dir=tmp_path)

    assert isinstance(restored, ConditionalNodeFieldGraphGenerator)
    assert restored.model_name == "zinc-streaming-n64-s0-05-w2048-b256-e5"


def test_load_graph_generator_restores_legacy_encoding_pipeline_defaults(tmp_path):
    generator = _make_fitted_sampling_generator()
    generator.model_name = "legacy-encoding"
    if hasattr(generator, "encoding_pipeline_"):
        delattr(generator, "encoding_pipeline_")
    if hasattr(generator, "use_embedding_svd"):
        delattr(generator, "use_embedding_svd")

    save_graph_generator(generator, model_dir=tmp_path, log=False)

    restored = load_graph_generator("legacy-encoding", model_dir=tmp_path)

    assert restored.use_embedding_svd is False
    assert isinstance(restored.encoding_pipeline_, EncodingPipeline)
    assert restored.encoding_pipeline_.owner is restored
    assert isinstance(restored.supervision_planner_, SupervisionPlanner)
    assert restored.supervision_planner_.owner is restored
    assert isinstance(restored.node_batch_builder_, NodeBatchBuilder)
    assert restored.node_batch_builder_.owner is restored
    assert isinstance(restored.conditioning_sampler_, ConditioningSampler)
    assert restored.conditioning_sampler_.owner is restored
    assert isinstance(restored.stream_fit_service_, StreamFitService)
    assert restored.stream_fit_service_.owner is restored


def test_load_graph_generator_restores_legacy_sample_oracle_runtime_defaults(tmp_path, monkeypatch):
    generator = _make_fitted_sampling_generator()
    generator.model_name = "legacy-sample-oracle"
    delattr(generator, "feasibility_oracle_candidates_per_attempt")
    delattr(generator, "max_decode_seconds_per_sample")
    delattr(generator, "max_decode_attempts_per_sample")
    generator.graph_decoder.adjacency_time_limit_seconds = 60.0
    generator.graph_decoder.parallel_decode_timeout_seconds = 30.0
    generator.graph_decoder.active_time_limit_seconds = None
    generator.graph_decoder.solver_threads = None
    delattr(generator.graph_decoder, "adjacency_time_limit_seconds")
    delattr(generator.graph_decoder, "parallel_decode_timeout_seconds")
    delattr(generator.graph_decoder, "active_time_limit_seconds")
    delattr(generator.graph_decoder, "solver_threads")
    delattr(generator, "decode_service_")

    save_graph_generator(generator, model_dir=tmp_path, log=False)

    restored = load_graph_generator("legacy-sample-oracle", model_dir=tmp_path)

    assert (
        restored.feasibility_oracle_candidates_per_attempt
        == ConditionalNodeFieldGraphGenerator._DEFAULT_FEASIBILITY_ORACLE_CANDIDATES_PER_ATTEMPT
    )
    assert restored.max_decode_seconds_per_sample is None
    assert restored.max_decode_attempts_per_sample == 1
    assert restored.graph_decoder.adjacency_time_limit_seconds == pytest.approx(60.0)
    assert restored.graph_decoder.parallel_decode_timeout_seconds == pytest.approx(30.0)
    assert restored.graph_decoder.active_time_limit_seconds is None
    assert restored.graph_decoder.solver_threads is None
    assert restored.decode_service_.owner is restored

    captured = {}

    def _fake_decode(graph_conditioning, **kwargs):
        del graph_conditioning
        captured.update(kwargs)
        return ["decoded"]

    monkeypatch.setattr(restored.decode_service_, "decode", _fake_decode)

    assert restored.sample(1, use_feasibility_oracle=False) == ["decoded"]
    assert captured["feasibility_oracle_candidates_per_attempt"] == 0


def test_adj_mtx_to_targets_preserves_expected_locality_pairs():
    decoder = ConditionalNodeFieldGraphDecoder(verbose=False)
    adj = [np.asarray([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int)]
    encodings = [np.asarray([[0.0], [1.0], [2.0]], dtype=float)]

    targets, pairs = decoder.adj_mtx_to_targets(
        adj,
        encodings,
        locality_sample_fraction=1.0,
        negative_sample_factor=1,
        force_bi_directional_edges=True,
        is_training=False,
        horizon=1,
    )

    assert targets.tolist().count(1) == 8
    assert targets.tolist().count(0) == 4
    assert pairs.count((0, 0, 1)) == 2
    assert pairs.count((0, 0, 2)) == 2
    assert pairs.count((0, 1, 0)) == 2
    assert pairs.count((0, 1, 2)) == 2


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
        oracle_use_edge_label_cuts=True,
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
    monkeypatch.setattr(
        oracle_decode_module,
        "oracle_adjacency_timeout_seconds",
        lambda owner: None,
    )

    decoded = generator._decode_generated_nodes(_oracle_generated_batch())

    assert len(decoded) == 1
    assert active_cut_counts[-2:] == [1, 0]


def test_oracle_edge_memory_helpers_are_symmetric_and_reduce_edge_probabilities():
    prior = np.zeros((4, 4), dtype=float)
    updated = cngg_module._update_oracle_edge_memory(
        prior,
        [frozenset({(0, 1), (2, 3)})],
        update_weight=1.5,
        decay=1.0,
        clip_value=5.0,
    )

    assert updated[0, 1] == pytest.approx(1.5)
    assert updated[1, 0] == pytest.approx(1.5)
    assert updated[2, 3] == pytest.approx(1.5)
    assert updated[3, 2] == pytest.approx(1.5)
    assert np.allclose(np.diag(updated), 0.0)

    prob_matrix = np.asarray(
        [
            [0.0, 0.9, 0.2, 0.2],
            [0.9, 0.0, 0.2, 0.2],
            [0.2, 0.2, 0.0, 0.9],
            [0.2, 0.2, 0.9, 0.0],
        ],
        dtype=float,
    )
    penalized = cngg_module._apply_oracle_edge_memory_penalty(
        prob_matrix,
        updated,
        penalty_weight=0.75,
    )

    assert penalized[0, 1] < prob_matrix[0, 1]
    assert penalized[2, 3] < prob_matrix[2, 3]
    assert penalized[0, 2] == pytest.approx(prob_matrix[0, 2])


def test_decode_generated_nodes_repairs_node_labels_before_structural_cuts():
    estimator = _LabelAwareOracleEstimator(
        edge_sets_per_call=[[], []],
        node_sets_per_call=[[[0]], []],
    )
    generator = ConditionalNodeFieldGraphGenerator(
        verbose=False,
        oracle_use_node_label_cuts=True,
    )
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
    generator = ConditionalNodeFieldGraphGenerator(
        verbose=False,
        oracle_use_edge_label_cuts=True,
    )
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


def test_oracle_edge_label_pressure_changes_multiple_labels_per_violation():
    generator = ConditionalNodeFieldGraphGenerator(
        verbose=False,
        oracle_use_edge_label_cuts=True,
        oracle_edge_label_min_changes_per_violation=2,
    )
    generator.edge_label_classes_ = np.asarray(["aromatic", "single"], dtype=object)
    generator.edge_label_to_index_ = {"aromatic": 0, "single": 1}
    adjacency = np.asarray(
        [
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=float,
    )
    current_labels = np.asarray(
        [
            [None, "aromatic", None, None],
            ["aromatic", None, "aromatic", None],
            [None, "aromatic", None, "aromatic"],
            [None, None, "aromatic", None],
        ],
        dtype=object,
    )
    label_probabilities = np.zeros((4, 4, 2), dtype=float)
    label_probabilities[..., 0] = 0.9
    label_probabilities[..., 1] = 0.1
    violating_edges = ((0, 1), (1, 2), (2, 3))

    _, repaired = generator._repair_labels_with_oracle(
        existence_mask=np.ones(4, dtype=bool),
        adj_mtx=adjacency,
        current_node_labels=np.asarray(["C"] * 4, dtype=object),
        current_edge_label_matrix=current_labels,
        node_label_probabilities=None,
        edge_label_probabilities=label_probabilities,
        forbidden_node_assignments=[],
        forbidden_edge_assignments=[(violating_edges, ("aromatic",) * 3)],
    )

    changed_count = sum(repaired[i, j] != "aromatic" for i, j in violating_edges)
    assert changed_count >= 2


def test_decode_generated_nodes_jointly_repairs_node_and_edge_labels():
    estimator = _LabelAwareOracleEstimator(
        edge_sets_per_call=[[[(0, 1)]], []],
        node_sets_per_call=[[[0]], []],
    )
    generator = ConditionalNodeFieldGraphGenerator(
        verbose=False,
        oracle_use_node_label_cuts=True,
        oracle_use_edge_label_cuts=True,
    )
    generator.feasibility_estimator = estimator
    generator.node_label_classes_ = np.asarray(["C", "O"], dtype=object)
    generator.node_label_to_index_ = {"C": 0, "O": 1}
    generator.edge_label_classes_ = np.asarray(["-", "="], dtype=object)
    generator.edge_label_to_index_ = {"-": 0, "=": 1}

    decoded = generator._decode_generated_nodes(
        _oracle_label_generated_batch(
            node_labels=["C", "C"],
            edge_label_matrix=[[None, "-"], ["-", None]],
            node_label_probabilities=[
                [0.05, 0.95],
                [0.95, 0.05],
            ],
            edge_label_probabilities=[
                [[1.0, 0.0], [0.05, 0.95]],
                [[0.05, 0.95], [1.0, 0.0]],
            ],
        )
    )

    assert len(decoded) == 1
    assert decoded[0].nodes[0]["label"] == "O"
    assert decoded[0].edges[(0, 1)]["label"] == "="
    assert estimator.node_calls >= 2
    assert estimator.edge_calls >= 2


def test_decode_generated_nodes_skips_node_label_repair_without_probabilities():
    estimator = _LabelAwareOracleEstimator(
        edge_sets_per_call=[[], []],
        node_sets_per_call=[[[0]], [[0]]],
    )
    generator = ConditionalNodeFieldGraphGenerator(
        verbose=False,
        max_oracle_iterations=2,
        oracle_use_node_label_cuts=True,
    )
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


def test_decode_generated_nodes_ignores_label_cuts_by_default_but_keeps_structural_cuts():
    estimator = _LabelAwareOracleEstimator(
        edge_sets_per_call=[[[(0, 1)]], []],
        node_sets_per_call=[[[0]], [[0]]],
    )
    generator = ConditionalNodeFieldGraphGenerator(verbose=False)
    generator.feasibility_estimator = estimator
    generator.node_label_classes_ = np.asarray(["C", "O"], dtype=object)
    generator.node_label_to_index_ = {"C": 0, "O": 1}
    generator.edge_label_classes_ = np.asarray(["-", "="], dtype=object)
    generator.edge_label_to_index_ = {"-": 0, "=": 1}

    decoded = generator._decode_generated_nodes(
        _oracle_label_generated_batch(
            node_labels=["C", "C"],
            edge_label_matrix=[[None, "-"], ["-", None]],
            node_label_probabilities=[
                [0.05, 0.95],
                [0.95, 0.05],
            ],
            edge_label_probabilities=[
                [[1.0, 0.0], [0.05, 0.95]],
                [[0.05, 0.95], [1.0, 0.0]],
            ],
        )
    )

    assert len(decoded) == 1
    assert decoded[0].nodes[0]["label"] == "C"
    assert decoded[0].edges[(0, 1)]["label"] == "-"
    assert estimator.node_calls == 0
    assert estimator.edge_calls >= 2


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


def test_oracle_candidate_score_components_smoke_uses_probability_eps_after_refactor():
    generator = ConditionalNodeFieldGraphGenerator(verbose=False)
    generator.node_label_to_index_ = {"C": 0}
    generator.edge_label_to_index_ = {"-": 0}

    total, edge_score, node_score, edge_label_score = generator._oracle_candidate_score_components(
        existence_mask=np.asarray([True, True], dtype=bool),
        adj_mtx=np.asarray([[0, 1], [1, 0]], dtype=float),
        node_labels=np.asarray(["C", "C"], dtype=object),
        edge_label_matrix=np.asarray([[None, "-"], ["-", None]], dtype=object),
        edge_probability_matrix=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float),
        node_label_probabilities=np.asarray([[1.0], [1.0]], dtype=float),
        edge_label_probabilities=np.asarray(
            [
                [[1.0], [1.0]],
                [[1.0], [1.0]],
            ],
            dtype=float,
        ),
    )

    assert np.isfinite(total)
    assert np.isfinite(edge_score)
    assert np.isfinite(node_score)
    assert np.isfinite(edge_label_score)


def test_decode_generated_nodes_with_oracle_falls_back_when_initial_seed_decode_fails(monkeypatch):
    estimator = _OracleOnceEstimator([[]])
    decoder = ConditionalNodeFieldGraphDecoder(verbose=False, enforce_connectivity=True)
    generator = ConditionalNodeFieldGraphGenerator(
        graph_decoder=decoder,
        feasibility_estimator=estimator,
        verbose=False,
    )

    def _fail_initial_decode(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("initial connected seed solve failed")

    optimize_calls = []

    def _fake_optimize(
        self,
        prob_matrix,
        target_degrees,
        target_edge_count=None,
        timeLimit=60,
        verbose=False,
        alpha=0.7,
        connectivity=None,
        forbidden_edge_sets=None,
        active_node_mask=None,
    ):
        del (
            self,
            target_degrees,
            target_edge_count,
            timeLimit,
            verbose,
            alpha,
            forbidden_edge_sets,
            active_node_mask,
        )
        optimize_calls.append(connectivity)
        n = prob_matrix.shape[0]
        adj = np.zeros((n, n), dtype=int)
        if n >= 2:
            for i in range(n - 1):
                adj[i, i + 1] = 1
                adj[i + 1, i] = 1
        return adj

    monkeypatch.setattr(decoder, "decode_adjacency_matrix", _fail_initial_decode)
    monkeypatch.setattr(ConditionalNodeFieldGraphDecoder, "optimize_adjacency_matrix", _fake_optimize)
    monkeypatch.setattr(
        oracle_decode_module,
        "oracle_adjacency_timeout_seconds",
        lambda owner: None,
    )

    decoded = generator._decode_generated_nodes(_oracle_generated_batch())

    assert len(decoded) == 1
    assert decoded[0].number_of_nodes() > 0
    assert optimize_calls
    assert optimize_calls[0] is False


def test_decode_generated_nodes_with_oracle_skips_seed_timeout(monkeypatch):
    estimator = _OracleOnceEstimator([[]])
    decoder = ConditionalNodeFieldGraphDecoder(verbose=False, enforce_connectivity=True)
    generator = ConditionalNodeFieldGraphGenerator(
        graph_decoder=decoder,
        feasibility_estimator=estimator,
        verbose=False,
    )

    def _timeout_initial_decode(*args, **kwargs):
        del args, kwargs
        raise TimeoutError("seed decode timed out")

    monkeypatch.setattr(decoder, "decode_adjacency_matrix", _timeout_initial_decode)

    decoded = generator._decode_generated_nodes_with_oracle(_oracle_generated_batch())

    assert decoded == [None]
    assert estimator.calls == []


def test_decode_generated_nodes_reruns_joint_label_repair_after_structural_change(monkeypatch):
    estimator = _OracleOnceEstimator([
        [[(1, 0), (2, 1), (3, 2), (3, 0)]],
        [],
    ])
    generator = ConditionalNodeFieldGraphGenerator(
        graph_decoder=ConditionalNodeFieldGraphDecoder(verbose=False, enforce_connectivity=True),
        feasibility_estimator=estimator,
        oracle_use_edge_label_cuts=True,
        verbose=False,
    )
    generator.node_label_classes_ = np.asarray(["C"], dtype=object)
    generator.node_label_to_index_ = {"C": 0}
    generator.edge_label_classes_ = np.asarray(["-", "="], dtype=object)
    generator.edge_label_to_index_ = {"-": 0, "=": 1}

    generated = GeneratedNodeBatch(
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
        edge_existence_probabilities=[
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
        edge_label_probabilities=[
            np.repeat(
                np.asarray(
                    [
                        [0.0, 1.0],
                        [0.8, 0.2],
                        [0.8, 0.2],
                        [0.8, 0.2],
                    ],
                    dtype=float,
                )[None, :, :],
                4,
                axis=0,
            )
        ],
    )

    call_adjacencies = []
    original_repair = generator._repair_labels_with_oracle

    def wrapped_repair(*args, **kwargs):
        call_adjacencies.append(np.asarray(kwargs["adj_mtx"], dtype=int).copy())
        return original_repair(*args, **kwargs)

    monkeypatch.setattr(generator, "_repair_labels_with_oracle", wrapped_repair)

    decoded = generator._decode_generated_nodes(generated)

    assert len(decoded) == 1
    assert len(call_adjacencies) >= 1


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


def test_fit_compresses_sparse_node_and_graph_embeddings_with_svd():
    node_model = _TrainableNodeModel(verbose=False)
    generator = ConditionalNodeFieldGraphGenerator(
        graph_vectorizer=_SparseGraphVectorizer(dimension=9),
        node_graph_vectorizer=_SparseNodeVectorizer(dimension=11),
        conditional_node_generator_model=node_model,
        graph_decoder=ConditionalNodeFieldGraphDecoder(verbose=False),
        use_embedding_svd=True,
        node_embedding_svd_dimension=3,
        graph_embedding_svd_dimension=2,
        verbose=False,
    )

    graphs = _sampling_graphs()
    generator.fit(graphs, train_node_generator=True)

    assert generator.node_embedding_svd_fitted_ is True
    assert generator.graph_embedding_svd_fitted_ is True
    assert generator.node_embedding_raw_dimension_ == 11
    assert generator.graph_embedding_raw_dimension_ == 9
    setup_call = node_model.setup_calls[0]
    assert setup_call["node_batch"].node_embeddings_list[0].shape[1] == 3
    assert setup_call["graph_conditioning"].graph_embeddings.shape[1] == 2
    assert generator.training_graph_conditioning_.graph_embeddings.shape[1] == 2

    encoded_nodes, encoded_conditioning = generator.encode(graphs[:2])
    assert encoded_nodes[0].shape[1] == 3
    assert encoded_conditioning.graph_embeddings.shape == (2, 2)


def test_embedding_svd_fit_row_sampling_and_chunked_projection():
    generator = ConditionalNodeFieldGraphGenerator(
        graph_vectorizer=_SparseGraphVectorizer(dimension=9),
        node_graph_vectorizer=_SparseNodeVectorizer(dimension=11),
        use_embedding_svd=True,
        node_embedding_svd_dimension=3,
        graph_embedding_svd_dimension=2,
        embedding_svd_fit_max_rows=4,
        embedding_svd_transform_batch_size=2,
        verbose=False,
    )
    pipeline = generator._ensure_encoding_pipeline()
    matrix = sparse.csr_matrix(np.arange(80, dtype=float).reshape(10, 8))

    sampled = pipeline.sample_svd_fit_rows(matrix, requested_dimension=3, label="node")

    assert sparse.issparse(sampled)
    assert sampled.shape == (4, 8)
    generator.fit(_sampling_graphs(), train_node_generator=False)
    encoded_nodes, encoded_conditioning = generator.encode(_sampling_graphs()[:2])
    assert encoded_nodes[0].shape[1] == 3
    assert encoded_conditioning.graph_embeddings.shape == (2, 2)


def test_embedding_svd_graph_dimension_defaults_to_node_dimension():
    generator = ConditionalNodeFieldGraphGenerator(
        graph_vectorizer=_SparseGraphVectorizer(dimension=8),
        node_graph_vectorizer=_SparseNodeVectorizer(dimension=10),
        use_embedding_svd=True,
        node_embedding_svd_dimension=4,
        graph_embedding_svd_dimension=None,
        verbose=False,
    )

    generator.fit(_sampling_graphs(), train_node_generator=False)

    assert generator.node_embedding_effective_dimension_ == 4
    assert generator.graph_embedding_effective_dimension_ == 4
    assert generator.graph_encode(_sampling_graphs()[:1]).graph_embeddings.shape == (1, 4)


def test_embedding_svd_skips_when_requested_dimension_exceeds_raw_dimension():
    generator = ConditionalNodeFieldGraphGenerator(
        graph_vectorizer=_SparseGraphVectorizer(dimension=3),
        node_graph_vectorizer=_SparseNodeVectorizer(dimension=4),
        use_embedding_svd=True,
        node_embedding_svd_dimension=99,
        graph_embedding_svd_dimension=99,
        verbose=False,
    )

    generator.fit(_sampling_graphs(), train_node_generator=False)

    assert generator.node_embedding_svd_fitted_ is False
    assert generator.graph_embedding_svd_fitted_ is False
    assert generator.node_encode(_sampling_graphs()[:1])[0].shape[1] == 4
    assert generator.graph_encode(_sampling_graphs()[:1]).graph_embeddings.shape == (1, 3)


def test_embedding_svd_disabled_preserves_raw_embedding_dimensions():
    generator = ConditionalNodeFieldGraphGenerator(
        graph_vectorizer=_SparseGraphVectorizer(dimension=5),
        node_graph_vectorizer=_SparseNodeVectorizer(dimension=6),
        use_embedding_svd=False,
        node_embedding_svd_dimension=2,
        graph_embedding_svd_dimension=2,
        verbose=False,
    )

    generator.fit(_sampling_graphs(), train_node_generator=False)

    assert generator.node_embedding_svd_fitted_ is False
    assert generator.graph_embedding_svd_fitted_ is False
    assert generator.node_encode(_sampling_graphs()[:1])[0].shape[1] == 6
    assert generator.graph_encode(_sampling_graphs()[:1]).graph_embeddings.shape == (1, 5)


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
        lambda n_samples, interpolate_between_n_samples=None: type(
            generator.training_graph_conditioning_
        )(
            graph_embeddings=np.asarray([[1.0, 2.0]], dtype=float),
            node_counts=np.asarray([3], dtype=np.int64),
            edge_counts=np.asarray([2], dtype=np.int64),
        ),
    )

    def _fake_decode(graph_conditioning, **kwargs):
        del kwargs
        captured["conditioning"] = graph_conditioning
        return ["decoded"]

    monkeypatch.setattr(generator.decode_service_, "decode", _fake_decode)

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
    monkeypatch.setattr(
        generator.decode_service_,
        "decode",
        lambda graph_conditioning, **kwargs: [graph_conditioning],
    )

    result = generator.sample(1, interpolate_between_n_samples=10)

    assert len(result) == 1
    assert captured == {"n_samples": 1, "interpolate_between_n_samples": 10}


def test_sample_forwards_direct_decoder_options(monkeypatch):
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
        del graph_conditioning
        captured.update(kwargs)
        return ["decoded"]

    monkeypatch.setattr(generator.decode_service_, "decode", _fake_decode)

    result = generator.sample(
        1,
        use_ilp_decoder=False,
        edge_probability_threshold=0.7,
    )

    assert result == ["decoded"]
    assert captured["use_ilp_decoder"] is False
    assert captured["edge_probability_threshold"] == pytest.approx(0.7)


def test_sample_can_disable_feasibility_oracle(monkeypatch):
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
        del graph_conditioning
        captured.update(kwargs)
        return ["decoded"]

    monkeypatch.setattr(generator.decode_service_, "decode", _fake_decode)

    result = generator.sample(1, use_feasibility_oracle=False)

    assert result == ["decoded"]
    assert captured["feasibility_oracle_candidates_per_attempt"] == 0


def test_sample_can_enable_feasibility_oracle_when_configured_budget_is_zero(monkeypatch):
    generator = _make_fitted_sampling_generator()
    generator.feasibility_oracle_candidates_per_attempt = 0
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
        del graph_conditioning
        captured.update(kwargs)
        return ["decoded"]

    monkeypatch.setattr(generator.decode_service_, "decode", _fake_decode)

    result = generator.sample(1, use_feasibility_oracle=True)

    assert result == ["decoded"]
    assert (
        captured["feasibility_oracle_candidates_per_attempt"]
        == ConditionalNodeFieldGraphGenerator._DEFAULT_FEASIBILITY_ORACLE_CANDIDATES_PER_ATTEMPT
    )


def test_graph_generator_rejects_invalid_feasibility_rejection_mode():
    with pytest.raises(ValueError, match="feasibility_rejection_mode"):
        ConditionalNodeFieldGraphGenerator(feasibility_rejection_mode="drop")


def test_optimize_adjacency_matrix_raises_when_solver_has_no_feasible_solution(monkeypatch):
    decoder = ConditionalNodeFieldGraphDecoder(verbose=False)

    def _fake_solve(self, solver):
        del solver
        self.status = pulp.LpStatusInfeasible
        return self.status

    monkeypatch.setattr(pulp.LpProblem, "solve", _fake_solve)

    with pytest.raises(RuntimeError, match="did not produce a feasible solution"):
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


def test_optimize_adjacency_matrix_accepts_valid_integer_feasible_incumbent(monkeypatch):
    decoder = ConditionalNodeFieldGraphDecoder(verbose=False)

    def _fake_solve(self, solver):
        del solver
        self.status = pulp.LpStatusOptimal
        self.sol_status = pulp.LpSolutionIntegerFeasible
        for variable in self.variables():
            if variable.name == "x_0_1":
                variable.varValue = 1.0
            elif variable.name.startswith("flow_"):
                variable.varValue = 1.0
            else:
                variable.varValue = 0.0
        return self.status

    monkeypatch.setattr(pulp.LpProblem, "solve", _fake_solve)

    adjacency = decoder.optimize_adjacency_matrix(
        prob_matrix=np.array([[0.0, 0.9], [0.9, 0.0]], dtype=float),
        target_degrees=[1, 1],
    )

    assert adjacency.tolist() == [[0, 1], [1, 0]]
    assert decoder.last_adjacency_solve_report_.used_incumbent is True
    assert decoder.last_adjacency_solve_report_.optimal is False


def test_optimize_adjacency_matrix_rejects_fractional_incumbent(monkeypatch):
    decoder = ConditionalNodeFieldGraphDecoder(verbose=False)

    def _fake_solve(self, solver):
        del solver
        self.status = pulp.LpStatusOptimal
        self.sol_status = pulp.LpSolutionIntegerFeasible
        for variable in self.variables():
            variable.varValue = 0.5 if variable.name == "x_0_1" else 0.0
        return self.status

    monkeypatch.setattr(pulp.LpProblem, "solve", _fake_solve)

    with pytest.raises(RuntimeError, match="fractional"):
        decoder.optimize_adjacency_matrix(
            prob_matrix=np.array([[0.0, 0.9], [0.9, 0.0]], dtype=float),
            target_degrees=[1, 1],
        )


def test_optimize_adjacency_matrix_rejects_timeout_without_incumbent(monkeypatch):
    decoder = ConditionalNodeFieldGraphDecoder(verbose=False)

    def _fake_solve(self, solver):
        del solver
        self.status = pulp.LpStatusNotSolved
        self.sol_status = pulp.LpSolutionNoSolutionFound
        return self.status

    monkeypatch.setattr(pulp.LpProblem, "solve", _fake_solve)

    with pytest.raises(RuntimeError, match="did not produce a feasible solution"):
        decoder.optimize_adjacency_matrix(
            prob_matrix=np.array([[0.0, 0.9], [0.9, 0.0]], dtype=float),
            target_degrees=[1, 1],
        )
