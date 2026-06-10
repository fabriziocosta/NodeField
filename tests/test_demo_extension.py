from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from conditional_node_field_graph_generator.extensions.demo.pipeline import (
    FeasibilityEstimator,
    FeasibilityEstimatorFeatureCannotExist,
    WithinRangeFeasibilityEstimatorFromNumericalFunction,
    benchmark_regression_guidance,
    build_graph_generator,
    build_zinc_dataset,
    prepare_zinc_data_split,
    combination,
    compose,
    cycle,
    ensure_demo_feasibility_estimator,
    fit_graph_generator,
    neighborhood,
    prepare_experiment,
    sample_hyperparameter_configuration,
    score_graph_generator_feasible_rate,
    unlabel,
)
from conditional_node_field_graph_generator.extensions.demo.visualization import (
    _temporary_decoder_n_jobs,
    compare_real_vs_generated,
    infer_display_mode,
    offset_neg_graphs,
    select_pos_neg,
    show_molecules,
)
from conditional_node_field_graph_generator.extensions.synthetic import ArtificialGraphDatasetConstructor


def _labeled_graph(label):
    graph = nx.Graph()
    graph.add_node(0, label=label)
    return graph


def test_offset_neg_graphs_only_offsets_negative_examples():
    pos_graph = _labeled_graph(1)
    neg_graph = _labeled_graph(2)

    out_graphs, out_targets = offset_neg_graphs([pos_graph, neg_graph], [1, 0], offset=10)

    assert out_targets == [1, 0]
    assert pos_graph.nodes[0]["label"] == 1
    assert neg_graph.nodes[0]["label"] == 2
    assert out_graphs[0].nodes[0]["label"] == 1
    assert out_graphs[1].nodes[0]["label"] == 12


def test_select_pos_neg_applies_cap_per_group():
    graphs = [_labeled_graph(idx) for idx in range(8)]
    targets = [1, 1, 1, 1, 0, 0, 0, 0]

    pos_graphs, neg_graphs = select_pos_neg(graphs, targets, n_lines=1, n_graphs_per_line=2)

    assert len(pos_graphs) == 2
    assert len(neg_graphs) == 2
    assert [graph.nodes[0]["label"] for graph in pos_graphs] == [0, 1]
    assert [graph.nodes[0]["label"] for graph in neg_graphs] == [4, 5]


def test_infer_display_mode_detects_molecule_metadata_and_labels():
    meta_graph = _labeled_graph("x")
    meta_graph.graph["smiles"] = "CCO"
    label_graph = _labeled_graph("C")
    plain_graph = _labeled_graph("custom")

    assert infer_display_mode([]) == "not_molecule"
    assert infer_display_mode([meta_graph]) == "molecule"
    assert infer_display_mode([label_graph]) == "molecule"
    assert infer_display_mode([plain_graph]) == "not_molecule"


def test_temporary_decoder_n_jobs_restores_original_value():
    decoder = type("_Decoder", (), {"n_jobs": 3})()
    graph_generator = type("_GraphGenerator", (), {"graph_decoder": decoder})()

    with _temporary_decoder_n_jobs(graph_generator, decoder_n_jobs=1):
        assert decoder.n_jobs == 1

    assert decoder.n_jobs == 3


def test_show_molecules_returns_none_for_empty_input(capsys):
    result = show_molecules([], title="Empty")

    assert result is None
    output = capsys.readouterr().out
    assert "Empty" in output
    assert "No graphs to display." in output


def test_prepare_experiment_splits_dataset_and_preserves_outputs(capsys):
    def build_dataset_fn(dataset_size, marker):
        graphs = [f"{marker}-{idx}" for idx in range(dataset_size)]
        targets = np.arange(dataset_size)
        return graphs, targets

    graphs, targets, train_graphs, test_graphs, train_targets, test_targets = prepare_experiment(
        build_dataset_fn,
        dataset_size=10,
        test_size=3,
        random_state=7,
        marker="demo",
    )

    assert len(graphs) == 10
    assert len(targets) == 10
    assert len(train_graphs) == 7
    assert len(test_graphs) == 3
    assert len(train_targets) == 7
    assert len(test_targets) == 3
    assert sorted(train_graphs + test_graphs) == [f"demo-{idx}" for idx in range(10)]
    assert "train_graphs:7   test_graphs:3" in capsys.readouterr().out


def test_build_zinc_dataset_uses_compact_size_interface(monkeypatch, tmp_path):
    calls = {}

    def fake_download(dataset_dir):
        calls["download"] = Path(dataset_dir)
        return Path(dataset_dir) / "zinc_250k.csv"

    class _FakeZINCLoader:
        def __init__(self, root, *, on_error="raise"):
            calls["loader"] = {"root": Path(root), "on_error": on_error}

        def load(self, dataset_name, *, limit=None, min_node_count=None, max_node_count=None):
            calls["load"] = {
                "dataset_name": dataset_name,
                "limit": limit,
                "min_node_count": min_node_count,
                "max_node_count": max_node_count,
            }
            graphs = []
            for idx in range(40):
                graph = nx.path_graph(10 + (idx % 2))
                graph.graph["zinc_id"] = f"z{idx}"
                graphs.append(graph)
            metadata = pd.DataFrame({"zinc_id": [f"z{idx}" for idx in range(40)]})
            return graphs, metadata

    monkeypatch.setattr(
        "conditional_node_field_graph_generator.extensions.demo.pipeline.download_zinc_dataset",
        fake_download,
    )
    monkeypatch.setattr(
        "conditional_node_field_graph_generator.extensions.demo.pipeline.ZINCLoader",
        _FakeZINCLoader,
    )

    graphs, metadata, manifest = build_zinc_dataset(
        dataset_dir=tmp_path,
        num_examples=25,
        min_size=10,
        max_size=12,
        random_state=7,
    )

    assert len(graphs) == 25
    assert len(metadata) == 25
    assert manifest["dataset_name"] == "zinc_250k"
    assert manifest["csv_path"] == str((tmp_path / "zinc_250k.csv").resolve())
    assert manifest["node_counts"] == [10, 11]
    assert manifest["total_graphs"] == 40
    assert calls["loader"] == {"root": tmp_path.resolve(), "on_error": "raise"}
    assert calls["load"] == {
        "dataset_name": "zinc_250k",
        "limit": None,
        "min_node_count": 10,
        "max_node_count": 12,
    }
    repeated_graphs = build_zinc_dataset(
        dataset_dir=tmp_path,
        num_examples=25,
        min_size=10,
        max_size=12,
        random_state=7,
    )[0]
    assert [graph.graph["zinc_id"] for graph in graphs] == [
        graph.graph["zinc_id"] for graph in repeated_graphs
    ]


def test_prepare_zinc_data_split_builds_targets_and_debug_subsets(monkeypatch, tmp_path):
    graphs = [nx.path_graph(10 + (idx % 12)) for idx in range(30)]
    metadata = pd.DataFrame(
        {
            "zinc_id": [f"z{idx}" for idx in range(30)],
            "logP": np.linspace(1.0, 3.9, num=30),
            "qed": np.linspace(0.1, 0.9, num=30),
            "SAS": np.linspace(2.0, 4.9, num=30),
        }
    )
    manifest = {"dataset_name": "zinc_250k", "node_counts": list(range(10, 22))}

    def fake_build_zinc_dataset(**kwargs):
        return list(graphs), metadata.copy(), dict(manifest)

    monkeypatch.setattr(
        "conditional_node_field_graph_generator.extensions.demo.pipeline.build_zinc_dataset",
        fake_build_zinc_dataset,
    )

    bundle = prepare_zinc_data_split(
        dataset_dir=tmp_path,
        num_examples=30,
        min_size=10,
        max_size=21,
        test_size=4,
        random_state=7,
        debug_mode=True,
        debug_train_subset=3,
        debug_test_subset=2,
    )

    assert len(bundle["graphs"]) == 30
    assert bundle["metadata"]["zinc_id"].tolist() == [f"z{idx}" for idx in range(30)]
    assert bundle["targets"].shape == (30, 3)
    assert bundle["target_columns"] == ["logP", "qed", "SAS"]
    assert len(bundle["train_graphs"]) == 3
    assert len(bundle["test_graphs"]) == 2
    assert len(bundle["train_metadata"]) == 3
    assert len(bundle["test_metadata"]) == 2
    assert tuple(bundle["train_targets"].shape) == (3, 3)
    assert tuple(bundle["test_targets"].shape) == (2, 3)


def test_fit_graph_generator_rejects_conflicting_resume_arguments():
    recorder = type("_Recorder", (), {"fit": lambda *args, **kwargs: None})()

    with pytest.raises(ValueError, match="Provide either ckpt_path or resume_latest_checkpoint"):
        fit_graph_generator(
            recorder,
            train_graphs=["g1"],
            ckpt_path="/tmp/a.ckpt",
            resume_latest_checkpoint=True,
        )


def test_fit_graph_generator_rejects_incompatible_latest_checkpoint(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    run_dir = checkpoint_root / "run_a"
    run_dir.mkdir(parents=True)
    checkpoint_path = run_dir / "last.ckpt"
    checkpoint_path.write_text("checkpoint")

    class _Recorder:
        def __init__(self):
            self.calls = []

        def fit(self, graphs, targets=None, ckpt_path=None):
            self.calls.append(
                {
                    "graphs": graphs,
                    "targets": targets,
                    "ckpt_path": ckpt_path,
                }
            )
            if ckpt_path is not None:
                raise RuntimeError(
                    "Error(s) in loading state_dict for ConditionalNodeFieldModule:\n\tsize mismatch for layernorm_in.weight"
                )

    recorder = _Recorder()

    with pytest.raises(RuntimeError, match="Checkpoint is incompatible with the current generator configuration"):
        fit_graph_generator(
            recorder,
            train_graphs=["g1", "g2"],
            targets=[1, 0],
            resume_latest_checkpoint=True,
            checkpoint_root=checkpoint_root,
        )

    assert recorder.calls == [
        {"graphs": ["g1", "g2"], "targets": [1, 0], "ckpt_path": str(checkpoint_path.resolve())},
    ]


def test_build_graph_generator_propagates_model_name_to_inner_generator():
    if any(
        dependency is None
        for dependency in (
            compose,
            cycle,
            neighborhood,
            unlabel,
            combination,
            FeasibilityEstimator,
            FeasibilityEstimatorFeatureCannotExist,
            WithinRangeFeasibilityEstimatorFromNumericalFunction,
        )
    ):
        pytest.skip("abstractgraph and abstractgraph_ml are not installed")

    generator = build_graph_generator(
        model_name="demo-artificial-n100-size8",
        model_dir="/tmp/models",
    )

    assert generator.model_name == "demo-artificial-n100-size8"
    assert generator.model_dir == "/tmp/models"
    assert generator.conditional_node_generator_model.model_name == "demo-artificial-n100-size8"
    assert generator.conditional_node_generator_model.model_dir == "/tmp/models"


def test_build_graph_generator_sets_oracle_budget_and_forwards_overrides():
    if any(
        dependency is None
        for dependency in (
            compose,
            cycle,
            neighborhood,
            unlabel,
            combination,
            FeasibilityEstimator,
            FeasibilityEstimatorFeatureCannotExist,
            WithinRangeFeasibilityEstimatorFromNumericalFunction,
        )
    ):
        pytest.skip("abstractgraph and abstractgraph_ml are not installed")

    default_generator = build_graph_generator()
    overridden_generator = build_graph_generator(
        feasibility_oracle_candidates_per_attempt=0,
        max_oracle_iterations=3,
        oracle_use_node_label_cuts=True,
        oracle_use_edge_label_cuts=True,
        sparse_supervision_mask_ratio=0.4,
        use_embedding_svd=True,
        node_embedding_svd_dimension=123,
        graph_embedding_svd_dimension=45,
    )

    assert default_generator.feasibility_oracle_candidates_per_attempt == 2
    assert default_generator.oracle_use_node_label_cuts is False
    assert default_generator.oracle_use_edge_label_cuts is False
    assert default_generator.node_graph_vectorizer.dense is False
    assert default_generator.graph_vectorizer.dense is False
    assert default_generator.use_embedding_svd is True
    assert overridden_generator.feasibility_oracle_candidates_per_attempt == 0
    assert overridden_generator.max_oracle_iterations == 3
    assert overridden_generator.oracle_use_node_label_cuts is True
    assert overridden_generator.oracle_use_edge_label_cuts is True
    assert overridden_generator.use_embedding_svd is True
    assert overridden_generator.node_embedding_svd_dimension == 123
    assert overridden_generator.graph_embedding_svd_dimension == 45
    assert (
        overridden_generator.conditional_node_generator_model.sparse_supervision_mask_ratio
        == pytest.approx(0.4)
    )


def test_build_graph_generator_disables_feasibility_when_optional_dependencies_are_missing(monkeypatch):
    monkeypatch.setattr(
        "conditional_node_field_graph_generator.extensions.demo.pipeline.compose",
        None,
    )
    monkeypatch.setattr(
        "conditional_node_field_graph_generator.extensions.demo.pipeline.cycle",
        None,
    )
    monkeypatch.setattr(
        "conditional_node_field_graph_generator.extensions.demo.pipeline.neighborhood",
        None,
    )
    monkeypatch.setattr(
        "conditional_node_field_graph_generator.extensions.demo.pipeline.unlabel",
        None,
    )
    monkeypatch.setattr(
        "conditional_node_field_graph_generator.extensions.demo.pipeline.combination",
        None,
    )

    generator = build_graph_generator()

    assert generator.feasibility_estimator is None
    assert generator.feasibility_oracle_candidates_per_attempt == 0
    assert generator.use_feasibility_filtering is False


def test_ensure_demo_feasibility_estimator_names_aromatic_level(monkeypatch):
    monkeypatch.setattr(
        "conditional_node_field_graph_generator.extensions.demo.pipeline._AbstractGraphFeasibilityEstimator",
        object,
    )
    children = [type("_Leaf", (), {"parallel": True})() for _ in range(5)]
    source = type("_Composite", (), {"feasibility_estimators": children})()

    wrapped = ensure_demo_feasibility_estimator(source)

    assert wrapped.estimator_names == ["edge", "path", "valence", "cycle", "aromatic"]


def test_artificial_graph_dataset_constructor_uses_internal_deduper_when_abstractgraph_is_missing():
    graphs, targets = ArtificialGraphDatasetConstructor(
        graph_generator_target_type_pos="cycle",
        graph_generator_context_type_pos="cycle",
        graph_generator_target_type_neg="tree",
        graph_generator_context_type_neg="tree",
        target_size_pos=5,
        context_size_pos=5,
        n_link_edges_pos=1,
        alphabet_size_pos=3,
        target_size_neg=5,
        context_size_neg=5,
        n_link_edges_neg=1,
        alphabet_size_neg=3,
    ).sample(8)

    assert len(graphs) == len(targets)
    assert len(graphs) > 0


def test_sample_hyperparameter_configuration_respects_typed_ranges():
    config = sample_hyperparameter_configuration(
        {
            "max_feasibility_attempts": {"type": "int", "low": 2, "high": 5},
            "sampling_step_size": {"type": "real", "low": 0.01, "high": 0.1},
        },
        random_state=7,
    )

    assert isinstance(config["max_feasibility_attempts"], int)
    assert 2 <= config["max_feasibility_attempts"] <= 5
    assert isinstance(config["sampling_step_size"], float)
    assert 0.01 <= config["sampling_step_size"] <= 0.1


def test_score_graph_generator_feasible_rate_forwards_to_member_function():
    calls = {}

    class _FakeScoringGenerator:
        def score_feasible_rate(self, **kwargs):
            calls["kwargs"] = kwargs
            return {"score": 0.25}

    result = score_graph_generator_feasible_rate(
        _FakeScoringGenerator(),
        n_samples=2,
        max_feasibility_attempts=4,
        feasibility_candidates_per_attempt=3,
        feasibility_oracle_candidates_per_attempt=1,
        verbose=True,
    )

    assert result == {"score": 0.25}
    assert calls["kwargs"] == {
        "n_samples": 2,
        "max_feasibility_attempts": 4,
        "feasibility_candidates_per_attempt": 3,
        "feasibility_oracle_candidates_per_attempt": 1,
        "interpolate_between_n_samples": None,
        "desired_target": None,
        "guidance_scale": 1.0,
        "verbose": True,
    }


class _FakeBenchmarkFeasibilityEstimator:
    def number_of_violations(self, decoded_graphs):
        return [int(graph.graph["violations"]) for graph in decoded_graphs]


class _FakeBenchmarkGenerator:
    def __init__(self):
        self.feasibility_estimator = _FakeBenchmarkFeasibilityEstimator()

    def _sample_conditions(self, n_samples, interpolate_between_n_samples=None):
        del interpolate_between_n_samples
        return [f"c{idx}" for idx in range(int(n_samples))]

    def _predict_generated_nodes(
        self,
        graph_conditioning,
        sampling_mode,
        desired_target,
        guidance_scale,
        predictor_scale,
    ):
        del desired_target, guidance_scale, predictor_scale
        return type(
            "_Generated",
            (),
            {
                "node_embeddings_list": [
                    np.asarray([[float(idx)]], dtype=float)
                    for idx, _conditioning in enumerate(graph_conditioning)
                ],
                "sampling_mode": sampling_mode,
            },
        )()

    def _decode_generated_nodes(self, generated_nodes):
        decoded = []
        for idx, _embedding in enumerate(generated_nodes.node_embeddings_list):
            graph = nx.Graph()
            if generated_nodes.sampling_mode == "unguided":
                graph.graph["violations"] = [0, 2, 5, 1][idx]
            else:
                graph.graph["violations"] = [0, 0, 6, 3][idx]
            decoded.append(graph)
        return decoded

    @staticmethod
    def _compute_guidance_targets(violation_counts):
        violations = np.asarray(violation_counts, dtype=float)
        return 1.0 / (1.0 + np.sqrt(violations))


def test_benchmark_regression_guidance_returns_paired_summary():
    result = benchmark_regression_guidance(
        _FakeBenchmarkGenerator(),
        n_samples=4,
        bootstrap_samples=200,
        random_state=7,
    )

    assert set(result.keys()) == {
        "summary",
        "paired_summary",
        "per_sample",
        "unguided_batch",
        "guided_batch",
    }
    summary = result["summary"]
    assert summary["label"].tolist() == ["unguided", "regression_guided"]
    assert summary["feasible_count"].tolist() == [1, 2]
    assert summary["count"].tolist() == [4, 4]
    paired = result["paired_summary"].iloc[0]
    assert int(paired["guided_only_feasible"]) == 1
    assert int(paired["unguided_only_feasible"]) == 0
    assert int(paired["both_feasible"]) == 1
    assert int(paired["neither_feasible"]) == 2
    assert paired["mean_feasible_rate_delta"] == pytest.approx(0.25)
    per_sample = result["per_sample"]
    assert per_sample["violation_delta"].tolist() == [0.0, -2.0, 1.0, 2.0]


class _FakeCompareGenerator:
    def graph_encode(self, graphs):
        return list(graphs)

    def _decode_with_feasibility_slots(self, conditioning, apply_feasibility_filtering=True):
        del apply_feasibility_filtering
        return [None for _ in conditioning]


class _FakeSuccessfulCompareGenerator:
    def graph_encode(self, graphs):
        return list(graphs)

    def _decode_with_feasibility_slots(self, conditioning, apply_feasibility_filtering=True):
        del apply_feasibility_filtering
        generated = []
        for graph in conditioning:
            copy = graph.copy()
            generated.append(copy)
        return generated


def test_compare_real_vs_generated_raises_when_no_feasible_outputs():
    with pytest.raises(RuntimeError, match="No feasible generated graphs"):
        compare_real_vs_generated(_FakeCompareGenerator(), [_labeled_graph("C")])


def test_compare_real_vs_generated_returns_summary_tables(monkeypatch):
    displayed = []
    monkeypatch.setattr(
        "conditional_node_field_graph_generator.extensions.demo.visualization.display",
        lambda obj: displayed.append(obj),
    )
    monkeypatch.setattr(
        "conditional_node_field_graph_generator.extensions.demo.visualization.plt.show",
        lambda: None,
    )

    graph_a = nx.Graph()
    graph_a.add_node(0, label="C")
    graph_a.add_node(1, label="O")
    graph_a.add_edge(0, 1, label="-")

    graph_b = nx.Graph()
    graph_b.add_node(0, label="N")
    graph_b.add_node(1, label="C")
    graph_b.add_edge(0, 1, label="=")

    result = compare_real_vs_generated(_FakeSuccessfulCompareGenerator(), [graph_a, graph_b])

    assert set(result.keys()) == {"summary", "comparison_tables", "real_graphs", "generated_graphs"}
    assert isinstance(result["summary"], pd.DataFrame)
    assert set(result["comparison_tables"].keys()) == {"node_count", "edge_count", "atom_label", "bond_label"}
    assert len(result["real_graphs"]) == 2
    assert len(result["generated_graphs"]) == 2
    assert len(displayed) >= 5


class _MaskAwareFeasibilityLeaf:
    def __init__(self, accepted_ids, violating_edge_sets=None):
        self.accepted_ids = set(accepted_ids)
        self._violating_edge_sets = list(violating_edge_sets or [])
        self.parallel = True

    def fit(self, graphs):
        return self

    def predict(self, graphs):
        return np.asarray([graph.graph["graph_id"] in self.accepted_ids for graph in graphs], dtype=bool)

    def number_of_violations(self, graphs):
        return np.asarray(
            [0 if graph.graph["graph_id"] in self.accepted_ids else 1 for graph in graphs],
            dtype=int,
        )

    def violating_edge_sets(self, graphs):
        return [list(self._violating_edge_sets) for _ in graphs]


def _oracle_test_graph(graph_id):
    graph = nx.Graph()
    graph.graph["graph_id"] = graph_id
    graph.add_node(0, label="C")
    graph.add_node(1, label="C")
    graph.add_edge(0, 1, label="single")
    return graph


def test_demo_feasibility_estimator_mask_selects_active_levels():
    graphs = [_oracle_test_graph(0), _oracle_test_graph(1)]
    estimator = FeasibilityEstimator(
        [
            _MaskAwareFeasibilityLeaf({0}),
            _MaskAwareFeasibilityLeaf({1}),
        ],
        estimator_names=["first", "second"],
    )

    assert estimator.predict(graphs).tolist() == [False, False]
    assert estimator.predict(graphs, estimator_mask=[1, 0]).tolist() == [True, False]
    assert estimator.predict(graphs, estimator_mask=[0, 1]).tolist() == [False, True]

    estimator.set_active_mask([0, 1])
    assert estimator.predict(graphs).tolist() == [False, True]


def test_demo_feasibility_estimator_caps_oracle_cuts_per_level():
    graph = _oracle_test_graph(0)
    estimator = FeasibilityEstimator(
        [
            _MaskAwareFeasibilityLeaf(
                {0},
                violating_edge_sets=[
                    frozenset({(0, 1)}),
                    frozenset({(0, 2)}),
                ],
            ),
            _MaskAwareFeasibilityLeaf(
                {0},
                violating_edge_sets=[
                    frozenset({(1, 2)}),
                    frozenset({(2, 3)}),
                ],
            ),
        ],
        estimator_names=["small", "large"],
    )

    estimator.set_oracle_cut_budget_per_estimator([1, 0])
    assert estimator.violating_edge_sets([graph]) == [[frozenset({(0, 1)})]]

    estimator.set_oracle_cut_budget_schedule(4, schedule="exponential", minimum_budget=1, decay=0.5)
    assert estimator.oracle_cut_budget_per_estimator.tolist() == [4, 2]
