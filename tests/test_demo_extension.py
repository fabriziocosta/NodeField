import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from conditional_node_field_graph_generator import nodefield_campaign as core_nodefield_campaign
from conditional_node_field_graph_generator.extensions.demo import (
    campaign_best_model,
    campaign_search,
    nodefield_campaign,
    zinc_hyperparameter_optimization as zinc_hopt,
)
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
        oracle_add_edge_repair_budget=7,
        oracle_use_node_label_cuts=True,
        oracle_use_edge_label_cuts=True,
        sparse_supervision_mask_ratio=0.4,
        use_embedding_svd=True,
        node_embedding_svd_dimension=123,
        graph_embedding_svd_dimension=45,
    )

    assert default_generator.feasibility_oracle_candidates_per_attempt == 8
    assert default_generator.max_oracle_iterations == 10
    assert default_generator.oracle_add_edge_repair_budget == 64
    assert default_generator.max_feasibility_seconds_per_sample == pytest.approx(200.0)
    assert default_generator.max_decode_attempts_per_sample == 4
    assert default_generator.oracle_use_node_label_cuts is False
    assert default_generator.oracle_use_edge_label_cuts is False
    assert default_generator.node_graph_vectorizer.dense is False
    assert default_generator.graph_vectorizer.dense is False
    assert default_generator.use_embedding_svd is True
    assert default_generator.embedding_svd_fit_max_rows == 10_000
    assert default_generator.embedding_svd_transform_batch_size == 10_000
    assert default_generator.embedding_svd_n_iter == 2
    assert default_generator.embedding_svd_n_oversamples == 5
    assert overridden_generator.feasibility_oracle_candidates_per_attempt == 0
    assert overridden_generator.max_oracle_iterations == 3
    assert overridden_generator.oracle_add_edge_repair_budget == 7
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


def _base_zinc_hyperparameter_optimization_config():
    return {
        "experiment": {
            "name": "unit-zinc-search",
            "n_trials": 2,
            "random_state": 10,
            "verbose": 0,
        },
        "dataset": {
            "num_graphs": 4,
            "min_size": 3,
            "max_size": 6,
            "random_state": 11,
        },
        "model": {
            "fixed": {
                "batch_size": 2,
                "maximum_epochs": 1,
            },
            "search_space": {
                "trial_quality": {"type": "real", "low": 0.0, "high": 1.0},
            },
        },
        "generation": {
            "n_samples": 3,
            "feasibility_effort": 4,
            "feasibility_filter": "strict",
        },
        "outputs": {
            "artifact_subdir": "unit-zinc-search",
            "results_csv": "results.csv",
        },
    }


def test_load_zinc_hyperparameter_optimization_config_validates_sections(tmp_path):
    config_path = tmp_path / "config.yaml"
    config = _base_zinc_hyperparameter_optimization_config()
    config_path.write_text(json.dumps(config), encoding="utf-8")

    loaded = zinc_hopt.load_zinc_hyperparameter_optimization_config(config_path)

    assert loaded["experiment"]["n_trials"] == 2
    assert loaded["generation"]["feasibility_effort"] == 4
    assert loaded["generation"]["feasibility_filter"] == "strict"

    missing = dict(config)
    missing.pop("dataset")
    missing_path = tmp_path / "missing.yaml"
    missing_path.write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing config sections: dataset"):
        zinc_hopt.load_zinc_hyperparameter_optimization_config(missing_path)

    bad_type = _base_zinc_hyperparameter_optimization_config()
    bad_type["model"]["search_space"]["trial_quality"]["type"] = "choice"
    bad_path = tmp_path / "bad_type.yaml"
    bad_path.write_text(json.dumps(bad_type), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported hyperparameter type"):
        zinc_hopt.load_zinc_hyperparameter_optimization_config(bad_path)


def test_run_zinc_hyperparameter_optimization_ranks_average_violations(
    monkeypatch,
    tmp_path,
):
    calls = {
        "build_zinc_dataset": None,
        "build_graph_generator": [],
        "fit_graph_generator": [],
        "sample": [],
    }
    graphs = [nx.path_graph(3), nx.path_graph(4)]
    metadata = pd.DataFrame({"zinc_id": ["z0", "z1"]})
    sampled_params_by_call = [{"trial_quality": 5.0}, {"trial_quality": 1.0}]

    def fake_build_zinc_dataset(**kwargs):
        calls["build_zinc_dataset"] = kwargs
        return list(graphs), metadata.copy(), {"dataset_name": "fake-zinc"}

    class _FakeFeasibilityEstimator:
        def __init__(self, trial_quality):
            self.trial_quality = float(trial_quality)

        def number_of_violations(self, decoded_graphs):
            del decoded_graphs
            if self.trial_quality == 5.0:
                return np.asarray([4, 5, 6], dtype=int)
            return np.asarray([0, 1, 1], dtype=int)

    class _FakeGenerator:
        def __init__(self, kwargs):
            self.kwargs = kwargs
            self.feasibility_estimator = _FakeFeasibilityEstimator(kwargs["trial_quality"])

        def sample(self, **kwargs):
            calls["sample"].append(kwargs)
            return [nx.Graph() for _ in range(int(kwargs["n_samples"]))]

    def fake_sample_hyperparameter_configuration(search_space, random_state=None):
        assert search_space == {
            "trial_quality": {"type": "real", "low": 0.0, "high": 1.0},
        }
        assert random_state in {11, 12}
        return sampled_params_by_call.pop(0)

    def fake_build_graph_generator(**kwargs):
        calls["build_graph_generator"].append(kwargs)
        return _FakeGenerator(kwargs)

    def fake_fit_graph_generator(graph_generator, train_graphs, **kwargs):
        calls["fit_graph_generator"].append((graph_generator, train_graphs, kwargs))
        return graph_generator

    monkeypatch.setattr(zinc_hopt, "build_zinc_dataset", fake_build_zinc_dataset)
    monkeypatch.setattr(
        zinc_hopt,
        "sample_hyperparameter_configuration",
        fake_sample_hyperparameter_configuration,
    )
    monkeypatch.setattr(zinc_hopt, "build_graph_generator", fake_build_graph_generator)
    monkeypatch.setattr(zinc_hopt, "fit_graph_generator", fake_fit_graph_generator)

    config = _base_zinc_hyperparameter_optimization_config()
    config["outputs"]["artifact_root"] = str(tmp_path / "artifact")
    config["outputs"]["run_timestamp"] = "20260625_091011"
    config["outputs"]["run_id"] = "unit"

    result = zinc_hopt.run_zinc_hyperparameter_optimization(
        config,
        notebook_context={
            "ARTIFACT_ROOT": tmp_path / "artifacts",
            "NOTEBOOK_DATA_ROOT": tmp_path / "datasets",
        },
    )

    assert calls["build_zinc_dataset"] == {
        "dataset_dir": tmp_path / "datasets" / "zinc",
        "num_examples": 4,
        "min_size": 3,
        "max_size": 6,
        "random_state": 11,
    }
    assert [call["batch_size"] for call in calls["build_graph_generator"]] == [2, 2]
    assert [call["trial_quality"] for call in calls["build_graph_generator"]] == [5.0, 1.0]
    assert calls["sample"] == [
        {"n_samples": 3, "feasibility_effort": 4, "feasibility_filter": "strict"},
        {"n_samples": 3, "feasibility_effort": 4, "feasibility_filter": "strict"},
    ]
    results_df = result["results_df"]
    assert results_df["trial_id"].tolist() == [2, 1]
    assert results_df.loc[0, "average_num_violations"] == pytest.approx(2 / 3)
    assert results_df.loc[1, "average_num_violations"] == pytest.approx(5.0)
    assert results_df.loc[0, "feasible_rate"] == pytest.approx(1 / 3)
    assert results_df.loc[1, "feasible_rate"] == pytest.approx(0.0)
    assert result["best_row"]["trial_id"] == 2
    assert result["results_csv_path"].is_file()
    assert ".artifacts" not in str(result["artifact_root"])
    assert result["artifact_root"].name == "unit-zinc-search_20260625_091011_unit"
    assert result["results_csv_path"].parent.name == "metrics"


def test_campaign_patch_space_validation_and_sampling_is_deterministic():
    patch_space = {
        "model": {
            "search_space": {
                "sampling_step_size": {"type": "real", "low": 0.02, "high": 0.05},
                "batch_size": {"type": "int", "low": 2, "high": 4},
            }
        }
    }

    first = campaign_search.sample_patch_space(
        patch_space,
        n_samples=2,
        random_state=7,
        allowed_paths=["model.search_space"],
        max_leaf_count=2,
    )
    second = campaign_search.sample_patch_space(
        patch_space,
        n_samples=2,
        random_state=7,
        allowed_paths=["model.search_space"],
        max_leaf_count=2,
    )

    assert first == second
    assert len(first) == 2
    assert 0.02 <= first[0]["model"]["search_space"]["sampling_step_size"] <= 0.05
    assert 2 <= first[0]["model"]["search_space"]["batch_size"] <= 4

    with pytest.raises(ValueError, match="non-allowlisted"):
        campaign_search.validate_patch_space(
            {"outputs": {"artifact_root": {"type": "choice", "values": ["bad"]}}},
            allowed_paths=["model.search_space"],
        )


def test_campaign_apply_exact_trial_patch_preserves_search_space_specs():
    base = {
        "experiment": {"n_trials": 3},
        "model": {
            "fixed": {"batch_size": 8},
            "search_space": {
                "sampling_step_size": {"type": "real", "low": 0.01, "high": 0.1},
            },
        },
    }
    patch = {
        "model": {
            "fixed": {"batch_size": 4},
            "search_space": {"sampling_step_size": 0.025},
        }
    }

    resolved = nodefield_campaign.apply_exact_trial_patch(base, patch)

    assert resolved["model"]["fixed"]["batch_size"] == 4
    assert resolved["model"]["search_space"]["sampling_step_size"] == {
        "type": "real",
        "low": 0.025,
        "high": 0.025,
    }


def test_campaign_dry_run_uses_artifact_root_without_legacy_artifacts(tmp_path):
    campaign_config = {
        "campaign": {"id": "molecules", "domain": "molecules", "prefix": "molecules"},
        "artifacts": {"root": str(tmp_path / "artifact")},
        "logbook": {"path": str(tmp_path / "LOGBOOK.md")},
        "runner": {"config_path": str(tmp_path / "workflow.yaml")},
        "random_search": {"batch_size": 2, "random_state": 3},
        "agent": {
            "allowed_paths": ["model.search_space.sampling_step_size"],
            "max_search_leaf_count": 1,
            "default_trial_patch_space": {
                "model": {
                    "search_space": {
                        "sampling_step_size": {"type": "real", "low": 0.02, "high": 0.03}
                    }
                }
            },
        },
        "_repo_root": str(Path(__file__).resolve().parents[1]),
    }
    (tmp_path / "workflow.yaml").write_text(
        json.dumps(_base_zinc_hyperparameter_optimization_config())
    )

    result = nodefield_campaign.run_campaign_once(
        campaign_config,
        dry_run=True,
        now=pd.Timestamp("2026-06-25 09:10:11").to_pydatetime(),
        short_id="dry001",
    )

    assert result["state"]["status"] == "dry_run"
    assert result["run_dir"].name == "molecules_20260625_091011_dry001"
    assert len(result["proposal"]["sampled_patches"]) == 2
    assert ".artifacts" not in str(result["run_dir"])
    assert not result["run_dir"].exists()


def test_campaign_loads_mutable_groups_and_exact_config_proposals(tmp_path):
    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text(json.dumps(_base_zinc_hyperparameter_optimization_config()))
    campaign_path = tmp_path / "campaign.yaml"
    campaign_path.write_text(
        json.dumps(
            {
                "campaign": {"id": "molecules", "domain": "molecules", "prefix": "molecules"},
                "artifacts": {"root": str(tmp_path / "artifact")},
                "runner": {"config_path": str(workflow_path)},
                "random_search": {"batch_size": 2, "random_state": 5},
                "agent": {
                    "proposal_mode": "exact_configs",
                    "mutable_groups": ["architecture"],
                    "default_trial_configs": [
                        {
                            "model": {"fixed": {"number_of_transformer_layers": 2}},
                        },
                        {
                            "model": {"fixed": {"number_of_transformer_layers": 3}},
                        },
                    ],
                },
            }
        )
    )

    config = nodefield_campaign.load_campaign_config(campaign_path)
    result = nodefield_campaign.run_campaign_once(
        config,
        dry_run=True,
        now=pd.Timestamp("2026-06-25 09:10:11").to_pydatetime(),
        short_id="exact1",
    )

    assert config["agent"]["proposal_mode"] == "exact_configs"
    assert "dataset" not in config["agent"]["allowed_paths"]
    assert "model.fixed.number_of_transformer_layers" in config["agent"]["allowed_paths"]
    assert config["logbook"]["path"].endswith("LOGBOOK_molecules.md")
    assert result["proposal"]["sampled_patches"] == config["agent"]["default_trial_configs"]

    bad_path = tmp_path / "bad_campaign.yaml"
    bad = dict(json.loads(campaign_path.read_text()))
    bad["agent"]["default_trial_configs"] = [{"dataset": {"num_graphs": 20}}]
    bad_path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="non-allowlisted"):
        nodefield_campaign.load_campaign_config(bad_path)

    bad_generation_path = tmp_path / "bad_generation_campaign.yaml"
    bad_generation = dict(json.loads(campaign_path.read_text()))
    bad_generation["agent"]["default_trial_configs"] = [{"generation": {"n_samples": 8}}]
    bad_generation_path.write_text(json.dumps(bad_generation))
    with pytest.raises(ValueError, match="non-allowlisted"):
        nodefield_campaign.load_campaign_config(bad_generation_path)


def test_builtin_campaign_configs_define_prompts_and_poll_intervals():
    repo_root = Path(__file__).resolve().parents[1]

    small = nodefield_campaign.load_campaign_config(
        repo_root / "configs" / "campaigns" / "artificial_graphs_small.yaml",
        repo_root=repo_root,
    )
    large = nodefield_campaign.load_campaign_config(
        repo_root / "configs" / "campaigns" / "artificial_graphs_large.yaml",
        repo_root=repo_root,
    )

    assert small["runner"]["poll_seconds"] == 1800
    assert large["runner"]["poll_seconds"] == 3600
    for config in (small, large):
        assert config["random_search"]["batch_size"] == 1
        proposal_prompt = Path(config["agent"]["prompts"]["proposal"])
        logbook_prompt = Path(config["agent"]["prompts"]["logbook"])
        assert proposal_prompt.name == "nodefield_campaign_proposal.md"
        assert logbook_prompt.name == "nodefield_campaign_logbook.md"
        assert proposal_prompt.is_file()
        assert logbook_prompt.is_file()


def test_campaign_logbook_upsert_replaces_existing_block(tmp_path):
    logbook_path = tmp_path / "LOGBOOK.md"

    nodefield_campaign.upsert_logbook_block(logbook_path, "run-a", "first")
    nodefield_campaign.upsert_logbook_block(logbook_path, "run-a", "second")

    text = logbook_path.read_text()
    assert "second" in text
    assert "first" not in text
    assert text.count("nodefield-campaign:run-a:begin") == 1


def _write_agent_loop_campaign(tmp_path):
    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text(json.dumps(_base_zinc_hyperparameter_optimization_config()))
    campaign_path = tmp_path / "campaign.yaml"
    campaign_path.write_text(
        json.dumps(
            {
                "campaign": {"id": "molecules", "domain": "molecules", "prefix": "molecules"},
                "artifacts": {"root": str(tmp_path / "artifact")},
                "logbook": {"path": str(tmp_path / "LOGBOOK.md")},
                "runner": {"config_path": str(workflow_path), "poll_seconds": 1},
                "random_search": {"batch_size": 1, "random_state": 5},
                "agent": {
                    "allowed_paths": ["model.search_space.trial_quality"],
                    "max_search_leaf_count": 1,
                    "default_trial_patch_space": {
                        "model": {
                            "search_space": {
                                "trial_quality": {"type": "real", "low": 0.0, "high": 1.0}
                            }
                        }
                    },
                },
            }
        )
    )
    return campaign_path


def test_agent_decision_schema_and_parser_validate_strict_contract():
    schema = nodefield_campaign.campaign_decision_text_format()

    assert schema["format"]["strict"] is True
    assert schema["format"]["schema"]["additionalProperties"] is False
    assert schema["format"]["schema"]["properties"]["campaign_patch"]["type"] == "string"

    decision = nodefield_campaign.parse_agent_campaign_decision(
        json.dumps(
            {
                "decision": "terminate_run_and_propose_trial",
                "reason": "Narrow the range after the latest result.",
                "logbook_markdown": "### analysis\n\nUse a smaller range.",
                "campaign_patch": json.dumps(
                    {
                        "agent": {
                            "default_trial_patch_space": {
                                "model": {
                                    "search_space": {
                                        "trial_quality": {
                                            "type": "real",
                                            "low": 0.1,
                                            "high": 0.2,
                                        }
                                    }
                                }
                            }
                        }
                    }
                ),
            }
        )
    )

    assert decision.decision == "terminate_run_and_propose_trial"
    assert decision.campaign_patch["agent"]["default_trial_patch_space"]["model"][
        "search_space"
    ]["trial_quality"]["low"] == 0.1

    with pytest.raises(ValueError, match="Unsupported agent decision"):
        nodefield_campaign.parse_agent_campaign_decision(
            json.dumps(
                {
                    "decision": "bad",
                    "reason": "x",
                    "logbook_markdown": "",
                    "campaign_patch": "{}",
                }
            )
        )
    with pytest.raises(json.JSONDecodeError):
        nodefield_campaign.parse_agent_campaign_decision(
            json.dumps(
                {
                    "decision": "no_action",
                    "reason": "x",
                    "logbook_markdown": "",
                    "campaign_patch": "{bad",
                }
            )
        )
    with pytest.raises(ValueError, match="campaign_patch must decode"):
        nodefield_campaign.parse_agent_campaign_decision(
            json.dumps(
                {
                    "decision": "no_action",
                    "reason": "x",
                    "logbook_markdown": "",
                    "campaign_patch": "[]",
                }
            )
        )


def test_agent_campaign_patch_validation_accepts_only_agent_fields(tmp_path):
    campaign_path = _write_agent_loop_campaign(tmp_path)
    config = nodefield_campaign.load_campaign_config(campaign_path)

    patched = nodefield_campaign.apply_campaign_patch(
        config,
        {
            "agent": {
                "reason": "Tighten the promising range.",
                "default_trial_patch_space": {
                    "model": {
                        "search_space": {
                            "trial_quality": {"type": "real", "low": 0.1, "high": 0.2}
                        }
                    }
                },
            }
        },
    )

    assert patched["agent"]["reason"] == "Tighten the promising range."
    assert patched["agent"]["default_trial_patch_space"]["model"]["search_space"][
        "trial_quality"
    ]["high"] == 0.2

    with pytest.raises(ValueError, match="non-allowlisted"):
        nodefield_campaign.apply_campaign_patch(config, {"dataset": {"num_graphs": 10}})
    with pytest.raises(ValueError, match="non-allowlisted"):
        nodefield_campaign.apply_campaign_patch(config, {"generation": {"n_samples": 8}})
    with pytest.raises(ValueError, match="non-allowlisted"):
        nodefield_campaign.apply_campaign_patch(config, {"runner": {"poll_seconds": 5}})


def test_campaign_loop_once_launches_child_without_openai(monkeypatch, tmp_path):
    campaign_path = _write_agent_loop_campaign(tmp_path)
    config = nodefield_campaign.load_campaign_config(campaign_path)
    launches = []

    def fake_launch(config, *, campaign_name, device, run_timestamp=None, run_id=None):
        del run_timestamp, run_id
        state = {
            "campaign": "molecules",
            "status": "running",
            "phase": "mini_batch",
            "run_dir": str(tmp_path / "artifact" / "molecules" / "molecules_fake"),
            "child_log_path": str(tmp_path / "artifact" / "molecules" / "logs" / "fake.log"),
            "poll_seconds": 1,
        }
        launches.append({"campaign_name": campaign_name, "device": device})
        core_nodefield_campaign._write_json(
            core_nodefield_campaign._campaign_state_path(config),
            state,
        )
        return state

    def fail_decision(*args, **kwargs):
        raise AssertionError("OpenAI should not be called before a mini-batch completes")

    monkeypatch.setattr(core_nodefield_campaign, "_launch_mini_batch_child", fake_launch)
    monkeypatch.setattr(
        core_nodefield_campaign,
        "request_agent_campaign_decision",
        fail_decision,
    )

    result = nodefield_campaign.run_campaign_loop(
        config,
        campaign_name="molecules",
        once=True,
        sleep_fn=lambda seconds: None,
    )

    assert result["state"]["status"] == "running"
    assert launches == [{"campaign_name": "molecules", "device": "cpu"}]


def test_campaign_child_process_log_lives_inside_run_directory(tmp_path):
    campaign_path = _write_agent_loop_campaign(tmp_path)
    config = nodefield_campaign.load_campaign_config(campaign_path)
    run_dir = tmp_path / "artifact" / "molecules" / "molecules_20260625_091011_child"

    assert core_nodefield_campaign._child_log_path(config, run_dir) == (
        run_dir / "logs" / "mini_batch.log"
    )


def test_campaign_loop_restarts_after_stale_termination_request(monkeypatch, tmp_path):
    campaign_path = _write_agent_loop_campaign(tmp_path)
    config = nodefield_campaign.load_campaign_config(campaign_path)
    old_run_dir = tmp_path / "artifact" / "molecules" / "molecules_20260625_091011_old"
    old_run_dir.mkdir(parents=True)
    core_nodefield_campaign._write_json(
        core_nodefield_campaign._campaign_state_path(config),
        {
            "campaign": "molecules",
            "status": "termination_requested",
            "phase": "mini_batch",
            "run_dir": str(old_run_dir),
            "pid": 999999,
            "poll_seconds": 1,
        },
    )
    launches = []

    def fake_launch(config, *, campaign_name, device, run_timestamp=None, run_id=None):
        del run_timestamp, run_id
        state = {
            "campaign": "molecules",
            "status": "running",
            "phase": "mini_batch",
            "run_dir": str(tmp_path / "artifact" / "molecules" / "molecules_new"),
            "child_log_path": str(
                tmp_path / "artifact" / "molecules" / "molecules_new" / "logs" / "mini_batch.log"
            ),
            "poll_seconds": 1,
        }
        launches.append({"campaign_name": campaign_name, "device": device})
        core_nodefield_campaign._write_json(
            core_nodefield_campaign._campaign_state_path(config),
            state,
        )
        return state

    monkeypatch.setattr(core_nodefield_campaign, "_launch_mini_batch_child", fake_launch)

    result = nodefield_campaign.run_campaign_loop(
        config,
        campaign_name="molecules",
        once=True,
        sleep_fn=lambda seconds: None,
    )

    assert result["state"]["status"] == "running"
    assert result["state"]["run_dir"].endswith("molecules_new")
    assert launches == [{"campaign_name": "molecules", "device": "cpu"}]


def _write_campaign_best_trial_fixture(tmp_path):
    repo_root = tmp_path
    (repo_root / "conditional_node_field_graph_generator").mkdir()
    domain_root = repo_root / "artifact" / "artificial_graphs"
    state_path = domain_root / "artificial_graphs_small_campaign_state.json"
    active_run = domain_root / "artificial_graphs_small_20260626_010203_active"
    older_run = domain_root / "artificial_graphs_small_20260625_010203_old"
    for run_dir, values in (
        (active_run, [3.0, 1.0]),
        (older_run, [0.5]),
    ):
        for index, average in enumerate(values, start=1):
            trial_dir = run_dir / "trials" / f"trial_{index:03d}"
            ckpt_dir = trial_dir / "trials" / "trial_001" / "checkpoints" / "model"
            ckpt_dir.mkdir(parents=True)
            (ckpt_dir / f"best-00{index}-{average:.4f}.ckpt").write_text("checkpoint")
            (trial_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "campaign_trial_id": index,
                        "average_num_violations": average,
                        "median_num_violations": average,
                        "feasible_rate": 0.0 if average else 1.0,
                    }
                )
            )
            (trial_dir / "config.yaml").write_text(
                json.dumps(
                    {
                        "experiment": {"random_state": 7, "verbose": 1},
                        "dataset": {
                            "num_graphs": 2,
                            "cycle_length": 3,
                            "path_length": 2,
                            "num_rays": 1,
                            "ray_length": 1,
                        },
                        "model": {"fixed": {}, "search_space": {"x": {"type": "real", "low": 1, "high": 1}}},
                        "generation": {
                            "n_samples": 4,
                            "feasibility_effort": 2,
                            "feasibility_filter": "none",
                        },
                        "outputs": {"artifact_subdir": "artificial_graphs"},
                    }
                )
            )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "campaign": "artificial_graphs",
                "prefix": "artificial_graphs_small",
                "status": "running",
                "run_dir": str(active_run),
            }
        )
    )
    return repo_root, state_path


def test_campaign_best_model_auto_selects_latest_active_campaign_and_ranks_trials(tmp_path):
    repo_root, state_path = _write_campaign_best_trial_fixture(tmp_path)

    found_state_path, found_state = campaign_best_model.find_latest_campaign_state(repo_root)
    ranking = campaign_best_model.collect_campaign_trial_results(repo_root=repo_root)
    selection = campaign_best_model.select_best_campaign_trial(repo_root=repo_root)

    assert found_state_path == state_path
    assert found_state["status"] == "running"
    assert ranking.iloc[0]["average_num_violations"] == 0.5
    assert ranking.iloc[0]["run_name"] == "artificial_graphs_small_20260625_010203_old"
    assert selection.metrics["average_num_violations"] == 0.5
    assert selection.checkpoint_path.name.startswith("best-")


def test_sample_from_best_campaign_trial_forwards_notebook_sampling_overrides(
    monkeypatch,
    tmp_path,
):
    repo_root, _state_path = _write_campaign_best_trial_fixture(tmp_path)
    calls = {}

    class _FakeEstimator:
        def number_of_violations(self, graphs):
            return [0 for _graph in graphs]

    class _FakeGenerator:
        feasibility_estimator = _FakeEstimator()

        def sample(self, **kwargs):
            calls["sample"] = kwargs
            return ["g1", "g2", "g3"]

    monkeypatch.setattr(
        campaign_best_model,
        "load_campaign_trial_generator",
        lambda selection, *, notebook_context, device="cpu": _FakeGenerator(),
    )

    result = campaign_best_model.sample_from_best_campaign_trial(
        repo_root=repo_root,
        notebook_context={"NOTEBOOK_DATA_ROOT": tmp_path / "datasets"},
        n_samples=3,
        feasibility_effort=4,
        feasibility_filter="strict",
        device="cpu",
    )

    assert calls["sample"] == {
        "n_samples": 3,
        "feasibility_effort": 4,
        "feasibility_filter": "strict",
    }
    assert result["sample_summary"]["returned_samples"] == 3
    assert result["sample_summary"]["average_num_violations"] == 0.0


def test_campaign_loop_completed_minibatch_patches_config_and_relaunches(
    monkeypatch,
    tmp_path,
):
    campaign_path = _write_agent_loop_campaign(tmp_path)
    config = nodefield_campaign.load_campaign_config(campaign_path)
    run_dir = tmp_path / "artifact" / "molecules" / "molecules_20260625_091011_done"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics").mkdir()
    (run_dir / "state.json").write_text(
        json.dumps({"status": "completed", "latest_metrics": {"average_num_violations": 0.5}})
    )
    core_nodefield_campaign._write_json(
        core_nodefield_campaign._campaign_state_path(config),
        {
            "campaign": "molecules",
            "status": "running",
            "phase": "mini_batch",
            "run_dir": str(run_dir),
            "pid": 999999,
            "poll_seconds": 1,
        },
    )
    launched_configs = []

    def fake_decision(config, campaign_state, result, *, client=None):
        del config, campaign_state, result, client
        return nodefield_campaign.AgentCampaignDecision(
            decision="propose_trial",
            reason="Narrow quality after completed run.",
            logbook_markdown="### molecules\n\nNarrow quality.",
            campaign_patch={
                "agent": {
                    "reason": "Narrow quality.",
                    "default_trial_patch_space": {
                        "model": {
                            "search_space": {
                                "trial_quality": {"type": "real", "low": 0.2, "high": 0.3}
                            }
                        }
                    },
                }
            },
        )

    def fake_launch(config, *, campaign_name, device, run_timestamp=None, run_id=None):
        del campaign_name, device, run_timestamp, run_id
        launched_configs.append(config)
        state = {
            "campaign": "molecules",
            "status": "running",
            "phase": "mini_batch",
            "run_dir": str(tmp_path / "artifact" / "molecules" / "molecules_next"),
            "child_log_path": str(tmp_path / "artifact" / "molecules" / "logs" / "next.log"),
            "poll_seconds": 1,
        }
        core_nodefield_campaign._write_json(
            core_nodefield_campaign._campaign_state_path(config),
            state,
        )
        return state

    monkeypatch.setattr(core_nodefield_campaign, "request_agent_campaign_decision", fake_decision)
    monkeypatch.setattr(core_nodefield_campaign, "_launch_mini_batch_child", fake_launch)

    result = nodefield_campaign.run_campaign_loop(
        config,
        campaign_name="molecules",
        once=True,
        sleep_fn=lambda seconds: None,
    )

    assert result["state"]["status"] == "running"
    assert (run_dir / "agent_decision.json").is_file()
    assert (tmp_path / "artifact" / "molecules" / "molecules_agent_decisions.jsonl").is_file()
    patched = nodefield_campaign.load_campaign_config(campaign_path)
    assert patched["agent"]["default_trial_patch_space"]["model"]["search_space"][
        "trial_quality"
    ] == {"type": "real", "low": 0.2, "high": 0.3}
    assert "Narrow quality." in (tmp_path / "LOGBOOK.md").read_text()
    assert "agent_decision.json" in (tmp_path / "LOGBOOK.md").read_text()
    assert "Files to inspect" in (tmp_path / "LOGBOOK.md").read_text()
    assert launched_configs


def test_campaign_loop_can_semantically_stop_active_run_and_relaunch(
    monkeypatch,
    tmp_path,
):
    campaign_path = _write_agent_loop_campaign(tmp_path)
    config = nodefield_campaign.load_campaign_config(campaign_path)
    run_dir = tmp_path / "artifact" / "molecules" / "molecules_20260625_091011_active"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "status": "running",
                "latest_metrics": {
                    "campaign_trial_id": 1,
                    "average_num_violations": 99.0,
                    "median_num_violations": 99.0,
                    "feasible_rate": 0.0,
                },
                "queued_trials": [{"trial_id": 1, "status": "running"}],
            }
        )
    )
    core_nodefield_campaign._write_json(
        core_nodefield_campaign._campaign_state_path(config),
        {
            "campaign": "molecules",
            "status": "running",
            "phase": "mini_batch",
            "run_dir": str(run_dir),
            "pid": 12345,
            "poll_seconds": 1,
        },
    )
    terminated = []
    launched_configs = []

    def fake_decision(config, campaign_state, result, *, client=None):
        del config, campaign_state, client
        assert result["status"] == "running"
        assert result["state"]["latest_metrics"]["average_num_violations"] == 99.0
        return nodefield_campaign.AgentCampaignDecision(
            decision="terminate_run_and_propose_trial",
            reason="Partial metrics show this run is clearly uninformative.",
            logbook_markdown=(
                "### molecules\n\n"
                "This run is not improving and should be stopped early.\n\n"
                "| Trial | Avg violations |\n| --- | ---: |\n| trial_001 | 99.0 |\n\n"
                "Next, tighten the trial-quality range around the previous stable region."
            ),
            campaign_patch={
                "agent": {
                    "reason": "Recover from a bad active run.",
                    "default_trial_patch_space": {
                        "model": {
                            "search_space": {
                                "trial_quality": {"type": "real", "low": 0.2, "high": 0.3}
                            }
                        }
                    },
                }
            },
        )

    def fake_launch(config, *, campaign_name, device, run_timestamp=None, run_id=None):
        del campaign_name, device, run_timestamp, run_id
        launched_configs.append(config)
        state = {
            "campaign": "molecules",
            "status": "running",
            "phase": "mini_batch",
            "run_dir": str(tmp_path / "artifact" / "molecules" / "molecules_next"),
            "child_log_path": str(
                tmp_path / "artifact" / "molecules" / "molecules_next" / "logs" / "mini_batch.log"
            ),
            "poll_seconds": 1,
        }
        core_nodefield_campaign._write_json(
            core_nodefield_campaign._campaign_state_path(config),
            state,
        )
        return state

    monkeypatch.setattr(core_nodefield_campaign, "_is_process_running", lambda pid: pid == 12345)
    monkeypatch.setattr(core_nodefield_campaign, "_terminate_process_group", terminated.append)
    monkeypatch.setattr(core_nodefield_campaign, "request_agent_campaign_decision", fake_decision)
    monkeypatch.setattr(core_nodefield_campaign, "_launch_mini_batch_child", fake_launch)

    result = nodefield_campaign.run_campaign_loop(
        config,
        campaign_name="molecules",
        once=True,
        sleep_fn=lambda seconds: None,
    )

    assert terminated == [12345]
    assert result["state"]["status"] == "running"
    assert result["state"]["run_dir"].endswith("molecules_next")
    assert launched_configs
    run_state = json.loads((run_dir / "state.json").read_text())
    assert run_state["status"] == "terminated_by_agent"
    assert (run_dir / "agent_decision.json").is_file()
    logbook_text = (tmp_path / "LOGBOOK.md").read_text()
    assert "Files to inspect" in logbook_text
    assert "agent_decision.json" in logbook_text


def test_campaign_proposal_records_latest_result_context(tmp_path):
    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text(json.dumps(_base_zinc_hyperparameter_optimization_config()))
    campaign_path = tmp_path / "campaign.yaml"
    campaign_path.write_text(
        json.dumps(
            {
                "campaign": {"id": "molecules", "domain": "molecules", "prefix": "molecules"},
                "artifacts": {"root": str(tmp_path / "artifact")},
                "logbook": {"path": str(tmp_path / "LOGBOOK.md")},
                "runner": {"config_path": str(workflow_path)},
                "random_search": {"batch_size": 1, "random_state": 5},
                "agent": {
                    "reason": "Narrow loss-weight ranges after reviewing the latest result.",
                    "allowed_paths": ["model.search_space.trial_quality"],
                    "max_search_leaf_count": 1,
                    "default_trial_patch_space": {
                        "model": {
                            "search_space": {
                                "trial_quality": {"type": "real", "low": 0.0, "high": 1.0}
                            }
                        }
                    },
                },
            }
        )
    )
    previous_run = tmp_path / "artifact" / "molecules" / "molecules_20260624_010203_prev"
    previous_run.mkdir(parents=True)
    (previous_run / "metrics").mkdir()
    (previous_run / "metrics" / "summary.csv").write_text("average_num_violations\n0.5\n")
    (previous_run / "proposal.json").write_text("{}")
    (previous_run / "state.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "latest_metrics": {
                    "average_num_violations": 0.5,
                    "feasible_rate": 0.25,
                },
            }
        )
    )

    config = nodefield_campaign.load_campaign_config(campaign_path)
    result = nodefield_campaign.run_campaign_once(
        config,
        dry_run=True,
        now=pd.Timestamp("2026-06-25 09:10:11").to_pydatetime(),
        short_id="next",
    )

    proposal = result["proposal"]
    assert Path(proposal["prompt_paths"]["proposal"]).name == "nodefield_campaign_proposal.md"
    assert Path(proposal["prompt_paths"]["logbook"]).name == "nodefield_campaign_logbook.md"
    assert proposal["previous_result"]["status"] == "completed"
    assert proposal["previous_result"]["latest_metrics"] == {
        "average_num_violations": 0.5,
        "feasible_rate": 0.25,
    }
    assert proposal["previous_result"]["summary_csv_path"] == str(
        previous_run / "metrics" / "summary.csv"
    )
    assert "Latest run" in proposal["reason"]
    assert "average_num_violations=0.5" in proposal["reason"]
    assert "Narrow loss-weight ranges" in proposal["reason"]


def test_campaign_mini_batch_execution_writes_state_and_metrics(monkeypatch, tmp_path):
    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text(json.dumps(_base_zinc_hyperparameter_optimization_config()))
    campaign_path = tmp_path / "campaign.yaml"
    campaign_path.write_text(
        json.dumps(
            {
                "campaign": {"id": "molecules", "domain": "molecules", "prefix": "molecules"},
                "artifacts": {"root": str(tmp_path / "artifact")},
                "logbook": {"path": str(tmp_path / "LOGBOOK.md")},
                "runner": {"config_path": str(workflow_path)},
                "dataset": {"num_graphs": 8, "max_size": 7},
                "generation": {"n_samples": 5, "feasibility_effort": 2},
                "random_search": {"batch_size": 2, "random_state": 5},
                "agent": {
                    "mutable_groups": ["architecture"],
                    "allowed_paths": ["model.search_space.trial_quality"],
                    "max_search_leaf_count": 2,
                    "default_trial_patch_space": {
                        "model": {
                            "fixed": {
                                "number_of_transformer_layers": {
                                    "type": "int",
                                    "low": 1,
                                    "high": 2,
                                },
                            },
                            "search_space": {
                                "trial_quality": {"type": "real", "low": 0.0, "high": 1.0}
                            }
                        }
                    },
                },
            }
        )
    )
    config = nodefield_campaign.load_campaign_config(campaign_path)
    captured_configs = []

    def fake_loader(path):
        try:
            import yaml
        except ImportError:
            data = json.loads(Path(path).read_text())
        else:
            data = yaml.safe_load(Path(path).read_text())
        captured_configs.append(data)
        return data

    def fake_runner(config, *, notebook_context):
        del notebook_context
        print("runner stdout should be logged")
        print("runner stderr should be logged", file=sys.stderr)
        value = float(config["model"]["search_space"]["trial_quality"]["low"])
        class _FakeGraphGenerator:
            def export_metrics_pdf(self, output_path):
                Path(output_path).write_text("pdf placeholder")

        return {
            "best_row": {
                "trial_id": 1,
                "average_num_violations": value,
                "median_num_violations": value,
                "feasible_rate": 1.0 - value,
            },
            "results_csv_path": (
                Path(config["outputs"]["run_dir"]) / "metrics" / "trial_results.csv"
            ),
            "best_graph_generator": _FakeGraphGenerator(),
        }

    monkeypatch.setattr(
        core_nodefield_campaign,
        "_domain_runner",
        lambda domain: (fake_loader, fake_runner),
    )

    result = nodefield_campaign.run_campaign_once(
        config,
        now=pd.Timestamp("2026-06-25 09:10:11").to_pydatetime(),
        short_id="batch1",
    )

    state = json.loads((result["run_dir"] / "state.json").read_text())
    assert state["status"] == "completed"
    assert [trial["status"] for trial in state["queued_trials"]] == ["completed", "completed"]
    assert (result["run_dir"] / "proposal.json").is_file()
    assert (result["run_dir"] / "metrics" / "summary.csv").is_file()
    assert state["logs_dir"] == str(result["run_dir"] / "logs")
    assert len(state["loss_pdf_paths"]) == 2
    assert all(Path(path).is_file() for path in state["loss_pdf_paths"])
    first_log = result["run_dir"] / "logs" / "trial_001.log"
    assert "runner stdout should be logged" in first_log.read_text()
    assert "runner stderr should be logged" in first_log.read_text()
    assert "loss_pdf_path" in state["latest_metrics"]
    assert len(captured_configs) == 2
    assert all(
        item["model"]["search_space"]["trial_quality"]["low"]
        == item["model"]["search_space"]["trial_quality"]["high"]
        for item in captured_configs
    )
    assert all(item["dataset"]["num_graphs"] == 8 for item in captured_configs)
    assert all(item["dataset"]["max_size"] == 7 for item in captured_configs)
    assert all(item["generation"]["n_samples"] == 5 for item in captured_configs)
    assert all(item["generation"]["feasibility_effort"] == 2 for item in captured_configs)
    assert all(item["experiment"]["verbose"] == 2 for item in captured_configs)
    assert all(
        1 <= item["model"]["fixed"]["number_of_transformer_layers"] <= 2
        for item in captured_configs
    )
    logbook_text = (tmp_path / "LOGBOOK.md").read_text()
    assert "Files to inspect" in logbook_text
    assert "| Trial | Status | Avg violations | Median violations | Feasible rate |" in logbook_text
    assert "This run completed" in logbook_text
    assert "proposal.json" in logbook_text
    assert "metrics/summary.csv" in logbook_text
    assert "trial_001.log" in logbook_text
    assert "loss_curves.pdf" in logbook_text


def test_campaign_mini_batch_failure_marks_state_failed(monkeypatch, tmp_path):
    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text(json.dumps(_base_zinc_hyperparameter_optimization_config()))
    campaign_path = tmp_path / "campaign.yaml"
    campaign_path.write_text(
        json.dumps(
            {
                "campaign": {"id": "molecules", "domain": "molecules", "prefix": "molecules"},
                "artifacts": {"root": str(tmp_path / "artifact")},
                "logbook": {"path": str(tmp_path / "LOGBOOK.md")},
                "runner": {"config_path": str(workflow_path)},
                "random_search": {"batch_size": 2, "random_state": 5},
                "agent": {
                    "allowed_paths": ["model.search_space.trial_quality"],
                    "max_search_leaf_count": 1,
                    "default_trial_patch_space": {
                        "model": {
                            "search_space": {
                                "trial_quality": {"type": "real", "low": 0.0, "high": 1.0}
                            }
                        }
                    },
                },
            }
        )
    )
    config = nodefield_campaign.load_campaign_config(campaign_path)

    def fake_loader(path):
        try:
            import yaml
        except ImportError:
            return json.loads(Path(path).read_text())
        return yaml.safe_load(Path(path).read_text())

    def failing_runner(config, *, notebook_context):
        del config, notebook_context
        raise RuntimeError("training failed")

    monkeypatch.setattr(
        core_nodefield_campaign,
        "_domain_runner",
        lambda domain: (fake_loader, failing_runner),
    )

    with pytest.raises(RuntimeError, match="training failed"):
        nodefield_campaign.run_campaign_once(
            config,
            now=pd.Timestamp("2026-06-25 09:10:11").to_pydatetime(),
            short_id="failed",
        )

    run_dir = tmp_path / "artifact" / "molecules" / "molecules_20260625_091011_failed"
    state = json.loads((run_dir / "state.json").read_text())
    assert state["status"] == "failed"
    assert [trial["status"] for trial in state["queued_trials"]] == ["failed", "queued"]
    assert state["latest_error"] == {
        "trial_id": 1,
        "type": "RuntimeError",
        "message": "training failed",
        "log_path": str(run_dir / "logs" / "trial_001.log"),
    }
    failure_log = run_dir / "logs" / "trial_001.log"
    assert "Trial failed with exception" in failure_log.read_text()


def test_campaign_status_reads_latest_state_only(tmp_path):
    config = {
        "campaign": {"domain": "molecules", "prefix": "molecules"},
        "artifacts": {"root": str(tmp_path / "artifact")},
        "_repo_root": str(Path(__file__).resolve().parents[1]),
    }
    run_dir = tmp_path / "artifact" / "molecules" / "molecules_20260625_091011_state1"
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "status": "running",
                "queued_trials": [{"trial_id": 1, "status": "running"}],
                "latest_metrics": {"average_num_violations": 0.5},
                "latest_error": {
                    "trial_id": 1,
                    "type": "RuntimeError",
                    "message": "training failed",
                },
                "loss_pdf_paths": [str(run_dir / "metrics" / "loss_curves.pdf")],
            }
        )
    )

    status = nodefield_campaign.campaign_status(config)

    assert status["status"] == "running"
    assert status["latest_metrics"] == {"average_num_violations": 0.5}
    assert status["latest_error"] == {
        "trial_id": 1,
        "type": "RuntimeError",
        "message": "training failed",
    }
    assert status["logs_dir"] == str(run_dir / "logs")
    assert status["loss_pdf_paths"] == [str(run_dir / "metrics" / "loss_curves.pdf")]
    assert "latest_error" in nodefield_campaign.format_campaign_status(status)
    assert "loss_pdfs:" in nodefield_campaign.format_campaign_status(status)
    assert "molecules_20260625_091011_state1" in status["run_dir"]


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
