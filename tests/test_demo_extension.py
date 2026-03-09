from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from conditional_node_field_graph_generator.extensions.demo.pipeline import (
    build_zinc_dataset,
    fit_graph_generator,
    prepare_experiment,
    sample_hyperparameter_configuration,
    score_graph_generator_feasible_rate,
)
from conditional_node_field_graph_generator.extensions.demo.visualization import (
    _temporary_decoder_n_jobs,
    compare_real_vs_generated,
    infer_display_mode,
    offset_neg_graphs,
    select_pos_neg,
)


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
        return Path(dataset_dir) / "zinc.csv"

    def fake_build(dataset_dir, csv_path):
        calls["build"] = {"dataset_dir": Path(dataset_dir), "csv_path": Path(csv_path)}
        return {"node_counts": [10, 11], "total_graphs": 200_000}

    def fake_load(dataset_dir, max_molecules, min_node_count, max_node_count):
        calls["load"] = {
            "dataset_dir": Path(dataset_dir),
            "max_molecules": max_molecules,
            "min_node_count": min_node_count,
            "max_node_count": max_node_count,
        }
        graphs = [f"g{idx}" for idx in range(40)]
        metadata = pd.DataFrame({"zinc_id": [f"z{idx}" for idx in range(40)]})
        return graphs, metadata

    monkeypatch.setattr(
        "conditional_node_field_graph_generator.extensions.demo.pipeline.download_zinc_dataset",
        fake_download,
    )
    monkeypatch.setattr(
        "conditional_node_field_graph_generator.extensions.demo.pipeline.build_zinc_graph_corpus",
        fake_build,
    )
    monkeypatch.setattr(
        "conditional_node_field_graph_generator.extensions.demo.pipeline.load_zinc_graph_dataset",
        fake_load,
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
    assert manifest == {"node_counts": [10, 11], "total_graphs": 200_000}
    assert calls["load"] == {
        "dataset_dir": tmp_path.resolve(),
        "max_molecules": 200000,
        "min_node_count": 10,
        "max_node_count": 12,
    }
    assert graphs == build_zinc_dataset(
        dataset_dir=tmp_path,
        num_examples=25,
        min_size=10,
        max_size=12,
        random_state=7,
    )[0]


def test_fit_graph_generator_rejects_conflicting_resume_arguments():
    recorder = type("_Recorder", (), {"fit": lambda *args, **kwargs: None})()

    with pytest.raises(ValueError, match="Provide either ckpt_path or resume_latest_checkpoint"):
        fit_graph_generator(
            recorder,
            train_graphs=["g1"],
            ckpt_path="/tmp/a.ckpt",
            resume_latest_checkpoint=True,
        )


def test_fit_graph_generator_falls_back_when_latest_checkpoint_is_incompatible(tmp_path, capsys):
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

    result = fit_graph_generator(
        recorder,
        train_graphs=["g1", "g2"],
        targets=[1, 0],
        resume_latest_checkpoint=True,
        checkpoint_root=checkpoint_root,
    )

    assert result is recorder
    assert recorder.calls == [
        {"graphs": ["g1", "g2"], "targets": [1, 0], "ckpt_path": str(checkpoint_path.resolve())},
        {"graphs": ["g1", "g2"], "targets": [1, 0], "ckpt_path": None},
    ]
    output = capsys.readouterr().out
    assert "Latest checkpoint is incompatible" in output


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
        verbose=True,
    )

    assert result == {"score": 0.25}
    assert calls["kwargs"] == {
        "n_samples": 2,
        "max_feasibility_attempts": 4,
        "feasibility_candidates_per_attempt": 3,
        "interpolate_between_n_samples": None,
        "desired_target": None,
        "guidance_scale": 1.0,
        "verbose": True,
    }


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
