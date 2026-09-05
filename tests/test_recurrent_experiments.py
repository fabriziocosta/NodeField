from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from conditional_node_field_graph_generator.extensions.demo.recurrent_experiments import (
    experiment_conditions,
    load_config,
    primary_metrics,
    sampled_steps,
    stabilization_step,
    summarize_results,
)


def test_experiment_config_and_matrix():
    config = load_config("recurrent_energy_annealed")
    assert config["model"]["recurrent_training_steps"] == 8
    smoke = list(experiment_conditions(8, smoke=True, config=config["experiment"]))
    full = list(experiment_conditions(8, smoke=False, config=config["experiment"]))
    assert len(smoke) == 4 and len(full) > len(smoke)
    assert any(row[0] == "reset_both_every_step" for row in full)


def test_stabilization_requires_consecutive_observations():
    records = [
        dict(hidden_delta_norm=h, prediction_delta=0.01, score_norm=0.1)
        for h in [0.01, 1.0, 0.01, 0.01, 0.01, 0.1]
    ]
    assert (
        stabilization_step(records, hidden_threshold=0.1, prediction_threshold=0.1, consecutive=3)
        == 5
    )


def test_failure_remains_primary_metric_zero():
    result = primary_metrics(None, nx.path_graph(3), None)
    assert result["feasible_condition_match"] is False and result["decoder_success"] is False


def test_learned_feasibility_estimator_does_not_gate_structural_metric(monkeypatch):
    from types import SimpleNamespace

    from conditional_node_field_graph_generator.extensions.demo import artificial_conditioning

    graph = nx.path_graph(2)
    stats = {
        "total_nodes": 2,
        "total_edges": 1,
        "cycle_count": 0,
        "cycle_sizes": (),
        "path_length": 2,
        "path_component_sizes": (2,),
        "ray_count": 0,
        "ray_sizes": (),
        "star_hub_count": 0,
        "cycles_are_valid": True,
        "path_is_valid": True,
        "rays_are_valid": True,
    }
    monkeypatch.setattr(artificial_conditioning, "artificial_graph_stats", lambda *_: stats)
    generator = SimpleNamespace(
        feasibility_estimator=SimpleNamespace(number_of_violations=lambda _: [99])
    )

    result = primary_metrics(graph, graph, generator)
    assert result["valid"] is True
    assert result["feasible_condition_match"] is True
    assert np.isnan(result["feasibility_violations"])
    diagnostic = primary_metrics(
        graph, graph, generator, include_feasibility_diagnostics=True
    )
    assert diagnostic["feasibility_violations"] == 99


def test_sampled_steps_always_include_full_budget():
    assert sampled_steps(16, 4) == [1, 5, 9, 13, 16]
    assert sampled_steps(16, 100) == [1, 16]


def test_statistics_pair_examples_then_seeds(tmp_path):
    rows = []
    for seed in [0, 1]:
        for model, quality in [("baseline", 0), ("recurrent_energy_annealed", 1)]:
            for example in range(3):
                rows.append(
                    dict(
                        seed=seed,
                        model=model,
                        checkpoint="best",
                        training_schedule="annealed",
                        K_test=8,
                        intervention="normal",
                        inference_noise="none",
                        example_id=example,
                        feasible_condition_match=quality,
                        valid=quality,
                        decoder_success=1,
                        node_count_accuracy=1,
                        edge_count_accuracy=1,
                        condition_error=0,
                        generation_seconds=0.1,
                        decode_seconds=0.1,
                    )
                )
    pd.DataFrame(rows).to_csv(tmp_path / "results.csv", index=False)
    summary = summarize_results(tmp_path)
    assert set(summary.seeds) == {2}
    paired = pd.read_csv(tmp_path / "paired_effects.csv")
    assert paired.iloc[0].paired_effect == 1
    assert paired.iloc[0].ci_low == 1


def test_notebook_is_thin_and_valid():
    import nbformat

    path = Path(__file__).parents[1] / "notebooks/recurrent_energy_nodefield_ablation.ipynb"
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    source = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "class " not in source and "def " not in source
    assert "RUN_FULL = False" in source
    for c in nb.cells:
        if c.cell_type == "code":
            compile(c.source, str(path), "exec")


def test_missing_star_component_is_a_metric_outcome():
    from conditional_node_field_graph_generator.extensions.demo.artificial_conditioning import (
        artificial_graph_stats,
    )

    graph = nx.path_graph(3)
    nx.set_node_attributes(graph, 0, "label")
    graph.graph["metadata"] = {
        "node_alphabets_by_component": {"cycle": [1], "path": [0], "star": [2]}
    }
    stats = artificial_graph_stats(graph)
    assert stats["ray_count"] == 0 and stats["star_hub_count"] == 0


def test_anytime_driver_uses_validation_and_reports_test_only(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import torch

    from conditional_node_field_graph_generator.extensions.demo import (
        recurrent_experiments as experiment_module,
    )
    from conditional_node_field_graph_generator.recurrent_diagnostics import (
        RecurrentNodeFieldTrajectory,
    )

    class Conditions:
        def take(self, indices):
            return indices

    class Owner:
        model = SimpleNamespace(langevin_noise_scale=0.0)

        def predict_recurrent(self, condition, total_steps, return_trajectory):
            trace = RecurrentNodeFieldTrajectory()
            trace.diagnostics = [
                dict(hidden_delta_norm=0.01, prediction_delta=0.01, score_norm=0.01)
                for _ in range(total_steps)
            ]
            return torch.zeros(1), trace

    experiment = experiment_module.RecurrentExperiment.__new__(
        experiment_module.RecurrentExperiment
    )
    experiment.models = {
        (0, "recurrent_energy_annealed"): SimpleNamespace(conditional_node_generator_model=Owner())
    }
    experiment.config = {
        "experiment": {
            "depths": [4],
            "stability_consecutive_steps": 2,
            "anytime_decoder_check_stride": 2,
        }
    }
    experiment.splits = {"validation": [0], "test": [1]}
    experiment.conditions = {"validation": Conditions(), "test": Conditions()}
    experiment.graphs = [nx.path_graph(2), nx.path_graph(2)]
    experiment.run_dir = tmp_path
    monkeypatch.setattr(experiment_module, "_readout_graph", lambda *args: nx.path_graph(2))
    monkeypatch.setattr(
        experiment_module, "primary_metrics", lambda *args: {"feasible_condition_match": True}
    )
    frame = experiment.evaluate_anytime()
    assert set(frame.split) == {"validation", "test"}
    assert (tmp_path / "anytime_thresholds.json").exists()
    assert (tmp_path / "anytime_summary.csv").exists()
    assert frame.quality_loss.eq(0).all()


def test_factory_exposes_and_forwards_every_recurrent_setting():
    import inspect

    from conditional_node_field_graph_generator.extensions.demo.pipeline import (
        build_graph_generator,
    )

    options = dict(
        node_field_mode="recurrent_energy",
        recurrent_hidden_dimension=11,
        recurrent_training_steps=3,
        recurrent_detach_interval=2,
        recurrent_update_scale=0.7,
        recurrent_initial_state="zeros",
        recurrent_state_normalization=False,
        recurrent_corruption_schedule="constant",
        recurrent_sigma_min=0.03,
        recurrent_sigma_max=0.4,
        recurrent_supervise_all_steps=False,
        recurrent_loss_discount=0.6,
    )
    assert options.keys() <= inspect.signature(build_graph_generator).parameters.keys()
    generator = build_graph_generator(
        **options,
        verbose=0,
        latent_embedding_dimension=8,
        transformer_attention_head_count=2,
        use_feasibility_filtering=False,
        feasibility_oracle_candidates_per_attempt=0,
    )
    owner = generator.conditional_node_generator_model
    for key, value in options.items():
        assert getattr(owner, key) == value
    assert (
        inspect.signature(build_graph_generator).parameters["node_field_mode"].default == "baseline"
    )


def test_model_building_notebooks_default_to_recurrence():
    import ast
    import inspect
    import json

    from conditional_node_field_graph_generator.extensions.demo.pipeline import (
        build_graph_generator,
    )

    root = Path(__file__).parents[1]
    count = 0
    required = {
        k for k in inspect.signature(build_graph_generator).parameters if k.startswith("recurrent_")
    } | {"node_field_mode"}
    for path in (root / "notebooks").rglob("*.ipynb"):
        if "datasets" in path.parts:
            continue
        notebook = json.loads(path.read_text())
        source = "\n".join(
            "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"
        )
        if "build_graph_generator(" not in source:
            continue
        source = "\n".join(line for line in source.splitlines() if not line.startswith(("%", "!")))
        tree = ast.parse(source)
        settings = next(
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "NODE_FIELD_CONFIG"
                for target in node.targets
            )
        )
        options = ast.literal_eval(settings)
        assert options.keys() == required, path
        assert options["node_field_mode"] == "recurrent_energy", path
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "build_graph_generator"
            ):
                assert any(
                    kw.arg is None
                    and isinstance(kw.value, ast.Name)
                    and kw.value.id == "NODE_FIELD_CONFIG"
                    for kw in node.keywords
                ), path
        count += 1
    assert count >= 9


def test_hyperparameter_notebook_configs_use_recurrence():
    import yaml

    root = Path(__file__).parents[1]
    for name in [
        "artificial_graph_hyperparameter_optimization",
        "zinc_molecule_hyperparameter_optimization",
    ]:
        config = yaml.safe_load((root / "notebooks" / "configs" / f"{name}.yaml").read_text())
        assert config["model"]["fixed"]["node_field_mode"] == "recurrent_energy"
        assert config["model"]["fixed"]["recurrent_training_steps"] == 8
