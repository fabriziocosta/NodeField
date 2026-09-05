from pathlib import Path

import networkx as nx
import pandas as pd

from conditional_node_field_graph_generator.extensions.demo.recurrent_experiments import (
    experiment_conditions,
    load_config,
    primary_metrics,
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
