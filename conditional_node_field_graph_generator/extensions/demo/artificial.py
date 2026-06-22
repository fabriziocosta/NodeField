"""Notebook-facing artificial graph generation, visualization, and feasibility helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import networkx as nx
import pandas as pd
import yaml

from abstractgraph.operators import compose, connected_component, filter_by_node_label, merge, node, unlabel
from abstractgraph_ml.feasibility import FeasibilityEstimatorFeatureCannotExist

from ...persistence import load_graph_generator, resolve_saved_generator_dir
from ..synthetic import generate_artificial_dataset, make_artificial_graph_plotter

ARTIFICIAL_PARTS = ("cycle", "path", "branching")


def find_latest_artificial_graph_generator(
    model_dir: str | Path | None = None,
    *,
    patterns: Iterable[str] = ("artificial*.pkl", "*artificial*.pkl"),
) -> Path:
    """Return the most recently modified saved artificial generator snapshot."""
    model_root = resolve_saved_generator_dir(model_dir=model_dir)
    matches: dict[Path, float] = {}
    for pattern in patterns:
        for path in model_root.glob(pattern):
            if path.is_file() and path.suffix == ".pkl":
                matches[path.resolve()] = path.stat().st_mtime
    if not matches:
        available = sorted(path.name for path in model_root.glob("*.pkl"))
        suffix = ""
        if available:
            preview = ", ".join(available[:8])
            if len(available) > 8:
                preview += ", ..."
            suffix = f" Available saved generators: {preview}"
        raise FileNotFoundError(
            f"No saved artificial graph generator snapshots found in {model_root}."
            f"{suffix}"
        )
    return max(matches, key=matches.get)


def load_latest_artificial_graph_generator(
    model_dir: str | Path | None = None,
    *,
    patterns: Iterable[str] = ("artificial*.pkl", "*artificial*.pkl"),
):
    """Load the latest saved artificial generator and return ``(generator, path)``."""
    path = find_latest_artificial_graph_generator(model_dir=model_dir, patterns=patterns)
    return load_graph_generator(path, model_dir=model_dir), path


def build_artificial_plotter(
    *,
    node_alphabet_size: int = 3,
    node_alphabet_kind: str = "int",
    component_specific_alphabets: bool = True,
):
    """Build the standard artificial graph plotter used by NodeField demos."""
    return make_artificial_graph_plotter(
        node_alphabet_size,
        node_alphabet_kind=node_alphabet_kind,
        component_specific_alphabets=component_specific_alphabets,
    )


def summarize_graphs(graphs: Iterable[nx.Graph]) -> dict[str, list[int]]:
    """Return compact sample diagnostics for a list of generated graphs."""
    graph_list = list(graphs)
    return {
        "node_counts": [graph.number_of_nodes() for graph in graph_list],
        "edge_counts": [graph.number_of_edges() for graph in graph_list],
        "connected_components": [nx.number_connected_components(graph) for graph in graph_list],
    }


def draw_artificial_graphs(
    graphs,
    *,
    n: int | None = None,
    title: str | None = None,
    titles=None,
    n_graphs_per_line: int = 7,
    plotter=None,
):
    """Draw artificial graphs with the same notebook-facing shape as ``draw_graphs``."""
    graph_list = list(graphs or [])
    if n is not None:
        graph_list = graph_list[: int(n)]
    if not graph_list:
        print("No graphs to display.")
        return None
    if titles is None:
        titles = [f"graph {idx}" for idx in range(len(graph_list))]
    if title is not None:
        titles = [f"{title} | {item}" for item in titles]
    if plotter is None:
        plotter = build_artificial_plotter()
    return plotter(
        graph_list,
        n_cols=max(1, min(int(n_graphs_per_line), len(graph_list))),
        titles=titles,
    )


def _unlabel_base_and_mapped_subgraphs(abstract_graph, label="-"):
    """Strip labels from both the base graph and selected mapped subgraphs."""
    out_abstract_graph = unlabel(label=label)(abstract_graph)
    for _, data in out_abstract_graph.interpretation_graph.nodes(data=True):
        mapped_subgraph = data.get("mapped_subgraph")
        if mapped_subgraph is None:
            continue
        for _, node_data in mapped_subgraph.nodes(data=True):
            node_data["label"] = label
        for _, _, edge_data in mapped_subgraph.edges(data=True):
            edge_data["label"] = label
    return out_abstract_graph


def find_latest_artificial_dataset_config(root: str | Path, pattern: str = "artificial-cycle-path-star*.yaml") -> Path:
    """Return the newest artificial cycle/path/star dataset config under ``root``."""
    matches = sorted(Path(root).glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No files matching {pattern!r} under {root}.")
    return matches[0]


def load_artificial_feasibility_graphs(
    config_path: str | Path,
    *,
    n_graphs: int = 100,
):
    """Generate a small true artificial graph sample from a saved config."""
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config = dict(config)
    config["num_graphs"] = int(n_graphs)
    graphs, _ = generate_artificial_dataset(
        **config,
        save_config=False,
    )
    return graphs


def artificial_part_label_intervals(
    node_alphabet_size: int,
    *,
    node_alphabet_kind: str = "int",
    component_specific_alphabets: bool = True,
) -> dict[str, set]:
    """Return cycle/path/branching node-label sets from artificial alphabet settings."""
    if not component_specific_alphabets:
        raise ValueError("Separate cycle/path/branching estimators require component-specific node alphabets.")
    size = int(node_alphabet_size)
    if size < 1:
        raise ValueError("node_alphabet_size must be >= 1.")

    def make_interval(component_index: int) -> set:
        offset = component_index * size
        if node_alphabet_kind == "int":
            return set(range(offset, offset + size))
        if node_alphabet_kind == "letter":
            return {chr(ord("A") + idx) for idx in range(offset, offset + size)}
        raise ValueError("node_alphabet_kind must be 'int' or 'letter'.")

    return {
        "cycle": make_interval(0),
        "path": make_interval(1),
        "branching": make_interval(2),
    }


def artificial_part_decomposition(labels: Iterable):
    """Build the abstractgraph operator decomposition for one artificial part."""
    return compose(
        _unlabel_base_and_mapped_subgraphs,
        connected_component(),
        merge(),
        filter_by_node_label(must_have_one_of=sorted(labels)),
        node(),
    )


def fit_artificial_part_feasibility_estimators(
    graphs,
    *,
    node_alphabet_size: int,
    node_alphabet_kind: str = "int",
    component_specific_alphabets: bool = True,
    backend: str = "threading",
    n_jobs: int = 8,
) -> tuple[dict[str, FeasibilityEstimatorFeatureCannotExist], dict[str, list]]:
    """Fit one upstream feature-forbidding estimator for each artificial part."""
    label_intervals = artificial_part_label_intervals(
        node_alphabet_size,
        node_alphabet_kind=node_alphabet_kind,
        component_specific_alphabets=component_specific_alphabets,
    )
    decomposition_functions = {
        part: artificial_part_decomposition(labels)
        for part, labels in label_intervals.items()
    }
    estimators = {
        part: FeasibilityEstimatorFeatureCannotExist(
            decomposition_function=decomposition_function,
            backend=backend,
            n_jobs=n_jobs,
        ).fit(graphs)
        for part, decomposition_function in decomposition_functions.items()
    }
    labels = {
        part: sorted(label_set)
        for part, label_set in label_intervals.items()
    }
    return estimators, labels


def artificial_part_estimator_summary(
    estimators: Mapping[str, FeasibilityEstimatorFeatureCannotExist],
    labels: Mapping[str, list],
) -> pd.DataFrame:
    """Return a compact table describing fitted artificial part estimators."""
    return pd.DataFrame(
        [
            {
                "part": part,
                "labels": labels[part],
                "seen_features": len(getattr(estimator, "seen_feature_labels", [])),
            }
            for part, estimator in estimators.items()
        ]
    )


def score_artificial_part_feasibility(
    estimators: Mapping[str, FeasibilityEstimatorFeatureCannotExist],
    graphs,
) -> pd.DataFrame:
    """Return per-part feasibility decisions for ``graphs``."""
    return pd.DataFrame({
        part: estimator.predict(graphs)
        for part, estimator in estimators.items()
    })


def artificial_true_count_histogram(part_feasibility: pd.DataFrame) -> pd.DataFrame:
    """Return count/fraction histogram by number of feasible artificial parts."""
    if len(part_feasibility) == 0:
        raise ValueError("part_feasibility must contain at least one row.")
    histogram = (
        part_feasibility.sum(axis=1)
        .value_counts(sort=False)
        .reindex(range(part_feasibility.shape[1], -1, -1), fill_value=0)
        .rename_axis("n_true")
        .rename("count")
        .reset_index()
    )
    histogram["fraction"] = histogram["count"] / len(part_feasibility)
    return histogram


def compare_artificial_feasibility_efforts(
    graph_generator,
    estimators: Mapping[str, FeasibilityEstimatorFeatureCannotExist],
    *,
    n_graphs: int,
    effort_min: int = 0,
    effort_max: int = 5,
    feasibility_filter: str = "none",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample graphs across feasibility efforts and summarize per-part feasibility."""
    n_graphs = int(n_graphs)
    if n_graphs < 1:
        raise ValueError("n_graphs must be >= 1.")
    if effort_max < effort_min:
        raise ValueError("effort_max must be >= effort_min.")

    rows = []
    for effort in range(int(effort_min), int(effort_max) + 1):
        samples = graph_generator.sample(
            n_samples=n_graphs,
            feasibility_effort=effort,
            feasibility_filter=feasibility_filter,
        )
        part_feasibility = score_artificial_part_feasibility(estimators, samples)
        effort_histogram = artificial_true_count_histogram(part_feasibility)
        effort_histogram.insert(0, "feasibility_effort", effort)
        rows.append(effort_histogram)

    histogram = pd.concat(rows, ignore_index=True)
    fraction_table = histogram.pivot(
        index="feasibility_effort",
        columns="n_true",
        values="fraction",
    ).reset_index()
    fraction_table.columns.name = None
    fraction_table = fraction_table.rename(columns={
        n_true: f"frac_{n_true}_true"
        for n_true in range(len(estimators) + 1)
    })
    return histogram, fraction_table


def assert_artificial_part_feasibility(
    estimators: Mapping[str, FeasibilityEstimatorFeatureCannotExist],
    graphs,
) -> pd.DataFrame:
    """Assert that all graphs are feasible for every artificial part."""
    feasibility = score_artificial_part_feasibility(estimators, graphs)
    assert feasibility.to_numpy(dtype=bool).all(), feasibility
    return feasibility


def artificial_feasibility_titles(
    estimators: Mapping[str, FeasibilityEstimatorFeatureCannotExist],
    graphs,
) -> list[str]:
    """Return plot titles showing exactly which artificial parts fail feasibility."""
    titles = []
    for graph_idx, graph in enumerate(graphs):
        failed_parts = [
            part
            for part, estimator in estimators.items()
            if not bool(estimator.predict([graph])[0])
        ]
        if not failed_parts:
            titles.append(f"{graph_idx}: feasible")
        else:
            titles.append(f"{graph_idx}: not feasible ({', '.join(failed_parts)})")
    return titles


__all__ = [
    "ARTIFICIAL_PARTS",
    "artificial_feasibility_titles",
    "artificial_part_decomposition",
    "artificial_part_estimator_summary",
    "artificial_part_label_intervals",
    "artificial_true_count_histogram",
    "assert_artificial_part_feasibility",
    "build_artificial_plotter",
    "compare_artificial_feasibility_efforts",
    "draw_artificial_graphs",
    "find_latest_artificial_dataset_config",
    "find_latest_artificial_graph_generator",
    "fit_artificial_part_feasibility_estimators",
    "generate_artificial_dataset",
    "load_artificial_feasibility_graphs",
    "load_latest_artificial_graph_generator",
    "make_artificial_graph_plotter",
    "score_artificial_part_feasibility",
    "summarize_graphs",
]
