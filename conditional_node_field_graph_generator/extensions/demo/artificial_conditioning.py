"""Reusable analysis helpers for artificial conditioning-vector notebooks."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


COMPONENTS = ("cycle", "path", "star")


def _label_sets(conditioning_graph: nx.Graph) -> dict[str, set[Any]]:
    metadata = conditioning_graph.graph.get("metadata", {})
    alphabets = metadata.get("node_alphabets_by_component", {}) or {}
    sets = {component: set(alphabets.get(component, [])) for component in COMPONENTS}
    if not any(sets.values()):
        for _, attrs in conditioning_graph.nodes(data=True):
            component = attrs.get("label_component")
            if component in sets:
                sets[component].add(attrs.get("label"))
    return sets


def _component_nodes(graph: nx.Graph, conditioning_graph: nx.Graph) -> dict[str, list[Any]]:
    labels = _label_sets(conditioning_graph)
    nodes = {component: [] for component in COMPONENTS}
    for node, attrs in graph.nodes(data=True):
        for component, component_labels in labels.items():
            if attrs.get("label") in component_labels:
                nodes[component].append(node)
                break
    return nodes


def _iteration_parameters(graph: nx.Graph) -> list[dict[str, int]]:
    metadata = graph.graph.get("metadata", {})
    values = metadata.get("iteration_parameters")
    if values:
        return [
            {
                key: int(unit.get(key, metadata.get(key, default)))
                for key, default in (
                    ("cycle_length", 0),
                    ("num_cycles", 1),
                    ("path_length", 0),
                    ("num_rays", 0),
                    ("ray_length", 0),
                )
            }
            for unit in values
        ]
    return [
        {
            key: int(metadata.get(key, default))
            for key, default in (
                ("cycle_length", 0),
                ("num_cycles", 1),
                ("path_length", 0),
                ("num_rays", 0),
                ("ray_length", 0),
            )
        }
    ]


def _path_stats(graph: nx.Graph, nodes: Iterable[Any]) -> dict[str, Any]:
    subgraph = graph.subgraph(nodes)
    components = [subgraph.subgraph(component).copy() for component in nx.connected_components(subgraph)]
    sizes = sorted(component.number_of_nodes() for component in components)
    valid = True
    for component in components:
        n_nodes = component.number_of_nodes()
        degrees = sorted(dict(component.degree()).values())
        valid = valid and (n_nodes <= 1 or degrees.count(1) == 2 and max(degrees, default=0) <= 2)
    return {
        "path_length": len(list(nodes)),
        "path_component_count": len(components),
        "path_component_sizes": tuple(sizes),
        "path_is_valid": bool(valid),
    }


def _star_stats(graph: nx.Graph, nodes: Iterable[Any]) -> dict[str, Any]:
    subgraph = graph.subgraph(nodes).copy()
    ray_sizes: list[int] = []
    hub_degrees: list[int] = []
    valid = nx.is_forest(subgraph)
    for component_nodes in nx.connected_components(subgraph):
        component = subgraph.subgraph(component_nodes).copy()
        if component.number_of_nodes() == 1:
            hubs = list(component.nodes())
        else:
            degrees = dict(component.degree())
            max_degree = max(degrees.values(), default=0)
            hubs = [node for node, degree in degrees.items() if degree == max_degree and max_degree >= 3]
            if not hubs:
                hubs = list(nx.center(component))[:1]
        hub_set = set(hubs)
        for hub in hubs:
            hub_degrees.append(int(component.degree(hub)))
            for neighbor in component.neighbors(hub):
                if neighbor in hub_set:
                    continue
                previous, current = hub, neighbor
                branch: set[Any] = set()
                while current not in branch:
                    branch.add(current)
                    next_nodes = [
                        node
                        for node in component.neighbors(current)
                        if node != previous and node not in hub_set
                    ]
                    if len(next_nodes) > 1:
                        valid = False
                    if not next_nodes:
                        break
                    previous, current = current, next_nodes[0]
                ray_sizes.append(len(branch))
    return {
        "ray_count": len(ray_sizes),
        "ray_sizes": tuple(sorted(ray_sizes)),
        "ray_size_median": float(np.median(ray_sizes)) if ray_sizes else np.nan,
        "ray_size_min": min(ray_sizes) if ray_sizes else np.nan,
        "ray_size_max": max(ray_sizes) if ray_sizes else np.nan,
        "star_hub_count": len(hub_degrees),
        "star_hub_degrees": tuple(sorted(hub_degrees)),
        "rays_are_valid": bool(valid),
    }


def artificial_graph_stats(graph: nx.Graph, conditioning_graph: nx.Graph | None = None) -> dict[str, Any]:
    """Measure artificial cycle/path/star structure relative to a condition graph."""
    conditioning_graph = conditioning_graph or graph
    nodes = _component_nodes(graph, conditioning_graph)
    cycle_subgraph = graph.subgraph(nodes["cycle"])
    cycle_sizes = sorted(len(cycle) for cycle in nx.cycle_basis(cycle_subgraph))
    iteration_parameters = _iteration_parameters(graph)
    cycle_config = [
        int(unit["cycle_length"])
        for unit in iteration_parameters
        for _ in range(max(0, int(unit["num_cycles"])))
        if int(unit["cycle_length"]) >= 3
    ]
    ray_config = [
        int(unit["ray_length"])
        for unit in iteration_parameters
        for _ in range(max(0, int(unit["num_rays"])))
    ]
    path_stats = _path_stats(graph, nodes["path"])
    star_stats = _star_stats(graph, nodes["star"])
    return {
        "total_nodes": graph.number_of_nodes(),
        "total_edges": graph.number_of_edges(),
        "cycle_count": len(cycle_sizes),
        "cycle_sizes": tuple(cycle_sizes),
        "cycle_size_median": float(np.median(cycle_sizes)) if cycle_sizes else np.nan,
        "cycles_are_valid": bool(all(size >= 3 for size in cycle_sizes)),
        "configured_cycle_count": len(cycle_config),
        "configured_cycle_size_median": float(np.median(cycle_config)) if cycle_config else np.nan,
        "configured_ray_count": len(ray_config),
        "configured_ray_size_median": float(np.median(ray_config)) if ray_config else np.nan,
        **path_stats,
        **star_stats,
    }


EXPERIMENTS = {
    "path_length": ("path_length", "path_is_valid", "conditioning path-node count", "generated path-node count"),
    "cycle_count": ("cycle_count", "cycles_are_valid", "conditioning cycle count", "generated cycle count"),
    "cycle_size": ("cycle_size_median", "cycles_are_valid", "conditioning median cycle size", "generated median cycle size"),
    "ray_count": ("ray_count", "rays_are_valid", "conditioning ray count", "generated ray count"),
    "ray_size": ("ray_size_median", "rays_are_valid", "conditioning median ray size", "generated ray size"),
}


def run_conditioning_vector_test(
    graphs: Iterable[nx.Graph],
    graph_generator: Any,
    *,
    experiment_type: str = "path_length",
    samples_per_conditioning_graph: int = 32,
    feasibility_effort: int = 0,
) -> dict[str, Any]:
    """Run one conditioning experiment and return dataframes plus generated graphs."""
    if experiment_type not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment_type={experiment_type!r}; choose from {sorted(EXPERIMENTS)}")
    generated_key, valid_key, x_label, y_label = EXPERIMENTS[experiment_type]
    graph_list = list(graphs)
    stats = {id(graph): artificial_graph_stats(graph) for graph in graph_list}
    selected: dict[Any, nx.Graph] = {}
    for graph in graph_list:
        value = stats[id(graph)].get(generated_key, np.nan)
        if not pd.isna(value):
            selected.setdefault(value, graph)
    conditioning_graphs = [selected[value] for value in sorted(selected)]
    conditioning_values = sorted(selected)
    generated_by_graph = graph_generator.conditional_sample(
        conditioning_graphs,
        n_samples=int(samples_per_conditioning_graph),
        feasibility_effort=int(feasibility_effort),
    )
    rows = []
    for index, (conditioning_graph, conditioning_value, generated_graphs) in enumerate(
        zip(conditioning_graphs, conditioning_values, generated_by_graph)
    ):
        conditioning_stats = artificial_graph_stats(conditioning_graph)
        for generated_index, generated_graph in enumerate(generated_graphs):
            generated_stats = artificial_graph_stats(generated_graph, conditioning_graph)
            rows.append(
                {
                    "conditioning_index": index,
                    "conditioning_value": conditioning_value,
                    "generated_index": generated_index,
                    **{f"conditioning_{key}": value for key, value in conditioning_stats.items()},
                    **{f"generated_{key}": value for key, value in generated_stats.items()},
                }
            )
    analysis_df = pd.DataFrame(rows)
    metric_column = f"generated_{generated_key}"
    valid_column = f"generated_{valid_key}"
    summary_rows = []
    for value, group in analysis_df.groupby("conditioning_value", sort=True):
        values = pd.to_numeric(group[metric_column], errors="coerce").dropna()
        summary_rows.append(
            {
                "conditioning_value": value,
                "n_generated": len(group),
                "valid_rate": group[valid_column].mean(),
                "generated_median": values.median() if not values.empty else np.nan,
                "generated_q25": values.quantile(0.25) if not values.empty else np.nan,
                "generated_q75": values.quantile(0.75) if not values.empty else np.nan,
                "generated_mean": values.mean() if not values.empty else np.nan,
                "generated_std": values.std(ddof=0) if not values.empty else np.nan,
            }
        )
    return {
        "experiment_type": experiment_type,
        "conditioning_graphs": conditioning_graphs,
        "conditioning_values": conditioning_values,
        "generated_by_graph": generated_by_graph,
        "analysis_df": analysis_df,
        "summary_df": pd.DataFrame(summary_rows),
        "metric_column": metric_column,
        "valid_column": valid_column,
        "x_label": x_label,
        "y_label": y_label,
    }


def plot_conditioning_vector_test(result: Mapping[str, Any], plotter: Any, *, examples_per_condition: int = 4):
    """Plot the summary relationship and a few conditioning/generated examples."""
    analysis_df = result["analysis_df"]
    summary_df = result["summary_df"]
    metric_column = result["metric_column"]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = analysis_df["conditioning_value"].to_numpy(dtype=float)
    y = pd.to_numeric(analysis_df[metric_column], errors="coerce").to_numpy(dtype=float)
    ax.scatter(x + rng.uniform(-0.05, 0.05, len(x)), y, alpha=0.35, s=28)
    ax.errorbar(
        summary_df["conditioning_value"],
        summary_df["generated_median"],
        yerr=[summary_df["generated_median"] - summary_df["generated_q25"], summary_df["generated_q75"] - summary_df["generated_median"]],
        fmt="o",
        color="black",
        capsize=4,
    )
    axis_values = np.concatenate([x, y[np.isfinite(y)]]) if len(y) else x
    if len(axis_values):
        limits = [float(np.nanmin(axis_values)), float(np.nanmax(axis_values))]
        ax.plot(limits, limits, "--", color="gray", linewidth=1)
    ax.set_xlabel(result["x_label"])
    ax.set_ylabel(result["y_label"])
    ax.set_title(f"{result['experiment_type']} conditioning test")
    fig.tight_layout()
    plt.show()
    for index, (conditioning_graph, generated_graphs, value) in enumerate(
        zip(result["conditioning_graphs"], result["generated_by_graph"], result["conditioning_values"])
    ):
        examples = list(generated_graphs)[:examples_per_condition]
        plotter(
            [conditioning_graph, *examples],
            n_cols=1 + len(examples),
            titles=[f"condition {value}", *[f"sample {sample_index}" for sample_index in range(len(examples))]],
        )
    return fig


def histogram_tables(graph: nx.Graph, generated_graphs: Iterable[nx.Graph]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return node-label and degree-count comparison tables."""
    generated = list(generated_graphs)

    def table(counter_fn, name: str) -> pd.DataFrame:
        conditioning = counter_fn(graph)
        counters = [counter_fn(item) for item in generated]
        values = sorted(set(conditioning).union(*(set(counter) for counter in counters)), key=str)
        matrix = np.asarray([[counter.get(value, 0) for value in values] for counter in counters], dtype=float)
        frame = pd.DataFrame(
            {
                name: values,
                "conditioning_count": [conditioning.get(value, 0) for value in values],
                "mean_generated_count": matrix.mean(axis=0),
                "std_generated_count": matrix.std(axis=0),
            }
        )
        frame["mean_abs_error"] = (frame["mean_generated_count"] - frame["conditioning_count"]).abs()
        return frame

    return table(lambda item: Counter(attrs.get("label") for _, attrs in item.nodes(data=True)), "label"), table(
        lambda item: Counter(dict(item.degree()).values()), "degree"
    )


__all__ = [
    "EXPERIMENTS",
    "artificial_graph_stats",
    "histogram_tables",
    "plot_conditioning_vector_test",
    "run_conditioning_vector_test",
]
