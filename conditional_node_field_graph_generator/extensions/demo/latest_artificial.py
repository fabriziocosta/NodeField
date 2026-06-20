"""Notebook helpers for sampling the latest saved artificial NodeField model."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import networkx as nx

from ...persistence import load_graph_generator, resolve_saved_generator_dir
from ..synthetic import make_artificial_graph_plotter


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
