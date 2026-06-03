"""Synthetic-graph helpers for demos and artificial datasets."""

from .composition import make_combined_graphs
from .datasets import (
    ArtificialGraphConstructor,
    ArtificialGraphDatasetConstructor,
    AttributeGenerator,
    artificial_node_label_colors,
    generate_artificial_dataset,
    generate_cycle_path_star_graph,
    link_graphs,
    make_graph,
    make_artificial_graph_plotter,
    make_graphs,
    make_graphs_classification_dataset,
    make_two_types_graphs_classification_dataset,
)
from .primitives import (
    RandomGraphConstructor,
    make_graph_generator,
    random_cycle_graph,
    random_degree_seq,
    random_dense_graph,
    random_path_graph,
    random_regular_graph,
    random_tree_graph,
)

__all__ = [
    "ArtificialGraphConstructor",
    "ArtificialGraphDatasetConstructor",
    "AttributeGenerator",
    "artificial_node_label_colors",
    "generate_artificial_dataset",
    "generate_cycle_path_star_graph",
    "RandomGraphConstructor",
    "link_graphs",
    "make_artificial_graph_plotter",
    "make_combined_graphs",
    "make_graph",
    "make_graph_generator",
    "make_graphs",
    "make_graphs_classification_dataset",
    "make_two_types_graphs_classification_dataset",
    "random_cycle_graph",
    "random_degree_seq",
    "random_dense_graph",
    "random_path_graph",
    "random_regular_graph",
    "random_tree_graph",
]
