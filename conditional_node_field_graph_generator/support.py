"""Backward-compatible re-export layer for utilities previously merged in support.py."""

from .runtime_utils import _verbosity_level, run_trainer_fit, timeit
from .extensions.synthetic.composition import make_combined_graphs
from .extensions.synthetic.datasets import (
    ArtificialGraphConstructor,
    ArtificialGraphDatasetConstructor,
    AttributeGenerator,
    link_graphs,
    make_graph,
    make_graphs,
    make_graphs_classification_dataset,
    make_two_types_graphs_classification_dataset,
)
from .extensions.synthetic.primitives import (
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
    "_verbosity_level",
    "timeit",
    "run_trainer_fit",
    "RandomGraphConstructor",
    "random_path_graph",
    "random_tree_graph",
    "random_cycle_graph",
    "random_regular_graph",
    "random_degree_seq",
    "random_dense_graph",
    "make_graph_generator",
    "AttributeGenerator",
    "make_graph",
    "ArtificialGraphConstructor",
    "link_graphs",
    "make_graphs",
    "make_graphs_classification_dataset",
    "make_two_types_graphs_classification_dataset",
    "ArtificialGraphDatasetConstructor",
    "make_combined_graphs",
]
