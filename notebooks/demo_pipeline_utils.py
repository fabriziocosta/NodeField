"""Compatibility wrapper for demo pipeline helpers."""

from conditional_node_field_graph_generator.extensions.demo.pipeline import (
    build_dataset,
    build_graph_generator,
    prepare_experiment,
)

__all__ = [
    "build_dataset",
    "build_graph_generator",
    "prepare_experiment",
]
