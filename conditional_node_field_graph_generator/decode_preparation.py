"""Compatibility re-exports for decode-preparation helpers.

Canonical decoder-related implementations now live in
``conditional_node_field_graph_decoder``.
"""

from .conditional_node_field_graph_decoder import (
    build_single_generated_node_batch,
    decode_generated_nodes,
    resolve_predicted_edge_labels,
    resolve_predicted_node_labels,
)

__all__ = [
    "build_single_generated_node_batch",
    "decode_generated_nodes",
    "resolve_predicted_edge_labels",
    "resolve_predicted_node_labels",
]
