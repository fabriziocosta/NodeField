# Synthetic Extension

This section documents the synthetic-graph extension layer for NodeField.

The synthetic extension is useful for:
- artificial graph primitives
- synthetic binary dataset construction
- linked source/context graph generation
- graph-composition helpers for demos and tests

The maintained artificial dataset family now supports iterative cycle/path/ray
units with `n_iterations`, cycle-free starts with `num_cycles=0`, path-only
graphs, ray-only trees, and per-unit sampled structural metadata.

Primary entry points live under:
- [`conditional_node_field_graph_generator/extensions/synthetic/__init__.py`](../../../conditional_node_field_graph_generator/extensions/synthetic/__init__.py)
- [`conditional_node_field_graph_generator/extensions/synthetic/primitives.py`](../../../conditional_node_field_graph_generator/extensions/synthetic/primitives.py)
- [`conditional_node_field_graph_generator/extensions/synthetic/datasets.py`](../../../conditional_node_field_graph_generator/extensions/synthetic/datasets.py)
- [`conditional_node_field_graph_generator/extensions/synthetic/composition.py`](../../../conditional_node_field_graph_generator/extensions/synthetic/composition.py)

Detailed module documentation:
- [`docs/extensions/synthetic/ARTIFICIAL_GRAPH_UTILS_README.md`](ARTIFICIAL_GRAPH_UTILS_README.md)

Boundary:
- this extension is not required for the core NodeField model
- it exists for demos, artificial datasets, and notebook-oriented experimentation
