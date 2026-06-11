## Synthetic Extension

This package contains artificial-graph and synthetic-dataset helpers used for demos, testing, and notebook workflows. They are not part of the core NodeField model.

Main entry points:
- [`primitives.py`](primitives.py)
- [`datasets.py`](datasets.py)
- [`composition.py`](composition.py)

Artificial cycle/path/ray datasets support repeated units through
`n_iterations`, cycle-free/path-only/ray-only shapes with `num_cycles=0`, and
per-unit sampled structural ranges recorded in graph metadata as
`iteration_parameters`.

New code should import from `conditional_node_field_graph_generator.extensions.synthetic`.
