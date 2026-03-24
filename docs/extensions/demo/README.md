# Demo Extension

This section documents the demo-oriented extension layer for NodeField.

The demo extension is useful for:
- notebook-facing dataset preparation
- reusable plotting and lightweight analysis helpers
- checkpoint discovery helpers for interactive training workflows
- oracle trace capture helpers for notebook diagnostics

Primary entry points live under:
- [`conditional_node_field_graph_generator/extensions/demo/__init__.py`](../../../conditional_node_field_graph_generator/extensions/demo/__init__.py)
- [`conditional_node_field_graph_generator/extensions/demo/pipeline.py`](../../../conditional_node_field_graph_generator/extensions/demo/pipeline.py)
- [`conditional_node_field_graph_generator/extensions/demo/visualization.py`](../../../conditional_node_field_graph_generator/extensions/demo/visualization.py)
- [`conditional_node_field_graph_generator/extensions/demo/oracle.py`](../../../conditional_node_field_graph_generator/extensions/demo/oracle.py)
- [`conditional_node_field_graph_generator/extensions/demo/storage.py`](../../../conditional_node_field_graph_generator/extensions/demo/storage.py)
- [`conditional_node_field_graph_generator/notebooks.py`](../../../conditional_node_field_graph_generator/notebooks.py)
- [`conditional_node_field_graph_generator/persistence.py`](../../../conditional_node_field_graph_generator/persistence.py)

Boundary:
- this extension is not required for the core NodeField model
- it exists to support notebooks, demos, and interactive experiment flows
- generic fitted-model serialization is now part of core package utilities, not the demo extension

Supported notebook bootstrap:
- maintained notebooks should import `configure_notebook` and `import_nsppk` from `conditional_node_field_graph_generator.notebooks`
- notebook cells should not modify `sys.path` or walk parent directories to find the repo root
- chemistry and demo notebook workflows expect installed `abstractgraph`, `abstractgraph-ml`, `abstractgraph-graphicalizer`, and `nsppk` packages rather than local checkouts or import shims

Transition note:
- new code should import demo helpers from `conditional_node_field_graph_generator.extensions.demo`
- oracle trace collection should use `collect_oracle_trace_rows()` and `parse_oracle_trace_title()` instead of notebook-local monkey patches
- incompatible Lightning resume checkpoints now fail explicitly instead of silently restarting from scratch
