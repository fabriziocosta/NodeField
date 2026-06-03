# Demo Extension

This section documents the demo-oriented extension layer for NodeField.

The demo extension is useful for:
- notebook-facing dataset preparation
- reusable plotting and lightweight analysis helpers
- checkpoint discovery helpers for interactive training workflows
- oracle trace capture helpers for notebook diagnostics
- demo feasibility-estimator composition, masking, and oracle-cut scheduling

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
- saved generators loaded through the core persistence helper are upgraded when possible so older demo feasibility-estimator composites gain the local masked wrapper and default adaptive oracle-cut allocation

Demo feasibility estimator notes:
- the maintained demo pipeline builds the feasibility estimator as an ordered stack of internal motif checks
- the wrapper accepts boolean masks to activate one-hot, cumulative, or full-stack subsets of those internal checks
- oracle-guided decode uses per-level structural-cut budgets
- oracle-guided decode keeps structural edge-set cuts as the hard refinement mechanism
- optional node-label and edge-label repairs are soft follow-up proposals evaluated against the full oracle state after each structural candidate, rather than isolated decode stages
- the default budget policy is adaptive: it starts from a decreasing prior and redistributes unused budget toward estimator levels that still expose violations on the current graph

Demo training notes:
- `build_graph_generator(...)` forwards orchestrator SVD options:
  `use_embedding_svd`, `node_embedding_svd_dimension`, and
  `graph_embedding_svd_dimension`
- SVD compression is enabled by default in new demo generators, with a default
  requested dimension of `256` for both node and graph embeddings
- `build_graph_generator(...)` now uses `locality_horizon=3` by default so new
  demo models train the auxiliary horizon-locality head used by horizon-aware
  ILP decoding
- horizon-aware ILP decoding is enabled by default on the demo decoder and can
  be tuned with `decoder_horizon_constraint_weight`,
  `decoder_horizon_positive_threshold`, `decoder_horizon_negative_threshold`,
  `decoder_horizon_pair_budget`, `decoder_horizon_paths_per_pair`, and
  `decoder_horizon_max_iterations`
- `fit_graph_generator(...)` forwards the training-progress PDF options accepted
  by the core generator, including `sample_training_progress`,
  `sample_training_progress_n_samples`,
  `sample_training_progress_every_n_epochs`,
  `sample_training_progress_pdf_path`, and
  `sample_training_progress_plot_kwargs`
- `fit_graph_generator(...)` also forwards
  `sample_training_progress_plot_fn`, which can provide a dataset-specific
  per-graph renderer such as an RDKit molecule drawer for the progress PDF
