## Artificial Graph Utilities

This document describes the artificial-graph and synthetic-dataset helpers that support demo workflows, synthetic experiments, and some tests.

These utilities are not part of the core NodeField model. New code should access them through the synthetic extension namespace:
- [`conditional_node_field_graph_generator/extensions/synthetic/__init__.py`](../../../conditional_node_field_graph_generator/extensions/synthetic/__init__.py)

The maintained implementation now lives in:
- [`conditional_node_field_graph_generator/extensions/synthetic/primitives.py`](../../../conditional_node_field_graph_generator/extensions/synthetic/primitives.py)
- [`conditional_node_field_graph_generator/extensions/synthetic/datasets.py`](../../../conditional_node_field_graph_generator/extensions/synthetic/datasets.py)
- [`conditional_node_field_graph_generator/extensions/synthetic/composition.py`](../../../conditional_node_field_graph_generator/extensions/synthetic/composition.py)

## Main Responsibilities

### 1. Primitive synthetic graph samplers

The primitive samplers generate unlabeled graph structures that are later decorated with node and edge labels.

Main functions in [`extensions/synthetic/primitives.py`](../../../conditional_node_field_graph_generator/extensions/synthetic/primitives.py):

- `random_path_graph(n)`
  Build a path graph with `n` nodes.

- `random_tree_graph(n)`
  Build a random tree with `n` nodes.

- `random_cycle_graph(n)`
  Build a cycle-like graph by augmenting a random tree with extra edges between terminal nodes.

- `random_regular_graph(d, n)`
  Build a `d`-regular graph with `n` nodes.

- `random_degree_seq(n, dmax)`
  Build an expected-degree graph from a simple degree sequence.

- `random_dense_graph(n, m)`
  Build a dense random graph with `n` nodes and `m` edges, then keep the largest connected component.

- `make_graph_generator(graph_type, instance_size)`
  Dispatch helper that selects one of the primitive samplers from a graph-type string.

Supported `graph_type` values:
- `path`
- `tree`
- `cycle`
- `degree`
- `regular`
- `dense`

Utility classes:
- `RandomGraphConstructor`
  Samples lightweight random graphs based on random integer edge endpoints rather than the graph-type dispatcher above.

## 2. Labeling and attribute decoration

The dataset helpers can decorate graph nodes with labels and optional feature vectors.

Main pieces in [`extensions/synthetic/datasets.py`](../../../conditional_node_field_graph_generator/extensions/synthetic/datasets.py):

- `AttributeGenerator`
  Samples node-level auxiliary attribute vectors conditioned on sampled class labels.

- `make_graph(graph_generator, alphabet_size, attribute_generator)`
  Takes a primitive graph and assigns:
  - node `true_label`
  - node `label`
  - optional node `vec`
  - edge label `"-"`

Label behavior:
- `true_label` is sampled in the full class space
- `label` is the class modulo `alphabet_size`
- if `attribute_generator` is present, each node also receives a sampled vector attribute

## 3. Linked source/context graph construction

The synthetic datasets are based on joining a target graph with a context graph.

Key functions:
- `link_graphs(graph_source, graph_target, n_link_edges=0)`
  Build a disjoint union of two graphs and add `n_link_edges` random cross-graph edges.

- `make_graphs(...)`
  Generate batches of linked source/context graphs using a target graph family and a context graph family.

Important behavior:
- `use_single_target=True`
  Reuses one target graph across all samples.

- `use_single_target=False`
  Samples a fresh target graph for each example.

This distinction is used to create positive and negative synthetic classes.

### Cycle/path/ray artificial datasets

- `generate_artificial_dataset(...)`
  Builds batches of connected graphs from repeated cycle -> path -> ray units.
  The original single-unit shape is still the default with `n_iterations=1`.

Structural parameters have node-count semantics:

- `num_cycles`
  Number of cycle motifs per unit. `num_cycles=0` omits the cycle section.

- `cycle_length`
  Size of each cycle when `num_cycles > 0`. If `num_cycles=0`,
  `cycle_length` is ignored.

- `path_length`
  Number of path-labeled nodes in the unit. The star/ray hub is not counted as
  a path node.

- `num_rays`
  Number of rays after the path section. `num_rays=0` omits the star/ray hub
  entirely.

- `ray_length`
  Number of star-labeled nodes added per ray after the hub. `ray_length=0`
  adds no ray nodes. If `num_rays > 0`, the star/ray hub may still exist as the
  ray attachment point.

- `n_iterations`
  Number of repeated attachment waves. Each unit attaches new units to the
  endpoints produced by the previous unit. With `n_iterations=2`, the shape is
  cycle -> path -> rays -> attached cycle/path/ray units. With
  `n_iterations=3`, attachment continues from the endpoints of those attached
  units.

When `num_cycles=0`, a unit starts from the next available section:

- path-only graphs are allowed with `path_length > 0` and `num_rays=0`
- ray-only trees are allowed with `path_length=0` and `num_rays > 0`
- fully empty units (`num_cycles=0`, `path_length=0`, `num_rays=0`) are invalid

When structural parameters are integer ranges, each materialized unit samples a
fresh size from the same parameter range. Generated graphs record the actual
per-unit draws in `graph.graph["metadata"]["iteration_parameters"]`.

By default, `generate_artificial_dataset` also writes a reproducibility config:

```python
graphs, plot_artificial_graphs = generate_artificial_dataset(
    num_graphs=100,
    cycle_length=(3, 6),
    n_iterations=1,
    num_cycles=1,
    path_length=(1, 4),
    num_rays=3,
    ray_length=(1, 3),
    seed=13,
)
```

Set `num_cycles > 1` to chain same-length cycles through shared edges. The
first cycle connects to the path/ray structure; each additional cycle shares one
random edge with the previous cycle. Set `num_cycles=0` to skip cycles entirely.

Examples:

```python
# A pure ray tree: one hub and three length-2 rays.
graphs, plot_artificial_graphs = generate_artificial_dataset(
    num_graphs=100,
    cycle_length=0,
    num_cycles=0,
    path_length=0,
    num_rays=3,
    ray_length=2,
    seed=13,
)

# Iterative cycle/path/ray graphs with fresh per-unit samples.
graphs, plot_artificial_graphs = generate_artificial_dataset(
    num_graphs=100,
    cycle_length=(3, 5),
    num_cycles=2,
    n_iterations=3,
    path_length=(0, 2),
    num_rays=2,
    ray_length=(0, 2),
    seed=13,
)
```

The function returns a plain list of NetworkX graphs plus an artificial-graph
plot function:

```python
graphs, plot_artificial_graphs = generate_artificial_dataset(...)
plot_artificial_graphs(graphs[:20], n_cols=10)
graph_generator.fit(
    graphs,
    sample_training_progress_plot_kwargs=plot_artificial_graphs.plot_kwargs,
    sample_training_progress_plot_fn=plot_artificial_graphs,
)
```

The plotter fixes cycle/path/star node colors to red/blue/green ramps derived
from `node_alphabet_size`; callers may still override labels, node size, edge
width, layout, and related rendering options. The `size` argument is the display
size per graph panel: one graph with `size=5` uses `figsize=(5, 5)`, while a
two-column row uses `figsize=(10, 5)`.

The function prints the generated YAML file name, for example:

```text
Saved artificial dataset config: artificial-cycle-path-star-n100-c3-6-p1-4-r3x1-3.yaml
```

For notebook workflows, pass
`save_config_dir=REPO_ROOT / "notebooks" / "configs" / "artificial_datasets"`
so generated dataset configs stay in the dedicated config directory rather
than the notebook root.

The same configuration can be loaded later:

```python
graphs, plot_artificial_graphs = generate_artificial_dataset(
    load_from_file="artificial-cycle-path-star-n100-c3-6-p1-4-r3x1-3.yaml",
    save_config=False,
)
```

Config filenames use the same sanitized crumb style as model names and include
only dataset-defining values, such as graph count, cycle length, cycle count,
iteration count, path length, ray count, ray length, and non-default label
alphabets. Pass `save_config=False` when no config file should be written.

### Conditioning-vector analysis notebook

`notebooks/synthetic/evaluate_conditioning.ipynb` compares generated graphs
against conditioning graphs for this artificial family.

Important analysis behavior:

- conditional samples are generated once per selected conditioning graph and
  reused for all tables, histograms, scatter plots, and examples
- conditioning graph statistics use `metadata["iteration_parameters"]` when
  present, so cycle, path, and ray counts are aggregated across all iterations
- generated graph statistics are measured from labels and topology only
- cycle count is measured across all cycle-labeled components with
  `networkx.cycle_basis`
- path size is the total number of path-labeled nodes
- generated ray count is observable-only; zero-length ray multiplicity is not
  inferred from the conditioning metadata
- node-label and node-degree histograms compare the conditioning graph against
  the cached generated samples

## 4. Synthetic classification datasets

The main binary dataset builders are:

- `make_graphs_classification_dataset(...)`
  Positive class:
  - same target graph reused across samples
  Negative class:
  - fresh target graph per sample

- `make_two_types_graphs_classification_dataset(...)`
  Positive and negative classes are built from different target/context graph families.

Both functions:
- deduplicate graphs using `abstractgraph.hashing.GraphHashDeduper`
- return:
  - `graphs`
  - `targets`
  - `pos_graphs`
  - `neg_graphs`

## 5. Dataset constructor classes

Class wrappers:

- `ArtificialGraphConstructor`
  Small wrapper around a single primitive graph family plus optional attribute generation.

- `ArtificialGraphDatasetConstructor`
  Higher-level wrapper for binary synthetic datasets with separate positive and negative graph-family settings.

Important methods:
- `get_graph_types()`
  Returns the supported graph-family names.

- `sample(n_samples, return_separate_classes=False)`
  Returns either:
  - `(graphs, targets)`, or
  - `(pos_graphs, neg_graphs)` when `return_separate_classes=True`

## 6. Graph composition utilities

[`extensions/synthetic/composition.py`](../../../conditional_node_field_graph_generator/extensions/synthetic/composition.py) contains:

- `make_combined_graphs(graphs1, targets1, graphs2=None, targets2=None, number_of_graphs=1, number_of_edges=1)`

This helper:
- samples pairs of graphs with matching targets
- relabels nodes to avoid collisions
- composes the graphs
- adds random cross-graph edges
- renumbers the final nodes consecutively

Typical use:
- making larger synthetic graphs from smaller target-matched components
- augmentation for demo experiments

## Dependency note

The synthetic dataset builders depend on `abstractgraph` for graph deduplication.

That dependency enters through:
- `GraphHashDeduper`

This is why the synthetic extension is considered auxiliary rather than core.

## Intended Scope

Use these utilities for:
- synthetic demos
- artificial benchmark construction
- quick graph-family experiments
- notebook support

Do not treat them as required for:
- the core Conditional Node Field model
- molecule workflows
- generic production graph ingestion
