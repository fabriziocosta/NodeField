# TODO

This file tracks cleanup work that remains after the ongoing core/extensions separation.

The current codebase works, but several compatibility layers and legacy entry points still exist to avoid breaking notebooks and older imports abruptly.

## High Priority

- Remove the local `coco_grape/` compatibility shim once all notebooks and any downstream code stop importing:
  - `coco_grape.data_loader.loader`
  - `coco_grape.data_loader.mol.mol_loader`
  - `coco_grape.visualizer.mol_display`

- Remove the backward-compatible [`support.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/conditional_node_field_graph_generator/support.py) re-export layer once all imports have moved to:
  - `conditional_node_field_graph_generator.runtime_utils`
  - `conditional_node_field_graph_generator.extensions.synthetic`

- Remove the legacy wrapper [`molecular_graph_utils.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/conditional_node_field_graph_generator/molecular_graph_utils.py) once callers use:
  - `conditional_node_field_graph_generator.extensions.molecular`

- Decide whether the root-level synthetic helper modules should remain as implementation files or also be folded fully under `extensions/synthetic/`:
  - [`synthetic_graph_primitives.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/conditional_node_field_graph_generator/synthetic_graph_primitives.py)
  - [`synthetic_graph_datasets.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/conditional_node_field_graph_generator/synthetic_graph_datasets.py)
  - [`graph_composition.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/conditional_node_field_graph_generator/graph_composition.py)

## Notebook Cleanup

- Remove the `.py` compatibility wrappers under `notebooks/` once notebook cells import directly from extensions:
  - [`notebooks/demo_pipeline_utils.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/notebooks/demo_pipeline_utils.py)
  - [`notebooks/notebook_utils.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/notebooks/notebook_utils.py)
  - [`notebooks/zinc_dataset_utils.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/notebooks/zinc_dataset_utils.py)

- Rewrite notebook cells that still use legacy `coco_grape` imports:
  - [`notebooks/demo_chem.ipynb`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/notebooks/demo_chem.ipynb)
  - [`notebooks/demo_optimization.ipynb`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/notebooks/demo_optimization.ipynb)
  - [`notebooks/demo_zinc.ipynb`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/notebooks/demo_zinc.ipynb)

- Update notebook prose/comments that still describe old helper locations, for example references to `notebooks/demo_pipeline_utils.py` as the main home of reusable logic.

- Decide whether saved-generator helpers should remain notebook/demo-specific or move to a more explicit non-notebook utility module.

## Documentation Cleanup

- Update docs so they stop referring to code-folder READMEs that have been moved under `docs/extensions/`.

- Audit [`README.md`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/README.md) project-layout and path descriptions so they reflect:
  - `extensions/molecular`
  - `extensions/synthetic`
  - `extensions/demo`
  - the reduced role of `notebooks/*.py`

- Add a dedicated extension README for the new demo extension:
  - `conditional_node_field_graph_generator/extensions/demo/`
  - matching docs entry under `docs/extensions/`

- Update [`docs/extensions/molecular/MOLECULAR_GRAPH_UTILS_README.md`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/docs/extensions/molecular/MOLECULAR_GRAPH_UTILS_README.md) so it describes the extension-first import path as the primary API and the legacy module as transitional only.

- Update [`docs/extensions/synthetic/ARTIFICIAL_GRAPH_UTILS_README.md`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/docs/extensions/synthetic/ARTIFICIAL_GRAPH_UTILS_README.md) so it describes `extensions/synthetic/*` as the primary import surface and the root modules as transitional only.

## API Cleanup

- Decide whether the demo extension should stay as one package or be split further into:
  - `extensions/demo/pipeline.py`
  - `extensions/demo/visualization.py`
  - `extensions/demo/storage.py`
  or collapsed differently.

- Review whether `runtime_utils.py` belongs in core or should move to a more explicit internal helpers namespace.

- Review whether `tests` should import compatibility wrappers (`support.py`, notebook wrappers) or migrate fully to the extension/core paths.

## Testing Cleanup

- Add tests for the new demo extension modules directly, instead of relying only on notebook wrapper coverage.

- Add tests that confirm the compatibility wrappers continue to match the new extension APIs while they still exist.

- Once wrappers are removed, simplify tests to target only:
  - core package modules
  - extension package modules

## Suggested Removal Order

1. Update all notebooks to import from `extensions/*`.
2. Update tests to stop depending on notebook wrappers and `support.py`.
3. Remove `notebooks/*.py` wrappers.
4. Remove `coco_grape/` compatibility shims.
5. Remove `support.py` compatibility re-exports.
6. Remove root-level legacy helper modules if the extension packages become the only supported API.
