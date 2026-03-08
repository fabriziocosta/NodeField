## Molecular Graph Utilities

This module centralizes all local operations related to molecular graphs in [`molecular_graph_utils.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/conditional_node_field_graph_generator/molecular_graph_utils.py).

It is intended to replace the previous dependency on `coco-grape` for:
- PubChem assay loading
- molecule drawing
- ZINC download, caching, and conversion
- small dataset-loader compatibility helpers used by notebooks

The implementation keeps the older PubChem loader and supervised-loader interfaces available locally so existing notebook flows can continue to work with minimal change.

For new code, prefer the extension namespace:
- [`extensions/molecular/__init__.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/conditional_node_field_graph_generator/extensions/molecular/__init__.py)

## Main Responsibilities

### 1. Molecule <-> graph conversion

These helpers convert between RDKit molecules and the NetworkX graph format used in this repository.

- `smiles_to_networkx_molecule(smiles, zinc_id=None, properties=None)`
  Builds a labeled `networkx.Graph` from a SMILES string.

- `molecule_to_networkx(mol, graph_metadata=None)`
  Converts an RDKit molecule to the repository graph format.

- `networkx_to_molecule(graph, sanitize=True)`
  Converts a labeled molecular graph back to an RDKit molecule.

- `nx_to_rdkit(graph)`
  Compatibility wrapper around `networkx_to_molecule(..., sanitize=False)`.

- `rdkmol_to_nx(mol)`
  Compatibility wrapper around `molecule_to_networkx(...)`.

Expected graph conventions:
- node label: atomic symbol, stored in `node["label"]`
- edge label: bond order string (`"1"`, `"2"`, `"3"`, `"AROMATIC"`)
- optional graph metadata such as `smiles`, `zinc_id`, `pubchem_cid`, `inchi`

## 2. Molecule drawing

These helpers render molecular graphs using RDKit instead of the previous external visualization helper.

- `set_coordinates(compounds)`
  Computes 2D coordinates for RDKit molecules before drawing.

- `compounds_to_image(compounds, n_graphs_per_line=5, size=250, legend=None)`
  Builds a grid image directly from RDKit molecules.

- `molecule_graphs_to_grid_image(graphs, legends=None, mols_per_row=4, sub_img_size=(250, 200))`
  Builds a grid image from molecular NetworkX graphs.

- `nx_to_image(graphs, n_graphs_per_line=5, size=250, title_key=None, titles=None)`
  Compatibility helper matching the older `coco_grape` drawing API.

- `draw_molecules(graphs, titles=None, num=None, n_graphs_per_line=7, size=7, ...)`
  Displays batches of molecular graphs.

Notes:
- this is notebook-oriented display code
- graphs are rendered in chunks of 50 molecules
- the API intentionally stays close to the old helper so legacy notebook cells remain usable

## 3. PubChem loading

The module includes a local copy of the older PubChem loading workflow.

- `resolve_pubchem_dir(pubchem_dir=None)`
  Resolves the local directory that stores assay SDF files.

- `PubChemLoader`
  Local replacement for the older `coco_grape.data_loader.mol.mol_loader.PubChemLoader`.

Important methods:
- `get_assay_description(assay_id)`
- `download(assay_id, active=True, stepsize=50)`
- `load(assay_id, dirname="PUBCHEM", format_type="sdf")`

Behavior:
- expects PubChem assay files named like `AID651610_active.sdf` and `AID651610_inactive.sdf`
- loads active molecules as class `1`
- loads inactive molecules as class `0`
- returns `(graphs, targets)`

Additional higher-level helper:
- `load_pubchem_graph_dataset(pubchem_dir=None, assay_id="651610", dataset_size=None, max_node_count=None, use_equalized=False, random_state=0)`

This higher-level loader:
- reads local SDFs
- preserves more metadata in a returned dataframe
- can filter by node count
- can equalize positive/negative classes
- returns `(graphs, targets, metadata)`

## 4. Supervised dataset compatibility loader

`SupervisedDataSetLoader` is a local copy of the older generic supervised-loader helper.

It supports:
- resizing datasets to a requested sample count
- equalizing classes
- restricting to a subset of targets
- multiclass-to-binary conversion
- regression-to-binary conversion

Main method:
- `load()`

Typical use:
```python
from conditional_node_field_graph_generator.extensions.molecular import (
    PubChemLoader,
    SupervisedDataSetLoader,
)

loader = PubChemLoader()
loader.pubchem_dir = "notebooks/datasets/PUBCHEM"

def pubchem_loader():
    return loader.load("651610", dirname=loader.pubchem_dir)

graphs, targets = SupervisedDataSetLoader(
    load_func=pubchem_loader,
    size=200,
    use_equalized=False,
).load()
```

## 5. ZINC helpers

The module also owns the maintained ZINC dataset helpers.

Constants:
- `ZINC_250K_URL`
- `DEFAULT_ZINC_TARGET_COLUMNS`

Main functions:
- `download_zinc_dataset(dataset_dir, url=..., filename="zinc_250k.csv", ...)`
- `build_zinc_graph_corpus(dataset_dir, csv_path, force=False, ...)`
- `load_zinc_graph_dataset(dataset_dir, max_molecules=100_000, min_node_count=None, max_node_count=40, ...)`
- `extract_zinc_targets(metadata, target_columns=("logP", "qed", "SAS"))`

Workflow:
1. download the source CSV once
2. convert SMILES rows to NetworkX molecular graphs
3. cache graphs bucketed by node count
4. load only the requested size slice later

This is the backend used by the maintained ZINC notebook flow.

## Compatibility Layer

To avoid breaking legacy notebook imports immediately, this repository also exposes thin local compatibility shims under:
- [`coco_grape/data_loader/loader.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/coco_grape/data_loader/loader.py)
- [`coco_grape/data_loader/mol/mol_loader.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/coco_grape/data_loader/mol/mol_loader.py)
- [`coco_grape/visualizer/mol_display.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/coco_grape/visualizer/mol_display.py)

These local wrappers forward the old import paths to this module.

## Intended Scope

Use this module for:
- molecular graph conversion
- molecular notebook visualization
- PubChem assay ingestion
- ZINC preparation and caching
- preserving compatibility with older molecule notebook code

Do not use it for:
- generic non-molecular graph rendering
- core generator logic unrelated to chemistry
