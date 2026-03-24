## Molecular Graph Utilities

NodeField no longer maintains a local `extensions.molecular` package.

Use the abstractgraph ecosystem directly:

```python
from abstractgraph_graphicalizer.chem import (
    PubChemLoader,
    SupervisedDataSetLoader,
    ZINCLoader,
    build_zinc_graph_corpus,
    download_zinc_dataset,
    draw_molecules,
    extract_zinc_targets,
    graph_to_rdmol,
    load_pubchem_graph_dataset,
    load_zinc_graph_dataset,
    normalize_graph_schema,
    rdmol_to_graph,
    smiles_to_graph,
)
```

Install the supporting package normally:
- `pip install abstractgraph-graphicalizer`
- or `pip install -e /path/to/abstractgraph-graphicalizer`

Canonical chemistry schema:
- node label: atomic symbol in `node["label"]`
- edge label: `"single"`, `"double"`, `"triple"`, or `"aromatic"`
- edge metadata also includes `bond_order`, `bond_type`, and `aromatic`

Ownership split:
- NodeField owns the conditional graph-generation model, decoder, persistence, and demo helpers
- `abstractgraph_graphicalizer.chem` owns chemistry data loading, conversion, cache helpers, and rendering

Migration notes:
- old numeric bond labels like `"1"`, `"2"`, `"3"`, and `"AROMATIC"` are normalized to the canonical schema at load time
- maintained notebooks now import chemistry helpers from `abstractgraph_graphicalizer.chem`
- old `conditional_node_field_graph_generator.extensions.molecular` imports have been removed

For API details, see `abstractgraph-graphicalizer/docs/CHEMISTRY.md` in the graphicalizer checkout.
