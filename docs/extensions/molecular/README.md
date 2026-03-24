# Molecular Support

NodeField no longer ships a local molecular extension.

All maintained chemistry support now lives in the abstractgraph ecosystem under
`abstractgraph_graphicalizer.chem`, including:
- molecule graph conversion
- molecule drawing
- PubChem assay loading
- ZINC download, caching, and corpus loading
- small supervised dataset shaping helpers used by notebooks

Use:
- `from abstractgraph_graphicalizer.chem import ...`

Reference:
- [`docs/extensions/molecular/MOLECULAR_GRAPH_UTILS_README.md`](/home/fabrizio/code/NodeField/docs/extensions/molecular/MOLECULAR_GRAPH_UTILS_README.md)
- `abstractgraph-graphicalizer/docs/CHEMISTRY.md` in the graphicalizer checkout
