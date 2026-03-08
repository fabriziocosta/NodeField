# Molecular Extension

This section documents the chemistry-oriented extension layer for NodeField.

The molecular extension is useful for:
- molecule/graph conversion
- PubChem assay loading
- ZINC download, caching, and preprocessing
- molecule drawing in notebooks
- compatibility with older molecule-oriented notebook workflows

Primary entry points live under:
- [`conditional_node_field_graph_generator/extensions/molecular/__init__.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/conditional_node_field_graph_generator/extensions/molecular/__init__.py)
- [`conditional_node_field_graph_generator/extensions/molecular/conversion.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/conditional_node_field_graph_generator/extensions/molecular/conversion.py)
- [`conditional_node_field_graph_generator/extensions/molecular/datasets.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/conditional_node_field_graph_generator/extensions/molecular/datasets.py)
- [`conditional_node_field_graph_generator/extensions/molecular/visualization.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/conditional_node_field_graph_generator/extensions/molecular/visualization.py)

Detailed module documentation:
- [`docs/extensions/molecular/MOLECULAR_GRAPH_UTILS_README.md`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/docs/extensions/molecular/MOLECULAR_GRAPH_UTILS_README.md)

Boundary:
- this extension is not required for the core NodeField model
- it exists to support chemistry-specific datasets and notebook workflows
