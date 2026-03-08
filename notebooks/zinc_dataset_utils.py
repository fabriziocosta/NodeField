"""Compatibility wrapper for molecule and ZINC dataset utilities."""

from conditional_node_field_graph_generator.molecular_graph_utils import (
    DEFAULT_ZINC_TARGET_COLUMNS,
    ZINC_250K_URL,
    build_zinc_graph_corpus,
    download_zinc_dataset,
    extract_zinc_targets,
    load_zinc_graph_dataset,
    smiles_to_networkx_molecule,
)

__all__ = [
    "DEFAULT_ZINC_TARGET_COLUMNS",
    "ZINC_250K_URL",
    "build_zinc_graph_corpus",
    "download_zinc_dataset",
    "extract_zinc_targets",
    "load_zinc_graph_dataset",
    "smiles_to_networkx_molecule",
]
