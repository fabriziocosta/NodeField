"""Versioned load-time migrations for persisted NodeField objects."""

from __future__ import annotations

from typing import Any

import numpy as np
from abstractgraph_graphicalizer.chem import normalize_bond_label


GRAPH_GENERATOR_PERSISTENCE_VERSION = 3


def get_graph_generator_persistence_version(graph_generator: Any) -> int:
    return int(getattr(graph_generator, "_persistence_schema_version", 0))


def _migrate_graph_generator_v1(graph_generator: Any) -> None:
    if not hasattr(graph_generator, "feasibility_oracle_candidates_per_attempt"):
        legacy_use_oracle = getattr(graph_generator, "use_feasibility_oracle", True)
        default_budget = int(getattr(graph_generator, "_DEFAULT_FEASIBILITY_ORACLE_CANDIDATES_PER_ATTEMPT", 2))
        graph_generator.feasibility_oracle_candidates_per_attempt = default_budget if bool(legacy_use_oracle) else 0
    graph_generator.feasibility_oracle_candidates_per_attempt = max(
        0,
        int(graph_generator.feasibility_oracle_candidates_per_attempt),
    )
    if not hasattr(graph_generator, "oracle_edge_memory_penalty"):
        graph_generator.oracle_edge_memory_penalty = 0.5
    if not hasattr(graph_generator, "oracle_edge_memory_update"):
        graph_generator.oracle_edge_memory_update = 1.0
    if not hasattr(graph_generator, "oracle_edge_memory_decay"):
        graph_generator.oracle_edge_memory_decay = 1.0
    if not hasattr(graph_generator, "oracle_edge_memory_clip"):
        graph_generator.oracle_edge_memory_clip = 5.0


def _migrate_graph_generator_v2(graph_generator: Any) -> None:
    if not hasattr(graph_generator, "node_label_classes_"):
        graph_generator.node_label_classes_ = None
    if not hasattr(graph_generator, "node_label_to_index_"):
        graph_generator.node_label_to_index_ = None
    if not hasattr(graph_generator, "edge_label_classes_"):
        graph_generator.edge_label_classes_ = None
    if not hasattr(graph_generator, "edge_label_to_index_"):
        graph_generator.edge_label_to_index_ = None

    restore = getattr(graph_generator, "_restore_label_vocab_metadata_from_node_model", None)
    if callable(restore):
        restore()

    if getattr(graph_generator, "node_label_classes_", None) is not None:
        graph_generator.node_label_classes_ = np.asarray(graph_generator.node_label_classes_, dtype=object)
    if getattr(graph_generator, "edge_label_classes_", None) is not None:
        graph_generator.edge_label_classes_ = np.asarray(graph_generator.edge_label_classes_, dtype=object)


def _migrate_graph_generator_v3(graph_generator: Any) -> None:
    edge_label_classes = getattr(graph_generator, "edge_label_classes_", None)
    if edge_label_classes is not None:
        graph_generator.edge_label_classes_ = np.asarray(
            [normalize_bond_label(label) for label in edge_label_classes],
            dtype=object,
        )

    edge_label_to_index = getattr(graph_generator, "edge_label_to_index_", None)
    if edge_label_to_index is not None:
        graph_generator.edge_label_to_index_ = {
            normalize_bond_label(label): int(index)
            for label, index in edge_label_to_index.items()
        }


def apply_graph_generator_persistence_migrations(graph_generator: Any) -> Any:
    version = get_graph_generator_persistence_version(graph_generator)
    if version < 1:
        _migrate_graph_generator_v1(graph_generator)
        version = 1
    if version < 2:
        _migrate_graph_generator_v2(graph_generator)
        version = 2
    if version < 3:
        _migrate_graph_generator_v3(graph_generator)
        version = 3
    graph_generator._persistence_schema_version = version
    return graph_generator
