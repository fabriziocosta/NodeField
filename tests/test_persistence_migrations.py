import numpy as np

from conditional_node_field_graph_generator.persistence_migrations import (
    GRAPH_GENERATOR_PERSISTENCE_VERSION,
    apply_graph_generator_persistence_migrations,
)


class _LegacyNodeModel:
    node_label_classes_ = ["C", "N"]
    node_label_to_index_ = {"C": 0, "N": 1}
    edge_label_classes_ = ["1", "2"]
    edge_label_to_index_ = {"1": 0, "2": 1}


class _LegacyGenerator:
    _DEFAULT_FEASIBILITY_ORACLE_CANDIDATES_PER_ATTEMPT = 2

    def __init__(self):
        self.use_feasibility_oracle = False
        self.conditional_node_generator_model = _LegacyNodeModel()
        self.restore_calls = 0

    def _restore_label_vocab_metadata_from_node_model(self):
        self.restore_calls += 1
        if getattr(self, "node_label_classes_", None) is None:
            self.node_label_classes_ = self.conditional_node_generator_model.node_label_classes_
        if getattr(self, "node_label_to_index_", None) is None:
            self.node_label_to_index_ = self.conditional_node_generator_model.node_label_to_index_
        if getattr(self, "edge_label_classes_", None) is None:
            self.edge_label_classes_ = self.conditional_node_generator_model.edge_label_classes_
        if getattr(self, "edge_label_to_index_", None) is None:
            self.edge_label_to_index_ = self.conditional_node_generator_model.edge_label_to_index_


def test_apply_graph_generator_persistence_migrations_repairs_legacy_state():
    generator = _LegacyGenerator()

    migrated = apply_graph_generator_persistence_migrations(generator)

    assert migrated is generator
    assert generator._persistence_schema_version == GRAPH_GENERATOR_PERSISTENCE_VERSION
    assert generator.feasibility_oracle_candidates_per_attempt == 0
    assert generator.oracle_edge_memory_penalty == 0.5
    assert generator.oracle_edge_memory_update == 1.0
    assert generator.oracle_edge_memory_decay == 1.0
    assert generator.oracle_edge_memory_clip == 5.0
    assert generator.restore_calls == 1
    assert np.asarray(generator.node_label_classes_, dtype=object).tolist() == ["C", "N"]
    assert np.asarray(generator.edge_label_classes_, dtype=object).tolist() == ["single", "double"]
    assert generator.edge_label_to_index_ == {"single": 0, "double": 1}
