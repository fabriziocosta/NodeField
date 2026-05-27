"""Node-generation batch assembly for graph generator training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import networkx as nx
import numpy as np

from .conditional_node_field_generator import NodeGenerationBatch


@dataclass
class NodeBatchBuilder:
    owner: Any

    def build_node_batch(
        self,
        graphs: List[nx.Graph],
        node_embeddings_list: List[np.ndarray],
        node_label_targets: Optional[List[np.ndarray]] = None,
        edge_pairs: Optional[List[Tuple[int, int, int]]] = None,
        edge_targets: Optional[np.ndarray] = None,
        edge_label_pairs: Optional[List[Tuple[int, int, int]]] = None,
        edge_label_targets: Optional[np.ndarray] = None,
        auxiliary_edge_pairs: Optional[List[Tuple[int, int, int]]] = None,
        auxiliary_edge_targets: Optional[np.ndarray] = None,
    ) -> NodeGenerationBatch:
        """Assemble padded node targets using graph node iteration order.

        The node vectorizer row order, graph node iteration order, masks, labels,
        edge pairs, and decoded graph assembly must stay aligned.
        """
        frozen_num_rows = None
        if self.owner.conditional_node_generator_model is not None:
            frozen_num_rows = getattr(
                self.owner.conditional_node_generator_model,
                "number_of_rows_per_example",
                None,
            )
        max_num_rows = (
            int(frozen_num_rows)
            if frozen_num_rows is not None
            else max(emb.shape[0] for emb in node_embeddings_list)
        )
        node_presence_mask = np.zeros((len(graphs), max_num_rows), dtype=bool)
        node_degree_targets = np.zeros((len(graphs), max_num_rows), dtype=np.int64)
        for graph_idx, graph in enumerate(graphs):
            nodes = list(graph.nodes())
            if len(nodes) > max_num_rows:
                raise ValueError(
                    "Graph exceeds the configured number_of_rows_per_example "
                    f"({len(nodes)} > {max_num_rows})."
                )
            node_presence_mask[graph_idx, :len(nodes)] = True
            node_degree_targets[graph_idx, :len(nodes)] = np.asarray(
                [graph.degree(node) for node in nodes],
                dtype=np.int64,
            )
        return NodeGenerationBatch(
            node_embeddings_list=node_embeddings_list,
            node_presence_mask=node_presence_mask,
            node_degree_targets=node_degree_targets,
            node_label_targets=node_label_targets,
            edge_pairs=edge_pairs,
            edge_targets=edge_targets,
            edge_label_pairs=edge_label_pairs,
            edge_label_targets=edge_label_targets,
            auxiliary_edge_pairs=auxiliary_edge_pairs,
            auxiliary_edge_targets=auxiliary_edge_targets,
        )

    def build_training_node_batch(
        self,
        graphs: List[nx.Graph],
        *,
        node_embeddings_list: List[np.ndarray],
        node_label_targets: List[np.ndarray],
        edge_label_targets: Optional[np.ndarray],
        edge_label_pairs: Optional[List[Tuple[int, int, int]]],
        supervision_plan,
        log_details: bool = True,
    ) -> NodeGenerationBatch:
        edge_pairs_for_cond_gen = None
        edge_targets_for_cond_gen = None
        auxiliary_edge_pairs_for_cond_gen = None
        auxiliary_edge_targets_for_cond_gen = None
        decoder_verbose = None
        if (
            not log_details
            and self.owner.graph_decoder is not None
            and hasattr(self.owner.graph_decoder, "verbose")
        ):
            decoder_verbose = getattr(self.owner.graph_decoder, "verbose")
            self.owner.graph_decoder.verbose = False
        if supervision_plan.direct_edges.enabled:
            if self.owner.graph_decoder is None:
                raise RuntimeError("Locality supervision requested but graph_decoder is None.")
            if log_details:
                self.owner._log_supervision_plan(supervision_plan)
            try:
                edge_targets_for_cond_gen, edge_pairs_for_cond_gen = (
                    self.owner.graph_decoder.compute_edge_supervision(
                        graphs,
                        node_embeddings_list,
                        locality_sample_fraction=self.owner.locality_sample_fraction,
                        negative_sample_factor=self.owner.negative_sample_factor,
                        locality_sampling_strategy=self.owner.locality_sampling_strategy,
                        locality_target_positive_ratio=self.owner.locality_target_positive_ratio,
                        horizon=1,
                        supervision_name="direct_edge",
                    )
                )
                if supervision_plan.auxiliary_locality.enabled:
                    auxiliary_edge_targets_for_cond_gen, auxiliary_edge_pairs_for_cond_gen = (
                        self.owner.graph_decoder.compute_edge_supervision(
                            graphs,
                            node_embeddings_list,
                            locality_sample_fraction=self.owner.locality_sample_fraction,
                            negative_sample_factor=self.owner.negative_sample_factor,
                            locality_sampling_strategy=self.owner.locality_sampling_strategy,
                            locality_target_positive_ratio=self.owner.locality_target_positive_ratio,
                            horizon=supervision_plan.auxiliary_locality.horizon,
                            supervision_name="aux_locality",
                        )
                    )
            finally:
                if decoder_verbose is not None:
                    self.owner.graph_decoder.verbose = decoder_verbose
        else:
            if log_details:
                self.owner._log_supervision_plan(supervision_plan)
        return self.build_node_batch(
            graphs,
            node_embeddings_list,
            node_label_targets=node_label_targets if supervision_plan.node_labels.enabled else None,
            edge_pairs=edge_pairs_for_cond_gen,
            edge_targets=edge_targets_for_cond_gen,
            edge_label_pairs=edge_label_pairs if supervision_plan.edge_labels.enabled else None,
            edge_label_targets=edge_label_targets if supervision_plan.edge_labels.enabled else None,
            auxiliary_edge_pairs=auxiliary_edge_pairs_for_cond_gen,
            auxiliary_edge_targets=auxiliary_edge_targets_for_cond_gen,
        )
