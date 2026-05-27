"""Internal graph/node embedding pipeline for the graph generator orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from .conditional_node_field_generator import GraphConditioningBatch
from .runtime_utils import verbose_log


@dataclass
class EncodingPipeline:
    """Fit vectorizers and optional SVD projections while preserving node order.

    The node-level vectorizer owns graph-local node row order. Every downstream
    mask, label array, edge pair, and decoded graph must use that same order.
    The transformation chain is:
    raw vectorizer output -> optional TruncatedSVD -> neural model input.
    """

    owner: Any

    @staticmethod
    def to_numpy_2d(matrix) -> np.ndarray:
        if sparse.issparse(matrix):
            return np.asarray(matrix.toarray(), dtype=float)
        array = np.asarray(matrix, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return array

    @staticmethod
    def stack_embedding_rows(embeddings: Sequence[Any]):
        if not embeddings:
            return np.zeros((0, 0), dtype=float)
        if any(sparse.issparse(embedding) for embedding in embeddings):
            return sparse.vstack(
                [
                    embedding if sparse.issparse(embedding) else sparse.csr_matrix(embedding)
                    for embedding in embeddings
                ],
                format="csr",
            )
        return np.vstack([np.asarray(embedding, dtype=float) for embedding in embeddings])

    @staticmethod
    def feature_dimension(matrix) -> int:
        if matrix is None:
            return 0
        if len(matrix.shape) != 2:
            matrix = np.asarray(matrix)
            if matrix.ndim == 1:
                return 1
        return int(matrix.shape[1])

    def resolved_graph_embedding_svd_dimension(self) -> int:
        if self.owner.graph_embedding_svd_dimension is None:
            return int(self.owner.node_embedding_svd_dimension)
        return int(self.owner.graph_embedding_svd_dimension)

    def raw_node_encode(self, graphs: List[nx.Graph]) -> List[Any]:
        if int(self.owner.verbose) >= 3:
            verbose_log(self.owner, f"Node encoding {len(graphs)} graphs", level=3)
        return self.owner.node_graph_vectorizer.transform(graphs)

    def raw_graph_encode(self, graphs: List[nx.Graph]):
        if int(self.owner.verbose) >= 3:
            verbose_log(self.owner, f"Encoding {len(graphs)} graphs", level=3)
        return self.owner.graph_vectorizer.transform(graphs)

    def fit_single_embedding_svd(self, matrix, requested_dimension: int, label: str):
        raw_dimension = self.feature_dimension(matrix)
        requested_dimension = int(requested_dimension)
        if not self.owner.use_embedding_svd:
            return None, raw_dimension, raw_dimension, False
        if requested_dimension < 1:
            raise ValueError(f"{label}_embedding_svd_dimension must be >= 1.")
        if raw_dimension <= 0:
            return None, raw_dimension, raw_dimension, False
        if requested_dimension >= raw_dimension:
            verbose_log(
                self.owner,
                f"Skipping {label} embedding SVD: requested dimension "
                f"{requested_dimension} >= raw dimension {raw_dimension}.",
                level=1,
            )
            return None, raw_dimension, raw_dimension, False
        svd = TruncatedSVD(n_components=requested_dimension, random_state=0)
        svd.fit(matrix)
        verbose_log(
            self.owner,
            f"Fitted {label} embedding SVD: {raw_dimension} -> {requested_dimension}.",
            level=1,
        )
        return svd, raw_dimension, requested_dimension, True

    def fit_embedding_svds(self, raw_node_embeddings_list: List[Any], raw_graph_embeddings) -> None:
        node_matrix = self.stack_embedding_rows(raw_node_embeddings_list)
        graph_matrix = raw_graph_embeddings
        if sparse.issparse(graph_matrix):
            graph_matrix = graph_matrix.tocsr()
        else:
            graph_matrix = np.asarray(graph_matrix, dtype=float)
            if graph_matrix.ndim == 1:
                graph_matrix = graph_matrix.reshape(1, -1)
        (
            self.owner.node_embedding_svd_,
            self.owner.node_embedding_raw_dimension_,
            self.owner.node_embedding_effective_dimension_,
            self.owner.node_embedding_svd_fitted_,
        ) = self.fit_single_embedding_svd(
            node_matrix,
            int(self.owner.node_embedding_svd_dimension),
            "node",
        )
        (
            self.owner.graph_embedding_svd_,
            self.owner.graph_embedding_raw_dimension_,
            self.owner.graph_embedding_effective_dimension_,
            self.owner.graph_embedding_svd_fitted_,
        ) = self.fit_single_embedding_svd(
            graph_matrix,
            self.resolved_graph_embedding_svd_dimension(),
            "graph",
        )

    def compress_node_embeddings(self, raw_node_embeddings_list: List[Any]) -> List[np.ndarray]:
        if not bool(getattr(self.owner, "node_embedding_svd_fitted_", False)):
            return [self.to_numpy_2d(embedding) for embedding in raw_node_embeddings_list]
        return [
            np.asarray(self.owner.node_embedding_svd_.transform(embedding), dtype=float)
            for embedding in raw_node_embeddings_list
        ]

    def compress_graph_embeddings(self, raw_graph_embeddings) -> np.ndarray:
        if not bool(getattr(self.owner, "graph_embedding_svd_fitted_", False)):
            return self.to_numpy_2d(raw_graph_embeddings)
        return np.asarray(self.owner.graph_embedding_svd_.transform(raw_graph_embeddings), dtype=float)

    def build_graph_conditioning_from_raw(
        self,
        graphs: List[nx.Graph],
        raw_graph_embeddings,
    ) -> GraphConditioningBatch:
        graph_embeddings = self.compress_graph_embeddings(raw_graph_embeddings)
        node_counts = np.asarray([graph.number_of_nodes() for graph in graphs], dtype=np.int64)
        edge_counts = np.asarray([graph.number_of_edges() for graph in graphs], dtype=np.int64)
        return GraphConditioningBatch(
            graph_embeddings=graph_embeddings,
            node_counts=node_counts,
            edge_counts=edge_counts,
        )

    def node_encode(self, graphs: List[nx.Graph]) -> List[np.ndarray]:
        return self.compress_node_embeddings(self.raw_node_encode(graphs))

    def graph_encode(self, graphs: List[nx.Graph]) -> GraphConditioningBatch:
        return self.build_graph_conditioning_from_raw(graphs, self.raw_graph_encode(graphs))

    def encode(self, graphs: List[nx.Graph]) -> Tuple[List[np.ndarray], GraphConditioningBatch]:
        raw_node_embeddings_list = self.raw_node_encode(graphs)
        raw_graph_embeddings = self.raw_graph_encode(graphs)
        return (
            self.compress_node_embeddings(raw_node_embeddings_list),
            self.build_graph_conditioning_from_raw(graphs, raw_graph_embeddings),
        )
