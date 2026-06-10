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

    @staticmethod
    def matrix_summary(matrix) -> str:
        shape = getattr(matrix, "shape", None)
        if sparse.issparse(matrix):
            nnz = int(matrix.nnz)
            total = int(matrix.shape[0]) * int(matrix.shape[1]) if len(matrix.shape) == 2 else 0
            density = (nnz / total) if total else 0.0
            return f"shape={shape}, sparse=True, nnz={nnz}, density={density:.6f}"
        array = np.asarray(matrix)
        size_mb = array.nbytes / (1024.0 * 1024.0)
        return f"shape={shape}, sparse=False, dtype={array.dtype}, size={size_mb:.1f} MiB"

    @staticmethod
    def row_count(matrix) -> int:
        shape = getattr(matrix, "shape", None)
        if shape is None or len(shape) == 0:
            return 0
        return int(shape[0])

    def resolved_graph_embedding_svd_dimension(self) -> int:
        if self.owner.graph_embedding_svd_dimension is None:
            return int(self.owner.node_embedding_svd_dimension)
        return int(self.owner.graph_embedding_svd_dimension)

    def svd_fit_max_rows(self) -> Optional[int]:
        value = getattr(self.owner, "embedding_svd_fit_max_rows", None)
        if value is None:
            return None
        value = int(value)
        if value < 1:
            return None
        return value

    def svd_transform_batch_size(self) -> Optional[int]:
        value = getattr(self.owner, "embedding_svd_transform_batch_size", None)
        if value is None:
            return None
        value = int(value)
        if value < 1:
            return None
        return value

    def svd_fit_random_state(self, label: str) -> int:
        base_seed = int(getattr(self.owner, "embedding_svd_fit_random_state", 0))
        return base_seed + (1 if label == "graph" else 0)

    def sample_svd_fit_rows(self, matrix, requested_dimension: int, label: str):
        max_rows = self.svd_fit_max_rows()
        row_count = self.row_count(matrix)
        if max_rows is None or row_count <= max_rows:
            return matrix
        min_rows = int(requested_dimension) + 1
        if max_rows < min_rows:
            raise ValueError(
                "embedding_svd_fit_max_rows must be greater than the requested "
                f"{label}_embedding_svd_dimension ({requested_dimension})."
            )
        rng = np.random.default_rng(self.svd_fit_random_state(label))
        row_indices = np.sort(rng.choice(row_count, size=max_rows, replace=False))
        sampled_matrix = matrix[row_indices]
        verbose_log(
            self.owner,
            f"Sampled {label} embedding SVD fit rows: {row_count} -> {max_rows}.",
            level=1,
        )
        return sampled_matrix

    def transform_with_optional_batches(self, svd, matrix, label: str) -> np.ndarray:
        batch_size = self.svd_transform_batch_size()
        row_count = self.row_count(matrix)
        if batch_size is None or row_count <= batch_size:
            return np.asarray(svd.transform(matrix), dtype=float)
        verbose_log(
            self.owner,
            f"Projecting {label} embeddings in batches of {batch_size} rows "
            f"({row_count} total rows).",
            level=1,
        )
        chunks = []
        for start_idx in range(0, row_count, batch_size):
            end_idx = min(start_idx + batch_size, row_count)
            chunks.append(np.asarray(svd.transform(matrix[start_idx:end_idx]), dtype=float))
        return np.vstack(chunks) if chunks else np.zeros((0, int(svd.n_components)), dtype=float)

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
        fit_matrix = self.sample_svd_fit_rows(matrix, requested_dimension, label)
        verbose_log(
            self.owner,
            f"Fitting {label} embedding SVD on {self.matrix_summary(fit_matrix)} "
            f"to {requested_dimension} dimensions.",
            level=1,
        )
        svd = TruncatedSVD(
            n_components=requested_dimension,
            n_iter=int(getattr(self.owner, "embedding_svd_n_iter", 2)),
            n_oversamples=int(getattr(self.owner, "embedding_svd_n_oversamples", 5)),
            random_state=0,
        )
        svd.fit(fit_matrix)
        verbose_log(
            self.owner,
            f"Fitted {label} embedding SVD: {raw_dimension} -> {requested_dimension}.",
            level=1,
        )
        return svd, raw_dimension, requested_dimension, True

    def fit_embedding_svds(self, raw_node_embeddings_list: List[Any], raw_graph_embeddings) -> None:
        verbose_log(self.owner, "Stacking raw node embeddings for SVD.", level=1)
        node_matrix = self.stack_embedding_rows(raw_node_embeddings_list)
        verbose_log(
            self.owner,
            f"Stacked node embedding matrix: {self.matrix_summary(node_matrix)}.",
            level=1,
        )
        graph_matrix = raw_graph_embeddings
        if sparse.issparse(graph_matrix):
            graph_matrix = graph_matrix.tocsr()
        else:
            graph_matrix = np.asarray(graph_matrix, dtype=float)
            if graph_matrix.ndim == 1:
                graph_matrix = graph_matrix.reshape(1, -1)
        verbose_log(
            self.owner,
            f"Prepared graph embedding matrix: {self.matrix_summary(graph_matrix)}.",
            level=1,
        )
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
            self.transform_with_optional_batches(self.owner.node_embedding_svd_, embedding, "node")
            for embedding in raw_node_embeddings_list
        ]

    def compress_graph_embeddings(self, raw_graph_embeddings) -> np.ndarray:
        if not bool(getattr(self.owner, "graph_embedding_svd_fitted_", False)):
            return self.to_numpy_2d(raw_graph_embeddings)
        return self.transform_with_optional_batches(
            self.owner.graph_embedding_svd_,
            raw_graph_embeddings,
            "graph",
        )

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
