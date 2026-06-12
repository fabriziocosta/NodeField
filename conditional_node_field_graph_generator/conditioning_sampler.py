"""Sampling helpers for cached graph-level conditioning rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from .conditional_node_field_generator import GraphConditioningBatch


@dataclass
class ConditioningSampler:
    owner: Any

    @staticmethod
    def sample_conditioning_rows(source: GraphConditioningBatch, indices: np.ndarray) -> GraphConditioningBatch:
        idx = np.asarray(indices, dtype=np.int64)
        return source.take(idx)

    @staticmethod
    def interpolated_conditioning_from_pair(
        conditioning: GraphConditioningBatch,
        first_idx: int,
        second_idx: int,
        t: float,
    ) -> Tuple[np.ndarray, np.int64, np.int64]:
        graph_embeddings = np.asarray(conditioning.graph_embeddings, dtype=float)
        node_counts = np.asarray(conditioning.node_counts, dtype=float)
        edge_counts = np.asarray(conditioning.edge_counts, dtype=float)

        interpolated_embedding = (1.0 - t) * graph_embeddings[first_idx] + t * graph_embeddings[second_idx]
        interpolated_node_count = np.int64(max(1, int(np.rint((1.0 - t) * node_counts[first_idx] + t * node_counts[second_idx]))))
        interpolated_edge_count = np.int64(max(0, int(np.rint((1.0 - t) * edge_counts[first_idx] + t * edge_counts[second_idx]))))
        return interpolated_embedding, interpolated_node_count, interpolated_edge_count

    def sample_conditions(
        self,
        n_samples: int,
        interpolate_between_n_samples: Optional[int] = None,
    ) -> GraphConditioningBatch:
        conditioning = self.owner._require_training_graph_conditioning()
        if interpolate_between_n_samples is not None and int(interpolate_between_n_samples) < 2:
            raise ValueError("interpolate_between_n_samples must be >= 2 when provided.")

        n_training = len(conditioning)
        if interpolate_between_n_samples is None or n_training == 1:
            sample_indices = np.random.choice(n_training, size=int(n_samples), replace=True)
            return self.sample_conditioning_rows(conditioning, sample_indices)

        subset_size = min(int(interpolate_between_n_samples), n_training)
        if subset_size < 2:
            sample_indices = np.random.choice(n_training, size=int(n_samples), replace=True)
            return self.sample_conditioning_rows(conditioning, sample_indices)

        graph_embeddings = np.asarray(conditioning.graph_embeddings, dtype=float)
        sampled_embeddings = []
        sampled_node_counts = []
        sampled_edge_counts = []

        for _ in range(int(n_samples)):
            candidate_indices = np.random.choice(n_training, size=subset_size, replace=False)
            if len(candidate_indices) < 2:
                fallback_idx = int(np.random.choice(n_training))
                direct_conditioning = self.sample_conditioning_rows(
                    conditioning,
                    np.asarray([fallback_idx], dtype=np.int64),
                )
                sampled_embeddings.append(np.asarray(direct_conditioning.graph_embeddings[0], dtype=float))
                sampled_node_counts.append(np.int64(direct_conditioning.node_counts[0]))
                sampled_edge_counts.append(np.int64(direct_conditioning.edge_counts[0]))
                continue

            pair_indices = []
            pair_weights = []
            raw_pair_cosines = []
            for local_i in range(len(candidate_indices)):
                for local_j in range(local_i + 1, len(candidate_indices)):
                    first_idx = int(candidate_indices[local_i])
                    second_idx = int(candidate_indices[local_j])
                    first_embedding = graph_embeddings[first_idx]
                    second_embedding = graph_embeddings[second_idx]
                    denom = float(np.linalg.norm(first_embedding) * np.linalg.norm(second_embedding))
                    cosine = 0.0 if denom == 0.0 else float(np.dot(first_embedding, second_embedding) / denom)
                    pair_indices.append((first_idx, second_idx))
                    raw_pair_cosines.append(cosine)
                    pair_weights.append(max(cosine, 0.0))

            pair_weights_array = np.asarray(pair_weights, dtype=float)
            if np.all(pair_weights_array == 0.0):
                raw_pair_cosines_array = np.asarray(raw_pair_cosines, dtype=float)
                max_cosine = float(np.max(raw_pair_cosines_array))
                candidate_pair_choices = np.flatnonzero(np.isclose(raw_pair_cosines_array, max_cosine))
                pair_choice = int(np.random.choice(candidate_pair_choices))
            else:
                pair_probabilities = pair_weights_array / pair_weights_array.sum()
                pair_choice = int(np.random.choice(len(pair_indices), p=pair_probabilities))
            first_idx, second_idx = pair_indices[pair_choice]
            t = float(np.random.uniform(0.0, 1.0))
            interpolated_embedding, interpolated_node_count, interpolated_edge_count = (
                self.interpolated_conditioning_from_pair(
                    conditioning,
                    first_idx,
                    second_idx,
                    t,
                )
            )
            sampled_embeddings.append(interpolated_embedding)
            sampled_node_counts.append(interpolated_node_count)
            sampled_edge_counts.append(interpolated_edge_count)

        return GraphConditioningBatch(
            graph_embeddings=np.asarray(sampled_embeddings, dtype=float),
            node_counts=np.asarray(sampled_node_counts, dtype=np.int64),
            edge_counts=np.asarray(sampled_edge_counts, dtype=np.int64),
        )
