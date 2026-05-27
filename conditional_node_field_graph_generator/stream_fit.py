"""Streaming fit helpers for the graph-generator orchestrator."""

from __future__ import annotations

from typing import List, Optional

import networkx as nx

from .runtime_utils import get_runtime_logger, run_with_fork_timeout, verbose_log


DEFAULT_DUMMY_NODE_LABEL = "__dummy_node_label__"
logger = get_runtime_logger(__name__)


class _StreamTransformError(RuntimeError):
    pass


class _StreamSupervisionError(RuntimeError):
    pass


class _StreamBatchTimeoutError(RuntimeError):
    pass


def _prepare_stream_training_payload_worker(graph_generator, graphs: List[nx.Graph]):
    return graph_generator._prepare_stream_training_payload(graphs)


class StreamFitService:
    """Owns stream-specific validation, batch preparation, and counters."""

    def __init__(self, owner):
        self.owner = owner

    def stream_rejection_reason(self, graph: nx.Graph) -> Optional[str]:
        node_model = self.owner.conditional_node_generator_model
        max_rows = getattr(node_model, "number_of_rows_per_example", None)
        if max_rows is not None and graph.number_of_nodes() > int(max_rows):
            return "too_large"
        node_label_vocab = getattr(node_model, "node_label_to_index_", None)
        if node_label_vocab:
            for node in graph.nodes():
                label = graph.nodes[node].get("label", DEFAULT_DUMMY_NODE_LABEL)
                if label not in node_label_vocab:
                    return "unknown_node_label"
        supervision_plan = getattr(self.owner, "supervision_plan_", None)
        edge_label_vocab = getattr(node_model, "edge_label_to_index_", None)
        edge_label_mode = None if supervision_plan is None else getattr(supervision_plan.edge_labels, "mode", None)
        if edge_label_vocab and edge_label_mode == "learned":
            for _, _, attrs in graph.edges(data=True):
                if "label" not in attrs or attrs["label"] not in edge_label_vocab:
                    return "unknown_edge_label"
        return None

    def increment_stream_skip(self, reason: str) -> None:
        owner = self.owner
        owner.stream_training_skipped_ += 1
        owner.stream_epoch_training_skipped_ += 1
        if reason == "too_large":
            owner.stream_skipped_too_large_ += 1
        elif reason == "unknown_node_label":
            owner.stream_skipped_unknown_node_label_ += 1
        elif reason == "unknown_edge_label":
            owner.stream_skipped_unknown_edge_label_ += 1
        elif reason == "transform_error":
            owner.stream_skipped_transform_error_ += 1
        elif reason == "supervision_error":
            owner.stream_skipped_supervision_error_ += 1

    def log_stream_skip(self, reason: str, graph: nx.Graph) -> None:
        if int(self.owner.verbose) < 2:
            return
        verbose_log(
            self.owner,
            "Skipping streamed graph "
            f"(nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}) due to {reason}.",
            level=2,
        )

    def finalize_stream_fit_stats(self) -> None:
        owner = self.owner
        with owner._ensure_stream_runtime_lock():
            denominator = max(1, int(owner.stream_training_seen_))
            owner.stream_acceptance_rate_ = float(owner.stream_training_accepted_) / float(denominator)
            owner.warmup_schema_frozen_ = True
            if int(owner.verbose) >= 1:
                verbose_log(
                    owner,
                    "Streaming fit summary: "
                    f"seen={owner.stream_seen_}, warmup={owner.stream_warmup_count_}, "
                    f"train_seen={owner.stream_training_seen_}, accepted={owner.stream_training_accepted_}, "
                    f"skipped={owner.stream_training_skipped_}, acceptance_rate={owner.stream_acceptance_rate_:.1%}.",
                )
            if owner.stream_training_seen_ > 0 and owner.stream_acceptance_rate_ < 0.5:
                logger.warning(
                    "Low streamed training acceptance rate: %.1f%% (%d/%d accepted).",
                    100.0 * owner.stream_acceptance_rate_,
                    owner.stream_training_accepted_,
                    owner.stream_training_seen_,
                )

    def prepare_stream_training_payload(self, graphs: List[nx.Graph]):
        owner = self.owner
        with owner._ensure_stream_runtime_lock():
            supervision_plan = getattr(owner, "supervision_plan_", None)
            if supervision_plan is None:
                raise RuntimeError("supervision_plan_ is not initialized.")
            try:
                node_embeddings_list, graph_conditioning = owner.encode(graphs)
            except Exception as exc:
                raise _StreamTransformError("Failed to encode streamed graphs under the frozen warmup schema.") from exc
            try:
                node_label_targets = owner.graphs_to_node_label_targets(graphs)
                edge_label_targets, edge_label_pairs = owner.graphs_to_edge_label_targets(graphs)
                node_batch = owner._build_training_node_batch(
                    graphs,
                    node_embeddings_list=node_embeddings_list,
                    node_label_targets=node_label_targets,
                    edge_label_targets=edge_label_targets,
                    edge_label_pairs=edge_label_pairs,
                    supervision_plan=supervision_plan,
                    log_details=False,
                )
                return owner.conditional_node_generator_model._build_processed_training_payload(
                    node_batch=node_batch,
                    graph_conditioning=graph_conditioning,
                    targets=None,
                )
            except Exception as exc:
                raise _StreamSupervisionError(
                    "Failed to derive streamed supervision under the frozen warmup schema."
                ) from exc

    def prepare_stream_training_batch(self, graphs: List[nx.Graph]):
        payload = self.prepare_stream_training_payload(graphs)
        return self.owner.conditional_node_generator_model._collate_processed_payload(payload)

    def prepare_stream_training_batch_with_timeout(self, graphs: List[nx.Graph]):
        owner = self.owner
        timeout_seconds = getattr(owner, "stream_batch_timeout_seconds", None)
        if timeout_seconds is None:
            return self.prepare_stream_training_batch(graphs)
        try:
            payload = run_with_fork_timeout(
                _prepare_stream_training_payload_worker,
                owner,
                graphs,
                timeout_seconds=float(timeout_seconds),
            )
            return owner.conditional_node_generator_model._collate_processed_payload(payload)
        except TimeoutError as exc:
            raise _StreamBatchTimeoutError(
                f"Streamed batch preparation exceeded {float(timeout_seconds):.1f}s."
            ) from exc
