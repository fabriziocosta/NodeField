"""Training orchestration for the node-field model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

from .graph_generator_state import CheckpointPolicy, MetricsPolicy, TrainingPolicy
from .metrics_collection import (
    GraphGeneratorBatchAndEpochSnapshotCallback,
    GraphGeneratorBatchSnapshotCallback,
    GraphGeneratorEpochSnapshotCallback,
    GraphGeneratorTrainingSampleCallback,
    MetricsLogger,
)
from .runtime_utils import verbose_log, run_trainer_fit
from .training_policy import (
    build_training_callbacks,
    create_trainer,
    format_restored_checkpoint_summary,
    suppress_output,
)
from .runtime_utils import get_runtime_logger

logger = get_runtime_logger(__name__)


@dataclass
class TrainingCoordinator:
    owner: Any

    def _build_snapshot_callback(self, snapshot_frequency: Optional[str]):
        snapshot_owner = getattr(self.owner, "_graph_generator_snapshot_owner", None)
        if snapshot_owner is None or getattr(snapshot_owner, "model_name", None) is None:
            return None
        if snapshot_frequency == "batch":
            return GraphGeneratorBatchAndEpochSnapshotCallback(snapshot_owner)
        if snapshot_frequency == "epoch":
            return GraphGeneratorEpochSnapshotCallback(snapshot_owner)
        return None

    def _build_sample_progress_callback(self, snapshot_frequency: Optional[str]):
        if snapshot_frequency != "epoch":
            return None
        if not bool(getattr(self.owner, "_graph_generator_sample_progress_enabled", False)):
            return None
        snapshot_owner = getattr(self.owner, "_graph_generator_snapshot_owner", None)
        if snapshot_owner is None:
            return None
        output_path = getattr(self.owner, "_graph_generator_sample_progress_pdf_path", None)
        if output_path is None:
            return None
        return GraphGeneratorTrainingSampleCallback(
            snapshot_owner,
            n_samples=int(getattr(self.owner, "_graph_generator_sample_progress_n_samples", 7)),
            every_n_epochs=int(
                getattr(self.owner, "_graph_generator_sample_progress_every_n_epochs", 1)
            ),
            output_path=output_path,
        )

    def run_training(
        self,
        train_loader,
        val_loader,
        *,
        ckpt_path: Optional[str],
        context: str,
        train_loader_length: int,
        training_policy: TrainingPolicy,
        checkpoint_policy: CheckpointPolicy,
        metrics_policy: MetricsPolicy,
        snapshot_frequency: Optional[str] = "epoch",
    ) -> None:
        owner = self.owner
        snapshot_callback = self._build_snapshot_callback(snapshot_frequency)
        sample_progress_callback = self._build_sample_progress_callback(snapshot_frequency)
        callbacks, checkpoint_dir, checkpoint_callback = build_training_callbacks(
            generator_name=owner.__class__.__name__,
            checkpoint_root_dir=checkpoint_policy.checkpoint_root_dir,
            early_stopping_monitor=training_policy.early_stopping_monitor,
            early_stopping_mode=training_policy.early_stopping_mode,
            enable_early_stopping=training_policy.enable_early_stopping,
            early_stopping_patience=training_policy.early_stopping_patience,
            early_stopping_min_delta=training_policy.early_stopping_min_delta,
            metrics_logger=MetricsLogger(),
            epoch_snapshot_callback=snapshot_callback,
            sample_progress_callback=sample_progress_callback,
        )
        if owner.model_name is not None:
            verbose_log(owner, f"Save target model_name={owner.model_name} model_dir={owner.model_dir}")
        verbose_log(owner, f"Writing checkpoints to {checkpoint_dir}")
        trainer = create_trainer(
            maximum_epochs=training_policy.maximum_epochs,
            callbacks=callbacks,
            artifact_root_dir=owner.artifact_root_dir,
            train_loader_length=max(1, int(train_loader_length)),
        )
        if not owner.verbose and training_policy.suppress_non_batch_output:
            with suppress_output():
                run_trainer_fit(trainer, owner.model, train_loader, val_loader, context=context, ckpt_path=ckpt_path)
        else:
            run_trainer_fit(trainer, owner.model, train_loader, val_loader, context=context, ckpt_path=ckpt_path)

        owner.best_checkpoint_path_ = checkpoint_callback.best_model_path or None
        best_score = checkpoint_callback.best_model_score
        owner.best_checkpoint_score_ = float(best_score.item()) if best_score is not None else None
        if checkpoint_policy.restore_best_checkpoint and owner.best_checkpoint_path_:
            checkpoint = torch.load(owner.best_checkpoint_path_, map_location=owner.device, weights_only=False)
            best_epoch = checkpoint.get("epoch")
            owner.best_checkpoint_epoch_ = int(best_epoch) if best_epoch is not None else None
            state_dict = checkpoint.get("state_dict", checkpoint)
            owner.model.load_state_dict(state_dict)
            owner.model.to(owner.device)
            if int(owner.verbose) >= 1:
                stopped_epoch = int(getattr(trainer, "current_epoch", -1)) + 1
                raw_best_val_node_field_loss = None
                if (
                    owner.best_checkpoint_epoch_ is not None
                    and hasattr(owner.model, "val_node_field")
                    and owner.best_checkpoint_epoch_ < len(owner.model.val_node_field)
                ):
                    raw_best_val_node_field_loss = float(owner.model.val_node_field[owner.best_checkpoint_epoch_])
                verbose_log(
                    owner,
                    format_restored_checkpoint_summary(
                        early_stopping_monitor=training_policy.early_stopping_monitor,
                        best_checkpoint_score=owner.best_checkpoint_score_,
                        best_checkpoint_epoch=owner.best_checkpoint_epoch_,
                        raw_best_val_node_field_loss=raw_best_val_node_field_loss,
                        stopped_epoch=stopped_epoch,
                    ),
                    level=1,
                )
                verbose_log(owner, f"  path={owner.best_checkpoint_path_}", level=1)
        if metrics_policy.plot_on_train_end and int(owner.verbose) >= 1:
            try:
                owner.plot_metrics()
            except Exception as exc:
                logger.warning("Unable to plot training metrics: %s", exc)

    def fit_from_prebuilt_batches(
        self,
        validation_node_batch,
        validation_graph_conditioning,
        batch_iter_factory,
        *,
        ckpt_path: Optional[str],
    ):
        owner = self.owner
        from .conditional_node_field_generator import PrebuiltBatchIterableDataset

        val_loader = owner._build_validation_loader(
            node_batch=validation_node_batch,
            graph_conditioning=validation_graph_conditioning,
            targets=None,
        )
        train_loader = DataLoader(
            PrebuiltBatchIterableDataset(
                batch_iter_factory,
                prefetch_batches=int(getattr(owner, "stream_prefetch_batches", 2)),
                batch_timeout_seconds=getattr(owner, "stream_batch_timeout_seconds", None),
                max_consecutive_timeouts=int(getattr(owner, "stream_max_consecutive_stalls", 3)),
            ),
            batch_size=None,
        )
        previous_batch_logging = bool(getattr(owner.model, "log_train_every_batch", False))
        previous_stream_progress_owner = getattr(owner.model, "_stream_progress_owner", None)
        stream_progress_owner = getattr(owner, "_graph_generator_snapshot_owner", None)
        owner.model._stream_progress_owner = stream_progress_owner
        owner.model.log_train_every_batch = bool(
            getattr(stream_progress_owner, "verbose", owner.verbose)
        )
        try:
            self.run_training(
                train_loader,
                val_loader,
                ckpt_path=ckpt_path,
                context=f"{owner.__class__.__name__}.fit_from_prebuilt_batches",
                train_loader_length=1,
                training_policy=owner._current_training_policy(suppress_non_batch_output=False),
                checkpoint_policy=owner._current_checkpoint_policy(),
                metrics_policy=owner.metrics_policy_,
                snapshot_frequency="batch",
            )
        finally:
            owner.model.log_train_every_batch = previous_batch_logging
            owner.model._stream_progress_owner = previous_stream_progress_owner
        return owner

    def fit(self, node_batch, graph_conditioning, *, targets=None, ckpt_path: Optional[str] = None):
        owner = self.owner
        from .conditional_node_field_generator import (
            ConditionalNodeFieldGraphWithEdgesDataset,
            collate_conditional_node_field_graph_with_edges,
        )

        payload = owner._build_processed_training_payload(
            node_batch=node_batch,
            graph_conditioning=graph_conditioning,
            targets=targets,
        )
        dataset = owner._build_dataset_from_processed_payload(payload)
        if isinstance(dataset, ConditionalNodeFieldGraphWithEdgesDataset):
            train_dataset, val_dataset = owner._build_train_val_subsets(dataset)
            train_loader = DataLoader(
                train_dataset,
                batch_size=owner.batch_size,
                shuffle=True,
                collate_fn=collate_conditional_node_field_graph_with_edges,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=owner.batch_size,
                shuffle=False,
                collate_fn=collate_conditional_node_field_graph_with_edges,
            )
        else:
            train_dataset, val_dataset = owner._build_train_val_subsets(dataset)
            train_loader = DataLoader(train_dataset, batch_size=owner.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=owner.batch_size, shuffle=False)

        self.run_training(
            train_loader,
            val_loader,
            ckpt_path=ckpt_path,
            context=f"{owner.__class__.__name__}.fit",
            train_loader_length=len(train_loader),
            training_policy=owner._current_training_policy(),
            checkpoint_policy=owner._current_checkpoint_policy(),
            metrics_policy=owner.metrics_policy_,
            snapshot_frequency="epoch",
        )
        return owner
