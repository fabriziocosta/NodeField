"""Training orchestration for the node-field model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

from .graph_generator_state import (
    CheckpointPolicy,
    MetricsPolicy,
    TrainingPolicy,
    TrainingProgressSamplingConfig,
)
from .metrics_collection import (
    GraphGeneratorBatchAndEpochSnapshotCallback,
    GraphGeneratorBatchSnapshotCallback,
    GraphGeneratorEpochSnapshotCallback,
    GraphGeneratorTrainingSampleCallback,
    MetricsLogger,
)
from .naming_utils import sanitize_model_token
from .runtime_paths import resolve_saved_generator_dir
from .runtime_utils import get_runtime_logger, run_trainer_fit, verbose_log
from .scientific_observations import ObservationPolicy, ScientificMetricsCallback
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
        config = getattr(self.owner, "_graph_generator_sample_progress_config", None)
        if config is None:
            config = TrainingProgressSamplingConfig(
                enabled=bool(getattr(self.owner, "_graph_generator_sample_progress_enabled", False)),
                n_samples=int(getattr(self.owner, "_graph_generator_sample_progress_n_samples", 7)),
                every_n_epochs=int(
                    getattr(self.owner, "_graph_generator_sample_progress_every_n_epochs", 1)
                ),
                output_path=getattr(self.owner, "_graph_generator_sample_progress_pdf_path", None),
                plot_kwargs=getattr(self.owner, "_graph_generator_sample_progress_plot_kwargs", None),
                plot_fn=getattr(self.owner, "_graph_generator_sample_progress_plot_fn", None),
            )
        if not bool(config.enabled):
            return None
        snapshot_owner = getattr(self.owner, "_graph_generator_snapshot_owner", None)
        if snapshot_owner is None:
            return None
        if config.output_path is None:
            return None
        return GraphGeneratorTrainingSampleCallback(
            snapshot_owner,
            n_samples=int(config.n_samples),
            every_n_epochs=int(config.every_n_epochs),
            output_path=config.output_path,
            plot_kwargs=config.plot_kwargs,
            plot_fn=config.plot_fn,
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
            scientific_metrics_callback=self._build_scientific_metrics_callback(),
        )
        if owner.model_name is not None:
            verbose_log(owner, f"Save target model_name={owner.model_name} model_dir={owner.model_dir}")
        verbose_log(owner, f"Writing checkpoints to {checkpoint_dir}")
        if owner.model_name is not None:
            loss_curves_pdf_path = (
                resolve_saved_generator_dir(model_dir=getattr(owner, "model_dir", None))
                / f"{sanitize_model_token(owner.model_name)}.loss-curves.pdf"
            )
            verbose_log(owner, f"Loss curves PDF: {loss_curves_pdf_path.resolve()}")
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
        if owner.best_checkpoint_path_:
            checkpoint = torch.load(owner.best_checkpoint_path_, map_location=owner.device, weights_only=False)
            best_epoch = checkpoint.get("epoch")
            owner.best_checkpoint_epoch_ = int(best_epoch) if best_epoch is not None else None
            if checkpoint_policy.restore_best_checkpoint:
                state_dict = checkpoint.get("state_dict", checkpoint)
                owner.model.load_state_dict(state_dict)
                owner.model.to(owner.device)
            if checkpoint_policy.restore_best_checkpoint and int(owner.verbose) >= 1:
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

    def _build_scientific_metrics_callback(self):
        telemetry_path = getattr(self.owner, "scientific_telemetry_path", None)
        if telemetry_path is None:
            return None
        policy = ObservationPolicy(
            plateau_window_epochs=int(getattr(self.owner, "scientific_plateau_window_epochs", 8)),
            plateau_minimum_improvement=float(
                getattr(self.owner, "scientific_plateau_minimum_improvement", 0.002)
            ),
            generalisation_gap_threshold=float(
                getattr(self.owner, "scientific_generalisation_gap_threshold", 0.12)
            ),
            gradient_norm_threshold=float(
                getattr(self.owner, "scientific_gradient_norm_threshold", 1000.0)
            ),
            runtime_multiplier=float(getattr(self.owner, "scientific_runtime_multiplier", 3.0)),
        )
        return ScientificMetricsCallback(
            telemetry_path,
            getattr(self.owner, "scientific_observations_path", None),
            policy=policy,
        )

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
        try:
            train_dataset = PrebuiltBatchIterableDataset(
                batch_iter_factory,
                prefetch_batches=int(getattr(owner, "stream_prefetch_batches", 2)),
                batch_timeout_seconds=getattr(owner, "stream_batch_timeout_seconds", None),
                max_consecutive_timeouts=int(getattr(owner, "stream_max_consecutive_stalls", 3)),
            )
        except TypeError as exc:
            # Preserve compatibility with lightweight iterable test doubles and
            # older extension implementations that only accepted prefetch_batches.
            if "unexpected keyword argument" not in str(exc):
                raise
            train_dataset = PrebuiltBatchIterableDataset(
                batch_iter_factory,
                prefetch_batches=int(getattr(owner, "stream_prefetch_batches", 2)),
            )
        train_loader = DataLoader(train_dataset, batch_size=None)
        previous_batch_logging = bool(getattr(owner.model, "log_train_every_batch", False))
        previous_stream_progress_owner = getattr(owner.model, "_stream_progress_owner", None)
        stream_progress_owner = getattr(owner, "_graph_generator_snapshot_owner", None)
        owner.model._stream_progress_owner = stream_progress_owner
        owner.model.log_train_every_batch = bool(
            getattr(stream_progress_owner, "verbose", owner.verbose)
        )
        training_policy_factory = getattr(owner, "_current_training_policy", None)
        checkpoint_policy_factory = getattr(owner, "_current_checkpoint_policy", None)
        metrics_policy = getattr(owner, "metrics_policy_", MetricsPolicy())
        training_policy = (
            training_policy_factory(suppress_non_batch_output=False)
            if callable(training_policy_factory)
            else TrainingPolicy(
                maximum_epochs=int(getattr(owner, "maximum_epochs", 1)),
                early_stopping_monitor=str(getattr(owner, "early_stopping_monitor", "val_total")),
                early_stopping_mode=str(getattr(owner, "early_stopping_mode", "min")),
                enable_early_stopping=bool(getattr(owner, "enable_early_stopping", False)),
                early_stopping_patience=int(getattr(owner, "early_stopping_patience", 1)),
                early_stopping_min_delta=float(getattr(owner, "early_stopping_min_delta", 0.0)),
                suppress_non_batch_output=False,
            )
        )
        checkpoint_policy = (
            checkpoint_policy_factory()
            if callable(checkpoint_policy_factory)
            else CheckpointPolicy(
                restore_best_checkpoint=bool(getattr(owner, "restore_best_checkpoint", False)),
                checkpoint_root_dir=str(getattr(owner, "checkpoint_root_dir", ".artifacts/checkpoints")),
            )
        )
        try:
            self.run_training(
                train_loader,
                val_loader,
                ckpt_path=ckpt_path,
                context=f"{owner.__class__.__name__}.fit_from_prebuilt_batches",
                train_loader_length=1,
                training_policy=training_policy,
                checkpoint_policy=checkpoint_policy,
                metrics_policy=metrics_policy,
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
        training_num_workers = max(0, int(getattr(owner, "training_num_workers", 0)))
        loader_worker_kwargs = {
            "num_workers": training_num_workers,
            "persistent_workers": training_num_workers > 0,
        }
        if isinstance(dataset, ConditionalNodeFieldGraphWithEdgesDataset):
            train_dataset, val_dataset = owner._build_train_val_subsets(dataset)
            train_loader = DataLoader(
                train_dataset,
                batch_size=owner.batch_size,
                shuffle=True,
                collate_fn=collate_conditional_node_field_graph_with_edges,
                **loader_worker_kwargs,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=owner.batch_size,
                shuffle=False,
                collate_fn=collate_conditional_node_field_graph_with_edges,
                **loader_worker_kwargs,
            )
        else:
            train_dataset, val_dataset = owner._build_train_val_subsets(dataset)
            train_loader = DataLoader(
                train_dataset,
                batch_size=owner.batch_size,
                shuffle=True,
                **loader_worker_kwargs,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=owner.batch_size,
                shuffle=False,
                **loader_worker_kwargs,
            )

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
