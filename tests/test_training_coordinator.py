import types

import torch

import conditional_node_field_graph_generator.training_coordinator as tc_module
from conditional_node_field_graph_generator.graph_generator_state import (
    CheckpointPolicy,
    MetricsPolicy,
    TrainingPolicy,
)
from conditional_node_field_graph_generator.training_coordinator import TrainingCoordinator


class _Owner:
    def __init__(self):
        self.model = torch.nn.Linear(1, 1)
        self.model.val_node_field = [3.0, 2.0]
        self.model_name = None
        self.model_dir = None
        self.artifact_root_dir = ".artifacts"
        self.checkpoint_root_dir = ".artifacts/checkpoints"
        self.device = "cpu"
        self.verbose = 1
        self.best_checkpoint_path_ = None
        self.best_checkpoint_score_ = None
        self.best_checkpoint_epoch_ = None
        self._graph_generator_snapshot_owner = None
        self.plot_calls = 0

    def plot_metrics(self):
        self.plot_calls += 1


def test_training_coordinator_restores_best_checkpoint_and_plots(monkeypatch):
    owner = _Owner()
    coordinator = TrainingCoordinator(owner)
    callback = types.SimpleNamespace(best_model_path="best.ckpt", best_model_score=torch.tensor(1.5))

    monkeypatch.setattr(
        tc_module,
        "build_training_callbacks",
        lambda **kwargs: ([], ".artifacts/checkpoints/run", callback),
    )
    monkeypatch.setattr(tc_module, "create_trainer", lambda **kwargs: types.SimpleNamespace(current_epoch=4))
    monkeypatch.setattr(tc_module, "run_trainer_fit", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        tc_module.torch,
        "load",
        lambda *args, **kwargs: {"epoch": 1, "state_dict": owner.model.state_dict()},
    )

    coordinator.run_training(
        train_loader=[1],
        val_loader=[2],
        ckpt_path=None,
        context="unit",
        train_loader_length=1,
        training_policy=TrainingPolicy(
            maximum_epochs=5,
            early_stopping_monitor="val_total",
            early_stopping_mode="min",
            enable_early_stopping=True,
            early_stopping_patience=2,
            early_stopping_min_delta=0.0,
        ),
        checkpoint_policy=CheckpointPolicy(
            restore_best_checkpoint=True,
            checkpoint_root_dir=".artifacts/checkpoints",
        ),
        metrics_policy=MetricsPolicy(plot_on_train_end=True),
    )

    assert owner.best_checkpoint_path_ == "best.ckpt"
    assert owner.best_checkpoint_score_ == 1.5
    assert owner.best_checkpoint_epoch_ == 1
    assert owner.plot_calls == 1


def test_training_coordinator_stream_batch_logging_uses_snapshot_owner_verbose(monkeypatch):
    owner = _Owner()
    owner.verbose = False
    owner.stream_prefetch_batches = 2
    owner.model.log_train_every_batch = False
    owner.model._stream_progress_owner = None
    owner._graph_generator_snapshot_owner = types.SimpleNamespace(verbose=True, model_name=None)
    owner._build_validation_loader = lambda **kwargs: []

    monkeypatch.setattr(
        tc_module,
        "DataLoader",
        lambda dataset, batch_size=None: dataset if batch_size is None else [dataset],
    )

    called = {}

    def _fake_run_training(self, train_loader, val_loader, **kwargs):
        del train_loader, val_loader, kwargs
        called["log_train_every_batch"] = owner.model.log_train_every_batch
        called["stream_progress_owner"] = owner.model._stream_progress_owner

    monkeypatch.setattr(TrainingCoordinator, "run_training", _fake_run_training)

    coordinator = TrainingCoordinator(owner)
    coordinator.fit_from_prebuilt_batches(
        validation_node_batch=[],
        validation_graph_conditioning=[],
        batch_iter_factory=lambda: iter(()),
        ckpt_path=None,
    )

    assert called["log_train_every_batch"] is True
    assert called["stream_progress_owner"] is owner._graph_generator_snapshot_owner


def test_training_coordinator_uses_prefetch_for_streamed_batches(monkeypatch):
    owner = _Owner()
    owner.stream_prefetch_batches = 5
    owner.verbose = False
    owner.model.log_train_every_batch = False
    owner.model._stream_progress_owner = None
    owner._graph_generator_snapshot_owner = None
    owner._build_validation_loader = lambda **kwargs: []

    created = {}

    class _FakeIterableDataset:
        def __init__(self, batch_iter_factory, prefetch_batches=0):
            created["prefetch_batches"] = prefetch_batches
            self.batch_iter_factory = batch_iter_factory

    monkeypatch.setattr(
        __import__("conditional_node_field_graph_generator.conditional_node_field_generator", fromlist=["PrebuiltBatchIterableDataset"]),
        "PrebuiltBatchIterableDataset",
        _FakeIterableDataset,
    )
    monkeypatch.setattr(
        tc_module,
        "DataLoader",
        lambda dataset, batch_size=None: dataset if batch_size is None else [dataset],
    )
    monkeypatch.setattr(TrainingCoordinator, "run_training", lambda *args, **kwargs: None)

    coordinator = TrainingCoordinator(owner)
    coordinator.fit_from_prebuilt_batches(
        validation_node_batch=[],
        validation_graph_conditioning=[],
        batch_iter_factory=lambda: iter(()),
        ckpt_path=None,
    )

    assert created["prefetch_batches"] == 5
