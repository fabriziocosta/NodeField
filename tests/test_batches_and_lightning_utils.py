import os
import logging
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytest
import torch
import warnings

import conditional_node_field_graph_generator as graphgen
from conditional_node_field_graph_generator.conditional_node_field_generator import (
    GeneratedNodeBatch,
    ConditionalNodeFieldGenerator,
    ConditionalNodeFieldModule,
    GraphConditioningBatch,
    MetricsLogger,
    NodeGenerationBatch,
    _StreamBatchTimeoutError,
)
from conditional_node_field_graph_generator.metrics_collection import (
    GraphGeneratorBatchAndEpochSnapshotCallback,
    GraphGeneratorEpochSnapshotCallback,
    GraphGeneratorTrainingSampleCallback,
)
from conditional_node_field_graph_generator.extensions.demo.pipeline import fit_graph_generator
from conditional_node_field_graph_generator.extensions.demo.storage import find_latest_checkpoint
from conditional_node_field_graph_generator.metrics_visualization import (
    plot_metrics,
)
from conditional_node_field_graph_generator.persistence import (
    GRAPH_GENERATOR_PERSISTENCE_VERSION,
    load_graph_generator,
    save_graph_generator,
)
from conditional_node_field_graph_generator.runtime_utils import run_trainer_fit
from conditional_node_field_graph_generator.training_policy import (
    format_restored_checkpoint_summary,
)


def test_graph_conditioning_batch_len():
    batch = GraphConditioningBatch(
        graph_embeddings=np.zeros((4, 8), dtype=float),
        node_counts=np.array([2, 2, 3, 1], dtype=np.int64),
        edge_counts=np.array([1, 1, 2, 0], dtype=np.int64),
    )
    assert len(batch) == 4


def test_node_generation_and_generated_batch_len():
    node_batch = NodeGenerationBatch(
        node_embeddings_list=[np.zeros((2, 4)), np.zeros((3, 4))],
        node_presence_mask=np.ones((2, 3), dtype=bool),
        node_degree_targets=np.zeros((2, 3), dtype=np.int64),
    )
    generated = GeneratedNodeBatch(
        node_presence_mask=np.ones((3, 2), dtype=bool)
    )
    assert len(node_batch) == 2
    assert len(generated) == 3


class _OkTrainer:
    def __init__(self):
        self.called_with = None

    def fit(self, model, train_dataloaders=None, val_dataloaders=None, ckpt_path=None):
        self.called_with = (model, train_dataloaders, val_dataloaders, ckpt_path)


class _ExitTrainer:
    def fit(self, model, train_dataloaders=None, val_dataloaders=None, ckpt_path=None):
        raise SystemExit(2)


class _InterruptedExitTrainer:
    def fit(self, model, train_dataloaders=None, val_dataloaders=None, ckpt_path=None):
        try:
            raise KeyboardInterrupt()
        except KeyboardInterrupt as exc:
            raise SystemExit(1) from exc


def test_run_trainer_fit_calls_fit_with_named_loaders():
    trainer = _OkTrainer()
    model = object()
    train_loader = object()
    val_loader = object()

    run_trainer_fit(trainer, model, train_loader, val_loader, context="unit-test")

    assert trainer.called_with == (model, train_loader, val_loader, None)


def test_run_trainer_fit_forwards_checkpoint_path():
    trainer = _OkTrainer()
    model = object()
    train_loader = object()
    val_loader = object()

    run_trainer_fit(
        trainer,
        model,
        train_loader,
        val_loader,
        context="unit-test",
        ckpt_path="/tmp/resume.ckpt",
    )

    assert trainer.called_with == (model, train_loader, val_loader, "/tmp/resume.ckpt")


def test_run_trainer_fit_wraps_system_exit():
    with pytest.raises(RuntimeError, match="unit-test aborted with SystemExit\\(2\\)"):
        run_trainer_fit(_ExitTrainer(), object(), object(), object(), context="unit-test")


def test_run_trainer_fit_reraises_keyboard_interrupt_wrapped_by_system_exit():
    with pytest.raises(KeyboardInterrupt):
        run_trainer_fit(_InterruptedExitTrainer(), object(), object(), object(), context="unit-test")


class _WarnTrainer:
    def __init__(self):
        self.called = False

    def fit(self, model, train_dataloaders=None, val_dataloaders=None, ckpt_path=None):
        del model, train_dataloaders, val_dataloaders, ckpt_path
        self.called = True
        warnings.warn(
            "The 'train_dataloader' does not have many workers which may be a bottleneck. "
            "Consider increasing the value of the `num_workers` argument` to `num_workers=15` in the `DataLoader` to improve performance.",
            UserWarning,
        )
        warnings.warn(
            "The 'val_dataloader' does not have many workers which may be a bottleneck. "
            "Consider increasing the value of the `num_workers` argument` to `num_workers=15` in the `DataLoader` to improve performance.",
            UserWarning,
        )
        warnings.warn(
            "Starting from v1.9.0, `tensorboardX` has been removed as a dependency of the "
            "`pytorch_lightning` package, due to potential conflicts with other packages in the ML ecosystem.",
            UserWarning,
        )


def test_run_trainer_fit_suppresses_lightning_worker_warnings():
    trainer = _WarnTrainer()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_trainer_fit(trainer, object(), object(), object(), context="unit-test")

    assert trainer.called is True
    assert caught == []


def test_stream_batch_timeout_skips_immediately_without_in_process_retry(monkeypatch):
    generator = graphgen.ConditionalNodeFieldGraphGenerator.__new__(graphgen.ConditionalNodeFieldGraphGenerator)
    generator.stream_batch_timeout_seconds = 2.5
    generator.conditional_node_generator_model = type(
        "_DummyModel",
        (),
        {"_collate_processed_payload": staticmethod(lambda payload: payload)},
    )()

    retry_called = {"value": False}

    def _unexpected_retry(graphs):
        del graphs
        retry_called["value"] = True
        raise AssertionError("in-process retry should not run after a timeout")

    generator._prepare_stream_training_batch = _unexpected_retry

    monkeypatch.setattr(
        "conditional_node_field_graph_generator.stream_fit.run_with_fork_timeout",
        lambda worker, *args, timeout_seconds=None: (_ for _ in ()).throw(TimeoutError("timed out")),
    )

    with pytest.raises(_StreamBatchTimeoutError, match=r"exceeded 2\.5s"):
        generator._prepare_stream_training_batch_with_timeout(["g1", "g2"])

    assert retry_called["value"] is False


def test_find_latest_checkpoint_prefers_last_ckpt(tmp_path):
    root = tmp_path / "checkpoints"
    older = root / "run_old"
    newer = root / "run_new"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    (older / "best-001.ckpt").write_text("old")
    (newer / "best-003.ckpt").write_text("best")
    (newer / "last.ckpt").write_text("last")

    latest = find_latest_checkpoint(checkpoint_root=root)

    assert latest is not None
    assert latest.endswith("last.ckpt")


class _FitRecorder:
    def __init__(self):
        self.calls = []

    def fit(self, graphs, targets=None, ckpt_path=None):
        self.calls.append(
            {
                "graphs": graphs,
                "targets": targets,
                "ckpt_path": ckpt_path,
            }
        )


def test_fit_graph_generator_resumes_from_latest_checkpoint(tmp_path):
    recorder = _FitRecorder()
    checkpoint_root = tmp_path / "checkpoints"
    run_dir = checkpoint_root / "run_a"
    run_dir.mkdir(parents=True)
    (run_dir / "last.ckpt").write_text("checkpoint")

    result = fit_graph_generator(
        recorder,
        train_graphs=["g1", "g2"],
        targets=[1, 0],
        resume_latest_checkpoint=True,
        checkpoint_root=checkpoint_root,
    )

    assert result is recorder
    assert recorder.calls[0]["graphs"] == ["g1", "g2"]
    assert recorder.calls[0]["targets"] == [1, 0]
    assert recorder.calls[0]["ckpt_path"].endswith("last.ckpt")


class _SaveableGenerator:
    def __init__(self, model_name=None, model_dir=None):
        self.model_name = model_name
        self.model_dir = model_dir


def test_save_graph_generator_uses_explicit_generator_metadata(tmp_path):
    generator = _SaveableGenerator(model_name="demo-chem", model_dir=tmp_path)

    filename = save_graph_generator(
        generator,
    )

    assert filename == "demo-chem.pkl"
    assert (tmp_path / filename).exists()


def test_save_graph_generator_uses_atomic_replace(monkeypatch, tmp_path):
    replace_calls = []
    real_replace = os.replace

    def recording_replace(src, dst):
        replace_calls.append((Path(src), Path(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", recording_replace)

    generator = _SaveableGenerator(model_name="demo-chem", model_dir=tmp_path)

    filename = save_graph_generator(generator)

    assert filename == "demo-chem.pkl"
    assert (tmp_path / filename).exists()
    assert len(replace_calls) == 1
    src, dst = replace_calls[0]
    assert src.parent == tmp_path
    assert src.name.startswith(".demo-chem.")
    assert dst == tmp_path / "demo-chem.pkl"
    assert not src.exists()


def test_save_graph_generator_cpu_fallback_handles_forked_cuda_init_error(monkeypatch, tmp_path):
    class _RecordingModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.to_calls = []

        def to(self, device):
            self.to_calls.append(device)
            return self

    import types

    module = _RecordingModule()
    node_generator = types.SimpleNamespace(model=module, device=torch.device("cuda:0"))
    generator = _SaveableGenerator(model_name="demo-chem", model_dir=tmp_path)
    generator.conditional_node_generator_model = node_generator
    generator.device = torch.device("cuda:0")
    dump_observations = []

    def fake_run_with_fork_timeout(*args, **kwargs):
        raise RuntimeError('AcceleratorError("CUDA error: initialization error")')

    def fake_module_device(candidate):
        assert candidate is module
        return torch.device("cuda:0")

    def fake_atomic_pickle_dump(obj, output_path):
        dump_observations.append(
            {
                "path": output_path,
                "generator_device": obj.device,
                "node_generator_device": obj.conditional_node_generator_model.device,
                "to_calls": list(module.to_calls),
            }
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"snapshot")

    monkeypatch.setattr(
        "conditional_node_field_graph_generator.persistence.run_with_fork_timeout",
        fake_run_with_fork_timeout,
    )
    monkeypatch.setattr(
        "conditional_node_field_graph_generator.persistence._module_device",
        fake_module_device,
    )
    monkeypatch.setattr(
        "conditional_node_field_graph_generator.persistence._atomic_pickle_dump",
        fake_atomic_pickle_dump,
    )

    filename = save_graph_generator(generator, save_loss_curves_pdf=False)

    assert filename == "demo-chem.pkl"
    assert (tmp_path / filename).read_bytes() == b"snapshot"
    assert dump_observations == [
        {
            "path": tmp_path / "demo-chem.pkl",
            "generator_device": torch.device("cpu"),
            "node_generator_device": torch.device("cpu"),
            "to_calls": ["cpu"],
        }
    ]
    assert module.to_calls == ["cpu", torch.device("cuda:0")]
    assert generator.device == torch.device("cuda:0")
    assert node_generator.device == torch.device("cuda:0")


def test_save_graph_generator_also_exports_loss_curves_pdf_when_supported(tmp_path):
    class _GeneratorWithPdf(_SaveableGenerator):
        def __init__(self, model_name=None, model_dir=None):
            super().__init__(model_name=model_name, model_dir=model_dir)
            self.exported_paths = []

        def export_metrics_pdf(self, output_path, window=10, alpha=0.3):
            path = Path(output_path)
            path.write_text("pdf placeholder")
            self.exported_paths.append((path, window, alpha))
            return path

    generator = _GeneratorWithPdf(model_name="demo-chem", model_dir=tmp_path)
    pdf_path = tmp_path / "demo-chem.loss-curves.pdf"
    pdf_path.write_text("stale pdf")
    original_inode = pdf_path.stat().st_ino

    filename = save_graph_generator(generator)

    assert filename == "demo-chem.pkl"
    assert (tmp_path / filename).exists()
    assert pdf_path.exists()
    assert pdf_path.read_text() == "pdf placeholder"
    assert pdf_path.stat().st_ino == original_inode
    assert len(generator.exported_paths) == 1
    exported_path, window, alpha = generator.exported_paths[0]
    assert exported_path.parent == tmp_path
    assert exported_path.name.startswith(".demo-chem.loss-curves.")
    assert exported_path.suffix == ".pdf"
    assert window == 10
    assert alpha == 0.3


def test_save_graph_generator_skips_when_model_name_is_none(tmp_path):
    generator = _SaveableGenerator(model_name=None, model_dir=tmp_path)

    filename = save_graph_generator(
        generator,
    )

    assert filename is None
    assert not list(tmp_path.glob("*.pkl"))


def test_load_graph_generator_rejects_incompatible_schema_version(tmp_path):
    import dill as pickle

    generator = _SaveableGenerator(model_name="demo-chem", model_dir=tmp_path)
    filename = save_graph_generator(generator)
    saved_path = tmp_path / filename

    with open(saved_path, "rb") as handle:
        restored = pickle.load(handle)
    restored._persistence_schema_version = GRAPH_GENERATOR_PERSISTENCE_VERSION - 1
    with open(saved_path, "wb") as handle:
        pickle.dump(restored, handle)

    with pytest.raises(RuntimeError, match="Saved graph generator schema is incompatible"):
        load_graph_generator(filename, model_dir=tmp_path)


def test_load_graph_generator_repairs_legacy_nsppk_fit_flags(tmp_path):
    class _BaseNSPPK:
        pass

    class _NSPPKWrapper:
        def __init__(self):
            self.base_nsppk = _BaseNSPPK()

    class _NodeNSPPKWrapper:
        def __init__(self):
            self.nsppk = _NSPPKWrapper()

    class _Generator(_SaveableGenerator):
        def __init__(self, model_name=None, model_dir=None):
            super().__init__(model_name=model_name, model_dir=model_dir)
            self.is_fitted_ = True
            self.graph_vectorizer = _NSPPKWrapper()
            self.node_graph_vectorizer = _NodeNSPPKWrapper()

    generator = _Generator(model_name="demo-chem", model_dir=tmp_path)
    filename = save_graph_generator(generator)

    restored = load_graph_generator(filename, model_dir=tmp_path)

    assert restored.graph_vectorizer.base_nsppk.is_fitted_ is True
    assert restored.node_graph_vectorizer.nsppk.base_nsppk.is_fitted_ is True


def test_load_graph_generator_repairs_legacy_node_generator_locality_horizon(tmp_path):
    import types

    generator = _SaveableGenerator(model_name="demo-chem", model_dir=tmp_path)
    generator.locality_horizon = 3
    generator.conditional_node_generator_model = types.SimpleNamespace()
    filename = save_graph_generator(generator)

    restored = load_graph_generator(filename, model_dir=tmp_path)

    assert restored.conditional_node_generator_model.locality_horizon_ == 3


def test_load_graph_generator_restores_legacy_oracle_runtime_defaults(tmp_path):
    class _Generator(_SaveableGenerator):
        def __init__(self, model_name=None, model_dir=None):
            super().__init__(model_name=model_name, model_dir=model_dir)
            self.is_fitted_ = True
            self.graph_decoder = type("_LegacyDecoder", (), {})()

    generator = _Generator(model_name="demo-chem", model_dir=tmp_path)
    filename = save_graph_generator(generator)

    restored = load_graph_generator(filename, model_dir=tmp_path)

    assert restored.oracle_use_node_label_cuts is False
    assert restored.oracle_use_edge_label_cuts is False
    assert restored.max_decode_seconds_per_sample is None
    assert restored.max_decode_attempts_per_sample == 1
    assert restored.graph_decoder.adjacency_time_limit_seconds == 60.0
    assert restored.graph_decoder.parallel_decode_timeout_seconds == 30.0
    assert restored.graph_decoder.active_time_limit_seconds is None
    assert restored.graph_decoder.solver_threads is None
    assert restored.graph_decoder.use_horizon_ilp_constraints is True
    assert restored.graph_decoder.horizon_constraint_weight == pytest.approx(2.0)
    assert restored.graph_decoder.horizon_positive_threshold == pytest.approx(0.8)
    assert restored.graph_decoder.horizon_negative_threshold == pytest.approx(0.2)
    assert restored.graph_decoder.horizon_pair_budget == 24
    assert restored.graph_decoder.horizon_paths_per_pair == 8
    assert restored.graph_decoder.horizon_max_iterations == 1


def test_graph_generator_epoch_snapshot_callback_saves_epoch_version(monkeypatch, tmp_path):
    calls = []

    def fake_save_graph_generator(
        graph_generator,
        model_name=None,
        model_dir=None,
        log=True,
        save_loss_curves_pdf=True,
    ):
        calls.append(
            {
                "graph_generator": graph_generator,
                "model_name": model_name,
                "model_dir": model_dir,
                "log": log,
                "save_loss_curves_pdf": save_loss_curves_pdf,
                "is_fitted": graph_generator.is_fitted_,
            }
        )
        return "saved.pkl"

    monkeypatch.setattr(
        "conditional_node_field_graph_generator.persistence.save_graph_generator",
        fake_save_graph_generator,
    )

    class _Owner:
        def __init__(self):
            self.model_name = "demo-chem"
            self.model_dir = tmp_path
            self.is_fitted_ = False

    class _Trainer:
        sanity_checking = False
        is_global_zero = True
        current_epoch = 2

    owner = _Owner()
    callback = GraphGeneratorEpochSnapshotCallback(owner)

    callback.on_validation_epoch_end(_Trainer(), object())

    assert owner.is_fitted_ is False
    assert calls == [
        {
            "graph_generator": owner,
            "model_name": "demo-chem",
            "model_dir": tmp_path,
            "log": False,
            "save_loss_curves_pdf": False,
            "is_fitted": True,
        }
    ]


def test_graph_generator_epoch_snapshot_callback_saves_loss_pdf_on_configured_interval(monkeypatch, tmp_path):
    calls = []

    def fake_save_graph_generator(
        graph_generator,
        model_name=None,
        model_dir=None,
        log=True,
        save_loss_curves_pdf=True,
    ):
        calls.append(save_loss_curves_pdf)
        return "saved.pkl"

    monkeypatch.setattr(
        "conditional_node_field_graph_generator.persistence.save_graph_generator",
        fake_save_graph_generator,
    )

    class _Owner:
        def __init__(self):
            self.model_name = "demo-chem"
            self.model_dir = tmp_path
            self.is_fitted_ = False
            self.loss_curves_pdf_every_n_epochs = 3

    class _Trainer:
        sanity_checking = False
        is_global_zero = True
        current_epoch = 2

    callback = GraphGeneratorEpochSnapshotCallback(_Owner())

    callback.on_validation_epoch_end(_Trainer(), object())

    assert calls == [True]


def test_graph_generator_epoch_snapshot_callback_logs_complete_epoch_time(monkeypatch, tmp_path, caplog):
    def fake_save_graph_generator(
        graph_generator,
        model_name=None,
        model_dir=None,
        log=True,
        save_loss_curves_pdf=True,
    ):
        del graph_generator, model_name, model_dir, log, save_loss_curves_pdf
        return "saved.pkl"

    monkeypatch.setattr(
        "conditional_node_field_graph_generator.persistence.save_graph_generator",
        fake_save_graph_generator,
    )
    times = [103.0, 108.5]
    monkeypatch.setattr(
        "conditional_node_field_graph_generator.metrics_collection.time.time",
        lambda: times.pop(0) if times else 108.5,
    )

    class _Owner:
        def __init__(self):
            self.model_name = "demo-chem"
            self.model_dir = tmp_path
            self.is_fitted_ = False
            self.verbose = 1

    class _Trainer:
        sanity_checking = False
        is_global_zero = True
        current_epoch = 2

    class _Module:
        _epoch_started_at = 100.0

    caplog.set_level(logging.INFO, logger="conditional_node_field_graph_generator")

    callback = GraphGeneratorEpochSnapshotCallback(_Owner())
    pl_module = _Module()
    callback.on_validation_epoch_end(_Trainer(), pl_module)

    assert "epoch 3: completed epoch in 8.50s" in caplog.text
    assert "finished generator snapshot" not in caplog.text
    assert pl_module._last_completed_epoch_seconds == pytest.approx(8.5)


def test_graph_generator_batch_and_epoch_snapshot_callback_saves_epoch_version(monkeypatch, tmp_path):
    calls = []

    def fake_save_graph_generator(
        graph_generator,
        model_name=None,
        model_dir=None,
        log=True,
        save_loss_curves_pdf=True,
    ):
        calls.append(
            {
                "graph_generator": graph_generator,
                "model_name": model_name,
                "model_dir": model_dir,
                "log": log,
                "save_loss_curves_pdf": save_loss_curves_pdf,
                "is_fitted": graph_generator.is_fitted_,
            }
        )
        return "saved.pkl"

    monkeypatch.setattr(
        "conditional_node_field_graph_generator.persistence.save_graph_generator",
        fake_save_graph_generator,
    )

    class _Owner:
        def __init__(self):
            self.model_name = "demo-stream"
            self.model_dir = tmp_path
            self.is_fitted_ = False
            self.stream_snapshot_every_n_batches = 100

    class _Trainer:
        sanity_checking = False
        is_global_zero = True
        current_epoch = 1

    owner = _Owner()
    callback = GraphGeneratorBatchAndEpochSnapshotCallback(owner)

    callback.on_validation_epoch_end(_Trainer(), object())

    assert owner.is_fitted_ is False
    assert calls == [
        {
            "graph_generator": owner,
            "model_name": "demo-stream",
            "model_dir": tmp_path,
            "log": False,
            "save_loss_curves_pdf": False,
            "is_fitted": True,
        }
    ]


def test_training_sample_callback_uses_sample_return_decode_stages(monkeypatch, tmp_path):
    write_calls = []

    monkeypatch.setattr(
        GraphGeneratorTrainingSampleCallback,
        "_write_pdf_page",
        lambda self, epoch_record: write_calls.append(epoch_record),
    )

    class _Owner:
        def __init__(self):
            self.is_fitted_ = False
            self.calls = 0

        def sample(self, n_samples=1, return_decode_stages=False, **kwargs):
            del kwargs
            assert return_decode_stages is True
            self.calls += 1
            variants = {}
            for key in ("raw", "ilp", "oracle"):
                graph = nx.Graph()
                graph.add_node(0, label=key)
                variants[key] = [graph] * int(n_samples)
            return variants

    class _Trainer:
        sanity_checking = False
        is_global_zero = True
        current_epoch = 0

    owner = _Owner()
    callback = GraphGeneratorTrainingSampleCallback(
        owner,
        n_samples=3,
        every_n_epochs=1,
        output_path=tmp_path / "samples.pdf",
    )

    callback.on_validation_epoch_end(_Trainer(), object())

    assert owner.is_fitted_ is False
    assert owner.calls == 1
    assert len(write_calls) == 1
    assert write_calls[0]["epoch"] == 1
    assert len(write_calls[0]["raw"]) == 3
    assert len(write_calls[0]["ilp"]) == 3
    assert len(write_calls[0]["oracle"]) == 3
    assert callback.epoch_samples == [{"epoch": 1}]


def test_training_sample_callback_respects_epoch_interval(monkeypatch, tmp_path):
    write_calls = []

    monkeypatch.setattr(
        GraphGeneratorTrainingSampleCallback,
        "_write_pdf_page",
        lambda self, epoch_record: write_calls.append(epoch_record["epoch"]),
    )

    class _Owner:
        is_fitted_ = False

        def __init__(self):
            self.calls = 0

        def sample(self, **kwargs):
            del kwargs
            self.calls += 1
            return []

    class _Trainer:
        sanity_checking = False
        is_global_zero = True
        current_epoch = 0

    owner = _Owner()
    callback = GraphGeneratorTrainingSampleCallback(
        owner,
        n_samples=1,
        every_n_epochs=2,
        output_path=tmp_path / "samples.pdf",
    )

    callback.on_validation_epoch_end(_Trainer(), object())
    _Trainer.current_epoch = 1
    callback.on_validation_epoch_end(_Trainer(), object())

    assert owner.calls == 2
    assert write_calls == [2]


def test_training_sample_callback_writes_incremental_pdf(tmp_path):
    write_calls = []

    class _Owner:
        is_fitted_ = False

        def sample(self, **kwargs):
            graphs = []
            for idx in range(int(kwargs["n_samples"])):
                graph = nx.path_graph(2)
                graph.nodes[0]["label"] = "C"
                graph.nodes[1]["label"] = str(idx)
                graphs.append(graph)
            return graphs

    class _Trainer:
        sanity_checking = False
        is_global_zero = True
        current_epoch = 0

    output_path = tmp_path / "samples.pdf"
    callback = GraphGeneratorTrainingSampleCallback(
        _Owner(),
        n_samples=2,
        every_n_epochs=1,
        output_path=output_path,
        plot_kwargs={
            "node_label_colors": {"C": "#ffaaaa", "0": "#00ff00", "1": "#0000ff"},
            "size": 2.0,
            "node_size": 250,
            "edge_width": 1.5,
            "show_label": True,
        },
    )

    callback.on_validation_epoch_end(_Trainer(), object())
    write_calls.append((output_path, [record["epoch"] for record in callback.epoch_samples]))
    first_size = output_path.stat().st_size
    _Trainer.current_epoch = 1
    callback.on_validation_epoch_end(_Trainer(), object())
    write_calls.append((output_path, [record["epoch"] for record in callback.epoch_samples]))

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert len(callback.epoch_samples) == 2
    assert write_calls == [
        (output_path, [1]),
        (output_path, [1, 2]),
    ]
    assert output_path.stat().st_size >= first_size
    assert callback.plot_kwargs["node_label_colors"]["C"] == "#ffaaaa"
    assert not list(tmp_path.glob(".samples.page.*.pdf"))


def test_training_sample_callback_starts_fresh_when_output_exists(tmp_path):
    output_path = tmp_path / "samples.pdf"
    output_path.write_bytes(b"stale pdf")

    GraphGeneratorTrainingSampleCallback(
        object(),
        n_samples=1,
        every_n_epochs=1,
        output_path=output_path,
    )

    assert not output_path.exists()


def test_training_sample_callback_uses_custom_plot_function(tmp_path):
    calls = []

    def _plot_fn(graph, size=None):
        calls.append((graph, size))
        return np.ones((4, 4, 3), dtype=np.uint8) * 255

    graph = nx.Graph()
    graph.add_node(0, label="C")

    callback = GraphGeneratorTrainingSampleCallback(
        object(),
        n_samples=1,
        every_n_epochs=1,
        output_path=tmp_path / "samples.pdf",
        plot_kwargs={"size": (500, 350), "cell_size": 2.0},
        plot_fn=_plot_fn,
    )

    record = {"epoch": 1, "raw": [graph], "ilp": [graph], "oracle": [graph]}
    callback.epoch_samples.append(record)
    callback._write_pdf_page(record)

    assert calls
    assert calls[0][0] is graph
    assert calls[0][1] == (500, 350)
    assert (tmp_path / "samples.pdf").exists()


def test_training_sample_callback_displays_pil_image_result(tmp_path):
    def _plot_fn(graph, size=None):
        del graph
        width, height = size
        return Image.new("RGB", (width, height), color="white")

    graph = nx.Graph()
    graph.add_node(0, label="C")

    callback = GraphGeneratorTrainingSampleCallback(
        object(),
        n_samples=1,
        every_n_epochs=1,
        output_path=tmp_path / "samples.pdf",
        plot_kwargs={"size": (32, 24), "cell_size": 2.0},
        plot_fn=_plot_fn,
    )

    record = {"epoch": 1, "raw": [graph], "ilp": [graph], "oracle": [graph]}
    callback.epoch_samples.append(record)
    callback._write_pdf_page(record)

    assert (tmp_path / "samples.pdf").exists()
    assert (tmp_path / "samples.pdf").stat().st_size > 0


def test_training_sample_callback_validates_configuration(tmp_path):
    with pytest.raises(ValueError, match="sample_training_progress_n_samples"):
        GraphGeneratorTrainingSampleCallback(
            object(),
            n_samples=0,
            every_n_epochs=1,
            output_path=tmp_path / "x.pdf",
        )
    with pytest.raises(ValueError, match="sample_training_progress_every_n_epochs"):
        GraphGeneratorTrainingSampleCallback(
            object(),
            n_samples=1,
            every_n_epochs=0,
            output_path=tmp_path / "x.pdf",
        )


def test_build_train_val_subsets_reuses_single_example_for_train_and_val():
    dataset = torch.utils.data.TensorDataset(torch.tensor([[1.0]], dtype=torch.float32))

    train_dataset, val_dataset = ConditionalNodeFieldGenerator._build_train_val_subsets(dataset)

    assert len(train_dataset) == 1
    assert len(val_dataset) == 1
    assert train_dataset[0][0].item() == 1.0
    assert val_dataset[0][0].item() == 1.0


def test_package_exports_only_new_primary_names():
    assert sorted(graphgen.__all__) == [
        "ConditionalNodeFieldGenerator",
        "ConditionalNodeFieldGraphDecoder",
        "ConditionalNodeFieldGraphGenerator",
    ]


def test_build_train_val_subsets_keeps_both_sides_non_empty_for_two_examples():
    dataset = torch.utils.data.TensorDataset(
        torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    )

    train_dataset, val_dataset = ConditionalNodeFieldGenerator._build_train_val_subsets(dataset)

    assert len(train_dataset) == 1
    assert len(val_dataset) == 1


def test_build_train_val_subsets_rejects_empty_dataset():
    dataset = torch.utils.data.TensorDataset(torch.empty((0, 1), dtype=torch.float32))

    with pytest.raises(ValueError, match="must contain at least one example"):
        ConditionalNodeFieldGenerator._build_train_val_subsets(dataset)


def test_update_ema_metric_tracks_smoothed_validation_signal():
    trainer = type("_Trainer", (), {"callback_metrics": {}, "logged_metrics": {}})()
    pl_module = type("_Module", (), {"_ema_metrics": {}, "early_stopping_ema_alpha": 0.25})()

    first = MetricsLogger._update_ema_metric(trainer, pl_module, "val_node_field", 100.0)
    second = MetricsLogger._update_ema_metric(trainer, pl_module, "val_node_field", 60.0)

    assert first == pytest.approx(100.0)
    assert second == pytest.approx(90.0)
    assert pl_module._ema_metrics["val_node_field"] == pytest.approx(90.0)
    assert trainer.callback_metrics["val_node_field_ema"].item() == pytest.approx(90.0)
    assert trainer.logged_metrics["val_node_field_ema"].item() == pytest.approx(90.0)


def test_component_summary_uses_raw_weighted_loss_scale():
    pl_module = type(
        "_Module",
        (),
        {
            "input_feature_dimension": 2048,
            "lambda_degree_importance": 1.0,
            "lambda_node_exist_importance": 0.0,
            "lambda_node_count_importance": 0.0,
            "lambda_node_label_importance": 1.0,
            "lambda_edge_label_importance": 0.0,
            "lambda_direct_edge_importance": 1.0,
            "lambda_edge_count_importance": 0.0,
            "lambda_degree_edge_consistency_importance": 0.0,
            "lambda_auxiliary_edge_importance": 0.0,
        },
    )()
    metrics = {
        "train_total": torch.tensor(121845.5),
        "train_node_field": torch.tensor(102374.0),
        "train_deg_ce": torch.tensor(4873.4),
        "train_node_label_ce": torch.tensor(5725.7),
        "train_edge_ce": torch.tensor(8872.4),
    }

    total, components, dominant_label, dominant_share = MetricsLogger._component_summary(
        pl_module,
        metrics,
        "train",
    )
    component_map = {label: (raw, weighted, share) for label, raw, weighted, share in components}

    assert component_map["node_field"][0] == pytest.approx(102374.0)
    assert component_map["node_field"][1] == pytest.approx(102374.0)
    assert component_map["deg"][1] == pytest.approx(4873.4)
    assert component_map["node_label"][1] == pytest.approx(5725.7)
    assert component_map["edge"][1] == pytest.approx(8872.4)
    assert total == pytest.approx(121845.5)
    assert dominant_label == "node_field"
    assert dominant_share == pytest.approx(102374.0 / 121845.5)


def test_format_metric_value_uses_more_precision_for_small_losses():
    assert MetricsLogger._format_metric_value(25.0).strip() == "25.000"
    assert MetricsLogger._format_metric_value(0.125).strip() == "0.12500"
    assert MetricsLogger._format_metric_value(3716.6).strip() == "3716.6"


def test_restored_checkpoint_summary_uses_node_field_label():
    summary = format_restored_checkpoint_summary(
        early_stopping_monitor="val_total",
        best_checkpoint_score=12.5,
        best_checkpoint_epoch=3,
        raw_best_val_node_field_loss=8.75,
        stopped_epoch=11,
    )

    assert "raw_val_node_field=8.7500" in summary


def test_compute_edge_count_loss_matches_target_on_consistent_probabilities():
    edge_probs = torch.tensor(
        [[[0.0, 0.8], [0.8, 0.0]]],
        dtype=torch.float32,
    )
    node_presence_mask = torch.tensor([[True, True]])
    target_edge_counts = torch.tensor([1.0], dtype=torch.float32)

    loss = ConditionalNodeFieldModule._compute_edge_count_loss(
        edge_probs=edge_probs,
        node_presence_mask=node_presence_mask,
        target_edge_counts=target_edge_counts,
    )

    assert loss.item() == pytest.approx(0.02, abs=1e-6)


def test_scale_normalized_huber_loss_is_invariant_to_shared_target_scale():
    small = ConditionalNodeFieldModule._scale_normalized_huber_loss(
        prediction=torch.tensor([12.0]),
        target=torch.tensor([10.0]),
        scale=torch.tensor([10.0]),
    )
    large = ConditionalNodeFieldModule._scale_normalized_huber_loss(
        prediction=torch.tensor([120.0]),
        target=torch.tensor([100.0]),
        scale=torch.tensor([100.0]),
    )

    assert small.item() == pytest.approx(large.item(), rel=1e-6)


def test_compute_degree_edge_consistency_loss_is_zero_when_handshake_identity_matches():
    logits_deg = torch.tensor(
        [[[ -10.0, 10.0], [ -10.0, 10.0]]],
        dtype=torch.float32,
    )
    node_presence_mask = torch.tensor([[True, True]])
    target_edge_counts = torch.tensor([1.0], dtype=torch.float32)

    loss = ConditionalNodeFieldModule(
        number_of_rows_per_example=2,
        input_feature_dimension=2,
        condition_feature_dimension=3,
        latent_embedding_dimension=4,
        number_of_transformer_layers=1,
        transformer_attention_head_count=1,
        max_degree=1,
    )._compute_degree_edge_consistency_loss(
        logits_deg=logits_deg,
        node_presence_mask=node_presence_mask,
        target_edge_counts=target_edge_counts,
    )

    assert loss.item() == pytest.approx(0.0, abs=1e-4)


def test_compute_node_count_loss_is_zero_when_expected_count_matches():
    logits_exist = torch.tensor(
        [[10.0, 10.0, -10.0]],
        dtype=torch.float32,
    )
    target_node_counts = torch.tensor([2.0], dtype=torch.float32)

    loss = ConditionalNodeFieldModule._compute_node_count_loss(
        logits_exist=logits_exist,
        target_node_counts=target_node_counts,
    )

    assert loss.item() == pytest.approx(0.0, abs=1e-4)


def test_compute_edge_count_loss_tracks_relative_not_absolute_error():
    edge_probs_small = torch.zeros((1, 6, 6), dtype=torch.float32)
    edge_probs_large = torch.zeros((1, 12, 12), dtype=torch.float32)
    for matrix in (edge_probs_small, edge_probs_large):
        matrix[0, 0, 1] = 1.0
        matrix[0, 1, 0] = 1.0
    edge_probs_small[0, 2, 3] = 1.0
    edge_probs_small[0, 3, 2] = 1.0
    edge_probs_small[0, 4, 5] = 0.2
    edge_probs_small[0, 5, 4] = 0.2
    edge_probs_large[0, 2, 3] = 1.0
    edge_probs_large[0, 3, 2] = 1.0
    edge_probs_large[0, 4, 5] = 1.0
    edge_probs_large[0, 5, 4] = 1.0
    edge_probs_large[0, 6, 7] = 1.0
    edge_probs_large[0, 7, 6] = 1.0
    edge_probs_large[0, 8, 9] = 1.0
    edge_probs_large[0, 9, 8] = 1.0
    edge_probs_large[0, 10, 11] = 0.2
    edge_probs_large[0, 11, 10] = 0.2

    loss_small = ConditionalNodeFieldModule._compute_edge_count_loss(
        edge_probs=edge_probs_small,
        node_presence_mask=torch.ones((1, 6), dtype=torch.bool),
        target_edge_counts=torch.tensor([2.5], dtype=torch.float32),
    )
    loss_large = ConditionalNodeFieldModule._compute_edge_count_loss(
        edge_probs=edge_probs_large,
        node_presence_mask=torch.ones((1, 12), dtype=torch.bool),
        target_edge_counts=torch.tensor([5.0], dtype=torch.float32),
    )

    assert loss_small.item() == pytest.approx(loss_large.item(), rel=1e-6)


def test_plot_metrics_accepts_node_field_key():
    plot_metrics(
        train_metrics={"total": [10.0, 9.0], "node_field": [8.0, 7.0]},
        val_metrics={"total": [11.0, 10.0], "node_field": [9.0, 8.0]},
        window=2,
    )


def test_tokenized_graph_conditioning_composes_and_scales_as_memory_tokens():
    graph_conditioning = GraphConditioningBatch(
        graph_embeddings=np.asarray([[0.25], [0.75]], dtype=float),
        node_counts=np.asarray([3, 5], dtype=np.int64),
        edge_counts=np.asarray([2, 4], dtype=np.int64),
        condition_node_embeddings=[
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=float),
            np.asarray([[0.5, 0.5]], dtype=float),
        ],
        condition_node_presence_mask=np.asarray(
            [
                [True, True],
                [True, False],
            ],
            dtype=bool,
        ),
    )
    node_batch = NodeGenerationBatch(
        node_embeddings_list=[
            np.asarray([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]], dtype=float),
            np.asarray([[0.4, 0.5], [0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [0.8, 0.9]], dtype=float),
        ],
        node_presence_mask=np.asarray(
            [
                [True, True, True, False, False],
                [True, True, True, True, True],
            ],
            dtype=bool,
        ),
        node_degree_targets=np.zeros((2, 5), dtype=np.int64),
    )
    generator = ConditionalNodeFieldGenerator(
        latent_embedding_dimension=8,
        number_of_transformer_layers=1,
        transformer_attention_head_count=1,
        maximum_epochs=1,
        batch_size=2,
        verbose=False,
    )

    condition_array = generator._compose_condition_array(graph_conditioning)

    assert condition_array.shape == (2, 2, 6)
    np.testing.assert_array_equal(condition_array[:, :, -2], np.asarray([[3.0, 3.0], [5.0, 5.0]]))
    np.testing.assert_array_equal(condition_array[:, :, -1], np.asarray([[2.0, 2.0], [4.0, 4.0]]))
    np.testing.assert_array_equal(condition_array[:, :, -3], np.asarray([[1.0, 1.0], [1.0, 0.0]]))

    generator.setup(node_batch=node_batch, graph_conditioning=graph_conditioning)
    payload = generator._build_processed_training_payload(node_batch, graph_conditioning)

    assert generator.condition_token_count == 2
    assert generator.condition_feature_dimension == 6
    assert payload["y_scaled"].shape == (2, 2, 6)
