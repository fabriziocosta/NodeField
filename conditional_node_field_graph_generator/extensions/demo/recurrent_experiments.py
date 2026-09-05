"""Reproducible RENF experiment driver. Core models remain outside notebooks.

Artifacts are append-only within unique run directories. A failed generation is a
row, never a reason to drop a conditioning example from the metric denominator.
"""

from __future__ import annotations

import inspect
import json
import platform
import random
import subprocess
import time
import uuid
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import dill
import networkx as nx
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from ...conditional_node_field_generator import (
    ConditionalNodeFieldGenerator,
    ConditionalNodeFieldGraphWithEdgesDataset,
    collate_conditional_node_field_graph_with_edges,
)
from ...recurrent_interventions import RecurrentIntervention

EXPERIMENT_NAME = "recurrent_energy_nodefield_ablation_v1"
SEEDS = [0, 1, 2, 3, 4]
CONFIG_NAMES = ("baseline", "recurrent_energy_constant", "recurrent_energy_annealed")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot serialize {type(value)}")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=_json_default))


def environment_metadata():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return dict(
        git_commit=commit,
        git_dirty=dirty,
        python_version=platform.python_version(),
        torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,
        device_name=torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else platform.processor() or "CPU",
        date=datetime.now(timezone.utc).isoformat(),
        determinism="Seeded Python/NumPy/Torch; deterministic algorithms warn on unsupported operations; solver and accelerator versions can affect results.",
    )


def load_config(name, root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[3]
    with (root / "configs" / "recurrent_nodefield" / f"{name}.yaml").open() as f:
        return yaml.safe_load(f)


def primary_metrics(graph, condition_graph, generator):
    """Structural fidelity uses measured statistics, never generated metadata."""
    from .artificial_conditioning import artificial_graph_stats

    missing = dict(
        valid=False,
        decoder_success=False,
        feasible_condition_match=False,
        node_count_accuracy=False,
        edge_count_accuracy=False,
        condition_error=np.nan,
        degree_consistency=np.nan,
        node_label_distribution_error=np.nan,
        edge_label_distribution_error=np.nan,
        feasibility_violations=np.nan,
    )
    if graph is None or graph.number_of_nodes() == 0:
        return missing
    actual, target = (
        artificial_graph_stats(graph, condition_graph),
        artificial_graph_stats(condition_graph),
    )
    keys = (
        "total_nodes",
        "total_edges",
        "cycle_count",
        "cycle_sizes",
        "path_length",
        "path_component_sizes",
        "ray_count",
        "ray_sizes",
        "star_hub_count",
    )
    mismatches = sum(actual[k] != target[k] for k in keys)
    structural_valid = all(
        actual[k] for k in ("cycles_are_valid", "path_is_valid", "rays_are_valid")
    ) and nx.is_connected(graph)
    estimator = generator.feasibility_estimator
    if estimator is None:
        raise RuntimeError("Canonical experiment requires the repository feasibility estimator")
    violations = float(np.asarray(estimator.number_of_violations([graph])).reshape(-1)[0])
    valid = structural_valid and violations == 0

    def distribution_error(first, second):
        a, b = Counter(first), Counter(second)
        na, nb = max(1, sum(a.values())), max(1, sum(b.values()))
        return 0.5 * sum(abs(a[k] / na - b[k] / nb) for k in a.keys() | b.keys())

    return dict(
        valid=valid,
        decoder_success=True,
        feasible_condition_match=bool(valid and mismatches == 0),
        node_count_accuracy=actual["total_nodes"] == target["total_nodes"],
        edge_count_accuracy=actual["total_edges"] == target["total_edges"],
        condition_error=mismatches / len(keys),
        degree_consistency=distribution_error(
            dict(graph.degree()).values(), dict(condition_graph.degree()).values()
        ),
        node_label_distribution_error=distribution_error(
            nx.get_node_attributes(graph, "label").values(),
            nx.get_node_attributes(condition_graph, "label").values(),
        ),
        edge_label_distribution_error=distribution_error(
            nx.get_edge_attributes(graph, "label").values(),
            nx.get_edge_attributes(condition_graph, "label").values(),
        ),
        feasibility_violations=violations,
    )


def experiment_conditions(depth, *, smoke, config):
    """Return (name, interventions, inference-noise label, Langevin scale)."""
    half = depth // 2
    yield "normal", None, "none", 0.0
    yield "reset_h_mid", RecurrentIntervention("reset_hidden", step=half), "none", 0.0
    yield (
        "fresh_x_mid",
        RecurrentIntervention("fresh_x_noise", step=half, seed=0),
        "unit_gaussian",
        0.0,
    )
    yield (
        "fresh_x_every_step",
        RecurrentIntervention("fresh_x_noise_every_step", seed=0),
        "unit_gaussian",
        0.0,
    )
    if smoke:
        return
    yield "reset_h_every_step", RecurrentIntervention("reset_hidden", every_step=True), "none", 0.0
    yield (
        "reset_both_mid",
        [
            RecurrentIntervention("fresh_x_noise", step=half, seed=0),
            RecurrentIntervention("reset_hidden", step=half),
        ],
        "unit_gaussian",
        0.0,
    )
    yield (
        "reset_both_every_step",
        [
            RecurrentIntervention("fresh_x_noise_every_step", seed=0),
            RecurrentIntervention("reset_hidden", every_step=True),
        ],
        "unit_gaussian",
        0.0,
    )
    for fraction in config["reset_fractions"]:
        step = int(fraction * depth)
        yield f"reset_h_{fraction}", RecurrentIntervention("reset_hidden", step=step), "none", 0.0
        for seed in config["shuffle_seeds"]:
            yield (
                f"shuffle_h_{fraction}_{seed}",
                RecurrentIntervention("shuffle_hidden_nodes", step=step, seed=seed),
                "none",
                0.0,
            )
    for scale in config["noise_scales"]:
        yield (
            f"fresh_x_sigma_{scale}",
            RecurrentIntervention("fresh_x_noise_every_step", seed=0, noise_scale=scale),
            f"normal_0_{scale}",
            0.0,
        )
    yield "langevin_original", None, "langevin_original", config["langevin_original"]


def _node_batch(generator, graphs):
    embeddings = generator.node_encode(graphs)
    labels = generator.graphs_to_node_label_targets(graphs)
    edge_labels, edge_pairs = generator.graphs_to_edge_label_targets(graphs)
    return generator.node_batch_builder_.build_training_node_batch(
        graphs,
        node_embeddings_list=embeddings,
        node_label_targets=labels,
        edge_label_targets=edge_labels,
        edge_label_pairs=edge_pairs,
        supervision_plan=generator.supervision_plan_,
        log_details=False,
    )


def _loader(owner, batch, condition, batch_size, seed, shuffle):
    payload = owner._build_processed_training_payload(batch, condition)
    dataset = owner._build_dataset_from_processed_payload(payload)
    collate = (
        collate_conditional_node_field_graph_with_edges
        if isinstance(dataset, ConditionalNodeFieldGraphWithEdgesDataset)
        else None
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )


class RecurrentExperiment:
    def __init__(self, output_root="artifact/recurrent_nodefield", *, smoke=True, config_root=None):
        self.smoke = smoke
        self.configs = {name: load_config(name, config_root) for name in CONFIG_NAMES}
        self.config = self.configs[CONFIG_NAMES[0]]
        self.run_dir = Path(output_root) / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        )
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.rows = []
        self.diagnostic_rows = []
        self.models = {}
        self.sanity_rows = []
        self.prepared = False
        write_json(self.run_dir / "environment.json", environment_metadata())
        write_json(
            self.run_dir / "config.json",
            dict(
                experiment_name=EXPERIMENT_NAME,
                smoke=smoke,
                configs=self.configs,
                decisions=dict(
                    primary_metric="feasible_condition_match",
                    minimum_effect=0.05,
                    significance="paired seed-level 95% CI excludes zero; smoke has no significance claim",
                    parameter_matching="raw RENF; compare equal field evaluations and report parameters",
                    checkpoint_D="alias of C selected by validation loss",
                    anytime="validation-calibrated thresholds, 3 consecutive stable steps",
                ),
            ),
        )

    def prepare(self):
        if self.prepared:
            return self
        from ..synthetic import generate_artificial_dataset
        from .pipeline import build_graph_generator

        torch.set_num_threads(self.config["experiment"]["threads"])
        seed_everything(self.config["dataset"]["seed"])
        dataset = dict(self.config["dataset"])
        if self.smoke:
            dataset["num_graphs"] = self.config["experiment"]["smoke_graphs"]
        self.graphs, _ = generate_artificial_dataset(**dataset, save_config=False)
        order = np.random.default_rng(self.config["experiment"]["split_seed"]).permutation(
            len(self.graphs)
        )
        a, b = int(0.8 * len(order)), int(0.9 * len(order))
        self.splits = dict(train=order[:a], validation=order[a:b], test=order[b:])
        write_json(self.run_dir / "split_indices.json", self.splits)
        with (self.run_dir / "dataset.pkl").open("wb") as f:
            dill.dump(self.graphs, f)
        options = dict(self.config["model"])
        options.update(
            artifact_root=str(self.run_dir),
            checkpoint_root=str(self.run_dir / "checkpoints"),
            model_dir=str(self.run_dir / "models"),
        )
        self.template = build_graph_generator(**options)
        training = [self.graphs[i] for i in self.splits["train"]]
        self.template.fit(training, train_node_generator=False)
        self.template.graph_decoder.adjacency_time_limit_seconds = self.config["experiment"][
            "decode_time_limit_seconds"
        ]
        self.batches = {}
        self.conditions = {}
        for split, indices in self.splits.items():
            graphs = [self.graphs[i] for i in indices]
            self.batches[split] = _node_batch(self.template, graphs)
            self.conditions[split] = self.template.graph_encode(graphs)
        # Shared fitted preprocessing and supervision are persisted before model initialization.
        with (self.run_dir / "preprocessing.pkl").open("wb") as f:
            dill.dump(
                dict(generator=self.template, batches=self.batches, conditions=self.conditions), f
            )
        self.prepared = True
        return self

    def make_model(self, name, seed):
        self.prepare()
        seed_everything(seed)
        generator = deepcopy(self.template)
        config = self.configs[name]
        keys = inspect.signature(ConditionalNodeFieldGenerator).parameters
        opts = {k: v for k, v in config["model"].items() if k in keys}
        owner = ConditionalNodeFieldGenerator(**opts)
        owner.supervision_plan_ = generator.supervision_plan_
        generator.conditional_node_generator_model = owner
        owner.setup(self.batches["train"], self.conditions["train"])
        owner.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        owner.model.to(owner.device)
        generator.is_fitted_ = True
        return generator

    def sanity_checks(self):
        from unittest.mock import patch

        for name in CONFIG_NAMES:
            generator = self.make_model(name, 0)
            owner = generator.conditional_node_generator_model
            m = owner.model
            batch = next(
                iter(_loader(owner, self.batches["train"], self.conditions["train"], 16, 0, False))
            )
            batch = tuple(v.to(owner.device) for v in batch)
            m.train()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            with patch.object(m, "log"):
                loss = m.training_step(batch, 0)
            loss.backward()
            grads = [p.grad for p in m.parameters() if p.grad is not None]
            if not torch.isfinite(loss) or not all(torch.isfinite(g).all() for g in grads):
                raise FloatingPointError(f"Nonfinite sanity gradients: {name}")
            x, c = batch[:2]
            mask = batch[-3] if m.use_node_label_head else batch[-2]
            if m.node_field_mode == "recurrent_energy":
                h = m._initialize_recurrent_state(*x.shape[:2], x.device, x.dtype)
                m.eval()
                score, _, _, hn = m._compute_recurrent_score_field(
                    x.detach().requires_grad_(True), h, c, mask, create_graph=False
                )
                altered, _, _, _ = m._compute_recurrent_score_field(
                    x.detach().requires_grad_(True),
                    torch.randn_like(h),
                    c,
                    mask,
                    create_graph=False,
                )
                assert not torch.allclose(score, altered), "Hidden state does not influence scores"
                assert torch.count_nonzero(hn[~mask]) == 0
                hidden_norm = float(hn.detach().square().mean().sqrt())
            else:
                score, _, _ = m._compute_score_field(
                    x.detach().requires_grad_(True), c, mask, create_graph=False
                )
                hidden_norm = 0.0
            row = dict(
                model=name,
                loss=float(loss.detach()),
                score_norm=float(score.detach().norm()),
                hidden_norm=hidden_norm,
                gradient_norm=float(
                    torch.stack([g.detach().norm() ** 2 for g in grads]).sum().sqrt()
                ),
                parameter_count=sum(p.numel() for p in m.parameters() if p.requires_grad),
                peak_gpu_memory=torch.cuda.max_memory_allocated()
                if torch.cuda.is_available()
                else None,
            )
            self.sanity_rows.append(row)
            print(row, flush=True)
        pd.DataFrame(self.sanity_rows).to_csv(self.run_dir / "sanity.csv", index=False)
        return pd.DataFrame(self.sanity_rows)

    def train(self):
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
        from pytorch_lightning.loggers import CSVLogger

        self.prepare()
        if not self.sanity_rows:
            self.sanity_checks()

        class FiniteGradients(Callback):
            def on_after_backward(self, trainer, module):
                if any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in module.parameters()
                ):
                    raise FloatingPointError("Nonfinite training gradient")

            def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs
                if not torch.isfinite(loss).all():
                    raise FloatingPointError("Nonfinite training loss")

            def on_validation_epoch_start(self, trainer, module):
                self.rng = torch.get_rng_state()
                self.cuda_rng = (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                )
                torch.manual_seed(12345)

            def on_validation_epoch_end(self, trainer, module):
                torch.set_rng_state(self.rng)
                if self.cuda_rng is not None:
                    torch.cuda.set_rng_state_all(self.cuda_rng)
                if not torch.isfinite(trainer.callback_metrics["val_total"]):
                    raise FloatingPointError("Nonfinite validation loss")

        seeds = self.config["experiment"]["smoke_seeds"] if self.smoke else self.config["seeds"]
        from .pipeline import build_graph_generator

        defaults = {
            k: v.default
            for k, v in inspect.signature(build_graph_generator).parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        for seed in seeds:
            for name in CONFIG_NAMES:
                generator = self.make_model(name, seed)
                owner = generator.conditional_node_generator_model
                directory = self.run_dir / f"{name}-seed{seed}"
                directory.mkdir(exist_ok=False)
                config = self.configs[name]
                epochs = (
                    config["experiment"]["smoke_epochs"]
                    if self.smoke
                    else config["model"]["maximum_epochs"]
                )
                resolved = dict(defaults, **config["model"])
                resolved.update(maximum_epochs=epochs, enable_early_stopping=not self.smoke)
                write_json(
                    directory / "config.json",
                    dict(
                        seed=seed,
                        model=resolved,
                        parameter_count=sum(
                            p.numel() for p in owner.model.parameters() if p.requires_grad
                        ),
                        sigma_sequence=owner.model._build_recurrent_sigma_schedule(
                            owner.recurrent_training_steps, "cpu", torch.float32
                        )
                        if name != "baseline"
                        else [owner.node_field_sigma],
                    ),
                )
                checkpoint = ModelCheckpoint(
                    dirpath=directory / "checkpoints",
                    filename="{epoch:03d}-{step}",
                    monitor="val_total",
                    mode="min",
                    save_top_k=-1,
                    save_last=True,
                )
                callbacks = [FiniteGradients(), checkpoint]
                if not self.smoke:
                    callbacks.append(
                        EarlyStopping(
                            "val_total",
                            patience=config["model"]["early_stopping_patience"],
                            min_delta=0.0,
                        )
                    )
                trainer = pl.Trainer(
                    max_epochs=epochs,
                    accelerator="gpu" if torch.cuda.is_available() else "cpu",
                    devices=1,
                    logger=CSVLogger(str(directory), name="logs"),
                    callbacks=callbacks,
                    enable_progress_bar=False,
                    enable_model_summary=False,
                    inference_mode=False,
                    deterministic="warn",
                    num_sanity_val_steps=0,
                    log_every_n_steps=1,
                )
                train_loader = _loader(
                    owner,
                    self.batches["train"],
                    self.conditions["train"],
                    owner.batch_size,
                    seed,
                    True,
                )
                val_loader = _loader(
                    owner,
                    self.batches["validation"],
                    self.conditions["validation"],
                    owner.batch_size,
                    seed,
                    False,
                )
                started = time.perf_counter()
                seed_everything(seed)
                trainer.fit(owner.model, train_loader, val_loader)
                owner.model.load_state_dict(
                    torch.load(
                        checkpoint.best_model_path, map_location=owner.device, weights_only=False
                    )["state_dict"]
                )
                owner.best_checkpoint_path_ = checkpoint.best_model_path
                owner.model.eval()
                with (directory / "generator.pkl").open("wb") as f:
                    dill.dump(generator, f)
                self.models[(seed, name)] = generator
                write_json(
                    directory / "training.json",
                    dict(
                        seconds=time.perf_counter() - started,
                        updates=trainer.global_step,
                        best_checkpoint=checkpoint.best_model_path,
                        best_validation_loss=float(checkpoint.best_model_score),
                    ),
                )
                print(f"Trained {name} seed={seed} updates={trainer.global_step}", flush=True)
        write_json(
            self.run_dir / "checkpoint_aliases.json",
            {"D": "recurrent_energy_annealed (C), same validation-selected checkpoint"},
        )
        return self

    def _decode(self, generator, batch, condition):
        from ...conditional_node_field_graph_decoder import decode_generated_nodes

        graphs = decode_generated_nodes(
            generator,
            batch,
            graph_conditioning=condition,
            feasibility_oracle_candidates_per_attempt=0,
            use_ilp_decoder=True,
        )
        return graphs[0] if graphs else None

    def _evaluate_one(
        self,
        generator,
        seed,
        name,
        depth,
        label,
        intervention,
        noise,
        langevin,
        split,
        index,
        example_id,
        checkpoint_label="best",
    ):
        owner = generator.conditional_node_generator_model
        m = owner.model
        condition = self.conditions[split].take([index])
        target = self.graphs[example_id]
        sampling_seed = seed * 100000 + int(example_id) + 1000
        owner.sampling_steps = depth
        m.langevin_noise_scale = langevin
        trajectory = None
        failure = None
        graph = None
        diagnostic_seconds = 0.0
        started = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        try:
            seed_everything(sampling_seed)
            if name == "baseline":
                batch = owner.predict(condition)
            else:
                batch = owner.predict_recurrent(
                    condition, total_steps=depth, intervention=intervention
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            generation_seconds = time.perf_counter() - started
            decode_started = time.perf_counter()
            graph = self._decode(generator, batch, condition)
            decode_seconds = time.perf_counter() - decode_started
            metrics = primary_metrics(graph, target, generator)
            if name != "baseline":
                ds = time.perf_counter()
                seed_everything(sampling_seed)
                _, trajectory = owner.predict_recurrent(
                    condition, total_steps=depth, intervention=intervention, return_trajectory=True
                )
                diagnostic_seconds = time.perf_counter() - ds
        except FloatingPointError:
            raise
        except (RuntimeError, ValueError, nx.NetworkXException) as exc:
            failure = f"{type(exc).__name__}: {exc}"
            generation_seconds = time.perf_counter() - started
            decode_seconds = 0.0
            metrics = primary_metrics(None, target, generator)
        items = (
            []
            if intervention is None
            else (
                [intervention] if isinstance(intervention, RecurrentIntervention) else intervention
            )
        )
        base = dict(
            seed=seed,
            checkpoint=checkpoint_label,
            model=name,
            model_mode=owner.node_field_mode,
            training_schedule="baseline_fixed"
            if name == "baseline"
            else owner.recurrent_corruption_schedule,
            K_train=owner.recurrent_training_steps if name != "baseline" else 1,
            K_test=depth,
            intervention=label,
            intervention_step=next((i.step for i in items if i.step is not None), None),
            inference_noise=noise,
            split=split,
            example_id=int(example_id),
            sampling_seed=sampling_seed,
        )
        row = dict(
            base,
            **metrics,
            runtime_seconds=generation_seconds + decode_seconds,
            generation_seconds=generation_seconds,
            decode_seconds=decode_seconds,
            diagnostic_seconds=diagnostic_seconds,
            field_evaluations=depth,
            diagnostic_readouts=depth if trajectory is not None else 0,
            final_readout_field_evaluations=1 if name == "baseline" else 0,
            peak_gpu_memory=torch.cuda.max_memory_allocated()
            if torch.cuda.is_available()
            else None,
            failure=failure,
        )
        self.rows.append(row)
        if trajectory is not None:
            path = (
                self.run_dir
                / "trajectories"
                / f"{seed}-{name}-{checkpoint_label}-{split}-{example_id}-{depth}-{label}.pt"
            )
            path.parent.mkdir(exist_ok=True)
            torch.save(trajectory, path)
            for step, diagnostic in enumerate(trajectory.diagnostics):
                readout = trajectory.readouts[step]
                exist = (
                    readout["exist_logits"].sigmoid().sum().item()
                    if readout["exist_logits"] is not None
                    else readout["node_embeddings"].shape[1]
                )
                head_error = abs(exist - target.number_of_nodes())
                self.diagnostic_rows.append(
                    dict(base, **diagnostic, head_node_count_error=head_error, trajectory=str(path))
                )
        return row

    def evaluate(self):
        if not self.models:
            raise RuntimeError("Call train() before evaluate()")
        depths = (
            self.config["experiment"]["smoke_depths"]
            if self.smoke
            else self.config["experiment"]["depths"]
        )
        for (seed, name), generator in self.models.items():
            for depth in depths:
                conditions = (
                    [("normal", None, "none", 0.0)]
                    if name == "baseline"
                    else list(
                        experiment_conditions(
                            depth, smoke=self.smoke, config=self.config["experiment"]
                        )
                    )
                )
                for label, intervention, noise, langevin in conditions:
                    for index, example_id in enumerate(self.splits["test"]):
                        self._evaluate_one(
                            generator,
                            seed,
                            name,
                            depth,
                            label,
                            intervention,
                            noise,
                            langevin,
                            "test",
                            index,
                            example_id,
                        )
                self.save_tables()
                print(f"Evaluated {name} seed={seed} depth={depth}", flush=True)
        if not self.smoke:
            self.evaluate_matched_updates()
            self.evaluate_anytime()
        return self.save_tables()

    def evaluate_matched_updates(self):
        """Compare curricula at the earlier stopping time using retained epoch checkpoints."""
        for seed in self.config["seeds"]:
            names = CONFIG_NAMES[1:]
            checkpoints = {}
            for name in names:
                paths = (self.run_dir / f"{name}-seed{seed}" / "checkpoints").glob("epoch=*.ckpt")
                checkpoints[name] = {
                    int(torch.load(p, map_location="cpu", weights_only=False)["global_step"]): p
                    for p in paths
                }
            common = set(checkpoints[names[0]]) & set(checkpoints[names[1]])
            step = max(common)
            for name in names:
                generator = self.models[(seed, name)]
                m = generator.conditional_node_generator_model.model
                best = deepcopy(m.state_dict())
                m.load_state_dict(
                    torch.load(checkpoints[name][step], map_location="cpu", weights_only=False)[
                        "state_dict"
                    ]
                )
                depth = generator.conditional_node_generator_model.recurrent_training_steps
                for index, example_id in enumerate(self.splits["test"]):
                    self._evaluate_one(
                        generator,
                        seed,
                        name,
                        depth,
                        "normal",
                        None,
                        "none",
                        0.0,
                        "test",
                        index,
                        example_id,
                        f"matched_updates_{step}",
                    )
                m.load_state_dict(best)
        self.save_tables()

    def evaluate_anytime(self):
        """Calibrate on validation, freeze, then compare test quality to the full rollout."""
        results = []
        thresholds = []
        depth = max(self.config["experiment"]["depths"])
        consecutive = self.config["experiment"]["stability_consecutive_steps"]
        for (seed, name), generator in self.models.items():
            if name == "baseline":
                continue
            owner = generator.conditional_node_generator_model
            owner.model.langevin_noise_scale = 0.0
            traces = {}
            for split in ("validation", "test"):
                traces[split] = []
                for index, example_id in enumerate(self.splits[split]):
                    condition = self.conditions[split].take([index])
                    seed_everything(seed * 100000 + int(example_id) + 1000)
                    _, trace = owner.predict_recurrent(
                        condition, total_steps=depth, return_trajectory=True
                    )
                    traces[split].append((index, int(example_id), condition, trace))
            rows = [r for _, _, _, trace in traces["validation"] for r in trace.diagnostics]
            h_values = [r["hidden_delta_norm"] for r in rows]
            p_values = [r["prediction_delta"] for r in rows if r["prediction_delta"] is not None]
            s_values = [r["score_norm"] for r in rows]
            candidates = []
            for q in (0.1, 0.25, 0.5, 0.75):
                ht = float(np.quantile(h_values, q))
                pt = float(np.quantile(p_values, q))
                st = float(np.quantile(s_values, q))
                candidates.extend(
                    [
                        dict(rule="hidden", hidden_threshold=ht, prediction_threshold=float("inf")),
                        dict(
                            rule="score",
                            hidden_threshold=float("inf"),
                            prediction_threshold=float("inf"),
                            score_threshold=st,
                        ),
                        dict(
                            rule="prediction",
                            hidden_threshold=float("inf"),
                            prediction_threshold=pt,
                        ),
                        dict(rule="combined", hidden_threshold=ht, prediction_threshold=pt),
                    ]
                )
            # Include a full-budget fallback; adaptive stopping is optional, never forced.
            candidates.append(
                dict(rule="full_budget", hidden_threshold=0.0, prediction_threshold=0.0)
            )
            cache = {}

            def quality(split, index, example_id, condition, trace, step):
                key = (split, index, step)
                if key not in cache:
                    try:
                        graph = _readout_graph(self, generator, trace, step, condition)
                    except (RuntimeError, ValueError, nx.NetworkXException):
                        graph = None
                    cache[key] = primary_metrics(graph, self.graphs[example_id], generator)[
                        "feasible_condition_match"
                    ]
                return float(cache[key])

            for candidate in candidates:
                kwargs = {k: v for k, v in candidate.items() if k != "rule"}
                losses = []
                steps = []
                for index, example_id, condition, trace in traces["validation"]:
                    step = stabilization_step(trace.diagnostics, consecutive=consecutive, **kwargs)
                    losses.append(
                        quality("validation", index, example_id, condition, trace, depth)
                        - quality("validation", index, example_id, condition, trace, step)
                    )
                    steps.append(step)
                candidate["validation_quality_loss"] = float(np.mean(losses))
                candidate["validation_mean_steps"] = float(np.mean(steps))
            eligible = [c for c in candidates if c["validation_quality_loss"] <= 0.05]
            chosen = min(
                eligible, key=lambda c: (c["validation_mean_steps"], c["validation_quality_loss"])
            )
            thresholds.append(dict(seed=seed, model=name, **chosen))
            kwargs = {
                k: chosen[k]
                for k in ("hidden_threshold", "prediction_threshold", "score_threshold")
                if k in chosen
            }
            for index, example_id, condition, trace in traces["test"]:
                step = stabilization_step(trace.diagnostics, consecutive=consecutive, **kwargs)
                full = quality("test", index, example_id, condition, trace, depth)
                early = quality("test", index, example_id, condition, trace, step)
                results.append(
                    dict(
                        seed=seed,
                        model=name,
                        example_id=example_id,
                        split="test",
                        rule=chosen["rule"],
                        steps=step,
                        steps_saved=depth - step,
                        quality_loss=full - early,
                        full_quality=full,
                        adaptive_quality=early,
                    )
                )
            # Separate decoder-unchanged candidate: evaluate isomorphism, not node ordering.
            # This expensive diagnostic is evaluated on validation first and uses the same fixed M on test.
            for split in ("validation", "test"):
                for index, example_id, condition, trace in traces[split]:
                    last = None
                    streak = 0
                    step = depth
                    for k in range(1, depth + 1):
                        try:
                            graph = _readout_graph(self, generator, trace, k, condition)
                        except (RuntimeError, ValueError, nx.NetworkXException):
                            graph = None
                        same = (
                            graph is not None
                            and last is not None
                            and nx.is_isomorphic(
                                graph,
                                last,
                                node_match=nx.algorithms.isomorphism.categorical_node_match(
                                    "label", None
                                ),
                                edge_match=nx.algorithms.isomorphism.categorical_edge_match(
                                    "label", None
                                ),
                            )
                        )
                        streak = streak + 1 if same else 0
                        last = graph
                        if streak >= consecutive:
                            step = k
                            break
                    full = quality(split, index, example_id, condition, trace, depth)
                    early = float(
                        primary_metrics(last, self.graphs[example_id], generator)[
                            "feasible_condition_match"
                        ]
                    )
                    results.append(
                        dict(
                            seed=seed,
                            model=name,
                            example_id=example_id,
                            split=split,
                            rule="decoder_unchanged_fixed_M",
                            steps=step,
                            steps_saved=depth - step,
                            quality_loss=full - early,
                            full_quality=full,
                            adaptive_quality=early,
                        )
                    )
        write_json(self.run_dir / "anytime_thresholds.json", thresholds)
        frame = pd.DataFrame(results)
        frame.to_csv(self.run_dir / "anytime.csv", index=False)
        frame[frame.split == "test"].groupby(["model", "rule"]).agg(
            average_steps_saved=("steps_saved", "mean"),
            quality_loss=("quality_loss", "mean"),
            worst_case_quality_loss=("quality_loss", "max"),
        ).to_csv(self.run_dir / "anytime_summary.csv")
        return frame

    def save_tables(self):
        frame = pd.DataFrame(self.rows)
        frame.to_csv(self.run_dir / "results.csv", index=False)
        pd.DataFrame(self.diagnostic_rows).to_csv(self.run_dir / "diagnostics.csv", index=False)
        return frame

    def run(self):
        try:
            self.prepare()
            self.sanity_checks()
            self.train()
            self.evaluate()
            summarize_results(self.run_dir)
            plot_results(self.run_dir)
            write_json(self.run_dir / "status.json", dict(status="complete", smoke=self.smoke))
        except Exception as exc:
            self.save_tables()
            write_json(
                self.run_dir / "failure.json", dict(type=type(exc).__name__, message=str(exc))
            )
            raise
        return self.run_dir


def summarize_results(run_dir):
    """Seed-level Student-t intervals and paired effects; never pool examples as seeds."""
    from scipy.stats import t

    run_dir = Path(run_dir)
    df = pd.read_csv(run_dir / "results.csv")
    metrics = [
        "feasible_condition_match",
        "valid",
        "decoder_success",
        "node_count_accuracy",
        "edge_count_accuracy",
        "condition_error",
        "generation_seconds",
        "decode_seconds",
    ]
    keys = ["model", "checkpoint", "training_schedule", "K_test", "intervention", "inference_noise"]
    means = df.groupby(keys + ["seed"], dropna=False)[metrics].mean().reset_index()
    rows = []
    for group, data in means.groupby(keys, dropna=False):
        for metric in metrics:
            values = data[metric].dropna()
            n = len(values)
            mean = values.mean()
            sd = values.std(ddof=1) if n > 1 else np.nan
            width = float(t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else np.nan
            rows.append(
                dict(
                    zip(keys, group),
                    metric=metric,
                    mean=mean,
                    std=sd,
                    ci_low=mean - width,
                    ci_high=mean + width,
                    seeds=n,
                )
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / "summary.csv", index=False)
    effects = []
    pivot = df[df.checkpoint == "best"].pivot_table(
        index=["seed", "example_id", "K_test"],
        columns=["model", "intervention", "inference_noise"],
        values="feasible_condition_match",
    )
    references = [
        (("baseline", "normal", "none"), ("recurrent_energy_annealed", "normal", "none")),
        (
            ("recurrent_energy_constant", "normal", "none"),
            ("recurrent_energy_annealed", "normal", "none"),
        ),
    ]
    for model in CONFIG_NAMES[1:]:
        normal = (model, "normal", "none")
        for column in pivot.columns:
            if column[0] == model and column != normal:
                references.append((column, normal))
    for before, after in references:
        if before not in pivot or after not in pivot:
            continue
        delta = (pivot[after] - pivot[before]).dropna()
        for depth, values in delta.groupby(level="K_test"):
            seed_values = values.groupby(level="seed").mean()
            n = len(seed_values)
            mean = seed_values.mean()
            width = t.ppf(0.975, n - 1) * seed_values.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            effects.append(
                dict(
                    reference=str(before),
                    comparison=str(after),
                    K_test=depth,
                    paired_effect=mean,
                    ci_low=mean - width,
                    ci_high=mean + width,
                    seeds=n,
                )
            )
    pd.DataFrame(effects).to_csv(run_dir / "paired_effects.csv", index=False)
    return summary


def stabilization_step(
    records, *, hidden_threshold, prediction_threshold, consecutive=3, score_threshold=float("inf")
):
    """Return completed-step count, requiring all enabled thresholds consecutively."""
    streak = 0
    for index, row in enumerate(records):
        pred = row.get("prediction_delta")
        stable = (
            pred is not None
            and np.isfinite(pred)
            and row["hidden_delta_norm"] < hidden_threshold
            and pred < prediction_threshold
            and row["score_norm"] < score_threshold
        )
        streak = streak + 1 if stable else 0
        if streak >= consecutive:
            return index + 1
    return len(records)


def _readout_graph(experiment, generator, trajectory, step, condition):
    owner = generator.conditional_node_generator_model
    owner.model._cache_generation_heads(
        trajectory.readouts[step - 1]["latent_tokens"].to(owner.device)
    )
    x = trajectory.x[step].numpy()
    batch = owner._build_generated_node_batch(owner._inverse_transform_input(x))
    return experiment._decode(generator, batch, condition)


def plot_results(run_dir):
    """Regenerate all plots from saved tables; figures contain no hand-entered observations."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    out = run_dir / "figures"
    out.mkdir(exist_ok=True)
    df = pd.read_csv(run_dir / "results.csv")
    diagnostics = pd.read_csv(run_dir / "diagnostics.csv")
    config = json.loads((run_dir / "config.json").read_text())
    ktrain = config["configs"]["recurrent_energy_annealed"]["model"]["recurrent_training_steps"]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    ax.text(0.02, 0.75, "Baseline:  xₖ → −∇ₓ φ(xₖ, c) → xₖ₊₁", fontsize=14)
    ax.text(0.02, 0.35, "RENF:  (xₖ, hₖ) → [−∇ₓ φ(xₖ, hₖ, c), hₖ₊₁] → (xₖ₊₁, hₖ₊₁)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "01_architecture.png", dpi=150)
    plt.close(fig)

    def depth_plot(data, metric, filename, groups=("model", "intervention")):
        fig, ax = plt.subplots(figsize=(8, 4))
        for key, group in data.groupby(list(groups)):
            per_seed = group.groupby(["seed", "K_test"])[metric].mean().reset_index()
            means = per_seed.groupby("K_test")[metric].mean()
            ax.plot(
                means.index,
                means.values,
                marker="o",
                label=" / ".join(key) if isinstance(key, tuple) else key,
            )
        ax.axvline(ktrain, color="gray", ls="--", label="K_train")
        ax.set_xscale("log", base=2)
        ax.set(xlabel="Field evaluations / inference depth", ylabel=metric)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / filename, dpi=150)
        plt.close(fig)

    main = df[(df.checkpoint == "best") & df.intervention.isin(["normal", "reset_h_every_step"])]
    depth_plot(main, "feasible_condition_match", "02_quality_depth.png")
    depth_plot(main, "decoder_success", "02_decoder_success.png")
    depth_plot(main, "condition_error", "02_condition_error.png")
    memory = df[
        df.intervention.str.startswith(("normal", "reset_h", "shuffle_h"))
        & (df.model == "recurrent_energy_annealed")
    ]
    depth_plot(memory, "feasible_condition_match", "03_memory_interventions.png", ("intervention",))
    channels = df[
        df.intervention.isin(
            [
                "normal",
                "fresh_x_every_step",
                "reset_h_every_step",
                "reset_both_every_step",
                "reset_h_mid",
                "fresh_x_mid",
                "reset_both_mid",
            ]
        )
        & (df.model == "recurrent_energy_annealed")
    ]
    depth_plot(
        channels, "feasible_condition_match", "04_information_channels.png", ("intervention",)
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, metric in zip(
        axes.flat, ["hidden_delta_norm", "score_norm", "phi", "prediction_delta"]
    ):
        subset = diagnostics[
            (diagnostics.intervention == "normal")
            & (diagnostics.K_test == diagnostics.K_test.max())
        ]
        for name, group in subset.groupby("model"):
            mean = group.groupby("step")[metric].mean()
            ax.plot(mean.index + 1, mean.values, label=name)
        ax.axvline(ktrain, color="gray", ls="--")
        ax.set(xlabel="Completed recurrent steps", ylabel=metric)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "05_state_dynamics.png", dpi=150)
    plt.close(fig)
    history = []
    for path in run_dir.glob("*-seed*/logs/version_*/metrics.csv"):
        part = pd.read_csv(path)
        part["model"] = path.parents[2].name
        history.append(part)
    if history:
        history = pd.concat(history, ignore_index=True)
        history.to_csv(run_dir / "training_history.csv", index=False)
        for metrics, name in [
            (["train_total", "val_total"], "training_total"),
            (["train_node_field", "val_node_field"], "training_score"),
            (
                [
                    "train_deg_ce",
                    "train_exist",
                    "train_edge_ce",
                    "train_node_label_ce",
                    "train_node_count_loss",
                    "train_edge_count_loss",
                ],
                "training_structural",
            ),
        ]:
            fig, ax = plt.subplots(figsize=(10, 5))
            for model, group in history.groupby("model"):
                for metric in metrics:
                    if metric in group:
                        values = group.dropna(subset=[metric])
                        ax.plot(values.step, values[metric], label=f"{model}/{metric}")
            ax.set(xlabel="Optimization step", ylabel="Loss")
            ax.legend(fontsize=6)
            fig.tight_layout()
            fig.savefig(out / f"{name}.png", dpi=150)
            plt.close(fig)
    return sorted(out.glob("*.png"))


def run_smoke(output_root="artifact/recurrent_nodefield"):
    return RecurrentExperiment(output_root, smoke=True).run()


def load_results(run_dir):
    root = Path(run_dir)
    return pd.read_csv(root / "results.csv"), pd.read_csv(root / "diagnostics.csv")


def analysis_section(results, diagnostics, section, *, run_dir=None, k_train=8):
    if section == "fixed_depth":
        return (
            results[(results.K_test == k_train) & (results.intervention == "normal")]
            .groupby("model")
            .mean(numeric_only=True)
        )
    if section == "depth":
        return (
            results[results.intervention.isin(["normal", "reset_h_every_step"])]
            .groupby(["model", "K_test"])
            .mean(numeric_only=True)
        )
    if section == "reset":
        return diagnostics[diagnostics.intervention.str.startswith(("normal", "reset_h"))]
    if section == "shuffle":
        return results[results.intervention.str.startswith(("normal", "reset_h", "shuffle_h"))]
    if section == "channels":
        return results[
            results.intervention.isin(
                [
                    "normal",
                    "fresh_x_every_step",
                    "reset_h_every_step",
                    "reset_both_every_step",
                    "fresh_x_mid",
                    "reset_h_mid",
                    "reset_both_mid",
                ]
            )
        ]
    if section == "curriculum":
        return (
            results[results.model_mode == "recurrent_energy"]
            .groupby(["training_schedule", "checkpoint", "K_test", "intervention"])
            .mean(numeric_only=True)
        )
    if section == "no_memory":
        return results[results.intervention.isin(["normal", "reset_h_every_step"])]
    if section == "noise":
        return results.groupby(["model", "inference_noise", "K_test"]).mean(numeric_only=True)
    if section == "stability":
        return diagnostics
    if section in ("anytime", "statistics"):
        path = Path(run_dir) / ("anytime_summary.csv" if section == "anytime" else "summary.csv")
        return (
            pd.read_csv(path)
            if path.exists()
            else "Prepared for full run; not executed in smoke mode."
        )
    raise ValueError(f"Unknown analysis section: {section}")


def decision_report(run_dir):
    root = Path(run_dir)
    config = json.loads((root / "config.json").read_text())
    if config["smoke"]:
        report = {
            "scope": "smoke",
            "conclusion": "Numerical and workflow checks only; no significance or mechanism claim from one seed.",
        }
    else:
        effects = pd.read_csv(root / "paired_effects.csv")
        effects["meets_predefined_effect_criterion"] = (
            (effects.paired_effect >= 0.05) & (effects.ci_low > 0) & (effects.seeds >= 2)
        )
        report = {
            "scope": "full",
            "paired_comparisons": effects.to_dict(orient="records"),
            "interpretation": "Intervention effects are evidence of functional dependence, not semantic decoding of memory or proof of convergence.",
        }
    write_json(root / "decision_report.json", report)
    write_json(root / "status.json", dict(status="complete", smoke=config["smoke"]))
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output-root", default="artifact/recurrent_nodefield")
    args = parser.parse_args()
    print(RecurrentExperiment(args.output_root, smoke=not args.full).run())
