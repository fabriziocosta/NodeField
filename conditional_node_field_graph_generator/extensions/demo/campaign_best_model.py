"""Inspect campaign results and sample from the best completed trial."""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch

from ...runtime_paths import resolve_repo_root
from ..synthetic import generate_artificial_dataset, make_artificial_graph_plotter
from .pipeline import build_graph_generator, build_zinc_dataset, sample_hyperparameter_configuration
from .trial_scoring import trial_sort_key
from .trial_snapshots import (
    load_trial_graph_generator_snapshot,
    trial_graph_generator_snapshot_path,
)


TERMINAL_CAMPAIGN_STATUSES = {"campaign_completed", "openai_credits_exhausted"}


@dataclass(frozen=True)
class CampaignTrialSelection:
    campaign_state_path: Path | None
    campaign_state: dict[str, Any]
    domain: str
    prefix: str
    run_dir: Path
    trial_dir: Path
    metrics_path: Path
    config_path: Path
    checkpoint_path: Path
    generator_snapshot_path: Path | None
    metrics: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        return _read_json(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def _artifact_root(repo_root: Path) -> Path:
    return repo_root / "artifact"


def list_campaign_state_paths(repo_root: str | Path | None = None) -> list[Path]:
    """Return campaign-level state files under ``artifact/``."""
    root = resolve_repo_root(repo_root)
    artifact_root = _artifact_root(root)
    if not artifact_root.is_dir():
        return []
    return sorted(artifact_root.glob("*/*_campaign_state.json"))


def find_latest_campaign_state(repo_root: str | Path | None = None) -> tuple[Path | None, dict[str, Any]]:
    """Find the most recent active campaign state, falling back to the latest state."""
    state_paths = list_campaign_state_paths(repo_root)
    states: list[tuple[Path, dict[str, Any]]] = []
    for path in state_paths:
        try:
            states.append((path, _read_json(path)))
        except (OSError, json.JSONDecodeError):
            continue
    if not states:
        return None, {}
    active = [
        (path, state)
        for path, state in states
        if str(state.get("status", "")) not in TERMINAL_CAMPAIGN_STATUSES
    ]
    candidates = active or states
    return max(candidates, key=lambda item: item[0].stat().st_mtime)


def _infer_campaign_from_state(
    repo_root: Path,
    campaign_state_path: Path | None,
    campaign_state: Mapping[str, Any],
) -> tuple[str, str]:
    domain = str(campaign_state.get("campaign") or "")
    prefix = str(campaign_state.get("prefix") or campaign_state.get("campaign_id") or "")
    if domain and prefix:
        return domain, prefix
    if campaign_state_path is not None:
        domain = campaign_state_path.parent.name
        suffix = "_campaign_state.json"
        prefix = campaign_state_path.name[: -len(suffix)] if campaign_state_path.name.endswith(suffix) else ""
        if domain and prefix:
            return domain, prefix
    artifact_root = _artifact_root(repo_root)
    run_dirs = sorted(
        (path for path in artifact_root.glob("*/*_*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        summary = run_dir / "metrics" / "summary.json"
        if not summary.is_file():
            continue
        domain = run_dir.parent.name
        parts = run_dir.name.split("_")
        if len(parts) >= 4:
            prefix = "_".join(parts[:-3])
        else:
            prefix = run_dir.name
        return domain, prefix
    raise FileNotFoundError("No campaign state or completed campaign run was found under artifact/.")


def _find_checkpoint(trial_dir: Path) -> Path:
    best = sorted(trial_dir.glob("**/checkpoints/**/best-*.ckpt"))
    if best:
        return best[-1]
    last = sorted(trial_dir.glob("**/checkpoints/**/last.ckpt"))
    if last:
        return last[-1]
    raise FileNotFoundError(f"No checkpoint found under {trial_dir}")


def _resolve_generator_snapshot_path(metrics: Mapping[str, Any], trial_dir: Path) -> Path | None:
    configured_path = metrics.get("generator_snapshot_path")
    if configured_path:
        path = Path(str(configured_path)).expanduser()
        if path.is_file():
            return path.resolve()
    stable_path = trial_graph_generator_snapshot_path(trial_dir)
    if stable_path.is_file():
        return stable_path
    return None


def collect_campaign_trial_results(
    *,
    repo_root: str | Path | None = None,
    campaign_state_path: str | Path | None = None,
) -> pd.DataFrame:
    """Collect ranked trial metrics for the latest active campaign."""
    root = resolve_repo_root(repo_root)
    if campaign_state_path is None:
        state_path, state = find_latest_campaign_state(root)
    else:
        state_path = Path(campaign_state_path).expanduser().resolve()
        state = _read_json(state_path)
    domain, prefix = _infer_campaign_from_state(root, state_path, state)
    domain_root = _artifact_root(root) / domain
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(domain_root.glob(f"{prefix}_*")):
        if not run_dir.is_dir():
            continue
        for metrics_path in sorted(run_dir.glob("trials/trial_*/metrics.json")):
            try:
                metrics = _read_json(metrics_path)
                checkpoint_path = _find_checkpoint(metrics_path.parent)
            except (OSError, json.JSONDecodeError, FileNotFoundError):
                continue
            config_path = metrics_path.parent / "config.yaml"
            if not config_path.is_file():
                continue
            generator_snapshot_path = _resolve_generator_snapshot_path(metrics, metrics_path.parent)
            average = metrics.get("average_num_violations")
            if average is None:
                continue
            sort_key = trial_sort_key(metrics)
            rows.append(
                {
                    **metrics,
                    "domain": domain,
                    "prefix": prefix,
                    "run_dir": str(run_dir),
                    "run_name": run_dir.name,
                    "trial_dir": str(metrics_path.parent),
                    "trial_name": metrics_path.parent.name,
                    "metrics_path": str(metrics_path),
                    "config_path": str(config_path),
                    "checkpoint_path": str(checkpoint_path),
                    "generator_snapshot_path": (
                        str(generator_snapshot_path) if generator_snapshot_path else ""
                    ),
                    "campaign_state_path": str(state_path) if state_path is not None else "",
                    "campaign_status": state.get("status", ""),
                    "_run_mtime": run_dir.stat().st_mtime,
                    "_optimization_sort": sort_key[0],
                    "_average_sort": sort_key[1],
                    "_distance_sort": sort_key[2],
                    "_feasible_sort": sort_key[3],
                }
            )
    if not rows:
        raise FileNotFoundError(
            f"No completed trial metrics with checkpoints found for campaign {domain}/{prefix}."
        )
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(
        [
            "_optimization_sort",
            "_average_sort",
            "_distance_sort",
            "_feasible_sort",
            "_run_mtime",
            "run_name",
            "trial_name",
        ],
        ascending=[True, True, True, True, False, True, True],
    ).reset_index(drop=True)
    return frame.drop(
        columns=[
            "_optimization_sort",
            "_average_sort",
            "_distance_sort",
            "_feasible_sort",
            "_run_mtime",
        ]
    )


def select_best_campaign_trial(
    *,
    repo_root: str | Path | None = None,
    campaign_state_path: str | Path | None = None,
) -> CampaignTrialSelection:
    """Select the lowest-average-violation completed trial for the latest campaign."""
    frame = collect_campaign_trial_results(
        repo_root=repo_root,
        campaign_state_path=campaign_state_path,
    )
    row = frame.iloc[0].to_dict()
    state_path = Path(row["campaign_state_path"]) if row.get("campaign_state_path") else None
    state = _read_json(state_path) if state_path is not None and state_path.is_file() else {}
    metrics = _read_json(Path(row["metrics_path"]))
    return CampaignTrialSelection(
        campaign_state_path=state_path,
        campaign_state=state,
        domain=str(row["domain"]),
        prefix=str(row["prefix"]),
        run_dir=Path(row["run_dir"]),
        trial_dir=Path(row["trial_dir"]),
        metrics_path=Path(row["metrics_path"]),
        config_path=Path(row["config_path"]),
        checkpoint_path=Path(row["checkpoint_path"]),
        generator_snapshot_path=(
            Path(row["generator_snapshot_path"]) if row.get("generator_snapshot_path") else None
        ),
        metrics=metrics,
    )


def _build_dataset_for_trial(config: Mapping[str, Any], notebook_context: Mapping[str, Any]) -> list[Any]:
    outputs = config.get("outputs", {})
    domain = str(outputs.get("artifact_subdir") or "")
    dataset_config = dict(config["dataset"])
    if domain == "molecules":
        notebook_data_root = Path(notebook_context.get("NOTEBOOK_DATA_ROOT", Path("notebooks") / "datasets"))
        graphs, _metadata, _manifest = build_zinc_dataset(
            dataset_dir=notebook_data_root / "zinc",
            num_examples=dataset_config["num_graphs"],
            min_size=dataset_config["min_size"],
            max_size=dataset_config["max_size"],
            random_state=dataset_config["random_state"],
        )
        return list(graphs)
    dataset_config["save_config"] = False
    graphs, _plotter = generate_artificial_dataset(**dataset_config)
    return list(graphs)


def _setup_generator_for_checkpoint(graph_generator: Any, graphs: list[Any]) -> None:
    artifacts = graph_generator._prepare_fit_artifacts(graphs, targets=None)
    supervision_plan = artifacts["supervision_plan"]
    edge_targets_for_cond_gen = None
    edge_pairs_for_cond_gen = None
    auxiliary_edge_targets_for_cond_gen = None
    auxiliary_edge_pairs_for_cond_gen = None
    if supervision_plan.direct_edges.enabled:
        graph_generator._log_supervision_plan(supervision_plan)
        edge_targets_for_cond_gen, edge_pairs_for_cond_gen = (
            graph_generator.graph_decoder.compute_edge_supervision(
                graphs,
                artifacts["node_embeddings_list"],
                locality_sample_fraction=graph_generator.locality_sample_fraction,
                negative_sample_factor=graph_generator.negative_sample_factor,
                locality_sampling_strategy=graph_generator.locality_sampling_strategy,
                locality_target_positive_ratio=graph_generator.locality_target_positive_ratio,
                horizon=1,
                supervision_name="direct_edge",
            )
        )
        if supervision_plan.auxiliary_locality.enabled:
            auxiliary_edge_targets_for_cond_gen, auxiliary_edge_pairs_for_cond_gen = (
                graph_generator.graph_decoder.compute_edge_supervision(
                    graphs,
                    artifacts["node_embeddings_list"],
                    locality_sample_fraction=graph_generator.locality_sample_fraction,
                    negative_sample_factor=graph_generator.negative_sample_factor,
                    locality_sampling_strategy=graph_generator.locality_sampling_strategy,
                    locality_target_positive_ratio=graph_generator.locality_target_positive_ratio,
                    horizon=supervision_plan.auxiliary_locality.horizon,
                    supervision_name="aux_locality",
                )
            )
    else:
        graph_generator._log_supervision_plan(supervision_plan)
    node_batch = graph_generator._build_node_batch(
        graphs,
        artifacts["node_embeddings_list"],
        node_label_targets=(
            artifacts["node_label_targets"] if supervision_plan.node_labels.enabled else None
        ),
        edge_pairs=edge_pairs_for_cond_gen,
        edge_targets=edge_targets_for_cond_gen,
        edge_label_pairs=(
            artifacts["edge_label_pairs"] if supervision_plan.edge_labels.enabled else None
        ),
        edge_label_targets=(
            artifacts["edge_label_targets"] if supervision_plan.edge_labels.enabled else None
        ),
        auxiliary_edge_pairs=auxiliary_edge_pairs_for_cond_gen,
        auxiliary_edge_targets=auxiliary_edge_targets_for_cond_gen,
    )
    graph_generator.conditional_node_generator_model.setup(
        node_batch=node_batch,
        graph_conditioning=artifacts["graph_conditioning"],
        targets=None,
    )
    setattr(
        graph_generator.conditional_node_generator_model,
        "_graph_generator_snapshot_owner",
        graph_generator,
    )


def _set_generator_device(graph_generator: Any, device: str) -> None:
    resolved_device = torch.device(str(device))
    node_model = graph_generator.conditional_node_generator_model
    node_model.device = resolved_device
    if getattr(node_model, "model", None) is not None:
        node_model.model.to(resolved_device)


def _load_checkpoint_into_generator(
    graph_generator: Any,
    checkpoint_path: Path,
    device: str,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    graph_generator.conditional_node_generator_model.model.load_state_dict(state_dict)
    _set_generator_device(graph_generator, device)
    graph_generator.is_fitted_ = True
    graph_generator.best_checkpoint_path_ = str(checkpoint_path)
    graph_generator.best_checkpoint_epoch_ = checkpoint.get("epoch")
    return checkpoint


def load_campaign_trial_generator(
    selection: CampaignTrialSelection,
    *,
    notebook_context: Mapping[str, Any],
    device: str = "cpu",
) -> Any:
    """Load a campaign trial generator snapshot and apply the selected checkpoint weights."""
    if selection.generator_snapshot_path is not None and selection.generator_snapshot_path.is_file():
        graph_generator = load_trial_graph_generator_snapshot(selection.generator_snapshot_path)
        _set_generator_device(graph_generator, device)
        _load_checkpoint_into_generator(graph_generator, selection.checkpoint_path, device)
        graph_generator.campaign_trial_load_mode_ = "snapshot"
        graph_generator.generator_snapshot_path_ = str(selection.generator_snapshot_path)
        return graph_generator

    warnings.warn(
        "Campaign trial generator snapshot is missing; rebuilding fit artifacts from the "
        "trial config for legacy compatibility. New campaign runs should record "
        "generator_snapshot_path in metrics.json.",
        RuntimeWarning,
        stacklevel=2,
    )
    config = _read_yaml_or_json(selection.config_path)
    experiment = config["experiment"]
    model_config = config["model"]
    graphs = _build_dataset_for_trial(config, notebook_context)
    sampled_params = sample_hyperparameter_configuration(
        model_config["search_space"],
        random_state=int(experiment["random_state"]) + 1,
    )
    generator_kwargs = {
        **model_config.get("fixed", {}),
        **sampled_params,
        "verbose": int(experiment.get("verbose", 1)),
        "artifact_root": selection.trial_dir / "reload_artifacts",
        "checkpoint_root": selection.trial_dir / "reload_checkpoints",
    }
    graph_generator = build_graph_generator(**generator_kwargs)
    _set_generator_device(graph_generator, device)
    _setup_generator_for_checkpoint(graph_generator, graphs)
    _load_checkpoint_into_generator(graph_generator, selection.checkpoint_path, device)
    graph_generator.campaign_trial_load_mode_ = "legacy_rebuild"
    graph_generator.generator_snapshot_path_ = None
    return graph_generator


def load_campaign_trial_training_examples(
    selection: CampaignTrialSelection,
    *,
    notebook_context: Mapping[str, Any],
    n_examples: int = 7,
) -> list[Any]:
    """Rebuild the selected trial training dataset and return example graphs."""
    config = _read_yaml_or_json(selection.config_path)
    graphs = _build_dataset_for_trial(config, notebook_context)
    return list(graphs[: int(n_examples)])


def build_campaign_trial_artificial_plotter(selection: CampaignTrialSelection):
    """Build the artificial-graph plotter matching the selected trial dataset config."""
    config = _read_yaml_or_json(selection.config_path)
    dataset = dict(config.get("dataset", {}))
    return make_artificial_graph_plotter(
        int(dataset.get("node_alphabet_size", 3)),
        node_alphabet_kind=str(dataset.get("node_alphabet_kind", "int")),
        component_specific_alphabets=bool(dataset.get("component_specific_alphabets", True)),
    )


def sample_from_best_campaign_trial(
    *,
    notebook_context: Mapping[str, Any],
    repo_root: str | Path | None = None,
    campaign_state_path: str | Path | None = None,
    n_samples: int | None = None,
    feasibility_effort: int | None = None,
    feasibility_filter: str | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load the best campaign trial and sample graphs from it."""
    selection = select_best_campaign_trial(
        repo_root=repo_root,
        campaign_state_path=campaign_state_path,
    )
    config = _read_yaml_or_json(selection.config_path)
    generation = dict(config.get("generation", {}))
    sample_count = int(n_samples or generation.get("n_samples") or 8)
    effort = int(feasibility_effort if feasibility_effort is not None else generation.get("feasibility_effort", 2))
    filter_mode = str(feasibility_filter if feasibility_filter is not None else generation.get("feasibility_filter", "none"))
    graph_generator = load_campaign_trial_generator(
        selection,
        notebook_context=notebook_context,
        device=device,
    )
    generated_graphs = list(
        graph_generator.sample(
            n_samples=sample_count,
            feasibility_effort=effort,
            feasibility_filter=filter_mode,
        )
    )
    violation_counts = graph_generator.feasibility_estimator.number_of_violations(generated_graphs)
    violation_values = [float(value) for value in violation_counts]
    average = float(sum(violation_values) / len(violation_values)) if violation_values else math.nan
    feasible_rate = (
        float(sum(value == 0 for value in violation_values) / len(violation_values))
        if violation_values
        else math.nan
    )
    return {
        "selection": selection,
        "ranking": collect_campaign_trial_results(
            repo_root=repo_root,
            campaign_state_path=campaign_state_path,
        ),
        "graph_generator": graph_generator,
        "generated_graphs": generated_graphs,
        "violation_counts": violation_values,
        "sample_summary": {
            "n_samples": sample_count,
            "returned_samples": len(generated_graphs),
            "feasibility_effort": effort,
            "feasibility_filter": filter_mode,
            "average_num_violations": average,
            "feasible_rate": feasible_rate,
        },
    }


__all__ = [
    "CampaignTrialSelection",
    "build_campaign_trial_artificial_plotter",
    "collect_campaign_trial_results",
    "find_latest_campaign_state",
    "list_campaign_state_paths",
    "load_campaign_trial_generator",
    "load_campaign_trial_training_examples",
    "sample_from_best_campaign_trial",
    "select_best_campaign_trial",
]
