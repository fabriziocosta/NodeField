"""YAML-driven artificial-graph hyperparameter optimization helpers."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ...feasibility_effort import resolve_feasibility_effort
from ...runtime_paths import (
    make_timestamped_run_dir,
    resolve_campaign_artifact_root,
    resolve_repo_root,
)
from ..synthetic import generate_artificial_dataset
from .pipeline import build_graph_generator, fit_graph_generator, sample_hyperparameter_configuration


_REQUIRED_TOP_LEVEL_SECTIONS = ("experiment", "dataset", "model", "generation", "outputs")
_VALID_FEASIBILITY_FILTERS = {"none", "fallback", "strict"}


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ModuleNotFoundError(
                "Loading YAML configs requires PyYAML. Install nodefield[notebooks]."
            ) from exc
    else:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Artificial-graph hyperparameter optimization config must contain a mapping.")
    return data


def _require_mapping(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config section {key!r} must be a mapping.")
    return value


def _require_positive_int(section: Mapping[str, Any], key: str) -> int:
    if key not in section:
        raise ValueError(f"{key} must be provided.")
    value = int(section[key])
    if value < 1:
        raise ValueError(f"{key} must be >= 1.")
    return value


def _validate_search_space(search_space: Mapping[str, Any]) -> None:
    for name, raw_spec in search_space.items():
        if not isinstance(raw_spec, dict):
            raise ValueError(f"Search-space entry {name!r} must be a mapping.")
        param_type = raw_spec.get("type")
        if param_type not in {"int", "real"}:
            raise ValueError(f"Unsupported hyperparameter type for {name!r}: {param_type!r}")
        if "low" not in raw_spec or "high" not in raw_spec:
            raise ValueError(f"Search-space entry {name!r} must define low and high.")
        if float(raw_spec["high"]) < float(raw_spec["low"]):
            raise ValueError(f"Search-space entry {name!r} has high < low.")


def _validate_artificial_hyperparameter_optimization_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    missing = [section for section in _REQUIRED_TOP_LEVEL_SECTIONS if section not in config]
    if missing:
        raise ValueError(f"Missing config sections: {', '.join(missing)}")

    normalized = deepcopy(dict(config))
    experiment = _require_mapping(normalized, "experiment")
    dataset = _require_mapping(normalized, "dataset")
    model = _require_mapping(normalized, "model")
    generation = _require_mapping(normalized, "generation")
    _require_mapping(normalized, "outputs")

    experiment["n_trials"] = _require_positive_int(experiment, "n_trials")
    experiment["random_state"] = int(experiment.get("random_state", 0))
    experiment["verbose"] = int(experiment.get("verbose", 1))

    for key in ("num_graphs", "cycle_length", "path_length", "num_rays", "ray_length"):
        _require_positive_int(dataset, key)
    dataset["seed"] = int(dataset.get("seed", experiment["random_state"]))
    dataset["save_config"] = False

    fixed = model.get("fixed", {})
    search_space = model.get("search_space", {})
    if not isinstance(fixed, dict):
        raise ValueError("model.fixed must be a mapping.")
    if not isinstance(search_space, dict) or not search_space:
        raise ValueError("model.search_space must be a non-empty mapping.")
    _validate_search_space(search_space)

    generation["n_samples"] = _require_positive_int(generation, "n_samples")
    effort = generation.get("feasibility_effort")
    resolve_feasibility_effort(effort)
    generation["feasibility_effort"] = int(effort)
    feasibility_filter = str(generation.get("feasibility_filter", "strict")).lower()
    if feasibility_filter not in _VALID_FEASIBILITY_FILTERS:
        raise ValueError(
            f"generation.feasibility_filter must be one of {sorted(_VALID_FEASIBILITY_FILTERS)}."
        )
    generation["feasibility_filter"] = feasibility_filter
    return normalized


def load_artificial_hyperparameter_optimization_config(path: str | Path) -> dict[str, Any]:
    """Load and validate an artificial-graph hyperparameter optimization config."""
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    return _validate_artificial_hyperparameter_optimization_config(_read_yaml_mapping(config_path))


def _context_path(context: Mapping[str, Any], key: str, default: str | Path) -> Path:
    return Path(context.get(key, default)).expanduser()


def _resolve_campaign_base_from_config(
    output_config: Mapping[str, Any],
    notebook_context: Mapping[str, Any],
) -> Path:
    repo_root = resolve_repo_root(notebook_context.get("REPO_ROOT", Path.cwd()))
    configured_root = output_config.get("artifact_root")
    return resolve_campaign_artifact_root(configured_root, repo_root=repo_root)


def _parse_run_timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    text = str(value)
    for fmt in ("%Y%m%d_%H%M%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(
        "outputs.run_timestamp must use YYYYMMDD_HHMMSS or ISO seconds format."
    )


def _number_of_violations(graph_generator: Any, generated_graphs: list[Any]) -> np.ndarray:
    feasibility_estimator = getattr(graph_generator, "feasibility_estimator", None)
    if feasibility_estimator is None or not hasattr(feasibility_estimator, "number_of_violations"):
        raise RuntimeError(
            "average_num_violations scoring requires graph_generator.feasibility_estimator."
        )
    return np.asarray(feasibility_estimator.number_of_violations(generated_graphs), dtype=float)


def _score_generated_graphs(graph_generator: Any, generated_graphs: list[Any]) -> dict[str, Any]:
    if len(generated_graphs) == 0:
        return {
            "returned_samples": 0,
            "average_num_violations": math.inf,
            "median_num_violations": math.inf,
            "feasible_count": 0,
            "feasible_rate": 0.0,
            "violation_counts": np.asarray([], dtype=float),
        }
    violation_counts = _number_of_violations(graph_generator, generated_graphs)
    if violation_counts.shape[0] != len(generated_graphs):
        raise RuntimeError(
            "Feasibility estimator returned an unexpected number of violation counts "
            f"({violation_counts.shape[0]} for {len(generated_graphs)} graphs)."
        )
    feasible_count = int(np.sum(violation_counts == 0))
    return {
        "returned_samples": len(generated_graphs),
        "average_num_violations": float(np.mean(violation_counts)),
        "median_num_violations": float(np.median(violation_counts)),
        "feasible_count": feasible_count,
        "feasible_rate": float(feasible_count / len(generated_graphs)),
        "violation_counts": violation_counts,
    }


def run_artificial_hyperparameter_optimization(
    config: Mapping[str, Any],
    *,
    notebook_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Train and score a random-search artificial-graph hyperparameter study."""
    config = _validate_artificial_hyperparameter_optimization_config(config)
    experiment = config["experiment"]
    dataset_config = config["dataset"]
    model_config = config["model"]
    generation_config = config["generation"]
    output_config = config["outputs"]

    artifact_base = _resolve_campaign_base_from_config(output_config, notebook_context)
    _context_path(notebook_context, "NOTEBOOK_DATA_ROOT", Path("notebooks") / "datasets")
    if output_config.get("run_dir"):
        artifact_root = Path(str(output_config["run_dir"])).expanduser().resolve()
        artifact_root.mkdir(parents=True, exist_ok=True)
    else:
        artifact_domain = str(output_config.get("artifact_subdir") or "artificial_graphs")
        artifact_prefix = str(output_config.get("artifact_prefix") or artifact_domain)
        artifact_root = make_timestamped_run_dir(
            artifact_base / artifact_domain,
            artifact_prefix,
            now=_parse_run_timestamp(output_config.get("run_timestamp")),
            short_id=output_config.get("run_id"),
        )
    for name in ("configs", "trials", "logs", "metrics", "samples"):
        (artifact_root / name).mkdir(parents=True, exist_ok=True)
    (artifact_root / "configs" / "resolved_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    graphs, _plot_artificial_graphs = generate_artificial_dataset(**dataset_config)

    rows: list[dict[str, Any]] = []
    best_graph_generator = None
    best_samples: list[Any] = []
    best_violation_counts = np.asarray([], dtype=float)
    best_sort_key = (math.inf, 0.0, 0)

    for trial_id in range(1, int(experiment["n_trials"]) + 1):
        trial_root = artifact_root / "trials" / f"trial_{trial_id:03d}"
        trial_root.mkdir(parents=True, exist_ok=True)
        sampled_params = sample_hyperparameter_configuration(
            model_config["search_space"],
            random_state=int(experiment["random_state"]) + trial_id,
        )
        generator_kwargs = {
            **model_config.get("fixed", {}),
            **sampled_params,
            "verbose": int(experiment["verbose"]),
            "artifact_root": trial_root / "artifacts",
            "checkpoint_root": trial_root / "checkpoints",
        }
        graph_generator = build_graph_generator(**generator_kwargs)
        graph_generator = fit_graph_generator(
            graph_generator,
            graphs,
            checkpoint_root=trial_root / "checkpoints",
        )
        generated_graphs = list(
            graph_generator.sample(
                n_samples=int(generation_config["n_samples"]),
                feasibility_effort=int(generation_config["feasibility_effort"]),
                feasibility_filter=generation_config["feasibility_filter"],
            )
        )
        score_info = _score_generated_graphs(graph_generator, generated_graphs)
        row = {
            "trial_id": trial_id,
            **sampled_params,
            "requested_samples": int(generation_config["n_samples"]),
            "returned_samples": score_info["returned_samples"],
            "average_num_violations": score_info["average_num_violations"],
            "median_num_violations": score_info["median_num_violations"],
            "feasible_count": score_info["feasible_count"],
            "feasible_rate": score_info["feasible_rate"],
            "feasibility_effort": int(generation_config["feasibility_effort"]),
            "feasibility_filter": generation_config["feasibility_filter"],
            "trial_root": str(trial_root),
        }
        rows.append(row)
        sort_key = (
            float(row["average_num_violations"]),
            -float(row["feasible_rate"]),
            int(row["trial_id"]),
        )
        if sort_key < best_sort_key:
            best_sort_key = sort_key
            best_graph_generator = graph_generator
            best_samples = generated_graphs
            best_violation_counts = score_info["violation_counts"]

    results_df = pd.DataFrame(rows).sort_values(
        ["average_num_violations", "feasible_rate", "trial_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    results_csv_path = None
    if output_config.get("results_csv"):
        results_csv_path = artifact_root / "metrics" / str(output_config["results_csv"])
        results_csv_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_csv_path, index=False)

    return {
        "config": config,
        "metadata": {"dataset": "artificial_cycle_path_star", **dataset_config},
        "manifest": {"dataset_name": "artificial_cycle_path_star", "num_graphs": len(graphs)},
        "artifact_root": artifact_root,
        "results_csv_path": results_csv_path,
        "results_df": results_df,
        "best_row": results_df.iloc[0].to_dict() if len(results_df) else None,
        "best_graph_generator": best_graph_generator,
        "best_samples": best_samples,
        "best_violation_counts": best_violation_counts,
    }


def summarize_artificial_hyperparameter_results(result: Mapping[str, Any]) -> pd.DataFrame:
    """Return the sorted trial summary table from an artificial optimization result bundle."""
    results_df = result.get("results_df")
    if results_df is None:
        raise ValueError("Result bundle does not contain results_df.")
    return pd.DataFrame(results_df).copy()


__all__ = [
    "load_artificial_hyperparameter_optimization_config",
    "run_artificial_hyperparameter_optimization",
    "summarize_artificial_hyperparameter_results",
]
