"""Filesystem-backed NodeField campaign controller utilities."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, TextIO

from .runtime_paths import (
    make_timestamped_run_dir,
    resolve_campaign_artifact_root,
    resolve_repo_root,
)
from .campaign_search import flatten_leaf_paths, sample_patch_space, validate_patch_space


CAMPAIGN_CONFIGS = {
    "molecules": Path("configs") / "campaigns" / "molecules.yaml",
    "artificial_graphs": Path("configs") / "campaigns" / "artificial_graphs.yaml",
    "artificial-graphs": Path("configs") / "campaigns" / "artificial_graphs.yaml",
}

RUN_SUBDIRECTORIES = ("configs", "trials", "logs", "metrics", "samples")

MUTABLE_GROUP_PATHS = {
    "dataset": ["dataset"],
    "generation": ["generation"],
    "loss_weights": [
        "model.search_space.lambda_degree_importance",
        "model.search_space.lambda_node_exist_importance",
        "model.search_space.lambda_node_count_importance",
        "model.search_space.lambda_node_label_importance",
        "model.search_space.lambda_edge_label_importance",
        "model.search_space.lambda_direct_edge_importance",
        "model.search_space.lambda_auxiliary_edge_importance",
        "model.search_space.lambda_edge_count_importance",
        "model.search_space.lambda_degree_edge_consistency_importance",
        "model.search_space.default_exist_pos_weight",
    ],
    "sampling": [
        "model.search_space.sampling_step_size",
        "model.search_space.sampling_steps",
        "model.search_space.langevin_noise_scale",
        "model.search_space.sparse_supervision_mask_ratio",
        "generation",
    ],
    "training": [
        "model.fixed.learning_rate",
        "model.fixed.batch_size",
        "model.fixed.maximum_epochs",
        "model.fixed.total_steps",
        "model.fixed.enable_early_stopping",
        "model.fixed.early_stopping_patience",
        "model.fixed.early_stopping_min_delta",
    ],
    "architecture": [
        "model.fixed.latent_embedding_dimension",
        "model.fixed.node_embedding_svd_dimension",
        "model.fixed.graph_embedding_svd_dimension",
        "model.fixed.number_of_transformer_layers",
        "model.fixed.transformer_attention_head_count",
        "model.fixed.transformer_dropout",
        "model.fixed.locality_horizon",
        "model.fixed.locality_sample_fraction",
    ],
}

VALID_PROPOSAL_MODES = {"range_search", "exact_configs"}


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def _dump_yaml_or_json(data: Mapping[str, Any]) -> str:
    try:
        import yaml
    except ImportError:
        return json.dumps(_json_safe(data), indent=2, sort_keys=True) + "\n"
    return yaml.safe_dump(_json_safe(data), sort_keys=False)


def _write_yaml_or_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dump_yaml_or_json(data), encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _deep_merge(base: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_repo_path(repo_root: Path, path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = repo_root / resolved
    return resolved.resolve()


def _deduplicate_paths(paths: list[str]) -> list[str]:
    deduplicated: list[str] = []
    seen = set()
    for path in paths:
        text = str(path)
        if text in seen:
            continue
        deduplicated.append(text)
        seen.add(text)
    return deduplicated


def _resolve_allowed_paths(agent: Mapping[str, Any]) -> list[str]:
    paths: list[str] = []
    groups = agent.get("mutable_groups", [])
    if groups is None:
        groups = []
    if not isinstance(groups, list):
        raise ValueError("agent.mutable_groups must be a list when provided.")
    unknown_groups = sorted(str(group) for group in groups if str(group) not in MUTABLE_GROUP_PATHS)
    if unknown_groups:
        raise ValueError("Unknown mutable group(s): " + ", ".join(unknown_groups))
    for group in groups:
        paths.extend(MUTABLE_GROUP_PATHS[str(group)])
    explicit_paths = agent.get("allowed_paths", [])
    if explicit_paths is None:
        explicit_paths = []
    if not isinstance(explicit_paths, list):
        raise ValueError("agent.allowed_paths must be a list when provided.")
    paths.extend(str(path) for path in explicit_paths)
    return _deduplicate_paths(paths)


def _path_allowed(path: str, allowed_paths: list[str]) -> bool:
    return any(path == allowed or path.startswith(f"{allowed}.") for allowed in allowed_paths)


def _validate_exact_patches(patches: list[Mapping[str, Any]], *, allowed_paths: list[str]) -> None:
    for index, patch in enumerate(patches, start=1):
        if not isinstance(patch, Mapping):
            raise ValueError(f"Exact trial config {index} must be a mapping.")
        rejected = sorted(
            path
            for path in flatten_leaf_paths(patch)
            if not _path_allowed(path, allowed_paths)
        )
        if rejected:
            raise ValueError(
                f"Exact trial config {index} contains non-allowlisted path(s): "
                + ", ".join(rejected)
            )


def load_campaign_config(
    path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load and normalize a NodeField campaign config."""
    root = resolve_repo_root(repo_root)
    config_path = _resolve_repo_path(root, path)
    config = _read_yaml_or_json(config_path)
    campaign = config.setdefault("campaign", {})
    if not isinstance(campaign, dict):
        raise ValueError("campaign section must be a mapping.")
    domain = str(campaign.get("domain") or campaign.get("id") or config_path.stem)
    if domain == "artificial-graphs":
        domain = "artificial_graphs"
    campaign["domain"] = domain
    campaign["prefix"] = str(campaign.get("prefix") or domain)
    campaign["id"] = str(campaign.get("id") or domain)
    campaign["config_path"] = str(config_path)

    random_search = config.setdefault("random_search", {})
    if not isinstance(random_search, dict):
        raise ValueError("random_search section must be a mapping.")
    random_search["batch_size"] = int(random_search.get("batch_size", 3))
    if random_search["batch_size"] < 1:
        raise ValueError("random_search.batch_size must be >= 1.")
    random_search["random_state"] = int(random_search.get("random_state", 0))

    runner = config.setdefault("runner", {})
    if not isinstance(runner, dict):
        raise ValueError("runner section must be a mapping.")
    if "config_path" not in runner:
        raise ValueError("runner.config_path must be provided.")
    runner["config_path"] = str(_resolve_repo_path(root, runner["config_path"]))

    agent = config.setdefault("agent", {})
    if not isinstance(agent, dict):
        raise ValueError("agent section must be a mapping.")
    proposal_mode = str(agent.get("proposal_mode") or "range_search")
    if proposal_mode not in VALID_PROPOSAL_MODES:
        raise ValueError(f"agent.proposal_mode must be one of {sorted(VALID_PROPOSAL_MODES)}.")
    agent["proposal_mode"] = proposal_mode
    allowed_paths = _resolve_allowed_paths(agent)
    if not allowed_paths:
        raise ValueError(
            "agent.allowed_paths or agent.mutable_groups must provide at least one path."
        )
    agent["allowed_paths"] = allowed_paths
    agent["max_search_leaf_count"] = int(agent.get("max_search_leaf_count", len(allowed_paths)))
    if proposal_mode == "range_search":
        if "default_trial_patch_space" not in agent:
            raise ValueError("agent.default_trial_patch_space must be provided for range_search.")
    else:
        exact_configs = agent.get("default_trial_configs")
        if not isinstance(exact_configs, list) or not exact_configs:
            raise ValueError(
                "agent.default_trial_configs must be a non-empty list for exact_configs."
            )
        _validate_exact_patches(exact_configs, allowed_paths=allowed_paths)

    artifacts = config.setdefault("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("artifacts section must be a mapping.")
    artifacts["root"] = str(artifacts.get("root") or "artifact")

    logbook = config.setdefault("logbook", {})
    if not isinstance(logbook, dict):
        raise ValueError("logbook section must be a mapping.")
    default_logbook = f"LOGBOOK_{campaign['domain']}.md"
    logbook["path"] = str(_resolve_repo_path(root, logbook.get("path") or default_logbook))
    config["_repo_root"] = str(root)
    return config


def resolve_campaign_config(campaign: str, *, repo_root: str | Path | None = None) -> Path:
    """Resolve a built-in campaign name or explicit config path."""
    root = resolve_repo_root(repo_root)
    key = str(campaign)
    rel_path = CAMPAIGN_CONFIGS.get(key, Path(key))
    return _resolve_repo_path(root, rel_path)


def list_campaigns(*, repo_root: str | Path | None = None) -> list[dict[str, Any]]:
    """Return known campaign configs and whether their config files exist."""
    root = resolve_repo_root(repo_root)
    rows = []
    for name in ("molecules", "artificial_graphs"):
        path = _resolve_repo_path(root, CAMPAIGN_CONFIGS[name])
        rows.append({"campaign": name, "config_path": str(path), "exists": path.is_file()})
    return rows


def _campaign_artifact_root(config: Mapping[str, Any]) -> Path:
    repo_root = Path(str(config["_repo_root"]))
    root = config["artifacts"].get("root")
    return resolve_campaign_artifact_root(root, repo_root=repo_root)


def _campaign_domain_root(config: Mapping[str, Any]) -> Path:
    return _campaign_artifact_root(config) / str(config["campaign"]["domain"])


def _make_run_dir(
    config: Mapping[str, Any],
    *,
    now: datetime | None = None,
    short_id: str | None = None,
    dry_run: bool = False,
) -> Path:
    return make_timestamped_run_dir(
        _campaign_domain_root(config),
        str(config["campaign"]["prefix"]),
        now=now,
        short_id=short_id,
        create=not dry_run,
    )


def _latest_run_dir(config: Mapping[str, Any]) -> Path | None:
    root = _campaign_domain_root(config)
    prefix = str(config["campaign"]["prefix"])
    if not root.is_dir():
        return None
    matches = sorted(
        (path for path in root.glob(f"{prefix}_*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def apply_exact_trial_patch(
    base_config: Mapping[str, Any],
    patch: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply an exact candidate patch, preserving one-point model search-space specs."""
    patched = deepcopy(dict(base_config))
    model_patch = patch.get("model") if isinstance(patch, Mapping) else None
    search_space_patch = None
    if isinstance(model_patch, Mapping):
        search_space_patch = model_patch.get("search_space")
    if isinstance(search_space_patch, Mapping):
        for name, exact_value in search_space_patch.items():
            if isinstance(exact_value, Mapping) and "type" in exact_value:
                continue
            current = patched.setdefault("model", {}).setdefault("search_space", {}).get(name, {})
            spec_type = current.get("type", "real") if isinstance(current, Mapping) else "real"
            patched.setdefault("model", {}).setdefault("search_space", {})[name] = {
                "type": spec_type,
                "low": exact_value,
                "high": exact_value,
            }
        patch = deepcopy(dict(patch))
        patch["model"] = dict(model_patch)
        patch["model"]["search_space"] = {
            name: value
            for name, value in search_space_patch.items()
            if isinstance(value, Mapping) and "type" in value
        }
        if not patch["model"]["search_space"]:
            patch["model"].pop("search_space")
        if not patch["model"]:
            patch.pop("model")
    return _deep_merge(patched, patch)


def _load_workflow_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return _read_yaml_or_json(Path(str(config["runner"]["config_path"])))


def _campaign_workflow_overrides(config: Mapping[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for section in ("experiment", "dataset", "generation"):
        value = config.get(section)
        if isinstance(value, Mapping):
            overrides[section] = deepcopy(dict(value))
    model = config.get("model")
    if isinstance(model, Mapping):
        overrides["model"] = deepcopy(dict(model))
    return overrides


def _load_campaign_workflow_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Load the base workflow YAML and apply campaign-level fixed overrides."""
    return _deep_merge(_load_workflow_config(config), _campaign_workflow_overrides(config))


def _domain_runner(
    domain: str,
) -> tuple[Callable[[str | Path], dict[str, Any]], Callable[..., dict[str, Any]]]:
    if domain == "molecules":
        from .extensions.demo.zinc_hyperparameter_optimization import (
            load_zinc_hyperparameter_optimization_config,
            run_zinc_hyperparameter_optimization,
        )

        return load_zinc_hyperparameter_optimization_config, run_zinc_hyperparameter_optimization
    if domain == "artificial_graphs":
        from .extensions.demo.artificial_hyperparameter_optimization import (
            load_artificial_hyperparameter_optimization_config,
            run_artificial_hyperparameter_optimization,
        )

        return (
            load_artificial_hyperparameter_optimization_config,
            run_artificial_hyperparameter_optimization,
        )
    raise ValueError(f"Unsupported campaign domain: {domain!r}")


def _notebook_context(config: Mapping[str, Any]) -> dict[str, Path]:
    repo_root = Path(str(config["_repo_root"]))
    return {
        "REPO_ROOT": repo_root,
        "NOTEBOOK_DATA_ROOT": repo_root / "notebooks" / "datasets",
        "ARTIFACT_ROOT": _campaign_artifact_root(config),
    }


def _format_logbook_entry(
    *,
    config: Mapping[str, Any],
    run_dir: Path,
    proposal: Mapping[str, Any],
    state: Mapping[str, Any],
) -> str:
    campaign = config["campaign"]["domain"]
    sampled_count = len(proposal.get("sampled_patches", []))
    metrics = state.get("latest_metrics") or {}
    metric_text = "pending"
    if metrics:
        metric_text = ", ".join(f"{key}={value}" for key, value in sorted(metrics.items()))
    return (
        f"### {campaign} - {run_dir.name}\n\n"
        f"- Tried: {sampled_count} candidate(s) from "
        f"`{proposal.get('proposal_mode', 'range_search')}`.\n"
        f"- Agent reasoning: {proposal.get('reason', 'Config-defined range proposal.')}\n"
        f"- Mutable groups: "
        f"{', '.join(proposal.get('mutable_groups', [])) or 'explicit paths only'}\n"
        f"- Ranges/exact configs: `{run_dir / 'proposal.json'}`\n"
        f"- Metrics: {metric_text}\n"
        f"- Artifacts: `{run_dir}`\n"
        f"- Next attempt: "
        f"{proposal.get('next_attempt', 'Review metrics and narrow the best ranges.')}\n"
    )


def upsert_logbook_block(logbook_path: str | Path, block_id: str, markdown: str) -> None:
    """Create or replace one marked campaign block in a domain logbook."""
    path = Path(logbook_path)
    begin = f"<!-- nodefield-campaign:{block_id}:begin -->"
    end = f"<!-- nodefield-campaign:{block_id}:end -->"
    block = f"{begin}\n{markdown.rstrip()}\n{end}\n"
    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        text = "# NodeField Campaign Logbook\n\n"
    if begin in text and end in text:
        before, rest = text.split(begin, 1)
        _old, after = rest.split(end, 1)
        text = before.rstrip() + "\n\n" + block + after.lstrip()
    else:
        text = text.rstrip() + "\n\n" + block
    path.write_text(text, encoding="utf-8")


def _initial_state(
    config: Mapping[str, Any],
    run_dir: Path,
    proposal: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "campaign": config["campaign"]["domain"],
        "prefix": config["campaign"]["prefix"],
        "status": "queued",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "proposal_mode": proposal.get("proposal_mode", "range_search"),
        "queued_trials": [
            {"trial_id": idx + 1, "status": "queued"}
            for idx, _patch in enumerate(proposal.get("sampled_patches", []))
        ],
        "completed_trials": [],
        "latest_metrics": {},
        "logbook_path": config["logbook"]["path"],
    }


def _proposal_from_config(
    config: Mapping[str, Any],
    *,
    run_dir: Path,
) -> dict[str, Any]:
    agent = config["agent"]
    proposal_mode = str(agent.get("proposal_mode") or "range_search")
    patch_space = None
    flattened = {}
    if proposal_mode == "range_search":
        patch_space = agent["default_trial_patch_space"]
        flattened = validate_patch_space(
            patch_space,
            allowed_paths=list(agent["allowed_paths"]),
            max_leaf_count=int(agent.get("max_search_leaf_count", 8)),
        )
        sampled = sample_patch_space(
            patch_space,
            n_samples=int(config["random_search"]["batch_size"]),
            random_state=int(config["random_search"]["random_state"]),
            allowed_paths=list(agent["allowed_paths"]),
            max_leaf_count=int(agent.get("max_search_leaf_count", 8)),
        )
    elif proposal_mode == "exact_configs":
        exact_configs = [deepcopy(dict(patch)) for patch in agent["default_trial_configs"]]
        _validate_exact_patches(exact_configs, allowed_paths=list(agent["allowed_paths"]))
        sampled = exact_configs[: int(config["random_search"]["batch_size"])]
    else:
        raise ValueError(f"Unsupported proposal mode: {proposal_mode!r}")
    return {
        "campaign": config["campaign"]["domain"],
        "run_dir": str(run_dir),
        "proposal_mode": proposal_mode,
        "mutable_groups": list(agent.get("mutable_groups", [])),
        "allowed_paths": list(agent["allowed_paths"]),
        "reason": str(agent.get("reason") or "Config-defined range proposal."),
        "next_attempt": str(
            agent.get("next_attempt") or "Review metrics and narrow the best ranges."
        ),
        "patch_space": patch_space,
        "flattened_patch_space": flattened,
        "sampled_patches": sampled,
    }


def _prepare_run_tree(run_dir: Path) -> None:
    for subdir in RUN_SUBDIRECTORIES:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)


def _summarize_result(result: Mapping[str, Any]) -> dict[str, Any]:
    best_row = result.get("best_row") or {}
    return {
        "average_num_violations": best_row.get("average_num_violations"),
        "median_num_violations": best_row.get("median_num_violations"),
        "feasible_rate": best_row.get("feasible_rate"),
        "trial_id": best_row.get("trial_id"),
        "results_csv_path": str(result.get("results_csv_path") or ""),
    }


def _write_summary_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_campaign_once(
    config: Mapping[str, Any],
    *,
    dry_run: bool = False,
    now: datetime | None = None,
    short_id: str | None = None,
    stream: TextIO | None = None,
) -> dict[str, Any]:
    """Sample one mini-batch proposal and optionally execute it sequentially."""
    stream = stream or sys.stdout
    run_dir = _make_run_dir(config, now=now, short_id=short_id, dry_run=dry_run)
    proposal = _proposal_from_config(config, run_dir=run_dir)
    state = _initial_state(config, run_dir, proposal)
    if dry_run:
        state["status"] = "dry_run"
        print(f"Dry run: would create {run_dir}", file=stream)
        print(f"Dry run: would queue {len(proposal['sampled_patches'])} trial(s)", file=stream)
        return {"run_dir": run_dir, "proposal": proposal, "state": state, "dry_run": True}

    _prepare_run_tree(run_dir)
    _write_json(run_dir / "proposal.json", proposal)
    _write_json(run_dir / "state.json", state)
    shutil.copyfile(str(config["campaign"]["config_path"]), run_dir / "configs" / "campaign.yaml")
    shutil.copyfile(
        str(config["runner"]["config_path"]),
        run_dir / "configs" / "base_workflow.yaml",
    )

    loader, runner = _domain_runner(str(config["campaign"]["domain"]))
    base_workflow_config = _load_campaign_workflow_config(config)
    metrics_rows: list[dict[str, Any]] = []

    state["status"] = "running"
    for index, patch in enumerate(proposal["sampled_patches"], start=1):
        trial_name = f"trial_{index:03d}"
        trial_dir = run_dir / "trials" / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)
        state["queued_trials"][index - 1]["status"] = "running"
        _write_json(run_dir / "state.json", state)

        workflow_config = apply_exact_trial_patch(base_workflow_config, patch)
        workflow_config.setdefault("experiment", {})["n_trials"] = 1
        workflow_config.setdefault("outputs", {})["run_dir"] = str(trial_dir)
        workflow_config["outputs"]["artifact_root"] = str(_campaign_artifact_root(config))
        workflow_config["outputs"]["artifact_subdir"] = str(config["campaign"]["domain"])
        workflow_config["outputs"]["artifact_prefix"] = str(config["campaign"]["prefix"])
        _write_yaml_or_json(run_dir / "configs" / f"{trial_name}.yaml", workflow_config)
        _write_yaml_or_json(trial_dir / "config.yaml", workflow_config)

        loaded_workflow_config = loader(run_dir / "configs" / f"{trial_name}.yaml")
        print(f"Running {trial_name} in {trial_dir}", file=stream)
        result = runner(loaded_workflow_config, notebook_context=_notebook_context(config))
        trial_metrics = {"campaign_trial_id": index, **_summarize_result(result)}
        metrics_rows.append(trial_metrics)
        _write_json(trial_dir / "metrics.json", trial_metrics)

        state["queued_trials"][index - 1]["status"] = "completed"
        state["completed_trials"].append({"trial_id": index, "trial_dir": str(trial_dir)})
        state["latest_metrics"] = trial_metrics
        _write_json(run_dir / "state.json", state)

    state["status"] = "completed"
    _write_json(run_dir / "state.json", state)
    _write_json(run_dir / "metrics" / "summary.json", {"trials": metrics_rows})
    _write_summary_csv(run_dir / "metrics" / "summary.csv", metrics_rows)
    logbook_entry = _format_logbook_entry(
        config=config,
        run_dir=run_dir,
        proposal=proposal,
        state=state,
    )
    upsert_logbook_block(config["logbook"]["path"], run_dir.name, logbook_entry)
    return {
        "run_dir": run_dir,
        "proposal": proposal,
        "state": state,
        "metrics": metrics_rows,
        "dry_run": False,
    }


def campaign_status(config: Mapping[str, Any], *, log_tail_lines: int = 20) -> dict[str, Any]:
    """Read status from the latest campaign run state file."""
    run_dir = _latest_run_dir(config)
    if run_dir is None:
        return {
            "campaign": config["campaign"]["domain"],
            "status": "not_started",
            "run_dir": None,
            "queued_trials": [],
            "latest_metrics": {},
            "log_tail": "",
        }
    state_path = run_dir / "state.json"
    state = _read_json(state_path) if state_path.is_file() else {}
    log_lines: list[str] = []
    log_root = run_dir / "logs"
    if log_root.is_dir():
        log_files = sorted(
            (path for path in log_root.glob("*.log") if path.is_file()),
            reverse=True,
        )
        if log_files:
            log_lines = log_files[0].read_text(encoding="utf-8", errors="replace").splitlines()
    return {
        "campaign": config["campaign"]["domain"],
        "status": state.get("status", "unknown"),
        "run_dir": str(run_dir),
        "queued_trials": state.get("queued_trials", []),
        "latest_metrics": state.get("latest_metrics", {}),
        "log_tail": "\n".join(log_lines[-int(log_tail_lines) :]),
    }


def format_campaign_status(status: Mapping[str, Any]) -> str:
    """Format campaign status for CLI output."""
    lines = [
        f"campaign: {status.get('campaign')}",
        f"status: {status.get('status')}",
        f"run_dir: {status.get('run_dir') or '-'}",
    ]
    queued = status.get("queued_trials") or []
    if queued:
        queued_text = ", ".join(
            f"{trial.get('trial_id')}:{trial.get('status')}" for trial in queued
        )
    else:
        queued_text = "-"
    lines.append(f"queued_trials: {queued_text}")
    metrics = status.get("latest_metrics") or {}
    lines.append("latest_metrics: " + (json.dumps(metrics, sort_keys=True) if metrics else "-"))
    log_tail = status.get("log_tail")
    if log_tail:
        lines.extend(["log_tail:", str(log_tail)])
    return "\n".join(lines)


def terminate_campaign(config: Mapping[str, Any]) -> dict[str, Any]:
    """Mark the latest campaign run as termination requested."""
    run_dir = _latest_run_dir(config)
    if run_dir is None:
        return {"campaign": config["campaign"]["domain"], "status": "not_started"}
    state_path = run_dir / "state.json"
    state = _read_json(state_path) if state_path.is_file() else {}
    state["status"] = "termination_requested"
    state["termination_requested_at"] = datetime.now().isoformat(timespec="seconds")
    _write_json(state_path, state)
    return {
        "campaign": config["campaign"]["domain"],
        "status": "termination_requested",
        "run_dir": str(run_dir),
    }


__all__ = [
    "CAMPAIGN_CONFIGS",
    "MUTABLE_GROUP_PATHS",
    "apply_exact_trial_patch",
    "campaign_status",
    "format_campaign_status",
    "list_campaigns",
    "load_campaign_config",
    "resolve_campaign_config",
    "run_campaign_once",
    "terminate_campaign",
    "upsert_logbook_block",
]
