"""Filesystem-backed NodeField campaign controller utilities."""

from __future__ import annotations

import csv
import contextlib
from dataclasses import dataclass
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, TextIO
import uuid

from .runtime_paths import (
    make_timestamped_run_dir,
    resolve_campaign_artifact_root,
    resolve_repo_root,
)
from .campaign_search import flatten_leaf_paths, sample_patch_space, validate_patch_space


CAMPAIGN_CONFIGS = {
    "molecules": Path("configs") / "campaigns" / "molecules_small.yaml",
    "molecules-small": Path("configs") / "campaigns" / "molecules_small.yaml",
    "molecules_small": Path("configs") / "campaigns" / "molecules_small.yaml",
    "molecules-large": Path("configs") / "campaigns" / "molecules_large.yaml",
    "molecules_large": Path("configs") / "campaigns" / "molecules_large.yaml",
    "artificial_graphs": Path("configs") / "campaigns" / "artificial_graphs_small.yaml",
    "artificial-graphs": Path("configs") / "campaigns" / "artificial_graphs_small.yaml",
    "artificial-graphs-small": Path("configs") / "campaigns" / "artificial_graphs_small.yaml",
    "artificial_graphs_small": Path("configs") / "campaigns" / "artificial_graphs_small.yaml",
    "artificial-graphs-simple": Path("configs") / "campaigns" / "artificial_graphs_small.yaml",
    "artificial_graphs_simple": Path("configs") / "campaigns" / "artificial_graphs_small.yaml",
    "artificial-graphs-large": Path("configs") / "campaigns" / "artificial_graphs_large.yaml",
    "artificial_graphs_large": Path("configs") / "campaigns" / "artificial_graphs_large.yaml",
    "artificial-graphs-complex": (
        Path("configs") / "campaigns" / "artificial_graphs_large.yaml"
    ),
    "artificial_graphs_complex": (
        Path("configs") / "campaigns" / "artificial_graphs_large.yaml"
    ),
}

RUN_SUBDIRECTORIES = ("configs", "trials", "logs", "metrics", "samples")
MANAGED_EXPERIMENT_VERBOSE = 2

MUTABLE_GROUP_PATHS = {
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
ALLOWED_AGENT_DECISIONS = {
    "no_action",
    "update_logbook",
    "propose_trial",
    "stop_campaign",
    "terminate_run_and_propose_trial",
}
ALLOWED_CAMPAIGN_PATCH_PATHS = {
    "agent.reason",
    "agent.next_attempt",
    "agent.default_trial_patch_space",
    "agent.default_trial_configs",
}
CREDIT_EXHAUSTION_CODES = {
    "billing_hard_limit_reached",
    "billing_not_active",
    "insufficient_quota",
    "quota_exceeded",
}
DEFAULT_AGENT_PROMPTS = {
    "proposal": Path("configs") / "campaigns" / "prompts" / "nodefield_campaign_proposal.md",
    "logbook": Path("configs") / "campaigns" / "prompts" / "nodefield_campaign_logbook.md",
}


@dataclass(frozen=True)
class AgentCampaignDecision:
    decision: str
    reason: str
    logbook_markdown: str
    campaign_patch: dict[str, Any]


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
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(
        json.dumps(_json_safe(data), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


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


def _campaign_patch_path_allowed(path: str) -> bool:
    return any(
        path == allowed or path.startswith(f"{allowed}.")
        for allowed in ALLOWED_CAMPAIGN_PATCH_PATHS
    )


def _validate_campaign_patch(config: Mapping[str, Any], patch: Mapping[str, Any]) -> None:
    if not isinstance(patch, Mapping):
        raise ValueError("campaign_patch must be a mapping.")
    rejected = sorted(
        path for path in flatten_leaf_paths(patch) if not _campaign_patch_path_allowed(path)
    )
    if rejected:
        raise ValueError(
            "campaign_patch contains non-allowlisted path(s): " + ", ".join(rejected)
        )
    patched_agent = _deep_merge(config.get("agent", {}), patch.get("agent", {}))
    proposal_mode = str(patched_agent.get("proposal_mode") or config["agent"]["proposal_mode"])
    allowed_paths = list(config["agent"]["allowed_paths"])
    max_leaf_count = int(config["agent"].get("max_search_leaf_count", len(allowed_paths)))
    if proposal_mode == "range_search" and "default_trial_patch_space" in patched_agent:
        validate_patch_space(
            patched_agent["default_trial_patch_space"],
            allowed_paths=allowed_paths,
            max_leaf_count=max_leaf_count,
        )
    if proposal_mode == "exact_configs" and "default_trial_configs" in patched_agent:
        exact_configs = patched_agent["default_trial_configs"]
        if not isinstance(exact_configs, list) or not exact_configs:
            raise ValueError("agent.default_trial_configs must be a non-empty list.")
        _validate_exact_patches(exact_configs, allowed_paths=allowed_paths)


def apply_campaign_patch(config: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and persist an agent-proposed patch to the tracked campaign YAML."""
    _validate_campaign_patch(config, patch)
    config_path = Path(str(config["campaign"]["config_path"]))
    raw_config = _read_yaml_or_json(config_path)
    patched = _deep_merge(raw_config, patch)
    _write_yaml_or_json(config_path, patched)
    return load_campaign_config(config_path, repo_root=config.get("_repo_root"))


def campaign_decision_text_format() -> dict[str, Any]:
    """Return the strict Responses API schema for one NodeField campaign decision."""
    return {
        "format": {
            "type": "json_schema",
            "name": "nodefield_campaign_decision",
            "description": "A single NodeField campaign orchestration decision.",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["decision", "reason", "logbook_markdown", "campaign_patch"],
                "properties": {
                    "decision": {
                        "type": "string",
                        "enum": sorted(ALLOWED_AGENT_DECISIONS),
                    },
                    "reason": {"type": "string"},
                    "logbook_markdown": {"type": "string"},
                    "campaign_patch": {
                        "type": "string",
                        "description": (
                            "JSON-encoded object patching only agent.reason, "
                            "agent.next_attempt, agent.default_trial_patch_space, or "
                            "agent.default_trial_configs. Use {} when no patch is proposed. "
                            "When an active run is going nowhere, use "
                            "terminate_run_and_propose_trial with a patch for the next run."
                        ),
                    },
                },
            },
        }
    }


def parse_agent_campaign_decision(text: str) -> AgentCampaignDecision:
    payload = json.loads(text.strip())
    if not isinstance(payload, dict):
        raise ValueError("Agent response must be a JSON object.")
    decision = str(payload.get("decision", ""))
    if decision not in ALLOWED_AGENT_DECISIONS:
        raise ValueError(f"Unsupported agent decision: {decision!r}")
    reason = str(payload.get("reason") or "").strip()
    logbook_markdown = str(payload.get("logbook_markdown") or "").strip()
    if not reason and not logbook_markdown:
        raise ValueError("Agent decision must include reason or logbook_markdown.")
    campaign_patch = payload.get("campaign_patch")
    if isinstance(campaign_patch, str):
        campaign_patch = json.loads(campaign_patch) if campaign_patch.strip() else {}
    if campaign_patch is None:
        campaign_patch = {}
    if not isinstance(campaign_patch, dict):
        raise ValueError("campaign_patch must decode to a mapping.")
    return AgentCampaignDecision(
        decision=decision,
        reason=reason,
        logbook_markdown=logbook_markdown,
        campaign_patch=campaign_patch,
    )


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
    random_search["batch_size"] = int(random_search.get("batch_size", 1))
    if random_search["batch_size"] < 1:
        raise ValueError("random_search.batch_size must be >= 1.")
    random_search["random_state"] = int(random_search.get("random_state", 0))

    runner = config.setdefault("runner", {})
    if not isinstance(runner, dict):
        raise ValueError("runner section must be a mapping.")
    if "config_path" not in runner:
        raise ValueError("runner.config_path must be provided.")
    runner["config_path"] = str(_resolve_repo_path(root, runner["config_path"]))
    runner["poll_seconds"] = int(runner.get("poll_seconds", 1800))
    if runner["poll_seconds"] < 1:
        raise ValueError("runner.poll_seconds must be >= 1.")

    agent = config.setdefault("agent", {})
    if not isinstance(agent, dict):
        raise ValueError("agent section must be a mapping.")
    agent.setdefault("model", "gpt-5.3-codex")
    agent.setdefault("reasoning_effort", "medium")
    agent["max_output_tokens"] = int(agent.get("max_output_tokens", 2000))
    agent.setdefault("api_key_env", "OPENAI_API_KEY")
    prompts = agent.setdefault("prompts", {})
    if prompts is None:
        prompts = {}
    if not isinstance(prompts, dict):
        raise ValueError("agent.prompts must be a mapping when provided.")
    resolved_prompts: dict[str, str] = {}
    for prompt_name, default_path in DEFAULT_AGENT_PROMPTS.items():
        prompt_path = prompts.get(prompt_name, default_path)
        resolved_path = _resolve_repo_path(root, prompt_path)
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Agent prompt file does not exist: {resolved_path}")
        resolved_prompts[prompt_name] = str(resolved_path)
    for prompt_name, prompt_path in prompts.items():
        if prompt_name in resolved_prompts:
            continue
        resolved_path = _resolve_repo_path(root, prompt_path)
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Agent prompt file does not exist: {resolved_path}")
        resolved_prompts[str(prompt_name)] = str(resolved_path)
    agent["prompts"] = resolved_prompts
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
    for name in (
        "molecules-small",
        "molecules-large",
        "artificial-graphs-small",
        "artificial-graphs-large",
    ):
        path = _resolve_repo_path(root, CAMPAIGN_CONFIGS[name])
        rows.append({"campaign": name, "config_path": str(path), "exists": path.is_file()})
    return rows


def _campaign_artifact_root(config: Mapping[str, Any]) -> Path:
    repo_root = Path(str(config["_repo_root"]))
    root = config["artifacts"].get("root")
    return resolve_campaign_artifact_root(root, repo_root=repo_root)


def _campaign_domain_root(config: Mapping[str, Any]) -> Path:
    return _campaign_artifact_root(config) / str(config["campaign"]["domain"])


def _campaign_state_path(config: Mapping[str, Any]) -> Path:
    return _campaign_domain_root(config) / f"{config['campaign']['prefix']}_campaign_state.json"


def _campaign_lock_path(config: Mapping[str, Any]) -> Path:
    return _campaign_domain_root(config) / f"{config['campaign']['prefix']}.lock"


def _agent_decisions_path(config: Mapping[str, Any]) -> Path:
    return _campaign_domain_root(config) / f"{config['campaign']['prefix']}_agent_decisions.jsonl"


def _now_iso(now: datetime | None = None) -> str:
    return (now or datetime.now()).isoformat(timespec="seconds")


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"state_error": f"Could not parse {path}"}
    return data if isinstance(data, dict) else {}


def _is_process_running(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_process_group(pid: int | None) -> None:
    if pid is None or pid <= 0:
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        os.kill(pid, signal.SIGTERM)


def _read_lock_pid(path: Path) -> int | None:
    payload = _read_json_if_exists(path)
    try:
        return int(payload.get("pid"))
    except (TypeError, ValueError, AttributeError):
        return None


def _acquire_campaign_lock(config: Mapping[str, Any], *, force: bool = False) -> Path:
    path = _campaign_lock_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    for attempt in range(2):
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            existing_pid = _read_lock_pid(path)
            if force:
                if existing_pid is not None and existing_pid != pid and _is_process_running(existing_pid):
                    try:
                        os.kill(existing_pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                path.unlink(missing_ok=True)
                continue
            if attempt == 0 and existing_pid is not None and not _is_process_running(existing_pid):
                path.unlink(missing_ok=True)
                continue
            raise RuntimeError(f"Campaign lock already exists at {path}; pid={existing_pid}")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": pid, "created_at": _now_iso()}, indent=2))
        return path
    raise RuntimeError(f"Could not acquire campaign lock at {path}")


def _release_campaign_lock(path: Path) -> None:
    if _read_lock_pid(path) == os.getpid():
        path.unlink(missing_ok=True)


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")


def _response_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text
    output = getattr(response, "output", None)
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for content_item in content:
                    text = getattr(content_item, "text", None)
                    if isinstance(text, str):
                        parts.append(text)
        if parts:
            return "\n".join(parts)
    raise ValueError("OpenAI response did not contain output_text.")


def _make_run_dir(
    config: Mapping[str, Any],
    *,
    now: datetime | None = None,
    short_id: str | None = None,
    dry_run: bool = False,
    allow_existing: bool = False,
) -> Path:
    run_dir = make_timestamped_run_dir(
        _campaign_domain_root(config),
        str(config["campaign"]["prefix"]),
        now=now,
        short_id=short_id,
        create=False,
    )
    if not dry_run:
        if allow_existing and ((run_dir / "state.json").exists() or (run_dir / "proposal.json").exists()):
            raise FileExistsError(f"Run directory already contains campaign state: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=allow_existing)
    return run_dir


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


def _latest_campaign_context(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return compact context from the latest completed/failed run before proposing."""
    run_dir = _latest_run_dir(config)
    if run_dir is None:
        return {
            "status": "not_started",
            "run_dir": None,
            "latest_metrics": {},
            "latest_error": {},
            "summary_csv_path": None,
            "proposal_path": None,
            "logbook_path": config["logbook"]["path"],
            "logbook_tail": "",
        }
    state_path = run_dir / "state.json"
    state = _read_json(state_path) if state_path.is_file() else {}
    summary_csv_path = run_dir / "metrics" / "summary.csv"
    logbook_path = Path(str(config["logbook"]["path"]))
    logbook_tail = ""
    if logbook_path.is_file():
        logbook_tail = logbook_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    return {
        "status": state.get("status", "unknown"),
        "run_dir": str(run_dir),
        "latest_metrics": state.get("latest_metrics", {}),
        "latest_error": state.get("latest_error", {}),
        "summary_csv_path": str(summary_csv_path) if summary_csv_path.is_file() else None,
        "proposal_path": str(run_dir / "proposal.json") if (run_dir / "proposal.json").is_file() else None,
        "logbook_path": str(logbook_path),
        "logbook_tail": logbook_tail,
    }


def _campaign_result_summary(config: Mapping[str, Any], run_dir: Path | None = None) -> dict[str, Any]:
    run_dir = run_dir or _latest_run_dir(config)
    if run_dir is None:
        return {
            "status": "not_started",
            "run_dir": None,
            "state": {},
            "proposal": {},
            "summary": {},
            "log_tail": "",
        }
    state = _read_json_if_exists(run_dir / "state.json")
    proposal = _read_json_if_exists(run_dir / "proposal.json")
    summary = _read_json_if_exists(run_dir / "metrics" / "summary.json")
    log_tail = ""
    log_files = sorted((run_dir / "logs").glob("*.log")) if (run_dir / "logs").is_dir() else []
    if log_files:
        lines = log_files[-1].read_text(encoding="utf-8", errors="replace").splitlines()
        log_tail = "\n".join(lines[-80:])
    return {
        "status": state.get("status", "unknown"),
        "run_dir": str(run_dir),
        "state": state,
        "proposal": proposal,
        "summary": summary,
        "log_tail": log_tail,
        "loss_pdf_paths": state.get("loss_pdf_paths", []),
        "artifact_links_markdown": _format_logbook_artifact_links(config, run_dir, state),
    }


def _agent_prompt_text(config: Mapping[str, Any], campaign_state: Mapping[str, Any], result: Mapping[str, Any]) -> str:
    proposal_prompt = Path(str(config["agent"]["prompts"]["proposal"])).read_text(
        encoding="utf-8"
    )
    logbook_prompt = Path(str(config["agent"]["prompts"]["logbook"])).read_text(
        encoding="utf-8"
    )
    logbook_path = Path(str(config["logbook"]["path"]))
    logbook_tail = ""
    if logbook_path.is_file():
        logbook_tail = logbook_path.read_text(encoding="utf-8", errors="replace")[-6000:]
    visible_config = {
        "campaign": config.get("campaign", {}),
        "random_search": config.get("random_search", {}),
        "dataset": config.get("dataset", {}),
        "generation": config.get("generation", {}),
        "agent": {
            key: value
            for key, value in config.get("agent", {}).items()
            if key not in {"prompts"}
        },
    }
    return "\n\n".join(
        [
            "You are managing a NodeField graph-generation hyperparameter campaign.",
            "The deterministic controller handles launching, polling, and validation.",
            (
                "If the campaign state shows an active running mini-batch, you may return "
                "`no_action` to keep waiting, `update_logbook` to record analysis only, "
                "`terminate_run_and_propose_trial` to stop an unpromising active child "
                "and launch a patched next run, or `stop_campaign` to stop the campaign."
            ),
            (
                "Use semantic early stopping only when partial metrics, logs, or repeated "
                "failure patterns make the active run clearly uninformative; otherwise "
                "continue and wait for more evidence."
            ),
            "Return only the strict JSON object requested by the response schema.",
            "Proposal prompt:",
            proposal_prompt,
            "Logbook prompt:",
            logbook_prompt,
            "Allowed campaign_patch paths:",
            json.dumps(sorted(ALLOWED_CAMPAIGN_PATCH_PATHS), indent=2),
            "Current campaign config JSON:",
            json.dumps(_json_safe(visible_config), indent=2, sort_keys=True),
            "Campaign controller state JSON:",
            json.dumps(_json_safe(campaign_state), indent=2, sort_keys=True),
            "Latest mini-batch result JSON:",
            json.dumps(_json_safe(result), indent=2, sort_keys=True),
            "Domain logbook tail:",
            logbook_tail,
        ]
    )


def request_agent_campaign_decision(
    config: Mapping[str, Any],
    campaign_state: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    client: Any | None = None,
) -> AgentCampaignDecision:
    if client is None:
        from openai import OpenAI

        api_key_env = str(config["agent"].get("api_key_env", "OPENAI_API_KEY"))
        api_key = os.environ.get(api_key_env)
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
    response = client.responses.create(
        model=config["agent"].get("model", "gpt-5.3-codex"),
        reasoning={"effort": config["agent"].get("reasoning_effort", "medium")},
        max_output_tokens=int(config["agent"].get("max_output_tokens", 2000)),
        text=campaign_decision_text_format(),
        input=_agent_prompt_text(config, campaign_state, result),
    )
    return parse_agent_campaign_decision(_response_output_text(response))


def _is_openai_credits_exhausted(exc: Exception) -> bool:
    structured_values: list[str] = []
    for attr in ("code", "type"):
        value = getattr(exc, attr, None)
        if value:
            structured_values.append(str(value).lower())
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            for key in ("code", "type", "message"):
                value = error.get(key)
                if value:
                    structured_values.append(str(value).lower())
    if any(value in CREDIT_EXHAUSTION_CODES for value in structured_values):
        return True
    message = f"{type(exc).__name__}: {exc}".lower()
    markers = [
        "insufficient_quota",
        "exceeded your current quota",
        "billing hard limit",
        "billing_not_active",
        "quota_exceeded",
        "credits exhausted",
        "credit balance",
    ]
    return any(marker in message for marker in markers)


def _format_metric_summary(metrics: Mapping[str, Any]) -> str:
    if not metrics:
        return "no metrics"
    preferred = [
        "average_num_violations",
        "median_num_violations",
        "feasible_rate",
        "campaign_trial_id",
        "trial_id",
    ]
    parts = []
    for key in preferred:
        if key in metrics:
            parts.append(f"{key}={metrics[key]}")
    for key in sorted(metrics):
        if key not in preferred and len(parts) < 6:
            parts.append(f"{key}={metrics[key]}")
    return ", ".join(parts)


def _proposal_reason(agent: Mapping[str, Any], previous_result: Mapping[str, Any]) -> str:
    base_reason = str(agent.get("reason") or "Config-defined range proposal.")
    status = str(previous_result.get("status") or "not_started")
    run_dir = previous_result.get("run_dir")
    if status == "not_started" or not run_dir:
        return f"No prior campaign result found; starting from configured ranges. {base_reason}"
    error = previous_result.get("latest_error") or {}
    if error:
        message = error.get("message") or "unknown error"
        return (
            f"Latest run {run_dir} ended with {error.get('type', 'error')}: {message}. "
            f"Retrying from the configured ranges while preserving the fixed campaign conditions. "
            f"{base_reason}"
        )
    metrics = previous_result.get("latest_metrics") or {}
    return (
        f"Latest run {run_dir} finished with {_format_metric_summary(metrics)}. "
        f"Proposing the next mini-batch from the configured mutable ranges and exact patches. "
        f"{base_reason}"
    )


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


def _logbook_link(config: Mapping[str, Any], path: str | Path, label: str | None = None) -> str:
    repo_root = Path(str(config["_repo_root"]))
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (repo_root / resolved).resolve()
    else:
        resolved = resolved.resolve()
    try:
        target = resolved.relative_to(repo_root)
    except ValueError:
        target = resolved
    text = label or resolved.name
    return f"[{text}]({target.as_posix()})"


def _existing_logbook_links(
    config: Mapping[str, Any],
    entries: list[tuple[str, str | Path]],
) -> list[str]:
    links = []
    for label, path in entries:
        resolved = Path(path).expanduser()
        if not resolved.is_absolute():
            resolved = Path(str(config["_repo_root"])) / resolved
        if resolved.exists():
            links.append(f"{label}: {_logbook_link(config, resolved)}")
    return links


def _format_logbook_artifact_links(
    config: Mapping[str, Any],
    run_dir: str | Path,
    state: Mapping[str, Any] | None = None,
) -> str:
    """Return deterministic Markdown links to the files worth inspecting for a run."""
    run_path = Path(run_dir).expanduser().resolve()
    state = state or _read_json_if_exists(run_path / "state.json")
    primary_links = _existing_logbook_links(
        config,
        [
            ("run", run_path),
            ("state", run_path / "state.json"),
            ("proposal", run_path / "proposal.json"),
            ("campaign result", run_path / "campaign_result.json"),
            ("agent decision", run_path / "agent_decision.json"),
            ("metrics csv", run_path / "metrics" / "summary.csv"),
            ("metrics json", run_path / "metrics" / "summary.json"),
            ("campaign config", run_path / "configs" / "campaign.yaml"),
            ("base workflow", run_path / "configs" / "base_workflow.yaml"),
            ("mini-batch log", run_path / "logs" / "mini_batch.log"),
        ],
    )
    trial_links: list[str] = []
    for trial_state in state.get("queued_trials", []) or []:
        if not isinstance(trial_state, Mapping):
            continue
        trial_id = trial_state.get("trial_id")
        trial_name = f"trial_{int(trial_id):03d}" if trial_id is not None else "trial"
        log_path = trial_state.get("log_path")
        if log_path:
            trial_links.extend(_existing_logbook_links(config, [(f"{trial_name} log", log_path)]))
        loss_pdf_path = trial_state.get("loss_pdf_path")
        if loss_pdf_path:
            trial_links.extend(
                _existing_logbook_links(config, [(f"{trial_name} loss PDF", loss_pdf_path)])
            )
        trial_dir = run_path / "trials" / trial_name
        trial_links.extend(
            _existing_logbook_links(
                config,
                [
                    (f"{trial_name} config", trial_dir / "config.yaml"),
                    (f"{trial_name} metrics", trial_dir / "metrics.json"),
                    (f"{trial_name} results CSV", trial_dir / "metrics" / "trial_results.csv"),
                    (f"{trial_name} loss PDF", trial_dir / "metrics" / "loss_curves.pdf"),
                ],
            )
        )
    for pdf_path in state.get("loss_pdf_paths", []) or []:
        trial_links.extend(_existing_logbook_links(config, [("loss PDF", pdf_path)]))
    seen = set()
    unique_lines = []
    for line in [*primary_links, *trial_links]:
        if line in seen:
            continue
        seen.add(line)
        unique_lines.append(f"- {line}")
    if not unique_lines:
        return ""
    return "#### Files to inspect\n\n" + "\n".join(unique_lines)


def _format_logbook_metrics_table(state: Mapping[str, Any]) -> str:
    rows: list[Mapping[str, Any]] = []
    latest_metrics = state.get("latest_metrics")
    completed_by_id = {
        int(trial.get("trial_id")): trial
        for trial in state.get("completed_trials", []) or []
        if isinstance(trial, Mapping) and trial.get("trial_id") is not None
    }
    for trial in state.get("queued_trials", []) or []:
        if not isinstance(trial, Mapping):
            continue
        trial_id = trial.get("trial_id")
        if trial_id is None:
            continue
        completed_trial_dir = completed_by_id.get(int(trial_id), {}).get("trial_dir")
        metrics_path = Path(str(completed_trial_dir)) / "metrics.json" if completed_trial_dir else None
        metrics = _read_json_if_exists(metrics_path) if metrics_path and metrics_path.is_file() else {}
        if not metrics and isinstance(latest_metrics, Mapping):
            latest_trial_id = latest_metrics.get("campaign_trial_id") or latest_metrics.get("trial_id")
            if latest_trial_id == trial_id:
                metrics = latest_metrics
        rows.append(
            {
                "trial": f"trial_{int(trial_id):03d}",
                "status": trial.get("status", ""),
                "average": metrics.get("average_num_violations", ""),
                "median": metrics.get("median_num_violations", ""),
                "feasible_rate": metrics.get("feasible_rate", ""),
            }
        )
    if not rows:
        return ""
    lines = [
        "| Trial | Status | Avg violations | Median violations | Feasible rate |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {trial} | {status} | {average} | {median} | {feasible_rate} |".format(**row)
        )
    return "\n".join(lines)


def _best_metric_row(state: Mapping[str, Any]) -> Mapping[str, Any]:
    rows: list[Mapping[str, Any]] = []
    for trial in state.get("completed_trials", []) or []:
        if not isinstance(trial, Mapping):
            continue
        trial_dir = trial.get("trial_dir")
        if not trial_dir:
            continue
        metrics = _read_json_if_exists(Path(str(trial_dir)) / "metrics.json")
        if metrics:
            rows.append(metrics)
    latest_metrics = state.get("latest_metrics")
    if isinstance(latest_metrics, Mapping) and latest_metrics:
        rows.append(latest_metrics)
    numeric_rows = [
        row
        for row in rows
        if row.get("average_num_violations") is not None
    ]
    if not numeric_rows:
        return latest_metrics if isinstance(latest_metrics, Mapping) else {}
    return sorted(
        numeric_rows,
        key=lambda row: (
            float(row.get("average_num_violations", float("inf"))),
            -float(row.get("feasible_rate", 0.0) or 0.0),
        ),
    )[0]


def _format_logbook_entry(
    *,
    config: Mapping[str, Any],
    run_dir: Path,
    proposal: Mapping[str, Any],
    state: Mapping[str, Any],
) -> str:
    campaign = config["campaign"]["domain"]
    sampled_count = len(proposal.get("sampled_patches", []))
    status = str(state.get("status", "unknown"))
    best_metrics = _best_metric_row(state)
    best_trial = best_metrics.get("campaign_trial_id") or best_metrics.get("trial_id")
    metrics_table = _format_logbook_metrics_table(state)
    if best_metrics:
        best_trial_name = f"trial_{int(best_trial):03d}" if best_trial is not None else "the best trial"
        conclusion = (
            f"This run {status} after testing {sampled_count} candidate(s). "
            f"The best candidate was {best_trial_name} with "
            f"average violations {best_metrics.get('average_num_violations')}, "
            f"median violations {best_metrics.get('median_num_violations')}, and "
            f"feasible rate {best_metrics.get('feasible_rate')}; use the table below "
            "to compare the sampled candidates."
        )
    else:
        conclusion = (
            f"This run {status} after queueing {sampled_count} candidate(s), but no "
            "completed metric row was available yet. Check the logs and state file "
            "linked below before changing ranges."
        )
    next_attempt = str(
        proposal.get("next_attempt") or "Review metrics and narrow the best ranges."
    )
    reasoning = str(proposal.get("reason") or "Config-defined range proposal.")
    return (
        f"### {campaign} - {run_dir.name}\n\n"
        f"{conclusion}\n\n"
        f"{reasoning}\n\n"
        f"Next, {next_attempt}\n\n"
        f"#### Metrics\n\n"
        f"{metrics_table or '_No completed trial metrics yet._'}\n\n"
        f"{_format_logbook_artifact_links(config, run_dir, state)}\n"
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
        "poll_seconds": int((config.get("runner") or {}).get("poll_seconds", 1800)),
        "logs_dir": str(run_dir / "logs"),
        "loss_pdf_paths": [],
        "queued_trials": [
            {
                "trial_id": idx + 1,
                "status": "queued",
                "log_path": str(run_dir / "logs" / f"trial_{idx + 1:03d}.log"),
            }
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
    previous_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    agent = config["agent"]
    previous_result = dict(previous_result or {})
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
        "prompt_paths": dict(agent.get("prompts", {})),
        "previous_result": previous_result,
        "reason": _proposal_reason(agent, previous_result),
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


def _collect_loss_pdf_paths(run_dir: Path) -> list[str]:
    pdfs = sorted(
        path
        for path in run_dir.rglob("*.pdf")
        if "loss" in path.name.lower() or "metric" in path.name.lower()
    )
    return [str(path) for path in pdfs]


def _export_trial_loss_pdf(result: Mapping[str, Any], trial_dir: Path) -> str | None:
    graph_generator = result.get("best_graph_generator")
    exporter = getattr(graph_generator, "export_metrics_pdf", None)
    if not callable(exporter):
        return None
    pdf_path = trial_dir / "metrics" / "loss_curves.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    exporter(str(pdf_path))
    return str(pdf_path)


def _save_result_generator_snapshot(result: Mapping[str, Any], trial_dir: Path) -> str | None:
    graph_generator = result.get("best_graph_generator")
    if graph_generator is None:
        return None
    from .extensions.demo.trial_snapshots import save_trial_graph_generator_snapshot

    return save_trial_graph_generator_snapshot(graph_generator, trial_dir)


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
    allow_existing_run_dir: bool = False,
) -> dict[str, Any]:
    """Sample one mini-batch proposal and optionally execute it sequentially."""
    stream = stream or sys.stdout
    loader = None
    runner = None
    base_workflow_config = None
    if not dry_run:
        loader, runner = _domain_runner(str(config["campaign"]["domain"]))
        base_workflow_config = _load_campaign_workflow_config(config)

    previous_result = _latest_campaign_context(config)
    run_dir = _make_run_dir(
        config,
        now=now,
        short_id=short_id,
        dry_run=dry_run,
        allow_existing=allow_existing_run_dir,
    )
    proposal = _proposal_from_config(
        config,
        run_dir=run_dir,
        previous_result=previous_result,
    )
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

    metrics_rows: list[dict[str, Any]] = []

    state["status"] = "running"
    print(f"campaign: {config['campaign']['id']}", file=stream)
    print(f"run_dir: {run_dir}", file=stream)
    print(f"proposal: {proposal['reason']}", file=stream)
    print(f"logs: {run_dir / 'logs'}", file=stream)
    for index, patch in enumerate(proposal["sampled_patches"], start=1):
        trial_name = f"trial_{index:03d}"
        trial_dir = run_dir / "trials" / trial_name
        trial_log_path = run_dir / "logs" / f"{trial_name}.log"
        trial_dir.mkdir(parents=True, exist_ok=True)
        state["queued_trials"][index - 1]["status"] = "running"
        state["queued_trials"][index - 1]["log_path"] = str(trial_log_path)
        _write_json(run_dir / "state.json", state)

        workflow_config = apply_exact_trial_patch(base_workflow_config, patch)
        experiment_config = workflow_config.setdefault("experiment", {})
        experiment_config["n_trials"] = 1
        experiment_config["verbose"] = MANAGED_EXPERIMENT_VERBOSE
        workflow_config.setdefault("outputs", {})["run_dir"] = str(trial_dir)
        workflow_config["outputs"]["artifact_root"] = str(_campaign_artifact_root(config))
        workflow_config["outputs"]["artifact_subdir"] = str(config["campaign"]["domain"])
        workflow_config["outputs"]["artifact_prefix"] = str(config["campaign"]["prefix"])
        _write_yaml_or_json(run_dir / "configs" / f"{trial_name}.yaml", workflow_config)
        _write_yaml_or_json(trial_dir / "config.yaml", workflow_config)

        loaded_workflow_config = loader(run_dir / "configs" / f"{trial_name}.yaml")
        print(f"{trial_name}: running; log={trial_log_path}", file=stream)
        try:
            with trial_log_path.open("w", encoding="utf-8") as log_handle:
                print(f"Running {trial_name} in {trial_dir}", file=log_handle)
                print(f"Config: {trial_dir / 'config.yaml'}", file=log_handle)
                with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
                    result = runner(loaded_workflow_config, notebook_context=_notebook_context(config))
                    loss_pdf_path = _export_trial_loss_pdf(result, trial_dir)
                    generator_snapshot_path = _save_result_generator_snapshot(result, trial_dir)
        except Exception as exc:
            state["status"] = "failed"
            state["queued_trials"][index - 1]["status"] = "failed"
            state["queued_trials"][index - 1]["log_path"] = str(trial_log_path)
            state["latest_error"] = {
                "trial_id": index,
                "type": type(exc).__name__,
                "message": str(exc),
                "log_path": str(trial_log_path),
            }
            _write_json(run_dir / "state.json", state)
            _write_json(
                run_dir / "campaign_result.json",
                {
                    "run_dir": str(run_dir),
                    "status": "failed",
                    "state": state,
                    "proposal": proposal,
                    "metrics": metrics_rows,
                },
            )
            with trial_log_path.open("a", encoding="utf-8") as log_handle:
                print("\nTrial failed with exception:", file=log_handle)
                traceback.print_exc(file=log_handle)
            print(f"{trial_name}: failed; log={trial_log_path}", file=stream)
            raise
        trial_metrics = {"campaign_trial_id": index, **_summarize_result(result)}
        if loss_pdf_path:
            trial_metrics["loss_pdf_path"] = loss_pdf_path
        if generator_snapshot_path:
            trial_metrics["generator_snapshot_path"] = generator_snapshot_path
        metrics_rows.append(trial_metrics)
        _write_json(trial_dir / "metrics.json", trial_metrics)

        state["queued_trials"][index - 1]["status"] = "completed"
        if loss_pdf_path:
            state["queued_trials"][index - 1]["loss_pdf_path"] = loss_pdf_path
        state["completed_trials"].append(
            {
                "trial_id": index,
                "trial_dir": str(trial_dir),
                "log_path": str(trial_log_path),
                "loss_pdf_path": loss_pdf_path,
                "generator_snapshot_path": generator_snapshot_path,
            }
        )
        state["latest_metrics"] = trial_metrics
        state["loss_pdf_paths"] = _collect_loss_pdf_paths(run_dir)
        _write_json(run_dir / "state.json", state)
        metric_summary = _format_metric_summary(trial_metrics)
        pdf_note = f"; loss_pdf={loss_pdf_path}" if loss_pdf_path else ""
        print(f"{trial_name}: completed; {metric_summary}{pdf_note}", file=stream)

    state["status"] = "completed"
    state["loss_pdf_paths"] = _collect_loss_pdf_paths(run_dir)
    _write_json(run_dir / "state.json", state)
    _write_json(run_dir / "metrics" / "summary.json", {"trials": metrics_rows})
    _write_summary_csv(run_dir / "metrics" / "summary.csv", metrics_rows)
    _write_json(
        run_dir / "campaign_result.json",
        {
            "run_dir": str(run_dir),
            "status": state["status"],
            "state": state,
            "proposal": proposal,
            "metrics": metrics_rows,
        },
    )
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


def _child_log_path(config: Mapping[str, Any], run_dir: Path) -> Path:
    return run_dir / "logs" / "mini_batch.log"


def _parse_run_timestamp(value: str) -> datetime:
    return datetime.strptime(value, "%Y%m%d_%H%M%S")


def _next_run_identity() -> tuple[datetime, str]:
    return datetime.now(), uuid.uuid4().hex[:6]


def _launch_mini_batch_child(
    config: Mapping[str, Any],
    *,
    campaign_name: str,
    device: str = "cpu",
    run_timestamp: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    now, generated_id = _next_run_identity()
    if run_timestamp is not None:
        now = _parse_run_timestamp(run_timestamp)
    run_id = run_id or generated_id
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    run_dir = _make_run_dir(config, now=now, short_id=run_id, dry_run=True)
    log_path = _child_log_path(config, run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    repo_root = Path(str(config["_repo_root"]))
    command = [
        sys.executable,
        str(repo_root / "run_nodefield_campaign.py"),
        "run-mini-batch",
        campaign_name,
        "--config",
        str(config["campaign"]["config_path"]),
        "--run-timestamp",
        timestamp,
        "--run-id",
        run_id,
        "--device",
        device,
    ]
    env = dict(os.environ)
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif device == "cuda":
        env.pop("CUDA_VISIBLE_DEVICES", None)
    log_handle = log_path.open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            cwd=str(repo_root),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_handle.close()
    state = {
        "campaign": config["campaign"]["domain"],
        "campaign_id": config["campaign"]["id"],
        "prefix": config["campaign"]["prefix"],
        "status": "running",
        "phase": "mini_batch",
        "pid": process.pid,
        "process_running": True,
        "run_dir": str(run_dir),
        "child_log_path": str(log_path),
        "command": command,
        "poll_seconds": int(config["runner"]["poll_seconds"]),
        "started_at": _now_iso(),
        "updated_at": _now_iso(),
        "config_path": str(config["campaign"]["config_path"]),
        "latest_decision": {},
        "latest_error": {},
    }
    _write_json(_campaign_state_path(config), state)
    return state


def _decision_to_json(decision: AgentCampaignDecision) -> dict[str, Any]:
    return {
        "decision": decision.decision,
        "reason": decision.reason,
        "logbook_markdown": decision.logbook_markdown,
        "campaign_patch": decision.campaign_patch,
    }


def _write_agent_decision(
    config: Mapping[str, Any],
    run_dir: Path,
    campaign_state: Mapping[str, Any],
    decision: AgentCampaignDecision,
) -> dict[str, Any]:
    row = {
        **_decision_to_json(decision),
        "campaign": config["campaign"]["id"],
        "run_dir": str(run_dir),
        "created_at": _now_iso(),
        "previous_status": campaign_state.get("status"),
    }
    _write_json(run_dir / "agent_decision.json", row)
    _append_jsonl(_agent_decisions_path(config), row)
    if decision.logbook_markdown:
        artifact_links = _format_logbook_artifact_links(config, run_dir, campaign_state)
        logbook_markdown = decision.logbook_markdown
        if artifact_links:
            logbook_markdown = logbook_markdown.rstrip() + "\n\n" + artifact_links
        upsert_logbook_block(
            config["logbook"]["path"],
            f"{run_dir.name}:agent",
            logbook_markdown,
        )
    return row


def _campaign_state_for_decision(
    config: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    status: str,
    phase: str,
    process_running: bool = False,
    decision_row: Mapping[str, Any] | None = None,
    error: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    next_state = dict(state)
    next_state.update(
        {
            "campaign": config["campaign"]["domain"],
            "campaign_id": config["campaign"]["id"],
            "prefix": config["campaign"]["prefix"],
            "status": status,
            "phase": phase,
            "process_running": process_running,
            "poll_seconds": int(config["runner"]["poll_seconds"]),
            "updated_at": _now_iso(),
            "config_path": str(config["campaign"]["config_path"]),
        }
    )
    if decision_row is not None:
        next_state["latest_decision"] = dict(decision_row)
    if error is not None:
        next_state["latest_error"] = dict(error)
    _write_json(_campaign_state_path(config), next_state)
    return next_state


def _mark_run_terminated_by_agent(
    run_dir: Path,
    *,
    decision_row: Mapping[str, Any],
) -> dict[str, Any]:
    run_state = _read_json_if_exists(run_dir / "state.json")
    run_state.update(
        {
            "status": "terminated_by_agent",
            "terminated_at": _now_iso(),
            "termination_reason": decision_row.get("reason", ""),
            "latest_decision": dict(decision_row),
        }
    )
    _write_json(run_dir / "state.json", run_state)
    _write_json(
        run_dir / "campaign_result.json",
        {
            "run_dir": str(run_dir),
            "status": "terminated_by_agent",
            "state": run_state,
            "proposal": _read_json_if_exists(run_dir / "proposal.json"),
            "metrics": _read_json_if_exists(run_dir / "metrics" / "summary.json").get(
                "trials", []
            ),
        },
    )
    return run_state


def _parse_state_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _seconds_until_next_poll(config: Mapping[str, Any], state: Mapping[str, Any]) -> int:
    poll_seconds = int(config["runner"]["poll_seconds"])
    last_poll = (
        _parse_state_time(state.get("updated_at"))
        or _parse_state_time(state.get("started_at"))
    )
    if last_poll is None:
        return 0
    due_at = last_poll + timedelta(seconds=poll_seconds)
    return max(0, int((due_at - datetime.now()).total_seconds()))


def _stream_new_log_lines(
    log_path: str | Path | None,
    *,
    offsets: dict[str, int],
    stream: TextIO,
    max_lines: int = 40,
) -> None:
    if not log_path:
        return
    path = Path(str(log_path))
    if not path.is_file():
        return
    key = str(path)
    size = path.stat().st_size
    offset = min(offsets.get(key, 0), size)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(offset)
        chunk = handle.read()
        offsets[key] = handle.tell()
    lines = chunk.splitlines()
    if not lines:
        return
    if len(lines) > max_lines:
        skipped = len(lines) - max_lines
        lines = [f"... skipped {skipped} older log line(s) ...", *lines[-max_lines:]]
    for line in lines:
        print(f"[child] {line}", file=stream)


def _wait_with_foreground_updates(
    config: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    seconds: int,
    sleep_fn: Callable[[float], None],
    stream: TextIO,
    log_offsets: dict[str, int],
) -> None:
    if seconds <= 0:
        return
    log_path = state.get("child_log_path")
    if log_path:
        print("monitoring child log; press Ctrl-C to terminate the active run", file=stream)
    if sleep_fn is not time.sleep:
        _stream_new_log_lines(log_path, offsets=log_offsets, stream=stream)
        sleep_fn(seconds)
        return
    deadline = time.monotonic() + seconds
    interval = max(1, min(10, seconds))
    while True:
        _stream_new_log_lines(log_path, offsets=log_offsets, stream=stream)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        sleep_fn(min(interval, remaining))


def _terminate_campaign_state_for_interrupt(
    config: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    stream: TextIO,
) -> dict[str, Any]:
    next_state = dict(state)
    pid = next_state.get("pid")
    try:
        pid_int = int(pid) if pid is not None else None
    except (TypeError, ValueError):
        pid_int = None
    if pid_int and _is_process_running(pid_int):
        _terminate_process_group(pid_int)
    next_state.update(
        {
            "status": "terminated",
            "phase": "interrupted",
            "process_running": False,
            "interrupted_at": _now_iso(),
            "updated_at": _now_iso(),
        }
    )
    run_dir_text = next_state.get("run_dir")
    if run_dir_text:
        run_dir = Path(str(run_dir_text))
        run_state = _read_json_if_exists(run_dir / "state.json")
        if run_state:
            run_state["status"] = "terminated_by_user"
            run_state["terminated_at"] = _now_iso()
            _write_json(run_dir / "state.json", run_state)
    _write_json(_campaign_state_path(config), next_state)
    print("\nCtrl-C received; terminated the active campaign child process.", file=stream)
    return next_state


def _force_cleanup_campaign_state(
    config: Mapping[str, Any],
    *,
    stream: TextIO,
) -> dict[str, Any]:
    state_path = _campaign_state_path(config)
    campaign_state = _read_json_if_exists(state_path)
    run_dir_text = campaign_state.get("run_dir")
    pid = campaign_state.get("pid")
    try:
        pid_int = int(pid) if pid is not None else None
    except (TypeError, ValueError):
        pid_int = None
    process_was_running = _is_process_running(pid_int)
    if process_was_running:
        _terminate_process_group(pid_int)
    previous_status = campaign_state.get("status", "not_started")
    cleanup = {
        "previous_status": previous_status,
        "previous_run_dir": run_dir_text,
        "previous_pid": pid_int,
        "terminated_process": bool(process_was_running),
        "cleaned_at": _now_iso(),
    }
    if run_dir_text and previous_status in {
        "running",
        "termination_requested",
        "terminated",
    }:
        run_dir = Path(str(run_dir_text))
        run_state = _read_json_if_exists(run_dir / "state.json")
        if run_state:
            run_state["status"] = "terminated_by_force_restart"
            run_state["terminated_at"] = _now_iso()
            run_state["force_restart_cleanup"] = cleanup
            _write_json(run_dir / "state.json", run_state)
    if campaign_state:
        archived_state = dict(campaign_state)
        archived_state.update(
            {
                "status": "force_restarted",
                "process_running": False,
                "force_restart_cleanup": cleanup,
                "updated_at": _now_iso(),
            }
        )
        _write_json(state_path, archived_state)
    print(
        "force-restart cleanup: "
        f"previous_status={previous_status}; "
        f"terminated_process={bool(process_was_running)}; "
        f"previous_run={run_dir_text or '-'}",
        file=stream,
    )
    return cleanup


def force_restart_campaign(
    config: Mapping[str, Any],
    *,
    campaign_name: str,
    device: str = "cpu",
    stream: TextIO | None = None,
) -> dict[str, Any]:
    """Terminate stale/running campaign state and launch a fresh mini-batch child."""
    stream = stream or sys.stdout
    cleanup = _force_cleanup_campaign_state(config, stream=stream)
    state = _launch_mini_batch_child(config, campaign_name=campaign_name, device=device)
    state["force_restarted_from"] = cleanup
    _write_json(_campaign_state_path(config), state)
    print(f"launched: {state['run_dir']}", file=stream)
    print(f"logs: {state['child_log_path']}", file=stream)
    return {"state": state, "cleanup": cleanup, "config": dict(config)}


def _handle_agent_decision(
    config: Mapping[str, Any],
    *,
    campaign_state: Mapping[str, Any],
    campaign_name: str,
    device: str = "cpu",
    active_process_pid: int | None = None,
    active_run_running: bool = False,
    client: Any | None = None,
    stream: TextIO | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    stream = stream or sys.stdout
    run_dir_text = campaign_state.get("run_dir")
    run_dir = Path(str(run_dir_text)) if run_dir_text else (_latest_run_dir(config) or Path())
    result = _campaign_result_summary(config, run_dir if run_dir_text else None)
    try:
        decision = request_agent_campaign_decision(
            config,
            campaign_state,
            result,
            client=client,
        )
        _validate_campaign_patch(config, decision.campaign_patch)
    except Exception as exc:
        status = "openai_credits_exhausted" if _is_openai_credits_exhausted(exc) else "agent_decision_failed"
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "at": _now_iso(),
        }
        state = _campaign_state_for_decision(
            config,
            campaign_state,
            status=status,
            phase="agent_decision",
            error=error,
        )
        if status == "openai_credits_exhausted":
            print("status: openai_credits_exhausted; not retrying", file=stream)
        else:
            print(f"status: agent_decision_failed; retry_after={state['poll_seconds']}s", file=stream)
        return state, config

    decision_row = _write_agent_decision(config, run_dir, campaign_state, decision)
    print(f"agent_decision: {decision.decision}", file=stream)
    if active_run_running and decision.decision in {"no_action", "update_logbook"}:
        state = _campaign_state_for_decision(
            config,
            campaign_state,
            status="running",
            phase="mini_batch",
            process_running=True,
            decision_row=decision_row,
        )
        print(f"status: running; continuing active run {run_dir}", file=stream)
        return state, config

    if active_run_running and decision.decision == "stop_campaign":
        _terminate_process_group(active_process_pid)
        _mark_run_terminated_by_agent(run_dir, decision_row=decision_row)
        state = _campaign_state_for_decision(
            config,
            campaign_state,
            status="campaign_completed",
            phase="stopped",
            decision_row=decision_row,
        )
        print(f"terminated: {run_dir}", file=stream)
        return state, config

    if active_run_running and decision.decision in {
        "propose_trial",
        "terminate_run_and_propose_trial",
    }:
        _terminate_process_group(active_process_pid)
        _mark_run_terminated_by_agent(run_dir, decision_row=decision_row)
        patched_config = apply_campaign_patch(config, decision.campaign_patch)
        state = _launch_mini_batch_child(
            patched_config,
            campaign_name=campaign_name,
            device=device,
        )
        state["latest_decision"] = decision_row
        _write_json(_campaign_state_path(patched_config), state)
        print(f"terminated: {run_dir}", file=stream)
        print(f"launched: {state['run_dir']}", file=stream)
        print(f"logs: {state['child_log_path']}", file=stream)
        return state, patched_config

    if decision.decision == "stop_campaign":
        state = _campaign_state_for_decision(
            config,
            campaign_state,
            status="campaign_completed",
            phase="stopped",
            decision_row=decision_row,
        )
        return state, config
    if decision.decision in {"no_action", "update_logbook"}:
        state = _campaign_state_for_decision(
            config,
            campaign_state,
            status="analysis_completed",
            phase="idle",
            decision_row=decision_row,
        )
        return state, config
    if decision.decision in {"propose_trial", "terminate_run_and_propose_trial"}:
        patched_config = apply_campaign_patch(config, decision.campaign_patch)
        state = _launch_mini_batch_child(
            patched_config,
            campaign_name=campaign_name,
            device=device,
        )
        state["latest_decision"] = decision_row
        _write_json(_campaign_state_path(patched_config), state)
        print(f"launched: {state['run_dir']}", file=stream)
        print(f"logs: {state['child_log_path']}", file=stream)
        return state, patched_config
    raise ValueError(f"Unsupported agent decision: {decision.decision!r}")


def run_campaign_loop(
    config: Mapping[str, Any],
    *,
    campaign_name: str,
    once: bool = False,
    dry_run: bool = False,
    force_restart: bool = False,
    device: str = "cpu",
    client: Any | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    stream: TextIO | None = None,
) -> dict[str, Any]:
    """Start or resume the OpenAI-backed campaign loop."""
    stream = stream or sys.stdout
    if dry_run:
        status = campaign_status(config)
        print("status: dry_run", file=stream)
        print(format_campaign_status(status), file=stream)
        return {"state": {"status": "dry_run"}, "status": status, "dry_run": True}

    lock_path = _acquire_campaign_lock(config, force=force_restart)
    active_config = dict(config)
    campaign_state: dict[str, Any] = {}
    log_offsets: dict[str, int] = {}
    try:
        if force_restart:
            restart_result = force_restart_campaign(
                active_config,
                campaign_name=campaign_name,
                device=device,
                stream=stream,
            )
            campaign_state = dict(restart_result["state"])
            if once:
                return restart_result

        while True:
            next_sleep_seconds = int(active_config["runner"]["poll_seconds"])
            state_path = _campaign_state_path(active_config)
            campaign_state = _read_json_if_exists(state_path)
            run_dir_text = campaign_state.get("run_dir")
            run_dir = Path(str(run_dir_text)) if run_dir_text else None
            run_state = _read_json_if_exists(run_dir / "state.json") if run_dir else {}
            child_status = str(run_state.get("status") or "")
            pid = campaign_state.get("pid")
            try:
                pid_int = int(pid) if pid is not None else None
            except (TypeError, ValueError):
                pid_int = None
            running = _is_process_running(pid_int)

            if not campaign_state:
                campaign_state = _launch_mini_batch_child(
                    active_config,
                    campaign_name=campaign_name,
                    device=device,
                )
                print(f"launched: {campaign_state['run_dir']}", file=stream)
                print(f"logs: {campaign_state['child_log_path']}", file=stream)
            elif campaign_state.get("status") == "running" and child_status in {"completed", "failed"}:
                campaign_state["child_status"] = child_status
                campaign_state["process_running"] = running
                campaign_state, active_config = _handle_agent_decision(
                    active_config,
                    campaign_state=campaign_state,
                    campaign_name=campaign_name,
                    device=device,
                    client=client,
                    stream=stream,
                )
            elif campaign_state.get("status") == "running" and pid_int is not None and not running:
                campaign_state["child_status"] = "failed"
                campaign_state["process_running"] = False
                campaign_state["latest_error"] = {
                    "type": "ChildProcessExited",
                    "message": "Mini-batch child process exited before writing final state.",
                    "at": _now_iso(),
                }
                _write_json(_campaign_state_path(active_config), campaign_state)
                campaign_state, active_config = _handle_agent_decision(
                    active_config,
                    campaign_state=campaign_state,
                    campaign_name=campaign_name,
                    device=device,
                    client=client,
                    stream=stream,
                )
            elif campaign_state.get("status") == "running" and running:
                seconds_until_poll = _seconds_until_next_poll(active_config, campaign_state)
                if seconds_until_poll > 0:
                    campaign_state["child_status"] = child_status or "running"
                    campaign_state["process_running"] = True
                    next_sleep_seconds = seconds_until_poll
                    next_poll_at = datetime.now() + timedelta(seconds=seconds_until_poll)
                    print(
                        f"status: running; run_dir={campaign_state.get('run_dir') or '-'}; "
                        f"next_poll={next_poll_at.isoformat(timespec='seconds')}",
                        file=stream,
                    )
                else:
                    campaign_state["child_status"] = child_status or "running"
                    campaign_state["process_running"] = True
                    campaign_state, active_config = _handle_agent_decision(
                        active_config,
                        campaign_state=campaign_state,
                        campaign_name=campaign_name,
                        device=device,
                        active_process_pid=pid_int,
                        active_run_running=True,
                        client=client,
                        stream=stream,
                    )
            elif campaign_state.get("status") == "agent_decision_failed":
                campaign_state, active_config = _handle_agent_decision(
                    active_config,
                    campaign_state=campaign_state,
                    campaign_name=campaign_name,
                    device=device,
                    client=client,
                    stream=stream,
                )
            elif campaign_state.get("status") in {"termination_requested", "terminated"}:
                if running and pid_int is not None:
                    _terminate_process_group(pid_int)
                campaign_state["status"] = "terminated"
                campaign_state["process_running"] = False
                campaign_state["terminated_at"] = _now_iso()
                _write_json(_campaign_state_path(active_config), campaign_state)
                campaign_state = _launch_mini_batch_child(
                    active_config,
                    campaign_name=campaign_name,
                    device=device,
                )
                print(f"launched: {campaign_state['run_dir']}", file=stream)
                print(f"logs: {campaign_state['child_log_path']}", file=stream)
            elif campaign_state.get("status") in {"campaign_completed", "openai_credits_exhausted"}:
                print(f"status: {campaign_state.get('status')}", file=stream)
                return {"state": campaign_state, "config": active_config}
            elif campaign_state.get("status") in {"analysis_completed", "not_started"}:
                campaign_state = _launch_mini_batch_child(
                    active_config,
                    campaign_name=campaign_name,
                    device=device,
                )
                print(f"launched: {campaign_state['run_dir']}", file=stream)
                print(f"logs: {campaign_state['child_log_path']}", file=stream)
            else:
                next_poll_at = datetime.now() + timedelta(
                    seconds=int(active_config["runner"]["poll_seconds"])
                )
                print(
                    f"status: {campaign_state.get('status', 'unknown')}; "
                    f"run_dir={campaign_state.get('run_dir') or '-'}; "
                    f"next_poll={next_poll_at.isoformat(timespec='seconds')}",
                    file=stream,
                )

            if once:
                return {"state": campaign_state, "config": active_config}
            sleep_seconds = max(1, int(next_sleep_seconds))
            next_poll_at = datetime.now() + timedelta(seconds=sleep_seconds)
            print(f"next_poll: {next_poll_at.isoformat(timespec='seconds')}", file=stream)
            _wait_with_foreground_updates(
                active_config,
                campaign_state,
                seconds=sleep_seconds,
                sleep_fn=sleep_fn,
                stream=stream,
                log_offsets=log_offsets,
            )
    except KeyboardInterrupt:
        interrupted_state = _terminate_campaign_state_for_interrupt(
            active_config,
            campaign_state,
            stream=stream,
        )
        return {"state": interrupted_state, "config": active_config, "interrupted": True}
    finally:
        _release_campaign_lock(lock_path)


def campaign_status(config: Mapping[str, Any], *, log_tail_lines: int = 20) -> dict[str, Any]:
    """Read status from the latest campaign run state file."""
    campaign_state = _read_json_if_exists(_campaign_state_path(config))
    run_dir = _latest_run_dir(config)
    if run_dir is None:
        return {
            "campaign": config["campaign"]["domain"],
            "prefix": config["campaign"].get("prefix"),
            "status": campaign_state.get("status", "not_started"),
            "campaign_state_path": str(_campaign_state_path(config)),
            "run_dir": campaign_state.get("run_dir"),
            "logs_dir": None,
            "child_log_path": campaign_state.get("child_log_path"),
            "pid": campaign_state.get("pid"),
            "process_running": _is_process_running(
                int(campaign_state["pid"]) if str(campaign_state.get("pid", "")).isdigit() else None
            ),
            "poll_seconds": int((config.get("runner") or {}).get("poll_seconds", 1800)),
            "queued_trials": [],
            "latest_metrics": {},
            "loss_pdf_paths": [],
            "latest_decision": campaign_state.get("latest_decision", {}),
            "latest_error": campaign_state.get("latest_error", {}),
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
        "prefix": config["campaign"].get("prefix"),
        "status": campaign_state.get("status") or state.get("status", "unknown"),
        "campaign_state_path": str(_campaign_state_path(config)),
        "run_dir": str(run_dir),
        "logs_dir": state.get("logs_dir") or str(run_dir / "logs"),
        "child_log_path": campaign_state.get("child_log_path"),
        "pid": campaign_state.get("pid"),
        "process_running": _is_process_running(
            int(campaign_state["pid"]) if str(campaign_state.get("pid", "")).isdigit() else None
        ),
        "poll_seconds": state.get(
            "poll_seconds",
            (config.get("runner") or {}).get("poll_seconds", 1800),
        ),
        "queued_trials": state.get("queued_trials", []),
        "latest_metrics": state.get("latest_metrics", {}),
        "latest_decision": campaign_state.get("latest_decision", {}),
        "latest_error": campaign_state.get("latest_error") or state.get("latest_error", {}),
        "loss_pdf_paths": state.get("loss_pdf_paths", []),
        "log_tail": "\n".join(log_lines[-int(log_tail_lines) :]),
    }


def format_campaign_status(status: Mapping[str, Any]) -> str:
    """Format campaign status for CLI output."""
    def _display_path(value: Any) -> str:
        if not value:
            return "-"
        text = str(value)
        path = Path(text).expanduser()
        if not path.is_absolute():
            return text
        try:
            return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except (OSError, ValueError):
            return text

    def _format_seconds(value: Any) -> str:
        try:
            seconds = int(value)
        except (TypeError, ValueError):
            return "-"
        if seconds < 60:
            return f"{seconds}s"
        minutes = seconds // 60
        if seconds % 60 == 0 and minutes < 60:
            return f"{seconds}s ({minutes} min)"
        hours = seconds / 3600
        return f"{seconds}s ({hours:.1f} h)"

    def _shorten_text_paths(text: str) -> str:
        shortened = str(text)
        roots = [Path.cwd(), Path.cwd().resolve()]
        for root in roots:
            root_text = root.as_posix().rstrip("/")
            if root_text:
                shortened = shortened.replace(root_text + "/", "")
        return shortened

    def _state_explanation(state: str) -> str:
        explanations = {
            "not_started": "No campaign run has been started yet.",
            "running": "A mini-batch is currently running or waiting for the next poll.",
            "completed": "The latest mini-batch completed.",
            "failed": "The latest mini-batch failed during execution.",
            "agent_decision_failed": (
                "Training is not running. The last agent decision failed, usually because "
                "the OpenAI response could not be parsed. Run the campaign again with "
                "`--once` to retry the decision."
            ),
            "openai_credits_exhausted": "The loop stopped because OpenAI quota or billing is exhausted.",
            "campaign_completed": "The agent stopped the campaign.",
            "analysis_completed": "The agent wrote analysis and did not launch another run.",
            "termination_requested": "Termination has been requested for the active child process.",
            "terminated": "The previous child process was terminated.",
        }
        return explanations.get(state, "See state and logs below for details.")

    def _next_action(state: str) -> str:
        prefix = status.get("prefix") or status.get("campaign") or "<campaign>"
        command = f"./run_nodefield_campaign run {prefix} --once"
        if state == "agent_decision_failed":
            return f"Retry the agent decision with `{command}`."
        if state == "running" and status.get("process_running"):
            return "Wait for the next poll, or inspect the mini-batch log listed below."
        if state == "running":
            return f"The process is not running; use `{command}` to let the controller reconcile state."
        if state in {"analysis_completed", "not_started", "terminated"}:
            return f"Start the next campaign tick with `{command}`."
        if state == "openai_credits_exhausted":
            return "Fix OpenAI billing/quota before retrying the campaign loop."
        return "Inspect the files below before deciding whether to retry or terminate."

    def _metric_rows(metrics: Mapping[str, Any]) -> list[tuple[str, Any]]:
        preferred = [
            "average_num_violations",
            "median_num_violations",
            "feasible_rate",
            "campaign_trial_id",
            "trial_id",
        ]
        rows = [(key, metrics[key]) for key in preferred if key in metrics]
        for key in sorted(metrics):
            if key in preferred or key.endswith("_path"):
                continue
            rows.append((key, metrics[key]))
        return rows

    def _summarize_decision(decision: Mapping[str, Any]) -> list[str]:
        if not decision:
            return ["- No agent decision recorded yet."]
        lines = [
            f"- Decision: `{decision.get('decision', 'unknown')}`",
        ]
        if decision.get("created_at"):
            lines.append(f"- Time: {decision['created_at']}")
        if decision.get("previous_status"):
            lines.append(f"- Previous state: `{decision['previous_status']}`")
        if decision.get("run_dir"):
            lines.append(f"- Decision run: {_display_path(decision['run_dir'])}")
        reason = str(decision.get("reason") or "").strip()
        if reason:
            lines.append(f"- Reason: {reason}")
        patch = decision.get("campaign_patch")
        if isinstance(patch, Mapping):
            agent_patch = patch.get("agent")
            if isinstance(agent_patch, Mapping):
                next_attempt = str(agent_patch.get("next_attempt") or "").strip()
                if next_attempt:
                    lines.append(f"- Next attempt: {next_attempt}")
                touched = sorted(flatten_leaf_paths(agent_patch))
                if touched:
                    lines.append(
                        f"- Patch summary: updates {len(touched)} agent field(s), "
                        "mostly under `agent.default_trial_patch_space`."
                    )
        return lines

    state = str(status.get("status") or "unknown")
    prefix = status.get("prefix")
    lines = [
        "Campaign status",
        f"- Campaign: {status.get('campaign')}" + (f" (`{prefix}`)" if prefix else ""),
        f"- State: `{state}`",
        f"- Meaning: {_state_explanation(state)}",
        f"- Next action: {_next_action(state)}",
        f"- Process: {'running' if status.get('process_running') else 'not running'}"
        f" (pid: {status.get('pid') or '-'})",
        f"- Poll interval: {_format_seconds(status.get('poll_seconds'))}",
        "",
        "Run files",
        f"- Run directory: {_display_path(status.get('run_dir'))}",
        f"- Logs directory: {_display_path(status.get('logs_dir'))}",
        f"- Mini-batch log: {_display_path(status.get('child_log_path'))}",
        f"- Campaign state: {_display_path(status.get('campaign_state_path'))}",
    ]

    queued = status.get("queued_trials") or []
    lines.extend(["", "Trials"])
    if queued:
        lines.extend(["| Trial | Status | Log |", "| --- | --- | --- |"])
        for trial in queued:
            trial_id = trial.get("trial_id")
            trial_name = f"trial_{int(trial_id):03d}" if trial_id is not None else "trial"
            lines.append(
                "| {trial} | {status} | {log} |".format(
                    trial=trial_name,
                    status=trial.get("status", "-"),
                    log=_display_path(trial.get("log_path")),
                )
            )
    else:
        lines.append("- No queued trials recorded.")

    metrics = status.get("latest_metrics") or {}
    lines.extend(["", "Latest metrics"])
    if metrics:
        lines.extend(["| Metric | Value |", "| --- | ---: |"])
        for key, value in _metric_rows(metrics):
            lines.append(f"| `{key}` | {value} |")
        for label, key in (
            ("Loss PDF", "loss_pdf_path"),
            ("Results CSV", "results_csv_path"),
        ):
            if metrics.get(key):
                lines.append(f"- {label}: {_display_path(metrics[key])}")
    else:
        lines.append("- No metrics recorded yet.")

    loss_pdf_paths = status.get("loss_pdf_paths") or []
    lines.extend(["", "Files to inspect"])
    if loss_pdf_paths:
        for index, path in enumerate(loss_pdf_paths, start=1):
            lines.append(f"- Loss PDF {index}: {_display_path(path)}")
    else:
        lines.append("- No loss PDFs recorded yet.")

    latest_decision = status.get("latest_decision") or {}
    lines.extend(["", "Latest agent decision"])
    lines.extend(_summarize_decision(latest_decision))

    latest_error = status.get("latest_error") or {}
    lines.extend(["", "Latest error"])
    if latest_error:
        if latest_error.get("type"):
            lines.append(f"- Type: `{latest_error['type']}`")
        if latest_error.get("at"):
            lines.append(f"- Time: {latest_error['at']}")
        if latest_error.get("message"):
            lines.append(f"- Message: {latest_error['message']}")
        if latest_error.get("log_path"):
            lines.append(f"- Log: {_display_path(latest_error['log_path'])}")
    else:
        lines.append("- No error recorded.")

    log_tail = status.get("log_tail")
    if log_tail:
        lines.extend(["", "Recent log tail", _shorten_text_paths(str(log_tail))])
    return "\n".join(lines)


def terminate_campaign(config: Mapping[str, Any]) -> dict[str, Any]:
    """Mark the latest campaign run as termination requested."""
    campaign_state_path = _campaign_state_path(config)
    campaign_state = _read_json_if_exists(campaign_state_path)
    pid = campaign_state.get("pid")
    pid_int = int(pid) if str(pid).isdigit() else None
    if pid_int and _is_process_running(pid_int):
        _terminate_process_group(pid_int)
        campaign_state["status"] = "termination_requested"
        campaign_state["termination_requested_at"] = _now_iso()
        campaign_state["process_running"] = _is_process_running(pid_int)
        _write_json(campaign_state_path, campaign_state)
        return {
            "campaign": config["campaign"]["domain"],
            "status": "termination_requested",
            "run_dir": campaign_state.get("run_dir"),
            "pid": pid_int,
        }
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
    "AgentCampaignDecision",
    "CAMPAIGN_CONFIGS",
    "MUTABLE_GROUP_PATHS",
    "apply_campaign_patch",
    "apply_exact_trial_patch",
    "campaign_decision_text_format",
    "campaign_status",
    "format_campaign_status",
    "force_restart_campaign",
    "list_campaigns",
    "load_campaign_config",
    "parse_agent_campaign_decision",
    "request_agent_campaign_decision",
    "resolve_campaign_config",
    "run_campaign_loop",
    "run_campaign_once",
    "terminate_campaign",
    "upsert_logbook_block",
]
