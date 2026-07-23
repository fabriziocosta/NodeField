"""Scientific belief-state storage and validation for campaign controllers.

The state file is deliberately a small, human-readable snapshot.  Large metric
curves and logs remain beside the experiment that produced them.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, Mapping


STATE_VERSION = 1
ENTITY_TYPES = (
    "components",
    "datasets",
    "experiments",
    "observations",
    "hypotheses",
    "beliefs",
    "questions",
    "candidate_experiments",
)
RELATION_TYPES = {
    "supports",
    "contradicts",
    "refines",
    "supersedes",
    "alternative_to",
    "derived_from",
    "tests",
    "produced",
    "replicates",
    "extends",
    "differs_from",
    "reuses_checkpoint",
    "motivates",
    "rules_out",
    "prioritises",
    "blocks",
    "uses_component",
    "trained_on",
    "evaluated_on",
    "modifies_parameter",
}
IMMUTABLE_ENTITY_TYPES = {"experiments", "observations"}


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


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


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:  # pragma: no cover - project dependencies include PyYAML
        return json.loads(path.read_text(encoding="utf-8"))
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Scientific state must be a mapping: {path}")
    return data


def _dump_yaml(data: Mapping[str, Any]) -> str:
    try:
        import yaml
    except ImportError:  # pragma: no cover
        return json.dumps(_json_safe(data), indent=2, sort_keys=True) + "\n"
    return yaml.safe_dump(_json_safe(data), sort_keys=False)


def state_path(artifact_root: str | Path, domain: str, prefix: str) -> Path:
    """Return the canonical per-campaign state path."""
    return Path(artifact_root) / str(domain) / f"{prefix}_state.yaml"


def new_state(
    *,
    campaign_id: str,
    domain: str,
    objective: str = "Iteratively improve NodeField generation quality.",
    primary_metric: str = "average_num_violations",
    budgets: Mapping[str, Any] | None = None,
    trigger_rules: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create the initial scientific belief state for a campaign."""
    entities = {entity_type: {} for entity_type in ENTITY_TYPES}
    return {
        "schema_version": STATE_VERSION,
        "project": {
            "id": str(campaign_id),
            "domain": str(domain),
            "objective": objective,
            "primary_metric": primary_metric,
            "updated_at": _now_iso(),
        },
        "entities": entities,
        "relations": [],
        "controller_state": {
            "mode": "scientific_campaign",
            "active_run": None,
            "budgets": {
                "remaining_gpu_hours": None,
                "maximum_single_experiment_gpu_hours": 20.0,
                "maximum_parallel_runs": 1,
                **dict(budgets or {}),
            },
            "trigger_rules": list(trigger_rules or default_trigger_rules()),
            "pending_decision": {"candidate_experiments": [], "selected": None},
            "last_decision_at": None,
        },
    }


def default_trigger_rules() -> list[dict[str, Any]]:
    return [
        {
            "id": "trigger_plateau",
            "type": "validation_plateau",
            "window_epochs": 8,
            "minimum_improvement": 0.002,
        },
        {"id": "trigger_generalisation_gap", "type": "generalisation_gap", "threshold": 0.12},
        {"id": "trigger_non_finite", "type": "non_finite_metric"},
        {"id": "trigger_gradient_instability", "type": "unstable_gradients"},
        {"id": "trigger_runtime", "type": "anomalous_runtime"},
    ]


class ScientificStateRepository:
    """Atomic YAML repository with semantic immutability checks."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        state = _read_yaml(self.path)
        validate_state(state)
        return state

    def initialize(self, state: Mapping[str, Any], *, overwrite: bool = False) -> dict[str, Any]:
        if self.path.exists() and not overwrite:
            return self.load()
        payload = deepcopy(dict(state))
        validate_state(payload)
        self.save(payload)
        return payload

    def save(self, state: Mapping[str, Any]) -> None:
        payload = deepcopy(dict(state))
        validate_state(payload)
        payload.setdefault("project", {})["updated_at"] = _now_iso()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        temporary.write_text(_dump_yaml(payload), encoding="utf-8")
        os.replace(temporary, self.path)

    def apply_operations(self, operations: list[Mapping[str, Any]]) -> dict[str, Any]:
        state = self.load()
        updated = apply_operations(state, operations)
        self.save(updated)
        return updated


def validate_state(state: Mapping[str, Any]) -> None:
    if int(state.get("schema_version", -1)) != STATE_VERSION:
        raise ValueError(f"Unsupported scientific state schema: {state.get('schema_version')!r}")
    entities = state.get("entities")
    if not isinstance(entities, Mapping):
        raise ValueError("Scientific state entities must be a mapping.")
    for entity_type in ENTITY_TYPES:
        if not isinstance(entities.get(entity_type), Mapping):
            raise ValueError(f"Scientific state entity collection is invalid: {entity_type}")
    relations = state.get("relations", [])
    if not isinstance(relations, list):
        raise ValueError("Scientific state relations must be a list.")
    for relation in relations:
        if not isinstance(relation, Mapping) or str(relation.get("type")) not in RELATION_TYPES:
            raise ValueError(f"Unsupported scientific relation: {relation!r}")


def _entity_collection(state: dict[str, Any], entity_type: str) -> dict[str, Any]:
    if entity_type not in ENTITY_TYPES:
        raise ValueError(f"Unsupported entity type: {entity_type!r}")
    collection = state["entities"][entity_type]
    if not isinstance(collection, dict):
        raise ValueError(f"Entity collection is not mutable: {entity_type!r}")
    return collection


def allocate_id(state: Mapping[str, Any], entity_type: str) -> str:
    """Allocate the next stable identifier for an entity collection."""
    prefixes = {
        "experiments": "exp",
        "observations": "obs",
        "hypotheses": "hyp",
        "beliefs": "belief",
        "questions": "q",
        "candidate_experiments": "candidate",
        "components": "component",
        "datasets": "dataset",
    }
    prefix = prefixes.get(entity_type, entity_type.rstrip("s"))
    collection = state["entities"][entity_type]
    numbers = []
    for key in collection:
        try:
            numbers.append(int(str(key).rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return f"{prefix}_{max(numbers, default=0) + 1:04d}"


def add_entity(
    state: Mapping[str, Any],
    entity_type: str,
    entity: Mapping[str, Any],
    *,
    entity_id: str | None = None,
) -> dict[str, Any]:
    updated = deepcopy(dict(state))
    collection = _entity_collection(updated, entity_type)
    key = str(entity_id or entity.get("id") or allocate_id(updated, entity_type))
    if key in collection:
        raise ValueError(f"Entity already exists: {entity_type}.{key}")
    value = deepcopy(dict(entity))
    value.pop("id", None)
    collection[key] = value
    return updated


def append_observation(state: Mapping[str, Any], observation: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    entity_id = str(observation.get("id") or allocate_id(state, "observations"))
    return add_entity(state, "observations", observation, entity_id=entity_id), entity_id


def append_experiment(state: Mapping[str, Any], experiment: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    entity_id = str(experiment.get("id") or allocate_id(state, "experiments"))
    return add_entity(state, "experiments", experiment, entity_id=entity_id), entity_id


def update_experiment_lifecycle(
    state: Mapping[str, Any],
    experiment_id: str,
    values: Mapping[str, Any],
) -> dict[str, Any]:
    """Update an unfinished experiment; completed records are immutable."""
    updated = deepcopy(dict(state))
    collection = _entity_collection(updated, "experiments")
    if experiment_id not in collection:
        raise KeyError(f"Unknown experiment: {experiment_id}")
    experiment = collection[experiment_id]
    if experiment.get("status") == "completed":
        raise ValueError(f"Completed experiment is immutable: {experiment_id}")
    for key, value in values.items():
        if key in {"configuration", "purpose"} and experiment.get(key) is not None:
            raise ValueError(f"Experiment definition is immutable after launch: {experiment_id}.{key}")
        experiment[str(key)] = deepcopy(value)
    return updated


def _get_path(target: Mapping[str, Any], path: str) -> Any:
    value: Any = target
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(path)
        value = value[part]
    return value


def _set_path(target: dict[str, Any], path: str, value: Any) -> None:
    cursor = target
    parts = path.split(".")
    for part in parts[:-1]:
        child = cursor.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"Cannot update non-mapping path component: {part}")
        cursor = child
    cursor[parts[-1]] = deepcopy(value)


def apply_operations(state: Mapping[str, Any], operations: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply typed LLM operations while protecting historical evidence."""
    updated = deepcopy(dict(state))
    validate_state(updated)
    for operation in operations:
        kind = str(operation.get("operation", ""))
        entity_type = str(operation.get("entity_type", ""))
        entity_id = str(operation.get("entity_id", ""))
        if kind == "create_entity":
            if entity_type in IMMUTABLE_ENTITY_TYPES:
                raise ValueError("LLM operations cannot create historical experiments or observations.")
            updated = add_entity(updated, entity_type, operation.get("value") or {}, entity_id=entity_id or None)
            continue
        if kind == "add_relation":
            relation = deepcopy(dict(operation.get("value") or {}))
            relation.setdefault("id", f"rel_{len(updated['relations']) + 1:04d}")
            if relation.get("type") not in RELATION_TYPES:
                raise ValueError(f"Unsupported scientific relation: {relation.get('type')!r}")
            updated["relations"].append(relation)
            continue
        if kind != "update_entity":
            raise ValueError(f"Unsupported scientific state operation: {kind!r}")
        if entity_type in IMMUTABLE_ENTITY_TYPES:
            raise ValueError(f"Historical entity collection is immutable: {entity_type}")
        collection = _entity_collection(updated, entity_type)
        if entity_id not in collection:
            raise KeyError(f"Unknown entity: {entity_type}.{entity_id}")
        path = str(operation.get("path", ""))
        current = _get_path(collection[entity_id], path)
        if "old_value" not in operation:
            raise ValueError("Entity updates require old_value for optimistic concurrency.")
        if current != operation["old_value"]:
            raise ValueError(f"Stale scientific state update: {entity_type}.{entity_id}.{path}")
        _set_path(collection[entity_id], path, operation.get("new_value"))
    validate_state(updated)
    return updated


def validate_candidate(
    candidate: Mapping[str, Any],
    *,
    allowed_paths: list[str],
    remaining_gpu_hours: float | None,
    maximum_single_experiment_gpu_hours: float | None,
) -> None:
    """Validate the policy-critical parts of an LLM candidate."""
    design = candidate.get("design")
    cost = candidate.get("cost") or {}
    if not isinstance(design, Mapping):
        raise ValueError("Candidate experiment must contain a design mapping.")
    if not isinstance(design.get("fixed"), Mapping) or not isinstance(design.get("varied"), Mapping):
        raise ValueError("Candidate design requires fixed and varied mappings.")
    seeds = design.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("Candidate design requires at least one seed.")
    expected = candidate.get("expected_outcomes")
    if not isinstance(expected, Mapping) or not expected:
        raise ValueError("Candidate experiment must state expected discriminating outcomes.")
    estimated = float(cost.get("estimated_gpu_hours"))
    if estimated < 0:
        raise ValueError("Candidate estimated GPU hours must be non-negative.")
    if maximum_single_experiment_gpu_hours is not None and estimated > maximum_single_experiment_gpu_hours:
        raise ValueError("Candidate exceeds maximum single-experiment budget.")
    if remaining_gpu_hours is not None and estimated > remaining_gpu_hours:
        raise ValueError("Candidate exceeds remaining campaign budget.")
    varied = design.get("varied")
    paths = list(_leaf_paths(varied))
    if any(not any(path == allowed or path.startswith(f"{allowed}.") for allowed in allowed_paths) for path in paths):
        raise ValueError("Candidate varies a parameter outside the campaign allowlist.")


def _leaf_paths(value: Mapping[str, Any], prefix: str = "") -> list[str]:
    paths: list[str] = []
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping) and not {"type", "low", "high", "values"}.intersection(item):
            paths.extend(_leaf_paths(item, path))
        else:
            paths.append(path)
    return paths


__all__ = [
    "ENTITY_TYPES",
    "IMMUTABLE_ENTITY_TYPES",
    "RELATION_TYPES",
    "STATE_VERSION",
    "ScientificStateRepository",
    "add_entity",
    "allocate_id",
    "append_experiment",
    "append_observation",
    "apply_operations",
    "default_trigger_rules",
    "new_state",
    "state_path",
    "update_experiment_lifecycle",
    "validate_candidate",
    "validate_state",
]
