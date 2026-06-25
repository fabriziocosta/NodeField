"""Range-search utilities for NodeField agent campaign trials."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Mapping

import numpy as np


_SPEC_TYPES = {"real", "int", "choice"}


def _is_range_spec(value: Any) -> bool:
    return isinstance(value, Mapping) and str(value.get("type", "")) in _SPEC_TYPES


def _flatten_patch_space(
    value: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, dict[str, Any]]:
    flattened: dict[str, dict[str, Any]] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if _is_range_spec(item):
            flattened[path] = dict(item)
        elif isinstance(item, Mapping):
            flattened.update(_flatten_patch_space(item, prefix=path))
        else:
            raise ValueError(f"Patch-space leaf {path!r} must be a range spec.")
    return flattened


def _path_allowed(path: str, allowed_paths: list[str]) -> bool:
    return any(path == allowed or path.startswith(f"{allowed}.") for allowed in allowed_paths)


def _set_nested_value(target: dict[str, Any], dotted_path: str, value: Any) -> None:
    cursor = target
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
        if not isinstance(cursor, dict):
            raise ValueError(f"Cannot set nested path {dotted_path!r}; {part!r} is not a mapping.")
    cursor[parts[-1]] = value


def _validate_spec(path: str, spec: Mapping[str, Any]) -> None:
    spec_type = str(spec.get("type"))
    if spec_type not in _SPEC_TYPES:
        raise ValueError(f"Unsupported range type for {path!r}: {spec_type!r}")
    if spec_type == "choice":
        values = spec.get("values")
        if not isinstance(values, list) or not values:
            raise ValueError(f"Choice range {path!r} must define a non-empty values list.")
        return
    if "low" not in spec or "high" not in spec:
        raise ValueError(f"Range {path!r} must define low and high.")
    low = float(spec["low"])
    high = float(spec["high"])
    if high < low:
        raise ValueError(f"Range {path!r} has high < low.")
    if str(spec.get("scale", "linear")) == "log" and (low <= 0 or high <= 0):
        raise ValueError(f"Log range {path!r} requires positive low and high.")
    if str(spec.get("scale", "linear")) not in {"linear", "log"}:
        raise ValueError(f"Range {path!r} scale must be 'linear' or 'log'.")


def validate_patch_space(
    patch_space: Mapping[str, Any],
    *,
    allowed_paths: list[str],
    max_leaf_count: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Validate and flatten an agent-proposed patch-space mapping."""
    flattened = _flatten_patch_space(patch_space)
    if max_leaf_count is not None and len(flattened) > int(max_leaf_count):
        raise ValueError(
            f"Patch space changes {len(flattened)} leaves, exceeding max_search_leaf_count={max_leaf_count}."
        )
    rejected = sorted(path for path in flattened if not _path_allowed(path, allowed_paths))
    if rejected:
        raise ValueError("Patch space contains non-allowlisted path(s): " + ", ".join(rejected))
    for path, spec in flattened.items():
        _validate_spec(path, spec)
    return flattened


def _sample_spec(spec: Mapping[str, Any], rng: np.random.Generator) -> Any:
    spec_type = str(spec["type"])
    if spec_type == "choice":
        values = list(spec["values"])
        return deepcopy(values[int(rng.integers(0, len(values)))])
    low = float(spec["low"])
    high = float(spec["high"])
    if str(spec.get("scale", "linear")) == "log":
        value = math.exp(rng.uniform(math.log(low), math.log(high)))
    else:
        value = rng.uniform(low, high)
    if spec_type == "int":
        return int(rng.integers(int(low), int(high) + 1))
    return float(value)


def sample_patch_space(
    patch_space: Mapping[str, Any],
    *,
    n_samples: int,
    random_state: int = 0,
    allowed_paths: list[str],
    max_leaf_count: int | None = None,
) -> list[dict[str, Any]]:
    """Sample exact nested patches from a validated patch-space mapping."""
    flattened = validate_patch_space(
        patch_space,
        allowed_paths=allowed_paths,
        max_leaf_count=max_leaf_count,
    )
    rng = np.random.default_rng(int(random_state))
    patches: list[dict[str, Any]] = []
    for _ in range(int(n_samples)):
        patch: dict[str, Any] = {}
        for path, spec in flattened.items():
            _set_nested_value(patch, path, _sample_spec(spec, rng))
        patches.append(patch)
    return patches


__all__ = ["sample_patch_space", "validate_patch_space"]
