"""Shared path and local-workspace resolution helpers."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _resolve_start_path(start: str | Path | None = None) -> Path:
    return Path.cwd().resolve() if start is None else Path(start).expanduser().resolve()


def resolve_repo_root(start: str | Path | None = None) -> Path:
    """Locate the NodeField repo root from the provided starting path."""
    start_path = _resolve_start_path(start)
    search_roots = [start_path, *start_path.parents] if start_path.is_dir() else [start_path.parent, *start_path.parent.parents]
    for root in search_roots:
        if (root / "conditional_node_field_graph_generator").exists():
            return root.resolve()
    raise ModuleNotFoundError(
        "Could not locate the NodeField repo root from the provided path."
    )


def ensure_repo_on_syspath(start: str | Path | None = None) -> Path:
    repo_root = resolve_repo_root(start=start)
    resolved = str(repo_root)
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    return repo_root


def resolve_artifact_root(
    artifact_root: str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
) -> Path:
    if artifact_root is not None:
        root = Path(artifact_root).expanduser().resolve()
    else:
        root = resolve_repo_root(repo_root) / ".artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_notebook_data_root(repo_root: str | Path | None = None) -> Path:
    return resolve_repo_root(repo_root) / "notebooks" / "datasets"


def resolve_checkpoint_root(
    checkpoint_root: str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
) -> Path:
    if checkpoint_root is not None:
        root = Path(checkpoint_root).expanduser().resolve()
    else:
        root = resolve_artifact_root(repo_root=repo_root) / "checkpoints" / "node_field"
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_saved_generator_dir(
    model_dir: str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
) -> Path:
    if model_dir is not None:
        root = Path(model_dir).expanduser().resolve()
    else:
        root = resolve_artifact_root(repo_root=repo_root) / "saved_generators"
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_pubchem_data_root(
    pubchem_dir: str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
) -> Path:
    if pubchem_dir is not None:
        return Path(pubchem_dir).expanduser().resolve()
    env_path = os.environ.get("PUBCHEM_DATA_DIR")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return resolve_notebook_data_root(repo_root=repo_root) / "PUBCHEM"


def resolve_zinc_data_root(
    dataset_dir: str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
) -> Path:
    if dataset_dir is not None:
        return Path(dataset_dir).expanduser().resolve()
    return resolve_notebook_data_root(repo_root=repo_root) / "zinc"


def resolve_nsppk_root(
    repo_root: str | Path | None = None,
    *,
    start: str | Path | None = None,
) -> Path | None:
    resolved_repo_root = resolve_repo_root(repo_root if repo_root is not None else start)
    cwd = _resolve_start_path(start)
    candidates: list[Path] = []
    env_root = os.environ.get("NSPPK_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend(
        [
            resolved_repo_root / "NSPPK",
            resolved_repo_root.parent / "NSPPK",
            cwd / "NSPPK",
            cwd.parent / "NSPPK",
        ]
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return None
