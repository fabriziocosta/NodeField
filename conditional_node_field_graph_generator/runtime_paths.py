"""Shared path and local-workspace resolution helpers."""

from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime
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


def find_local_nsppk_repo(start: str | Path | None = None) -> Path | None:
    """Return a likely local NSPPK checkout next to the NodeField repo."""
    repo_root = ensure_repo_on_syspath(start)
    candidates = [
        repo_root.parent / "NSPPK",
        repo_root.parent / "nsppk",
        repo_root.parent / "INACTIVE" / "NSPPK",
        repo_root.parent / "INACTIVE" / "nsppk",
    ]
    for candidate in candidates:
        if (candidate / "pyproject.toml").exists() or (candidate / "setup.py").exists():
            return candidate.resolve()
    return None


def find_local_nsppk_import_path(start: str | Path | None = None) -> Path | None:
    """Return the importable path for a local NSPPK checkout if one exists."""
    repo = find_local_nsppk_repo(start)
    if repo is None:
        return None
    src_dir = repo / "src"
    if src_dir.is_dir():
        return src_dir.resolve()
    return repo.resolve()


def ensure_local_nsppk_on_syspath(start: str | Path | None = None) -> Path | None:
    """Add a sibling local NSPPK checkout to sys.path when available."""
    import_path = find_local_nsppk_import_path(start)
    if import_path is None:
        return None
    resolved = str(import_path)
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    return import_path


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


def resolve_campaign_artifact_root(
    artifact_root: str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
) -> Path:
    """Resolve the NodeField campaign artifact root.

    New campaign workflows intentionally use ``artifact/`` instead of the legacy
    ``.artifacts/`` root used by existing notebooks and demos.
    """
    if artifact_root is not None:
        root = Path(artifact_root).expanduser()
        if not root.is_absolute():
            root = resolve_repo_root(repo_root) / root
        root = root.resolve()
    else:
        root = resolve_repo_root(repo_root) / "artifact"
    root.mkdir(parents=True, exist_ok=True)
    return root


def make_timestamped_run_dir(
    root: str | Path,
    prefix: str,
    *,
    now: datetime | None = None,
    short_id: str | None = None,
    create: bool = True,
) -> Path:
    """Create ``<root>/<prefix>_YYYYMMDD_HHMMSS_<shortid>`` and return it."""
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    token = short_id or uuid.uuid4().hex[:6]
    run_dir = Path(root).expanduser().resolve() / f"{prefix}_{timestamp}_{token}"
    if create:
        run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


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
