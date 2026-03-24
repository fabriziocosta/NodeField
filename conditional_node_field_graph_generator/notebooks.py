"""Notebook bootstrap helpers exposed through the installed package."""

from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path

from .runtime_paths import (
    ensure_repo_on_syspath as _ensure_repo_on_syspath,
    resolve_artifact_root,
    resolve_checkpoint_root,
    resolve_notebook_data_root,
    resolve_nsppk_root,
    resolve_saved_generator_dir,
)


def _add_to_syspath(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def find_repo_root() -> Path:
    """Locate the NodeField repo root from a notebook working directory."""
    return _ensure_repo_on_syspath(Path.cwd())


def ensure_repo_on_syspath() -> Path:
    return _ensure_repo_on_syspath(Path.cwd())


def ensure_nsppk_on_syspath(repo_root: Path) -> Path | None:
    """Add a local NSPPK checkout to sys.path when present."""
    nsppk_root = resolve_nsppk_root(repo_root=repo_root, start=Path.cwd())
    if nsppk_root is not None:
        _add_to_syspath(nsppk_root)
    return nsppk_root


def import_nsppk():
    """Import the preferred NSPPK entry points across legacy layouts."""
    try:
        module = importlib.import_module("NSPPPK.nsppk")
    except ModuleNotFoundError:
        module = importlib.import_module("nsppk")
    node_nsppk = getattr(module, "NodeNSPPK", None)
    return module.NSPPK, node_nsppk


def configure_notebook(*, require_nsppk: bool = False, print_torch: bool = True) -> dict[str, Path]:
    """Resolve local paths, silence noisy warnings, and optionally expose NSPPK."""
    warnings.filterwarnings("ignore", message=".*PossibleUserWarning.*")
    warnings.filterwarnings("ignore", message=".*does not have many workers.*")
    warnings.filterwarnings("ignore", message=".*to enable TensorBoard support.*")

    repo_root = ensure_repo_on_syspath()
    nsppk_root = ensure_nsppk_on_syspath(repo_root)
    if require_nsppk:
        try:
            import_nsppk()
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Could not locate NSPPK. Set NSPPK_ROOT or clone NSPPK next to this repo."
            ) from exc
        if nsppk_root is None and "nsppk" not in sys.modules:
            raise ModuleNotFoundError(
                "Could not locate NSPPK. Set NSPPK_ROOT or clone NSPPK next to this repo."
            )

    if print_torch:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

    artifact_root = resolve_artifact_root(repo_root=repo_root)
    context: dict[str, Path] = {
        "REPO_ROOT": repo_root,
        "ARTIFACT_ROOT": artifact_root,
        "NOTEBOOK_DATA_ROOT": resolve_notebook_data_root(repo_root=repo_root),
        "CHECKPOINT_ROOT": resolve_checkpoint_root(repo_root=repo_root),
        "SAVED_GENERATOR_ROOT": resolve_saved_generator_dir(repo_root=repo_root),
    }
    if nsppk_root is not None:
        context["NSPPK_ROOT"] = nsppk_root
    return context


__all__ = [
    "configure_notebook",
    "ensure_nsppk_on_syspath",
    "ensure_repo_on_syspath",
    "find_repo_root",
    "import_nsppk",
]
