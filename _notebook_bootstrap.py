"""Shared notebook bootstrap helpers for local development."""

from __future__ import annotations

import importlib
import os
import sys
import warnings
from pathlib import Path


def _add_to_syspath(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def find_repo_root() -> Path:
    """Locate the NodeField repo root from a notebook working directory."""
    for root in [Path.cwd(), *Path.cwd().parents]:
        if (root / "conditional_node_field_graph_generator").exists():
            return root.resolve()
    raise ModuleNotFoundError(
        "Could not locate the NodeField repo root from the current notebook directory."
    )


def ensure_repo_on_syspath() -> Path:
    repo_root = find_repo_root()
    _add_to_syspath(repo_root)
    return repo_root


def ensure_nsppk_on_syspath(repo_root: Path) -> Path | None:
    """Add a local NSPPK checkout to sys.path when present."""
    candidates: list[Path] = []
    if os.environ.get("NSPPK_ROOT"):
        candidates.append(Path(os.environ["NSPPK_ROOT"]).expanduser())
    candidates.extend(
        [
            repo_root / "NSPPK",
            repo_root.parent / "NSPPK",
            Path.cwd() / "NSPPK",
            Path.cwd().parent / "NSPPK",
        ]
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            _add_to_syspath(resolved)
            return resolved
    return None


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

    artifact_root = repo_root / ".artifacts"
    context: dict[str, Path] = {
        "REPO_ROOT": repo_root,
        "ARTIFACT_ROOT": artifact_root,
        "NOTEBOOK_DATA_ROOT": repo_root / "notebooks" / "datasets",
        "CHECKPOINT_ROOT": artifact_root / "checkpoints" / "node_field",
        "SAVED_GENERATOR_ROOT": artifact_root / "saved_generators",
    }
    if nsppk_root is not None:
        context["NSPPK_ROOT"] = nsppk_root
    return context
