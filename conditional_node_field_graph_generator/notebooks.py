"""Notebook bootstrap helpers exposed through the installed package."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

from abstractgraph_graphicalizer.chem import download_zinc_dataset as _chem_download_zinc_dataset

from .runtime_paths import (
    ensure_repo_on_syspath as _ensure_repo_on_syspath,
    resolve_artifact_root,
    resolve_checkpoint_root,
    resolve_notebook_data_root,
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


def import_nsppk():
    """Import the installed NSPPK entry points."""
    import nsppk as module

    node_nsppk = getattr(module, "NodeNSPPK", None)
    return module.NSPPK, node_nsppk


def find_local_nsppk_repo(start: str | Path | None = None) -> Path | None:
    """Return a likely local NSPPK checkout if one exists near the repo."""
    repo_root = _ensure_repo_on_syspath(Path.cwd() if start is None else Path(start))
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


def configure_notebook(*, require_nsppk: bool = False, print_torch: bool = True) -> dict[str, Path]:
    """Resolve local paths, silence noisy warnings, and optionally expose NSPPK."""
    warnings.filterwarnings("ignore", message=".*PossibleUserWarning.*")
    warnings.filterwarnings("ignore", message=".*does not have many workers.*")
    warnings.filterwarnings("ignore", message=".*to enable TensorBoard support.*")

    repo_root = ensure_repo_on_syspath()
    if require_nsppk:
        try:
            import_nsppk()
        except ModuleNotFoundError as exc:
            local_nsppk_repo = find_local_nsppk_repo(repo_root)
            install_hint = "Install the 'nsppk' package in the current environment."
            if local_nsppk_repo is not None:
                install_hint = (
                    "Install the local NSPPK checkout in the current environment, "
                    f"for example: pip install -e '{local_nsppk_repo}'."
                )
            raise ModuleNotFoundError(
                f"Could not import NSPPK. {install_hint}"
            ) from exc

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
    return context


def download_zinc_dataset(dataset_dir: str | Path, filename: str | None = None) -> Path:
    """Return a ZINC CSV path, defaulting to the external downloader for the canonical file.

    When ``filename`` is provided, this helper returns ``dataset_dir / filename`` and requires
    that the file already exists. This keeps notebook workflows repo-local while still allowing
    alternate filtered copies such as ``zinc_18.csv``.
    """
    dataset_path = Path(dataset_dir).expanduser()
    dataset_path.mkdir(parents=True, exist_ok=True)
    if filename is None:
        return Path(_chem_download_zinc_dataset(dataset_path))
    requested_path = dataset_path / str(filename)
    if not requested_path.is_file():
        raise FileNotFoundError(f"Requested ZINC dataset file does not exist: {requested_path}")
    return requested_path


__all__ = [
    "configure_notebook",
    "download_zinc_dataset",
    "ensure_repo_on_syspath",
    "find_repo_root",
    "find_local_nsppk_repo",
    "import_nsppk",
]
