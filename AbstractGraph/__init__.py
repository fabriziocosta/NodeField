"""Compatibility shim for local sibling checkouts of ``AbstractGraph``.

This package exposes the canonical ``AbstractGraph`` import root and resolves
it against a sibling source checkout when the package is not installed in the
current environment.
"""

from pathlib import Path

_PROJECTS_ROOT = Path(__file__).resolve().parents[2]
_CANDIDATE_ROOTS = [
    _PROJECTS_ROOT / "AbstractGraph",
    _PROJECTS_ROOT / "AbstractGraph_dev",
]
_SOURCE_ROOT = next((path for path in _CANDIDATE_ROOTS if path.exists()), None)

if _SOURCE_ROOT is None:
    raise ModuleNotFoundError(
        "Could not locate an 'AbstractGraph' source checkout next to this repo. "
        "Tried: "
        + ", ".join(str(path) for path in _CANDIDATE_ROOTS)
    )

# Make ``AbstractGraph.<submodule>`` resolve against the source tree.
__path__ = [str(_SOURCE_ROOT)]

from .core.graphs import AbstractGraph  # noqa: F401,E402
