"""Compatibility shim for local source checkouts of legacy ``AbstractGraph``."""

from _external_imports import build_optional_dependency_candidates, resolve_source_checkout

_CANDIDATE_ROOTS = [
    base / relative_root
    for base in build_optional_dependency_candidates()
    for relative_root in ("AbstractGraph", "AbstractGraph_dev")
]
_SOURCE_ROOT = resolve_source_checkout("AbstractGraph", "AbstractGraph_dev")

if _SOURCE_ROOT is None:
    raise ModuleNotFoundError(
        "Could not locate an 'AbstractGraph' source checkout next to this repo. "
        "Tried: "
        + ", ".join(str(path) for path in _CANDIDATE_ROOTS)
    )

# Make ``AbstractGraph.<submodule>`` resolve against the source tree.
__path__ = [str(_SOURCE_ROOT)]

from .core.graphs import AbstractGraph  # noqa: F401,E402
