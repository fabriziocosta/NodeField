"""Compatibility shim for a local ``abstractgraph`` source checkout."""

from _external_imports import resolve_source_checkout

_SOURCE_ROOT = resolve_source_checkout(
    "abstractgraph/src/abstractgraph",
    "repos/abstractgraph/src/abstractgraph",
)

if _SOURCE_ROOT is None:
    raise ModuleNotFoundError(
        "Could not locate an 'abstractgraph' source checkout. "
        "Expected a sibling checkout or an abstractgraph-ecosystem workspace."
    )

__path__ = [str(_SOURCE_ROOT)]

from .graphs import AbstractGraph  # noqa: F401,E402
