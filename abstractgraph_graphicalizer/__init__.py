"""Compatibility shim for a local ``abstractgraph_graphicalizer`` source checkout."""

from _external_imports import resolve_source_checkout

_SOURCE_ROOT = resolve_source_checkout(
    "abstractgraph-graphicalizer/src/abstractgraph_graphicalizer",
    "repos/abstractgraph-graphicalizer/src/abstractgraph_graphicalizer",
)

if _SOURCE_ROOT is None:
    raise ModuleNotFoundError(
        "Could not locate an 'abstractgraph_graphicalizer' source checkout. "
        "Expected a sibling checkout or an abstractgraph-ecosystem workspace."
    )

__path__ = [str(_SOURCE_ROOT)]

