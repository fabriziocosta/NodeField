"""Compatibility shim for a local ``abstractgraph_ml`` source checkout."""

from _external_imports import resolve_source_checkout

_SOURCE_ROOT = resolve_source_checkout(
    "abstractgraph-ml/src/abstractgraph_ml",
    "repos/abstractgraph-ml/src/abstractgraph_ml",
)

if _SOURCE_ROOT is None:
    raise ModuleNotFoundError(
        "Could not locate an 'abstractgraph_ml' source checkout. "
        "Expected a sibling checkout or an abstractgraph-ecosystem workspace."
    )

__path__ = [str(_SOURCE_ROOT)]

from .estimators import GraphEstimator, IsolationForestProba  # noqa: F401,E402
