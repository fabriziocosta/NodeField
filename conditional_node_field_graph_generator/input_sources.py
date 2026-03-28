"""Helpers for iterating graph sources used by streaming training APIs."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional

import networkx as nx
import numpy as np


def make_stream_rng(random_state=None):
    if random_state is None:
        return np.random.default_rng()
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _normalize_limit(limit):
    if limit is None:
        return None
    if isinstance(limit, (int, np.integer)):
        if int(limit) < 0:
            raise ValueError("limit must be >= 0 when provided as an integer.")
        return int(limit)
    if isinstance(limit, float):
        if not 0.0 < float(limit) < 1.0:
            raise ValueError("float limit must be strictly between 0 and 1.")
        return float(limit)
    raise TypeError("limit must be None, int, or float.")


def _iter_molecular_source_graphs(uri, source_type) -> Iterable[nx.Graph]:
    try:
        from abstractgraph_graphicalizer.chem import MolecularGraphSourceLoader
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Molecular stream sources require abstractgraph_graphicalizer.chem.MolecularGraphSourceLoader."
        ) from exc
    loader = MolecularGraphSourceLoader(on_error="skip")
    yield from loader.iter_graphs(uri, source_type)


_LOCAL_READERS: dict[str, Callable[[str | Path], Iterable[nx.Graph]]] = {
    "smiles_csv": lambda uri: _iter_molecular_source_graphs(uri, "smiles_csv"),
    "csv_smiles": lambda uri: _iter_molecular_source_graphs(uri, "csv_smiles"),
    "zinc_csv": lambda uri: _iter_molecular_source_graphs(uri, "zinc_csv"),
}


def iter_selected_source_graphs(
    uri,
    source_type,
    *,
    reader: Optional[Callable] = None,
    limit=None,
    random_state=None,
    verbose: bool = False,
    start_after_instance: int = 0,
):
    if start_after_instance is None:
        start_after_instance = 0
    start_after_instance = int(start_after_instance)
    if start_after_instance < 0:
        raise ValueError("start_after_instance must be >= 0")

    normalized_type = str(source_type).strip().lower()
    if reader is None and normalized_type not in _LOCAL_READERS:
        try:
            import graph_io as _graph_io
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Streaming graph loading requires the NSPPK graph_io module "
                "when a local reader for the selected type is unavailable."
            ) from exc
        yield from _graph_io._iter_loaded_graphs(
            uri,
            source_type,
            reader=None,
            limit=limit,
            random_state=random_state,
            verbose=verbose,
            mode="stream",
            start_after_instance=start_after_instance,
        )
        return

    graph_iterable = reader(uri) if reader is not None else _LOCAL_READERS[normalized_type](uri)
    normalized_limit = _normalize_limit(limit)
    rng = make_stream_rng(random_state)
    yielded = 0
    for raw_index, graph in enumerate(graph_iterable):
        if raw_index < start_after_instance:
            continue
        if normalized_limit is None:
            pass
        elif isinstance(normalized_limit, int):
            if yielded >= normalized_limit:
                break
        else:
            if rng.random() > normalized_limit:
                continue
        if not isinstance(graph, nx.Graph):
            raise TypeError(
                "Graph readers must yield networkx.Graph instances "
                f"(got {type(graph)!r})."
            )
        yielded += 1
        yield graph
