"""Helpers for iterating graph sources used by streaming training APIs."""

from __future__ import annotations

import csv
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
    except ImportError:
        MolecularGraphSourceLoader = None

    if MolecularGraphSourceLoader is not None:
        loader = MolecularGraphSourceLoader(on_error="skip")
        yield from loader.iter_graphs(uri, source_type)
        return

    try:
        from abstractgraph_graphicalizer.chem import ZINCLoader, smiles_to_graph
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Molecular stream sources require abstractgraph_graphicalizer chemistry support."
        ) from exc

    normalized_type = str(source_type).strip().lower()
    path = Path(uri)
    if normalized_type == "zinc_csv":
        loader = ZINCLoader(path.parent, on_error="skip")
        dataset_name = path.stem
        with open(path, "r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            for row_index, row in enumerate(reader):
                graph = loader._graph_from_row(row, dataset_name=dataset_name, row_index=row_index)
                if graph is not None:
                    yield graph
        return

    if normalized_type in {"smiles_csv", "csv_smiles"}:
        with open(path, "r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = [str(name).strip() for name in (reader.fieldnames or []) if name]
            smiles_field = "smiles" if "smiles" in fieldnames else None
            if smiles_field is None and fieldnames:
                smiles_field = fieldnames[0]
            if smiles_field is None:
                raise ValueError(f"Could not determine a SMILES column for source: {path}")
            for row_index, row in enumerate(reader):
                smiles = row.get(smiles_field)
                if smiles is None or not str(smiles).strip():
                    continue
                try:
                    graph = smiles_to_graph(str(smiles))
                except Exception:
                    continue
                graph.graph["source"] = normalized_type
                graph.graph["input"] = f"{path.stem}[{row_index}]"
                for key, value in row.items():
                    if key is None or value is None or value == "":
                        continue
                    graph.graph[key] = value
                yield graph
        return

    raise ValueError(f"Unsupported molecular source type: {source_type!r}")


_LOCAL_READERS: dict[str, Callable[[str | Path], Iterable[nx.Graph]]] = {
    "smiles_csv": lambda uri: _iter_molecular_source_graphs(uri, "smiles_csv"),
    "csv_smiles": lambda uri: _iter_molecular_source_graphs(uri, "csv_smiles"),
    "zinc_csv": lambda uri: _iter_molecular_source_graphs(uri, "zinc_csv"),
}


def estimate_source_instance_count(uri, source_type) -> Optional[int]:
    normalized_type = str(source_type).strip().lower()
    if normalized_type not in _LOCAL_READERS:
        return None
    path = Path(uri)
    if not path.is_file():
        return None
    if path.suffix.lower() != ".csv":
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            line_count = sum(1 for _ in handle)
    except OSError:
        return None
    if line_count <= 1:
        return 0
    return max(0, line_count - 1)


def iter_selected_source_graphs(
    uri,
    source_type,
    *,
    reader: Optional[Callable] = None,
    limit=None,
    random_state=None,
    verbose: bool = False,
    start_after_instance: int = 0,
    max_selected: Optional[int] = None,
):
    if start_after_instance is None:
        start_after_instance = 0
    start_after_instance = int(start_after_instance)
    if start_after_instance < 0:
        raise ValueError("start_after_instance must be >= 0")
    if max_selected is not None:
        max_selected = int(max_selected)
        if max_selected < 0:
            raise ValueError("max_selected must be >= 0 when provided")

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
        if max_selected is not None and yielded >= max_selected:
            break
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
