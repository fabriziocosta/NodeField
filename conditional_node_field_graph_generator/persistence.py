"""General persistence helpers for fitted NodeField objects."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

import dill as pickle
import pandas as pd

from .runtime_paths import resolve_saved_generator_dir as _resolve_saved_generator_dir


GRAPH_GENERATOR_PERSISTENCE_VERSION = 3

try:
    from IPython.display import display
except Exception:  # pragma: no cover
    def display(obj):  # type: ignore
        print(obj)


def resolve_saved_generator_dir(model_dir=None):
    return _resolve_saved_generator_dir(model_dir=model_dir)


def _sanitize_model_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")
    return token or "gg"


def _restore_loaded_vectorizer_fit_state(graph_generator) -> None:
    """Repair legacy NSPPK-style fitted flags on trusted loaded generators."""
    if not bool(getattr(graph_generator, "is_fitted_", False)):
        return
    for attr_name in ("graph_vectorizer", "node_graph_vectorizer"):
        vectorizer = getattr(graph_generator, attr_name, None)
        if vectorizer is None:
            continue
        if hasattr(vectorizer, "base_nsppk"):
            base_nsppk = getattr(vectorizer, "base_nsppk", None)
            if base_nsppk is not None and not bool(getattr(base_nsppk, "is_fitted_", False)):
                base_nsppk.is_fitted_ = True
        nsppk = getattr(vectorizer, "nsppk", None)
        if nsppk is not None and hasattr(nsppk, "base_nsppk"):
                base_nsppk = getattr(nsppk, "base_nsppk", None)
                if base_nsppk is not None and not bool(getattr(base_nsppk, "is_fitted_", False)):
                    base_nsppk.is_fitted_ = True


def _restore_loaded_generator_runtime_defaults(graph_generator) -> None:
    """Backfill runtime attrs added after older persisted generators were saved."""
    if not hasattr(graph_generator, "oracle_use_node_label_cuts"):
        graph_generator.oracle_use_node_label_cuts = False
    if not hasattr(graph_generator, "oracle_use_edge_label_cuts"):
        graph_generator.oracle_use_edge_label_cuts = False


def save_graph_generator(graph_generator, model_name=None, model_dir=None, log=True):
    resolved_model_name = model_name if model_name is not None else getattr(graph_generator, "model_name", None)
    if resolved_model_name is None:
        print("Skipping graph generator save because model_name is None.")
        return None
    resolved_model_dir = model_dir if model_dir is not None else getattr(graph_generator, "model_dir", None)
    model_root = resolve_saved_generator_dir(model_dir=resolved_model_dir)
    graph_generator._persistence_schema_version = GRAPH_GENERATOR_PERSISTENCE_VERSION
    stem = _sanitize_model_token(resolved_model_name)
    filename = f"{stem}.pkl"
    path = model_root / filename
    with open(path, "wb") as handle:
        pickle.dump(graph_generator, handle)
    if log:
        print(f"Saved graph generator as: {filename}")
        print(path)
    return filename


def list_saved_graph_generators(model_dir=None):
    model_root = resolve_saved_generator_dir(model_dir=model_dir)
    files = sorted(model_root.glob("*.pkl"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not files:
        print(f"No saved graph generators found in {model_root}")
        return []
    rows = [
        {
            "name": path.name,
            "modified": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            "size_mb": round(path.stat().st_size / (1024 * 1024), 1),
        }
        for path in files
    ]
    frame = pd.DataFrame(rows)
    display(frame)
    return [path.name for path in files]


def load_graph_generator(model_name, model_dir=None):
    model_root = resolve_saved_generator_dir(model_dir=model_dir)
    requested = str(model_name).strip()
    candidates = []
    direct_path = Path(requested).expanduser()
    if direct_path.is_file():
        candidates = [direct_path.resolve()]
    else:
        names_to_try = {requested}
        if not requested.endswith(".pkl"):
            names_to_try.add(f"{requested}.pkl")
        for candidate_name in names_to_try:
            candidate_path = model_root / candidate_name
            if candidate_path.is_file():
                candidates.append(candidate_path.resolve())
        if not candidates:
            pattern = requested[:-4] if requested.endswith(".pkl") else requested
            matches = sorted(model_root.glob(f"{pattern}*.pkl"))
            candidates = [path.resolve() for path in matches]
    if not candidates:
        raise FileNotFoundError(f"Could not find a saved graph generator matching {requested!r} in {model_root}.")
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple saved graph generators match {requested!r}: "
            + ", ".join(path.name for path in candidates)
        )
    path = candidates[0]
    with open(path, "rb") as handle:
        graph_generator = pickle.load(handle)
    schema_version = int(getattr(graph_generator, "_persistence_schema_version", 0))
    if schema_version != GRAPH_GENERATOR_PERSISTENCE_VERSION:
        raise RuntimeError(
            "Saved graph generator schema is incompatible with this NodeField version. "
            f"Expected schema v{GRAPH_GENERATOR_PERSISTENCE_VERSION}, found v{schema_version}: {path}"
        )
    print(f"Loaded graph generator: {path.name}")
    print(path)
    _restore_loaded_vectorizer_fit_state(graph_generator)
    _restore_loaded_generator_runtime_defaults(graph_generator)
    try:
        from .extensions.demo.pipeline import ensure_demo_feasibility_estimator
    except Exception:
        ensure_demo_feasibility_estimator = None
    if ensure_demo_feasibility_estimator is not None and hasattr(graph_generator, "feasibility_estimator"):
        graph_generator.feasibility_estimator = ensure_demo_feasibility_estimator(
            graph_generator.feasibility_estimator
        )
    return graph_generator
