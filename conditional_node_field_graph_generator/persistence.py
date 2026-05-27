"""General persistence helpers for fitted NodeField objects."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import tempfile
from contextlib import nullcontext

import dill as pickle

from .encoding_pipeline import EncodingPipeline
from .conditioning_sampler import ConditioningSampler
from .naming_utils import sanitize_model_token
from .node_batch_builder import NodeBatchBuilder
from .runtime_paths import resolve_saved_generator_dir as _resolve_saved_generator_dir
from .runtime_utils import get_runtime_logger, run_with_fork_timeout
from .stream_fit import StreamFitService
from .supervision import SupervisionPlanner


GRAPH_GENERATOR_PERSISTENCE_VERSION = 3
logger = get_runtime_logger(__name__)

try:
    from IPython.display import display
except Exception:  # pragma: no cover
    def display(obj):  # type: ignore
        logger.info("%s", obj)


def resolve_saved_generator_dir(model_dir=None):
    return _resolve_saved_generator_dir(model_dir=model_dir)

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
    if not hasattr(graph_generator, "stream_prefetch_batches"):
        graph_generator.stream_prefetch_batches = 2
    if not hasattr(graph_generator, "stream_snapshot_every_n_batches"):
        graph_generator.stream_snapshot_every_n_batches = 10
    if not hasattr(graph_generator, "stream_batch_timeout_seconds"):
        graph_generator.stream_batch_timeout_seconds = 30.0
    if not hasattr(graph_generator, "stream_snapshot_timeout_seconds"):
        graph_generator.stream_snapshot_timeout_seconds = 30.0
    if not hasattr(graph_generator, "stream_pdf_timeout_seconds"):
        graph_generator.stream_pdf_timeout_seconds = 60.0
    if not hasattr(graph_generator, "stream_max_consecutive_stalls"):
        graph_generator.stream_max_consecutive_stalls = 3
    if not hasattr(graph_generator, "use_embedding_svd"):
        graph_generator.use_embedding_svd = False
    if not hasattr(graph_generator, "node_embedding_svd_dimension"):
        graph_generator.node_embedding_svd_dimension = 256
    if not hasattr(graph_generator, "graph_embedding_svd_dimension"):
        graph_generator.graph_embedding_svd_dimension = None
    for attr_name, default_value in (
        ("node_embedding_svd_", None),
        ("graph_embedding_svd_", None),
        ("node_embedding_svd_fitted_", False),
        ("graph_embedding_svd_fitted_", False),
        ("node_embedding_raw_dimension_", None),
        ("graph_embedding_raw_dimension_", None),
        ("node_embedding_effective_dimension_", None),
        ("graph_embedding_effective_dimension_", None),
    ):
        if not hasattr(graph_generator, attr_name):
            setattr(graph_generator, attr_name, default_value)
    if not hasattr(graph_generator, "encoding_pipeline_"):
        graph_generator.encoding_pipeline_ = EncodingPipeline(graph_generator)
    if not hasattr(graph_generator, "supervision_planner_"):
        graph_generator.supervision_planner_ = SupervisionPlanner(graph_generator)
    if not hasattr(graph_generator, "node_batch_builder_"):
        graph_generator.node_batch_builder_ = NodeBatchBuilder(graph_generator)
    if not hasattr(graph_generator, "conditioning_sampler_"):
        graph_generator.conditioning_sampler_ = ConditioningSampler(graph_generator)
    if not hasattr(graph_generator, "stream_fit_service_"):
        graph_generator.stream_fit_service_ = StreamFitService(graph_generator)
    graph_decoder = getattr(graph_generator, "graph_decoder", None)
    if graph_decoder is not None:
        if not hasattr(graph_decoder, "parallel_decode_timeout_seconds"):
            graph_decoder.parallel_decode_timeout_seconds = 30.0
        if not hasattr(graph_decoder, "active_time_limit_seconds"):
            graph_decoder.active_time_limit_seconds = None
    if getattr(graph_generator, "model_name", None) is not None:
        graph_generator.model_name = sanitize_model_token(graph_generator.model_name)


def _atomic_pickle_dump_worker(graph_generator, output_path: str) -> None:
    _atomic_pickle_dump(graph_generator, Path(output_path))


def _save_graph_generator_loss_curves_pdf_worker(graph_generator, output_path: str, log: bool) -> None:
    _save_graph_generator_loss_curves_pdf(graph_generator, output_path=Path(output_path), log=log)


def _save_graph_generator_loss_curves_pdf(graph_generator, *, output_path: Path, log: bool) -> None:
    exporter = getattr(graph_generator, "export_metrics_pdf", None)
    if not callable(exporter):
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=output_path.suffix,
        prefix=f".{output_path.stem}.",
        dir=output_path.parent,
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
    try:
        saved_path = exporter(str(temp_path))
    except Exception as exc:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass
        logger.warning("Unable to save graph generator loss curves PDF %s: %s", output_path, exc)
        return
    if saved_path is None:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass
        return
    saved_path = Path(saved_path)
    if saved_path != temp_path:
        os.replace(saved_path, temp_path)
    os.replace(temp_path, output_path)
    if log:
        logger.info("Saved graph generator loss curves as: %s", Path(saved_path).name)
        logger.info("%s", output_path)


def _atomic_pickle_dump(obj, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=output_path.suffix,
            prefix=f".{output_path.stem}.",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            pickle.dump(obj, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, output_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def save_graph_generator(
    graph_generator,
    model_name=None,
    model_dir=None,
    log=True,
    save_loss_curves_pdf=True,
):
    resolved_model_name = model_name if model_name is not None else getattr(graph_generator, "model_name", None)
    if resolved_model_name is None:
        logger.info("Skipping graph generator save because model_name is None.")
        return None
    resolved_model_dir = model_dir if model_dir is not None else getattr(graph_generator, "model_dir", None)
    model_root = resolve_saved_generator_dir(model_dir=resolved_model_dir)
    graph_generator._persistence_schema_version = GRAPH_GENERATOR_PERSISTENCE_VERSION
    stem = sanitize_model_token(resolved_model_name)
    filename = f"{stem}.pkl"
    path = model_root / filename
    pdf_path = model_root / f"{stem}.loss-curves.pdf"
    lock_factory = getattr(graph_generator, "_ensure_stream_runtime_lock", None)
    lock_context = lock_factory() if callable(lock_factory) else nullcontext()
    with lock_context:
        snapshot_timeout_seconds = getattr(graph_generator, "stream_snapshot_timeout_seconds", None)
        pdf_timeout_seconds = getattr(graph_generator, "stream_pdf_timeout_seconds", None)
        try:
            run_with_fork_timeout(
                _atomic_pickle_dump_worker,
                graph_generator,
                str(path),
                timeout_seconds=snapshot_timeout_seconds,
            )
        except TimeoutError:
            logger.warning("Timed out while saving graph generator snapshot %s; skipping snapshot.", path)
            return None
        except Exception as exc:
            logger.warning("Unable to save graph generator snapshot %s: %s", path, exc)
            return None
        if save_loss_curves_pdf:
            try:
                run_with_fork_timeout(
                    _save_graph_generator_loss_curves_pdf_worker,
                    graph_generator,
                    str(pdf_path),
                    log,
                    timeout_seconds=pdf_timeout_seconds,
                )
            except TimeoutError:
                logger.warning("Timed out while saving graph generator loss curves PDF %s; skipping PDF.", pdf_path)
            except Exception as exc:
                logger.warning("Unable to save graph generator loss curves PDF %s: %s", pdf_path, exc)
    if log:
        logger.info("Saved graph generator as: %s", filename)
        logger.info("%s", path)
    return filename


def list_saved_graph_generators(model_dir=None):
    import pandas as pd

    model_root = resolve_saved_generator_dir(model_dir=model_dir)
    files = sorted(model_root.glob("*.pkl"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not files:
        logger.info("No saved graph generators found in %s", model_root)
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
        sanitized_stem = sanitize_model_token(requested[:-4] if requested.endswith(".pkl") else requested)
        names_to_try.add(f"{sanitized_stem}.pkl")
        for candidate_name in names_to_try:
            candidate_path = model_root / candidate_name
            if candidate_path.is_file():
                candidates.append(candidate_path.resolve())
        if not candidates:
            pattern = requested[:-4] if requested.endswith(".pkl") else requested
            matches = sorted(model_root.glob(f"{pattern}*.pkl"))
            if not matches and sanitized_stem != pattern:
                matches = sorted(model_root.glob(f"{sanitized_stem}*.pkl"))
            candidates = [path.resolve() for path in matches]
    if not candidates:
        available = sorted(path.name for path in model_root.glob("*.pkl"))
        available_suffix = ""
        if available:
            preview = ", ".join(available[:5])
            if len(available) > 5:
                preview += ", ..."
            available_suffix = f" Available saved generators: {preview}"
        raise FileNotFoundError(
            f"Could not find a saved graph generator matching {requested!r} in {model_root}."
            f"{available_suffix}"
        )
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
    logger.info("Loaded graph generator: %s", path.name)
    logger.info("%s", path)
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
