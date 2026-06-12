"""General persistence helpers for fitted NodeField objects."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import shutil
import tempfile
from contextlib import nullcontext

import dill as pickle
import torch

from .encoding_pipeline import EncodingPipeline
from .conditioning_sampler import ConditioningSampler
from .decode_service import DecodeService
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
    node_generator = getattr(graph_generator, "conditional_node_generator_model", None)
    if node_generator is not None and not hasattr(node_generator, "locality_horizon_"):
        node_generator.locality_horizon_ = int(getattr(graph_generator, "locality_horizon", 1))
    if not hasattr(graph_generator, "feasibility_oracle_candidates_per_attempt"):
        default_oracle_candidates = int(
            getattr(graph_generator, "_DEFAULT_FEASIBILITY_ORACLE_CANDIDATES_PER_ATTEMPT", 2)
        )
        graph_generator.feasibility_oracle_candidates_per_attempt = default_oracle_candidates
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
    if not hasattr(graph_generator, "max_decode_seconds_per_sample"):
        graph_generator.max_decode_seconds_per_sample = None
    if not hasattr(graph_generator, "max_decode_attempts_per_sample"):
        graph_generator.max_decode_attempts_per_sample = 1
    if not hasattr(graph_generator, "use_embedding_svd"):
        graph_generator.use_embedding_svd = False
    if not hasattr(graph_generator, "node_embedding_svd_dimension"):
        graph_generator.node_embedding_svd_dimension = 256
    if not hasattr(graph_generator, "graph_embedding_svd_dimension"):
        graph_generator.graph_embedding_svd_dimension = None
    if not hasattr(graph_generator, "embedding_svd_fit_max_rows"):
        graph_generator.embedding_svd_fit_max_rows = None
    if not hasattr(graph_generator, "embedding_svd_fit_random_state"):
        graph_generator.embedding_svd_fit_random_state = 0
    if not hasattr(graph_generator, "embedding_svd_transform_batch_size"):
        graph_generator.embedding_svd_transform_batch_size = None
    if not hasattr(graph_generator, "embedding_svd_n_iter"):
        graph_generator.embedding_svd_n_iter = 2
    if not hasattr(graph_generator, "embedding_svd_n_oversamples"):
        graph_generator.embedding_svd_n_oversamples = 5
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
    if (
        not hasattr(graph_generator, "decode_service_")
        or getattr(graph_generator.decode_service_, "owner", None) is not graph_generator
    ):
        graph_generator.decode_service_ = DecodeService(graph_generator)
    graph_decoder = getattr(graph_generator, "graph_decoder", None)
    if graph_decoder is not None:
        for attr_name, default_value in (
            ("adjacency_time_limit_seconds", 60.0),
            ("parallel_decode_timeout_seconds", 30.0),
            ("active_time_limit_seconds", None),
            ("solver_threads", None),
            ("use_horizon_ilp_constraints", True),
            ("horizon_constraint_weight", 2.0),
            ("horizon_positive_threshold", 0.8),
            ("horizon_negative_threshold", 0.2),
            ("horizon_pair_budget", 24),
            ("horizon_paths_per_pair", 8),
            ("horizon_max_iterations", 1),
        ):
            if not hasattr(graph_decoder, attr_name):
                setattr(graph_decoder, attr_name, default_value)
    if getattr(graph_generator, "model_name", None) is not None:
        graph_generator.model_name = sanitize_model_token(graph_generator.model_name)


def _atomic_pickle_dump_worker(graph_generator, output_path: str) -> None:
    _atomic_pickle_dump(graph_generator, Path(output_path))


def _save_graph_generator_loss_curves_pdf_worker(graph_generator, output_path: str, log: bool) -> None:
    _save_graph_generator_loss_curves_pdf(graph_generator, output_path=Path(output_path), log=log)


def _copy_file_contents(source_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(source_path, "rb") as source, open(output_path, "wb") as target:
        shutil.copyfileobj(source, target)


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
    _copy_file_contents(temp_path, output_path)
    temp_path.unlink(missing_ok=True)
    if log:
        logger.info("Saved graph generator loss curves as: %s", output_path.name)
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


def _iter_snapshot_modules(graph_generator):
    """Yield known torch modules whose tensors should be CPU-backed for snapshots."""
    seen = set()

    def _yield_module(candidate):
        if isinstance(candidate, torch.nn.Module) and id(candidate) not in seen:
            seen.add(id(candidate))
            yield candidate

    yield from _yield_module(getattr(graph_generator, "model", None))
    yield from _yield_module(getattr(graph_generator, "guidance_predictor_", None))

    node_generator = getattr(graph_generator, "conditional_node_generator_model", None)
    if node_generator is not None:
        yield from _yield_module(getattr(node_generator, "model", None))
        yield from _yield_module(getattr(node_generator, "guidance_predictor_", None))


def _module_device(module: torch.nn.Module):
    for tensor in list(module.parameters(recurse=True)) + list(module.buffers(recurse=True)):
        return tensor.device
    return None


def _snapshot_to_cpu_and_dump(graph_generator, output_path: Path) -> None:
    modules = list(_iter_snapshot_modules(graph_generator))
    original_devices = [(module, _module_device(module)) for module in modules]
    original_owner_devices = []
    for owner in (
        graph_generator,
        getattr(graph_generator, "conditional_node_generator_model", None),
    ):
        if owner is not None and hasattr(owner, "device"):
            original_owner_devices.append((owner, getattr(owner, "device")))
    try:
        for module, device in original_devices:
            if device is not None and device.type != "cpu":
                module.to("cpu")
        for owner, _device in original_owner_devices:
            owner.device = torch.device("cpu")
        _atomic_pickle_dump(graph_generator, output_path)
    finally:
        for owner, device in original_owner_devices:
            owner.device = device
        for module, device in original_devices:
            if device is not None and device.type != "cpu":
                module.to(device)


def _looks_like_fork_cuda_initialization_error(exc: Exception) -> bool:
    message = str(exc)
    return (
        "CUDA error: initialization error" in message
        or "cudaErrorInitializationError" in message
        or "Cannot re-initialize CUDA in forked subprocess" in message
    )


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
            if not _looks_like_fork_cuda_initialization_error(exc):
                logger.warning("Unable to save graph generator snapshot %s: %s", path, exc)
                return None
            try:
                _snapshot_to_cpu_and_dump(graph_generator, path)
            except Exception as fallback_exc:
                logger.warning("Unable to save graph generator snapshot %s: %s", path, fallback_exc)
                return None
            logger.debug(
                "Saved graph generator snapshot %s with CPU fallback after forked CUDA initialization failed.",
                path,
            )
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
    display_frame = frame.style.set_properties(
        subset=["name"],
        **{
            "max-width": "none",
            "white-space": "nowrap",
            "overflow": "visible",
            "text-overflow": "clip",
        },
    )
    with pd.option_context("display.max_colwidth", None):
        display(display_frame)
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
