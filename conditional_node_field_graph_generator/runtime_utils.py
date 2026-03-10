"""Runtime helpers for verbose logging and trainer execution."""

import logging
import sys
import time
import warnings
from functools import wraps

_PACKAGE_LOGGER_NAME = "conditional_node_field_graph_generator"


def _verbosity_level(instance) -> int:
    """Interpret verbosity values as numeric levels, defaulting to 0."""
    if instance is None or not hasattr(instance, "verbose"):
        return 0
    value = getattr(instance, "verbose")
    if isinstance(value, bool):
        return 1 if value else 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1 if value else 0


def get_runtime_logger(name: str | None = None) -> logging.Logger:
    """Return a package logger configured for notebook-visible INFO output."""
    package_logger = logging.getLogger(_PACKAGE_LOGGER_NAME)
    if not getattr(package_logger, "_codex_configured", False):
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
        package_logger.setLevel(logging.INFO)
        package_logger.propagate = True
        package_logger._codex_configured = True  # type: ignore[attr-defined]
    return logging.getLogger(name or _PACKAGE_LOGGER_NAME)


def verbose_log(instance, message: str, level: int = 1, logger_name: str | None = None) -> None:
    """Emit a verbose message through the package logger when the instance level allows it."""
    if _verbosity_level(instance) < level:
        return
    get_runtime_logger(logger_name).info(message)


def timeit(func):
    """Time a method and print timing info when verbose>=3."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        elapsed_time = end_time - start_time
        elapsed_minutes = elapsed_time / 60
        elapsed_hours = elapsed_minutes / 60

        instance = args[0] if args else None
        if _verbosity_level(instance) >= 3:
            class_name = instance.__class__.__name__ if instance else "UnknownClass"
            verbose_log(
                instance,
                f"Class '{class_name}', Function '{func.__name__}' executed in "
                f"{elapsed_time:.2f} seconds ({elapsed_minutes:.2f} minutes, {elapsed_hours:.2f} hours).",
                level=3,
            )

        return result

    return wrapper


def run_trainer_fit(trainer, model, train_loader, val_loader, context: str, ckpt_path: str | None = None) -> None:
    """Run Lightning training while surfacing notebook-hostile SystemExit failures."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The '.*_dataloader' does not have many workers which may be a bottleneck\..*",
            )
            warnings.filterwarnings(
                "ignore",
                message=r"Starting from v1\.9\.0, `tensorboardX` has been removed as a dependency.*",
            )
            trainer.fit(
                model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                ckpt_path=ckpt_path,
            )
    except SystemExit as exc:
        code = exc.code if exc.code is not None else "None"
        argv_preview = " ".join(sys.argv[:5])
        raise RuntimeError(
            f"{context} aborted with SystemExit({code}). "
            "This usually means some code inside the training stack called "
            "CLI-style argument parsing or sys.exit(). "
            f"Current sys.argv starts with: {argv_preview!r}"
        ) from exc
