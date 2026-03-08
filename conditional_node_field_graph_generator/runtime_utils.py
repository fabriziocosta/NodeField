"""Runtime helpers for verbose logging and trainer execution."""

import sys
import time
import warnings
from functools import wraps


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
            print(
                f"Class '{class_name}', Function '{func.__name__}' executed in "
                f"{elapsed_time:.2f} seconds ({elapsed_minutes:.2f} minutes, {elapsed_hours:.2f} hours)."
            )

        return result

    return wrapper


def run_trainer_fit(trainer, model, train_loader, val_loader, context: str) -> None:
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
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    except SystemExit as exc:
        code = exc.code if exc.code is not None else "None"
        argv_preview = " ".join(sys.argv[:5])
        raise RuntimeError(
            f"{context} aborted with SystemExit({code}). "
            "This usually means some code inside the training stack called "
            "CLI-style argument parsing or sys.exit(). "
            f"Current sys.argv starts with: {argv_preview!r}"
        ) from exc
