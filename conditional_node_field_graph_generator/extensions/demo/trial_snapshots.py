"""Helpers for per-trial fitted generator snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...persistence import load_graph_generator, save_graph_generator


TRIAL_GENERATOR_MODEL_NAME = "graph_generator"
TRIAL_GENERATOR_MODEL_DIR = "model"


def save_trial_graph_generator_snapshot(graph_generator: Any, trial_root: str | Path) -> str | None:
    """Persist a fitted trial generator and return the absolute snapshot path."""
    model_dir = Path(trial_root).expanduser().resolve() / TRIAL_GENERATOR_MODEL_DIR
    filename = save_graph_generator(
        graph_generator,
        model_name=TRIAL_GENERATOR_MODEL_NAME,
        model_dir=model_dir,
        log=False,
        save_loss_curves_pdf=False,
    )
    if filename is None:
        return None
    return str((model_dir / filename).resolve())


def trial_graph_generator_snapshot_path(trial_root: str | Path) -> Path:
    """Return the stable per-trial fitted generator snapshot path."""
    return (
        Path(trial_root).expanduser().resolve()
        / TRIAL_GENERATOR_MODEL_DIR
        / f"{TRIAL_GENERATOR_MODEL_NAME}.pkl"
    )


def load_trial_graph_generator_snapshot(snapshot_path: str | Path) -> Any:
    """Load a fitted trial generator snapshot from an explicit path."""
    return load_graph_generator(str(Path(snapshot_path).expanduser().resolve()))


__all__ = [
    "TRIAL_GENERATOR_MODEL_DIR",
    "TRIAL_GENERATOR_MODEL_NAME",
    "load_trial_graph_generator_snapshot",
    "save_trial_graph_generator_snapshot",
    "trial_graph_generator_snapshot_path",
]
