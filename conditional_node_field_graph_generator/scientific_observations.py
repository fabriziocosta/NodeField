"""Deterministic training telemetry and scientific observation detection."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping

import pytorch_lightning as pl
import torch


@dataclass(frozen=True)
class ObservationPolicy:
    plateau_window_epochs: int = 8
    plateau_minimum_improvement: float = 0.002
    generalisation_gap_threshold: float = 0.12
    gradient_norm_threshold: float = 1_000.0
    runtime_multiplier: float = 3.0


def _number(value: Any) -> float | int | None:
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return value if math.isfinite(float(value)) else None
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        number = float(value.detach().cpu().item())
        return number if math.isfinite(number) else None
    return None


def _gradient_norm(module: Any) -> float | None:
    squared = 0.0
    found = False
    for parameter in module.parameters():
        gradient = getattr(parameter, "grad", None)
        if gradient is None:
            continue
        found = True
        value = gradient.detach().float()
        squared += float(torch.sum(value * value).cpu().item())
    return math.sqrt(squared) if found else None


def epoch_telemetry(trainer: Any, module: Any) -> dict[str, Any]:
    """Extract a JSON-safe, compact record from Lightning's epoch state."""
    metrics: dict[str, Any] = {}
    callback_metrics = getattr(trainer, "callback_metrics", {}) or {}
    non_finite_metric = False
    for name, value in callback_metrics.items():
        number = _number(value)
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            non_finite_metric = non_finite_metric or not math.isfinite(float(value.detach().cpu().item()))
        elif isinstance(value, (float, int)) and not isinstance(value, bool):
            non_finite_metric = non_finite_metric or not math.isfinite(float(value))
        if number is not None:
            metrics[str(name)] = number
    epoch = int(getattr(trainer, "current_epoch", -1)) + 1
    durations = getattr(module, "_epoch_duration_seconds", []) or []
    duration = float(durations[-1]) if durations else None
    finite = not non_finite_metric and all(math.isfinite(float(value)) for value in metrics.values())
    return {
        "epoch": epoch,
        "metrics": metrics,
        "epoch_duration_seconds": duration,
        "gradient_norm": _gradient_norm(module),
        "finite_metrics": finite,
        "checkpoint_path": getattr(getattr(trainer, "checkpoint_callback", None), "best_model_path", None),
    }


def detect_observations(
    records: list[Mapping[str, Any]],
    *,
    policy: ObservationPolicy | None = None,
) -> list[dict[str, Any]]:
    """Return reproducible observations detected from epoch records."""
    policy = policy or ObservationPolicy()
    if not records:
        return []
    latest = records[-1]
    metrics = latest.get("metrics") or {}
    epoch = latest.get("epoch")
    observations: list[dict[str, Any]] = []

    if not latest.get("finite_metrics", True):
        observations.append(
            {
                "local_id": f"obs_epoch_{int(epoch):04d}_non_finite",
                "type": "non_finite_metric",
                "statement": "At least one training metric became non-finite.",
                "epoch": epoch,
                "measurements": {"metrics": metrics},
                "detection": {"method": "deterministic_rule", "rule": "non_finite_metric"},
                "reliability": 1.0,
            }
        )

    gradient_norm = latest.get("gradient_norm")
    if gradient_norm is not None and float(gradient_norm) > policy.gradient_norm_threshold:
        observations.append(
            {
                "local_id": f"obs_epoch_{int(epoch):04d}_gradient",
                "type": "unstable_gradients",
                "statement": "The gradient norm exceeded the configured stability threshold.",
                "epoch": epoch,
                "measurements": {"gradient_norm": gradient_norm},
                "detection": {
                    "method": "deterministic_rule",
                    "rule": "gradient_norm_above_threshold",
                    "threshold": policy.gradient_norm_threshold,
                },
                "reliability": 0.99,
            }
        )

    train = _number(metrics.get("train_total"))
    validation = _number(metrics.get("val_total"))
    if train is not None and validation is not None:
        gap = float(validation) - float(train)
        if gap >= policy.generalisation_gap_threshold:
            observations.append(
                {
                    "local_id": f"obs_epoch_{int(epoch):04d}_gap",
                    "type": "generalisation_gap",
                    "statement": "Validation loss exceeded training loss by the configured gap threshold.",
                    "epoch": epoch,
                    "measurements": {
                        "training_loss": train,
                        "validation_loss": validation,
                        "generalisation_gap": gap,
                    },
                    "detection": {
                        "method": "deterministic_rule",
                        "rule": "generalisation_gap_above_threshold",
                        "threshold": policy.generalisation_gap_threshold,
                    },
                    "reliability": 0.94,
                }
            )

    window = records[-max(1, int(policy.plateau_window_epochs)):]
    values = [_number((record.get("metrics") or {}).get("val_total")) for record in window]
    values = [float(value) for value in values if value is not None]
    if len(values) >= max(2, int(policy.plateau_window_epochs)):
        improvement = max(values) - min(values)
        if improvement < policy.plateau_minimum_improvement:
            observations.append(
                {
                    "local_id": f"obs_epoch_{int(epoch):04d}_plateau",
                    "type": "validation_plateau",
                    "statement": "Validation loss showed insufficient improvement across the monitoring window.",
                    "epoch": epoch,
                    "measurements": {
                        "window_epochs": len(values),
                        "validation_range": improvement,
                    },
                    "detection": {
                        "method": "deterministic_rule",
                        "rule": "validation_plateau",
                        "threshold": policy.plateau_minimum_improvement,
                    },
                    "reliability": 0.9,
                }
            )

    durations = [
        float(record["epoch_duration_seconds"])
        for record in records[:-1]
        if record.get("epoch_duration_seconds") is not None
    ]
    current_duration = latest.get("epoch_duration_seconds")
    if durations and current_duration is not None and float(current_duration) > policy.runtime_multiplier * median(durations):
        observations.append(
            {
                "local_id": f"obs_epoch_{int(epoch):04d}_runtime",
                "type": "anomalous_runtime",
                "statement": "The latest epoch took substantially longer than recent epochs.",
                "epoch": epoch,
                "measurements": {
                    "epoch_duration_seconds": current_duration,
                    "recent_median_seconds": median(durations),
                },
                "detection": {
                    "method": "deterministic_rule",
                    "rule": "runtime_above_recent_median_multiplier",
                    "threshold": policy.runtime_multiplier,
                },
                "reliability": 0.85,
            }
        )
    return observations


class ScientificMetricsCallback(pl.callbacks.Callback):
    """Persist epoch telemetry and deterministic observations during training."""

    def __init__(
        self,
        telemetry_path: str | Path,
        observations_path: str | Path | None = None,
        policy: ObservationPolicy | None = None,
    ):
        self.telemetry_path = Path(telemetry_path)
        self.observations_path = Path(observations_path) if observations_path else self.telemetry_path.with_name("observations.jsonl")
        self.policy = policy or ObservationPolicy()
        self.records: list[dict[str, Any]] = []

    @staticmethod
    def _append(path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, sort_keys=True) + "\n")

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        record = epoch_telemetry(trainer, pl_module)
        self.records.append(record)
        self._append(self.telemetry_path, record)
        for observation in detect_observations(self.records, policy=self.policy):
            observation["source_telemetry"] = str(self.telemetry_path)
            self._append(self.observations_path, observation)
            if observation["type"] in {"non_finite_metric", "unstable_gradients"}:
                setattr(pl_module, "scientific_stop_reason", observation["type"])
                # Lightning checks this flag after the current validation epoch.
                setattr(trainer, "should_stop", True)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


__all__ = [
    "ObservationPolicy",
    "ScientificMetricsCallback",
    "detect_observations",
    "epoch_telemetry",
    "read_jsonl",
]
