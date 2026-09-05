"""Detached trajectory diagnostics; these describe stability, not convergence."""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class RecurrentNodeFieldState:
    x: torch.Tensor
    h: torch.Tensor


@dataclass
class RecurrentNodeFieldTrajectory:
    # x/h contain the initial state then one state per completed update.
    x: list[torch.Tensor] = field(default_factory=list)
    h: list[torch.Tensor] = field(default_factory=list)
    score: list[torch.Tensor] = field(default_factory=list)
    phi: list[torch.Tensor] = field(default_factory=list)
    sigma: list[float] = field(default_factory=list)
    evaluated_x: list[torch.Tensor] = field(default_factory=list)
    evaluated_h: list[torch.Tensor] = field(default_factory=list)
    diagnostics: list[dict] = field(default_factory=list)
    interventions: list[dict] = field(default_factory=list)
    readouts: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def detached(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: detached(v) for k, v in value.items()}
    return value


def rms(value):
    return float(value.detach().float().square().mean().sqrt().cpu())


def cosine(a, b: Optional[torch.Tensor]):
    if b is None or a.norm() == 0 or b.norm() == 0:
        return None
    return float(
        F.cosine_similarity(
            a.detach().float().flatten(), b.detach().to(a).float().flatten(), dim=0
        ).cpu()
    )


def state_diagnostics(x, h, x_next, h_next, score, phi, previous_score=None):
    return dict(
        hidden_norm=rms(h_next),
        hidden_delta_norm=rms(h_next - h),
        score_norm=rms(score),
        x_delta_norm=rms(x_next - x),
        phi=float(phi.detach().mean().cpu()),
        cosine_hidden_consecutive=cosine(h_next, h),
        cosine_score_consecutive=cosine(score, previous_score),
    )


def prediction_deltas(current, previous):
    result = {}
    for key in ("exist_logits", "degree_logits", "node_label_logits", "edge_probabilities"):
        value = current.get(key)
        old = None if previous is None else previous.get(key)
        if value is None or old is None:
            result[key + "_delta"] = None
            continue
        if key == "exist_logits":
            value, old = value.sigmoid(), old.sigmoid()
        elif key.endswith("logits"):
            value, old = value.softmax(-1), old.softmax(-1)
        result[key + "_delta"] = rms(value - old.to(value))
    defined = [v for v in result.values() if v is not None]
    result["prediction_delta"] = max(defined) if defined else None
    return result
