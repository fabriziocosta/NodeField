"""Seeded, pre-evaluation interventions on recurrent state (zero-based steps)."""
from dataclasses import dataclass
from typing import Optional
import math
import torch


@dataclass(frozen=True)
class RecurrentIntervention:
    kind: str = "none"
    step: Optional[int] = None
    every_step: bool = False
    seed: Optional[int] = None
    noise_scale: float = 1.0

    def __post_init__(self):
        if self.kind not in {"none", "reset_hidden", "shuffle_hidden_nodes", "fresh_x_noise", "fresh_x_noise_every_step"}:
            raise ValueError(f"Unsupported recurrent intervention: {self.kind}")
        if self.step is not None and (not isinstance(self.step, int) or self.step < 0):
            raise ValueError("intervention step must be a nonnegative integer")
        if not math.isfinite(self.noise_scale) or self.noise_scale < 0:
            raise ValueError("noise_scale must be finite and nonnegative")
        if self.kind not in {"none", "fresh_x_noise_every_step"} and self.step is None and not self.every_step:
            raise ValueError("intervention requires step or every_step=True")

    def active(self, step):
        return self.kind != "none" and (self.every_step or self.kind == "fresh_x_noise_every_step" or self.step == step)


def normalize_interventions(intervention, steps):
    items = [] if intervention is None else ([intervention] if isinstance(intervention, RecurrentIntervention) else list(intervention))
    for item in items:
        if not isinstance(item, RecurrentIntervention):
            raise TypeError("intervention must contain RecurrentIntervention objects")
        if item.step is not None and item.step >= steps:
            raise ValueError("intervention step is outside the sampling rollout")
    return items


def apply_recurrent_intervention(x, h, intervention, step, node_mask=None, *, generator=None):
    """Return new tensors without mutating callers; random draws use a local CPU RNG."""
    if not intervention.active(step):
        return x, h
    if generator is None:
        generator = torch.Generator().manual_seed(intervention.seed if intervention.seed is not None else 0)
    if intervention.kind == "reset_hidden":
        h = torch.zeros_like(h)
    elif intervention.kind == "shuffle_hidden_nodes":
        h = h.clone()
        for b in range(h.shape[0]):
            valid = torch.arange(h.shape[1], device=h.device) if node_mask is None else node_mask[b].nonzero().flatten()
            perm = torch.randperm(len(valid), generator=generator).to(h.device)
            h[b, valid] = h[b, valid[perm]]
    elif intervention.kind.startswith("fresh_x_noise"):
        x = torch.randn(x.shape, generator=generator, dtype=x.dtype).to(x.device) * intervention.noise_scale
        if node_mask is not None:
            x = x * node_mask.unsqueeze(-1)
    if node_mask is not None:
        h = h * node_mask.unsqueeze(-1)
    return x, h
