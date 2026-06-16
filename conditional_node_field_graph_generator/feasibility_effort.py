"""Feasibility-effort profiles for generation-time decoding."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class FeasibilityEffortProfile:
    effort: int
    apply_feasibility_filtering: bool
    use_feasibility_oracle: bool
    feasibility_oracle_candidates_per_attempt: int
    max_oracle_iterations: int
    oracle_add_edge_repair_budget: int
    max_feasibility_attempts: int
    feasibility_candidates_per_attempt: int
    max_decode_attempts_per_sample: int
    max_feasibility_seconds_per_sample: Optional[float]

    def to_dict(self) -> dict:
        return asdict(self)


_EFFORT_PROFILES = {
    0: FeasibilityEffortProfile(0, False, False, 0, 1, 0, 1, 1, 1, None),
    1: FeasibilityEffortProfile(1, True, False, 0, 1, 0, 1, 1, 1, 2.0),
    2: FeasibilityEffortProfile(2, True, True, 1, 2, 8, 2, 1, 1, 2.0),
    3: FeasibilityEffortProfile(3, True, True, 2, 5, 16, 8, 3, 2, 9.0),
    4: FeasibilityEffortProfile(4, True, True, 4, 7, 32, 14, 6, 3, 43.0),
    5: FeasibilityEffortProfile(5, True, True, 8, 10, 64, 20, 8, 4, 200.0),
}


def resolve_feasibility_effort(feasibility_effort: int) -> FeasibilityEffortProfile:
    """Return the concrete generation settings for an effort level in [0, 5]."""
    if isinstance(feasibility_effort, bool):
        raise ValueError("feasibility_effort must be an integer from 0 to 5.")
    try:
        effort = int(feasibility_effort)
    except (TypeError, ValueError) as exc:
        raise ValueError("feasibility_effort must be an integer from 0 to 5.") from exc
    if effort != feasibility_effort and not (
        isinstance(feasibility_effort, float) and feasibility_effort.is_integer()
    ):
        raise ValueError("feasibility_effort must be an integer from 0 to 5.")
    if effort not in _EFFORT_PROFILES:
        raise ValueError("feasibility_effort must be an integer from 0 to 5.")
    return _EFFORT_PROFILES[effort]


def feasibility_effort_map() -> Dict[int, dict]:
    """Return a DataFrame-friendly map of effort levels to concrete settings."""
    return {level: profile.to_dict() for level, profile in _EFFORT_PROFILES.items()}


DEFAULT_FEASIBILITY_EFFORT = 5
DEFAULT_FEASIBILITY_EFFORT_PROFILE = _EFFORT_PROFILES[DEFAULT_FEASIBILITY_EFFORT]

