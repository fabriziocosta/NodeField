"""Formatting helpers for feasibility retry reporting."""

from __future__ import annotations


def format_elapsed_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m {seconds:04.1f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:d}h {minutes:02d}m {seconds:04.1f}s"


def format_feasibility_attempt_status(
    *,
    attempt: int,
    max_attempts: int,
    attempted_total: int,
    feasible_now: int,
    filled_now: int,
    pending_now: int,
    acceptance_rate: float,
    filled_total: int,
    missing_total: int,
    attempt_elapsed_seconds: float,
    total_elapsed_seconds: float,
) -> str:
    remaining_attempts = max(0, int(max_attempts) - int(attempt))
    eta_seconds = 0.0 if pending_now <= 0 else (total_elapsed_seconds / max(1, attempt)) * remaining_attempts
    return (
        f"Feasibility attempt {attempt:>2}/{max_attempts:<2} | "
        f"generated={attempted_total:>4} | "
        f"feasible_candidates={feasible_now:>2} | "
        f"fulfilled_slots={filled_now:>2} | "
        f"pending_slots={pending_now:>2} | "
        f"feasible_rate={acceptance_rate:>6.1%} | "
        f"fulfilled_total={filled_total:>2} | "
        f"missing_total={missing_total:>2} | "
        f"attempt_time={format_elapsed_seconds(attempt_elapsed_seconds):>8} | "
        f"eta={format_elapsed_seconds(eta_seconds):>8}"
    )
