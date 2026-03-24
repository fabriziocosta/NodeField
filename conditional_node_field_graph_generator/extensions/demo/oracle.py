"""Oracle-tracing helpers for notebook and demo workflows."""

from __future__ import annotations

import contextlib
import re
from typing import Any, Iterator

import pandas as pd

import conditional_node_field_graph_generator.conditional_node_field_graph_generator as cngg_module


def _extract_numeric(title: str, key: str, *, kind: str) -> int | float | None:
    if kind == "int":
        match = re.search(rf"\b{re.escape(key)}=([-+]?\d+)", title)
        return None if match is None else int(match.group(1))
    match = re.search(rf"\b{re.escape(key)}=([-+]?\d*\.\d+|[-+]?\d+)", title)
    return None if match is None else float(match.group(1))


def parse_oracle_trace_title(title: str) -> dict[str, Any] | None:
    if not str(title).startswith("Oracle "):
        return None
    phase_match = re.match(r"^Oracle\s+(.+?)\s+graph=", title)
    return {
        "phase": phase_match.group(1) if phase_match is not None else "Unknown",
        "iteration": _extract_numeric(title, "iteration", kind="int"),
        "violating_node_sets": _extract_numeric(title, "violating_node_sets", kind="int"),
        "violating_edge_sets": _extract_numeric(title, "violating_edge_sets", kind="int"),
        "new_structural_cuts": _extract_numeric(title, "new_structural_cuts", kind="int"),
        "accepted_structural_cuts": _extract_numeric(title, "accepted_structural_cuts", kind="int"),
        "log_total": _extract_numeric(title, "log_total", kind="float"),
        "log_edge": _extract_numeric(title, "log_edge", kind="float"),
        "log_node": _extract_numeric(title, "log_node", kind="float"),
        "log_edge_label": _extract_numeric(title, "log_edge_label", kind="float"),
        "best_log_total": _extract_numeric(title, "best_log_total", kind="float"),
        "best_feasible_log_total": _extract_numeric(title, "best_feasible_log_total", kind="float"),
    }


@contextlib.contextmanager
def collect_oracle_trace_rows() -> Iterator[list[dict[str, Any]]]:
    """Temporarily collect oracle-trace rows emitted through decoder diagnostics."""
    original_plot = cngg_module._plot_decoder_diagnostics
    rows: list[dict[str, Any]] = []

    def wrapped_plot(**kwargs):
        parsed = parse_oracle_trace_title(kwargs.get("title", ""))
        if parsed is not None:
            rows.append(parsed)
        return original_plot(**kwargs)

    cngg_module._plot_decoder_diagnostics = wrapped_plot
    try:
        yield rows
    finally:
        cngg_module._plot_decoder_diagnostics = original_plot


def oracle_trace_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)
