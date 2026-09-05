"""Demo-oriented helpers for notebook workflows.

Exports are loaded lazily so importing a demo submodule does not require every
optional notebook/persistence dependency up front.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "apply_exact_trial_patch": "...nodefield_campaign",
    "benchmark_regression_guidance": ".pipeline",
    "build_campaign_trial_artificial_plotter": ".campaign_best_model",
    "build_dataset": ".pipeline",
    "build_graph_generator": ".pipeline",
    "build_zinc_dataset": ".pipeline",
    "campaign_status": "...nodefield_campaign",
    "collect_campaign_trial_results": ".campaign_best_model",
    "collect_oracle_trace_rows": ".oracle",
    "compare_real_vs_generated": ".visualization",
    "describe_resume_checkpoint": ".storage",
    "discover_latest_campaign_state": ".campaign_dashboard",
    "display_latest_campaign_dashboard": ".campaign_dashboard",
    "draw_artificial_graphs": ".artificial",
    "find_latest_checkpoint": ".storage",
    "find_latest_campaign_state": ".campaign_best_model",
    "fit_graph_generator": ".pipeline",
    "format_campaign_status": "...nodefield_campaign",
    "force_restart_campaign": "...nodefield_campaign",
    "infer_display_mode": ".visualization",
    "list_campaigns": "...nodefield_campaign",
    "list_saved_graph_generators": "...persistence",
    "list_training_checkpoints": ".storage",
    "load_artificial_hyperparameter_optimization_config": ".artificial_hyperparameter_optimization",
    "load_campaign_config": "...nodefield_campaign",
    "load_campaign_trial_generator": ".campaign_best_model",
    "load_campaign_trial_training_examples": ".campaign_best_model",
    "load_graph_generator": "...persistence",
    "load_zinc_hyperparameter_optimization_config": ".zinc_hyperparameter_optimization",
    "offset_neg_graphs": ".visualization",
    "oracle_trace_frame": ".oracle",
    "parse_oracle_trace_title": ".oracle",
    "plot_label_histogram_alignment": ".visualization",
    "plot_networkx_graphs": ".visualization",
    "plot_sample": ".visualization",
    "plot_similarity_distribution_with_iqr": ".visualization",
    "prepare_experiment": ".pipeline",
    "prepare_zinc_data_split": ".pipeline",
    "resolve_campaign_config": "...nodefield_campaign",
    "run_artificial_hyperparameter_optimization": ".artificial_hyperparameter_optimization",
    "run_campaign_once": "...nodefield_campaign",
    "run_label_histogram_analysis": ".visualization",
    "run_zinc_hyperparameter_optimization": ".zinc_hyperparameter_optimization",
    "sample_hyperparameter_configuration": ".pipeline",
    "sample_from_best_campaign_trial": ".campaign_best_model",
    "sample_positive_endpoint_pair": ".visualization",
    "score_graph_generator_feasible_rate": ".pipeline",
    "select_best_campaign_trial": ".campaign_best_model",
    "save_graph_generator": "...persistence",
    "select_pos_neg": ".visualization",
    "show_molecules": ".visualization",
    "show_samples": ".visualization",
    "summarize_artificial_hyperparameter_results": ".artificial_hyperparameter_optimization",
    "summarize_label_histogram_alignment": ".visualization",
    "summarize_zinc_hyperparameter_results": ".zinc_hyperparameter_optimization",
    "terminate_campaign": "...nodefield_campaign",
    "upsert_logbook_block": "...nodefield_campaign",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load demo exports on first access."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, package=__name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
