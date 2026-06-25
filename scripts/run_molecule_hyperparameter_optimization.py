#!/usr/bin/env python
"""Run the YAML-driven ZINC molecule hyperparameter optimization workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from conditional_node_field_graph_generator.extensions.demo.zinc_hyperparameter_optimization import (
    load_zinc_hyperparameter_optimization_config,
    run_zinc_hyperparameter_optimization,
    summarize_zinc_hyperparameter_results,
)
from conditional_node_field_graph_generator.runtime_paths import (
    resolve_campaign_artifact_root,
    resolve_notebook_data_root,
    resolve_repo_root,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("notebooks/configs/zinc_molecule_hyperparameter_optimization.yaml"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = resolve_repo_root(Path.cwd())
    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    artifact_root = resolve_campaign_artifact_root(repo_root=repo_root)
    context = {
        "REPO_ROOT": repo_root,
        "ARTIFACT_ROOT": artifact_root,
        "NOTEBOOK_DATA_ROOT": resolve_notebook_data_root(repo_root=repo_root),
        "CHECKPOINT_ROOT": artifact_root / "checkpoints" / "node_field",
        "SAVED_GENERATOR_ROOT": artifact_root / "saved_generators",
    }
    config = load_zinc_hyperparameter_optimization_config(config_path)
    result = run_zinc_hyperparameter_optimization(config, notebook_context=context)
    print(summarize_zinc_hyperparameter_results(result).to_string(index=False))
    print(f"artifact_root: {result['artifact_root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
