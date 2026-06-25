"""Command-line entrypoint for NodeField campaign runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from .nodefield_campaign import (
    format_campaign_status,
    list_campaigns,
    load_campaign_config,
    resolve_campaign_config,
    run_campaign_once,
    terminate_campaign,
    campaign_status,
)
from .runtime_paths import resolve_repo_root


def _load_named_campaign(name: str) -> dict:
    repo_root = resolve_repo_root(Path.cwd())
    config_path = resolve_campaign_config(name, repo_root=repo_root)
    return load_campaign_config(config_path, repo_root=repo_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_nodefield_campaign")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list", help="List configured campaigns.")

    status_parser = subparsers.add_parser("status", help="Show latest campaign status.")
    status_parser.add_argument("campaign", choices=["molecules", "artificial_graphs", "artificial-graphs"])

    terminate_parser = subparsers.add_parser("terminate", help="Mark latest campaign run for termination.")
    terminate_parser.add_argument("campaign", choices=["molecules", "artificial_graphs", "artificial-graphs"])

    for name in ("molecules", "artificial-graphs"):
        run_parser = subparsers.add_parser(name, help=f"Run one {name} campaign batch.")
        run_parser.add_argument("--once", action="store_true", help="Run one mini-batch and exit.")
        run_parser.add_argument("--dry-run", action="store_true", help="Sample and print without executing.")
        run_parser.add_argument("--config", type=Path, help="Override campaign config path.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2

    if args.command == "list":
        for row in list_campaigns():
            marker = "ok" if row["exists"] else "missing"
            print(f"{row['campaign']}\t{marker}\t{row['config_path']}")
        return 0

    if args.command == "status":
        config = _load_named_campaign(args.campaign)
        print(format_campaign_status(campaign_status(config)))
        return 0

    if args.command == "terminate":
        config = _load_named_campaign(args.campaign)
        result = terminate_campaign(config)
        print(format_campaign_status({**result, "queued_trials": [], "latest_metrics": {}}))
        return 0

    campaign_name = "artificial_graphs" if args.command == "artificial-graphs" else args.command
    repo_root = resolve_repo_root(Path.cwd())
    config_path = args.config or resolve_campaign_config(campaign_name, repo_root=repo_root)
    config = load_campaign_config(config_path, repo_root=repo_root)
    if not args.once:
        print("Only --once execution is currently supported.")
        return 2
    result = run_campaign_once(config, dry_run=bool(args.dry_run))
    print(f"run_dir: {result['run_dir']}")
    print(f"queued_trials: {len(result['proposal']['sampled_patches'])}")
    print(f"status: {result['state']['status']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
