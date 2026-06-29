"""Command-line entrypoint for NodeField campaign runs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from datetime import datetime

from .nodefield_campaign import (
    format_campaign_status,
    list_campaigns,
    load_campaign_config,
    resolve_campaign_config,
    run_campaign_loop,
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
    parser = argparse.ArgumentParser(
        prog="run_nodefield_campaign",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  ./run_nodefield_campaign list\n"
            "  ./run_nodefield_campaign run artificial-graphs-small --once --dry-run\n"
            "  ./run_nodefield_campaign force-restart artificial-graphs-large\n"
            "  ./run_nodefield_campaign status molecules-small\n"
            "  ./run_nodefield_campaign terminate molecules-large"
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list", help="List configured campaigns.")

    run_parser = subparsers.add_parser("run", help="Start or resume a campaign agent loop.")
    run_parser.add_argument("campaign", metavar="campaign")
    run_parser.add_argument(
        "--once",
        action="store_true",
        help="Run one campaign loop tick and exit.",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect status without launching jobs, calling OpenAI, or mutating files.",
    )
    run_parser.add_argument(
        "--force-restart",
        action="store_true",
        help="Terminate stale/running campaign state and start a clean new run.",
    )
    run_parser.add_argument(
        "--device",
        choices=("cpu", "auto", "cuda"),
        default="cpu",
        help="Training device policy. Defaults to cpu; use auto/cuda to allow CUDA.",
    )
    run_parser.add_argument("--config", type=Path, help="Override campaign config path.")

    mini_batch_parser = subparsers.add_parser(
        "run-mini-batch",
        help="Run one deterministic campaign mini-batch (internal).",
    )
    mini_batch_parser.add_argument("campaign", metavar="campaign")
    mini_batch_parser.add_argument("--config", type=Path, required=True)
    mini_batch_parser.add_argument("--run-timestamp", required=True)
    mini_batch_parser.add_argument("--run-id", required=True)
    mini_batch_parser.add_argument(
        "--device",
        choices=("cpu", "auto", "cuda"),
        default="cpu",
    )
    mini_batch_parser.add_argument("--dry-run", action="store_true")

    status_parser = subparsers.add_parser("status", help="Show latest campaign status.")
    status_parser.add_argument("campaign", metavar="campaign")

    terminate_parser = subparsers.add_parser(
        "terminate",
        help="Mark latest campaign run for termination.",
    )
    terminate_parser.add_argument("campaign", metavar="campaign")

    force_restart_parser = subparsers.add_parser(
        "force-restart",
        help="Terminate stale/running state and start a clean new campaign run.",
    )
    force_restart_parser.add_argument("campaign", metavar="campaign")
    force_restart_parser.add_argument(
        "--once",
        action="store_true",
        help="Launch the clean run and exit instead of monitoring it.",
    )
    force_restart_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect status without launching jobs or mutating files.",
    )
    force_restart_parser.add_argument(
        "--device",
        choices=("cpu", "auto", "cuda"),
        default="cpu",
        help="Training device policy. Defaults to cpu; use auto/cuda to allow CUDA.",
    )
    force_restart_parser.add_argument("--config", type=Path, help="Override campaign config path.")

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

    if args.command not in {"run", "run-mini-batch", "force-restart"}:
        parser.error(f"Unsupported command: {args.command}")

    campaign_name = args.campaign
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif args.device == "cuda":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    repo_root = resolve_repo_root(Path.cwd())
    config_path = args.config or resolve_campaign_config(campaign_name, repo_root=repo_root)
    config = load_campaign_config(config_path, repo_root=repo_root)
    if args.command == "run-mini-batch":
        now = datetime.strptime(args.run_timestamp, "%Y%m%d_%H%M%S")
        result = run_campaign_once(
            config,
            dry_run=bool(args.dry_run),
            now=now,
            short_id=args.run_id,
            allow_existing_run_dir=True,
        )
        print(f"run_dir: {result['run_dir']}")
        print(f"queued_trials: {len(result['proposal']['sampled_patches'])}")
        print(f"status: {result['state']['status']}")
        return 0

    run_campaign_loop(
        config,
        campaign_name=campaign_name,
        once=bool(args.once),
        dry_run=bool(args.dry_run),
        force_restart=bool(getattr(args, "force_restart", False) or args.command == "force-restart"),
        device=args.device,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
