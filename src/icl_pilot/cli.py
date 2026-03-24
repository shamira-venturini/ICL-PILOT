from __future__ import annotations

import argparse

from .manifest import load_counterbalance_rules, load_generation_manifest
from .paths import expected_layout


def _cmd_validate_layout() -> int:
    missing = [f"{name}: {path}" for name, path in expected_layout().items() if not path.exists()]
    if missing:
        print("Missing layout entries:")
        for item in missing:
            print(f"- {item}")
        return 1

    print("Layout OK")
    return 0


def _cmd_show_layout() -> int:
    for name, path in expected_layout().items():
        print(f"{name}: {path}")
    return 0


def _cmd_show_counterbalance() -> int:
    for rule in load_counterbalance_rules():
        print(
            f"{rule.target_set} {rule.target_story}: "
            f"E1={rule.e1_story} E2={rule.e2_story}"
        )
    return 0


def _cmd_show_manifest(limit: int | None) -> int:
    rows = load_generation_manifest()
    if limit is not None:
        rows = rows[:limit]

    for row in rows:
        print(
            f"{row.cohort} | target={row.target_story} ({row.target_subject_ids}, {row.target_age}) "
            f"| E1={row.e1_story} x{row.td_baselines} "
            f"| E2={row.e2_story} ({row.e2_subject_ids}, {row.e2_subject_age}) "
            f"| DLD outputs={row.dld_outputs}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Utilities for the cleaned ICL-PILOT scaffold.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("validate-layout")
    subparsers.add_parser("show-layout")
    subparsers.add_parser("show-counterbalance")

    manifest_parser = subparsers.add_parser("show-manifest")
    manifest_parser.add_argument("--limit", type=int, default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate-layout":
        return _cmd_validate_layout()
    if args.command == "show-layout":
        return _cmd_show_layout()
    if args.command == "show-counterbalance":
        return _cmd_show_counterbalance()
    if args.command == "show-manifest":
        return _cmd_show_manifest(args.limit)

    parser.error(f"Unknown command: {args.command}")
    return 2
