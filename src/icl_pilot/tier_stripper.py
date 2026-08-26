from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path


MORPHOSYNTACTIC_TIERS = ("%mor:", "%gra:", "%umor:", "%ugra:")


def _normalize_roots(
    root_dir: str | Path | Sequence[str | Path],
) -> list[Path]:
    values = [root_dir] if isinstance(root_dir, (str, Path)) else list(root_dir)
    roots = [Path(value).expanduser().resolve() for value in values]
    if not roots:
        raise ValueError("At least one transcript root is required")
    missing = [str(root) for root in roots if not root.is_dir()]
    if missing:
        raise FileNotFoundError(f"Transcript roots do not exist: {', '.join(missing)}")
    if len(set(roots)) != len(roots):
        raise ValueError("Transcript roots must be unique")
    return roots


def _scan_root(root: Path) -> tuple[dict[Path, list[str]], dict[str, object]]:
    plan: dict[Path, list[str]] = {}
    tier_counts = {
        tier.removeprefix("%").removesuffix(":"): 0
        for tier in MORPHOSYNTACTIC_TIERS
    }
    transcript_files = 0
    speaker_lines = 0
    retained_lines = 0

    for path in sorted(root.rglob("*.cha")):
        transcript_files += 1
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        kept: list[str] = []
        file_changed = False

        for line in lines:
            matching_tier = next(
                (tier for tier in MORPHOSYNTACTIC_TIERS if line.startswith(tier)),
                None,
            )
            if matching_tier is not None:
                tier_name = matching_tier.removeprefix("%").removesuffix(":")
                tier_counts[tier_name] += 1
                file_changed = True
                continue
            kept.append(line)
            retained_lines += 1
            if line.startswith("*"):
                speaker_lines += 1

        if file_changed:
            plan[path] = kept

    return plan, {
        "root": str(root),
        "transcript_files": transcript_files,
        "files_with_tiers": len(plan),
        "tier_lines": tier_counts,
        "tier_lines_total": sum(tier_counts.values()),
        "retained_lines": retained_lines,
        "speaker_lines": speaker_lines,
    }


def strip_tiers(
    root_dir: str | Path | Sequence[str | Path],
    dry_run: bool = False,
    summary_json: str | Path | None = None,
) -> int:
    """Remove CHAT morphosyntactic tiers while retaining every other line."""
    roots = _normalize_roots(root_dir)
    plans: list[tuple[dict[Path, list[str]], dict[str, object]]] = []
    for root in roots:
        plans.append(_scan_root(root))

    if not dry_run:
        for plan, _ in plans:
            for path, kept in plan.items():
                path.write_text("\n".join(kept) + "\n", encoding="utf-8")

    root_summaries: list[dict[str, object]] = []
    for root, (plan, before) in zip(roots, plans, strict=True):
        after_plan, after = _scan_root(root) if not dry_run else (plan, before)
        root_summary = {
            **before,
            "files_changed": 0 if dry_run else len(plan),
            "files_would_change": len(plan) if dry_run else 0,
            "tier_lines_remaining": (
                before["tier_lines_total"] if dry_run else after["tier_lines_total"]
            ),
            "transcript_files_after": after["transcript_files"],
            "retained_lines_after": after["retained_lines"],
            "speaker_lines_after": after["speaker_lines"],
        }
        if not dry_run and after_plan:
            raise RuntimeError(f"Tier removal verification failed under {root}")
        if not dry_run and (
            before["transcript_files"] != after["transcript_files"]
            or before["retained_lines"] != after["retained_lines"]
            or before["speaker_lines"] != after["speaker_lines"]
        ):
            raise RuntimeError(f"Non-tier content verification failed under {root}")
        root_summaries.append(root_summary)

    summary = {
        "operation": "remove_morphosyntactic_tiers",
        "dry_run": dry_run,
        "tiers": [tier.removesuffix(":") for tier in MORPHOSYNTACTIC_TIERS],
        "roots": root_summaries,
        "totals": {
            "transcript_files": sum(int(item["transcript_files"]) for item in root_summaries),
            "files_with_tiers": sum(int(item["files_with_tiers"]) for item in root_summaries),
            "tier_lines": {
                tier.removeprefix("%").removesuffix(":"): sum(
                    int(item["tier_lines"][tier.removeprefix("%").removesuffix(":")])
                    for item in root_summaries
                )
                for tier in MORPHOSYNTACTIC_TIERS
            },
            "tier_lines_total": sum(int(item["tier_lines_total"]) for item in root_summaries),
            "tier_lines_remaining": sum(
                int(item["tier_lines_remaining"]) for item in root_summaries
            ),
            "speaker_lines": sum(int(item["speaker_lines"]) for item in root_summaries),
            "speaker_lines_after": sum(
                int(item["speaker_lines_after"]) for item in root_summaries
            ),
        },
    }

    if summary_json is not None and not dry_run:
        output_path = Path(summary_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    action = "Would remove" if dry_run else "Removed"
    totals = summary["totals"]
    print(
        f"{action} {totals['tier_lines_total']} morphosyntactic tier lines "
        f"from {totals['files_with_tiers']} of {totals['transcript_files']} CHAT files."
    )
    print(json.dumps(totals["tier_lines"], sort_keys=True))
    if not dry_run:
        print(f"Remaining morphosyntactic tier lines: {totals['tier_lines_remaining']}")
    return 0
