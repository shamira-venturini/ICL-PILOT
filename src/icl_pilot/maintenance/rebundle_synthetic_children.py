from __future__ import annotations

import argparse
from pathlib import Path

STORY_SLOTS = ["A1", "A2", "A3", "B1", "B2", "B3"]


def _bundle_prefix(lines: list[str]) -> list[str]:
    for index, line in enumerate(lines):
        if line.startswith("@Comment:\tbundled_story_start="):
            return lines[:index]
    raise ValueError("Bundle file is missing bundled_story_start header marker.")


def _story_body(lines: list[str]) -> list[str]:
    start = next((index for index, line in enumerate(lines) if line.startswith("*CHI:")), None)
    if start is None:
        raise ValueError("Story file is missing a *CHI tier.")
    end = next((index for index, line in enumerate(lines[start:], start) if line == "@End"), len(lines))
    return lines[start:end]


def _source_path(bundle_path: Path, slot: str) -> Path:
    if bundle_path.parent.name == "stage1_td":
        suffix = "_TD_bundle.cha"
        source_dir = bundle_path.parents[2] / "stage1_td_reannotated"
        source_suffix = "_TD.cha"
    elif bundle_path.parent.name == "stage2_sli":
        suffix = "_SLI_bundle.cha"
        source_dir = bundle_path.parents[2] / "stage2_sli_reannotated"
        source_suffix = "_SLI.cha"
    else:
        raise ValueError(f"Unrecognized bundle directory for {bundle_path}")

    if not bundle_path.name.endswith(suffix):
        raise ValueError(f"Unexpected bundle filename: {bundle_path.name}")

    stem = bundle_path.name[: -len(suffix)]
    return source_dir / f"{stem}_{slot}{source_suffix}"


def rebundle_directory(root_dir: str | Path) -> int:
    root = Path(root_dir).expanduser().resolve()
    changed_files = 0

    for bundle_path in sorted(root.glob("*.cha")):
        original_lines = bundle_path.read_text(encoding="utf-8", errors="replace").splitlines()
        rebuilt_lines = _bundle_prefix(original_lines)

        for slot in STORY_SLOTS:
            source_path = _source_path(bundle_path, slot)
            source_lines = source_path.read_text(encoding="utf-8", errors="replace").splitlines()
            rebuilt_lines.append(f"@Comment:\tbundled_story_start={slot}")
            rebuilt_lines.extend(_story_body(source_lines))

        rebuilt_lines.append("@End")

        if rebuilt_lines != original_lines:
            bundle_path.write_text("\n".join(rebuilt_lines) + "\n", encoding="utf-8")
            changed_files += 1

    return changed_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild bundled synthetic CHAT files from story-level sources.")
    parser.add_argument("root_dir", nargs="+", help="One or more bundle directories to rebuild.")
    args = parser.parse_args()

    total_changed = 0
    for root_dir in args.root_dir:
        changed = rebundle_directory(root_dir)
        total_changed += changed
        print(f"{root_dir}: rebuilt {changed} files")
    print(f"total_rebuilt={total_changed}")


if __name__ == "__main__":
    main()
