from __future__ import annotations

from pathlib import Path


def strip_tiers(root_dir: str, dry_run: bool = False) -> int:
    root = Path(root_dir).expanduser().resolve()
    changed_files = 0
    removed_mor = 0
    removed_gra = 0

    for path in sorted(root.rglob("*.cha")):
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        kept: list[str] = []
        file_changed = False

        for line in lines:
            if line.startswith("%mor:"):
                removed_mor += 1
                file_changed = True
                continue
            if line.startswith("%gra:"):
                removed_gra += 1
                file_changed = True
                continue
            kept.append(line)

        if file_changed:
            changed_files += 1
            if not dry_run:
                path.write_text("\n".join(kept) + "\n", encoding="utf-8")

    print(
        f"{'Would strip' if dry_run else 'Stripped'} legacy %mor/%gra tiers from "
        f"{changed_files} files under {root}"
    )
    print(f"%mor removed: {removed_mor}")
    print(f"%gra removed: {removed_gra}")
    return 0
