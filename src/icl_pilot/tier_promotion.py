from __future__ import annotations

from pathlib import Path


def _ugra_to_gra(ugra_line: str) -> str:
    prefix = "%gra:\t" if "\t" in ugra_line else "%gra:"
    body = ugra_line.split("\t", 1)[1] if "\t" in ugra_line else ugra_line.split(":", 1)[1]
    converted: list[str] = []
    for token in body.split():
        parts = token.split("|")
        if len(parts) == 3 and parts[2] == "ROOT":
            parts[1] = "0"
            token = "|".join(parts)
        converted.append(token)
    return prefix + " ".join(converted)


def promote_batchalign_tiers(root_dir: str, dry_run: bool = False) -> int:
    root = Path(root_dir).expanduser().resolve()
    changed_files = 0
    promoted_mor = 0
    promoted_gra = 0

    for path in sorted(root.rglob("*.cha")):
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        file_changed = False

        for index, line in enumerate(lines):
            if line.startswith("%umor:"):
                replacement = line.replace("%umor:", "%mor:", 1)
                if index + 2 < len(lines) and lines[index + 2].startswith("%mor:"):
                    if lines[index + 2] != replacement:
                        lines[index + 2] = replacement
                        promoted_mor += 1
                        file_changed = True
            elif line.startswith("%ugra:"):
                replacement = _ugra_to_gra(line)
                if index + 2 < len(lines) and lines[index + 2].startswith("%gra:"):
                    if lines[index + 2] != replacement:
                        lines[index + 2] = replacement
                        promoted_gra += 1
                        file_changed = True

        if file_changed:
            changed_files += 1
            if not dry_run:
                path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        f"{'Would promote' if dry_run else 'Promoted'} %umor/%ugra over legacy %mor/%gra "
        f"in {changed_files} files under {root}"
    )
    print(f"%mor replacements: {promoted_mor}")
    print(f"%gra replacements: {promoted_gra}")
    return 0
