from __future__ import annotations

from pathlib import Path
import re


_INITIAL_PAUSE_PATTERNS = (
    (re.compile(r"^(\*[^:\n]+:\s*)\(\s*\.\s*\)"), r"\1(.)"),
    (re.compile(r"^(\*[^:\n]+:\s*)\(\s*\.\.\s*\)"), r"\1(..)"),
    (re.compile(r"^(\*[^:\n]+:\s*)\(\s*\.\.\.\s*\)"), r"\1(...)"),
)


def normalize_initial_pauses(root_dir: str, dry_run: bool = False) -> int:
    root = Path(root_dir).expanduser().resolve()
    changed_files = 0
    changed_lines = 0

    for path in sorted(root.rglob("*.cha")):
        original_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        updated_lines: list[str] = []
        file_changed = False

        for line in original_lines:
            new_line = line
            for pattern, replacement in _INITIAL_PAUSE_PATTERNS:
                new_line, n = pattern.subn(replacement, new_line, count=1)
                if n:
                    changed_lines += 1
                    file_changed = True
                    break
            updated_lines.append(new_line)

        if file_changed:
            changed_files += 1
            if not dry_run:
                path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")

    print(
        f"{'Would normalize' if dry_run else 'Normalized'} {changed_lines} utterance-initial "
        f"pause markers across {changed_files} files under {root}"
    )
    return 0
