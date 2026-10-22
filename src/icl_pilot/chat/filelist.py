from __future__ import annotations

from pathlib import Path


def write_cha_filelist(root_dir: str, output_file: str) -> int:
    root = Path(root_dir).expanduser().resolve()
    output = Path(output_file).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {root}")

    files = sorted(path.resolve() for path in root.rglob("*.cha") if path.is_file())
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as handle:
        for path in files:
            handle.write(f"{path}\n")

    print(f"Wrote {len(files)} .cha paths to {output}")
    return 0
