from __future__ import annotations

import csv
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .batchalign_runner import resolve_batchalign_executable


@dataclass(frozen=True)
class BatchalignBatch:
    batch_index: int
    total_batches: int
    relpaths: tuple[Path, ...]


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _load_manifest_relpaths(input_root: Path, manifest_path: Path) -> list[Path]:
    relpaths: list[Path] = []
    seen: set[Path] = set()
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_path = row.get("output_path", "").strip()
            if not raw_path:
                continue
            candidate = Path(raw_path).resolve()
            if not candidate.exists() or not _is_relative_to(candidate, input_root):
                continue
            relpath = candidate.relative_to(input_root)
            if relpath not in seen:
                relpaths.append(relpath)
                seen.add(relpath)
    return relpaths


def _discover_relpaths(input_root: Path, manifest_csv: str | None = None) -> list[Path]:
    manifest_path = Path(manifest_csv) if manifest_csv else input_root / "chat_manifest.csv"
    if manifest_path.exists():
        relpaths = _load_manifest_relpaths(input_root=input_root, manifest_path=manifest_path)
        if relpaths:
            return relpaths
    return sorted(path.relative_to(input_root) for path in input_root.rglob("*.cha"))


def _batched(relpaths: list[Path], batch_size: int) -> list[BatchalignBatch]:
    total_batches = max(1, math.ceil(len(relpaths) / batch_size))
    batches: list[BatchalignBatch] = []
    for batch_index, start in enumerate(range(0, len(relpaths), batch_size), start=1):
        chunk = tuple(relpaths[start : start + batch_size])
        batches.append(
            BatchalignBatch(
                batch_index=batch_index,
                total_batches=total_batches,
                relpaths=chunk,
            )
        )
    return batches


def _write_batch_manifest(output_root: Path, batches: list[BatchalignBatch]) -> None:
    manifest_path = output_root / "batchalign_batches.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["batch_index", "total_batches", "batch_size", "relative_path"])
        for batch in batches:
            for relpath in batch.relpaths:
                writer.writerow(
                    [
                        batch.batch_index,
                        batch.total_batches,
                        len(batch.relpaths),
                        relpath.as_posix(),
                    ]
                )


def build_batched_morphotag_command(
    input_dir: str,
    output_dir: str,
    keeptokens: bool = True,
    lexicon: str | None = None,
) -> list[str]:
    cmd = [resolve_batchalign_executable(), "morphotag"]
    cmd.append("--keeptokens" if keeptokens else "--retokenize")
    if lexicon:
        cmd.extend(["--lexicon", lexicon])
    cmd.extend([input_dir, output_dir])
    return cmd


def run_morphotag_batched(
    input_dir: str,
    output_dir: str,
    batch_size: int = 200,
    keeptokens: bool = True,
    lexicon: str | None = None,
    manifest_csv: str | None = None,
    start_batch: int = 1,
    max_batches: int | None = None,
    skip_existing: bool = True,
    dry_run: bool = False,
) -> int:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if start_batch < 1:
        raise ValueError("start_batch must be >= 1")

    input_root = Path(input_dir).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    relpaths = _discover_relpaths(input_root=input_root, manifest_csv=manifest_csv)
    if skip_existing:
        relpaths = [
            relpath for relpath in relpaths if not (output_root / relpath).exists()
        ]

    if not relpaths:
        print("No pending .cha files to process.")
        return 0

    batches = _batched(relpaths=relpaths, batch_size=batch_size)
    _write_batch_manifest(output_root=output_root, batches=batches)

    for batch in batches:
        if batch.batch_index < start_batch:
            continue
        if max_batches is not None and batch.batch_index >= start_batch + max_batches:
            break

        print(
            f"[batch {batch.batch_index}/{batch.total_batches}] "
            f"processing {len(batch.relpaths)} transcripts"
        )

        with tempfile.TemporaryDirectory(
            prefix=f"batchalign_{batch.batch_index:04d}_"
        ) as temp_root_str:
            temp_root = Path(temp_root_str)
            temp_input = temp_root / "input"
            temp_output = temp_root / "output"
            temp_input.mkdir(parents=True, exist_ok=True)
            temp_output.mkdir(parents=True, exist_ok=True)

            for relpath in batch.relpaths:
                source_path = input_root / relpath
                batch_input_path = temp_input / relpath
                batch_input_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, batch_input_path)

            cmd = build_batched_morphotag_command(
                input_dir=str(temp_input),
                output_dir=str(temp_output),
                keeptokens=keeptokens,
                lexicon=lexicon,
            )

            print("Command:")
            print(" ".join(cmd))

            if dry_run:
                continue

            completed = subprocess.run(cmd, check=False)
            if completed.returncode != 0:
                return completed.returncode

            for annotated_path in temp_output.rglob("*.cha"):
                destination = output_root / annotated_path.relative_to(temp_output)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(annotated_path), str(destination))

    source_manifest = input_root / "chat_manifest.csv"
    if source_manifest.exists():
        target_manifest = output_root / "source_chat_manifest.csv"
        if not target_manifest.exists():
            shutil.copy2(source_manifest, target_manifest)

    return 0
