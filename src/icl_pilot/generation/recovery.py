"""Atomic call-boundary checkpoints and conservative journal recovery.

One latest RNG snapshot is retained. Every completed output stays in the journal.
The snapshot also contains the last output, so a crash between committing the
checkpoint and appending its output line cannot lose that completed call.
"""
from __future__ import annotations

from datetime import datetime, timezone
import errno
import hashlib
import functools
import json
import os
from pathlib import Path
import tempfile

CHECKPOINT = "generation_checkpoint.json"
JOURNAL = "prompt_records.jsonl"


def single_writer(function):
    """Use an OS advisory lock: a second kernel must not write the same run."""
    @functools.wraps(function)
    def wrapped(prepared, *args, **kwargs):
        path = prepared.run_manifest_path.parent / "generation.lock"
        with path.open("a+b") as handle:
            if os.name == "nt":
                import msvcrt
                if handle.tell() == 0:
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError as error:
                    raise RuntimeError("Another process is writing this generation run.") from error
                try:
                    return function(prepared, *args, **kwargs)
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as error:
                    if error.errno in (errno.EAGAIN, errno.EACCES):
                        raise RuntimeError("Another process is writing this generation run.") from error
                    if error.errno in (errno.ENOSYS, errno.EOPNOTSUPP):
                        print("This filesystem cannot enforce advisory locking; use only one kernel for this run.")
                        return function(prepared, *args, **kwargs)
                    raise
                try:
                    return function(prepared, *args, **kwargs)
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return wrapped


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        # Some mounted filesystems do not support directory fsync.
        try:
            directory = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError:
            pass
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: dict) -> None:
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def event_bytes(record: dict) -> bytes:
    return (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def execution_order(frame) -> list[dict]:
    calls = []
    for job_index, row in enumerate(frame.itertuples(index=False), 1):
        job_id = "/".join(str(getattr(row, name)) for name in ("cohort", "round_id", "bundle_id", "target_story"))
        for stage in ("stage1_td", "stage2_sli"):
            calls.append({"call_index": len(calls) + 1, "job_index": job_index, "job_id": job_id, "stage": stage})
    if len({(call["job_id"], call["stage"]) for call in calls}) != len(calls):
        raise ValueError("Execution schedule contains duplicate jobs.")
    return calls


def order_sha256(calls: list) -> str:
    return hashlib.sha256(json.dumps(calls, sort_keys=True).encode("utf-8")).hexdigest()


def read_checkpoint(folder: Path) -> dict:
    path = folder / CHECKPOINT
    if not path.is_file():
        raise FileNotFoundError("No RNG checkpoint exists. Preserve this run; start a fresh run rather than inventing historical random states.")
    record = json.loads(path.read_text())
    digest = record.pop("checkpoint_sha256", None)
    if digest != hashlib.sha256(json.dumps(record, sort_keys=True).encode("utf-8")).hexdigest():
        raise ValueError("RNG checkpoint checksum failed; existing artifacts preserved.")
    record["checkpoint_sha256"] = digest
    return record


def write_checkpoint(folder: Path, record: dict) -> dict:
    record = {**record, "checkpoint_sha256": hashlib.sha256(json.dumps(record, sort_keys=True).encode("utf-8")).hexdigest()}
    atomic_json(folder / CHECKPOINT, record)
    return record


def recover_journal(folder: Path, checkpoint: dict) -> None:
    """Verify committed prefix, recover last output, retain partial tail evidence."""
    path = folder / JOURNAL
    raw = path.read_bytes()
    offset = checkpoint["journal_prefix_bytes"]
    if len(raw) < offset or hashlib.sha256(raw[:offset]).hexdigest() != checkpoint["journal_prefix_sha256"]:
        raise ValueError("Journal differs from the committed checkpoint prefix; artifacts preserved.")
    event = checkpoint.get("output_event")
    expected = event_bytes(event) if event else b""
    tail = raw[offset:]
    if expected and not tail.startswith(expected):
        if tail and not expected.startswith(tail):
            raise ValueError("Completed output differs from its checkpoint; artifacts preserved.")
        if tail:
            atomic_bytes(folder / "journal_recovery" / (hashlib.sha256(tail).hexdigest() + ".partial"), tail)
        atomic_bytes(path, raw[:offset] + expected)
        return
    # An interrupted prompt write may leave an incomplete final JSONL line.
    if raw and not raw.endswith(b"\n"):
        boundary = raw.rfind(b"\n") + 1
        if boundary < offset + len(expected):
            raise ValueError("Committed journal output is incomplete.")
        tail = raw[boundary:]
        atomic_bytes(folder / "journal_recovery" / (hashlib.sha256(tail).hexdigest() + ".partial"), tail)
        atomic_bytes(path, raw[:boundary])


_mirror_offsets = {}


def sync_generation_checkpoint(source: str | Path, destination: str | Path) -> Path:
    """Append journal changes before publishing their RNG checkpoint, then output.

    Called at every completed call (and initially). Only the appended journal
    bytes and the latest RNG snapshot are transferred, rather than recopying the
    complete prompt history at every call. One process must own each run.
    """
    source, target = Path(source).resolve(), Path(destination).resolve()
    if source == target:
        return target
    if source in target.parents or target in source.parents:
        raise ValueError("Persistent copies must be outside the local run tree")
    checkpoint = read_checkpoint(source)
    target.mkdir(parents=True, exist_ok=True)
    remote = read_checkpoint(target) if (target / CHECKPOINT).exists() else None
    if remote:
        if remote["run_id"] != checkpoint["run_id"] or remote["execution_order_sha256"] != checkpoint["execution_order_sha256"]:
            raise FileExistsError("Persistent checkpoint belongs to a different run.")
        if remote["completed_calls"] not in (checkpoint["completed_calls"], checkpoint["completed_calls"] - 1):
            raise ValueError("Persistent checkpoint is out of sequence; preserved. Use a new destination for a full copy.")
        if remote["completed_calls"] == checkpoint["completed_calls"] and remote != checkpoint:
            raise ValueError("Conflicting checkpoints at the same call; preserved.")
    local, copied = source / JOURNAL, target / JOURNAL
    key = (str(source), str(target))
    size = copied.stat().st_size if copied.exists() else 0
    signature = (size, copied.stat().st_mtime_ns) if copied.exists() else (0, 0)
    verified = _mirror_offsets.get(key)
    if verified != signature:
        if copied.exists():
            with local.open("rb") as handle:
                if handle.read(size) != copied.read_bytes():
                    # Repair only a checkpoint-backed missing/torn last output.
                    if not remote:
                        raise ValueError("Unverified persistent journal; preserved.")
                    recover_journal(target, remote)
                    size = copied.stat().st_size
                    handle.seek(0)
                    if handle.read(size) != copied.read_bytes():
                        raise ValueError("Persistent journal differs; preserved.")
    boundary = checkpoint["journal_prefix_bytes"]
    end = boundary + len(event_bytes(checkpoint["output_event"])) if checkpoint.get("output_event") else boundary
    if size > end:
        # A failed attempt is retained by the normal final copy, but not mirrored
        # ahead of the authoritative checkpoint during a boundary commit.
        if remote != checkpoint:
            raise ValueError("Persistent journal is ahead of its checkpoint; preserved.")
        return target
    with local.open("rb") as reader, copied.open("ab") as writer:
        reader.seek(size)
        if size < boundary:
            remaining = boundary - size
            while remaining:
                chunk = reader.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise OSError("Local journal ended before checkpoint prefix.")
                writer.write(chunk)
                remaining -= len(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        atomic_json(target / CHECKPOINT, checkpoint)
        reader.seek(max(size, boundary))
        remaining = end - max(size, boundary)
        if remaining:
            writer.write(reader.read(remaining))
        writer.flush()
        os.fsync(writer.fileno())
    _mirror_offsets[key] = (copied.stat().st_size, copied.stat().st_mtime_ns)
    return target
