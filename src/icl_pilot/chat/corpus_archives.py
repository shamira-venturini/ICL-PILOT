"""One checksum-verified original ZIP per corpus, with exact-member recovery."""
from __future__ import annotations

import hashlib
from functools import lru_cache
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

from ..generation.recovery import atomic_bytes, atomic_json, timestamp
from .chatter_runtime import sha256_file

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs/annotation/retention.json"


@lru_cache(maxsize=4)
def _check_zip(path, digest):
    # Caller always rechecks the complete ZIP SHA; CRC/name checks can be reused
    # for that identical byte sequence without repeatedly decompressing a corpus.
    with zipfile.ZipFile(path) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)) or bundle.testzip() is not None:
            raise ValueError("Original corpus archive failed integrity checks")
        for item in bundle.infolist():
            relative = PurePosixPath(item.filename)
            if relative.is_absolute() or ".." in relative.parts or "\\" in item.filename or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unsafe original corpus archive member")
    return tuple(names)


def verified_archive(corpus, *, project_root=ROOT, paths=None, config_path=DEFAULT_CONFIG):
    config = json.loads(Path(config_path).read_text())
    if config.get("policy") != "validated-annotation-retention-v1" or config.get("preserve_failed_runs") is not True:
        raise ValueError("Use the validated retention policy with failed-run preservation")
    if corpus not in config["archives"]:
        raise ValueError("Unknown original corpus: " + corpus)
    spec = config["archives"][corpus]
    path = Path((paths or {}).get(corpus, Path(project_root) / spec["path"])).expanduser().resolve()
    if not path.is_file() or sha256_file(path) != spec["sha256"]:
        raise ValueError("Original " + corpus + " ZIP is missing or differs from the study checksum: " + str(path))
    _check_zip(str(path), spec["sha256"])
    return {"corpus": corpus, "path": str(path), "sha256": spec["sha256"], "verified_at_utc": timestamp()}


def register_archives(output, *, required=(), project_root=ROOT, paths=None, config_path=DEFAULT_CONFIG):
    config = json.loads(Path(config_path).read_text())
    selected = set(required) | set(paths or {})
    # Register existing canonical ZIPs once by reference; never copy them into runs.
    selected |= {name for name, spec in config["archives"].items() if (Path(project_root) / spec["path"]).is_file()}
    records = {name: verified_archive(name, project_root=project_root, paths=paths, config_path=config_path) for name in sorted(selected)}
    atomic_json(Path(output), {"schema_version": 1, "archives": records})
    return records


@lru_cache(maxsize=4)
def _pid_index(path, digest):
    result = {}
    with zipfile.ZipFile(path) as bundle:
        for item in bundle.infolist():
            if item.filename.endswith(".cha"):
                raw = bundle.read(item)
                pids = re.findall(r"^@PID:[ \t]*(\S+)", raw.decode("utf-8-sig"), re.M)
                for pid in set(pids):
                    result.setdefault(pid, []).append({"member": item.filename, "sha256": hashlib.sha256(raw).hexdigest()})
    return result


def archive_member_for_pid(archive, pid):
    if sha256_file(Path(archive["path"])) != archive["sha256"]:
        raise ValueError("Original archive changed")
    found = _pid_index(archive["path"], archive["sha256"]).get(pid, [])
    if len(found) != 1:
        raise ValueError("Original CHAT PID does not identify one archive member: " + str(pid))
    return found[0]


def read_archive_member(archive, member):
    if sha256_file(Path(archive["path"])) != archive["sha256"]:
        raise ValueError("Original corpus ZIP changed")
    with zipfile.ZipFile(archive["path"]) as bundle:
        raw = bundle.read(member["member"])
    if hashlib.sha256(raw).hexdigest() != member["sha256"]:
        raise ValueError("Original corpus member changed")
    return raw


def restore_original_member(archive, member, destination):
    """Restore only a requested original; never guess edits or overwrite a file."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Restore to a fresh path; existing files are preserved")
    raw = read_archive_member(archive, member)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_bytes(destination, raw)
    return destination


def original_source_bytes(path, project_root):
    """Read an extracted source or its verified canonical ZIP without extraction."""
    path, root = Path(path), Path(project_root).resolve()
    config = json.loads(DEFAULT_CONFIG.read_text())
    for corpus, spec in config["archives"].items():
        transcript_root = root / Path(spec["path"]).parent / "transcripts"
        try:
            member = path.resolve().relative_to(transcript_root.resolve()).as_posix()
        except ValueError:
            continue
        archive = verified_archive(corpus, project_root=root, config_path=DEFAULT_CONFIG)
        with zipfile.ZipFile(archive["path"]) as bundle:
            raw = bundle.read(member)
        if path.is_file() and path.read_bytes() != raw:
            raise ValueError("Expanded original differs from its verified ZIP: " + str(path))
        return raw
    if path.is_file():
        return path.read_bytes()
    raise FileNotFoundError(path)


def original_member_paths(transcript_root, project_root, corpus):
    """Return logical paths without expanding the original ZIP again."""
    folder, root = Path(transcript_root).resolve(), Path(project_root).resolve()
    local = sorted(folder.rglob("*.cha"))
    config = json.loads(DEFAULT_CONFIG.read_text())
    expected = root / Path(config["archives"][corpus]["path"]).parent / "transcripts"
    if folder != expected.resolve():
        if local:
            return local
        raise FileNotFoundError("No CHAT files at selected transcript root: " + str(folder))
    archive = verified_archive(corpus, project_root=root, config_path=DEFAULT_CONFIG)
    with zipfile.ZipFile(archive["path"]) as bundle:
        paths = [folder / name for name in sorted(bundle.namelist()) if name.endswith(".cha")]
    if set(local) - set(paths):
        raise ValueError("Unexpected files in expanded original corpus")
    return paths


def compact_extracted_originals(project_root, *, config_path=DEFAULT_CONFIG, dry_run=False):
    """Remove only byte-identical extracted originals after verifying both ZIPs."""
    root = Path(project_root).resolve()
    config = json.loads(Path(config_path).read_text())
    plan, archives = [], {}
    for corpus, spec in config["archives"].items():
        archive = verified_archive(corpus, project_root=root, config_path=config_path)
        archives[corpus] = archive
        folder = root / Path(spec["path"]).parent / "transcripts"
        if folder.is_symlink() or (folder.exists() and root not in folder.resolve().parents):
            raise ValueError("Expanded originals must be an ordinary folder inside the project")
        with zipfile.ZipFile(archive["path"]) as bundle:
            for path in sorted(folder.rglob("*")):
                if path.is_symlink():
                    raise ValueError("Preserve unexpected symlink in extracted sources")
                if path.is_file():
                    name = path.relative_to(folder).as_posix()
                    raw = bundle.read(name)
                    if path.read_bytes() != raw:
                        raise ValueError("Extracted source differs from its recovery ZIP; preserved: " + str(path))
                    plan.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "corpus": corpus, "member": name})
    if dry_run:
        return {"archives": archives, "verified_files": len(plan), "files": plan}
    receipt = root / "docs/reproducibility/original_archive_retention.json"
    if receipt.exists():
        record = json.loads(receipt.read_text())
        recorded = {item["path"]: item for item in record["removed"]}
        if any(recorded.get(item["path"]) != item for item in plan):
            raise ValueError("Unexpected extracted files after original archive retention; preserved")
        if record.get("status") == "complete" and not plan:
            return receipt
        if record.get("status") not in {"verified_before_removal", "complete"}:
            raise ValueError("Unrecognized source-retention receipt")
    else:
        record = {"schema_version": 1, "status": "verified_before_removal", "archives": archives, "removed": plan}
        atomic_json(receipt, record)
    for item in plan:
        path = Path(item["path"])
        if sha256_file(path) != item["sha256"]:
            raise ValueError("Extracted source changed after archive verification")
        path.unlink()
    for spec in config["archives"].values():
        folder = root / Path(spec["path"]).parent / "transcripts"
        for path in sorted(folder.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()
    record.update(status="complete", completed_at_utc=timestamp())
    atomic_json(receipt, record)
    return receipt
