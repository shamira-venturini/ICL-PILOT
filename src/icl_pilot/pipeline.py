"""Small file-based helpers for the generation/annotation notebook."""
from __future__ import annotations

from datetime import datetime, timezone
import csv
import json
import re
import shutil
import zipfile
from pathlib import Path

from .chat.annotation import sha256_file
from .chat.generated_chat import plan_generated_story_csv, export_chat_plans


def stage_flags(mode: str) -> tuple[bool, bool, bool]:
    routes = {"full": (True, True, True), "generation_only": (True, False, False),
              "from_csv": (False, True, True), "convert_only": (False, True, False),
              "from_chat": (False, False, True), "morphosyntax_only": (False, False, False),
              "prepare_only": (False, False, False), "review_only": (False, False, False)}
    if mode not in routes:
        raise ValueError(f"Unknown pipeline mode: {mode}")
    return routes[mode]


def checked_sources(sources: dict, *, directories: bool = False) -> dict[str, Path]:
    resolved = {}
    for label, value in sources.items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", label):
            raise ValueError(f"Use a simple unique input label: {label!r}")
        path = Path(value).expanduser().resolve()
        if not (path.is_dir() if directories else path.is_file()):
            raise FileNotFoundError(path)
        if path in resolved.values():
            raise ValueError(f"Duplicate source path: {path}")
        resolved[label] = path
    return resolved


def begin_pipeline(output_dir: str | Path, *, mode: str, csvs: dict, chat_dirs: dict,
                   smoke_test: bool = False, resume: bool = False) -> Path:
    if type(smoke_test) is not bool:
        raise ValueError("smoke_test must be a boolean")
    generate, convert, annotate = stage_flags(mode)
    csvs, chat_dirs = checked_sources(csvs), checked_sources(chat_dirs, directories=True)
    if (convert and not generate or mode == "review_only") and not csvs:
        raise ValueError("This mode requires GENERATION_CSVS")
    if (annotate and not convert or mode in {"morphosyntax_only", "prepare_only"}) and not chat_dirs:
        raise ValueError("from_chat/morphosyntax_only/prepare_only mode requires CHAT_INPUT_DIRS")
    out = Path(output_dir).expanduser().resolve()
    if any(source == out or source in out.parents or out in source.parents for source in chat_dirs.values()):
        raise ValueError("Pipeline outputs must be separate from CHAT input trees")
    if resume:
        manifest = json.loads((out / "pipeline_manifest.json").read_text())
        if not generate or manifest["mode"] != mode or manifest["test_only"] != smoke_test:
            raise ValueError("Resume the original generation mode and smoke-test selection.")
        for stage in ("chat_conversion", "annotation"):
            if manifest.get("stages", {}).get(stage, {}).get("status") not in (None, "pending", "skipped"):
                raise ValueError("Generation resume is for interrupted generation. Use from_csv/from_chat in a new run for later stages.")
        if not (out / "generation_checkpoint.json").is_file():
            raise FileNotFoundError("This run has no RNG checkpoint; existing outputs preserved.")
        return out
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "mode": mode, "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "test_only": smoke_test,
                "source_csvs": {key: {"path": str(path), "sha256": sha256_file(path)} for key, path in csvs.items()},
                "source_chat_dirs": {key: str(path) for key, path in chat_dirs.items()},
                "stages": {}, "helper_sha256": sha256_file(Path(__file__))}
    (out / "pipeline_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def record_stage(output_dir: str | Path, stage: str, status: str, **details) -> None:
    path = Path(output_dir) / "pipeline_manifest.json"
    manifest = json.loads(path.read_text())
    event = {"stage": stage, "status": status, "recorded_at_utc": datetime.now(timezone.utc).isoformat(), **details}
    manifest.setdefault("stage_history", []).append(event)
    manifest["stages"][stage] = {"status": status, **details}
    from .generation.recovery import atomic_json
    atomic_json(path, manifest)


def convert_csv_inputs(csvs: dict, output_dir: str | Path, *, roster_csv, segmenter=None) -> dict[str, Path]:
    sources, out = checked_sources(csvs), Path(output_dir)
    if not sources:
        raise ValueError("Select at least one generation CSV")
    seen = set()
    for path in sources.values():
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                key = (row.get("bundle_id"), row.get("target_story"))
                if key in seen:
                    raise ValueError("Overlapping generation rows selected; use cohort CSVs or their aggregate")
                seen.add(key)
    plans = [plan_generated_story_csv(input_csv=path, output_dir=out / "chat" / label,
                                     roster_csv=roster_csv) for label, path in sources.items()]
    results = export_chat_plans(plans, segmenter=segmenter, report_dir=out / "segmentation")
    return {label: result["output_dir"] for label, result in zip(sources, results)}


def annotation_inputs(chat_dirs: dict, stages: list[str]) -> dict[str, Path]:
    if not stages or len(stages) != len(set(stages)) or set(stages) - {"stage1_td", "stage2_sli"}:
        raise ValueError("Select distinct valid generation stages for annotation")
    result = {}
    for label, root in checked_sources(chat_dirs, directories=True).items():
        structured = any((root / name).is_dir() for name in ("stage1_td", "stage2_sli"))
        if structured:
            for stage in stages:
                path = root / stage
                if not path.is_dir():
                    raise FileNotFoundError(f"Selected annotation stage is missing: {path}")
                output_label = f"{label}_{stage}"
                if output_label in result:
                    raise ValueError(f"Annotation output label collision: {output_label}")
                result[output_label] = path
        else:
            if label in result:
                raise ValueError(f"Annotation output label collision: {label}")
            result[label] = root
    if not result or any(not list(path.rglob("*.cha")) for path in result.values()):
        raise ValueError("Each selected annotation input must contain CHAT files")
    paths = list(result.values())
    if any(a == b or a in b.parents or b in a.parents for i, a in enumerate(paths) for b in paths[i + 1:]):
        raise ValueError("Annotation inputs must not overlap")
    return result


def annotation_collection_metadata(chat_dirs, selected, metadata=None):
    """Expand explicit root metadata after stage selection; never guess labels."""
    roots = checked_sources(chat_dirs, directories=True)
    metadata = metadata or {}
    if set(metadata) - set(roots):
        raise ValueError("CHAT_INPUT_METADATA contains unknown collection labels")
    result = {}
    for label, value in selected.items():
        path = Path(value).resolve()
        owners = [name for name, root in roots.items() if path == root or root in path.parents]
        if len(owners) != 1:
            raise ValueError("Selected annotation input has ambiguous source metadata")
        owner = owners[0]
        info = dict(metadata.get(owner, {"source_type": "unspecified"}))
        info["input_collection"] = owner
        relative = path.relative_to(roots[owner]).as_posix()
        if relative in {"stage1_td", "stage2_sli"}:
            info["generation_stage"] = relative
        result[label] = info
    return result


def sync_run(output_dir: str | Path, destination: str | Path) -> Path:
    """Copy artifacts and mirror only checksum-verified annotation pruning."""
    source, target = Path(output_dir).resolve(), Path(destination).resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Persistent copies must be outside the local run tree")
    if not source.is_dir():
        raise FileNotFoundError(source)
    if target.exists() and any(target.iterdir()):
        local, remote = source / "pipeline_manifest.json", target / "pipeline_manifest.json"
        if not local.is_file() or not remote.is_file():
            raise FileExistsError("Persistent destination already contains artifacts from an unverified run")
        before, after = json.loads(remote.read_text()), json.loads(local.read_text())
        identity = ("started_at_utc", "mode", "source_csvs", "source_chat_dirs", "helper_sha256")
        if any(before.get(key) != after.get(key) for key in identity):
            raise FileExistsError("Persistent destination belongs to a different pipeline run")
    pruned = []
    retention = source / "annotation_retention.json"
    if retention.is_file():
        from .chat.retention import verify_retained_annotations, _inside
        pipeline = json.loads((source / "pipeline_manifest.json").read_text())
        stage = pipeline.get("stages", {}).get("annotation_retention", {})
        if stage.get("status") != "complete":
            raise ValueError("Save the completed annotation-retention reference before synchronization")
        record = verify_retained_annotations(stage, run_root=source)
        pruned = record["removed"]
        for item in pruned:
            existing = _inside(target, item["path"])
            if existing.exists() and (not existing.is_file() or sha256_file(existing) != item["sha256"]):
                raise ValueError("Changed persistent intermediate preserved: " + str(existing))
    paths = sorted(source.rglob("*"), key=lambda p: (p.name != "pipeline_manifest.json", str(p)))
    from .generation.recovery import CHECKPOINT, JOURNAL, sync_generation_checkpoint
    has_checkpoint = (source / CHECKPOINT).is_file()
    for path in paths:
        if not path.is_file():
            continue
        if path.name == "generation.lock":
            continue
        if has_checkpoint and path.relative_to(source).as_posix() in (CHECKPOINT, JOURNAL):
            continue
        copied = target / path.relative_to(source)
        copied.parent.mkdir(parents=True, exist_ok=True)
        if copied.is_file() and sha256_file(copied) == sha256_file(path):
            continue
        temporary = copied.with_name(copied.name + ".copying")
        shutil.copy2(path, temporary)
        if sha256_file(temporary) != sha256_file(path):
            raise OSError(f"Persistent copy failed verification: {copied}")
        temporary.replace(copied)
    if has_checkpoint:
        sync_generation_checkpoint(source, target)
    for item in pruned:
        existing = _inside(target, item["path"])
        if existing.is_file():
            if sha256_file(existing) != item["sha256"]:
                raise ValueError("Persistent intermediate changed during synchronization")
            existing.unlink()
    return target


def package_run(output_dir: str | Path) -> Path:
    source = Path(output_dir).resolve()
    paths = [p for p in source.rglob("*") if p.is_file() and "exports" not in p.relative_to(source).parts]
    if not paths:
        raise ValueError("No pipeline artifacts to export")
    if shutil.disk_usage(source).free < sum(p.stat().st_size for p in paths) + 500 * 1024 * 1024:
        raise OSError("Insufficient space for an export archive; persistent files remain available")
    exports = source / "exports"
    exports.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = exports / f"{source.name}_{stamp}.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as handle:
        for path in sorted(paths):
            handle.write(path, arcname=path.relative_to(source).as_posix())
    return archive
