"""TalkTag execution on copied CHAT inputs, with provenance and source QC."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_annotation_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text())
    if config.get("schema_version") != 1 or not re.fullmatch(r"[a-f0-9]{40}", config.get("revision", "")):
        raise ValueError("Annotation settings require schema 1 and a full source commit hash")
    if config.get("repository") != "https://github.com/OliverHennhoefer/talk-tag.git":
        raise ValueError("Unexpected TalkTag source repository")
    if config.get("target_speaker") != "*CHI" or config.get("device") not in {"auto", "cuda", "cpu"}:
        raise ValueError("Use the CHI target and a supported TalkTag device")
    if config.get("granularity") not in {"light", "standard", "strict"}:
        raise ValueError("Invalid TalkTag granularity")
    if type(config.get("limit")) is not int or config["limit"] < 0 or type(config.get("show_target")) is not bool:
        raise ValueError("Invalid annotation limit or show_target setting")
    return config


def install_annotation_runtime(environment: str | Path, config: dict) -> Path:
    """Install in a separate environment using the caller's pip (>=22.3)."""
    if sys.version_info < (3, 10):
        raise RuntimeError("TalkTag requires Python >=3.10; the project helpers still support Python 3.8")
    environment = Path(environment).expanduser().resolve()
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    # Some hosted Python installations lack ensurepip. Skip that bootstrap and
    # let the kernel's pip target the environment explicitly instead. Re-running
    # venv also repairs the configuration left by an interrupted creation; never
    # clear the directory or mistake an existing Python executable for success.
    try:
        subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", "--without-pip",
                        str(environment)], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        details = "\n".join(part for part in (error.stdout, error.stderr) if part)
        raise RuntimeError(f"Could not prepare the TalkTag environment at {environment}:\n{details}") from error
    requirement = f"talk-tag[runtime] @ git+{config['repository']}@{config['revision']}"
    subprocess.run([sys.executable, "-m", "pip", "--python", str(python), "install",
                    requirement, "transformers>=4.52,<5"], check=True)
    return python


def annotation_runtime_metadata(python: str | Path, config: dict) -> dict:
    probe = '''import importlib.metadata as m, json, sys
names = ["talk-tag", "torch", "transformers", "peft", "bitsandbytes", "numpy", "pandas", "huggingface_hub"]
versions = {}
for name in names:
    try: versions[name] = m.version(name)
    except m.PackageNotFoundError: versions[name] = None
direct = json.loads(m.distribution("talk-tag").read_text("direct_url.json") or "{}")
print(json.dumps({"python": sys.version, "packages": versions,
                  "source_commit": direct.get("vcs_info", {}).get("commit_id")}))
'''
    result = subprocess.run([str(python), "-c", probe], check=True, capture_output=True, text=True)
    metadata = json.loads(result.stdout)
    if metadata.get("source_commit") != config["revision"]:
        raise ValueError("Installed TalkTag commit does not match the selected annotation configuration")
    return metadata


def _unfold_target(text: str, target: str) -> str:
    """Join wrapped target main tiers only in the working input copy."""
    result, pending = [], []
    for line in text.splitlines():
        if line.startswith(target + ":"):
            if pending:
                result.append(" ".join(pending))
            pending = [line]
        elif line[:1] in {"@", "%", "*"}:
            if pending:
                result.append(" ".join(pending))
                pending = []
            result.append(line)
        elif pending and line.strip():
            if re.search(r"\S+\|\S+", line):
                raise ValueError("Orphan annotation continuation in a target main tier; prepare a checked CHAT input")
            pending.append(line.strip())
        else:
            result.append(line)
    if pending:
        result.append(" ".join(pending))
    return "\n".join(result) + "\n"


def _target_utterances(text: str, target: str) -> list[str]:
    return [line.split(":", 1)[1].strip() for line in _unfold_target(text, target).splitlines()
            if line.startswith(target + ":")]


def _surface(utterance: str) -> list[str]:
    # Ignore added error/target annotations, retaining repetition, pauses,
    # grammatical forms, punctuation and utterance boundaries for comparison.
    utterance = re.sub(r"\[(?:\*|::?|=)\s+[^\]]*\]", "", utterance)
    return re.findall(r"\w+|[^\w\s]", utterance)


_INLINE_ANNOTATION = re.compile(r"\[(?:\*|::?|=)\s+[^\]]*\]")
_ANNOTATED_TOKEN = re.compile(_INLINE_ANNOTATION.pattern + r"|\w+|[^\w\s]")


def _restore_utterance_surface(original: str, annotated: str) -> str:
    """Keep annotation anchors while restoring source case and token spacing."""
    original_tokens, original_annotations = [], {}
    for match in _ANNOTATED_TOKEN.finditer(original):
        if _INLINE_ANNOTATION.fullmatch(match.group()):
            original_annotations.setdefault(len(original_tokens), []).append(match)
        else:
            original_tokens.append(match)
    annotated_tokens, annotations = [], {}
    for match in _ANNOTATED_TOKEN.finditer(annotated):
        if _INLINE_ANNOTATION.fullmatch(match.group()):
            annotations.setdefault(len(annotated_tokens), []).append(match.group())
        else:
            annotated_tokens.append(match.group())
    if ([match.group().lower() for match in original_tokens]
            != [token.lower() for token in annotated_tokens]):
        raise ValueError("Surface words, punctuation or their order changed beyond capitalization")
    if not original_tokens:
        if original_annotations or annotations:
            raise ValueError("Cannot anchor annotations to an empty utterance")
        return original
    # Insert only new tags into the exact source string. Existing annotations
    # must form an unchanged subsequence at each surface-token boundary.
    insertions = {}
    for anchor in sorted(set(annotations) | set(original_annotations)):
        existing, index = original_annotations.get(anchor, []), 0
        for tag in annotations.get(anchor, []):
            if index < len(existing) and tag == existing[index].group():
                index += 1
                continue
            if index < len(existing):
                position, addition = existing[index].start(), tag + " "
            elif existing:
                position, addition = existing[-1].end(), " " + tag
            elif anchor:
                position, addition = original_tokens[anchor - 1].end(), " " + tag
            else:
                position, addition = original_tokens[0].start(), tag + " "
            insertions.setdefault(position, []).append(addition)
        if index != len(existing):
            raise ValueError("An existing source annotation was removed, changed or moved")
    parts, cursor = [], 0
    for position, additions in sorted(insertions.items()):
        parts.extend([original[cursor:position], "".join(additions)])
        cursor = position
    parts.append(original[cursor:])
    return "".join(parts)


def postprocess_annotated_chat(original: str, annotated: str, *, target: str = "*CHI") -> str:
    """Restore source orthography in a derived copy; never repair grammar."""
    source_lines = [line for line in _unfold_target(original, target).splitlines()
                    if line.startswith(target + ":")]
    output_lines = _unfold_target(annotated, target).splitlines()
    if len(source_lines) != sum(line.startswith(target + ":") for line in output_lines):
        raise ValueError("Target utterance count changed")
    sources, restored = iter(source_lines), []
    for line in output_lines:
        if line.startswith(target + ":"):
            source = next(sources)
            body = _restore_utterance_surface(source.split(":", 1)[1], line.split(":", 1)[1])
            restored.append(target + ":" + body)
        else:
            restored.append(line)  # Headers and dependent tiers remain subject to strict QC.
    return "\n".join(restored) + "\n"


def _check_and_postprocess(out: Path, manifest: dict, raw: Path, annotated: Path) -> dict:
    """Derive annotated files and apply the unchanged, case-sensitive QC."""
    inventory, config = manifest["inputs"], manifest["config"]
    for item in inventory:
        relative = Path(item["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Annotation file paths must stay inside their output directory")
    inputs = out / "input_snapshot"
    annotated.mkdir(parents=True, exist_ok=False)
    issues, outputs, raw_outputs, records = [], [], [], []
    for item in inventory:
        relative = item["relative_path"]
        original, snapshot = Path(item["source_path"]), inputs / relative
        raw_path, path = raw / relative, annotated / relative
        if sha256_file(original) != item["source_sha256"]:
            issues.append({"file": relative, "issue": "source_file_changed"})
        if sha256_file(snapshot) != item["snapshot_sha256"]:
            issues.append({"file": relative, "issue": "input_snapshot_changed"})
        if not raw_path.is_file():
            issues.append({"file": relative, "issue": "missing_output"})
            continue
        raw_outputs.append({"relative_path": relative, "sha256": sha256_file(raw_path)})
        before, raw_text = snapshot.read_text(), raw_path.read_text()
        try:
            after = postprocess_annotated_chat(before, raw_text, target=config["target_speaker"])
        except ValueError as error:
            issues.append({"file": relative, "issue": "target_text_or_boundaries_changed", "detail": str(error)})
            continue  # Retain raw output; do not produce a restored file for this case.
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            handle.write(after)
        records.append({"file": relative, "formatting_restored": after != raw_text})
        original_tiers = [_surface(x) for x in _target_utterances(before, config["target_speaker"])]
        new_tiers = [_surface(x) for x in _target_utterances(after, config["target_speaker"])]
        if original_tiers != new_tiers:
            issues.append({"file": relative, "issue": "target_text_or_boundaries_changed"})
        other = lambda text: [line for line in text.splitlines()
                              if not line.startswith(config["target_speaker"] + ":")]
        if other(before) != other(after):
            issues.append({"file": relative, "issue": "non_target_content_changed"})
        outputs.append({"relative_path": relative, "sha256": sha256_file(path)})
    expected = {item["relative_path"] for item in inventory}
    unexpected = {p.relative_to(raw).as_posix() for p in raw.rglob("*.cha")} - expected
    for path in sorted(unexpected):
        issues.append({"file": path, "issue": "unexpected_output"})
    report_path = raw / "_talk_tag_report.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    total = sum(item["target_utterances"] for item in inventory)
    if (report.get("total_files") != len(inventory) or report.get("processed_files") != len(inventory)
            or report.get("failed_files") != 0):
        issues.append({"issue": "missing_or_incomplete_tool_report"})
    if config["limit"] == 0 and report.get("target_lines") != total:
        issues.append({"issue": "incomplete_target_coverage"})
    return {"outputs": outputs, "raw_outputs": raw_outputs, "qc_issues": issues,
            "expected_target_utterances": total, "reported_target_utterances": report.get("target_lines"),
            "raw_output_directory": raw.name, "annotated_output_directory": annotated.name,
            "tool_report_sha256": sha256_file(report_path) if report_path.is_file() else None,
            "postprocessing": {"method": "restore_source_case_and_spacing", "version": 1, "files": records},
            "qc_comparison": "case_sensitive_surface_tokens_and_utterance_boundaries"}


def _raise_qc_issues(manifest: dict, manifest_path: Path) -> None:
    counts = Counter(item["issue"] for item in manifest["qc_issues"])
    details = ", ".join(f"{issue}={count}" for issue, count in sorted(counts.items()))
    raise RuntimeError(f"Annotation QC needs review ({details}); see {manifest_path}")


def postprocess_annotation_run(manifest_path: str | Path, *, output_name: str = "postprocessed") -> Path:
    """Postprocess/recheck retained outputs without inference or rewriting a prior manifest."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", output_name):
        raise ValueError("Use a simple new output_name")
    source_manifest = Path(manifest_path).resolve()
    out = source_manifest.parent
    manifest = json.loads(source_manifest.read_text())
    if manifest.get("exit_code") != 0:
        raise ValueError("Postprocessing requires an annotation process that returned successfully")
    raw_name = manifest.get("raw_output_directory", "annotated")  # Earlier runs stored raw files here.
    if Path(raw_name).name != raw_name or raw_name in {".", ".."}:
        raise ValueError("Invalid raw output directory in manifest")
    raw, destination = out / raw_name, out / output_name
    reviewed_manifest = out / (output_name + "_manifest.json")
    if destination.exists() or reviewed_manifest.exists():
        raise FileExistsError("Postprocessed artifacts already exist; choose a fresh output_name")
    for item in manifest.get("raw_outputs", manifest.get("outputs", [])):
        if sha256_file(raw / item["relative_path"]) != item["sha256"]:
            raise ValueError("Raw annotation output changed since the original manifest; retain for review")
    if (manifest.get("tool_report_sha256")
            and sha256_file(raw / "_talk_tag_report.json") != manifest["tool_report_sha256"]):
        raise ValueError("TalkTag report changed since the original manifest; retain for review")
    reviewed = {**manifest, "source_annotation_manifest": str(source_manifest),
                "source_annotation_manifest_sha256": sha256_file(source_manifest),
                "prior_status": manifest.get("status"), "prior_qc_issues": manifest.get("qc_issues", []),
                "wrapper_sha256": sha256_file(Path(__file__)), "status": "postprocessing_in_progress"}
    with reviewed_manifest.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(reviewed, indent=2) + "\n")
    try:
        reviewed.update(_check_and_postprocess(out, manifest, raw, destination))
        if reviewed["qc_issues"]:
            _raise_qc_issues(reviewed, reviewed_manifest)
        reviewed["status"] = "partial_debug" if manifest["config"]["limit"] else "annotation_complete"
    except BaseException:
        reviewed["status"] = "failed_or_interrupted; outputs retained for review"
        raise
    finally:
        reviewed_manifest.write_text(json.dumps(reviewed, indent=2) + "\n")
    return reviewed_manifest


def annotate_chat_directory(input_dir: str | Path | dict[str, Path], output_dir: str | Path, *,
                            config: dict, python: str | Path, runtime: dict | None = None) -> Path:
    """Annotate one directory or labelled directories in a single model process."""
    sources = ({label: Path(path).resolve() for label, path in input_dir.items()}
               if isinstance(input_dir, dict) else {"": Path(input_dir).resolve()})
    out = Path(output_dir).resolve()
    for label, source in sources.items():
        if label and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", label):
            raise ValueError(f"Unsafe annotation input label: {label!r}")
        if not source.is_dir():
            raise FileNotFoundError(f"Missing CHAT input directory: {source}")
        if source == out or source in out.parents or out in source.parents:
            raise ValueError("Annotation inputs and outputs must be separate directory trees")
    if out.exists():
        raise FileExistsError(f"Annotation output already exists; choose a fresh directory: {out}")
    files = []
    for label, source in sources.items():
        selected = sorted(source.rglob("*.cha"))
        if not selected:
            raise ValueError(f"No CHAT files found: {source}")
        files.extend((Path(label) / path.relative_to(source), path) for path in selected)
    if not files:
        raise ValueError("No CHAT files found in the selected annotation inputs")
    if len({path for relative, path in files}) != len(files):
        raise ValueError("Annotation source directories overlap")
    snapshots = {}
    for relative, path in files:
        text = _unfold_target(path.read_text(encoding="utf-8-sig"), config["target_speaker"])
        count = len(_target_utterances(text, config["target_speaker"]))
        if not count:
            raise ValueError(f"No target utterances in {path}")
        snapshots[relative] = (path, text, count, sha256_file(path))
    runtime = annotation_runtime_metadata(python, config) if runtime is None else runtime
    if runtime.get("source_commit") != config["revision"]:
        raise ValueError("Installed TalkTag commit does not match annotation settings")
    out.mkdir(parents=True, exist_ok=False)
    inputs, raw, annotated = out / "input_snapshot", out / "raw_talktag", out / "annotated"
    inventory = []
    for relative, (path, text, count, digest) in snapshots.items():
        copied = inputs / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        copied.write_text(text, encoding="utf-8")
        inventory.append({"source_path": str(path), "relative_path": relative.as_posix(),
                          "source_sha256": digest, "snapshot_sha256": sha256_file(copied),
                          "target_utterances": count})
    command = [str(python), "-m", "talk_tag.cli", "annotate", "--input-dir", str(inputs),
               "--output-dir", str(raw), "--target-speaker", config["target_speaker"],
               "--device", config["device"], "--granularity", config["granularity"],
               "--limit", str(config["limit"]), "--speaker-field", "speaker", "--text-field", "utterance",
               "--fail-fast", "--no-progress"]
    if config["show_target"]:
        command.append("--show-target")
    manifest_path = out / "annotation_manifest.json"
    manifest = {"schema_version": 1, "status": "annotation_in_progress", "config": config,
                "runtime": runtime, "command": command, "inputs": inventory,
                "raw_output_directory": raw.name, "annotated_output_directory": annotated.name,
                "wrapper_sha256": sha256_file(Path(__file__))}

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    try:
        print(f"TalkTag: {len(inventory)} CHAT files, {sum(item['target_utterances'] for item in inventory)} target utterances")
        with (out / "talktag.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
        manifest["exit_code"] = completed.returncode
        if completed.returncode:
            raise RuntimeError(f"TalkTag failed; see {out / 'talktag.log'}")
        manifest.update(_check_and_postprocess(out, manifest, raw, annotated))
        if manifest["qc_issues"]:
            _raise_qc_issues(manifest, manifest_path)
        manifest["status"] = "partial_debug" if config["limit"] else "annotation_complete"
        save()
    except BaseException:
        manifest["status"] = "failed_or_interrupted; outputs retained for review"
        save()
        raise
    return manifest_path
