"""TalkTag execution on copied CHAT inputs, with provenance and source QC."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from . import validation


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
_RESTORATION_TOKEN = re.compile(
    _INLINE_ANNOTATION.pattern + r"|\[[^\[\]]*\]"
    r"|\(\s*(?:\.{1,3}|\d+(?:\.\d+)?)\s*\)|&[=+~*][^\s]+"
    r"|(?:\([A-Za-z]+\))?\w+(?:(?:['‘’:@-]|\([A-Za-z]+\))\w*)*|[^\w\s]"
)


def _restoration_units(bodies):
    tokens, annotations = [], []
    for line_index, body in enumerate(bodies):
        line_start = len(tokens)
        for match in _RESTORATION_TOKEN.finditer(body):
            item = {"text": match.group(), "line": line_index,
                    "start": match.start(), "end": match.end()}
            if _INLINE_ANNOTATION.fullmatch(match.group()):
                annotations.append({**item, "anchor": len(tokens), "leading": len(tokens) == line_start})
            elif match.group() not in {'"', '“', '”'}:
                tokens.append(item)
    return tokens, annotations


def _is_restoration_word(token):
    return (not token.startswith(("[", "&")) and not re.fullmatch(r"\([.\d\s]+\)", token)
            and bool(re.search(r"\w", token)))


def _restoration_key(token):
    return token.lower().replace("‘", "'").replace("’", "'")


def _align_restoration_tokens(source, raw):
    """Return all optimal source anchors, refusing arbitrary repeated-word ties.

    Text always comes from source. Only tags on unchanged, uniquely aligned
    tokens are transferable; substituted/deleted words remain unassessed.
    """
    a, b = [_restoration_key(x["text"]) for x in source], [_restoration_key(x["text"]) for x in raw]
    n, m = len(a), len(b)
    if a == b:
        return [{j} for j in range(m)], [], 0

    def substitution(i, j):
        if a[i] == b[j]:
            return 0
        return 1 if _is_restoration_word(a[i]) and _is_restoration_word(b[j]) else 3

    forward = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        forward[i][0] = i
    for j in range(m + 1):
        forward[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            forward[i][j] = min(forward[i - 1][j] + 1, forward[i][j - 1] + 1,
                                forward[i - 1][j - 1] + substitution(i - 1, j - 1))
    backward = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n, -1, -1):
        backward[i][m] = n - i
    for j in range(m, -1, -1):
        backward[n][j] = m - j
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            backward[i][j] = min(backward[i + 1][j] + 1, backward[i][j + 1] + 1,
                                 backward[i + 1][j + 1] + substitution(i, j))
    total = forward[n][m]
    possible = [set() for _ in range(m)]
    for j in range(m):
        for i in range(n + 1):
            if forward[i][j] + 1 + backward[i][j + 1] == total:
                possible[j].add(None)
            if i < n and forward[i][j] + substitution(i, j) + backward[i + 1][j + 1] == total:
                possible[j].add(i)
    # One reproducible edit path is explanatory only. Tag transfer uses every
    # optimal path above, so tie-breaking cannot move tags between repetitions.
    edits, i, j = [], 0, 0
    while i < n or j < m:
        if i < n and j < m and backward[i][j] == substitution(i, j) + backward[i + 1][j + 1]:
            if a[i] != b[j]:
                edits.append({"kind": "substitution_restored", "source_token": i, "raw_token": j,
                              "source": source[i]["text"], "raw": raw[j]["text"]})
            i, j = i + 1, j + 1
        elif i < n and backward[i][j] == 1 + backward[i + 1][j]:
            edits.append({"kind": "deletion_restored", "source_token": i, "source": source[i]["text"]})
            i += 1
        else:
            edits.append({"kind": "model_insertion_removed", "raw_token": j, "raw": raw[j]["text"]})
            j += 1
    return possible, edits, total


def _restore_source_bodies(originals, annotated):
    source, existing_tags = _restoration_units(originals)
    raw, raw_tags = _restoration_units(annotated)
    possible, edits, distance = _align_restoration_tokens(source, raw)
    confident = {j: next(iter(anchors)) for j, anchors in enumerate(possible)
                 if len(anchors) == 1 and None not in anchors}
    confident = {j: i for j, i in confident.items()
                 if _restoration_key(source[i]["text"]) == _restoration_key(raw[j]["text"])}
    source_annotations, new_annotations, discarded = {}, {}, []
    for tag in existing_tags:
        source_annotations.setdefault((tag["line"], tag["anchor"]), []).append(tag)
    for tag in raw_tags:
        previous = tag["anchor"] - 1
        if tag["leading"] or previous not in confident:
            discarded.append({"tag": tag["text"], "raw_utterance": tag["line"] + 1,
                              "raw_anchor": tag["anchor"], "reason": "changed_inserted_or_ambiguous_anchor"})
            continue
        index = confident[previous]
        key = (source[index]["line"], index + 1)
        new_annotations.setdefault(key, []).append(tag["text"])
    missing_existing = Counter()
    for key, existing in source_annotations.items():
        available = Counter(new_annotations.get(key, []))
        for tag in existing:
            if available[tag["text"]]:
                available[tag["text"]] -= 1
            else:
                missing_existing[tag["text"]] += 1
    insertions, retained = {}, 0
    for key, tags in sorted(new_annotations.items()):
        line, anchor = key
        existing = source_annotations.get(key, [])
        index = 0
        for tag in tags:
            if index < len(existing) and tag == existing[index]["text"]:
                index += 1
        if index != len(existing):
            discarded.extend({"tag": tag, "source_utterance": line + 1,
                              "reason": "existing_source_annotations_take_precedence"}
                             for tag in tags if tag not in {x["text"] for x in existing})
            continue
        index = 0
        for tag in tags:
            if index < len(existing) and tag == existing[index]["text"]:
                index += 1
                continue
            if missing_existing[tag]:
                missing_existing[tag] -= 1
                discarded.append({"tag": tag, "source_utterance": line + 1,
                                  "reason": "existing_source_annotation_moved"})
                continue
            if index < len(existing):
                position, addition = existing[index]["start"], tag + " "
            elif existing:
                position, addition = existing[-1]["end"], " " + tag
            else:
                position, addition = source[anchor - 1]["end"], " " + tag
            insertions.setdefault(line, {}).setdefault(position, []).append(addition)
            retained += 1
    restored = []
    for line, body in enumerate(originals):
        parts, cursor = [], 0
        for position, additions in sorted(insertions.get(line, {}).items()):
            parts.extend([body[cursor:position], "".join(additions)])
            cursor = position
        parts.append(body[cursor:])
        restored.append("".join(parts))
    assessed = set(confident.values())
    unassessed = [{"source_utterance": token["line"] + 1, "token": token["text"],
                   "start": token["start"], "end": token["end"]}
                  for i, token in enumerate(source) if _is_restoration_word(token["text"]) and i not in assessed]
    word_edits = any(_is_restoration_word(edit.get("source", ""))
                     or _is_restoration_word(edit.get("raw", ""))
                     or edit.get("raw", "").startswith(("[*", "[:", "[=")) for edit in edits)
    return restored, {"edit_distance": distance, "example_edit_path": edits,
                      "model_word_edits": word_edits,
                      "annotations_transferred": retained, "unmapped_annotations": discarded,
                      "unassessed_source_words": unassessed,
                      "annotation_coverage": "partial" if word_edits or unassessed or discarded else "complete"}


def _restore_utterance_surface(original: str, annotated: str) -> str:
    """Always restore source text; copy only uniquely anchored annotation tags."""
    restored, _ = _restore_source_bodies([original], [annotated])
    return restored[0]


def postprocess_annotated_chat(original: str, annotated: str, *, target: str = "*CHI", audit=None) -> str:
    """Rebuild from source, automatically undoing annotator edits and deletions."""
    source_lines = _unfold_target(original, target).splitlines()
    indices = [i for i, line in enumerate(source_lines) if line.startswith(target + ":")]
    originals = [source_lines[i].split(":", 1)[1] for i in indices]
    try:
        raw_lines = _unfold_target(annotated, target).splitlines()
    except ValueError:
        raw_lines = annotated.splitlines()
    raw = [line.split(":", 1)[1] for line in raw_lines if line.startswith(target + ":")]
    reports = []
    if len(originals) == len(raw):
        bodies = []
        for utterance, (before, after) in enumerate(zip(originals, raw), 1):
            restored, report = _restore_source_bodies([before], [after])
            bodies.extend(restored)
            for field in ("unmapped_annotations", "unassessed_source_words"):
                for item in report[field]:
                    if "source_utterance" in item:
                        item["source_utterance"] = utterance
                    if "raw_utterance" in item:
                        item["raw_utterance"] = utterance
            reports.append({"source_utterance": utterance, **report})
    else:
        bodies, report = _restore_source_bodies(originals, raw)
        reports.append({"source_utterance": "global_boundary_recovery", **report})
    for index, body in zip(indices, bodies):
        source_lines[index] = target + ":" + body
    if audit is not None:
        audit.update(source_utterances=len(originals), raw_utterances=len(raw),
                     boundaries_restored=len(originals) != len(raw), utterances=reports,
                     annotation_coverage="partial" if any(r["annotation_coverage"] == "partial" for r in reports) else "complete",
                     raw_non_target_content_ignored=[line for line in raw_lines if not line.startswith(target + ":")]
                         != [line for line in source_lines if not line.startswith(target + ":")])
    return "\n".join(source_lines) + "\n"


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
        restoration = {}
        try:
            after = postprocess_annotated_chat(before, raw_text, target=config["target_speaker"], audit=restoration)
        except ValueError as error:
            issues.append({"file": relative, "issue": "target_text_or_boundaries_changed", "detail": str(error)})
            continue  # Retain raw output; do not produce a restored file for this case.
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            handle.write(after)
        records.append({"file": relative, "formatting_restored": after != raw_text,
                        "source_restoration": restoration})
        original_tiers = [_surface(x) for x in _target_utterances(before, config["target_speaker"])]
        new_tiers = [_surface(x) for x in _target_utterances(after, config["target_speaker"])]
        if original_tiers != new_tiers:
            issues.append({"file": relative, "issue": "target_text_or_boundaries_changed"})
        other = lambda text: [line for line in text.splitlines()
                              if not line.startswith(config["target_speaker"] + ":")]
        if other(before) != other(after):
            issues.append({"file": relative, "issue": "non_target_content_changed"})
        from .preservation import text_fingerprint, require_text_preserved
        baseline = item.get("text_fingerprint", text_fingerprint(before))
        try:
            fingerprint = require_text_preserved(baseline, after)
        except ValueError as error:
            issues.append({"file": relative, "issue": "text_fingerprint_changed", "detail": str(error)})
            fingerprint = text_fingerprint(after)
        outputs.append({"relative_path": relative, "sha256": sha256_file(path),
                        "text_fingerprint": fingerprint,
                        "annotation_coverage": restoration["annotation_coverage"]})
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
            "postprocessing": {"method": "source_authoritative_automatic_restoration", "version": 3,
                               "word_changes_and_deletions": "automatically_restored_from_source",
                               "unmapped_tags": "recorded_without_arbitrary_assignment",
                               "partial_annotation_files": sum(item["annotation_coverage"] == "partial" for item in outputs),
                               "files": records},
            "qc_comparison": "case_sensitive_surface_tokens_and_utterance_boundaries"}


def _raise_qc_issues(manifest: dict, manifest_path: Path) -> None:
    counts = Counter(item["issue"] for item in manifest["qc_issues"])
    details = ", ".join(f"{issue}={count}" for issue, count in sorted(counts.items()))
    raise RuntimeError(f"Annotation QC needs review ({details}); see {manifest_path}")


def _chatter_stage(manifest, name, inputs, output, validator, *, blocking=True):
    """Retain the diagnostic reference on failure as well as success."""
    try:
        report = validation.validate_chat_directories(inputs, output, validator=validator,
                                                      raise_on_failure=blocking)
    except validation.ChatterValidationError as error:
        manifest.setdefault("chatter", {})[name] = validation.validation_reference(error.manifest_path)
        raise
    manifest.setdefault("chatter", {})[name] = validation.validation_reference(report)
    return report


def postprocess_annotation_run(manifest_path: str | Path, *, output_name: str = "postprocessed",
                              validator: dict | None = None) -> Path:
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
        selected_validator = validator or validation.ensure_chatter()
        reviewed["chatter_tool"] = selected_validator
        _chatter_stage(reviewed, "snapshot_recheck", {"input": out / "input_snapshot"},
                       out / "validation" / (output_name + "_before"), selected_validator)
        _chatter_stage(reviewed, "raw_output_recheck", {"raw": raw},
                       out / "validation" / (output_name + "_raw"), selected_validator, blocking=False)
        reviewed.update(_check_and_postprocess(out, manifest, raw, destination))
        if reviewed["qc_issues"]:
            _raise_qc_issues(reviewed, reviewed_manifest)
        _chatter_stage(reviewed, "after", {"annotated": destination},
                       out / "validation" / (output_name + "_after"), selected_validator)
        reviewed["status"] = "partial_debug" if manifest["config"]["limit"] else "annotation_complete"
    except BaseException:
        reviewed["status"] = "failed_or_interrupted; outputs retained for review"
        raise
    finally:
        reviewed_manifest.write_text(json.dumps(reviewed, indent=2) + "\n")
    return reviewed_manifest


def annotate_chat_directory(input_dir: str | Path | dict[str, Path], output_dir: str | Path, *,
                            config: dict, python: str | Path, runtime: dict | None = None,
                            validator: dict | None = None, preparation_reference: dict | None = None) -> Path:
    """Annotate one directory or labelled directories in a single model process."""
    sources = ({label: Path(path).resolve() for label, path in input_dir.items()}
               if isinstance(input_dir, dict) else {"": Path(input_dir).resolve()})
    preparation = None
    if preparation_reference is not None:
        from .annotation_preparation import require_prepared_annotations
        preparation = require_prepared_annotations(preparation_reference)
        expected = {label: Path(item["prepared"]).resolve() for label, item in preparation["collections"].items()}
        if sources != expected:
            raise ValueError("TalkTag inputs do not match the checked shared preparation")
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
    if len({relative for relative, path in files}) != len(files):
        raise ValueError("Annotation input labels create colliding snapshot paths")
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
    from .preservation import text_fingerprint
    for relative, (path, text, count, digest) in snapshots.items():
        copied = inputs / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        copied.write_text(text, encoding="utf-8")
        inventory.append({"source_path": str(path), "relative_path": relative.as_posix(),
                          "source_sha256": digest, "snapshot_sha256": sha256_file(copied),
                          "target_utterances": count,
                          "text_fingerprint": text_fingerprint(text)})
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
    if preparation is not None:
        manifest["input_preparation"] = preparation_reference
        manifest["collections"] = preparation["collections"]

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    try:
        selected_validator = validator or validation.ensure_chatter()
        manifest["chatter_tool"] = selected_validator
        _chatter_stage(manifest, "before_sources", {f"source_{index:04d}": path
                       for index, path in enumerate(sources.values(), 1)},
                       out / "validation/before_sources", selected_validator)
        _chatter_stage(manifest, "before_snapshot", {"input": inputs},
                       out / "validation/before_snapshot", selected_validator)
        save()
        print(f"TalkTag: {len(inventory)} CHAT files, {sum(item['target_utterances'] for item in inventory)} target utterances")
        with (out / "talktag.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
        manifest["exit_code"] = completed.returncode
        if completed.returncode:
            raise RuntimeError(f"TalkTag failed; see {out / 'talktag.log'}")
        _chatter_stage(manifest, "raw_output", {"raw": raw}, out / "validation/raw_output",
                       selected_validator, blocking=False)
        manifest.update(_check_and_postprocess(out, manifest, raw, annotated))
        if manifest["qc_issues"]:
            _raise_qc_issues(manifest, manifest_path)
        _chatter_stage(manifest, "after", {"annotated": annotated}, out / "validation/after", selected_validator)
        if preparation_reference is not None:
            require_prepared_annotations(preparation_reference)
        manifest["status"] = "partial_debug" if config["limit"] else "annotation_complete"
        save()
    except BaseException:
        manifest["status"] = "failed_or_interrupted; outputs retained for review"
        save()
        raise
    return manifest_path
