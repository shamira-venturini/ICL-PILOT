"""TalkTag execution on copied CHAT inputs, with provenance and source QC."""
from __future__ import annotations

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
    inputs, annotated = out / "input_snapshot", out / "annotated"
    inventory = []
    for relative, (path, text, count, digest) in snapshots.items():
        copied = inputs / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        copied.write_text(text, encoding="utf-8")
        inventory.append({"source_path": str(path), "relative_path": relative.as_posix(),
                          "source_sha256": digest, "snapshot_sha256": sha256_file(copied),
                          "target_utterances": count})
    command = [str(python), "-m", "talk_tag.cli", "annotate", "--input-dir", str(inputs),
               "--output-dir", str(annotated), "--target-speaker", config["target_speaker"],
               "--device", config["device"], "--granularity", config["granularity"],
               "--limit", str(config["limit"]), "--speaker-field", "speaker", "--text-field", "utterance",
               "--fail-fast", "--no-progress"]
    if config["show_target"]:
        command.append("--show-target")
    manifest_path = out / "annotation_manifest.json"
    manifest = {"schema_version": 1, "status": "annotation_in_progress", "config": config,
                "runtime": runtime, "command": command, "inputs": inventory,
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
        issues, outputs = [], []
        for item in inventory:
            relative = item["relative_path"]
            original = Path(item["source_path"])
            path = annotated / relative
            if sha256_file(original) != item["source_sha256"]:
                issues.append({"file": relative, "issue": "source_file_changed"})
            if not path.is_file():
                issues.append({"file": relative, "issue": "missing_output"})
                continue
            before, after = (inputs / relative).read_text(), path.read_text()
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
        unexpected = {p.relative_to(annotated).as_posix() for p in annotated.rglob("*.cha")} - expected
        for path in sorted(unexpected):
            issues.append({"file": path, "issue": "unexpected_output"})
        report_path = annotated / "_talk_tag_report.json"
        report = json.loads(report_path.read_text()) if report_path.is_file() else {}
        total = sum(item["target_utterances"] for item in inventory)
        if (report.get("total_files") != len(inventory) or report.get("processed_files") != len(inventory)
                or report.get("failed_files") != 0):
            issues.append({"issue": "missing_or_incomplete_tool_report"})
        if config["limit"] == 0 and report.get("target_lines") != total:
            issues.append({"issue": "incomplete_target_coverage"})
        manifest.update(outputs=outputs, qc_issues=issues, expected_target_utterances=total,
                        reported_target_utterances=report.get("target_lines"))
        if issues:
            raise RuntimeError(f"Annotation QC needs review; see {manifest_path}")
        manifest["status"] = "partial_debug" if config["limit"] else "annotation_complete"
        save()
    except BaseException:
        manifest["status"] = "failed_or_interrupted; outputs retained for review"
        save()
        raise
    return manifest_path
