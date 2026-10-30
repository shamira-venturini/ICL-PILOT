"""Required read-only CHAT validation with complete coverage and saved reports."""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess

from ..generation.recovery import atomic_json, timestamp
from .chatter_runtime import ensure_chatter, sha256_file, verify_chatter


class ChatterValidationError(RuntimeError):
    def __init__(self, manifest_path):
        self.manifest_path = Path(manifest_path)
        super().__init__(f"Chatter validation did not pass; see {self.manifest_path}")


def _inventory(inputs: dict) -> list[dict]:
    result, seen = [], set()
    for label, value in inputs.items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", label):
            raise ValueError("Use simple unique validation input labels.")
        root = Path(value).resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        paths = sorted(root.rglob("*.cha"))
        if not paths:
            raise ValueError(f"No CHAT files to validate: {root}")
        for path in paths:
            resolved = path.resolve()
            if resolved in seen:
                raise ValueError("CHAT validation input collections overlap.")
            seen.add(resolved)
            result.append({"label": label, "root": str(root), "relative_path": path.relative_to(root).as_posix(),
                           "path": str(resolved), "sha256": sha256_file(path)})
    if not result:
        raise ValueError("Select CHAT files before validation.")
    return result


def validate_chat_directories(inputs: dict, output_dir, *, validator=None, raise_on_failure=True) -> Path:
    """Fail closed: require success, fresh per-file results and complete summary.

    Only ``validate`` is invoked. Source bytes are checked before/after; no fix,
    normalize, suppressed error, skipped alignment or reduced parser is used.
    """
    output = Path(output_dir).resolve()
    for value in inputs.values():
        root = Path(value).resolve()
        if output == root or root in output.parents:
            raise ValueError("Place validation reports outside the CHAT input tree.")
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "validation_manifest.json"
    manifest = {"schema_version": 1, "started_at_utc": timestamp(), "status": "validation_in_progress",
                "inputs": {label: str(Path(path).resolve()) for label, path in inputs.items()},
                "inventory": [], "commands": [], "issues": [],
                "wrapper_sha256": sha256_file(Path(__file__))}
    atomic_json(manifest_path, manifest)
    try:
        inventory = _inventory(inputs)
        manifest["inventory"] = inventory
        selected = validator or ensure_chatter()
        tool = verify_chatter(selected["path"], config_path=selected.get("config_path"))
        manifest["tool"] = tool
        for index, (label, value) in enumerate(inputs.items(), 1):
            root = Path(value).resolve()
            command = [tool["path"], "validate", str(root), *tool["validation_args"]]
            stdout, stderr = output / f"{index:02d}_{label}.jsonl", output / f"{index:02d}_{label}.stderr.log"
            with stdout.open("x", encoding="utf-8") as out, stderr.open("x", encoding="utf-8") as err:
                done = subprocess.run(command, stdout=out, stderr=err, text=True, check=False)
            record = {"label": label, "command": command, "exit_code": done.returncode,
                      "stdout": stdout.name, "stdout_sha256": sha256_file(stdout),
                      "stderr": stderr.name, "stderr_sha256": sha256_file(stderr)}
            manifest["commands"].append(record)
            if done.returncode:
                manifest["issues"].append({"label": label, "issue": "validator_nonzero_exit", "exit_code": done.returncode})
            try:
                events = [json.loads(line) for line in stdout.read_text().splitlines() if line.strip()]
                files = [event for event in events if event.get("type") == "file"]
                summaries = [event for event in events if event.get("type") == "summary"]
                expected = {item["path"] for item in inventory if item["label"] == label}
                reported = [str(Path(event["file"]).resolve()) for event in files]
                if set(reported) != expected or len(reported) != len(expected):
                    manifest["issues"].append({"label": label, "issue": "incomplete_or_duplicate_file_coverage"})
                if any(event.get("status") != "valid" or event.get("cache_hit") is not False
                       or event.get("error_count", 0) for event in files):
                    manifest["issues"].append({"label": label, "issue": "invalid_or_unchecked_file"})
                if (len(summaries) != 1 or summaries[0].get("outcome") != "complete"
                        or summaries[0].get("total_files") != len(expected)
                        or summaries[0].get("valid") != len(expected)
                        or summaries[0].get("invalid") != 0 or summaries[0].get("internal_failures") != 0
                        or summaries[0].get("cache_hits") != 0):
                    manifest["issues"].append({"label": label, "issue": "missing_or_unsuccessful_summary"})
                record["summary"] = summaries[0] if len(summaries) == 1 else None
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                manifest["issues"].append({"label": label, "issue": "invalid_validator_report", "detail": str(error)})
        if _inventory(inputs) != inventory:
            manifest["issues"].append({"issue": "source_files_changed_during_validation"})
        manifest["status"] = "failed" if manifest["issues"] else "passed"
        manifest["files_checked"] = len(inventory)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["issues"].append({"issue": "validation_unavailable_or_interrupted", "detail": str(error)})
    except BaseException:
        manifest["status"] = "interrupted"
        raise
    finally:
        manifest["finished_at_utc"] = timestamp()
        atomic_json(manifest_path, manifest)
    if manifest["status"] != "passed" and raise_on_failure:
        raise ChatterValidationError(manifest_path)
    return manifest_path


def validation_reference(path) -> dict:
    path = Path(path).resolve()
    return {"manifest": str(path), "sha256": sha256_file(path)}


def require_validated_report(reference: dict) -> dict:
    """Reject stale/edited files or reports before presenting/exporting a pass."""
    path = Path(reference["manifest"])
    if sha256_file(path) != reference["sha256"]:
        raise ValueError("Chatter validation manifest changed; retained for review.")
    report = json.loads(path.read_text())
    if report["status"] != "passed" or _inventory(report["inputs"]) != report["inventory"]:
        raise ValueError("CHAT files have not passed validation or changed since their Chatter check.")
    for command in report["commands"]:
        for field in ("stdout", "stderr"):
            if sha256_file(path.parent / command[field]) != command[field + "_sha256"]:
                raise ValueError("Chatter diagnostic log changed; retained for review.")
    return report
