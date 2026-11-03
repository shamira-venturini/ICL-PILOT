"""Fresh Batchalign 3 UD tiers on immutable CHAT snapshots, with strict QC."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.request import urlopen

from ..generation.recovery import atomic_json, timestamp
from .annotation import sha256_file
from . import validation
from .morphosyntax_worker import file_digest


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs/annotation/batchalign.json"
MORPH_TIERS = {"%mor", "%gra", "%umor", "%ugra"}
_SOURCE_UD_PROVENANCE = re.compile(
    r"batchalign(?:3)? [a-f0-9]{7,40} \| morphosyntax: "
    r"stanza:\d+\.\d+\.\d+:noretok:native-mwt\d+ \| [0-9T:Z.+-]+")


def load_morphosyntax_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text())
    expected = {"schema_version": 1, "policy": "batchalign3-english-ud-preserve-v1",
                "version": "0.10.7", "stanza_version": "1.15.0", "torch_version": "2.11.0",
                "language": "eng", "target_speaker": "*CHI", "annotated_speakers": "all", "retokenize": False,
                "device": "cpu", "workers": 1, "torch_threads": 2, "seed": 42,
                "cache_policy": "Bypass", "output_tiers": ["%mor", "%gra"],
                "replaced_tiers": ["%mor", "%gra", "%umor", "%ugra"]}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Use the reviewed English UD configuration with preserved tokens and boundaries")
    if (len(config.get("models", {})) != 8 or not config.get("source_sha256")
            or not re.fullmatch(r"[a-f0-9]{64}", config.get("resources", {}).get("sha256", ""))):
        raise ValueError("Annotation requires model/catalog/source provenance")
    resources = config["resources"]
    if (resources.get("version") != config["stanza_version"]
            or not re.fullmatch(r"[a-f0-9]{40}", resources.get("revision", ""))
            or resources.get("url") != "https://raw.githubusercontent.com/stanfordnlp/stanza-resources/"
                + resources["revision"] + "/resources_" + resources["version"] + ".json"
            or not re.fullmatch(r"[a-f0-9]{40}", config.get("models_revision", ""))):
        raise ValueError("Annotation downloads require immutable catalog/model revisions")
    model_prefix = ("https://huggingface.co/stanfordnlp/stanza-en/resolve/"
                    + config["models_revision"] + "/models/")
    for name, spec in config["models"].items():
        if (not name.startswith("en/") or ".." in Path(name).parts
                or not re.fullmatch(r"[a-f0-9]{32}", spec.get("md5", ""))
                or not re.fullmatch(r"[a-f0-9]{64}", spec.get("sha256", ""))
                or spec.get("url") != model_prefix + name[3:]):
            raise ValueError("Invalid English Stanza model pin")
    return config


def install_morphosyntax_runtime(environment, config_path=DEFAULT_CONFIG):
    if sys.version_info < (3, 10):
        raise RuntimeError("Batchalign 3 annotation needs Python >=3.10")
    config = load_morphosyntax_config(config_path)
    environment = Path(environment).expanduser().resolve()
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(environment)], check=True)
    pip = [sys.executable, "-m", "pip", "--python", str(python), "install"]
    if sys.platform.startswith("linux"):
        subprocess.run(pip + ["--index-url", "https://download.pytorch.org/whl/cpu", "torch==" + config["torch_version"]], check=True)
    subprocess.run(pip + ["--only-binary=batchalign", "batchalign[stanza]==" + config["version"],
                          "stanza==" + config["stanza_version"], "torch==" + config["torch_version"],
                          "transformers==4.57.1"], check=True)
    return python


def prepare_morphosyntax_models(model_dir, config_path=DEFAULT_CONFIG, *, download=True):
    config = load_morphosyntax_config(config_path)
    root = Path(model_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    specifications = {"resources.json": {**config["resources"], "algorithm": "sha256"},
                      **{name: {**spec, "algorithm": "sha256"} for name, spec in config["models"].items()}}
    for name, spec in specifications.items():
        path = root / name
        algorithm = spec["algorithm"]
        if path.is_file():
            if file_digest(path, algorithm) != spec[algorithm]:
                raise ValueError("Annotation resource checksum mismatch; existing file preserved: " + str(path))
            continue
        if not download:
            raise FileNotFoundError("Annotation model missing: " + str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".downloading")
        with urlopen(spec["url"], timeout=60) as source, temporary.open("wb") as out:
            shutil.copyfileobj(source, out)
        actual = file_digest(temporary, algorithm)
        if actual != spec[algorithm]:
            raise ValueError("Annotation download checksum mismatch: " + str(temporary)
                             + "; expected " + spec[algorithm] + "; received " + actual
                             + "; source " + spec["url"])
        temporary.rename(path)
    return root


def _records(text):
    result = []
    for line in text.splitlines(keepends=True):
        if line.startswith(("@", "*", "%")):
            result.append(line)
        elif result:
            result[-1] += line
        elif line.strip():
            raise ValueError("CHAT content precedes its first tier")
    return result


def _tier(record):
    return record.split(":", 1)[0] if ":" in record.splitlines()[0] else record.splitlines()[0].strip()


def _body(record):
    return " ".join(record.split(":", 1)[1].split()) if ":" in record else ""


def prepare_morphosyntax_input(text):
    """Drop stale morphology on the worker copy only; log all four tier families."""
    result, removed = [], Counter()
    for record in _records(text):
        tier = _tier(record)
        if tier in MORPH_TIERS:
            removed[tier] += 1
        else:
            result.append(record)
    return "".join(result), dict(removed)


def _protected(text):
    return [(_tier(record), _body(record)) for record in _records(text) if _tier(record) not in MORPH_TIERS]


def _speaker_blocks(text):
    result, block = [], None
    for record in _records(text):
        tier = _tier(record)
        if tier.startswith(("*", "@")):
            block = {"main": record, "speaker": tier} if tier.startswith("*") else None
            if block is not None:
                result.append(block)
        elif block is not None and tier in MORPH_TIERS:
            if tier in block:
                raise ValueError("Duplicate target morphosyntactic tier")
            block[tier] = record
    return result


def _target_blocks(text):
    return [block for block in _speaker_blocks(text) if block["speaker"] == "*CHI"]


def _has_alignable_speech(main):
    body = re.sub(r"\[[^\]]*\]|\([^)]*\)|&[^\s]+|\+[^\s]+", " ", _body(main))
    return any(token.casefold() not in {"xxx", "yyy", "www"} for token in re.findall(r"[^\W\d_]+", body))


def check_morphosyntax_output(source, output):
    original = _protected(source)
    projected, added, restored, index = [], [], [], 0

    def restore_source_provenance(next_record=None):
        nonlocal index
        # Batchalign replaces its earlier tool-version comment. The final
        # merger copies the complete original header, so account explicitly
        # for that metadata-only omission in the raw backend output.
        while (index < len(original) and original[index] != next_record
               and original[index][0] == "@Comment"
               and _SOURCE_UD_PROVENANCE.fullmatch(original[index][1])):
            projected.append(original[index])
            restored.append(original[index][1])
            index += 1

    for tier, body in _protected(output):
        restore_source_provenance((tier, body))
        if index < len(original) and (tier, body) == original[index]:
            projected.append((tier, body))
            index += 1
        elif tier == "@Comment" and re.fullmatch(
                r"batchalign c8cd2e0 \| morphosyntax: stanza:1\.15\.0:noretok:native-mwt10 \| [0-9T:Z.+-]+", body):
            added.append(body)
        else:
            raise ValueError("Batchalign changed headers, words, codes, punctuation, order or utterance boundaries")
    restore_source_provenance()
    if original != projected or len(added) > 1:
        raise ValueError("Batchalign changed headers, words, codes, punctuation, order or utterance boundaries")
    blocks = _speaker_blocks(output)
    if not _target_blocks(output):
        raise ValueError("No CHI utterances selected")
    tagged, excluded = 0, []
    for index, block in enumerate(blocks, 1):
        if _body(block.get("%mor", "")) and _body(block.get("%gra", "")):
            tagged += block["speaker"] == "*CHI"
        elif _has_alignable_speech(block["main"]):
            raise ValueError("Missing UD morphology/dependencies at speaker utterance " + str(index))
        else:
            excluded.append({"speaker_utterance_index": index, "speaker": block["speaker"]})
    return {"target_utterances": len(_target_blocks(output)), "tagged_utterances": tagged,
            "all_speaker_utterances": len(blocks),
            "nonlexical_utterance_indices": excluded, "native_added_provenance": added,
            "restored_source_provenance": restored}


def merge_morphosyntax_tiers(source, raw):
    """Insert fresh UD tiers into source layout for all speakers, retaining other content."""
    check_morphosyntax_output(source, raw)
    tagged, result = iter(_speaker_blocks(raw)), []
    for record in _records(source):
        tier = _tier(record)
        if tier in MORPH_TIERS:
            continue
        result.append(record)
        if tier.startswith("*"):
            block = next(tagged)
            result.extend(block[name] for name in ("%mor", "%gra") if name in block)
    result = "".join(result)
    check_morphosyntax_output(source, result)
    return result


def _inventory(inputs):
    result, seen = [], set()
    for label, value in inputs.items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", label):
            raise ValueError("Use simple unique annotation labels")
        root = Path(value).expanduser().resolve()
        files = sorted(root.rglob("*.cha"))
        if not root.is_dir() or not files:
            raise ValueError("Each annotation input must contain CHAT files")
        for path in files:
            if path.resolve() in seen or path.is_symlink() or root not in path.resolve().parents:
                raise ValueError("Annotation inputs overlap or escape their selected directory")
            seen.add(path.resolve())
            result.append({"source_path": str(path.resolve()),
                           "relative_path": (Path(label) / path.relative_to(root)).as_posix(),
                           "source_sha256": sha256_file(path)})
    if not result:
        raise ValueError("Select annotation inputs")
    return result


def _verify_runtime(runtime, config):
    for package, key in (("batchalign", "version"), ("stanza", "stanza_version"), ("torch", "torch_version")):
        if runtime.get("packages", {}).get(package, "").split("+")[0] != config[key]:
            raise ValueError("Annotation runtime version mismatch: " + package)
    if (runtime.get("source_sha256") != config["source_sha256"]
            or runtime.get("resources_sha256") != config["resources"]["sha256"]
            or set(runtime.get("model_files", {})) != set(config["models"])
            or runtime.get("retokenize") is not False or runtime.get("resegment") is not False
            or runtime.get("cache_policy") != "Bypass" or runtime.get("device") != config["device"]
            or runtime.get("workers") != config["workers"] or not runtime.get("engine_sha256")):
        raise ValueError("Incomplete or mismatched annotation runtime/model provenance")
    for name, spec in config["models"].items():
        if runtime["model_files"][name].get("sha256") != spec["sha256"]:
            raise ValueError("Annotation model checksum mismatch: " + name)


def annotate_morphosyntax(inputs, output_dir, *, python, model_dir, config_path=DEFAULT_CONFIG,
                          validator=None, test_only=False, upstream_annotation=None):
    config = load_morphosyntax_config(config_path)
    lineage = None
    if upstream_annotation is not None:
        from .annotation_preparation import checked_talktag_collections
        checked_inputs, preparation, parent = checked_talktag_collections(upstream_annotation)
        if {key: Path(value).resolve() for key, value in inputs.items()} != checked_inputs:
            raise ValueError("UD inputs do not match the checked TalkTag collections")
        lineage = {"upstream_annotation": upstream_annotation, "input_preparation": parent["input_preparation"],
                   "collections": preparation["collections"]}
    inventory = _inventory(inputs)
    output = Path(output_dir).expanduser().resolve()
    for root in (Path(value).expanduser().resolve() for value in inputs.values()):
        if output == root or output in root.parents or root in output.parents:
            raise ValueError("Annotation outputs must be separate from input trees")
    if not Path(python).is_file():
        raise FileNotFoundError("Run the Batchalign annotation runtime setup first")
    model_dir = prepare_morphosyntax_models(model_dir, config_path, download=False)
    output.mkdir(parents=True, exist_ok=False)
    snapshots, prepared, raw, annotated = (output / name for name in ("input_snapshot", "prepared", "raw_batchalign", "annotated"))
    path = output / "morphosyntax_manifest.json"
    manifest = {"schema_version": 1, "status": "running", "started_at_utc": timestamp(),
                "test_only": bool(test_only), "config": config, "inputs": inventory, "outputs": [],
                "wrapper_sha256": sha256_file(Path(__file__)), "chatter": {},
                "annotated_output_directory": "annotated"}
    if lineage is not None:
        manifest.update(lineage)
    atomic_json(path, manifest)
    try:
        for item in inventory:
            source = Path(item["source_path"])
            snapshot, target = snapshots / item["relative_path"], prepared / item["relative_path"]
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, snapshot)
            if sha256_file(snapshot) != item["source_sha256"]:
                raise ValueError("Source changed during snapshot preparation")
            text = snapshot.read_text(encoding="utf-8-sig")
            from .preservation import text_fingerprint, require_text_preserved
            item["text_fingerprint"] = text_fingerprint(text, "exact")
            if not re.search(r"^@Languages:\s*eng\s*$", text, re.M):
                raise ValueError("This annotation block requires explicit English CHAT headers")
            if not _target_blocks(text):
                raise ValueError("No CHI utterances in " + item["relative_path"])
            ready, removed = prepare_morphosyntax_input(text)
            target.write_text(ready, encoding="utf-8")
            item.update(snapshot_sha256=sha256_file(snapshot), prepared_sha256=sha256_file(target), removed_tiers=removed)
        selected = validator or validation.ensure_chatter()
        report = validation.validate_chat_directories({"prepared": prepared}, output / "validation/before", validator=selected)
        manifest["chatter"]["before"] = validation.validation_reference(report)
        request, response, log = output / "request.json", output / "response.json", output / "batchalign.log"
        worker = Path(__file__).with_name("morphosyntax_worker.py")
        atomic_json(request, {"schema_version": 1, "config": config, "model_dir": str(model_dir), "output_dir": str(raw),
                              "inputs": [{"relative_path": item["relative_path"], "path": str(prepared / item["relative_path"])} for item in inventory]})
        command = [str(python), str(worker), str(request), str(response)]
        manifest.update(command=command, worker_sha256=sha256_file(worker), request_sha256=sha256_file(request))
        atomic_json(path, manifest)
        with log.open("x") as handle:
            done = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, check=False)
        manifest.update(exit_code=done.returncode, log_sha256=sha256_file(log))
        if done.returncode or not response.is_file():
            raise RuntimeError("Batchalign annotation failed; see " + str(log))
        payload = json.loads(response.read_text())
        expected = {item["relative_path"] for item in inventory}
        if (payload.get("request_sha256") != manifest["request_sha256"] or sha256_file(request) != manifest["request_sha256"]
                or set(payload.get("completed_files", [])) != expected or len(payload["completed_files"]) != len(expected)
                or {p.relative_to(raw).as_posix() for p in raw.rglob("*.cha")} != expected):
            raise ValueError("Batchalign request identity/file coverage mismatch")
        _verify_runtime(payload["runtime"], config)
        manifest.update(runtime=payload["runtime"], response_sha256=sha256_file(response))
        for item in inventory:
            raw_path, target = raw / item["relative_path"], annotated / item["relative_path"]
            source = (snapshots / item["relative_path"]).read_text(encoding="utf-8-sig")
            raw_text = raw_path.read_text(encoding="utf-8-sig")
            raw_qc = check_morphosyntax_output(source, raw_text)
            final = merge_morphosyntax_tiers(source, raw_text)
            final_fingerprint = require_text_preserved(item["text_fingerprint"], final)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(final, encoding="utf-8")
            manifest["outputs"].append({"relative_path": item["relative_path"], "sha256": sha256_file(target),
                                        "raw_sha256": sha256_file(raw_path), **check_morphosyntax_output(source, final),
                                        "native_added_provenance": raw_qc["native_added_provenance"],
                                        "restored_source_provenance": raw_qc["restored_source_provenance"],
                                        "text_fingerprint": final_fingerprint})
        report = validation.validate_chat_directories({"annotated": annotated}, output / "validation/after", validator=selected)
        manifest["chatter"]["after"] = validation.validation_reference(report)
        validation.require_validated_report(manifest["chatter"]["after"])
        if _inventory(inputs) != [{key: item[key] for key in ("source_path", "relative_path", "source_sha256")} for item in inventory]:
            raise ValueError("Source files changed during annotation")
        if any(sha256_file(snapshots / item["relative_path"]) != item["snapshot_sha256"]
               or sha256_file(prepared / item["relative_path"]) != item["prepared_sha256"] for item in inventory):
            raise ValueError("Annotation input snapshots changed during processing")
        if upstream_annotation is not None:
            checked_talktag_collections(upstream_annotation)
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed_or_interrupted", error=str(error))
        raise
    finally:
        manifest["finished_at_utc"] = timestamp()
        atomic_json(path, manifest)
    return path
