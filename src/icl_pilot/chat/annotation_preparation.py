"""Shared, audited deletion-only CHAT preparation before fresh annotation."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil

from ..generation.recovery import atomic_json, timestamp
from .annotation import sha256_file
from . import validation


DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs/annotation/preparation.json"
INLINE_KINDS = ["error", "replacement", "correction", "gloss", "alternative", "comment", "interruption_comment", "postcode"]


def load_preparation_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text())
    expected = {"schema_version": 1, "policy": "shared-chat-annotation-preparation-v1",
                "dependent_tiers": "remove_all", "remove_inline": INLINE_KINDS,
                "preserve_postcodes": ["bch", "trn"], "preserve_headers": True,
                "preserve_main_tier_boundaries": True, "unwrap_scopes_without_retained_codes": True,
                "normalization": "none", "annotation_adjacent_whitespace": "delete_forbidden_only",
                "retain_source_snapshots": {"real": False, "synthetic": True, "unspecified": True}}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Use the reviewed shared annotation preparation policy")
    return config


def _hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _records(text):
    records, offset = [], 0
    for line in text.splitlines(keepends=True):
        if line.startswith(("@", "*", "%")):
            records.append({"start": offset, "text": line})
        elif records:
            records[-1]["text"] += line
        elif line.strip():
            raise ValueError("CHAT content precedes its first header/tier")
        offset += len(line)
    return records


def _brackets(body):
    result, stack = [], []
    for index, char in enumerate(body):
        if char == "[":
            stack.append(index)
        elif char == "]":
            if not stack:
                raise ValueError("Unmatched closing bracket in CHAT speech")
            start = stack.pop()
            if not stack:
                result.append((start, index + 1, body[start:index + 1]))
    if stack:
        raise ValueError("Unmatched opening bracket in CHAT speech")
    return result


def _removal_kind(code, config):
    content = code[1:-1].lstrip()
    if content.startswith("*"):
        return "error"
    if content.startswith("::") and (len(content) == 2 or content[2].isspace()):
        return "correction"
    if content.startswith(":") and (len(content) == 1 or content[1].isspace()):
        return "replacement"
    if content.startswith("=?") and (len(content) == 2 or content[2].isspace()):
        return "alternative"
    if content.startswith("=") and (len(content) == 1 or content[1].isspace()):
        return "gloss"
    if content.startswith("%") and (len(content) == 1 or content[1].isspace()):
        return "comment"
    if content.startswith("^") and (len(content) == 1 or content[1].isspace()):
        return "interruption_comment"  # Keep structural [^c] clause boundaries.
    if content.startswith("+") and (len(content) == 1 or content[1].isspace()):
        name = content[1:].strip()
        if name not in config["preserve_postcodes"]:
            return "postcode"
    return None


def _speech_tokens(body):
    parts, cursor = [], 0
    for start, end, code in _brackets(body):
        parts.extend((body[cursor:start], " "))
        cursor = end
    parts.append(body[cursor:])
    # Scope punctuation is not spoken material. Every other non-whitespace
    # token, including grammatical forms, timing and transcription symbols,
    # remains in the fingerprint. Never expand fragments or use target forms.
    plain = "".join(parts).replace("<", " ").replace(">", " ")
    return re.findall(r"\w+|[^\w\s]", plain)


def _signature(text):
    headers, main, codes = [], [], []
    for record in _records(text):
        value = record["text"]
        if value.startswith("@"):
            headers.append(value)
        elif value.startswith("*"):
            speaker, body = value.split(":", 1)
            main.append((speaker, _speech_tokens(body)))
            codes.append([(code, _hash(json.dumps(_speech_tokens(body[:start]), ensure_ascii=False)))
                          for start, end, code in _brackets(body)])
    return headers, main, codes


def _main_edits(record, config):
    value, base = record["text"], record["start"]
    colon = value.index(":") + 1
    body = value[colon:]
    brackets = _brackets(body)
    by_start = {start: (end, code, _removal_kind(code, config)) for start, end, code in brackets}
    edits = []
    for start, end, code in brackets:
        kind = _removal_kind(code, config)
        if kind:
            edits.append({"start": base + colon + start, "end": base + colon + end,
                          "kind": "inline_" + kind, "removed_text": code})
    # Unwrap only a scope whose adjacent annotations are all being removed.
    # A retained repetition/retracing/uncertainty code keeps its original scope.
    stack, pairs, cursor = [], [], 0
    while cursor < len(body):
        if cursor in by_start:
            cursor = by_start[cursor][0]
            continue
        char = body[cursor]
        if char == "<" and (cursor == 0 or body[cursor - 1] != "+"):
            stack.append(cursor)
        elif char == ">":
            if not stack:
                raise ValueError("Unmatched CHAT scope")
            pairs.append((stack.pop(), cursor))
        cursor += 1
    if stack:
        raise ValueError("Unmatched CHAT scope")
    for opening, closing in pairs:
        cursor, removed, retained = closing + 1, False, False
        while cursor < len(body):
            if body[cursor].isspace():
                cursor += 1
                continue
            annotation = by_start.get(cursor)
            if annotation is None:
                break
            end, code, kind = annotation
            removed |= kind is not None
            retained |= kind is None
            cursor = end
        if removed and not retained:
            for offset in (opening, closing):
                edits.append({"start": base + colon + offset, "end": base + colon + offset + 1,
                              "kind": "unwrapped_scope_delimiter", "removed_text": body[offset]})
    # Deleting a code can expose whitespace that CHAT forbids immediately
    # before '>' or after a tier's initial tab. Delete only whitespace adjacent
    # to a removed span; never normalize existing speech or wrapped lines.
    removed_positions = {i for edit in edits for i in range(edit["start"] - base, edit["end"] - base)}
    remaining = [(i, char) for i, char in enumerate(value) if i not in removed_positions]
    preview = "".join(char for i, char in remaining)
    ranges = [match.span() for match in re.finditer(r"[ \t]+(?=>)", preview)]
    initial = re.match(r"\*[^:]+:\t([ \t]+)", preview)
    if initial:
        ranges.append(initial.span(1))
    for start, end in ranges:
        first, last = remaining[start - 1][0] + 1, remaining[end][0]
        if not any(first <= i < last for i in removed_positions):
            continue
        for offset, char in remaining[start:end]:
            edits.append({"start": base + offset, "end": base + offset + 1,
                          "kind": "annotation_adjacent_whitespace", "removed_text": char})
    return edits


def clean_chat_annotations(text, config=None):
    config = load_preparation_config() if config is None else config
    edits = []
    for record in _records(text):
        value = record["text"]
        if value.startswith("%"):
            edits.append({"start": record["start"], "end": record["start"] + len(value),
                          "kind": "dependent_tier", "tier": value.split(":", 1)[0],
                          "removed_text_sha256": _hash(value)})
        elif value.startswith("*"):
            edits.extend(_main_edits(record, config))
    edits.sort(key=lambda item: item["start"])
    pieces, cursor = [], 0
    for edit in edits:
        if edit["start"] < cursor:
            raise ValueError("Overlapping preparation deletions")
        pieces.append(text[cursor:edit["start"]])
        cursor = edit["end"]
    pieces.append(text[cursor:])
    cleaned = "".join(pieces)
    before_headers, before_main, before_codes = _signature(text)
    after_headers, after_main, after_codes = _signature(cleaned)
    expected_codes = [[item for item in row if _removal_kind(item[0], config) is None] for row in before_codes]
    if before_headers != after_headers or before_main != after_main or expected_codes != after_codes:
        raise ValueError("Preparation changed headers, speech, boundaries or retained annotation anchors")
    counts = Counter(edit["kind"] for edit in edits)
    tiers = Counter(edit["tier"] for edit in edits if edit["kind"] == "dependent_tier")
    return cleaned, {"policy": config["policy"], "offset_unit": "unicode_codepoints",
                     "source_text_sha256": _hash(text), "prepared_text_sha256": _hash(cleaned),
                     "main_tiers": len(before_main), "chi_main_tiers": sum(speaker == "*CHI" for speaker, tokens in before_main),
                     "header_records": len(before_headers), "spoken_content_sha256": _hash(json.dumps(before_main, ensure_ascii=False)),
                     "removed": dict(counts), "removed_dependent_tiers": dict(tiers), "edits": edits}


def _inventory(inputs):
    records, seen = [], set()
    for label, value in inputs.items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", label):
            raise ValueError("Use simple unique collection labels")
        root = Path(value).expanduser().resolve()
        files = sorted(root.rglob("*.cha"))
        if not root.is_dir() or not files:
            raise ValueError("Each selected collection must contain CHAT files: " + str(root))
        for path in files:
            if path.resolve() in seen or path.is_symlink() or root not in path.resolve().parents:
                raise ValueError("Overlapping or escaping annotation inputs")
            seen.add(path.resolve())
            records.append({"collection": label, "source_path": str(path.resolve()),
                            "relative_path": (Path(label) / path.relative_to(root)).as_posix(),
                            "source_sha256": sha256_file(path)})
    if not records:
        raise ValueError("Select annotation collections")
    return records


def prepare_annotation_collections(inputs, output_dir, *, config_path=DEFAULT_CONFIG,
                                   collection_metadata=None, validator=None, test_only=False):
    """One shared pass for real and synthetic collections, preserving labels."""
    config = load_preparation_config(config_path)
    inputs = {key: Path(value).expanduser().resolve() for key, value in inputs.items()}
    inventory = _inventory(inputs)
    metadata = collection_metadata or {}
    if set(metadata) - set(inputs):
        raise ValueError("Collection metadata has unknown input labels")
    for value in metadata.values():
        if not isinstance(value, dict) or value.get("source_type", "unspecified") not in {"real", "synthetic", "unspecified"}:
            raise ValueError("Use explicit real/synthetic collection metadata")
    output = Path(output_dir).expanduser().resolve()
    if any(output == root or output in root.parents or root in output.parents for root in inputs.values()):
        raise ValueError("Preparation outputs must be separate from source trees")
    output.mkdir(parents=True, exist_ok=False)
    path = output / "preparation_manifest.json"
    snapshots, prepared, audits = (output / name for name in ("source_snapshot", "prepared", "audits"))
    manifest = {"schema_version": 1, "status": "running", "started_at_utc": timestamp(),
                "config": config, "test_only": bool(test_only), "wrapper_sha256": sha256_file(Path(__file__)),
                "collections": {label: {"source": str(root), "prepared": str(prepared / label),
                                          "original_snapshot_retained": config["retain_source_snapshots"][metadata.get(label, {}).get("source_type", "unspecified")],
                                          "metadata": metadata.get(label, {"source_type": "unspecified"})}
                                for label, root in inputs.items()}, "inputs": inventory, "outputs": [], "totals": {}}
    atomic_json(path, manifest)
    try:
        totals = Counter()
        for item in inventory:
            source = Path(item["source_path"])
            snapshot, target = snapshots / item["relative_path"], prepared / item["relative_path"]
            audit_path = (audits / item["relative_path"]).with_suffix(".json")
            retain_snapshot = manifest["collections"][item["collection"]]["original_snapshot_retained"]
            for destination in (target, audit_path):
                destination.parent.mkdir(parents=True, exist_ok=True)
            if retain_snapshot:
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, snapshot)
            source_bytes = (snapshot if retain_snapshot else source).read_bytes()
            if hashlib.sha256(source_bytes).hexdigest() != item["source_sha256"]:
                raise ValueError("Source changed during preparation")
            text = source_bytes.decode("utf-8-sig")
            cleaned, audit = clean_chat_annotations(text, config)
            from .preservation import text_fingerprint, require_text_preserved
            source_fingerprint = text_fingerprint(text, "speech")
            require_text_preserved(source_fingerprint, cleaned, compare_dependent=False)
            target.write_bytes(cleaned.encode("utf-8"))
            atomic_json(audit_path, audit)
            manifest["outputs"].append({"relative_path": item["relative_path"], "sha256": sha256_file(target),
                                         "source_pid": next(iter(re.findall(r"^@PID:[ \t]*([^\r\n]+)", text, re.M)), None),
                                         "audit": audit_path.relative_to(output).as_posix(), "audit_sha256": sha256_file(audit_path),
                                         "main_tiers": audit["main_tiers"], "chi_main_tiers": audit["chi_main_tiers"],
                                         "spoken_content_sha256": audit["spoken_content_sha256"], "removed": audit["removed"],
                                         "source_fingerprint": source_fingerprint,
                                         "prepared_fingerprint": text_fingerprint(cleaned)})
            totals.update(audit["removed"])
        report = validation.validate_chat_directories({"prepared": prepared}, output / "validation",
                                                     validator=validator or validation.ensure_chatter())
        manifest["validation"] = validation.validation_reference(report)
        validation.require_validated_report(manifest["validation"])
        if _inventory(inputs) != inventory:
            raise ValueError("Source files changed during preparation")
        manifest.update(status="prepared", totals=dict(totals), files=len(inventory))
    except BaseException as error:
        if isinstance(error, validation.ChatterValidationError):
            manifest["validation"] = validation.validation_reference(error.manifest_path)
        manifest.update(status="failed_or_interrupted", error=str(error), totals=dict(totals), files=len(inventory))
        raise
    finally:
        manifest["finished_at_utc"] = timestamp()
        atomic_json(path, manifest)
    return path


def preparation_reference(path):
    path = Path(path).resolve()
    return {"manifest": str(path), "sha256": sha256_file(path)}


def require_prepared_annotations(reference):
    path = Path(reference["manifest"])
    if sha256_file(path) != reference["sha256"]:
        raise ValueError("Annotation preparation manifest changed")
    record = json.loads(path.read_text())
    if record.get("status") != "prepared":
        raise ValueError("Annotation preparation did not complete")
    # Real corpus originals are referenced by identity/hash, not retained as run
    # dependencies. Older manifests retained all snapshots and keep that policy.
    retained = {label for label, item in record["collections"].items()
                if item.get("original_snapshot_retained", True)}
    sources = {label: item["source"] for label, item in record["collections"].items() if label in retained}
    if sources and _inventory(sources) != [item for item in record["inputs"] if item["collection"] in retained]:
        raise ValueError("Annotation source files changed since preparation")
    for item in record["inputs"]:
        if item["collection"] not in retained:
            source = Path(item["source_path"])
            if source.is_symlink() or (source.exists() and (not source.is_file() or sha256_file(source) != item["source_sha256"])):
                raise ValueError("Available real annotation source changed since preparation")
    output = path.parent
    expected = {item["relative_path"] for item in record["inputs"]}
    if ({item["relative_path"] for item in record["outputs"]} != expected
            or len(record["outputs"]) != len(expected)):
        raise ValueError("Preparation output inventory changed")
    if {p.relative_to(output / "prepared").as_posix() for p in (output / "prepared").rglob("*.cha")} != expected:
        raise ValueError("Prepared annotation file coverage changed")
    expected_snapshots = {item["relative_path"] for item in record["inputs"] if item["collection"] in retained}
    if {p.relative_to(output / "source_snapshot").as_posix() for p in (output / "source_snapshot").rglob("*.cha")} != expected_snapshots:
        raise ValueError("Original annotation snapshot coverage changed")
    for item in record["inputs"]:
        if item["collection"] in retained and sha256_file(output / "source_snapshot" / item["relative_path"]) != item["source_sha256"]:
            raise ValueError("Annotation source snapshot changed")
    for item in record["outputs"]:
        if (sha256_file(output / "prepared" / item["relative_path"]) != item["sha256"]
                or sha256_file(output / item["audit"]) != item["audit_sha256"]):
            raise ValueError("Prepared annotation text or deletion audit changed")
    validation.require_validated_report(record["validation"])
    return record


def checked_talktag_collections(reference):
    """Verify the prepared-input chain before handing fresh TalkTag to UD."""
    path = Path(reference["manifest"]).resolve()
    if sha256_file(path) != reference["sha256"]:
        raise ValueError("Upstream TalkTag manifest changed")
    record = json.loads(path.read_text())
    if record.get("status") != "annotation_complete" or not record.get("input_preparation"):
        raise ValueError("UD requires complete TalkTag with shared preparation provenance")
    prepared = require_prepared_annotations(record["input_preparation"])
    validation.require_validated_report(record["chatter"]["after"])
    root = path.parent / record["annotated_output_directory"]
    expected = {item["relative_path"] for item in prepared["outputs"]}
    if ({item["relative_path"] for item in record["outputs"]} != expected
            or len(record["outputs"]) != len(expected)
            or {p.relative_to(root).as_posix() for p in root.rglob("*.cha")} != expected):
        raise ValueError("Upstream TalkTag file coverage changed")
    for item in record["outputs"]:
        if sha256_file(root / item["relative_path"]) != item["sha256"]:
            raise ValueError("Upstream TalkTag output changed")
    return ({label: root / label for label in prepared["collections"]}, prepared, record)
