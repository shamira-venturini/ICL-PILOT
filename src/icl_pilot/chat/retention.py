"""Finalize verified annotations before removing redundant run-owned CHAT copies.

The receipt replaces live intermediate-file checks after compaction. Historical
stage manifests and diagnostics remain immutable; their deleted inputs are not
misrepresented as having been revalidated.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..generation.recovery import atomic_json, timestamp
from .annotation import sha256_file
from .annotation_preparation import checked_talktag_collections
from .corpus_archives import DEFAULT_CONFIG, ROOT, archive_member_for_pid, verified_archive
from .preservation import require_text_preserved, text_fingerprint
from . import validation

RECEIPT = "annotation_retention.json"


def _inside(root, relative):
    path = root / relative
    if (Path(relative).is_absolute() or ".." in Path(relative).parts or path.is_symlink()
            or any(p.is_symlink() for p in path.parents if p != root and root in p.parents)
            or root not in path.resolve().parents):
        raise ValueError("Retention paths must be ordinary files inside this run")
    return path


def _require_hash(path, digest):
    if not path.is_file() or sha256_file(path) != digest:
        raise ValueError("Retention checksum mismatch: " + str(path))


def _prune(root, record):
    # Preflight the entire set, including resumed cleanups, before unlinking.
    protected = {item["path"] for item in record["outputs"] + record["retained_audits"]}
    for item in record["removed"]:
        path = _inside(root, item["path"])
        if path.suffix != ".cha" or item["path"] in protected:
            raise ValueError("Only redundant CHAT intermediates may be removed")
        if path.exists():
            _require_hash(path, item["sha256"])
    for item in record["removed"]:
        path = _inside(root, item["path"])
        if path.exists():
            _require_hash(path, item["sha256"])
            path.unlink()
    # Empty directories only. Reports, diagnostics and unknown files survive.
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not path.is_symlink() and not any(path.iterdir()):
            path.rmdir()


def verify_retained_annotations(reference, *, run_root=None, allow_pruning=False):
    """Verify retained output and audit bytes without requiring deleted copies.

    ``reference`` is the checksum-anchored receipt saved in pipeline_manifest.
    ``run_root`` supports a relocated Drive copy or unpacked export.
    """
    receipt = Path(reference["manifest"])
    root = Path(run_root).resolve() if run_root is not None else receipt.resolve().parent
    if run_root is not None:
        receipt = root / RECEIPT
    _require_hash(receipt, reference["sha256"])
    record = json.loads(receipt.read_text())
    if record.get("status") not in ({"complete", "pruning"} if allow_pruning else {"complete"}):
        raise ValueError("Annotation retention did not complete")
    expected = {item["path"] for item in record["outputs"]}
    actual = {p.relative_to(root).as_posix() for p in _inside(root, record["final_directory"]).rglob("*.cha")}
    if not expected or actual != expected:
        raise ValueError("Final retained CHAT coverage changed")
    for item in record["outputs"]:
        path = _inside(root, item["path"])
        _require_hash(path, item["sha256"])
        require_text_preserved(item["fingerprint"], path.read_text(encoding="utf-8-sig"))
    for item in record["retained_audits"]:
        _require_hash(_inside(root, item["path"]), item["sha256"])
    if record["status"] == "complete" and any(_inside(root, item["path"]).exists() for item in record["removed"]):
        raise ValueError("Completed retention still contains declared intermediate copies")
    return record


def finalize_annotation_run(run_root, *, annotation_manifest, morphosyntax_manifest=None,
                            archive_paths=None, project_root=ROOT, config_path=DEFAULT_CONFIG):
    """Compact a complete shared-preparation/TalkTag[/UD] run, never a failure.

    A missing ZIP, text mismatch, incomplete stage or changed input raises before
    cleanup. Interrupted pruning is resumable using the recorded exact file set.
    External input folders and original generation outputs are never removed.
    """
    root = Path(run_root).resolve()
    receipt = root / RECEIPT
    if receipt.exists():
        reference = {"manifest": str(receipt), "sha256": sha256_file(receipt)}
        record = verify_retained_annotations(reference, allow_pruning=True)
        if record["status"] == "pruning":
            _prune(root, record)
            record.update(status="complete", completed_at_utc=timestamp())
            atomic_json(receipt, record)
        return {"manifest": str(receipt), "sha256": sha256_file(receipt)}

    talk_path = Path(annotation_manifest).resolve()
    _inside(root, talk_path.relative_to(root))
    talk_ref = {"manifest": str(talk_path), "sha256": sha256_file(talk_path)}
    _, prepared, talk = checked_talktag_collections(talk_ref)
    prep_path = Path(talk["input_preparation"]["manifest"]).resolve()
    _inside(root, prep_path.relative_to(root))
    prep_root, talk_root = prep_path.parent, talk_path.parent
    prepared_outputs = {item["relative_path"]: item for item in prepared["outputs"]}
    expected = set(prepared_outputs)
    final_root = talk_root / talk["annotated_output_directory"]
    final = talk
    ud_path = None
    if morphosyntax_manifest is not None:
        ud_path = Path(morphosyntax_manifest).resolve()
        _inside(root, ud_path.relative_to(root))
        final = json.loads(ud_path.read_text())
        if final.get("status") != "complete" or final.get("upstream_annotation") != talk_ref:
            raise ValueError("Complete UD with matching TalkTag provenance is required before cleanup")
        validation.require_validated_report(final["chatter"]["before"])
        validation.require_validated_report(final["chatter"]["after"])
        final_root = ud_path.parent / final["annotated_output_directory"]
    if ({item["relative_path"] for item in final["outputs"]} != expected
            or len(final["outputs"]) != len(expected)
            or {p.relative_to(final_root).as_posix() for p in final_root.rglob("*.cha")} != expected):
        raise ValueError("Final annotation coverage differs from shared preparation")

    archives, originals = {}, []
    prep_inputs = {item["relative_path"]: item for item in prepared["inputs"]}
    outputs = []
    for item in final["outputs"]:
        relative = item["relative_path"]
        path, baseline = final_root / relative, prepared_outputs[relative]
        _require_hash(path, item["sha256"])
        text = path.read_text(encoding="utf-8-sig")
        # Link final words, boundaries and headers back to pre-cleaning real or
        # generated input. Only documented inline additions/removals are ignored.
        require_text_preserved(baseline["source_fingerprint"], text, compare_dependent=False)
        require_text_preserved(baseline["prepared_fingerprint"], text)
        require_text_preserved(item["text_fingerprint"], text)
        outputs.append({"path": path.relative_to(root).as_posix(), "sha256": item["sha256"],
                        "fingerprint": text_fingerprint(text, "exact")})
        collection = prepared["collections"][prep_inputs[relative]["collection"]]
        metadata = collection.get("metadata", {})
        if metadata.get("source_type") == "real":
            corpus = metadata.get("corpus")
            if not corpus or not baseline.get("source_pid"):
                raise ValueError("Real-data cleanup requires explicit corpus and original CHAT PID")
            if corpus not in archives:
                archives[corpus] = verified_archive(corpus, project_root=project_root, paths=archive_paths, config_path=config_path)
            originals.append({"relative_path": relative, "pid": baseline["source_pid"], "corpus": corpus,
                              "source_sha256": prep_inputs[relative]["source_sha256"],
                              **archive_member_for_pid(archives[corpus], baseline["source_pid"])})

    candidates = {}
    def add(folder, records, key):
        wanted = {item["relative_path"] for item in records}
        actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*.cha")}
        if actual != wanted:
            raise ValueError("Intermediate CHAT coverage changed: " + str(folder))
        for item in records:
            path = folder / item["relative_path"]
            relative = path.relative_to(root).as_posix()
            _inside(root, relative)
            _require_hash(path, item[key])
            candidates[relative] = item[key]

    retained_snapshots = [item for item in prepared["inputs"]
                          if prepared["collections"][item["collection"]].get("original_snapshot_retained", True)]
    add(prep_root / "source_snapshot", retained_snapshots, "source_sha256")
    add(prep_root / "prepared", prepared["outputs"], "sha256")
    add(talk_root / "input_snapshot", talk["inputs"], "snapshot_sha256")
    add(talk_root / "raw_talktag", talk["raw_outputs"], "sha256")
    if ud_path is not None:
        add(talk_root / talk["annotated_output_directory"], talk["outputs"], "sha256")
        add(ud_path.parent / "input_snapshot", final["inputs"], "snapshot_sha256")
        add(ud_path.parent / "prepared", final["inputs"], "prepared_sha256")
        add(ud_path.parent / "raw_batchalign", final["outputs"], "raw_sha256")
        for item in final["inputs"]:
            _require_hash(Path(item["source_path"]), item["source_sha256"])

    # Keep every non-CHAT stage artifact, including deletion audits, runtime
    # identities, manifests, tool reports and failed raw diagnostics.
    stage_roots = [prep_root, talk_root] + ([ud_path.parent] if ud_path is not None else [])
    audits = [{"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
              for stage in stage_roots for path in sorted(stage.rglob("*"))
              if path.is_file() and path.suffix != ".cha"]
    record = {"schema_version": 1, "policy": "validated-annotation-retention-v1", "status": "pruning",
              "verified_at_utc": timestamp(), "final_directory": final_root.relative_to(root).as_posix(),
              "outputs": outputs, "retained_audits": audits, "original_archives": archives,
              "original_members": originals,
              "removed": [{"path": key, "sha256": value} for key, value in sorted(candidates.items())],
              "note": "Stage validations describe pre-cleanup checks. Use this receipt to verify retained files."}
    atomic_json(receipt, record)
    reference = {"manifest": str(receipt), "sha256": sha256_file(receipt)}
    verify_retained_annotations(reference, allow_pruning=True)
    _prune(root, record)
    record.update(status="complete", completed_at_utc=timestamp())
    atomic_json(receipt, record)
    reference["sha256"] = sha256_file(receipt)
    verify_retained_annotations(reference)
    return reference
