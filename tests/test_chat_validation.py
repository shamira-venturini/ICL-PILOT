"""Native Chatter gates plus isolated coverage/installer failure checks."""
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import tarfile

import pytest

from icl_pilot.chat import annotation, chatter_runtime, validation

ROOT = Path(__file__).resolve().parents[1]
CHAT = ("@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n"
        "@ID:\teng|test|CHI|4;01.00||TD||Target_Child|||\n@Types:\tcross, narrative, TD\n"
        "@G:\tA1\n*CHI:\the go home .\n@End\n")


@pytest.fixture
def native():
    path = Path(os.environ.get("ICL_PILOT_CHATTER", ROOT / "tools/chatter/chatter"))
    if not path.is_file():
        pytest.skip("Native Chatter integration needs the pinned executable; no test downloads.")
    return chatter_runtime.verify_chatter(path)


def source(tmp_path, text=CHAT):
    folder = tmp_path / "source"
    folder.mkdir()
    (folder / "one.cha").write_text(text)
    return folder


def report(path):
    return json.loads(Path(path).read_text())


def test_native_checks_complete_collection_read_only_and_records_identity(tmp_path, native):
    folder = source(tmp_path)
    nested = folder / "nested/two.cha"
    nested.parent.mkdir()
    nested.write_text(CHAT.replace("he go", "her goed"))
    before = {str(p): p.read_bytes() for p in folder.rglob("*.cha")}
    path = validation.validate_chat_directories({"input": folder}, tmp_path / "qc", validator=native)
    checked = validation.require_validated_report(validation.validation_reference(path))
    assert checked["status"] == "passed" and checked["files_checked"] == 2
    assert checked["tool"]["version"] == "0.30.0"
    assert checked["tool"]["binary_sha256"] == native["binary_sha256"]
    args = checked["commands"][0]["command"]
    assert "--strict-linkers" not in args and "--force" in args and "tree-sitter" in args
    assert "--suppress" not in args and "--skip-alignment" not in args
    assert {str(p): p.read_bytes() for p in folder.rglob("*.cha")} == before


def test_invalid_native_input_blocks_with_retained_diagnostics(tmp_path, native):
    folder = source(tmp_path, CHAT.replace("@End\n", ""))
    with pytest.raises(validation.ChatterValidationError) as caught:
        validation.validate_chat_directories({"input": folder}, tmp_path / "qc", validator=native)
    checked = report(caught.value.manifest_path)
    assert checked["status"] == "failed"
    assert checked["commands"][0]["exit_code"] == 1
    assert (tmp_path / "qc/01_input.jsonl").is_file()
    assert (folder / "one.cha").read_text() == CHAT.replace("@End\n", "")


def test_native_alignment_checks_are_enabled(tmp_path, native):
    folder = source(tmp_path, CHAT.replace("@End", "%mor:\tpro|he v|go .\n@End"))
    with pytest.raises(validation.ChatterValidationError):
        validation.validate_chat_directories({"input": folder}, tmp_path / "qc", validator=native)
    assert report(tmp_path / "qc/validation_manifest.json")["status"] == "failed"


def test_source_quotation_and_completion_markers_remain_unchanged_without_optional_linker_rules(tmp_path, native):
    text = CHAT.replace("*CHI:\the go home .", '*CHI:\t+" he go home .\n*CHI:\t+, and come back .')
    folder = source(tmp_path, text)
    path = validation.validate_chat_directories({"input": folder}, tmp_path / "qc", validator=native)
    checked = validation.require_validated_report(validation.validation_reference(path))
    assert checked["status"] == "passed" and checked["files_checked"] == 1
    assert (folder / "one.cha").read_text() == text
    assert "--strict-linkers" not in checked["commands"][0]["command"]


@pytest.mark.parametrize("change", ["input", "extra_file", "log", "manifest"])
def test_stale_validation_cannot_be_presented_as_a_current_pass(tmp_path, native, change):
    folder = source(tmp_path)
    path = validation.validate_chat_directories({"input": folder}, tmp_path / "qc", validator=native)
    reference = validation.validation_reference(path)
    if change == "input": (folder / "one.cha").write_text(CHAT.replace("home", "away"))
    if change == "extra_file": (folder / "extra.cha").write_text(CHAT)
    if change == "log": (path.parent / "01_input.jsonl").write_text("changed")
    if change == "manifest": path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError): validation.require_validated_report(reference)


@pytest.mark.parametrize("mode", ["empty", "missing_file", "duplicate", "incomplete", "cached", "source_changed"])
def test_zero_exit_without_fresh_complete_coverage_does_not_pass(tmp_path, native, monkeypatch, mode):
    folder = source(tmp_path)
    real_run = subprocess.run
    def run(command, **kwargs):
        if "--version" in command:
            return real_run(command, **kwargs)
        event = {"type": "file", "file": str(folder / "one.cha"), "status": "valid", "cache_hit": mode == "cached"}
        summary = {"type": "summary", "outcome": "complete", "total_files": 1, "valid": 1,
                   "invalid": 0, "internal_failures": 0, "cache_hits": int(mode == "cached")}
        events = [] if mode == "empty" else [event, summary]
        if mode == "missing_file": events = [summary]
        if mode == "duplicate": events = [event, event, summary]
        if mode == "incomplete": summary["outcome"] = "interrupted"
        if mode == "source_changed": (folder / "one.cha").write_text(CHAT.replace("home", "away"))
        kwargs["stdout"].write("\n".join(json.dumps(item) for item in events) + "\n")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(validation.subprocess, "run", run)
    with pytest.raises(validation.ChatterValidationError):
        validation.validate_chat_directories({"input": folder}, tmp_path / "qc", validator=native)
    assert report(tmp_path / "qc/validation_manifest.json")["status"] == "failed"


def fake_talktag(monkeypatch, mode="normal"):
    calls, real_run = [], subprocess.run
    def run(command, **kwargs):
        if "--input-dir" not in command:
            return real_run(command, **kwargs)  # Real pinned Chatter including --version.
        calls.append(command)
        inputs = Path(command[command.index("--input-dir") + 1])
        outputs = Path(command[command.index("--output-dir") + 1])
        files = list(inputs.rglob("*.cha"))
        utterances = 0
        for path in files:
            text = path.read_text()
            utterances += sum(line.startswith("*CHI:") for line in text.splitlines())
            text = text.replace("go home", "go [* m:03s:a] home")
            if mode == "bad_tag": text = text.replace("[* m:03s:a]", "[:: ]")
            if mode == "header": text = text.replace("@G:\tA1", "@G:\tA2")
            if mode == "punctuation": text = text.replace("home .", "home ?")
            if mode == "structure": text = text.replace("@End\n", "")
            if mode == "orthography": text = text.replace("*CHI:\the", "*CHI:\tHe").replace("home .", "home.")
            destination = outputs / path.relative_to(inputs)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text)
        (outputs / "_talk_tag_report.json").write_text(json.dumps({"total_files": len(files),
            "processed_files": len(files), "failed_files": 0, "target_lines": utterances}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(annotation.subprocess, "run", run)
    return calls


def annotate(folder, out, native):
    config = annotation.load_annotation_config(ROOT / "configs/annotation/talktag.json")
    return annotation.annotate_chat_directory(folder, out, config=config, python="fixture-python",
        runtime={"source_commit": config["revision"]}, validator=native)


@pytest.mark.parametrize("mode", ["normal", "orthography"])
def test_native_before_and_after_talktag_gates_and_preservation(tmp_path, native, monkeypatch, mode):
    folder = source(tmp_path)
    original = (folder / "one.cha").read_bytes()
    calls = fake_talktag(monkeypatch, mode)
    path = annotate(folder, tmp_path / "annotation", native)
    checked = report(path)
    assert len(calls) == 1 and checked["status"] == "annotation_complete"
    assert set(checked["chatter"]) == {"before_sources", "before_snapshot", "raw_output", "after"}
    for key in ["before_sources", "before_snapshot", "after"]:
        assert validation.require_validated_report(checked["chatter"][key])["files_checked"] == 1
    assert (path.parent / "annotated/one.cha").read_text() == CHAT.replace("go home", "go [* m:03s:a] home")
    assert (folder / "one.cha").read_bytes() == original


def test_invalid_input_stops_before_any_talktag_call(tmp_path, native, monkeypatch):
    folder = source(tmp_path, CHAT.replace("@End\n", ""))
    calls = fake_talktag(monkeypatch)
    with pytest.raises(validation.ChatterValidationError): annotate(folder, tmp_path / "annotation", native)
    assert not calls
    checked = report(tmp_path / "annotation/annotation_manifest.json")
    assert checked["status"].startswith("failed")
    assert report(checked["chatter"]["before_sources"]["manifest"])["status"] == "failed"


def test_exact_snapshot_is_checked_before_talktag(tmp_path, native, monkeypatch):
    folder = source(tmp_path)
    calls = fake_talktag(monkeypatch)
    unfold = annotation._unfold_target
    monkeypatch.setattr(annotation, "_unfold_target", lambda text, target: unfold(text, target).replace("@End\n", ""))
    with pytest.raises(validation.ChatterValidationError): annotate(folder, tmp_path / "annotation", native)
    assert not calls
    checked = report(tmp_path / "annotation/annotation_manifest.json")
    assert report(checked["chatter"]["before_sources"]["manifest"])["status"] == "passed"
    assert report(checked["chatter"]["before_snapshot"]["manifest"])["status"] == "failed"


def test_postprocessing_saved_annotation_also_requires_native_checks(tmp_path, native, monkeypatch):
    folder = source(tmp_path)
    calls = fake_talktag(monkeypatch, "orthography")
    path = annotate(folder, tmp_path / "annotation", native)
    original = path.read_bytes()
    reviewed_path = annotation.postprocess_annotation_run(path, validator=native)
    reviewed = report(reviewed_path)
    assert reviewed["status"] == "annotation_complete" and len(calls) == 1
    for name in ["snapshot_recheck", "after"]:
        assert validation.require_validated_report(reviewed["chatter"][name])["status"] == "passed"
    assert path.read_bytes() == original


def test_annotation_labels_cannot_silently_overwrite_a_snapshot(tmp_path, native):
    first = source(tmp_path)
    nested = first / "other/one.cha"
    nested.parent.mkdir()
    nested.write_text(CHAT)
    second = tmp_path / "second"
    second.mkdir()
    (second / "one.cha").write_text(CHAT)
    config = annotation.load_annotation_config(ROOT / "configs/annotation/talktag.json")
    with pytest.raises(ValueError, match="colliding snapshot"):
        annotation.annotate_chat_directory({"": first, "other": second}, tmp_path / "annotation",
            config=config, python="unused", runtime={"source_commit": config["revision"]}, validator=native)
    assert not (tmp_path / "annotation").exists()


def test_malformed_added_tag_cannot_pass_only_surface_preservation(tmp_path, native, monkeypatch):
    folder = source(tmp_path)
    fake_talktag(monkeypatch, "bad_tag")
    with pytest.raises(validation.ChatterValidationError): annotate(folder, tmp_path / "annotation", native)
    checked = report(tmp_path / "annotation/annotation_manifest.json")
    assert checked["qc_issues"] == []  # The source words/punctuation were preserved.
    assert report(checked["chatter"]["after"]["manifest"])["status"] == "failed"
    assert "[:: ]" in (tmp_path / "annotation/raw_talktag/one.cha").read_text()


@pytest.mark.parametrize("mode,issue", [("header", "non_target_content_changed"),
    ("punctuation", "target_text_or_boundaries_changed"), ("structure", "non_target_content_changed")])
def test_preservation_blocks_valid_but_changed_content_as_well_as_broken_structure(tmp_path, native, monkeypatch, mode, issue):
    folder = source(tmp_path)
    fake_talktag(monkeypatch, mode)
    with pytest.raises(RuntimeError, match="QC needs review"): annotate(folder, tmp_path / "annotation", native)
    checked = report(tmp_path / "annotation/annotation_manifest.json")
    assert issue in {entry["issue"] for entry in checked["qc_issues"]}
    if mode in {"header", "punctuation"}:
        assert report(checked["chatter"]["raw_output"]["manifest"])["status"] == "passed"


def test_existing_reports_are_not_overwritten_and_missing_tool_fails_closed(tmp_path, native):
    folder = source(tmp_path)
    out = tmp_path / "qc"
    path = validation.validate_chat_directories({"input": folder}, out, validator=native)
    before = path.read_bytes()
    with pytest.raises(FileExistsError): validation.validate_chat_directories({"input": folder}, out, validator=native)
    assert path.read_bytes() == before
    with pytest.raises(validation.ChatterValidationError):
        validation.validate_chat_directories({"input": folder}, tmp_path / "missing_tool",
                                             validator={"path": str(tmp_path / "absent")})


def test_convert_only_notebook_still_requires_native_chatter_without_model_loading(tmp_path, native, monkeypatch):
    from segmentation_stub import StubSegmenter
    from icl_pilot.chat import segmentation
    monkeypatch.setattr(segmentation, "BatchalignSegmenter", StubSegmenter)
    import csv
    import sys
    notebook = report(ROOT / "notebooks/colab_direct_generation_pipeline.ipynb")
    def cell(name): return "".join(next(c for c in notebook["cells"] if c.get("id") == name)["source"])
    saved = tmp_path / "generation.csv"
    row = {"bundle_id": "pair_sli444_td410_rep01", "pair_id": "pair_sli444_td410", "target_story": "A1",
           "td_child_id": "410", "sli_child_id": "444", "td_age": "4;01.27", "sli_age": "4;02.04",
           "stage1_td_story": "he go home .", "stage2_sli_story": "he go home ."}
    with saved.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    namespace = {"Path": Path, "os": os, "sys": sys, "REPO_DIR": ROOT, "IS_COLAB": False}
    exec(cell("settings"), namespace)
    namespace.update(MODE="convert_only", RUN_GENERATION=False, RUN_CHAT_CONVERSION=True,
        RUN_ANNOTATION=False, GENERATION_CSVS={"age4": saved}, INSTALL_DEPENDENCIES=False,
        RUN_OUTPUT_DIR=tmp_path / "run", RUN_ID="convert", CHATTER_EXECUTABLE=native["path"], MAKE_ARCHIVE=False)
    for name in ["common-setup", "generation-input", "segmentation-setup", "chat-conversion", "chat-validation-before", "annotation-setup",
                 "annotation-run", "chat-validation-after", "export"]:
        exec(cell(name), namespace)
    stages = report(tmp_path / "run/pipeline_manifest.json")["stages"]
    assert stages["chat_validation_before"]["status"] == "complete"
    assert stages["segmentation"]["status"] == "complete"
    assert stages["annotation"]["status"] == stages["chat_validation_after"]["status"] == "skipped"
    checked = validation.require_validated_report(namespace["chat_validation_before"])
    assert checked["files_checked"] == 2 and "model" not in namespace


def test_generation_only_does_not_install_or_run_chatter(monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError("No CHAT files in generation-only mode")
    monkeypatch.setattr(validation, "ensure_chatter", forbidden)
    notebook = report(ROOT / "notebooks/colab_direct_generation_pipeline.ipynb")
    source = "".join(next(c for c in notebook["cells"] if c.get("id") == "chat-validation-before")["source"])
    exec(source, {"RUN_CHAT_CONVERSION": False, "RUN_ANNOTATION": False})


def test_config_refuses_reduced_checks_and_unknown_binary(tmp_path):
    config = chatter_runtime.load_chatter_config()
    config["validation_args"].append("--skip-alignment")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="required validation checks"): chatter_runtime.load_chatter_config(path)
    path.write_text("not a pinned executable")
    with pytest.raises(ValueError, match="differs from the pinned release"): chatter_runtime.verify_chatter(path)


@pytest.mark.parametrize("corrupt", [False, True])
def test_portable_installer_verifies_archive_and_executable_without_extracting_paths(tmp_path, monkeypatch, corrupt):
    payload = b"fixture CLI bytes"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:xz") as archive:
        info = tarfile.TarInfo("../../chatter")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    data = buffer.getvalue()
    config = chatter_runtime.load_chatter_config()
    config["assets"] = {"linux-x86_64": {"name": "talkbank-chatter-x86_64-unknown-linux-musl.tar.xz",
        "archive_sha256": "0" * 64 if corrupt else hashlib.sha256(data).hexdigest(),
        "binary_sha256": hashlib.sha256(payload).hexdigest()}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.delenv("ICL_PILOT_CHATTER", raising=False)
    monkeypatch.setattr(chatter_runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(chatter_runtime.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(chatter_runtime.shutil, "which", lambda name: None)
    monkeypatch.setattr(chatter_runtime.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(data))
    monkeypatch.setattr(chatter_runtime, "verify_chatter", lambda p, **kw: {"path": str(p)})
    destination = tmp_path / "runtime/linux-x86_64/chatter"
    if corrupt:
        with pytest.raises(ValueError, match="checksum"):
            chatter_runtime.ensure_chatter(config_path=path, cache_dir=tmp_path / "runtime")
        assert not destination.exists()
    else:
        result = chatter_runtime.ensure_chatter(config_path=path, cache_dir=tmp_path / "runtime")
        assert Path(result["path"]).read_bytes() == payload
        assert destination.stat().st_mode & 0o111
        assert not (tmp_path / "chatter").exists()
