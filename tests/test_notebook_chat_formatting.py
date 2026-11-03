"""The official notebook upgrades only known code and gates CHAT before Chatter."""
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from icl_pilot.pipeline import begin_pipeline, convert_csv_inputs, record_stage
from segmentation_stub import StubSegmenter


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/colab_direct_generation_pipeline.ipynb"


def notebook_cell(ident):
    notebook = json.loads(NOTEBOOK.read_text())
    return "".join(next(c["source"] for c in notebook["cells"] if c["id"] == ident))


@pytest.mark.parametrize("known_source", [True, False])
def test_notebook_source_upgrade_is_hash_pinned_and_records_provenance(tmp_path, known_source):
    project = tmp_path / "project"
    shutil.copytree(ROOT / "src", project / "src", ignore=shutil.ignore_patterns("__pycache__"))
    converter = project / "src/icl_pilot/chat/generated_chat.py"
    current = converter.read_text()
    start = current.index("    # CHAT rejects typographic single quotes")
    end = current.index('    substitute(r"\\(', start)
    old = current[:start] + current[end:]
    if known_source:
        assert hashlib.sha256(old.encode()).hexdigest() == "716f86a06a006920209e3b01aa6cbb0a44585526836e6c9a696bd45f3a7538a8"
    else:
        old += "\n# An unreviewed source revision must not be patched.\n"
    converter.write_text(old)
    run = tmp_path / "run"
    run.mkdir()
    (run / "pipeline_manifest.json").write_text(json.dumps({"stages": {}, "test_only": True}))
    script = tmp_path / "execute.py"
    setup = ("from pathlib import Path\nfrom icl_pilot.pipeline import record_stage\n"
             f"REPO_DIR = Path({str(project)!r})\nRUN_OUTPUT_DIR = Path({str(run)!r})\nRUN_CHAT_CONVERSION = True\n")
    script.write_text(setup + notebook_cell("chat-formatting-policy"))
    environment = dict(os.environ, PYTHONPATH=str(project / "src"), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, str(script)], env=environment, capture_output=True, text=True)
    if known_source:
        assert result.returncode == 0, result.stderr
        assert converter.read_text() == current
        report = json.loads((run / "chat_formatting_policy.json").read_text())
        assert report["reviewed_compatibility_update_applied"] is True
        assert report["source_before_sha256"] != report["source_after_sha256"]
        assert report["raw_generation_editing"] is report["hyphen_conversion"] is False
        manifest = json.loads((run / "pipeline_manifest.json").read_text())
        assert manifest["stages"]["chat_formatting_policy"]["status"] == "complete"
    else:
        assert result.returncode != 0
        assert "required audited apostrophe" in result.stderr
        assert converter.read_text() == old


def test_notebook_audits_derived_chat_and_blocks_illegal_apostrophes(tmp_path):
    from test_segmentation import saved_csv
    source = saved_csv(tmp_path / "source.csv", text="they’re playing.")
    raw = source.read_bytes()
    run = begin_pipeline(tmp_path / "run", mode="convert_only", csvs={"one": source},
                         chat_dirs={}, smoke_test=True)
    roots = convert_csv_inputs({"one": source}, run, roster_csv=None, segmenter=StubSegmenter())
    namespace = {"RUN_CHAT_CONVERSION": True, "RUN_OUTPUT_DIR": run, "chat_roots": roots,
                 "record_stage": record_stage, "Path": Path}
    code = notebook_cell("chat-formatting-check")
    exec(code, namespace)
    report = json.loads((run / "chat_formatting_manifest.json").read_text())
    assert report["files_checked"] == report["apostrophes_normalized"] == 2
    assert source.read_bytes() == raw

    # Even a self-consistent older conversion manifest must not hide E256 from
    # the pre-Chatter gate: it checks actual speech, not just stored hashes.
    manifest_path = roots["one"] / "chat_manifest.csv"
    with manifest_path.open(newline="") as handle:
        entries = list(csv.DictReader(handle))
    chat = Path(entries[0]["output_path"])
    chat.write_text(chat.read_text().replace("they're", "they’re"))
    entries[0]["chat_sha256"] = hashlib.sha256(chat.read_bytes()).hexdigest()
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=entries[0].keys())
        writer.writeheader()
        writer.writerows(entries)
    with pytest.raises(ValueError, match="Unconverted curly apostrophe"):
        exec(code, namespace)
    state = json.loads((run / "pipeline_manifest.json").read_text())
    assert state["stages"]["chat_formatting"]["status"] == "failed; Chatter/annotation blocked"
    assert source.read_bytes() == raw


def test_notebook_validation_error_includes_saved_chatter_diagnostics(tmp_path, monkeypatch):
    from icl_pilot.chat import annotation_preparation, validation
    failure_dir = tmp_path / "validation"
    failure_dir.mkdir()
    (failure_dir / "stdout.jsonl").write_text(json.dumps({
        "type": "file", "status": "invalid", "file": "/generated/story.cha",
        "errors": [{"code": "E256", "message": "Curly single quotation mark is illegal"}],
    }) + "\n")
    (failure_dir / "stderr.log").write_text("")
    manifest_path = failure_dir / "validation_manifest.json"
    manifest_path.write_text(json.dumps({
        "commands": [{"stdout": "stdout.jsonl", "stderr": "stderr.log"}], "issues": [],
    }))
    failure = validation.ChatterValidationError(manifest_path)
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(annotation_preparation, "prepare_annotation_collections", fail)
    monkeypatch.setattr(validation, "ensure_chatter", lambda **kwargs: {"path": "unused"})
    namespace = {"RUN_ANNOTATION": True, "MODE": "full", "RUN_CHAT_CONVERSION": False,
                 "RUN_OUTPUT_DIR": tmp_path, "REPO_DIR": ROOT, "INSTALL_DEPENDENCIES": False,
                 "chat_roots": {}, "ANNOTATION_STAGES": ["stage1_td"], "Path": Path, "json": json,
                 "annotation_inputs": lambda *args: {},
                 "annotation_collection_metadata": lambda *args: {},
                 "record_stage": lambda *args, **kwargs: None, "persist_run": lambda: None}
    (tmp_path / "pipeline_manifest.json").write_text(json.dumps({"test_only": True}))
    with pytest.raises(validation.ChatterValidationError, match="story.cha — E256"):
        exec(notebook_cell("annotation-preparation"), namespace)
    assert "Chatter diagnostic details:" in str(failure)


@pytest.mark.parametrize("baseline", ["original", "quote_restoration_v2", "automatic_v3_early", "unknown"])
def test_annotation_policy_upgrades_only_the_reviewed_source(tmp_path, baseline):
    code = notebook_cell("annotation-preservation-policy")
    tree = ast.parse(code)
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "reviewed_updates" for target in node.targets)]
    updates = ast.literal_eval(assignments[0].value)
    expected = {"original": "024647de64dc8d64183ea01d7ecc4471f2c03ddc1fa166eafaa96277e15aac13",
                "automatic_v3_early": "7d1c89eb5cc1f70a4b46c7b89597d37d741ab1f858674cbd99aa92cb39351620"
                }.get(baseline, "d4ac35ccab272fcda64d4ec2dd41e43438d5cdc1b33a3ed56f4b1355f439cf13")
    replacements = updates[expected]
    project = tmp_path / "project"
    shutil.copytree(ROOT / "src", project / "src", ignore=shutil.ignore_patterns("__pycache__"))
    module = project / "src/icl_pilot/chat/annotation.py"
    current = module.read_text()
    old = current
    for previous, fixed in reversed(replacements):
        assert old.count(fixed) == 1
        old = old.replace(fixed, previous, 1)
    assert hashlib.sha256(old.encode()).hexdigest() == expected
    if baseline == "unknown":
        old += "\n# Unreviewed code must not be rewritten.\n"
    module.write_text(old)
    run = tmp_path / "run"
    run.mkdir()
    (run / "pipeline_manifest.json").write_text(json.dumps({"stages": {}, "test_only": True}))
    setup = ("from pathlib import Path\nfrom icl_pilot.pipeline import record_stage\n"
             f"REPO_DIR = Path({str(project)!r})\nRUN_OUTPUT_DIR = Path({str(run)!r})\nRUN_ANNOTATION = True\n")
    script = tmp_path / "execute.py"
    script.write_text(setup + code)
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
        env=dict(os.environ, PYTHONPATH=str(project / "src"), PYTHONDONTWRITEBYTECODE="1"))
    if baseline != "unknown":
        assert result.returncode == 0, result.stderr
        assert module.read_text() == current
        report = json.loads((run / "annotation_preservation_policy.json").read_text())
        assert report["reviewed_compatibility_update_applied"] is True
        assert report["postprocessing_version"] == 3 and report["raw_talktag_outputs_edited"] is False
    else:
        assert result.returncode != 0
        assert module.read_text() == old
