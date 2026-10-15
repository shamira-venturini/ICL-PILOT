from __future__ import annotations

import ast
import csv
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from icl_pilot.annotation import annotate_chat_directory, install_annotation_runtime, load_annotation_config
from icl_pilot.pipeline import annotation_inputs, begin_pipeline, convert_csv_inputs, package_run, stage_flags, sync_run


ROOT = Path(__file__).resolve().parents[1]
CHAT = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n*CHI:\the go\n\thome .\n%mor:\tpro|he v|go n|home .\n%gra:\t1|2|SUBJ 2|0|ROOT 3|2|OBJ\n@End\n"


@pytest.fixture
def config():
    return load_annotation_config(ROOT / "configs/annotation/talktag.json")


@pytest.mark.parametrize("partial", [False, True])
def test_annotation_install_skips_ensurepip_and_targets_separate_environment(tmp_path, config, monkeypatch, partial):
    from icl_pilot import annotation

    environment = tmp_path / "talktag_env"
    python = environment / "bin/python"
    if partial:
        python.parent.mkdir(parents=True)
        python.write_text("partial environment")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(annotation, "sys", SimpleNamespace(version_info=(3, 13), platform="linux",
                                                          executable="/host/python"))
    monkeypatch.setattr(annotation.subprocess, "run", run)
    assert install_annotation_runtime(environment, config) == python
    assert calls[0][0] == ["/host/python", "-m", "venv", "--system-site-packages", "--without-pip", str(environment)]
    assert calls[0][1]["capture_output"] and calls[0][1]["text"]
    assert calls[1][0] == ["/host/python", "-m", "pip", "--python", str(python), "install",
        f"talk-tag[runtime] @ git+{config['repository']}@{config['revision']}", "transformers>=4.52,<5"]
    if partial:
        assert python.read_text() == "partial environment"  # No directory deletion/clear step.


def test_annotation_environment_failure_includes_underlying_diagnostic(tmp_path, config, monkeypatch):
    import subprocess
    from icl_pilot import annotation

    calls = []

    def run(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command, output="setup failed", stderr="No module named venv")

    monkeypatch.setattr(annotation, "sys", SimpleNamespace(version_info=(3, 13), platform="linux",
                                                          executable="/host/python"))
    monkeypatch.setattr(annotation.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="No module named venv"):
        install_annotation_runtime(tmp_path / "environment", config)
    assert len(calls) == 1  # No package install after unsuccessful environment creation.


def fake_tool(monkeypatch, *, mode="normal", limit=0):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        source = Path(command[command.index("--input-dir") + 1])
        out = Path(command[command.index("--output-dir") + 1])
        if mode == "exit_failure":
            kwargs["stdout"].write("fixture failure\n")
            return SimpleNamespace(returncode=1)
        files = list(source.rglob("*.cha"))
        total = 0
        for path in files:
            text = path.read_text()
            total += sum(line.startswith("*CHI:") for line in text.splitlines())
            destination = out / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if mode == "missing_output":
                continue
            annotated = text.replace("go home", "go [* m:03s:a] home")
            if mode == "text_drift":
                annotated = annotated.replace("home", "away")
            if mode == "tier_loss":
                annotated = "\n".join(line for line in annotated.splitlines() if not line.startswith("%gra:")) + "\n"
            destination.write_text(annotated)
        out.mkdir(exist_ok=True)
        if mode == "extra_output":
            (out / "extra.cha").write_text(CHAT)
        report = {"total_files": len(files), "processed_files": len(files), "failed_files": 0,
                  "target_lines": min(total, limit) if limit else total}
        if mode == "incomplete_coverage":
            report["target_lines"] = 0
        (out / "_talk_tag_report.json").write_text(json.dumps(report))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("icl_pilot.annotation.subprocess.run", run)
    return calls


def annotate(tmp_path, config, **kwargs):
    source = tmp_path / "source"
    source.mkdir()
    (source / "sample.cha").write_text(CHAT)
    return annotate_chat_directory(source, tmp_path / "annotation", config=config, python="fake_python",
                                    runtime={"source_commit": config["revision"]}, **kwargs)


def test_annotation_uses_copy_unfolds_main_tier_and_preserves_source_and_dependencies(tmp_path, config, monkeypatch):
    calls = fake_tool(monkeypatch)
    path = annotate(tmp_path, config)
    manifest = json.loads(path.read_text())
    assert manifest["status"] == "annotation_complete"
    assert manifest["qc_issues"] == []
    assert (tmp_path / "source/sample.cha").read_text() == CHAT
    snapshot = tmp_path / "annotation/input_snapshot/sample.cha"
    assert "*CHI:\the go home .\n%mor:" in snapshot.read_text()
    assert manifest["inputs"][0]["source_sha256"] != manifest["inputs"][0]["snapshot_sha256"]
    output = (tmp_path / "annotation/annotated/sample.cha").read_text()
    assert "go [* m:03s:a] home" in output and "%mor:" in output and "%gra:" in output
    assert str(snapshot.parent) in calls[0] and "--fail-fast" in calls[0]


@pytest.mark.parametrize("mode,issue", [
    ("text_drift", "target_text_or_boundaries_changed"), ("tier_loss", "non_target_content_changed"),
    ("missing_output", "missing_output"), ("extra_output", "unexpected_output"),
    ("incomplete_coverage", "incomplete_target_coverage"),
])
def test_qc_failures_retain_outputs_and_report_issues(tmp_path, config, monkeypatch, mode, issue):
    fake_tool(monkeypatch, mode=mode)
    with pytest.raises(RuntimeError, match="QC needs review"):
        annotate(tmp_path, config)
    manifest = json.loads((tmp_path / "annotation/annotation_manifest.json").read_text())
    assert manifest["status"].startswith("failed_or_interrupted")
    assert issue in {item["issue"] for item in manifest["qc_issues"]}
    assert (tmp_path / "source/sample.cha").read_text() == CHAT


def test_cli_failure_retains_log_and_attempt_manifest(tmp_path, config, monkeypatch):
    fake_tool(monkeypatch, mode="exit_failure")
    with pytest.raises(RuntimeError, match="TalkTag failed"):
        annotate(tmp_path, config)
    manifest = json.loads((tmp_path / "annotation/annotation_manifest.json").read_text())
    assert manifest["exit_code"] == 1 and manifest["status"].startswith("failed")
    assert "fixture failure" in (tmp_path / "annotation/talktag.log").read_text()


def test_debug_limit_is_not_reported_as_complete_annotation(tmp_path, config, monkeypatch):
    config = {**config, "limit": 1}
    fake_tool(monkeypatch, limit=1)
    path = annotate(tmp_path, config)
    assert json.loads(path.read_text())["status"] == "partial_debug"


def test_existing_annotation_output_is_never_replaced(tmp_path, config):
    source = tmp_path / "source"
    source.mkdir()
    (source / "file.cha").write_text(CHAT)
    out = tmp_path / "out"
    out.mkdir()
    sentinel = out / "existing.cha"
    sentinel.write_text("existing annotation")
    with pytest.raises(FileExistsError):
        annotate_chat_directory(source, out, config=config, python="unused")
    assert sentinel.read_text() == "existing annotation"


def test_annotation_output_cannot_be_inside_source_tree(tmp_path, config):
    with pytest.raises(ValueError, match="separate directory trees"):
        annotate_chat_directory(tmp_path, tmp_path / "output", config=config, python="unused")


def test_incorrect_runtime_commit_is_rejected_before_writing(tmp_path, config):
    source = tmp_path / "source"
    source.mkdir()
    (source / "file.cha").write_text(CHAT)
    with pytest.raises(ValueError, match="commit"):
        annotate_chat_directory(source, tmp_path / "out", config=config, python="unused",
                                runtime={"source_commit": "incorrect"})
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("mode,flags", [
    ("full", (True, True, True)), ("generation_only", (True, False, False)),
    ("from_csv", (False, True, True)), ("convert_only", (False, True, False)),
    ("from_chat", (False, False, True)),
])
def test_routes_enable_only_the_selected_stages(mode, flags):
    assert stage_flags(mode) == flags


def test_missing_selected_inputs_and_stages_fail_explicitly(tmp_path):
    with pytest.raises(ValueError, match="GENERATION_CSVS"):
        begin_pipeline(tmp_path / "run", mode="from_csv", csvs={}, chat_dirs={})
    assert not (tmp_path / "run").exists()
    root = tmp_path / "chat"
    (root / "stage2_sli").mkdir(parents=True)
    (root / "stage2_sli/file.cha").write_text(CHAT)
    with pytest.raises(FileNotFoundError, match="missing"):
        annotation_inputs({"age4": root}, ["stage1_td", "stage2_sli"])


def test_archives_include_annotation_logs_inputs_and_qc_without_recursive_archives(tmp_path, config, monkeypatch):
    fake_tool(monkeypatch)
    annotate(tmp_path, config)
    first = package_run(tmp_path)
    second = package_run(tmp_path)
    with zipfile.ZipFile(second) as archive:
        names = archive.namelist()
        assert "annotation/annotation_manifest.json" in names
        assert "annotation/input_snapshot/sample.cha" in names
        assert "annotation/talktag.log" in names
        assert all(not name.startswith("exports/") for name in names)
    copied = sync_run(tmp_path, tmp_path.parent / (tmp_path.name + "_copy"))
    assert (copied / first.relative_to(tmp_path)).read_bytes() == first.read_bytes()


@pytest.mark.parametrize("mode,smoke", [("full", False), ("full", True), ("from_csv", False), ("from_chat", False)])
def test_notebook_can_start_from_saved_inputs_in_a_fresh_namespace(tmp_path, config, monkeypatch, mode, smoke):
    calls = fake_tool(monkeypatch)
    monkeypatch.setattr("icl_pilot.annotation.annotation_runtime_metadata",
                        lambda python, cfg: {"source_commit": cfg["revision"]})
    source_csv = tmp_path / "generation.csv"
    row = {"bundle_id": "pair_sli444_td410_rep01", "pair_id": "pair_sli444_td410", "target_story": "A1",
           "td_child_id": "410", "sli_child_id": "444", "td_age": "4;01.27", "sli_age": "4;02.04",
           "stage1_td_story": "he go home .", "stage2_sli_story": "he go home ."}
    with source_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    source_chat = tmp_path / "chat_source"
    source_chat.mkdir()
    (source_chat / "sample.cha").write_text(CHAT)
    input_config = tmp_path / "input_config.json"
    input_config.write_text(json.dumps({"inputs": {"narratives_csv": "generation.csv"}}))
    original_csv, original_chat = source_csv.read_bytes(), (source_chat / "sample.cha").read_bytes()
    generate, convert, annotate_enabled = stage_flags(mode)
    namespace = {
        "Path": Path, "sys": sys, "MODE": mode, "SAVE_TO_DRIVE": False,
        "GENERATION_CSVS": {"age4": source_csv}, "CHAT_INPUT_DIRS": {"existing": source_chat},
        "RUN_OUTPUT_DIR": tmp_path / "run", "RUN_ID": "fixture", "RUN_GENERATION": generate,
        "RUN_CHAT_CONVERSION": convert, "RUN_ANNOTATION": annotate_enabled,
        "ROSTER_CSV": ROOT / "configs/generation/master_frozen_roster_4_to_7_year_old.csv",
        "ANNOTATION_STAGES": ["stage1_td", "stage2_sli"], "INSTALL_DEPENDENCIES": False,
        "ANNOTATION_ENV": tmp_path / "not_installed_fixture",
        "ANNOTATION_CONFIG_PATH": ROOT / "configs/annotation/talktag.json",
        "MAKE_ARCHIVE": True, "DOWNLOAD_ARCHIVE": False, "RELEASE_COLAB_RUNTIME": False,
        "IS_COLAB": False,
        "CONFIG_PATH": input_config, "DATA_ROOT": tmp_path, "GENERATION_INPUT_UPLOAD": None,
        "SAVE_EVERY": 1,
        "SMOKE_TEST": smoke,
    }
    notebook = json.loads((ROOT / "notebooks/colab_direct_generation_pipeline.ipynb").read_text())
    ids = ["common-setup", "generation-input", "generation-setup", "generation-model", "generation-function", "generation-preview",
           "generation-run", "generation-copy-check", "release-generator", "chat-conversion", "annotation-setup", "annotation-run", "export"]
    for name in ids:
        if mode == "full" and name in {"generation-setup", "generation-model", "generation-function", "generation-preview"}:
            # Inference is a fixture; exercise the actual generation runner,
            # conversion, annotation wrapper, QC and export cells together.
            continue
        cell = next(c for c in notebook["cells"] if c.get("id") == name)
        exec("".join(cell["source"]), namespace)
        if mode == "full" and name == "common-setup":
            import pandas as pd
            from icl_pilot.generation.prompts import load_prompt_builder
            builder = load_prompt_builder(ROOT / "configs/generation/experiment.json")
            generation_row = {**row, "cohort": "4-year-old", "cohort_label": "4-year-old",
                "round_id": "round_01", "replicate_id": 1, "story_slot_order": 1,
                "story_title": "Fixture", "story_units_text": "1. Fixture event.",
                "content_invariants_text": "Fixture", "optional_content_text": "None",
                "prompt_constraints_text": "Fixture", "prompt_td_story_e1": "B1", "prompt_sli_story_e2": "B3",
                "prompt_td_exemplar_story": "he go .", "prompt_sli_exemplar_story": "he go ."}
            manifest = tmp_path / "run/run_manifest.json"
            manifest.write_text(json.dumps({"execution": {"test_only": smoke}}))
            namespace["prepared"] = SimpleNamespace(run_manifest_path=manifest, prompt_builder=builder,
                config={"model": {"max_new_tokens_stage1": 180, "max_new_tokens_stage2": 180}},
                run_df=pd.DataFrame([generation_row]))
            def generate(messages, *, row, stage, journal, max_new_tokens, parent_record_id=None, stage1_text=None):
                record_id = journal.record_prompt(row=row, stage=stage, messages=messages,
                    rendered_prompt=json.dumps(messages), input_token_ids=[1], generation_parameters={},
                    parent_record_id=parent_record_id, stage1_text=stage1_text)
                journal.record_output(record_id, "he go home .", generated_token_count=4)
                return "he go home .", record_id
            namespace["generate_from_messages"] = generate
    assert "model" not in namespace
    if mode != "full":
        assert "prepared" not in namespace
    manifest = json.loads((tmp_path / "run/pipeline_manifest.json").read_text())
    assert manifest["stages"]["generation"]["status"] == ("complete" if mode == "full" else "skipped")
    assert manifest["stages"]["annotation"]["status"] == "complete"
    assert manifest["test_only"] == (smoke and namespace["RUN_GENERATION"])
    assert source_csv.read_bytes() == original_csv and (source_chat / "sample.cha").read_bytes() == original_chat
    annotation = json.loads((tmp_path / "run/annotation/annotation_manifest.json").read_text())
    assert len(annotation["inputs"]) == (1 if mode == "from_chat" else 2)
    assert len(list((tmp_path / "run/annotation").rglob("annotation_manifest.json"))) == 1
    assert len(calls) == 1  # One annotation model load for the selected collection.
    if mode == "full":
        import pandas as pd
        saved = pd.read_csv(tmp_path / "run/all_cohorts_rotating_story_generations.csv")
        assert saved["test_only"].eq(smoke).all()


def test_all_notebook_code_is_valid_python_and_cell_ids_are_unique():
    notebook = json.loads((ROOT / "notebooks/colab_direct_generation_pipeline.ipynb").read_text())
    ids = [c["id"] for c in notebook["cells"]]
    assert len(ids) == len(set(ids))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))


def notebook_cell(name):
    notebook = json.loads((ROOT / "notebooks/colab_direct_generation_pipeline.ipynb").read_text())
    return "".join(next(cell["source"] for cell in notebook["cells"] if cell.get("id") == name))


@pytest.mark.parametrize("mode", ["full", "from_csv", "convert_only"])
def test_stage_copy_check_flags_identity_without_changing_inputs(tmp_path, capsys, mode):
    path = tmp_path / "generated.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pair_id", "target_story", "stage1_td_story", "stage2_sli_story"])
        writer.writeheader()
        for story, first, second in [("A1", "he goes .", "he goes ."),
                                     ("A2", "he goes .", "he\n goes   ."),
                                     ("A3", "he goes .", "he go .")]:
            writer.writerow({"pair_id": "fixture", "target_story": story,
                             "stage1_td_story": first, "stage2_sli_story": second})
    before = path.read_bytes()
    ns = {"RUN_GENERATION": mode == "full", "MODE": mode, "generation_csvs": {"fixture": path}}
    exec(notebook_cell("generation-copy-check"), ns)
    assert ns["stage2_copy_summary"] == [{"source_csv": str(path), "jobs": 3, "unchanged_after_whitespace": 2}]
    assert "2/3 jobs" in capsys.readouterr().out
    assert path.read_bytes() == before
    assert sorted(tmp_path.iterdir()) == [path]  # No resampling, filtering or replacement artifacts.


def test_stage_copy_check_skips_chat_only_routes():
    ns = {"RUN_GENERATION": False, "MODE": "from_chat"}
    exec(notebook_cell("generation-copy-check"), ns)
    assert ns["stage2_copy_summary"] == []


@pytest.fixture
def input_namespace(tmp_path):
    config_path = tmp_path / "experiment.json"
    config_path.write_text(json.dumps({"inputs": {"narratives_csv": "data/processed/exemplars.csv"}}))
    upload = tmp_path / "uploaded.csv"
    upload.write_bytes(b"File_ID,Group,story_id\n410,TD,A1\n")
    out = begin_pipeline(tmp_path / "run", mode="generation_only", csvs={}, chat_dirs={})
    return {"Path": Path, "json": json, "RUN_GENERATION": True, "CONFIG_PATH": config_path,
            "DATA_ROOT": tmp_path / "drive_inputs", "GENERATION_INPUT_UPLOAD": upload,
            "RUN_OUTPUT_DIR": out, "persist_run": lambda: None}


def test_notebook_input_upload_is_verified_and_reused_without_the_upload(input_namespace):
    from icl_pilot.annotation import sha256_file

    ns = input_namespace
    source = ns["GENERATION_INPUT_UPLOAD"]
    original = source.read_bytes()
    exec(notebook_cell("generation-input"), ns)
    target = ns["DATA_ROOT"] / "data/processed/exemplars.csv"
    assert target.read_bytes() == source.read_bytes() == original
    manifest_path = ns["RUN_OUTPUT_DIR"] / "pipeline_manifest.json"
    assert json.loads(manifest_path.read_text())["generation_input"] == {
        "path": str(target), "sha256": sha256_file(source), "action": "copied_and_verified"}
    before = target.stat().st_mtime_ns
    exec(notebook_cell("generation-input"), ns)  # An identical upload does not replace the input.
    assert target.stat().st_mtime_ns == before
    ns["GENERATION_INPUT_UPLOAD"] = None  # A later runtime has only the persistent copy.
    exec(notebook_cell("generation-input"), ns)
    assert target.stat().st_mtime_ns == before
    assert json.loads(manifest_path.read_text())["generation_input"]["action"] == "reused"


def test_notebook_rejects_different_upload_without_overwriting_existing_input(input_namespace):
    ns = input_namespace
    exec(notebook_cell("generation-input"), ns)
    target = ns["DATA_ROOT"] / "data/processed/exemplars.csv"
    original = target.read_bytes()
    ns["GENERATION_INPUT_UPLOAD"].write_bytes(b"different corpus\n")
    with pytest.raises(FileExistsError, match="differs"):
        exec(notebook_cell("generation-input"), ns)
    assert target.read_bytes() == original
    assert ns["GENERATION_INPUT_UPLOAD"].read_bytes() == b"different corpus\n"


def test_missing_generation_input_explains_one_time_upload_without_creating_input_tree(input_namespace):
    ns = input_namespace
    ns["GENERATION_INPUT_UPLOAD"] = ns["DATA_ROOT"].parent / "absent.csv"
    with pytest.raises(FileNotFoundError, match="Upload to Colab"):
        exec(notebook_cell("generation-input"), ns)
    assert not ns["DATA_ROOT"].exists()


@pytest.mark.parametrize("path", ["../outside.csv", "/absolute.csv"])
def test_notebook_input_upload_cannot_escape_selected_root(input_namespace, path):
    ns = input_namespace
    ns["CONFIG_PATH"].write_text(json.dumps({"inputs": {"narratives_csv": path}}))
    with pytest.raises(ValueError, match="inside DATA_ROOT"):
        exec(notebook_cell("generation-input"), ns)
    assert not ns["DATA_ROOT"].exists()


def test_interrupted_input_copy_preserves_upload_and_cleans_partial_destination(input_namespace, monkeypatch):
    ns = input_namespace
    source = ns["GENERATION_INPUT_UPLOAD"]
    original = source.read_bytes()

    def interrupted(source_handle, destination_handle):
        destination_handle.write(b"partial")
        raise OSError("copy interrupted")

    with monkeypatch.context() as patch:
        patch.setattr("shutil.copyfileobj", interrupted)
        with pytest.raises(OSError, match="copy interrupted"):
            exec(notebook_cell("generation-input"), ns)
    assert source.read_bytes() == original
    assert not list(ns["DATA_ROOT"].rglob("*.copying"))
    assert not (ns["DATA_ROOT"] / "data/processed/exemplars.csv").exists()
    exec(notebook_cell("generation-input"), ns)
    assert (ns["DATA_ROOT"] / "data/processed/exemplars.csv").read_bytes() == original


def test_saved_input_modes_skip_exemplar_staging():
    exec(notebook_cell("generation-input"), {"RUN_GENERATION": False})


@pytest.mark.parametrize("colab,override", [(False, None), (True, None), (True, "/custom/data")])
def test_notebook_defaults_to_drive_inputs_only_in_colab(tmp_path, monkeypatch, colab, override):
    import os

    if override is None:
        monkeypatch.delenv("ICL_PILOT_DATA_ROOT", raising=False)
    else:
        monkeypatch.setenv("ICL_PILOT_DATA_ROOT", override)
    ns = {"Path": Path, "os": os, "REPO_DIR": tmp_path, "IS_COLAB": colab}
    exec(notebook_cell("settings"), ns)
    expected = Path(override) if override else Path("/content/drive/MyDrive/icl_pilot_inputs") if colab else tmp_path
    assert ns["DATA_ROOT"] == expected
    assert ns["SMOKE_TEST"] is True
    assert ns["SMOKE_TEST_JOBS"] == 12


def test_drive_inputs_mount_even_when_output_persistence_is_disabled(tmp_path, monkeypatch):
    import os

    monkeypatch.delenv("ICL_PILOT_DATA_ROOT", raising=False)
    mounted = []
    fake_drive = SimpleNamespace(mount=lambda path: mounted.append(path))
    monkeypatch.setitem(sys.modules, "google.colab", SimpleNamespace(drive=fake_drive))
    original_is_dir = Path.is_dir

    def is_dir(path):
        if str(path) == "/content/drive/MyDrive":
            return bool(mounted)
        return original_is_dir(path)

    monkeypatch.setattr(Path, "is_dir", is_dir)
    ns = {"Path": Path, "os": os, "REPO_DIR": tmp_path, "IS_COLAB": True}
    exec(notebook_cell("settings"), ns)
    ns["SAVE_TO_DRIVE"] = False
    exec(notebook_cell("common-setup"), ns)
    assert mounted == ["/content/drive"]
    assert (ns["RUN_OUTPUT_DIR"] / "pipeline_manifest.json").is_file()


def test_persistent_copy_preserves_artifacts_from_another_run(tmp_path):
    source = begin_pipeline(tmp_path / "one", mode="generation_only", csvs={}, chat_dirs={})
    target = tmp_path / "persistent"
    sync_run(source, target)
    original = (target / "pipeline_manifest.json").read_bytes()
    other = begin_pipeline(tmp_path / "two", mode="generation_only", csvs={}, chat_dirs={})
    with pytest.raises(FileExistsError, match="different pipeline run"):
        sync_run(other, target)
    assert (target / "pipeline_manifest.json").read_bytes() == original
    # A second copy from the same run updates progress without losing ownership.
    sync_run(source, target)


def test_selecting_aggregate_and_cohort_rows_is_rejected_before_conversion(tmp_path):
    text = "bundle_id,target_story\npair_1_rep01,A1\n"
    (tmp_path / "cohort.csv").write_text(text)
    (tmp_path / "aggregate.csv").write_text(text)
    with pytest.raises(ValueError, match="Overlapping generation rows"):
        convert_csv_inputs({"cohort": tmp_path / "cohort.csv", "all": tmp_path / "aggregate.csv"},
                           tmp_path / "out", roster_csv=None)
    assert not (tmp_path / "out").exists()
