from pathlib import Path
import hashlib
import json
import sys

import pandas as pd
import pytest

from icl_pilot.evaluation import story_text_exports as exports


@pytest.mark.parametrize("sources", [None, []])
def test_export_requires_explicit_generation_sources_before_writing(tmp_path, sources):
    with pytest.raises(ValueError, match="Provide generation_sources"):
        exports.build_all_story_text_records(project_root=tmp_path, generation_sources=sources)
    assert list(tmp_path.iterdir()) == []


def test_export_uses_selected_run_and_preserves_source_text_and_provenance(tmp_path):
    run = tmp_path / "selected_run"
    run.mkdir()
    generated = run / "generations.csv"
    pd.DataFrame([{
        "round_id": "round_01", "bundle_id": "pair_sli401_td410_rep01",
        "pair_id": "pair_sli401_td410", "replicate_id": 1, "target_story": "A1",
        "stage1_td_story": "The dog plays.", "stage2_sli_story": "Dog play.",
        "td_child_id": 410, "sli_child_id": 401,
    }]).to_csv(generated, index=False)
    measures = tmp_path / "real_measures.csv"
    pd.DataFrame([{"File_ID": 420, "Group": "TD", "Age(Month)": 50}]).to_csv(measures, index=False)
    stories = tmp_path / "real_stories.csv"
    pd.DataFrame([{
        "File_ID": 420, "Group": "TD", "Age_Raw": "4;02.00",
        "transcript_path": "TD/420.cha", "story_id": "A1", "story_text": "A dog runs.",
    }]).to_csv(stories, index=False)
    original = {p: p.read_bytes() for p in (generated, measures, stories)}
    artifacts = exports.build_all_story_text_records(
        project_root=tmp_path, generation_sources=[(4, "selected_run/generations.csv")],
        real_measures_csv=measures.name, real_stories_csv=stories.name,
        output_csv="exports/records.csv",
    )
    records = pd.read_csv(artifacts["output_csv"])
    synthetic = records.loc[records["source"] == "synthetic"]
    assert synthetic["story_text"].tolist() == ["The dog plays.", "Dog play."]
    assert synthetic["stage"].tolist() == ["stage1_td", "stage2_sli"]
    assert synthetic["source_child_id"].tolist() == [410, 401]
    assert synthetic["source_csv"].eq("selected_run/generations.csv").all()
    assert synthetic["source_row_number"].eq(2).all()
    summary = json.loads(artifacts["summary_json"].read_text())
    assert summary["total_records"] == 3
    assert summary["generated_sources"][0]["sha256"] == hashlib.sha256(original[generated]).hexdigest()
    assert all(p.read_bytes() == content for p, content in original.items())


def test_cli_forwards_multiple_selected_cohorts(monkeypatch, tmp_path):
    calls = []

    def build(**kwargs):
        calls.append(kwargs)
        return {"output_csv": Path("fixture.csv"), "summary_json": Path("fixture.json")}

    monkeypatch.setattr(exports, "build_all_story_text_records", build)
    monkeypatch.setattr(sys, "argv", ["export", "--project-root", str(tmp_path),
        "--generation-source", "4=selected run/four.csv", "--generation-source", "7=seven.csv",
        "--output-csv", "out.csv", "--real-measures-csv", "real.csv", "--real-stories-csv", "stories.csv"])
    exports.main()
    assert calls[0]["generation_sources"] == [(4, "selected run/four.csv"), (7, "seven.csv")]
    assert calls[0]["project_root"] == str(tmp_path)


@pytest.mark.parametrize("argument", [None, "4", "4=", "x=run.csv", "8=run.csv"])
def test_cli_rejects_missing_or_invalid_generation_sources(monkeypatch, capsys, argument):
    monkeypatch.setattr(sys, "argv", ["export"] + ([] if argument is None else ["--generation-source", argument]))
    with pytest.raises(SystemExit) as exc:
        exports.main()
    assert exc.value.code == 2
    assert "AGE" in capsys.readouterr().err


def test_gillam_only_cli_keeps_its_separate_input_contract(monkeypatch):
    calls = []
    monkeypatch.setattr(exports, "build_gillam_external_text_records",
        lambda **kwargs: calls.append(kwargs) or {"output_csv": Path("fixture.csv"), "summary_json": Path("fixture.json")})
    monkeypatch.setattr(sys, "argv", ["export", "--gillam-external-only", "--gillam-transcripts-dir", "external",
        "--gillam-output-csv", "external_records.csv"])
    exports.main()
    assert calls[0]["transcripts_dir"] == "external"


def test_export_requires_explicit_reference_and_destination_paths(tmp_path):
    with pytest.raises(ValueError, match="Provide explicit paths"):
        exports.build_all_story_text_records(project_root=tmp_path, generation_sources=[(4, "selected.csv")])
    with pytest.raises(ValueError, match="Provide explicit paths"):
        exports.build_gillam_external_text_records(project_root=tmp_path)
    assert list(tmp_path.iterdir()) == []
