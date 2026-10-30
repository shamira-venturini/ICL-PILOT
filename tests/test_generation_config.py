from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from icl_pilot.generation.frozen_roster import build_frozen_roster_manifest
from icl_pilot.chat.generated_chat import convert_generated_story_csv_to_cha
from icl_pilot.generation.config import (
    compile_experiment_design, load_experiment_config, prepare_colab_generation,
    validate_prompt_evaluation_separation,
)
from icl_pilot.generation.workflow import build_rotating_generation_schedule


ROOT = Path(__file__).resolve().parents[1]


def test_default_schedule_balances_exemplar_roles_and_complexity() -> None:
    config, schedule, _evaluation, rosters = compile_experiment_design(
        ROOT / "configs/generation/experiment.json"
    )
    assert config["n_replicates"] == 10
    assert sum(len(roster) for roster in rosters.values()) == 48
    assert len(schedule) == 2880
    allocation = schedule.groupby(
        ["pair_id", "target_story", "counterbalance_variant_index"]
    ).size().unstack("counterbalance_variant_index")
    assert set(allocation.columns) == {1, 2}
    assert len(allocation) == 48 * 6
    assert allocation.eq(5).all().all()

    e1_column, e2_column = "prompt_td_story_e1", "prompt_sli_story_e2"
    for column in (e1_column, e2_column):
        assert schedule[column].str[0].ne(schedule["target_story"].str[0]).all()
    assert schedule[e1_column].ne(schedule[e2_column]).all()
    for _, rows in schedule.groupby(["pair_id", "target_story"]):
        e1_counts = rows[e1_column].value_counts().sort_index()
        e2_counts = rows[e2_column].value_counts().sort_index()
        assert len(e1_counts) == 2
        assert e1_counts.to_dict() == e2_counts.to_dict()
        assert e1_counts.eq(5).all()
    for _, rows in schedule.groupby("pair_id"):
        for column in (e1_column, e2_column):
            assert rows[column].value_counts().to_dict() == {
                story: 10 for story in ("A1", "A2", "A3", "B1", "B2", "B3")
            }
            assert rows[column].str[-1].value_counts().to_dict() == {
                "1": 20, "2": 20, "3": 20
            }


def test_packaged_exemplars_prepare_all_cohorts_and_match_their_qc_hash(tmp_path: Path) -> None:
    config_path = ROOT / "configs/generation/experiment.json"
    config = json.loads(config_path.read_text())
    csv_path = ROOT / config["inputs"]["narratives_csv"]
    qc = json.loads(csv_path.with_suffix(".parsing_qc.json").read_text())
    original_csv = csv_path.read_bytes()
    prepared = prepare_colab_generation(config_path, ROOT, tmp_path / "run", n_replicates=1)
    assert len(prepared.run_df) == 288
    assert set(prepared.run_df["cohort"]) == {"4-year-old", "5-year-old", "6-year-old", "7-year-old"}
    assert prepared.run_df.groupby("pair_id")["target_story"].nunique().eq(6).all()
    manifest = json.loads(prepared.run_manifest_path.read_text())
    assert manifest["split_validation"] == "passed"
    assert manifest["config"]["model"]["top_k"] == 0
    assert manifest["input_sha256"]["narratives"] == qc["output_csv_sha256"]
    assert csv_path.read_bytes() == original_csv
    preparation = manifest["exemplar_preparation"]
    expected = {(str(row.td_child_id), "TD", row.prompt_td_story_e1)
                for row in prepared.run_df.itertuples(index=False)}
    expected |= {(str(row.sli_child_id), "SLI", row.prompt_sli_story_e2)
                 for row in prepared.run_df.itertuples(index=False)}
    assert preparation["unique_exemplars"] == len(expected)
    snapshot = tmp_path / "run" / preparation["snapshot_csv"]
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == preparation["snapshot_sha256"]
    inventory = pd.read_csv(snapshot, dtype={"File_ID": str}, keep_default_na=False)
    assert set(zip(inventory["File_ID"], inventory["Group"], inventory["story_id"])) == expected
    assert len(inventory) == len(expected) and inventory["story_text"].str.len().gt(0).all()
    for column in ("prompt_td_exemplar_story", "prompt_sli_exemplar_story"):
        assert not prepared.run_df[column].str.contains(
            r"\((?:\.{1,3}|\d+(?:\.\d+)?)\)|\b(?:xxx|yyy|www)\b|&=", case=False, regex=True
        ).any()


@pytest.fixture
def experiment(tmp_path: Path) -> Path:
    source = ROOT / "configs/generation"
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "models.json").write_bytes((source / "models.json").read_bytes())
    (config_dir / "prompts").mkdir()
    for stage in ("stage1", "stage2"):
        (config_dir / "prompts" / (stage + ".json")).write_bytes((source / "prompts" / (stage + ".json")).read_bytes())
    config = json.loads((source / "experiment.json").read_text())
    config["cohorts"] = config["cohorts"][:2]
    for cohort in config["cohorts"]:
        cohort["expected_pairs"] = 2
    config["n_replicates"] = 2
    config["pairs_per_round"] = 1
    config["inputs"] = {"narratives_csv": "narratives.csv"}
    config["story_reference_json"] = "reference.json"
    master = pd.read_csv(source / config["roster_csv"])
    roster = pd.concat([master.loc[master["cohort"] == cohort["label"]].iloc[:2]
                        for cohort in config["cohorts"]], ignore_index=True)
    roster.to_csv(config_dir / config["roster_csv"], index=False)
    (config_dir / "cross_story_counterbalancing.csv").write_bytes((source / "cross_story_counterbalancing.csv").read_bytes())
    stories = ["A1", "A2", "A3", "B1", "B2", "B3"]
    rows = []
    for group, column in (("TD", "td_child_id"), ("SLI", "sli_child_id")):
        for pair in roster.itertuples(index=False):
            child = getattr(pair, column)
            age = pair.td_age if group == "TD" else pair.sli_age
            for story in stories:
                rows.append({"File_ID": str(child), "Group": group, "Age_Raw": age, "story_id": story,
                             "story_text": f"Fixture exemplar {child} {story}.",
                             "story_units_json": json.dumps([f"Fixture exemplar {child} {story}."]),
                             "story_unit_count": 1})
    pd.DataFrame(rows).to_csv(tmp_path / "narratives.csv", index=False)
    (config_dir / "reference.json").write_text(json.dumps({story: {"title": story, "units": ["Fixture event."]} for story in stories}))
    path = config_dir / "experiment.json"
    path.write_text(json.dumps(config))
    return path


@pytest.mark.parametrize("value", [None, -1, True, 0.0, "0"])
def test_invalid_top_k_fails_preflight(experiment: Path, value) -> None:
    config = json.loads(experiment.read_text())
    config["model"]["top_k"] = value
    experiment.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="top_k must be an explicit nonnegative integer"):
        load_experiment_config(experiment)


def test_missing_top_k_cannot_silently_inherit_a_default(experiment: Path) -> None:
    config = json.loads(experiment.read_text())
    del config["model"]["top_k"]
    experiment.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="top_k must be an explicit nonnegative integer"):
        load_experiment_config(experiment)


def test_setup_creates_complete_jobs_without_changing_rosters(experiment: Path, tmp_path: Path) -> None:
    source = experiment.parent / json.loads(experiment.read_text())["roster_csv"]
    original = source.read_bytes()
    prepared = prepare_colab_generation(experiment, tmp_path, tmp_path / "run")
    assert len(prepared.run_df) == 48
    assert prepared.run_df.groupby("bundle_id")["target_story"].nunique().eq(6).all()
    assert prepared.run_df["prompt_td_exemplar_story"].str.len().gt(0).all()
    assert source.read_bytes() == original
    assert json.loads(prepared.run_manifest_path.read_text())["split_validation"] == "passed"
    for label, package in prepared.artifacts.cohort_packages.items():
        snapshot = pd.read_csv(package.roster_csv)
        expected = pd.read_csv(source).loc[lambda df: df["cohort"] == label].reset_index(drop=True)
        pd.testing.assert_frame_equal(snapshot, expected)
    manifest = json.loads(prepared.run_manifest_path.read_text())
    assert manifest["input_sha256"]["roster"] == hashlib.sha256(original).hexdigest()
    assert manifest["input_sha256"]["story_reference"] == hashlib.sha256((experiment.parent / "reference.json").read_bytes()).hexdigest()


def test_both_stages_receive_cleaned_exemplars_with_source_provenance(experiment: Path, tmp_path: Path) -> None:
    from icl_pilot.chat.story_unit_semrp import prepare_extracted_story_csv
    source = tmp_path / "narratives.csv"
    data = pd.read_csv(source)
    data["story_text"] = "her xxx (.) runned &=laughs ."
    data["story_units_json"] = json.dumps(["her xxx (.) runned &=laughs ."])
    data.to_csv(source, index=False)
    original = source.read_bytes()
    output = prepare_extracted_story_csv(source, tmp_path / "ready.csv")
    config = json.loads(experiment.read_text())
    config["inputs"]["narratives_csv"] = output.name
    experiment.write_text(json.dumps(config))
    prepared = prepare_colab_generation(experiment, tmp_path, tmp_path / "run")
    row = next(prepared.run_df.itertuples(index=False))
    for messages in (prepared.prompt_builder.stage1_messages(row),
                     prepared.prompt_builder.stage2_messages(row, "unchanged parent .")):
        assert "her runned ." in messages[1]["content"]
        assert "xxx" not in messages[1]["content"] and "&=" not in messages[1]["content"]
    manifest = json.loads(prepared.run_manifest_path.read_text())
    audit = manifest["exemplar_preparation"]
    assert audit["generation_action"] == "validate_and_select_only"
    assert audit["performed_at"] == "chat_to_csv_extraction"
    assert audit["extraction_qc"]["text_preparation"]["changed_stories"] == len(data)
    assert manifest["input_sha256"]["narratives"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert source.read_bytes() == original


def test_unusable_exemplars_fail_before_writing_a_run(experiment: Path, tmp_path: Path) -> None:
    source = tmp_path / "narratives.csv"
    data = pd.read_csv(source)
    data.loc[0, "story_text"] = "xxx (.) &=laughs ."
    data.loc[0, "story_units_json"] = json.dumps(["xxx (.) &=laughs ."])
    data.to_csv(source, index=False)
    with pytest.raises(ValueError, match="CHAT-to-CSV extraction"):
        prepare_colab_generation(experiment, tmp_path, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_unknown_exemplar_preparation_policy_fails_before_writing_a_run(experiment: Path, tmp_path: Path) -> None:
    config = json.loads(experiment.read_text())
    config["exemplar_preparation"]["policy"] = "unknown_policy"
    experiment.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="exemplar_preparation policy"):
        prepare_colab_generation(experiment, tmp_path, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_generation_only_validates_prepared_text_and_does_not_call_cleaner(experiment, tmp_path, monkeypatch):
    from icl_pilot.chat import text_cleaning
    def forbidden(*args, **kwargs):
        raise AssertionError("cleanup must already have happened during extraction")
    monkeypatch.setattr(text_cleaning, "prepare_extracted_story", forbidden)
    prepared = prepare_colab_generation(experiment, tmp_path, tmp_path / "run")
    assert len(prepared.run_df) == 48


@pytest.mark.parametrize("wrong", ["hash", "policy", "stage"])
def test_generation_checks_available_extraction_audit(experiment, tmp_path, wrong):
    source = tmp_path / "narratives.csv"
    qc = {"output_csv_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
          "text_preparation": {"policy": "chat-story-text-v3", "performed_at": "chat_to_csv_extraction"}}
    if wrong == "hash": qc["output_csv_sha256"] = "wrong"
    if wrong == "policy": qc["text_preparation"]["policy"] = "wrong"
    if wrong == "stage": qc["text_preparation"]["performed_at"] = "generation"
    source.with_suffix(".parsing_qc.json").write_text(json.dumps(qc))
    with pytest.raises(ValueError, match="extraction QC/policy"):
        prepare_colab_generation(experiment, tmp_path, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_packaged_text_matches_its_definitive_extraction_hashes():
    import csv
    source = ROOT / "configs/generation/inputs/story_unit_narratives_enni_2026-10-28.csv"
    audited = ROOT / "docs/reproducibility/tokenizer_audit/per_story.csv"
    with source.open() as handle: rows = list(csv.DictReader(handle))
    with audited.open() as handle: references = list(csv.DictReader(handle))
    assert len(rows) == len(references) == 2160
    for row, reference in zip(rows, references):
        assert hashlib.sha256(row["story_text"].encode()).hexdigest() == reference["prepared_text_sha256"]
        assert " ".join(json.loads(row["story_units_json"])) == row["story_text"]


def test_master_cannot_mix_cohorts_into_one_round(experiment: Path) -> None:
    config = json.loads(experiment.read_text())
    with pytest.raises(ValueError, match="Select one nonempty cohort"):
        build_rotating_generation_schedule(experiment.parent / config["roster_csv"],
                                           experiment.parent / config["counterbalance_csv"])


def test_unknown_master_cohort_is_rejected(experiment: Path) -> None:
    config = json.loads(experiment.read_text())
    path = experiment.parent / config["roster_csv"]
    roster = pd.read_csv(path)
    roster.loc[0, "cohort"] = "unknown"
    roster.to_csv(path, index=False)
    with pytest.raises(ValueError, match="cohort labels must match"):
        compile_experiment_design(experiment)


def test_master_rejects_child_reused_across_cohorts(experiment: Path) -> None:
    config = json.loads(experiment.read_text())
    path = experiment.parent / config["roster_csv"]
    roster = pd.read_csv(path)
    roster.loc[2, "td_child_id"] = roster.loc[0, "td_child_id"]
    roster.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique within and across cohorts"):
        compile_experiment_design(experiment)


def test_changed_master_cannot_reuse_run_directory(experiment: Path, tmp_path: Path) -> None:
    prepared = prepare_colab_generation(experiment, tmp_path, tmp_path / "run")
    manifest = prepared.run_manifest_path.read_bytes()
    path = experiment.parent / json.loads(experiment.read_text())["roster_csv"]
    roster = pd.read_csv(path)
    roster.loc[0, "notes"] = "changed input provenance"
    roster.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Run inputs changed"):
        prepare_colab_generation(experiment, tmp_path, tmp_path / "run")
    assert prepared.run_manifest_path.read_bytes() == manifest


def test_split_check_rejects_another_pairs_prompt_child(experiment: Path) -> None:
    _config, schedule, evaluation, _rosters = compile_experiment_design(experiment)
    row = evaluation.iloc[0].copy()
    prompts = schedule.loc[(schedule["cohort"] == row["cohort"]) & (schedule["round_id"] == row["round_id"])]
    row["eval_child_id"] = int(prompts.iloc[0]["td_child_id"])
    with pytest.raises(ValueError, match="Prompt/evaluation overlap"):
        validate_prompt_evaluation_separation(schedule, pd.concat([evaluation, row.to_frame().T], ignore_index=True))


def test_missing_exemplar_fails_before_creating_output(experiment: Path, tmp_path: Path) -> None:
    data = pd.read_csv(tmp_path / "narratives.csv")
    data.loc[data["Group"] == "TD", "story_text"] = ""
    data.to_csv(tmp_path / "narratives.csv", index=False)
    with pytest.raises(ValueError, match="Missing or empty prompt exemplar"):
        prepare_colab_generation(experiment, tmp_path, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_duplicate_exemplar_keys_fail(experiment: Path, tmp_path: Path) -> None:
    data = pd.read_csv(tmp_path / "narratives.csv")
    pd.concat([data, data.iloc[:1]], ignore_index=True).to_csv(tmp_path / "narratives.csv", index=False)
    with pytest.raises(ValueError, match="Duplicate child/group/story"):
        prepare_colab_generation(experiment, tmp_path, tmp_path / "run")


def test_changed_inputs_cannot_reuse_run_directory(experiment: Path, tmp_path: Path) -> None:
    prepared = prepare_colab_generation(experiment, tmp_path, tmp_path / "run")
    manifest = prepared.run_manifest_path.read_bytes()
    config = json.loads(experiment.read_text())
    config["master_seed"] += 1
    experiment.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="master_seed=42"):
        prepare_colab_generation(experiment, tmp_path, tmp_path / "run")
    assert prepared.run_manifest_path.read_bytes() == manifest


@pytest.mark.parametrize("setting,value", [
    ("max_new_tokens_stage1", 641), ("max_new_tokens_stage2", 641),
    ("temperature", 0.85), ("top_p", 0.92), ("top_k", 50),
])
def test_changed_decoding_setting_requires_fresh_run(experiment: Path, tmp_path: Path, setting: str, value) -> None:
    prepared = prepare_colab_generation(experiment, tmp_path, tmp_path / "run")
    original_manifest = prepared.run_manifest_path.read_bytes()
    config = json.loads(experiment.read_text())
    config["model"][setting] = value
    experiment.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Run inputs changed"):
        prepare_colab_generation(experiment, tmp_path, tmp_path / "run")
    assert prepared.run_manifest_path.read_bytes() == original_manifest


def test_changed_prompt_template_cannot_reuse_run_directory(experiment: Path, tmp_path: Path) -> None:
    prepared = prepare_colab_generation(experiment, tmp_path, tmp_path / "run")
    manifest = prepared.run_manifest_path.read_bytes()
    template_path = experiment.parent / "prompts/stage1.json"
    original_template = template_path.read_text()
    changed_template = original_template.replace("You produce", "You create")
    assert changed_template != original_template
    template_path.write_text(changed_template)
    with pytest.raises(ValueError, match="Run inputs changed"):
        prepare_colab_generation(experiment, tmp_path, tmp_path / "run")
    assert prepared.run_manifest_path.read_bytes() == manifest


def test_changed_counterbalance_cannot_reuse_run_directory(experiment: Path, tmp_path: Path) -> None:
    prepared = prepare_colab_generation(experiment, tmp_path, tmp_path / "run")
    manifest = prepared.run_manifest_path.read_bytes()
    path = experiment.parent / "cross_story_counterbalancing.csv"
    table = pd.read_csv(path)
    table.loc[0, "e2_story"] = "B3"
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Run inputs changed"):
        prepare_colab_generation(experiment, tmp_path, tmp_path / "run")
    assert prepared.run_manifest_path.read_bytes() == manifest


def test_existing_generation_output_is_preserved(experiment: Path, tmp_path: Path) -> None:
    prepare_colab_generation(experiment, tmp_path, tmp_path / "run")
    saved = tmp_path / "run/all_cohorts_rotating_story_generations.csv"
    saved.write_text("existing research output\n")
    with pytest.raises(FileExistsError, match="Existing generation outputs"):
        prepare_colab_generation(experiment, tmp_path, tmp_path / "run")
    assert saved.read_text() == "existing research output\n"


def test_invalid_counterbalancing_is_rejected(experiment: Path, tmp_path: Path) -> None:
    table = pd.read_csv(experiment.parent / "cross_story_counterbalancing.csv")
    table.loc[0, "e1_story"] = table.loc[0, "target_story"]
    table.to_csv(experiment.parent / "cross_story_counterbalancing.csv", index=False)
    with pytest.raises(ValueError, match="opposite story set"):
        compile_experiment_design(experiment)


def test_frozen_roster_builder_refuses_existing_output(tmp_path: Path) -> None:
    roster = tmp_path / "roster.csv"
    roster.write_text("frozen data\n")
    with pytest.raises(FileExistsError, match="Frozen roster already exists"):
        build_frozen_roster_manifest("unused_input.csv", str(roster))
    assert roster.read_text() == "frozen data\n"


def test_exemplar_age_mismatch_is_rejected(experiment: Path, tmp_path: Path) -> None:
    data = pd.read_csv(tmp_path / "narratives.csv")
    data.loc[data["Group"] == "TD", "Age_Raw"] = "9;01.01"
    data.to_csv(tmp_path / "narratives.csv", index=False)
    with pytest.raises(ValueError, match="Exemplar age disagrees"):
        prepare_colab_generation(experiment, tmp_path, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_reconciled_pairs_export_known_TD_ages(tmp_path: Path) -> None:
    roster = pd.read_csv(ROOT / "configs/generation/master_frozen_roster_enni_2026-10-28.csv")
    selected = roster.loc[roster["sli_child_id"].isin([474, 475])]
    rows = [{"bundle_id": row.pair_id + "_rep01", "pair_id": row.pair_id,
             "target_story": "A1", "sli_age": row.sli_age,
             "td_child_id": row.td_child_id, "sli_child_id": row.sli_child_id,
             "stage1_td_story": "Fixture story.", "stage2_sli_story": "Fixture story."}
            for row in selected.itertuples(index=False)]
    source = tmp_path / "generation.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    from segmentation_stub import StubSegmenter
    output = convert_generated_story_csv_to_cha(input_csv=source, output_dir=tmp_path / "chat", segmenter=StubSegmenter(),
        roster_csv=ROOT / "configs/generation/master_frozen_roster_enni_2026-10-28.csv")
    manifest = pd.read_csv(output["manifest_csv"], keep_default_na=False)
    assert set(manifest.loc[manifest["stage"] == "stage1_td", "age"]) == {"4;08.10", "4;08.15"}


def test_replicate_override_is_recorded_without_editing_study_config(experiment, tmp_path):
    original = experiment.read_bytes()
    prepared = prepare_colab_generation(experiment, tmp_path, tmp_path / "run", n_replicates=1)
    assert len(prepared.run_df) == 24
    assert prepared.config["n_replicates"] == 1
    assert experiment.read_bytes() == original
    manifest = json.loads(prepared.run_manifest_path.read_text())
    assert manifest["config"]["n_replicates"] == 1
    assert "effective_config" in manifest["input_sha256"]
    with pytest.raises(ValueError, match="Run inputs changed"):
        prepare_colab_generation(experiment, tmp_path, tmp_path / "run", n_replicates=2)


@pytest.mark.parametrize("replicates", [0, -1, 1.5, True])
def test_invalid_replicate_override_is_rejected(experiment, replicates):
    with pytest.raises(ValueError, match="positive integer"):
        compile_experiment_design(experiment, n_replicates=replicates)


@pytest.mark.parametrize("count", [1, 6, 12])
def test_smoke_test_selects_canonical_prefix_and_labels_outputs(experiment, tmp_path, count):
    from icl_pilot.generation.execution import run_generation
    original = experiment.read_bytes()
    prepared = prepare_colab_generation(experiment, tmp_path, tmp_path / "smoke", smoke_test=True,
                                        smoke_test_jobs=count)
    complete = prepare_colab_generation(experiment, tmp_path, tmp_path / "study")
    pd.testing.assert_frame_equal(prepared.run_df, complete.run_df.iloc[:count])
    for row in prepared.run_df.itertuples(index=False):
        assert prepared.prompt_builder.stage1_messages(row) == complete.prompt_builder.stage1_messages(row)
        assert prepared.prompt_builder.stage2_messages(row, "Same parent.") == complete.prompt_builder.stage2_messages(row, "Same parent.")
    assert experiment.read_bytes() == original
    manifest = json.loads(prepared.run_manifest_path.read_text())
    assert manifest["generation_jobs"] == count
    assert manifest["execution"]["planned_jobs"] == 48
    assert manifest["execution"]["test_only"] is True
    assert manifest["execution"]["mode"] == "smoke_test"
    selection = pd.read_csv(tmp_path / "smoke" / manifest["execution"]["selection_csv"])
    assert len(selection) == count
    calls = []
    def generate(messages, *, row, stage, journal, max_new_tokens, parent_record_id=None, stage1_text=None):
        calls.append(stage)
        record_id = journal.record_prompt(row=row, stage=stage, messages=messages,
            rendered_prompt=json.dumps(messages), input_token_ids=[1], generation_parameters={},
            parent_record_id=parent_record_id, stage1_text=stage1_text)
        journal.record_output(record_id, "Fixture narrative .", generated_token_count=3)
        return "Fixture narrative .", record_id
    paths = run_generation(prepared, generate, run_id="smoke_test")
    assert len(paths) == 1
    assert calls == ["stage1_td", "stage2_sli"] * count
    records = pd.read_csv(paths[0])
    assert len(records) == count and records["test_only"].eq(True).all()
    final = json.loads(prepared.run_manifest_path.read_text())
    assert final["completed_generation_jobs"] == count and final["generation_jobs"] == count


@pytest.mark.parametrize("count", [0, -1, 1.5, True, None])
def test_invalid_smoke_test_job_count_is_rejected(experiment, tmp_path, count):
    with pytest.raises(ValueError, match="positive integer"):
        prepare_colab_generation(experiment, tmp_path, tmp_path / "run", smoke_test=True, smoke_test_jobs=count)
    assert not (tmp_path / "run").exists()


def test_smoke_test_count_does_not_silently_truncate_or_change_study_mode(experiment, tmp_path):
    with pytest.raises(ValueError, match="exceeds"):
        prepare_colab_generation(experiment, tmp_path, tmp_path / "run", smoke_test=True, smoke_test_jobs=49)
    with pytest.raises(ValueError, match="smoke_test=True"):
        prepare_colab_generation(experiment, tmp_path, tmp_path / "run", smoke_test_jobs=12)
    assert not (tmp_path / "run").exists()


def test_smoke_run_size_change_requires_fresh_directory(experiment, tmp_path):
    prepared = prepare_colab_generation(experiment, tmp_path, tmp_path / "run", smoke_test=True)
    original = prepared.run_manifest_path.read_bytes()
    with pytest.raises(ValueError, match="Run inputs changed"):
        prepare_colab_generation(experiment, tmp_path, tmp_path / "run", smoke_test=True, smoke_test_jobs=12)
    assert prepared.run_manifest_path.read_bytes() == original


def test_smoke_run_directory_cannot_be_reused_for_full_study(experiment, tmp_path):
    prepared = prepare_colab_generation(experiment, tmp_path, tmp_path / "smoke", smoke_test=True)
    original = prepared.run_manifest_path.read_bytes()
    with pytest.raises(ValueError, match="Run inputs changed"):
        prepare_colab_generation(experiment, tmp_path, tmp_path / "smoke", smoke_test=False)
    assert prepared.run_manifest_path.read_bytes() == original


@pytest.mark.parametrize("value", [1, "True", None])
def test_smoke_test_requires_a_boolean(experiment, tmp_path, value):
    with pytest.raises(ValueError, match="smoke_test must be a boolean"):
        prepare_colab_generation(experiment, tmp_path, tmp_path / "smoke", smoke_test=value)
    assert not (tmp_path / "smoke").exists()
