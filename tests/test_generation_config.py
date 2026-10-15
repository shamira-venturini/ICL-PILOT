from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from icl_pilot.frozen_roster import build_frozen_roster_manifest
from icl_pilot.generated_chat import convert_generated_story_csv_to_cha
from icl_pilot.generation.config import (
    compile_experiment_design, prepare_colab_generation,
    validate_prompt_evaluation_separation,
)
from icl_pilot.generation.workflow import build_rotating_generation_schedule


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def experiment(tmp_path: Path) -> Path:
    source = ROOT / "configs/generation"
    config_dir = tmp_path / "config"
    config_dir.mkdir()
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
                             "story_text": f"Fixture exemplar {child} {story}."})
    pd.DataFrame(rows).to_csv(tmp_path / "narratives.csv", index=False)
    (config_dir / "reference.json").write_text(json.dumps({story: {"title": story, "units": ["Fixture event."]} for story in stories}))
    path = config_dir / "experiment.json"
    path.write_text(json.dumps(config))
    return path


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
    config["seed"] += 1
    experiment.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Run inputs changed"):
        prepare_colab_generation(experiment, tmp_path, tmp_path / "run")
    assert prepared.run_manifest_path.read_bytes() == manifest


def test_changed_prompt_template_cannot_reuse_run_directory(experiment: Path, tmp_path: Path) -> None:
    prepared = prepare_colab_generation(experiment, tmp_path, tmp_path / "run")
    manifest = prepared.run_manifest_path.read_bytes()
    template_path = experiment.parent / "prompts/stage1.json"
    template_path.write_text(template_path.read_text().replace("You write", "You create"))
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
    roster = pd.read_csv(ROOT / "configs/generation/master_frozen_roster_4_to_7_year_old.csv")
    selected = roster.loc[roster["sli_child_id"].isin([474, 475])]
    rows = [{"bundle_id": row.pair_id + "_rep01", "pair_id": row.pair_id,
             "target_story": "A1", "sli_age": row.sli_age,
             "td_child_id": row.td_child_id, "sli_child_id": row.sli_child_id,
             "stage1_td_story": "Fixture story.", "stage2_sli_story": "Fixture story."}
            for row in selected.itertuples(index=False)]
    source = tmp_path / "generation.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    output = convert_generated_story_csv_to_cha(input_csv=source, output_dir=tmp_path / "chat")
    manifest = pd.read_csv(output["manifest_csv"], keep_default_na=False)
    assert set(manifest.loc[manifest["stage"] == "stage1_td", "age"]) == {"4;07.26", "4;08.04"}


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
