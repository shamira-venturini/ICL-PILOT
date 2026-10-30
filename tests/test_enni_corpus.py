from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import pandas as pd
import pytest

from icl_pilot.chat.enni_metadata import completed_age_months, read_chi_metadata
from icl_pilot.evaluation.story_text_exports import _real_records
from icl_pilot.generation.config import compile_experiment_design, prepare_colab_generation
from icl_pilot.maintenance.enni_corpus import intake, INPUT_REL, STORIES_NAME, PARTICIPANTS_NAME


ROOT = Path(__file__).resolve().parents[1]


def transcript(age="5;05.29", group="SLI", pid="fixture/1"):
    return ("@PID:\t" + pid + "\n@ID:\teng|ENNI|CHI|" + age + "|male|" + group + "||Target_Child|||\n" +
            "".join("@G:\t" + story + "\n*CHI:\ther her goed (be)cause xxx .\n%mor:\tn|dog .\n"
                    for story in ("A1", "A2", "A3", "B1", "B2", "B3")))


@pytest.mark.parametrize("age,expected", [("4;11.18",59),("5;11.21",71),("7;11.19",95),("8;00",96)])
def test_days_never_round_a_child_into_another_cohort(age, expected):
    assert completed_age_months(age) == expected


@pytest.mark.parametrize("age", ["", "4;", "4;12.00", "4;11.99", "4.11"])
def test_unknown_or_invalid_age_is_not_imputed(age):
    with pytest.raises(ValueError):
        completed_age_months(age)
    assert read_chi_metadata(transcript(age=age))["Age_Month"] is None


def test_intake_preserves_raw_files_and_quarantines_missing_header_collision(tmp_path):
    archive = tmp_path / "ENNI.zip"
    payload = {"SLI/A/523.cha": transcript(), "TD/B/523.cha": transcript("", "", "fixture/2")}
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, text in payload.items():
            bundle.writestr(name, text)
    original = archive.read_bytes()
    root = tmp_path / "project"
    qc = intake(archive, root)
    assert qc["included_participants"] == 1 and qc["story_rows"] == 6
    assert qc["duplicate_filename_ids"] == {"523": ["SLI/A/523.cha", "TD/B/523.cha"]}
    assert qc["excluded_sources"][0]["transcript_path"] == "TD/B/523.cha"
    assert (root / "data/raw/enni/2026-10-28/ENNI.zip").read_bytes() == original
    for name, text in payload.items():
        assert (root / "data/raw/enni/2026-10-28/transcripts" / name).read_bytes() == text.encode()
    stories = pd.read_csv(root / INPUT_REL / STORIES_NAME)
    assert set(stories["Group"]) == {"SLI"}
    assert stories["story_text"].eq("her her goed (be)cause .").all()
    assert archive.read_bytes() == original
    with pytest.raises(FileExistsError):
        intake(archive, root)


def test_intake_rejects_two_fully_specified_conflicting_ids_before_writing(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("SLI/523.cha", transcript())
        bundle.writestr("TD/523.cha", transcript(group="TD",pid="fixture/2"))
    with pytest.raises(ValueError, match="share an ID"):
        intake(archive, tmp_path / "project")
    assert not (tmp_path / "project").exists()


def test_intake_rejects_unsafe_zip_before_extracting(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.cha", transcript())
    with pytest.raises(ValueError, match="Unsafe"):
        intake(archive, tmp_path / "project")
    assert not (tmp_path / "escape.cha").exists()


def test_verified_reference_has_one_child_per_id_and_header_ages():
    source = ROOT / INPUT_REL / STORIES_NAME
    participants = ROOT / INPUT_REL / PARTICIPANTS_NAME
    frame = pd.read_csv(source)
    meta = pd.read_csv(participants)
    assert len(meta) == 360 and meta["File_ID"].is_unique
    assert len(frame) == 2160 and not frame.duplicated(["File_ID","story_id"]).any()
    assert frame.groupby("File_ID")["story_id"].nunique().eq(6).all()
    assert set(meta.loc[meta["File_ID"].eq(523), "Group"]) == {"SLI"}
    records, missing = _real_records(measures_path=participants, stories_path=source, project_root=ROOT,
                                    minimum_age_months=48, maximum_age_months=96)
    real = pd.DataFrame(records)
    assert len(real) == 1428 and not missing
    assert real[["source_child_id","group"]].drop_duplicates()["group"].value_counts().to_dict() == {"TD":190,"SLI":48}
    assert real.loc[real["source_child_id"].isin([732,766]), "cohort_year"].eq(7).all()
    assert real.loc[real["source_child_id"].isin([414,443]), "cohort_year"].eq(4).all()
    assert real.loc[real["source_child_id"].isin([543,553]), "cohort_year"].eq(5).all()


def test_original_demographics_supports_523_and_resolves_sex_with_traceability():
    frame = pd.read_csv(ROOT/INPUT_REL/PARTICIPANTS_NAME,keep_default_na=False).set_index("File_ID")
    assert frame.loc[523,"demographic_group_raw"] == "SLI-A"
    assert frame.loc[523,"Sex"] == "male" and frame.loc[523,"demographic_no_audio"] == "x"
    assert (frame.loc[460,"Sex_CHI"],frame.loc[460,"Sex"]) == ("female","male")
    assert (frame.loc[601,"Sex_CHI"],frame.loc[601,"Sex"]) == ("male","female")
    assert (frame.loc[959,"Sex_CHI"],frame.loc[959,"Sex"]) == ("","female")


def test_export_refuses_directory_group_imputation_and_legacy_523_label(tmp_path):
    source = tmp_path / "stories.csv"
    measures = tmp_path / "meta.csv"
    row = dict(File_ID=523, Group="", Age_Raw="", transcript_path="TD/B/523.cha",story_id="A1",story_text="He go .")
    pd.DataFrame([row]).to_csv(source,index=False)
    pd.DataFrame([{"File_ID":523,"Group":"TD","Age(Month)":65}]).to_csv(measures,index=False)
    args = dict(measures_path=measures,stories_path=source,project_root=tmp_path,minimum_age_months=48,maximum_age_months=96)
    with pytest.raises(ValueError,match="never infer"):
        _real_records(**args)
    row.update(Group="SLI",Age_Raw="5;05.29",transcript_path="SLI/A/523.cha")
    pd.DataFrame([row]).to_csv(source,index=False)
    with pytest.raises(ValueError,match="no matching literal CHI header"):
        _real_records(**args)


def test_export_uses_header_age_even_if_measurement_age_was_rounded(tmp_path):
    source, measures = tmp_path / "stories.csv", tmp_path / "meta.csv"
    pd.DataFrame([dict(File_ID=766,Group="TD",Age_Raw="7;11.15",transcript_path="TD/A/766.cha",story_id="A1",story_text="He go .")]).to_csv(source,index=False)
    pd.DataFrame([{"File_ID":766,"Group":"TD","Age(Month)":96}]).to_csv(measures,index=False)
    records,_ = _real_records(measures_path=measures,stories_path=source,project_root=tmp_path,minimum_age_months=48,maximum_age_months=96)
    assert len(records) == 1 and records[0]["age_months"] == 95


def test_revised_roster_includes_all_48_sli_children_and_setup_keeps_them_separate(tmp_path):
    path = ROOT / "configs/generation/experiment.json"
    config,schedule,evaluation,rosters = compile_experiment_design(path)
    assert {key:len(value) for key,value in rosters.items()} == {"4-year-old":11,"5-year-old":13,"6-year-old":11,"7-year-old":13}
    assert len(schedule) == 2880
    assert len(schedule[["cohort","round_id"]].drop_duplicates()) == 18
    assert set(schedule.loc[schedule["sli_child_id"].eq(523),"td_child_id"]) == {517}
    prepared = prepare_colab_generation(path,ROOT,tmp_path / "run",n_replicates=1)
    assert len(prepared.run_df) == 288
    assert not prepared.run_df["td_child_id"].eq(523).any()


def test_all_raw_zip_members_and_active_story_hash_are_verified():
    root = ROOT / "data/raw/enni/2026-10-28"
    if not root.exists():
        pytest.skip("Raw ZIP is an intentionally unversioned corpus input.")
    qc = json.loads((ROOT / INPUT_REL / STORIES_NAME).with_suffix(".parsing_qc.json").read_text())
    assert hashlib.sha256((root/"ENNI.zip").read_bytes()).hexdigest() == qc["archive_sha256"]
    with zipfile.ZipFile(root/"ENNI.zip") as bundle:
        for member in bundle.infolist():
            if not member.is_dir():
                extracted = root / "transcripts" / member.filename
                if extracted.exists():
                    assert bundle.read(member) == extracted.read_bytes()
        assert bundle.testzip() is None
    view = ROOT / "data/inputs/enni_verified_2026-10-28/chat_4_to_7"
    assert len(list(view.rglob("*.cha"))) == 238
    assert not (view / "TD/B/523.cha").exists()
    assert (view / "SLI/A/0noaudio/523.cha").exists()
    inventory = json.loads((ROOT/"docs/reproducibility/study_chat_inventory.json").read_text())
    assert {r["File_ID"] for r in inventory["header_corrections"]} == {"460","601"}
    from icl_pilot.chat.story_unit_semrp import _iter_story_utterances
    from icl_pilot.chat.corpus_archives import original_source_bytes
    for row in inventory["sources"]:
        raw = original_source_bytes(ROOT/row["source_path"], ROOT)
        assert hashlib.sha256(raw).hexdigest() == row['source_sha256']
        source, prepared = raw.decode('utf-8-sig'), (ROOT/row["input_path"]).read_text()
        assert _iter_story_utterances(source,diagnostics=[]) == _iter_story_utterances(prepared,diagnostics=[])


def test_base_r_metadata_guard_corrects_boundary_ages_and_rejects_523():
    executable = shutil.which("Rscript")
    if executable is None:
        pytest.skip("Rscript is unavailable")
    helper = (ROOT / "configs/analysis/enni_metadata.R").as_posix()
    manifest = (ROOT / INPUT_REL / PARTICIPANTS_NAME).as_posix()
    script = """
source('%s')
manifest <- '%s'
x <- data.frame(File_ID=c('414','443','543','553','732','766'),Group='TD',check.names=FALSE)
x[['Age(Month)']] <- c(60,60,72,72,96,96)
y <- reconcile_enni_metadata(x, manifest)
stopifnot(identical(as.integer(y[['Age(Month)']]), c(59L,59L,71L,71L,95L,95L)))
bad <- data.frame(File_ID='523',Group='TD')
rejected <- tryCatch({reconcile_enni_metadata(bad,manifest); FALSE},error=function(e) grepl('523',conditionMessage(e)))
stopifnot(rejected)
""" % (helper, manifest)
    result = subprocess.run([executable,"--vanilla","-e",script],capture_output=True,text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_full_revised_schedule_builds_shared_classifier_splits_without_fitting(tmp_path):
    from icl_pilot.evaluation.framework_data import load_protocol, prepare_evaluation
    from icl_pilot.evaluation.framework_classifier import training_jobs
    from icl_pilot.evaluation.framework_text import TEXT_POLICY

    protocol = load_protocol(ROOT)
    _, schedule, _, _ = compile_experiment_design(ROOT / "configs/generation/experiment.json")
    real, _ = _real_records(measures_path=ROOT/INPUT_REL/PARTICIPANTS_NAME,
        stories_path=ROOT/INPUT_REL/STORIES_NAME,project_root=ROOT,minimum_age_months=48,maximum_age_months=96)
    generated = []
    # Software fixtures stay inside this test's temporary directory. No model
    # outputs or scientific results are manufactured or installed in the project.
    for model in protocol["generator_keys"]:
        identity = protocol["generator_identities"][model]
        for row in schedule.itertuples(index=False):
            parent = "Fixture parent {} {} {} {}.".format(model,row.pair_id,row.replicate_id,row.target_story)
            parent_hash = hashlib.sha256(parent.encode()).hexdigest()
            for stage,group,child in [("stage1_td","TD",row.td_child_id),("stage2_sli","SLI",row.sli_child_id)]:
                text = parent if stage == "stage1_td" else "Fixture rewrite " + parent
                generated.append(dict(source="synthetic",cohort_year=int(row.cohort[0]),source_child_id=child,
                    td_child_id=row.td_child_id,sli_child_id=row.sli_child_id,pair_id=row.pair_id,round_id=row.round_id,
                    replicate_id=row.replicate_id,story_id=row.target_story,stage=stage,group=group,
                    model_key=model,model_id=identity["id"],model_revision=identity["revision"],run_id="software_fixture",
                    story_text=text,raw_story_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    stage1_text_sha256=parent_hash,stage2_parent_text_sha256=parent_hash,
                    stage1_prompt_record_id=parent+"stage1",stage2_prompt_record_id=parent+"stage2",
                    text_normalization=TEXT_POLICY))
    external = pd.DataFrame([dict(source="real_external",cohort_year=5,source_child_id=child,
        group=group,story_id="fixture",story_text="External fixture.",text_normalization=TEXT_POLICY)
        for child,group in [(1,"TD"),(2,"SLI")]])
    study = prepare_evaluation(pd.concat([pd.DataFrame(real),pd.DataFrame(generated)],ignore_index=True),
                               external,protocol,tmp_path/"fixture_plan")
    # The current all-child protocol has five shared folds and 24 fits per fold.
    assert len(study.rounds) == 5 and len(training_jobs(study)) == 120
    for frames in study.rounds.values():
        sets = [set(frames[key]["child_key"]) for key in ("train","validation","test")]
        assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])


def test_official_zip_alone_reproduces_the_definitive_inputs(tmp_path):
    from icl_pilot.maintenance.enni_corpus import build_roster, build_study_chat_view
    archive = ROOT/"data/raw/enni/2026-10-28/ENNI.zip"
    if not archive.exists():
        pytest.skip("Official corpus ZIP is an unversioned source input")
    fresh = tmp_path/"fresh_intake"
    qc = intake(archive,fresh)
    report = build_roster(fresh)
    view = build_study_chat_view(fresh)
    assert qc["study_children"] == 238 and report["pairs"] == 48 and view["files"] == 238
    for relative in (INPUT_REL/STORIES_NAME,INPUT_REL/PARTICIPANTS_NAME,
                     Path("configs/generation/master_frozen_roster_enni_2026-10-28.csv")):
        assert (fresh/relative).read_bytes() == (ROOT/relative).read_bytes()
