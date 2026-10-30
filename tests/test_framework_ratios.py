"""Ratio selection, held-out pairing, fit reuse and notebook execution without GPUs."""
import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from test_framework_revision import fixture_study, prepared, fixture_backend
from icl_pilot.evaluation.framework_classifier import run_classifiers
from icl_pilot.evaluation.framework_data import JOB_FIELDS, frame_digest, digest
from icl_pilot.evaluation.framework_ratios import (
    prepare_ratio_experiment, run_ratio_experiment, summarize_ratio_experiment,
)

ROOT = Path(__file__).resolve().parents[1]


def test_ratios_keep_real_splits_balance_and_nested_common_descendants(fixture_study, tmp_path):
    study = prepared(fixture_study, tmp_path)
    before = {name: frame_digest(getattr(study, name)) for name in ("real", "synthetic", "external")}
    experiment = prepare_ratio_experiment(study, tmp_path/"ratios", ratios=(1, 2))
    assert len(experiment.plan) == 3*2*(1+2*(1+2))
    assert set(experiment.plan["training_setting"]) == {"real-only", "real-only-update-matched", "synthetic-train-augment"}
    for round_id, original in study.rounds.items():
        for ratio in (1, 2):
            frames = experiment.studies[ratio].rounds[round_id]
            for split in ("train", "validation", "test"):
                pd.testing.assert_frame_equal(frames[split], original[split])
            schedule = None
            for generator in study.protocol["generator_keys"]:
                rows = frames["synthetic"][generator]
                assert len(rows) == ratio*len(frames["train"])
                real_counts = frames["train"]["Label"].value_counts().sort_index()
                pd.testing.assert_series_equal(rows["Label"].value_counts().sort_index(), real_counts*ratio)
                keys = set(map(tuple, rows[list(JOB_FIELDS)+["stage", "bootstrap_draw_id"]].to_numpy()))
                if schedule is not None:assert keys == schedule
                schedule = keys
                low = experiment.studies[1].rounds[round_id]["synthetic"][generator]
                assert set(low["bootstrap_draw_id"]) <= set(rows["bootstrap_draw_id"])
                assert rows.groupby("bootstrap_draw_id")["story_id"].nunique().eq(len(study.protocol["story_ids"])).all()
                parents = {"enni:"+x for x in rows["td_child_id"]} | {"enni:"+x for x in rows["sli_child_id"]}
                assert parents <= set(frames["train"]["child_key"])
                assert parents.isdisjoint(set(frames["validation"]["child_key"]) | set(frames["test"]["child_key"]))
        for seed in study.protocol["classifier"]["seeds"]:
            for ratio in (1, 2):
                jobs = experiment.plan.loc[lambda x: x["training_round"].eq(round_id) & x["classifier_seed"].eq(seed) & x["ratio_level"].eq(ratio)]
                control = jobs.loc[jobs["training_setting"].eq("real-only-update-matched")].iloc[0]
                assert jobs["max_steps"].eq(control["max_steps"]).all()
                augmented = jobs.loc[jobs["training_setting"].eq("synthetic-train-augment")]
                assert augmented["effective_synthetic_to_real_ratio"].eq(ratio).all()
    assert before == {name: frame_digest(getattr(study, name)) for name in before}


@pytest.mark.parametrize("ratios", [(), (0, 1), (True,), (1.5,), (1, 1)])
def test_invalid_ratios_stop_without_creating_outputs(fixture_study, tmp_path, ratios):
    study = prepared(fixture_study, tmp_path)
    destination = tmp_path/"invalid"
    with pytest.raises(ValueError):prepare_ratio_experiment(study, destination, ratios=ratios)
    assert not destination.exists()


def test_changed_ratio_plan_does_not_replace_existing_evidence(fixture_study, tmp_path):
    study = prepared(fixture_study, tmp_path)
    output = tmp_path/"ratios"
    prepare_ratio_experiment(study, output, ratios=(1, 2))
    original = (output/"experiment_manifest.json").read_bytes()
    with pytest.raises(ValueError, match="Existing ratio experiment"):
        prepare_ratio_experiment(study, output, ratios=(1,))
    assert (output/"experiment_manifest.json").read_bytes() == original


def test_bootstrap_can_exceed_unique_pool_and_rejects_empty_class_support(fixture_study, tmp_path):
    study = prepared(fixture_study, tmp_path)
    experiment = prepare_ratio_experiment(study, tmp_path/"large_ratio", ratios=(10,))
    for frames in experiment.studies[10].rounds.values():
        for sample in frames["synthetic"].values():
            assert len(sample) == 10*len(frames["train"])
            assert sample["sample_id"].nunique() < len(sample)
    first = next(iter(study.rounds.values()))
    for generator, rows in first["synthetic_pool"].items():
        first["synthetic_pool"][generator] = rows.loc[rows["group"].eq("TD")]
    with pytest.raises(ValueError, match="No training-only synthetic bundle"):
        prepare_ratio_experiment(study, tmp_path/"ratios", ratios=(1, 2))


def test_completed_main_fits_are_reused_and_ratio_resume_never_refits(fixture_study, tmp_path):
    study = prepared(fixture_study, tmp_path)
    calls = []
    def backend(*args):
        calls.append(args[0]["classifier_run_id"])
        return fixture_backend(*args)
    main = tmp_path/"main"
    run_classifiers(study, main, backend=backend)
    main_calls = len(calls)
    experiment = prepare_ratio_experiment(study, tmp_path/"ratios", ratios=(1, 2))
    predictions = run_ratio_experiment(experiment, backend=backend, reuse_from=(main,))
    # Only the lower-ratio control + two lower-ratio generator fits are new.
    assert len(calls)-main_calls == 3*2*(1+2)
    assert set(predictions["ratio_level"]) == {0, 1, 2}
    count = len(calls)
    resumed = run_ratio_experiment(experiment, backend=backend, reuse_from=(main,))
    assert len(calls) == count and len(predictions) == len(resumed)
    assert len(list(experiment.output_dir.rglob("reused_from.json"))) == 3*2*(1+1+2)
    summary = summarize_ratio_experiment(predictions, study.protocol, experiment.output_dir/"summary")
    assert summary["ratio_differences"]["difference"].eq(0).all()
    assert set(summary["summary"]["ratio_level"]) == {0, 1, 2}


def test_corrupt_cache_predictions_are_rejected(fixture_study, tmp_path):
    study = prepared(fixture_study, tmp_path)
    main = tmp_path/"main"
    run_classifiers(study, main, backend=fixture_backend)
    for path in main.rglob("story_predictions.csv"):
        path.write_text(path.read_text()+"\n")
    experiment = prepare_ratio_experiment(study, tmp_path/"ratios", ratios=(1, 2))
    with pytest.raises(ValueError, match="failed verification"):
        run_ratio_experiment(experiment, backend=fixture_backend, reuse_from=(main,))


def test_direct_ratio_contrast_is_higher_minus_lower(fixture_study, tmp_path):
    study = prepared(fixture_study, tmp_path)
    def backend(job, train, validation, tests, protocol, target):
        output = fixture_backend(job, train, validation, tests, protocol, target)
        if job["training_setting"] == "synthetic-train-augment" and job["effective_synthetic_to_real_ratio"] == 1:
            output = {name: values[:, ::-1] for name, values in output.items()}
        return output
    experiment = prepare_ratio_experiment(study, tmp_path/"ratios", ratios=(1, 2))
    predictions = run_ratio_experiment(experiment, backend=backend)
    result = summarize_ratio_experiment(predictions, study.protocol, experiment.output_dir/"summary")
    differences = result["ratio_differences"].loc[lambda x: x["metric"].eq("balanced_accuracy")]
    assert differences["difference"].eq(1).all()
    assert differences["ci_low"].eq(1).all() and differences["ci_high"].eq(1).all()
    assert differences["lower_ratio"].eq(1).all() and differences["higher_ratio"].eq(2).all()


def test_notebook_ratio_block_runs_without_main_classifier_cells(fixture_study, tmp_path, monkeypatch):
    from icl_pilot.evaluation import framework_classifier
    records, external, protocol = fixture_study
    protocol = copy.deepcopy(protocol)
    protocol["synthetic_to_real_ratio"] = 10
    protocol.pop("protocol_sha256")
    protocol["protocol_sha256"] = digest(protocol)
    template = records.loc[records["source"].eq("synthetic") & records["replicate_id"].eq(1)]
    generated = []
    for rep in range(1, 11):
        rows = template.copy(); rows["replicate_id"] = rep
        for field in ("stage1_prompt_record_id", "stage2_prompt_record_id"):
            rows[field] = rows[field].astype(str) + ":rep" + str(rep)
        generated.append(rows)
    records = pd.concat([records.loc[records["source"].eq("real")]]+generated, ignore_index=True)
    study = prepared((records, external, protocol), tmp_path)
    monkeypatch.setattr(framework_classifier, "package_backend", fixture_backend)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    notebook = json.loads((ROOT/"notebooks/synthtexteval_demo.ipynb").read_text())
    cells = {c["id"]: ''.join(c["source"]) for c in notebook["cells"]}
    assert 'SYNTHETIC_REAL_RATIOS = (1, 10)' in cells["ratio-experiment-plan"]
    namespace = dict(prepared=study, protocol=study.protocol, RESULTS_DIR=tmp_path/"notebook",
                     pd=pd, display=lambda *_: None)
    exec(cells["ratio-experiment-plan"], namespace)
    exec(cells["ratio-experiment-run"], namespace)
    assert namespace["ratio_story_predictions"].empty
    assert not list((tmp_path/"notebook").rglob("completed.json"))
    namespace["RUN_RATIO_EXPERIMENT"] = True
    exec(cells["ratio-experiment-run"], namespace)
    exec(cells["ratio-experiment-summary"], namespace)
    assert (tmp_path/"notebook/ratio_experiment/summary/ratio_differences.csv").exists()
    assert (tmp_path/"notebook/ratio_experiment/child_balanced_accuracy_by_ratio.png").exists()
    assert "story_predictions" not in namespace
    assert notebook["cells"][-1]["id"] == "7acc85f1"
