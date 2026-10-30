"""Scientific invariants of the revised framework; no checkpoint downloads."""
import copy
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from icl_pilot.evaluation.framework_data import digest, load_protocol, prepare_evaluation
from icl_pilot.evaluation.framework_classifier import run_classifiers, training_jobs, tokenize_audit
from icl_pilot.evaluation.framework_metrics import metric_values, prediction_units, paired_intervals, summarize_predictions
from icl_pilot.evaluation.framework_text import normalize_framework_text, TEXT_POLICY
from icl_pilot.evaluation.framework_quality import frechet_distance
from icl_pilot.evaluation.framework_descriptive import lexical_comparison

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fixture_study(tmp_path):
    protocol = json.loads((ROOT / "configs/evaluation/protocol.json").read_text())
    protocol.update(generator_keys=["alpha", "beta"], cohorts=[4], story_ids=["A1", "B1"], synthetic_to_real_ratio=2)
    protocol["classification_split"]["n_splits"] = 3
    protocol["classifier"].update(seeds=[3, 4], epochs=1, batch_size=4)
    protocol["uncertainty"].update(bootstrap_samples=20)
    protocol["generator_identities"] = {key: {"id": "fixture/" + key, "revision": letter * 40}
                                         for key, letter in [("alpha", "a"), ("beta", "b")]}
    protocol["protocol_sha256"] = digest(protocol)
    real, generated = [], []
    for index in range(9):
        td, sli = 100 + index, 200 + index
        pair = "pair_" + str(index)
        round_id = "round_0" + str(index // 3 + 1)
        for group, child in [("TD", td), ("SLI", sli)]:
            for story in protocol["story_ids"]:
                real.append(dict(source="real", cohort_year=4, source_child_id=child, group=group,
                                 story_id=story, story_text=f"Child {child} tell {story}.", text_normalization=TEXT_POLICY))
        for model in protocol["generator_keys"]:
            identity = protocol["generator_identities"][model]
            for rep in [1, 2]:
                for story in protocol["story_ids"]:
                    parent = f"{model} stage1_td {story} {rep}."
                    parent_hash = hashlib.sha256(parent.encode()).hexdigest()
                    for stage, group, child in [("stage1_td", "TD", td), ("stage2_sli", "SLI", sli)]:
                        text = f"{model} {stage} {story} {rep}."
                        generated.append(dict(source="synthetic", cohort_year=4, source_child_id=child,
                            td_child_id=td, sli_child_id=sli, pair_id=pair, round_id=round_id,
                            replicate_id=rep, story_id=story, stage=stage, group=group,
                            model_key=model, model_id=identity["id"], model_revision=identity["revision"],
                            run_id="run_" + model, story_text=text,
                            raw_story_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                            stage1_text_sha256=parent_hash, stage2_parent_text_sha256=parent_hash,
                            stage1_prompt_record_id=f"{model}:{pair}:{rep}:{story}:stage1",
                            stage2_prompt_record_id=f"{model}:{pair}:{rep}:{story}:stage2",
                            test_only=False, text_normalization=TEXT_POLICY))
    for child in range(300, 309):
        for story in protocol["story_ids"]:
            real.append(dict(source="real", cohort_year=4, source_child_id=child, group="TD",
                             story_id=story, story_text=f"Unmatched TD child {child} tell {story}.", text_normalization=TEXT_POLICY))
    # These equal local child numbers are legitimate in a different corpus namespace.
    external = pd.DataFrame([dict(source="real_external", cohort_year=5, source_child_id=child,
        group=group, story_id=story, story_text=f"Gillam {child} {story}.", text_normalization=TEXT_POLICY)
        for group, children in [("TD", [100, 101]), ("SLI", [200, 201])]
        for child in children for story in ["Aliens", "LateForSchool"]])
    # CSV roundtrip reproduces float promotion of prompt-child/replicate fields.
    source = tmp_path / "records.csv"
    pd.DataFrame(real + generated).to_csv(source, index=False)
    return pd.read_csv(source), external, protocol


def prepared(fixture_study, tmp_path):
    records, external, protocol = fixture_study
    return prepare_evaluation(records, external, protocol, tmp_path / "evaluation")


def test_protocol_pins_model_and_validates_incompatible_token_limit(tmp_path):
    protocol = json.loads((ROOT / "configs/evaluation/protocol.json").read_text())
    protocol["generator_catalog"] = str(ROOT / "configs/generation/models.json")
    path = tmp_path / "protocol.json";path.write_text(json.dumps(protocol))
    loaded = load_protocol(tmp_path, path)
    assert set(loaded["generator_identities"]) == set(protocol["generator_keys"])
    protocol["classifier"]["max_length"] = 256
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="512"):
        load_protocol(tmp_path, path)


def test_shared_splits_keep_prompt_children_and_all_descendants_out_of_test(fixture_study, tmp_path):
    study = prepared(fixture_study, tmp_path)
    assert len(study.rounds) == 3
    for frames in study.rounds.values():
        train, val, test = [set(frames[k]["child_key"]) for k in ["train", "validation", "test"]]
        assert not train & val and not train & test and not val & test
        for generated in frames["synthetic"].values():
            parents = {"enni:" + str(x) for x in generated["td_child_id"]} | {"enni:" + str(x) for x in generated["sli_child_id"]}
            assert parents <= train and not parents & (test | val)
        pool = next(iter(frames["synthetic_pool"].values()))
        parents = {"enni:"+str(x) for x in pool["td_child_id"]} | {"enni:"+str(x) for x in pool["sli_child_id"]}
        assert parents == set(frames["train"].loc[frames["train"]["is_prompt_child"], "child_key"])
    assert set(study.external["child_key"]).isdisjoint(study.real["child_key"])


def test_reference_sample_and_replicates_are_common_across_generators(fixture_study, tmp_path):
    study = prepared(fixture_study, tmp_path)
    pd.testing.assert_frame_equal(study.samples["alpha"]["real"], study.samples["beta"]["real"])
    for frames in study.rounds.values():
        keys = ["pair_id", "replicate_id", "story_id", "stage", "bootstrap_draw_id"]
        assert set(map(tuple, frames["synthetic"]["alpha"][keys].to_numpy())) == set(map(tuple, frames["synthetic"]["beta"][keys].to_numpy()))
    synthetic_ids = study.synthetic["sample_id"]
    assert synthetic_ids.is_unique


@pytest.mark.parametrize("failure", ["missing_model", "missing_stage", "missing_target", "wrong_revision", "ambiguous_pair", "test_fixture", "wrong_parent"])
def test_invalid_lineage_stops_before_creating_results(fixture_study, tmp_path, failure):
    records, external, protocol = fixture_study
    records = records.copy()
    indexes = records.index[records["source"].eq("synthetic")]
    if failure == "missing_model":records = records.loc[~records["model_key"].eq("beta")]
    elif failure == "missing_stage":records = records.drop(indexes[0])
    elif failure == "missing_target":records = records.drop(indexes[:2])
    elif failure == "wrong_revision":records.loc[indexes[0], "model_revision"] = "c" * 40
    elif failure == "ambiguous_pair":records.loc[indexes[0], "td_child_id"] = 999
    elif failure == "wrong_parent":records.loc[indexes[1], "stage2_parent_text_sha256"] = "f" * 64
    else:records.loc[indexes[0], "test_only"] = True
    destination = tmp_path / "must_not_exist"
    with pytest.raises(ValueError):prepare_evaluation(records, external, protocol, destination)
    assert not destination.exists()


def test_unmatched_td_is_audited_and_included_in_classifier_folds(fixture_study, tmp_path):
    records, external, protocol = fixture_study
    extra = records.loc[records["source"].eq("real") & records["source_child_id"].eq(100)].copy()
    extra["source_child_id"] = 999
    study = prepare_evaluation(pd.concat([records, extra]), external, protocol, tmp_path / "eval")
    for split in ("train", "validation", "test"):
        assert sum("enni:999" in set(frames[split]["child_key"]) for frames in study.rounds.values()) == 1
    audit = pd.read_csv(study.output_dir / "unmatched_TD_included_in_classification.csv")
    assert set(audit["source_child_id"]) == set(range(300, 309)) | {999}
    tests = pd.concat([f["test"] for f in study.rounds.values()])
    assert not tests["sample_id"].duplicated().any() and len(tests) == len(study.real)


def test_one_shared_baseline_and_step_control_per_round_seed(fixture_study, tmp_path):
    study = prepared(fixture_study, tmp_path)
    jobs = training_jobs(study)
    assert len(jobs) == 3 * 2 * 6
    for round_id in study.rounds:
        for seed in [3, 4]:
            scope = [j for j, *_ in jobs if j["training_round"] == round_id and j["classifier_seed"] == seed]
            assert sum(j["training_setting"] == "real-only" for j in scope) == 1
            augmented = [j for j in scope if j["training_setting"] == "synthetic-train-augment"]
            control = next(j for j in scope if j["training_setting"] == "real-only-update-matched")
            assert all(j["max_steps"] == control["max_steps"] for j in augmented)
            assert all(j["effective_synthetic_to_real_ratio"] == 2 for j in augmented)


def fixture_backend(job, train, validation, tests, protocol, target):
    assert set(tests) == {"enni", "gillam"}
    assert set(tests["gillam"]["source"]) == {"real_external"}
    assert "real_external" not in set(train["source"]) | set(validation["source"])
    output = {}
    for name, frame in tests.items():
        probabilities = np.where(frame["Label"].eq(0), .8, .2)
        output[name] = np.column_stack([probabilities, 1 - probabilities])
    return output


def test_prediction_exports_resume_and_detect_modified_results(fixture_study, tmp_path):
    study = prepared(fixture_study, tmp_path)
    calls = []
    def backend(*args):
        calls.append(args[0]["classifier_run_id"])
        return fixture_backend(*args)
    predictions = run_classifiers(study, backend=backend)
    assert set(predictions["evaluation_dataset"]) == {"enni", "gillam"}
    assert predictions.groupby("classifier_run_id")["evaluation_dataset"].nunique().eq(2).all()
    count = len(calls)
    resumed = run_classifiers(study, backend=backend)
    assert len(calls) == count
    assert len(resumed) == len(predictions)
    with pytest.raises(ValueError, match="Completed classifier job"):
        run_classifiers(study, backend=fixture_backend)
    path = next((study.output_dir / "classification").rglob("story_predictions.csv"))
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="Completed classifier job"):
        run_classifiers(study, backend=backend)


def test_child_aggregation_and_cluster_inference_do_not_count_repeats_as_children(fixture_study, tmp_path):
    study = prepared(fixture_study, tmp_path)
    stories = run_classifiers(study, backend=fixture_backend)
    units = prediction_units(stories)
    external_children = units.loc[units["evaluation_dataset"].eq("gillam") & units["unit"].eq("child")]
    assert external_children.groupby("classifier_run_id").size().eq(4).all()
    assert external_children["n_stories"].eq(2).all()
    result = summarize_predictions(stories, study.protocol, study.output_dir / "summary")
    differences = result["paired_differences"]
    assert np.allclose(differences[["difference", "ci_low", "ci_high"]], 0)
    internal = differences.loc[differences["evaluation_dataset"].eq("enni")]
    assert set(internal["unique_resampling_clusters"]) == {18}
    assert set(differences.loc[differences["evaluation_dataset"].eq("gillam"), "unique_resampling_clusters"]) == {4}
    assert set(result["by_run"]["unit"]) == {"story", "child"}


def test_pairing_refuses_different_test_samples(fixture_study, tmp_path):
    study = prepared(fixture_study, tmp_path)
    stories = run_classifiers(study, backend=fixture_backend)
    units = prediction_units(stories)
    index = units.index[units["training_setting"].eq("synthetic-train-augment") & units["unit"].eq("child")][0]
    with pytest.raises(ValueError, match="identical real cases"):
        paired_intervals(units.drop(index), study.protocol["uncertainty"], .5)


def test_auc_ties_orientation_and_weighting():
    assert metric_values([0, 1], [.5, .5])["auc"] == .5
    assert metric_values([0, 1], [.9, .1])["auc"] == 1
    assert metric_values([0, 1], [.1, .9])["auc"] == 0
    weighted = metric_values([0, 1, 0, 1], [.8, .7, .4, .1], [2, 1, 1, 3])
    repeated = metric_values([0, 0, 1, 0, 1, 1, 1], [.8, .8, .7, .4, .1, .1, .1])
    assert weighted == repeated


def test_common_normalization_preserves_errors_fragments_and_repetitions():
    raw = 'He he (be)cause go [: went] (.) xxx &=laughs .'
    expected = 'He he (be)cause go.'
    assert normalize_framework_text(raw) == expected
    assert normalize_framework_text(expected) == expected


def test_truncation_audit_counts_special_tokens():
    class Tokenizer:
        def __call__(self, text, **kwargs):return {"input_ids": [1] + text.split() + [2]}
    frame = pd.DataFrame([dict(sample_id="x", source="real", group="TD", cohort_year=4,
                               source_child_id=100, output_text="one two three")])
    audit = tokenize_audit(Tokenizer(), frame, 4, {"split": "test"})
    assert audit.iloc[0]["original_tokens_including_special"] == 5
    assert bool(audit.iloc[0]["was_truncated"])


def test_frechet_uses_covariances_and_handles_singular_features():
    x = np.array([[-1., 0], [0, 0], [1, 0]])
    assert frechet_distance(x, x) == pytest.approx(0, abs=1e-7)
    assert frechet_distance(x, x + [2, 0]) == pytest.approx(4)
    assert frechet_distance(x, x * 2) == pytest.approx(1)
    assert frechet_distance(x, x * 2) == pytest.approx(frechet_distance(x * 2, x))
    with pytest.raises(ValueError):frechet_distance(x[:1], x)
    with pytest.raises(ValueError):frechet_distance(x, [[np.nan, 0], [1, 0]])


def test_shared_real_tfidf_reports_novel_vocabulary_and_document_ttr():
    result = lexical_comparison(["dog dog"], ["dog newword"])
    assert result["cosine_real_reference_tfidf_all_cross_pairs"] == pytest.approx(1)
    assert result["synthetic_tfidf_oov_token_share"] == .5
    assert result["jaccard_whitespace_all_cross_pairs"] == .5


def test_weighted_auc_matches_pairwise_oracle_with_ties():
    rng = np.random.default_rng(101)
    for _ in range(20):
        labels = np.array([0] * 5 + [1] * 5)
        probabilities = np.round(rng.uniform(size=10), 1)
        weights = rng.integers(1, 4, size=10)
        positive, negative = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
        expected = sum(weights[i] * weights[j] * ((probabilities[i] > probabilities[j]) + .5 * (probabilities[i] == probabilities[j]))
                       for i in positive for j in negative) / (weights[positive].sum() * weights[negative].sum())
        assert metric_values(labels, probabilities, weights)["auc"] == pytest.approx(expected)


def test_changed_protocol_does_not_overwrite_existing_snapshots(fixture_study, tmp_path):
    records, external, protocol = fixture_study
    destination = tmp_path / "evaluation"
    prepare_evaluation(records, external, protocol, destination)
    before = (destination / "effective_protocol.json").read_bytes()
    altered = copy.deepcopy(protocol);altered["classifier"]["seeds"] = [99]
    with pytest.raises(ValueError, match="Existing evaluation protocol"):
        prepare_evaluation(records, external, altered, destination)
    assert (destination / "effective_protocol.json").read_bytes() == before


def test_notebook_cells_wire_full_fixture_workflow_without_model_downloads(fixture_study, tmp_path, monkeypatch):
    from icl_pilot.evaluation import framework_classifier
    records, external, protocol = fixture_study
    protocol = copy.deepcopy(protocol)
    protocol["generator_catalog"] = "catalog.json"
    (tmp_path / "catalog.json").write_text(json.dumps({"models": protocol["generator_identities"]}))
    path = tmp_path / "protocol.json";path.write_text(json.dumps(protocol))
    records_path = tmp_path / "records.csv";records.to_csv(records_path,index=False)
    external_path = tmp_path / "external.csv";external.to_csv(external_path,index=False)
    output = tmp_path / "notebook-run";output.mkdir()
    namespace = dict(PROJECT_ROOT=tmp_path, PROTOCOL_CONFIG=path, TEXT_RECORDS_CSV=records_path,
        GILLAM_EXTERNAL_TEXT_CSV=external_path, RESULTS_DIR=output,
        RUN_CLASSIFICATION=True, RUN_TEXT_QUALITY=False, display=lambda *_: None)
    monkeypatch.setattr(framework_classifier, "package_backend", fixture_backend)
    notebook = json.loads((ROOT / "notebooks/synthtexteval_demo.ipynb").read_text())
    cells = {c["id"]: ''.join(c["source"]) for c in notebook["cells"]}
    for cell_id in ['hqqiVwMjHGG5','qpYZMtr0HGG5','9_LXLLYsHGG5','SWniz-ECHGG5','Eu4ZqaIMHGG6','SbMsoPGzHGG6','SOBQImycHGG6']:
        exec(cells[cell_id], namespace)
    assert (output / "classification/summary/paired_differences_cluster_bootstrap.csv").exists()
    assert set(namespace["classification_results"]["by_run"]["evaluation_dataset"]) == {"enni", "gillam"}
    assert set(namespace["classification_results"]["by_run"]["unit"]) == {"story", "child"}
