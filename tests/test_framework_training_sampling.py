"""All-child classifier folds and training-only prior-matched bundle draws."""
import copy
from collections import Counter

import pandas as pd
import pytest

from test_framework_revision import fixture_study, prepared
from icl_pilot.evaluation.framework_data import prepare_evaluation
from icl_pilot.evaluation.framework_sampling import bootstrap_synthetic
from icl_pilot.evaluation.framework_classifier import training_jobs


def test_all_children_rotate_through_test_and_validation_once(fixture_study, tmp_path):
    study = prepared(fixture_study, tmp_path)
    for split in ("test", "validation"):
        counts = Counter(child for frames in study.rounds.values() for child in frames[split]["child_key"].unique())
        assert set(counts) == set(study.real["child_key"]) and set(counts.values()) == {1}
    membership = study.real[["split_group_id", "heldout_fold"]].drop_duplicates()
    assert not membership["split_group_id"].duplicated().any()
    assert study.real["is_prompt_child"].sum() < len(study.real)
    assert not (study.output_dir/"unmatched_real_children_excluded_from_classification.csv").exists()


def test_classification_folds_do_not_follow_generation_batches(fixture_study, tmp_path):
    records, external, protocol = fixture_study
    original = prepare_evaluation(records, external, protocol, tmp_path/"a")
    changed = records.copy()
    changed.loc[changed["source"].eq("synthetic"), "round_id"] = "one_generation_batch"
    revised = prepare_evaluation(changed, external, protocol, tmp_path/"b")
    for fold in original.rounds:
        for split in ("train", "validation", "test"):
            assert set(original.rounds[fold][split]["sample_id"]) == set(revised.rounds[fold][split]["sample_id"])


def test_age_group_story_counts_are_preserved_with_different_age_distributions(fixture_study, tmp_path):
    records, external, protocol = fixture_study
    older = records.copy()
    older["cohort_year"] = 5
    for field in ("source_child_id", "td_child_id", "sli_child_id"):
        older[field] = older[field]+1000
    older.loc[older["source"].eq("synthetic"), "pair_id"] += "_older"
    older = older.loc[~(older["source"].eq("real") & older["source_child_id"].between(1301, 1308))]
    protocol = copy.deepcopy(protocol); protocol["cohorts"] = [4, 5]
    study = prepare_evaluation(pd.concat([records, older], ignore_index=True), external, protocol, tmp_path/"multiage")
    for fold, frames in study.rounds.items():
        low, _ = bootstrap_synthetic(frames["synthetic_pool"], frames["train"], protocol, fold, 1)
        high, _ = bootstrap_synthetic(frames["synthetic_pool"], frames["train"], protocol, fold, 10)
        expected = frames["train"].groupby(["cohort_year", "group", "story_id"]).size()
        for model in protocol["generator_keys"]:
            pd.testing.assert_series_equal(high[model].groupby(["cohort_year", "group", "story_id"]).size(), expected*10)
            small = Counter(zip(low[model]["sample_id"], low[model]["bootstrap_draw_id"]))
            large = Counter(zip(high[model]["sample_id"], high[model]["bootstrap_draw_id"]))
            assert all(large[key] >= count for key, count in small.items())


def test_training_jobs_preserve_real_prior_and_log_unique_counts(fixture_study, tmp_path):
    study = prepared(fixture_study, tmp_path)
    for job, train, validation, internal in training_jobs(study):
        assert job["n_TD_train"] == 2*job["n_SLI_train"]
        assert job["n_unique_synthetic_train"] <= job["n_synthetic_train"]
        assert job["real_reference_TD"] == 2*job["real_reference_SLI"]
        assert not set(train["source_child_id"]) & (set(validation["source_child_id"]) | set(internal["source_child_id"]))


def test_bootstrap_rejects_heldout_ancestors_and_incomplete_bundles(fixture_study, tmp_path):
    study = prepared(fixture_study, tmp_path)
    fold, frames = next(iter(study.rounds.items()))
    pools = {key: value.copy() for key, value in frames["synthetic_pool"].items()}
    key = study.protocol["generator_keys"][0]
    pools[key].iloc[0, pools[key].columns.get_loc("td_child_id")] = "999999"
    with pytest.raises(ValueError, match="validation/test prompt ancestor"):
        bootstrap_synthetic(pools, frames["train"], study.protocol, fold, 1)
    pools = {key: value.iloc[1:].copy() for key, value in frames["synthetic_pool"].items()}
    with pytest.raises(ValueError, match="complete"):
        bootstrap_synthetic(pools, frames["train"], study.protocol, fold, 1)
