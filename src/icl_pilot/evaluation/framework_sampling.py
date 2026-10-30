"""Training-only bundle bootstrap matched to the real fold's clinical/age counts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .framework_data import digest

BUNDLE_FIELDS = ["cohort_year", "pair_id", "replicate_id", "stage"]


def bootstrap_synthetic(pools, real_train, protocol, fold_id, ratio):
    if type(ratio) is not int or ratio < 1:
        raise ValueError("Synthetic:real ratios must be positive integer training-draw ratios.")
    if set(pools) != set(protocol["generator_keys"]):
        raise ValueError("Synthetic training pools must contain all configured generators.")
    parents = set(real_train["source_child_id"].astype(str))
    reference = pools[protocol["generator_keys"][0]]
    story_ids = set(protocol["story_ids"])
    expected_schedule = None
    for generator, pool in pools.items():
        if pool.duplicated(BUNDLE_FIELDS+["story_id"]).any() or not pool["source"].eq("synthetic").all():
            raise ValueError("Bootstrap pools must contain unique original synthetic stories.")
        if not set(pool["td_child_id"].astype(str)) <= parents or not set(pool["sli_child_id"].astype(str)) <= parents:
            raise ValueError("A bootstrap pool includes a validation/test prompt ancestor.")
        for _, rows in pool.groupby(BUNDLE_FIELDS):
            if set(rows["story_id"]) != story_ids or len(rows) != len(story_ids) or rows["group"].nunique() != 1:
                raise ValueError("Bootstrap requires complete, single-class story bundles.")
        schedule = set(map(tuple, pool[BUNDLE_FIELDS+["story_id"]].to_numpy()))
        if expected_schedule is not None and schedule != expected_schedule:
            raise ValueError("Bootstrap source schedules differ between generators.")
        expected_schedule = schedule
    draws = []
    for (age, group), rows in real_train.groupby(["cohort_year", "group"], sort=True):
        n_children = rows["source_child_id"].nunique()
        if len(rows) != n_children*len(story_ids):
            raise ValueError("Real training children must contribute complete story bundles.")
        eligible = reference.loc[reference["cohort_year"].eq(age) & reference["group"].eq(group)]
        bundles = eligible[BUNDLE_FIELDS].drop_duplicates().sort_values(BUNDLE_FIELDS).reset_index(drop=True)
        if bundles.empty:
            raise ValueError("No training-only synthetic bundle supports an age/group stratum.")
        seed = int(digest([protocol["sampling_seed"], fold_id, int(age), group, "bundle_bootstrap"] )[:8], 16)
        rng = np.random.default_rng(seed)
        # The seed does not depend on ratio, so smaller levels are prefixes of larger draws.
        selected = bundles.iloc[rng.integers(0, len(bundles), size=ratio*n_children)].to_dict("records")
        for index, bundle in enumerate(selected, start=1):
            draws.append({**bundle, "bootstrap_draw_index": index,
                          "bootstrap_draw_id": digest([fold_id, int(age), group, seed, index])[:24],
                          "bootstrap_source_bundle_id": digest([bundle[k] for k in BUNDLE_FIELDS])[:24],
                          "bootstrap_seed": seed})
    selection = pd.DataFrame(draws)
    result, membership = {}, []
    target = real_train.groupby(["cohort_year", "group", "story_id"]).size()*ratio
    for generator, pool in pools.items():
        sample = selection.merge(pool, on=BUNDLE_FIELDS, how="left", validate="many_to_many")
        if sample["sample_id"].isna().any():
            raise ValueError("A selected source bundle was absent from a generator.")
        actual = sample.groupby(["cohort_year", "group", "story_id"]).size()
        if not actual.reindex(target.index, fill_value=0).equals(target) or len(sample) != ratio*len(real_train):
            raise ValueError("Synthetic draws do not match the real training age/group/story distribution.")
        sample = sample.sort_values(["bootstrap_draw_id", "story_id"]).reset_index(drop=True)
        if sample.duplicated(["bootstrap_draw_id", "story_id"]).any():
            raise ValueError("A bootstrap draw contains duplicate story slots.")
        result[generator] = sample
        fields = ["sample_id"]+BUNDLE_FIELDS+["story_id", "group", "source_child_id",
                 "bootstrap_draw_id", "bootstrap_draw_index", "bootstrap_source_bundle_id", "bootstrap_seed"]
        membership.extend({"ratio_level": ratio, "training_round": fold_id, "generator_key": generator, **row}
                          for row in sample[fields].to_dict("records"))
    return result, membership
