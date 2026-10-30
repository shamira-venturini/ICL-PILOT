"""Age-stratified classifier folds, independent of generation-batch boundaries."""
from __future__ import annotations

import numpy as np

from .framework_data import digest


def assign_classifier_folds(real, pair_children, protocol):
    settings = protocol["classification_split"]
    n_folds = settings["n_splits"]
    frame = real.drop(columns=["pair_id", "round_id"], errors="ignore").merge(
        pair_children, on=["cohort_year", "group", "source_child_id"], how="left", validate="many_to_one")
    identity = ["cohort_year", "group", "source_child_id"]
    expected = set(map(tuple, pair_children[identity].to_numpy()))
    observed = set(map(tuple, frame.loc[frame["pair_id"].notna(), identity].to_numpy()))
    if not expected <= observed:
        raise ValueError("A synthetic prompt child has no matching real age/group reference records.")
    frame["pair_id"] = frame["pair_id"].fillna("")
    frame["is_prompt_child"] = frame["pair_id"].ne("")
    if frame.loc[~frame["is_prompt_child"], "group"].ne("TD").any():
        raise ValueError("Every eligible SLI child must have a documented generation prompt pair.")
    frame["split_group_id"] = ["pair:"+str(age)+":"+pair if pair else child
                                for age, pair, child in zip(frame["cohort_year"], frame["pair_id"], frame["child_key"])]
    children = frame[["child_key", "source_child_id", "group", "cohort_year", "pair_id", "split_group_id", "is_prompt_child"]].drop_duplicates()
    for child, rows in frame.groupby("child_key"):
        if len(rows) != len(protocol["story_ids"]) or set(rows["story_id"]) != set(protocol["story_ids"]):
            raise ValueError("A real child lacks the complete configured story set: "+child)
    mapping = {}
    for age in protocol["cohorts"]:
        cohort = children.loc[children["cohort_year"].eq(age)]
        for kind, prompt in (("paired", True), ("unmatched_TD", False)):
            groups = sorted(cohort.loc[cohort["is_prompt_child"].eq(prompt), "split_group_id"].unique())
            if prompt and len(groups) < n_folds:
                raise ValueError("Each age cohort needs at least n_splits complete prompt pairs.")
            rng = np.random.default_rng(int(digest([settings["seed"], age, kind])[:8], 16))
            rng.shuffle(groups)
            order = rng.permutation(n_folds)
            mapping.update({group: "fold_{:02d}".format(int(order[i % n_folds])+1) for i, group in enumerate(groups)})
    frame["heldout_fold"] = frame["split_group_id"].map(mapping)
    if frame["heldout_fold"].isna().any():
        raise ValueError("An eligible real child was not assigned to a classifier fold.")
    return frame


def classifier_partitions(real, protocol):
    n_folds = protocol["classification_split"]["n_splits"]
    for index in range(n_folds):
        test_fold = "fold_{:02d}".format(index+1)
        validation_fold = "fold_{:02d}".format((index+1) % n_folds+1)
        test = real["heldout_fold"].eq(test_fold)
        validation = real["heldout_fold"].eq(validation_fold)
        frames = {"train": real.loc[~test & ~validation].copy().reset_index(drop=True),
                  "validation": real.loc[validation].copy().reset_index(drop=True),
                  "test": real.loc[test].copy().reset_index(drop=True)}
        groups = {key: set(rows["split_group_id"]) for key, rows in frames.items()}
        if any(groups[a] & groups[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
            raise ValueError("Classifier partitions share a child or prompt pair.")
        yield test_fold, frames
