"""Paired, child-aware summaries of fixed held-out classifier predictions.

Bootstrap intervals are conditional on the fitted classifiers and recorded
splits. They do not simulate new training samples or replace training-seed
replication. A child's repeated predictions always receive the same resampling
weight; ENNI prompt pairs stay together and unmatched TD children are individual clusters.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RUN_FIELDS = ["classifier_run_id", "generator_key", "training_setting", "training_round", "classifier_seed", "evaluation_dataset", "unit"]
METRICS = ("balanced_accuracy", "auc", "sli_recall", "td_recall", "accuracy", "f1_macro", "predicted_sli_rate")


def _divide(a, b):
    return float(a / b) if b else float("nan")


def metric_values(labels, probabilities, weights=None, threshold=0.5):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities, dtype=float)
    if weights is None:
        weights = np.ones(len(labels), dtype=float)
    weights = np.asarray(weights, dtype=float)
    if not len(labels) or len(probabilities) != len(labels) or len(weights) != len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Nonempty aligned binary labels/probabilities/weights are required.")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Resampling weights must be finite and nonnegative.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Classifier probabilities must be finite and within [0, 1].")
    positive, predicted = labels == 0, probabilities >= threshold
    tp = weights[positive & predicted].sum(); fn = weights[positive & ~predicted].sum()
    tn = weights[~positive & ~predicted].sum(); fp = weights[~positive & predicted].sum()
    sensitivity, specificity = _divide(tp, tp + fn), _divide(tn, tn + fp)
    # Weighted Mann-Whitney AUC; ties receive half credit.
    order = np.argsort(probabilities, kind="stable")
    scores, pos, w = probabilities[order], positive[order], weights[order]
    _, first = np.unique(scores, return_index=True)
    positive_weight = np.add.reduceat(w * pos, first)
    negative_weight = np.add.reduceat(w * ~pos, first)
    lower_negative = np.cumsum(negative_weight) - negative_weight
    auc = _divide(np.sum(positive_weight * (lower_negative + 0.5 * negative_weight)), (tp + fn) * (tn + fp))
    f1_sli = _divide(2 * tp, 2 * tp + fp + fn)
    f1_td = _divide(2 * tn, 2 * tn + fp + fn)
    return {"balanced_accuracy": (sensitivity + specificity) / 2, "auc": auc,
            "sli_recall": sensitivity, "td_recall": specificity,
            "accuracy": _divide(tp + tn, weights.sum()), "f1_macro": (f1_sli + f1_td) / 2,
            "predicted_sli_rate": _divide(tp + fp, weights.sum())}


def prediction_units(stories):
    """Return story predictions and equal-story-weight child probabilities."""
    stories = stories.copy()
    if stories.duplicated(["classifier_run_id", "evaluation_dataset", "sample_id"]).any():
        raise ValueError("Duplicate story predictions within a classifier/test dataset.")
    for _key, group in stories.groupby(["classifier_run_id", "evaluation_dataset", "child_key"]):
        if group[["Label", "group", "cohort_year"]].nunique().gt(1).any():
            raise ValueError("Inconsistent metadata among a child's held-out stories.")
    stories["unit"] = "story"
    stories["n_stories"] = 1
    keys = [f for f in RUN_FIELDS if f != "unit"] + ["child_key", "source_child_id", "group", "cohort_year", "pair_id"]
    children = stories.groupby(keys, as_index=False, dropna=False).agg(
        p_SLI=("p_SLI", "mean"), p_TD=("p_TD", "mean"), Label=("Label", "first"), n_stories=("sample_id", "size"))
    children["sample_id"] = children["child_key"]
    children["unit"] = "child"
    return pd.concat([stories, children], ignore_index=True, sort=False)


def _cluster_columns(frame):
    frame = frame.copy()
    internal = frame["evaluation_dataset"].eq("enni")
    paired = internal & frame["pair_id"].fillna("").astype(str).ne("")
    if frame.loc[internal & ~paired, "group"].ne("TD").any():
        raise ValueError("Unmatched ENNI resampling units must be documented TD children.")
    frame["_cluster"] = np.where(paired,
        "enni:" + frame["cohort_year"].astype(str) + ":" + frame["pair_id"].astype(str),
        frame["child_key"].astype(str))
    frame["_stratum"] = np.where(internal,
        "enni:" + frame["cohort_year"].astype(str) + np.where(paired, ":prompt_pair", ":unmatched_TD"),
        "gillam:" + frame["cohort_year"].astype(str) + ":" + frame["group"].astype(str))
    return frame


def paired_intervals(predictions, settings, threshold):
    """Cluster bootstrap paired metric differences; no naive fold-level t tests."""
    comparisons = []
    frame = _cluster_columns(predictions)
    for (dataset, unit), scope in frame.groupby(["evaluation_dataset", "unit"]):
        if unit not in settings.get("units", ["child"]):
            continue
        baseline = scope.loc[scope["training_setting"].eq("real-only")]
        if baseline.empty:
            raise ValueError("A shared real-only baseline is required for paired comparisons.")
        for (generator, condition), candidate in scope.groupby(["generator_key", "training_setting"]):
            if condition == "real-only":
                continue
            references = [("real-only", baseline)]
            if condition == "synthetic-train-augment":
                control = scope.loc[scope["training_setting"].eq("real-only-update-matched")]
                if not control.empty:
                    references.append(("real-only-update-matched", control))
            for reference_name, reference in references:
                align = ["training_round", "classifier_seed", "sample_id"]
                if candidate.duplicated(align).any() or reference.duplicated(align).any():
                    raise ValueError("Ambiguous classifier pairing.")
                left_keys = set(map(tuple, candidate[align].to_numpy()))
                right_keys = set(map(tuple, reference[align].to_numpy()))
                if left_keys != right_keys:
                    raise ValueError("Compared conditions must predict identical real cases in identical rounds/seeds.")
                paired = candidate.merge(reference[align + ["p_SLI", "Label"]], on=align, validate="one_to_one", suffixes=("", "_reference"))
                if not paired["Label"].eq(paired["Label_reference"]).all():
                    raise ValueError("Compared predictions disagree about ground truth.")
                clusters = paired[["_cluster", "_stratum"]].drop_duplicates()
                if clusters.duplicated("_cluster").any():
                    raise ValueError("A resampling cluster belongs to multiple strata.")
                cluster_list = clusters["_cluster"].tolist()
                cluster_index = {key: index for index, key in enumerate(cluster_list)}
                strata = [np.asarray([cluster_index[x] for x in g["_cluster"]]) for _, g in clusters.groupby("_stratum")]
                runs = []
                for _, rows in paired.groupby(["training_round", "classifier_seed"]):
                    runs.append((rows["Label"].to_numpy(), rows["p_SLI"].to_numpy(), rows["p_SLI_reference"].to_numpy(),
                                 np.asarray([cluster_index[x] for x in rows["_cluster"]])))
                def contrast(weights):
                    diffs = []
                    for y, p, ref, indexes in runs:
                        a = metric_values(y, p, weights[indexes], threshold)
                        b = metric_values(y, ref, weights[indexes], threshold)
                        diffs.append([a[m] - b[m] for m in METRICS])
                    values = np.asarray(diffs)
                    # Do not silently average a different subset of runs when a
                    # bootstrap draw leaves a run without both true classes.
                    return values.mean(axis=0)
                estimate = contrast(np.ones(len(cluster_list)))
                rng = np.random.default_rng(settings["seed"])
                draws = []
                for _ in range(settings["bootstrap_samples"]):
                    weights = np.zeros(len(cluster_list))
                    for indexes in strata:
                        chosen = rng.choice(indexes, size=len(indexes), replace=True)
                        weights += np.bincount(chosen, minlength=len(cluster_list))
                    draws.append(contrast(weights))
                draws = np.asarray(draws)
                alpha = (1 - settings["confidence"]) / 2
                for index, metric in enumerate(METRICS):
                    valid = draws[:, index][np.isfinite(draws[:, index])]
                    low, high = np.quantile(valid, [alpha, 1 - alpha]) if len(valid) else (np.nan, np.nan)
                    comparisons.append({"evaluation_dataset": dataset, "unit": unit, "generator_key": generator,
                        "training_setting": condition, "reference_setting": reference_name, "metric": metric,
                        "difference": estimate[index], "ci_low": low, "ci_high": high,
                        "confidence": settings["confidence"], "unique_resampling_clusters": len(cluster_list),
                        "n_round_seed_comparisons": len(runs), "bootstrap_samples": settings["bootstrap_samples"],
                        "valid_bootstrap_samples": len(valid), "inference_scope": "conditional_on_recorded_fitted_models_and_splits"})
    return pd.DataFrame(comparisons)


def summarize_predictions(stories, protocol, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if stories.empty:
        raise ValueError("No held-out predictions to summarize.")
    units = prediction_units(stories)
    threshold = protocol["classifier"]["decision_threshold_sli"]
    rows, recalls = [], []
    for key, group in units.groupby(RUN_FIELDS, dropna=False):
        metadata = dict(zip(RUN_FIELDS, key))
        rows.append({**metadata, **metric_values(group["Label"], group["p_SLI"], threshold=threshold),
                     "n_observations": len(group), "unique_children": group["child_key"].nunique()})
        for (cohort, label), age_group in group.groupby(["cohort_year", "group"]):
            predicted = np.where(age_group["p_SLI"] >= threshold, 0, 1)
            recalls.append({**metadata, "cohort_year": cohort, "group": label,
                            "recall": float(np.mean(predicted == age_group["Label"])),
                            "n_observations": len(age_group), "unique_children": age_group["child_key"].nunique()})
    by_run = pd.DataFrame(rows)
    summary = by_run.groupby(["generator_key", "training_setting", "evaluation_dataset", "unit"])[list(METRICS)].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reset_index()
    summary_keys = ["generator_key", "training_setting", "evaluation_dataset", "unit"]
    counts = units.groupby(summary_keys, as_index=False).agg(
        n_fitted_classifiers=("classifier_run_id", "nunique"), n_rounds=("training_round", "nunique"),
        n_classifier_seeds=("classifier_seed", "nunique"), unique_test_children=("child_key", "nunique"),
        n_repeated_predictions=("sample_id", "size"))
    summary = summary.merge(counts, on=summary_keys, validate="one_to_one")
    intervals = paired_intervals(units, protocol["uncertainty"], threshold)
    units.to_csv(output / "all_story_and_child_predictions.csv", index=False)
    by_run.to_csv(output / "metrics_by_round_seed.csv", index=False)
    summary.to_csv(output / "summary_across_rounds_and_seeds.csv", index=False)
    pd.DataFrame(recalls).to_csv(output / "age_class_recall_by_round_seed.csv", index=False)
    age_summary = pd.DataFrame(recalls).groupby(summary_keys + ["cohort_year", "group"], as_index=False).agg(
        recall_mean=("recall", "mean"), recall_sd=("recall", "std"), n_fits=("classifier_run_id", "nunique"))
    age_summary.to_csv(output / "age_class_recall_summary.csv", index=False)
    intervals.to_csv(output / "paired_differences_cluster_bootstrap.csv", index=False)
    (output / "inference_manifest.json").write_text(json.dumps({
        "primary_unit": "child", "primary_metric": "balanced_accuracy", "secondary_metric": "auc",
        "sd_meaning": "descriptive variation across recorded round/seed fits; not an independent-sample standard error",
        "interval_scope": "conditional on fitted classifiers and recorded split schedule; no bootstrap refitting",
        "internal_clusters": "prompt pair jointly, unmatched TD child individually; separate age-by-unit-type strata; repeated predictions linked",
        "external_clusters": "unique Gillam child, stratified by age cohort and literal group",
        "multiplicity": "unadjusted descriptive confidence intervals; no automatic significance or equivalence claim",
        "settings": protocol["uncertainty"],
    }, indent=2) + "\n")
    return {"by_run": by_run, "summary": summary, "paired_differences": intervals,
            "age_class_recall": pd.DataFrame(recalls)}
