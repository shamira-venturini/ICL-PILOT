"""Separate, prespecified augmentation-ratio experiment with shared real splits."""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
import json
from pathlib import Path

import pandas as pd

from .framework_classifier import training_jobs, write_training_plan, run_classifiers
from .framework_data import frame_digest, _write_json
from .framework_sampling import bootstrap_synthetic
from .framework_metrics import prediction_units, paired_intervals, summarize_predictions


@dataclass
class RatioExperiment:
    prepared: object
    output_dir: Path
    ratios: tuple
    studies: dict
    jobs: dict
    baseline_jobs: list
    plan: pd.DataFrame


def _ratio_study(prepared, ratio):
    rounds, memberships = {}, []
    for round_id, frames in prepared.rounds.items():
        if "synthetic_pool" not in frames:
            raise ValueError("Reprepare classifier folds with unique training-only synthetic pools.")
        synthetic, draws = bootstrap_synthetic(frames["synthetic_pool"], frames["train"],
                                                prepared.protocol, round_id, ratio)
        rounds[round_id] = {**frames, "synthetic": synthetic}
        memberships.extend(draws)
    return replace(prepared, rounds=rounds), memberships


def prepare_ratio_experiment(prepared, output_dir, *, ratios=(1, 10)):
    ratios = tuple(ratios)
    if not ratios or any(type(r) is not int or r < 1 for r in ratios) or len(set(ratios)) != len(ratios):
        raise ValueError("Ratios must be unique positive integers; the real-only baseline is added automatically.")
    ratios = tuple(sorted(ratios))
    if not prepared.protocol["classifier"]["include_update_matched_real_control"]:
        raise ValueError("Ratio experiments require update-matched real-only controls.")
    output = Path(output_dir)
    studies, jobs, memberships = {}, {}, []
    baseline = [entry for entry in training_jobs(prepared) if entry[0]["training_setting"] == "real-only"]
    plan_rows = [{"ratio_level": 0, **job} for job, *_ in baseline]
    for ratio in ratios:
        studies[ratio], selected = _ratio_study(prepared, ratio)
        memberships.extend(selected)
        jobs[ratio] = [entry for entry in training_jobs(studies[ratio])
                       if entry[0]["training_setting"] in {"synthetic-train-augment", "real-only-update-matched"}]
        plan_rows.extend({"ratio_level": ratio, **job} for job, *_ in jobs[ratio])
    manifest = {"schema_version": 1, "ratios": list(ratios), "real_only_baseline_ratio": 0,
        "real_counts_fixed": True, "real_clinical_proportions_preserved": True, "splits_and_classifier_seeds_shared": True,
        "synthetic_selection": "training-only age/group-stratified bundle bootstrap with replacement; nested draw prefixes shared across generators",
        "training_effort": "epoch-derived updates, with a shared real-only update-matched control at every ratio",
        "ratio_selection": "prespecified reporting; no best-ratio selection using ENNI test or Gillam",
        "effective_protocol": prepared.protocol, "real_sha256": frame_digest(prepared.real),
        "synthetic_sha256": frame_digest(prepared.synthetic), "external_sha256": frame_digest(prepared.external),
        "planned_fits": len(plan_rows), "classifier_run_ids": [r["classifier_run_id"] for r in plan_rows]}
    path = output/"experiment_manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Existing ratio experiment differs; choose a fresh experiment directory.")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(path, manifest)
    pd.DataFrame(memberships).to_csv(output/"synthetic_membership.csv", index=False)
    plan = pd.DataFrame(plan_rows)
    plan.to_csv(output/"ratio_training_plan.csv", index=False)
    write_training_plan(prepared, output/"baseline", jobs=baseline)
    for ratio in ratios:
        write_training_plan(studies[ratio], output/("ratio_"+str(ratio)), jobs=jobs[ratio])
    return RatioExperiment(prepared, output, ratios, studies, jobs, baseline, plan)


def run_ratio_experiment(experiment, *, backend=None, reuse_from=()):
    # Recheck immutable plans if the caller has edited inputs since the plan cell.
    experiment = prepare_ratio_experiment(experiment.prepared, experiment.output_dir, ratios=experiment.ratios)
    output = experiment.output_dir
    baseline = run_classifiers(experiment.prepared, output/"baseline", backend=backend,
                               jobs=experiment.baseline_jobs, reuse_from=reuse_from)
    parts = [baseline.assign(ratio_level=0)]
    for ratio in experiment.ratios:
        predicted = run_classifiers(experiment.studies[ratio], output/("ratio_"+str(ratio)), backend=backend,
                                    jobs=experiment.jobs[ratio], reuse_from=reuse_from)
        parts.append(predicted.assign(ratio_level=ratio))
    result = pd.concat(parts, ignore_index=True)
    result.to_csv(output/"all_ratio_story_predictions.csv", index=False)
    return result


def summarize_ratio_experiment(stories, protocol, output_dir):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    baseline = stories.loc[stories["ratio_level"].eq(0)].drop(columns="ratio_level")
    if baseline.empty or not baseline["training_setting"].eq("real-only").all():
        raise ValueError("Ratio summaries require the shared real-only baseline at ratio zero.")
    pieces = {key: [] for key in ("by_run", "summary", "paired_differences", "age_class_recall")}
    base_result = summarize_predictions(baseline, protocol, output/"baseline")
    for key in pieces:
        if not base_result[key].empty:
            pieces[key].append(base_result[key].assign(ratio_level=0))
    levels = sorted(set(stories["ratio_level"])-{0})
    for ratio in levels:
        rows = stories.loc[stories["ratio_level"].eq(ratio)].drop(columns="ratio_level")
        result = summarize_predictions(pd.concat([baseline, rows], ignore_index=True), protocol,
                                       output/("ratio_"+str(ratio)))
        for key in pieces:
            kept = result[key].loc[~result[key]["training_setting"].eq("real-only")]
            pieces[key].append(kept.assign(ratio_level=ratio))
    direct = []
    for generator in protocol["generator_keys"]:
        for lower, higher in combinations(levels, 2):
            selected = stories.loc[stories["generator_key"].eq(generator) & stories["training_setting"].eq("synthetic-train-augment")]
            left = selected.loc[selected["ratio_level"].eq(higher)].copy()
            right = selected.loc[selected["ratio_level"].eq(lower)].copy()
            right["training_setting"] = "real-only"; right["generator_key"] = "shared"
            units = prediction_units(pd.concat([left, right], ignore_index=True))
            contrast = paired_intervals(units, protocol["uncertainty"], protocol["classifier"]["decision_threshold_sli"])
            contrast["reference_setting"] = "synthetic-train-augment"
            direct.append(contrast.assign(lower_ratio=lower, higher_ratio=higher,
                                           difference_direction="higher_ratio_minus_lower_ratio"))
    result = {key: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame() for key, parts in pieces.items()}
    result["ratio_differences"] = pd.concat(direct, ignore_index=True) if direct else pd.DataFrame()
    for key, frame in result.items():
        frame.to_csv(output/(key+".csv"), index=False)
    return result
