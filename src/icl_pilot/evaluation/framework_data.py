"""Generator-aware, child-separated inputs for the revised framework evaluation.

All data-dependent fitting remains in training. Generic corpus descriptions use
the full ENNI reference and are explicitly not held-out fidelity estimates.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .framework_text import TEXT_POLICY

LABELS = {"SLI": 0, "TD": 1}
MODEL_FIELDS = ("model_key", "model_id", "model_revision", "run_id")
PARENT_FIELDS = ("raw_story_text_sha256", "stage1_text_sha256", "stage2_parent_text_sha256",
                 "stage1_prompt_record_id", "stage2_prompt_record_id")
JOB_FIELDS = ("cohort_year", "round_id", "pair_id", "replicate_id", "story_id")
SCHEDULE_FIELDS = JOB_FIELDS + ("stage",)


def child_id(value):
    """Undo CSV float promotion of integer IDs without inventing new identities."""
    text = str(value)
    return text[:-2] if re.fullmatch(r"\d+\.0", text) else text


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def frame_digest(frame):
    """Hash ordered identities and texts, preserving training order in provenance."""
    return digest(frame.astype(object).where(frame.notna(), "").to_dict(orient="records"))


def load_protocol(root, path=None):
    root = Path(root).resolve()
    path = Path(path) if path else root / "configs/evaluation/protocol.json"
    path = path if path.is_absolute() else root / path
    protocol = json.loads(path.read_text())
    protocol.pop("generator_identities", None)
    protocol.pop("protocol_sha256", None)
    if protocol.get("schema_version") != 2:
        raise ValueError("Unsupported framework protocol schema.")
    for field in ("generator_keys", "cohorts", "story_ids"):
        values = protocol.get(field, [])
        if not values or len(values) != len(set(values)):
            raise ValueError("Protocol requires unique, nonempty " + field)
    if not set(protocol["cohorts"]) <= {4, 5, 6, 7}:
        raise ValueError("Framework cohorts must be within ages four through seven.")
    repeats = protocol.get("synthetic_to_real_ratio")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("synthetic_to_real_ratio must be a positive integer training-draw ratio.")
    splits = protocol.get("classification_split", {})
    if type(splits.get("n_splits")) is not int or splits["n_splits"] < 3 or type(splits.get("seed")) is not int or splits.get("validation") != "next_fold":
        raise ValueError("Configure at least three classifier folds, an integer split seed and next_fold validation.")
    sampling = protocol.get("synthetic_sampling", {})
    if sampling != {"method": "stratified_bundle_bootstrap", "unit": "complete_story_bundle", "replacement": True,
                    "match_real_strata": ["cohort_year", "group", "story_id"]}:
        raise ValueError("Use the explicit training-only, real-distribution-matched bundle bootstrap.")
    classifier = protocol["classifier"]
    for name in ("perplexity_revision", "embedding_revision"):
        if not re.fullmatch(r"[0-9a-f]{40}", protocol["text_quality"][name]):
            raise ValueError("Pin text-quality evaluator checkpoints to full revisions.")
    if not re.fullmatch(r"[0-9a-f]{40}", classifier["revision"]):
        raise ValueError("Pin the classifier checkpoint to a full revision.")
    seeds = classifier["seeds"]
    if not seeds or len(seeds) != len(set(seeds)) or any(type(x) is not int for x in seeds):
        raise ValueError("Classifier seeds must be unique integers.")
    for field in ("max_length", "batch_size", "eval_batch_size", "gradient_accumulation_steps"):
        if type(classifier[field]) is not int or classifier[field] < 1:
            raise ValueError("Invalid classifier " + field)
    if classifier["max_length"] != 512:
        raise ValueError("The pinned SynthTextEval tokenizer uses 512 tokens including special tokens.")
    if classifier["epochs"] <= 0 or classifier["learning_rate"] <= 0:
        raise ValueError("Epochs and learning rate must be positive.")
    if classifier["weight_decay"] < 0 or not 0 <= classifier["warmup_ratio"] <= 1:
        raise ValueError("Invalid optimizer decay or warmup settings.")
    for flag in ("retain_model_checkpoints", "include_update_matched_real_control"):
        if type(classifier[flag]) is not bool:
            raise ValueError("Classifier control flags must be booleans.")
    if classifier["class_weighting"] != "none":
        raise ValueError("This protocol uses unweighted cross entropy; no implicit weighting.")
    if not 0 < classifier["decision_threshold_sli"] < 1:
        raise ValueError("Invalid fixed decision threshold.")
    uncertainty = protocol["uncertainty"]
    if type(uncertainty["bootstrap_samples"]) is not int or uncertainty["bootstrap_samples"] < 2 or not 0 < uncertainty["confidence"] < 1:
        raise ValueError("Invalid bootstrap settings.")
    if not uncertainty.get("units") or not set(uncertainty["units"]) <= {"story", "child"}:
        raise ValueError("Invalid uncertainty units.")
    catalog_path = Path(protocol["generator_catalog"])
    catalog_path = catalog_path if catalog_path.is_absolute() else root / catalog_path
    catalog = json.loads(catalog_path.read_text())["models"]
    if not set(protocol["generator_keys"]) <= set(catalog):
        raise ValueError("Unknown generator key in evaluation protocol.")
    protocol["generator_identities"] = {key: catalog[key] for key in protocol["generator_keys"]}
    protocol["protocol_sha256"] = digest(protocol)
    return protocol


def require(frame, fields, name):
    missing = set(fields) - set(frame.columns)
    if missing:
        raise ValueError(name + " is missing columns: " + ", ".join(sorted(missing)))
    if frame[list(fields)].isna().any().any():
        raise ValueError(name + " has missing required values.")


def _identity(frame, dataset):
    frame = frame.copy()
    frame["evaluation_dataset"] = dataset
    frame["source_child_id"] = frame["source_child_id"].map(child_id)
    frame["child_key"] = dataset + ":" + frame["source_child_id"]
    frame["Label"] = frame["group"].map(LABELS)
    if frame["Label"].isna().any():
        raise ValueError("Unknown corpus group; literal TD/SLI labels are required.")
    frame["Label"] = frame["Label"].astype(int)
    frame["output_text"] = frame["story_text"].astype(str)
    if frame["output_text"].str.strip().eq("").any():
        raise ValueError("Empty story text is not an evaluation sample.")
    frame["sample_id"] = [
        dataset + ":" + digest([child, story])[:24]
        for child, story in zip(frame["source_child_id"], frame["story_id"])
    ]
    return frame


@dataclass
class PreparedEvaluation:
    protocol: dict
    real: pd.DataFrame
    synthetic: pd.DataFrame
    external: pd.DataFrame
    rounds: dict
    samples: dict
    output_dir: Path


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n")


def prepare_evaluation(records, external, protocol, output_dir):
    """Validate lineage and build common real splits and model-specific descendants."""
    if protocol.get("schema_version") != 2:
        raise ValueError("Reprepare inputs with the schema-2 all-child classification protocol.")
    records = records.copy()
    external = external.copy()
    require(records, ("source", "group", "cohort_year", "source_child_id", "story_id", "story_text", "text_normalization"), "ENNI")
    require(external, ("source", "group", "cohort_year", "source_child_id", "story_id", "story_text", "text_normalization"), "Gillam")
    if not records["text_normalization"].eq(TEXT_POLICY).all() or not external["text_normalization"].eq(TEXT_POLICY).all():
        raise ValueError("Re-export all sources with the common framework text policy; legacy normalization is not silently mixed.")
    if not records["source"].isin(["real", "synthetic"]).all() or not external["source"].eq("real_external").all():
        raise ValueError("Gillam must remain a separate test-only dataset.")
    records = records.loc[records["cohort_year"].isin(protocol["cohorts"])].copy()
    real = _identity(records.loc[records["source"].eq("real")], "enni")
    synthetic = records.loc[records["source"].eq("synthetic")].copy()
    external = _identity(external, "gillam")
    for frame in (real, external):
        if frame["sample_id"].duplicated().any():
            raise ValueError("Duplicate real child/story identities.")
        if frame.groupby("child_key")[["group", "cohort_year"]].nunique().gt(1).any().any():
            raise ValueError("Inconsistent real child group or age cohort.")
    require(synthetic, MODEL_FIELDS + PARENT_FIELDS + SCHEDULE_FIELDS + ("td_child_id", "sli_child_id"), "Synthetic export")
    for field in MODEL_FIELDS:
        if synthetic[field].astype(str).str.strip().eq("").any():
            raise ValueError("Missing generator identity: " + field)
    for field in ("raw_story_text_sha256", "stage1_text_sha256", "stage2_parent_text_sha256"):
        if not synthetic[field].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
            raise ValueError("Invalid recorded text hash: " + field)
    if "test_only" in synthetic and synthetic["test_only"].astype(str).str.lower().isin(["true", "1"]).any():
        raise ValueError("Software-fixture generation cannot enter scientific evaluation.")
    if set(synthetic["model_key"]) != set(protocol["generator_keys"]):
        raise ValueError("Supply exactly the configured generator models; no silent pooling or missing models.")
    if set(synthetic["story_id"]) != set(protocol["story_ids"]):
        raise ValueError("Synthetic target stories disagree with the evaluation protocol.")
    synthetic["replicate_id"] = pd.to_numeric(synthetic["replicate_id"], errors="raise")
    if (synthetic["replicate_id"] < 1).any() or (synthetic["replicate_id"] % 1 != 0).any():
        raise ValueError("Replicate IDs must be positive integers.")
    synthetic["replicate_id"] = synthetic["replicate_id"].astype(int)
    for key, identities in protocol["generator_identities"].items():
        rows = synthetic.loc[synthetic["model_key"].eq(key)]
        if not rows["model_id"].eq(identities["id"]).all() or not rows["model_revision"].eq(identities["revision"]).all():
            raise ValueError("Generator checkpoint differs from the pinned catalog: " + key)
        if rows["run_id"].nunique() != 1:
            raise ValueError("Select one generation run per generator; multiple runs require an explicit design.")
    if synthetic.duplicated(["model_key"] + list(SCHEDULE_FIELDS)).any():
        raise ValueError("Duplicate generator/job/stage identities.")
    synthetic["_job"] = list(map(tuple, synthetic[list(JOB_FIELDS)].to_numpy()))
    for (_model, _job), rows in synthetic.groupby(["model_key", "_job"]):
        if set(rows["stage"]) != {"stage1_td", "stage2_sli"} or len(rows) != 2:
            raise ValueError("Each synthetic job requires its own Stage 1 and Stage 2.")
        first = rows.loc[rows["stage"].eq("stage1_td")].iloc[0]
        if not rows["stage1_text_sha256"].eq(first["raw_story_text_sha256"]).all() or not rows["stage2_parent_text_sha256"].eq(first["raw_story_text_sha256"]).all():
            raise ValueError("Stage 2 parent hash disagrees with its own generator's Stage 1.")
        if any(rows[field].nunique() != 1 for field in ("stage1_prompt_record_id", "stage2_prompt_record_id")):
            raise ValueError("Ambiguous paired stage prompt-record identities.")
    stage_group = synthetic["stage"].map({"stage1_td": "TD", "stage2_sli": "SLI"})
    if not synthetic["group"].eq(stage_group).all():
        raise ValueError("Synthetic stage/group labels disagree.")
    for field in ("td_child_id", "sli_child_id"):
        synthetic[field] = synthetic[field].map(child_id)
    synthetic["source_child_id"] = synthetic["source_child_id"].map(child_id)
    expected_child = np.where(synthetic["group"].eq("TD"), synthetic["td_child_id"], synthetic["sli_child_id"])
    if not synthetic["source_child_id"].eq(expected_child).all():
        raise ValueError("Synthetic prompt-child identity disagrees with stage.")
    for _key, rows in synthetic.groupby(["model_key", "cohort_year", "pair_id", "replicate_id", "stage"]):
        if set(rows["story_id"]) != set(protocol["story_ids"]):
            raise ValueError("A synthetic replicate has an incomplete target-story set.")
    reference_schedule = None
    for key in protocol["generator_keys"]:
        rows = synthetic.loc[synthetic["model_key"].eq(key)]
        schedule = set(map(tuple, rows[list(SCHEDULE_FIELDS)].to_numpy()))
        if reference_schedule is not None and schedule != reference_schedule:
            raise ValueError("Generator schedules differ. Resolve missing jobs before comparing models.")
        reference_schedule = schedule
    synthetic["sample_id"] = [
        "synthetic:" + digest([row.model_key, row.run_id] + [getattr(row, f) for f in SCHEDULE_FIELDS])[:24]
        for row in synthetic.itertuples()
    ]
    synthetic["output_text"] = synthetic["story_text"].astype(str)
    synthetic["Label"] = synthetic["group"].map(LABELS).astype(int)
    if synthetic["output_text"].str.strip().eq("").any():
        raise ValueError("Synthetic text is empty.")
    # Identical source pairs/rounds across generators define shared real partitions.
    pair_fields = ["cohort_year", "pair_id", "round_id", "td_child_id", "sli_child_id"]
    pairs = synthetic[pair_fields].drop_duplicates()
    if pairs.duplicated(["cohort_year", "pair_id"]).any():
        raise ValueError("Ambiguous pair membership or generation-round mapping.")
    pair_children = []
    for row in pairs.itertuples():
        for group, field in (("TD", "td_child_id"), ("SLI", "sli_child_id")):
            pair_children.append({"cohort_year": row.cohort_year, "pair_id": row.pair_id,
                                  "round_id": row.round_id, "group": group,
                                  "source_child_id": getattr(row, field)})
    children = pd.DataFrame(pair_children)
    if children.duplicated(["cohort_year", "group", "source_child_id"]).any():
        raise ValueError("A real child belongs to more than one prompt pair.")
    from .framework_splits import assign_classifier_folds, classifier_partitions
    from .framework_sampling import bootstrap_synthetic

    real = assign_classifier_folds(real, children, protocol)
    paired_real = real.loc[real["is_prompt_child"]]
    rounds, split_rows, selection_rows, bootstrap_rows = {}, [], [], []
    for round_id, frames in classifier_partitions(real, protocol):
        train_pairs = set(map(tuple, frames["train"].loc[frames["train"]["is_prompt_child"], ["cohort_year", "pair_id"]].to_numpy()))
        mask = pd.Series([(age, pair) in train_pairs for age, pair in zip(synthetic["cohort_year"], synthetic["pair_id"])], index=synthetic.index)
        pools = {key: synthetic.loc[mask & synthetic["model_key"].eq(key)].drop(columns=["_job"]).sort_values(list(SCHEDULE_FIELDS)).reset_index(drop=True)
                 for key in protocol["generator_keys"]}
        model_frames, draws = bootstrap_synthetic(pools, frames["train"], protocol, round_id, protocol["synthetic_to_real_ratio"])
        bootstrap_rows.extend(draws)
        for name, frame in list(frames.items()) + [("synthetic/"+k, v) for k, v in model_frames.items()]:
            if set(frame["group"]) != set(LABELS):
                raise ValueError("Both clinical groups are required in each classifier partition.")
            split_rows.append({"round_id": round_id, "split": name, "status": "prepared",
                "rows": len(frame), "unique_prompt_children_or_real_children": frame["source_child_id"].nunique(),
                "unique_story_records": frame["sample_id"].nunique(),
                "n_TD": int(frame["group"].eq("TD").sum()), "n_SLI": int(frame["group"].eq("SLI").sum()),
                "train_validation_child_overlap": 0, "train_test_child_overlap": 0,
                "validation_test_child_overlap": 0, "training_prompt_test_child_overlap": 0})
            if name in frames:
                selection_rows.extend({"round_id": round_id, "split": name, "sample_id": sid} for sid in frame["sample_id"])
        rounds[round_id] = {**frames, "synthetic_pool": pools, "synthetic": model_frames}
    if set(external["group"]) != set(LABELS):
        raise ValueError("Both literal clinical groups are required in the external test sample.")
    output = Path(output_dir)
    if (output / "effective_protocol.json").exists():
        previous = json.loads((output / "effective_protocol.json").read_text())
        if previous != protocol:
            raise ValueError("Existing evaluation protocol differs; use a fresh evaluation run.")
    if (output / "input_audit.json").exists():
        previous = json.loads((output / "input_audit.json").read_text())
        if previous["records_sha256"] != frame_digest(records) or previous["external_sha256"] != frame_digest(external):
            raise ValueError("Existing evaluation inputs differ; use a fresh evaluation run.")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "effective_protocol.json", protocol)
    pd.DataFrame(split_rows).to_csv(output / "split_audit.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(output / "split_membership.csv", index=False)
    for name, frame in (("enni", real), ("gillam", external)):
        coverage = frame.groupby(["child_key", "group", "cohort_year"], as_index=False).agg(
            n_stories=("story_id", "size"), stories=("story_id", lambda x: "|".join(sorted(x))))
        coverage.to_csv(output / (name + "_story_coverage_by_child.csv"), index=False)
    unmatched = real.loc[~real["is_prompt_child"]]
    unmatched.to_csv(output / "unmatched_TD_included_in_classification.csv", index=False)
    fold_fields = ["source_child_id", "child_key", "group", "cohort_year", "pair_id", "split_group_id", "is_prompt_child", "heldout_fold"]
    real[fold_fields].drop_duplicates().to_csv(output / "classification_child_folds.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(output / "synthetic_bootstrap_membership.csv", index=False)
    _write_json(output / "input_audit.json", {
        "real_children": int(real["child_key"].nunique()), "matched_real_children": int(paired_real["child_key"].nunique()),
        "unmatched_real_children": int(unmatched["child_key"].nunique()), "external_children": int(external["child_key"].nunique()),
        "external_use": "test_only", "classifier_splits": "all eligible real children; age-stratified grouped cross-validation independent of generation batches",
        "training_sampling": "synthetic bundle bootstrap matched to real training age/group/story counts; unweighted loss",
        "classifier_folds": protocol["classification_split"]["n_splits"],
        "unmatched_TD_use": "included as individual child groups in ordinary train/validation/test folds",
        "raw_id_overlap_between_corpora": sorted(set(real["source_child_id"]) & set(external["source_child_id"])),
        "identity_namespace": "corpus + child ID; equal local IDs across corpora are not evidence of participant identity",
        "records_sha256": frame_digest(records), "external_sha256": frame_digest(external),
    })
    samples = balanced_corpora(real, synthetic, protocol, output)
    return PreparedEvaluation(protocol, real, synthetic.drop(columns=["_job"]), external, rounds, samples, output)


def balanced_corpora(real, synthetic, protocol, output):
    """Same real rows and age/group/story composition for every generator."""
    refs, selected = [], {key: [] for key in protocol["generator_keys"]}
    for stratum, reference in real.groupby(["cohort_year", "group", "story_id"], sort=True):
        candidates = {}
        for key in selected:
            mask = synthetic["model_key"].eq(key)
            for field, value in zip(("cohort_year", "group", "story_id"), stratum):
                mask &= synthetic[field].eq(value)
            candidates[key] = synthetic.loc[mask]
        n = min([len(reference)] + [len(x) for x in candidates.values()])
        if not n:
            raise ValueError("No common corpus-comparison support for stratum " + repr(stratum))
        seed = int(digest([protocol["sampling_seed"], list(stratum)])[:8], 16)
        refs.append(reference.sample(n=n, random_state=seed))
        for key in selected:
            selected[key].append(candidates[key].sample(n=n, random_state=seed))
    reference = pd.concat(refs, ignore_index=True)
    result = {}
    for key, parts in selected.items():
        generated = pd.concat(parts, ignore_index=True).drop(columns=["_job"], errors="ignore")
        target = output / "corpus_comparison" / key
        target.mkdir(parents=True, exist_ok=True)
        reference.to_csv(target / "real_sample.csv", index=False)
        generated.to_csv(target / "synthetic_sample.csv", index=False)
        comparison = pd.DataFrame({"source": generated["output_text"], "reference": reference["output_text"]})
        comparison.to_csv(target / "text_quality_input.csv", index=False)
        _write_json(target / "sampling_manifest.json", {
            "model_key": key, "scope": "descriptive full-corpus comparison; not held-out fidelity",
            "strata": ["age cohort", "literal corpus group", "target story"], "rows_per_source": len(reference),
            "semantic_row_pairing": False, "real_sample_sha256": frame_digest(reference),
            "synthetic_sample_sha256": frame_digest(generated),
        })
        result[key] = {"real": reference.copy(), "synthetic": generated, "comparison": comparison, "path": target}
    return result
