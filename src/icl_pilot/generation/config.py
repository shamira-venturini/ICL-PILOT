"""Portable, frozen-roster setup for the Colab generation notebook.

This module prepares generation jobs and split bookkeeping, not analyses.
It never chooses new child pairs or rewrites experiment inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import pandas as pd

from .prompts import PromptBuilder, load_prompt_builder
from .workflow import (
    MultiAgeRotatingGenerationArtifacts, RotatingGenerationPackageArtifacts,
    build_bundle_feature_template, build_bundle_index, build_rotating_eval_manifest,
    build_rotating_generation_schedule, build_rotating_prompt_rounds, load_frozen_roster,
)


def _relative_input(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Configuration input paths must be relative and stay inside their input root.")
    resolved = root / path
    if not resolved.is_file():
        raise FileNotFoundError(f"Required generation input is missing: {resolved}")
    return resolved


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_experiment_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 4:
        raise ValueError("Unsupported generation configuration schema.")
    for name in ("n_replicates", "pairs_per_round"):
        if type(config.get(name)) is not int or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if type(config.get("seed")) is not int:
        raise ValueError("seed must be an integer.")
    cohorts = config.get("cohorts", [])
    if not cohorts or len({row["label"] for row in cohorts}) != len(cohorts):
        raise ValueError("Cohort labels must be nonempty and unique.")
    if len({row["slug"] for row in cohorts}) != len(cohorts):
        raise ValueError("Cohort output slugs must be unique.")
    if not config.get("roster_csv") or any("roster_csv" in row for row in cohorts):
        raise ValueError("Specify one master roster_csv at the top level, with no cohort-specific roster paths.")
    if not config.get("story_reference_json") or "story_reference_json" in config.get("inputs", {}):
        raise ValueError("Specify story_reference_json at the top level, relative to the configuration directory.")
    if set(config.get("prompt_templates", {})) != {"stage1_td", "stage2_sli"}:
        raise ValueError("Specify prompt_templates for both generation stages.")
    model = config["model"]
    if not model.get("id") or type(model.get("do_sample")) is not bool:
        raise ValueError("Specify a model ID and a boolean do_sample setting.")
    if model["temperature"] <= 0 or not 0 < model["top_p"] <= 1:
        raise ValueError("Invalid decoding temperature or top_p.")
    for name in ("max_new_tokens_stage1", "max_new_tokens_stage2"):
        if type(model[name]) is not int or model[name] < 1:
            raise ValueError(f"{name} must be a positive integer.")
    quantization = model["quantization"]
    if type(quantization["load_in_4bit"]) is not bool or quantization["quant_type"] not in {"nf4", "fp4"}:
        raise ValueError("Invalid quantization configuration.")
    if quantization["compute_dtype"] not in {"bfloat16", "float16", "float32"}:
        raise ValueError("Unsupported quantization compute dtype.")
    return config


def validate_prompt_evaluation_separation(schedule: pd.DataFrame, evaluation: pd.DataFrame) -> None:
    """Reject any child reused for evaluation within its age-specific prompt round."""
    required_schedule = {"cohort", "round_id", "td_child_id", "sli_child_id"}
    required_eval = {"cohort", "round_id", "eval_child_id"}
    if not required_schedule.issubset(schedule) or not required_eval.issubset(evaluation):
        raise ValueError("Split checks require cohort, round, and child identity columns.")
    for (cohort, round_id), rows in schedule.groupby(["cohort", "round_id"], sort=False):
        prompts = set(rows["td_child_id"].astype(int)) | set(rows["sli_child_id"].astype(int))
        held = evaluation.loc[(evaluation["cohort"] == cohort) & (evaluation["round_id"] == round_id)]
        if held.empty:
            raise ValueError(f"No evaluation children for {cohort}/{round_id}.")
        overlap = prompts & set(held["eval_child_id"].astype(int))
        if overlap:
            raise ValueError(f"Prompt/evaluation overlap in {cohort}/{round_id}: {sorted(overlap)}")


def compile_experiment_design(
    config_path: str | Path, *, n_replicates: int | None = None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
    config_path = Path(config_path).resolve()
    config = load_experiment_config(config_path)
    if n_replicates is not None:
        if type(n_replicates) is not int or n_replicates < 1:
            raise ValueError("n_replicates override must be a positive integer.")
        config["n_replicates"] = n_replicates
    counterbalance = _relative_input(config_path.parent, config["counterbalance_csv"])
    rules = pd.read_csv(counterbalance)
    if set(rules["target_story"]) != {"A1", "A2", "A3", "B1", "B2", "B3"}:
        raise ValueError("Counterbalancing must cover all six ENNI target stories.")
    if any(row.target_story[0] == row.e1_story[0] or row.target_story[0] == row.e2_story[0]
           or row.e1_story == row.e2_story for row in rules.itertuples(index=False)):
        raise ValueError("Counterbalancing requires distinct exemplars from the opposite story set.")
    schedules, evaluations, rosters = [], [], {}
    master_path = _relative_input(config_path.parent, config["roster_csv"])
    master = pd.read_csv(master_path)
    if "cohort" not in master or set(master["cohort"]) != {row["label"] for row in config["cohorts"]}:
        raise ValueError("Master roster cohort labels must match the configured cohorts exactly.")
    seen_children: set[int] = set()
    for cohort in config["cohorts"]:
        subset = master.loc[master["cohort"] == cohort["label"]].copy()
        roster = load_frozen_roster(subset)
        if len(roster) != cohort["expected_pairs"]:
            raise ValueError(f"Unexpected frozen pair count for {cohort['label']}.")
        ids = roster["sli_child_id"].astype(int).tolist() + roster["td_child_id"].astype(int).tolist()
        if len(set(ids)) != len(ids) or seen_children.intersection(ids):
            raise ValueError("Frozen child IDs must be unique within and across cohorts.")
        seen_children.update(ids)
        if not roster["cohort"].eq(cohort["label"]).all() or not roster["pair_id"].is_unique:
            raise ValueError("Frozen cohort labels and pair IDs must be consistent.")
        if roster["pair_order"].astype(int).tolist() != list(range(1, len(roster) + 1)):
            raise ValueError("Frozen pair orders must be consecutive and unique.")
        for row in roster.itertuples(index=False):
            if row.pair_id != f"pair_sli{int(row.sli_child_id)}_td{int(row.td_child_id)}":
                raise ValueError("Pair ID disagrees with the recorded child identities.")
        for column in ("sli_age", "td_age"):
            for age in roster[column]:
                years, month_days = str(age).split(";")
                months = 12 * int(years) + int(month_days.split(".")[0])
                if not cohort["age_min_months"] <= months <= cohort["age_max_months"]:
                    raise ValueError(f"Age outside configured cohort: {age}")
        schedule = build_rotating_generation_schedule(roster, counterbalance, config["n_replicates"], config["pairs_per_round"])
        evaluation = build_rotating_eval_manifest(roster, config["pairs_per_round"])
        evaluation["cohort"] = cohort["label"]
        schedules.append(schedule)
        evaluations.append(evaluation)
        rosters[cohort["label"]] = subset.sort_values("pair_order").reset_index(drop=True)
    schedule = pd.concat(schedules, ignore_index=True)
    evaluation = pd.concat(evaluations, ignore_index=True)
    validate_prompt_evaluation_separation(schedule, evaluation)
    return config, schedule, evaluation, rosters


@dataclass(frozen=True)
class PreparedColabGeneration:
    config: dict
    run_df: pd.DataFrame
    eval_manifest: pd.DataFrame
    artifacts: MultiAgeRotatingGenerationArtifacts
    run_manifest_path: Path
    prompt_builder: PromptBuilder


def prepare_colab_generation(
    config_path: str | Path, data_root: str | Path, output_dir: str | Path,
    *, n_replicates: int | None = None, smoke_test: bool = False,
) -> PreparedColabGeneration:
    if type(smoke_test) is not bool:
        raise ValueError("smoke_test must be a boolean.")
    config_path = Path(config_path).resolve()
    config, schedule, evaluation, rosters = compile_experiment_design(config_path, n_replicates=n_replicates)
    prompt_builder = load_prompt_builder(config_path, config)
    data_root, output = Path(data_root).expanduser().resolve(), Path(output_dir).expanduser().resolve()
    narratives_path = _relative_input(data_root, config["inputs"]["narratives_csv"])
    reference_path = _relative_input(config_path.parent, config["story_reference_json"])
    narratives = pd.read_csv(narratives_path, dtype={"File_ID": str, "Group": str, "story_id": str}, keep_default_na=False)
    if narratives.duplicated(["File_ID", "Group", "story_id"]).any():
        raise ValueError("Duplicate child/group/story exemplar keys.")
    if "Age_Raw" not in narratives:
        raise ValueError("Exemplar metadata must include Age_Raw for age-matching checks.")
    lookup = {(r.File_ID, r.Group, r.story_id): r for r in narratives.itertuples(index=False)}
    reference = json.loads(reference_path.read_text())
    run = schedule.copy()
    run["cohort_label"] = run["cohort"]
    for column, child, group, slot in (("prompt_td_exemplar_story", "td_child_id", "TD", "prompt_td_story_e1"),
                                       ("prompt_sli_exemplar_story", "sli_child_id", "SLI", "prompt_sli_story_e2")):
        texts = []
        for row in run.itertuples(index=False):
            key = (str(int(getattr(row, child))), group, getattr(row, slot))
            exemplar = lookup.get(key)
            text = exemplar.story_text if exemplar is not None else None
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Missing or empty prompt exemplar: {key}")
            expected_age = row.td_age if group == "TD" else row.sli_age
            if exemplar.Age_Raw != expected_age:
                raise ValueError(f"Exemplar age disagrees with the frozen roster: {key}")
            texts.append(text)
        run[column] = texts
    for story in run["target_story"].unique():
        if story not in reference or not reference[story].get("units"):
            raise ValueError(f"Missing canonical story scaffold: {story}")
    run["story_title"] = run["target_story"].map(lambda s: reference[s]["title"])
    run["story_units_text"] = run["target_story"].map(lambda s: "\n".join(f"{i + 1}. {text}" for i, text in enumerate(reference[s]["units"])))
    for column, key in (("content_invariants_text", "content_invariants"), ("optional_content_text", "optional_content"), ("prompt_constraints_text", "prompt_constraints")):
        run[column] = run["target_story"].map(lambda s: "\n".join(f"- {text}" for text in reference[s].get(key, [])) or "None")

    # Stage 2 content is supplied only after inference. A marker checks template
    # rendering here; it is never saved as a model output or generation prompt.
    for row in run.itertuples(index=False):
        prompt_builder.stage1_messages(row)
        prompt_builder.stage2_messages(row, "Preflight parent marker; not a generated narrative.")

    # Preserve the complete validated study design; record the execution subset
    # separately so a one-story integration check cannot masquerade as a full run.
    execution_run = run.iloc[:1].copy() if smoke_test else run
    execution = {"mode": "smoke_test" if smoke_test else "study", "test_only": smoke_test,
                 "planned_jobs": len(run), "selected_jobs": len(execution_run),
                 "selection_csv": "generation_design/execution_schedule.csv"}

    notebook = config_path.parents[2] / "notebooks/colab_direct_generation_pipeline.ipynb"
    input_hashes = {"config": _sha(config_path), "counterbalance": _sha(config_path.parent / config["counterbalance_csv"]),
                    "execution_selection": hashlib.sha256(json.dumps(execution, sort_keys=True).encode("utf-8")).hexdigest(),
                    "effective_config": hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest(),
                    "notebook": _sha(notebook) if notebook.is_file() else None,
                    "setup_code": _sha(Path(__file__)), "workflow_code": _sha(Path(__file__).with_name("workflow.py")),
                    "prompt_builder_code": _sha(Path(__file__).with_name("prompts.py")),
                    "prompt_recording_code": _sha(Path(__file__).with_name("prompt_records.py")),
                    "execution_code": _sha(Path(__file__).with_name("execution.py")),
                    "prompt_templates": prompt_builder.template_hashes,
                    "narratives": _sha(narratives_path), "story_reference": _sha(reference_path),
                    "roster": _sha(config_path.parent / config["roster_csv"])}
    manifest_path = output / "run_manifest.json"
    if output.exists() and any(output.glob("*story_generations.csv")):
        raise FileExistsError("Existing generation outputs found. Choose a fresh run directory.")
    if (output / "prompt_records.jsonl").exists():
        raise FileExistsError("Existing prompt records found. Choose a fresh run directory.")
    if manifest_path.exists() and json.loads(manifest_path.read_text())["input_sha256"] != input_hashes:
        raise ValueError("Run inputs changed. Choose a fresh run directory.")
    output.mkdir(parents=True, exist_ok=True)
    design = output / "generation_design"
    prompt_builder.snapshot(design / "prompts")
    artifacts, summaries = {}, []
    for cohort in config["cohorts"]:
        label, slug = cohort["label"], cohort["slug"]
        folder = design / slug
        folder.mkdir(parents=True, exist_ok=True)
        roster_path = folder / f"{slug}_frozen_roster.csv"
        rosters[label].to_csv(roster_path, index=False)
        rounds_path = folder / f"{slug}_prompt_rounds.csv"
        build_rotating_prompt_rounds(roster_path, config["pairs_per_round"], rounds_path)
        schedule_path = folder / f"{slug}_rotating_generation_schedule.csv"
        part = schedule.loc[schedule["cohort"] == label].copy()
        part.to_csv(schedule_path, index=False)
        eval_path = folder / f"{slug}_rotating_eval_manifest.csv"
        evaluation.loc[evaluation["cohort"] == label].to_csv(eval_path, index=False)
        bundle_path = folder / f"{slug}_rotating_bundle_index.csv"
        bundles = build_bundle_index(part, bundle_path)
        template_path = folder / f"{slug}_rotating_synthetic_bundle_template.csv"
        build_bundle_feature_template(bundles, template_path)
        summary_path = folder / f"{slug}_rotating_generation_package_summary.json"
        summary = {"cohort": label, "pairs": cohort["expected_pairs"], "generation_jobs": len(part),
                   "n_replicates": config["n_replicates"], "pairs_per_round": config["pairs_per_round"]}
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        summaries.append(summary)
        artifacts[label] = RotatingGenerationPackageArtifacts(folder, roster_path, rounds_path, schedule_path, bundle_path, template_path, eval_path, summary_path)
    summary_path = design / "multi_age_rotating_generation_packages_summary.json"
    summary_path.write_text(json.dumps({"cohorts": summaries, "generation_jobs": len(run)}, indent=2) + "\n")
    selected_columns = ["cohort", "round_id", "pair_id", "bundle_id", "replicate_id", "target_story",
                        "story_slot_order", "td_child_id", "sli_child_id", "td_age", "sli_age",
                        "prompt_td_story_e1", "prompt_sli_story_e2"]
    execution_run[selected_columns].to_csv(output / execution["selection_csv"], index=False)
    manifest = {"schema_version": 1, "config": config, "input_sha256": input_hashes,
                "execution": execution, "generation_jobs": len(execution_run),
                "prompt_records": {"path": "prompt_records.jsonl", "schema_version": 1, "written_during_inference": True},
                "split_validation": "passed", "status": "generation_setup_prepared; model not yet loaded"}
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return PreparedColabGeneration(config, execution_run, evaluation, MultiAgeRotatingGenerationArtifacts(output, artifacts, summary_path), manifest_path, prompt_builder)
