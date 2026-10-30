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

from .exemplars import EXEMPLAR_POLICY, validate_exemplar
from .prompts import PromptBuilder, load_prompt_builder
from .models import load_model_catalog, select_model
from .recovery import CHECKPOINT, atomic_json, execution_order, order_sha256, read_checkpoint
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


def load_experiment_config(path: str | Path, *, model_key: str | None = None) -> dict:
    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 7:
        raise ValueError("Unsupported generation configuration schema.")
    catalog_path = _relative_input(path.parent, config["models_file"])
    config["model"] = select_model(config["model"], load_model_catalog(catalog_path), model_key)
    for name in ("n_replicates", "pairs_per_round"):
        if type(config.get(name)) is not int or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if type(config.get("master_seed")) is not int or config["master_seed"] != 42 or config.get("seed_policy") != "run_level":
        raise ValueError("Specify master_seed=42 and seed_policy=run_level.")
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
    if config.get("exemplar_preparation", {}).get("policy") != EXEMPLAR_POLICY:
        raise ValueError(f"Specify exemplar_preparation policy {EXEMPLAR_POLICY}.")
    if config["exemplar_preparation"].get("performed_at") != "chat_to_csv_extraction":
        raise ValueError("Exemplar preparation must be performed during chat_to_csv_extraction.")
    model = config["model"]
    if not model.get("id") or type(model.get("do_sample")) is not bool:
        raise ValueError("Specify a model ID and a boolean do_sample setting.")
    if model["temperature"] <= 0 or not 0 < model["top_p"] <= 1:
        raise ValueError("Invalid decoding temperature or top_p.")
    if type(model.get("top_k")) is not int or model["top_k"] < 0:
        raise ValueError("top_k must be an explicit nonnegative integer; 0 disables top-k filtering.")
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
    config_path: str | Path, *, n_replicates: int | None = None, model_key: str | None = None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
    config_path = Path(config_path).resolve()
    config = load_experiment_config(config_path, model_key=model_key)
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
        schedule = build_rotating_generation_schedule(roster, counterbalance, config["n_replicates"], config["pairs_per_round"], master_seed=config["master_seed"])
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
    *, n_replicates: int | None = None, smoke_test: bool = False, smoke_test_jobs: int = 1,
    model_key: str | None = None, resume: bool = False,
) -> PreparedColabGeneration:
    if type(smoke_test) is not bool:
        raise ValueError("smoke_test must be a boolean.")
    if type(smoke_test_jobs) is not int or smoke_test_jobs < 1:
        raise ValueError("smoke_test_jobs must be a positive integer.")
    if not smoke_test and smoke_test_jobs != 1:
        raise ValueError("Set smoke_test=True to request smoke_test_jobs other than 1.")
    config_path = Path(config_path).resolve()
    config, schedule, evaluation, rosters = compile_experiment_design(
        config_path, n_replicates=n_replicates, model_key=model_key)
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
    prepared_exemplars, exemplar_inventory = {}, []
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
            if key not in prepared_exemplars:
                validated = validate_exemplar(text, getattr(exemplar, "story_units_json", None))
                prepared_exemplars[key] = validated.text
                exemplar_inventory.append({
                    "File_ID": key[0], "Group": key[1], "story_id": key[2], "Age_Raw": exemplar.Age_Raw,
                    "story_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "story_unit_count": len(validated.utterances),
                    "story_units_json": json.dumps(validated.utterances, ensure_ascii=True),
                    "story_text": text,
                })
            texts.append(prepared_exemplars[key])
        run[column] = texts
    exemplar_snapshot = pd.DataFrame(exemplar_inventory).to_csv(index=False).encode("utf-8")
    exemplar_preparation = {
        "policy": EXEMPLAR_POLICY, "scope": "full_validated_design",
        "performed_at": "chat_to_csv_extraction", "generation_action": "validate_and_select_only",
        "snapshot_csv": "generation_design/prepared_exemplars.csv",
        "snapshot_sha256": hashlib.sha256(exemplar_snapshot).hexdigest(),
        "unique_exemplars": len(exemplar_inventory),
        "unintelligible_replacement": "deletion_only; no inferred words or replacement tokens",
    }
    extraction_qc_path = narratives_path.with_suffix(".parsing_qc.json")
    extraction_qc_hash = None
    if extraction_qc_path.is_file():
        extraction_qc_hash = _sha(extraction_qc_path)
        extraction_qc = json.loads(extraction_qc_path.read_text())
        if (extraction_qc.get("output_csv_sha256") != _sha(narratives_path)
                or extraction_qc.get("text_preparation", {}).get("policy") != EXEMPLAR_POLICY
                or extraction_qc.get("text_preparation", {}).get("performed_at") != "chat_to_csv_extraction"):
            raise ValueError("Prepared exemplar CSV does not match its extraction QC/policy.")
        exemplar_preparation["extraction_qc"] = {"path": str(extraction_qc_path), "sha256": extraction_qc_hash,
                                                "text_preparation": extraction_qc["text_preparation"]}
    else:
        exemplar_preparation["extraction_qc"] = {"available": False, "text_validation": "passed"}
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
    # separately so a small integration check cannot masquerade as a full run.
    if smoke_test and smoke_test_jobs > len(run):
        raise ValueError(f"smoke_test_jobs exceeds the {len(run)} configured generation jobs.")
    execution_run = run.iloc[:smoke_test_jobs].copy() if smoke_test else run
    execution_run = execution_run.copy()
    execution_run["execution_job_index"] = range(1, len(execution_run) + 1)
    calls = execution_order(execution_run)
    execution = {"mode": "smoke_test" if smoke_test else "study", "test_only": smoke_test,
                 "planned_jobs": len(run), "selected_jobs": len(execution_run),
                 "selection_csv": "generation_design/execution_schedule.csv",
                 "call_order_csv": "generation_design/execution_call_order.csv",
                 "call_order_sha256": order_sha256(calls), "master_seed": config["master_seed"], "seed_policy": "run_level"}

    notebook = config_path.parents[2] / "notebooks/colab_direct_generation_pipeline.ipynb"
    input_hashes = {"config": _sha(config_path), "counterbalance": _sha(config_path.parent / config["counterbalance_csv"]),
                    "model_catalog": _sha(config_path.parent / config["models_file"]),
                    "model_adapter_code": _sha(Path(__file__).with_name("models.py")),
                    "inference_code": _sha(Path(__file__).with_name("inference.py")),
                    "randomness_code": _sha(Path(__file__).with_name("randomness.py")),
                    "recovery_code": _sha(Path(__file__).with_name("recovery.py")),
                    "pipeline_code": _sha(Path(__file__).parents[1] / "pipeline.py"),
                    "execution_selection": hashlib.sha256(json.dumps(execution, sort_keys=True).encode("utf-8")).hexdigest(),
                    "effective_config": hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest(),
                    "notebook": _sha(notebook) if notebook.is_file() else None,
                    "setup_code": _sha(Path(__file__)), "workflow_code": _sha(Path(__file__).with_name("workflow.py")),
                    "prompt_builder_code": _sha(Path(__file__).with_name("prompts.py")),
                    "exemplar_preparation_code": _sha(Path(__file__).with_name("exemplars.py")),
                    "extraction_cleaning_code": _sha(Path(__file__).parents[1] / "chat/text_cleaning.py"),
                    "prepared_exemplars": exemplar_preparation["snapshot_sha256"],
                    "prompt_recording_code": _sha(Path(__file__).with_name("prompt_records.py")),
                    "execution_code": _sha(Path(__file__).with_name("execution.py")),
                    "prompt_templates": prompt_builder.template_hashes,
                    "narratives": _sha(narratives_path), "extraction_qc": extraction_qc_hash,
                    "story_reference": _sha(reference_path),
                    "roster": _sha(config_path.parent / config["roster_csv"])}
    if notebook.is_file():
        cells = json.loads(notebook.read_text())["cells"]
        generation_cells = {cell["id"]: cell["source"] for cell in cells if cell.get("id") in
                            {"generation-setup", "generation-model", "generation-function", "generation-run"}}
        input_hashes["notebook_generation_code"] = hashlib.sha256(json.dumps(generation_cells, sort_keys=True).encode()).hexdigest()
    manifest_path = output / "run_manifest.json"
    if resume:
        saved = json.loads(manifest_path.read_text())
        # Notebook output cells and the RESUME_RUN setting can change. Executed
        # generation cells, effective parameters and all scientific inputs cannot.
        before = {key: value for key, value in saved["input_sha256"].items() if key != "notebook"}
        after = {key: value for key, value in input_hashes.items() if key != "notebook"}
        if before != after or saved["config"] != config or saved["execution"] != execution:
            raise ValueError("Run inputs/settings/code changed; preserve this run and choose a fresh run directory.")
        read_checkpoint(output)
        if not saved.get("design_sha256"):
            raise ValueError("Legacy run has no verified recovery design; existing artifacts preserved.")
        for relative, digest in saved["design_sha256"].items():
            if _sha(output / relative) != digest:
                raise ValueError(f"Saved generation design changed: {relative}")
        artifacts = {}
        for cohort in config["cohorts"]:
            slug = cohort["slug"]
            folder = output / "generation_design" / slug
            artifacts[cohort["label"]] = RotatingGenerationPackageArtifacts(
                folder, folder / f"{slug}_frozen_roster.csv", folder / f"{slug}_prompt_rounds.csv",
                folder / f"{slug}_rotating_generation_schedule.csv", folder / f"{slug}_rotating_bundle_index.csv",
                folder / f"{slug}_rotating_synthetic_bundle_template.csv", folder / f"{slug}_rotating_eval_manifest.csv",
                folder / f"{slug}_rotating_generation_package_summary.json")
        summary_path = output / "generation_design/multi_age_rotating_generation_packages_summary.json"
        return PreparedColabGeneration(config, execution_run, evaluation, MultiAgeRotatingGenerationArtifacts(output, artifacts, summary_path), manifest_path, prompt_builder)
    if output.exists() and any(output.glob("*story_generations.csv")):
        raise FileExistsError("Existing generation outputs found. Choose a fresh run directory.")
    if (output / "prompt_records.jsonl").exists():
        raise FileExistsError("Existing prompt records found. Choose a fresh run directory.")
    if manifest_path.exists() and json.loads(manifest_path.read_text())["input_sha256"] != input_hashes:
        raise ValueError("Run inputs changed. Choose a fresh run directory.")
    output.mkdir(parents=True, exist_ok=True)
    design = output / "generation_design"
    prompt_builder.snapshot(design / "prompts")
    (output / exemplar_preparation["snapshot_csv"]).write_bytes(exemplar_snapshot)
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
    selected_columns = ["execution_job_index", "master_seed", "seed_policy", "cohort", "round_id", "pair_id", "bundle_id", "replicate_id", "target_story",
                        "story_slot_order", "td_child_id", "sli_child_id", "td_age", "sli_age",
                        "prompt_td_story_e1", "prompt_sli_story_e2"]
    execution_run[selected_columns].to_csv(output / execution["selection_csv"], index=False)
    pd.DataFrame(calls).to_csv(output / execution["call_order_csv"], index=False)
    manifest = {"schema_version": 2, "config": config, "input_sha256": input_hashes,
                "master_seed": config["master_seed"], "seed_policy": "run_level",
                "design_sha256": {path.relative_to(output).as_posix(): _sha(path) for path in sorted(design.rglob("*")) if path.is_file()},
                "execution": execution, "generation_jobs": len(execution_run),
                "exemplar_preparation": exemplar_preparation,
                "prompt_records": {"path": "prompt_records.jsonl", "schema_version": 2, "written_during_inference": True},
                "recovery": {"checkpoint": CHECKPOINT, "boundary": "completed_generation_call", "latest_rng_snapshot_only": True},
                "split_validation": "passed", "status": "generation_setup_prepared; model not yet loaded"}
    if not manifest_path.exists():
        atomic_json(manifest_path, manifest)
    return PreparedColabGeneration(config, execution_run, evaluation, MultiAgeRotatingGenerationArtifacts(output, artifacts, summary_path), manifest_path, prompt_builder)
