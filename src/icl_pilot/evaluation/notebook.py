"""File-based entry points for independent evaluation blocks in the main notebook.

The notebook kernel uses only the standard library here. Analysis imports occur
inside the separate evaluation process; generation and annotation are never run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _path(root, value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def create_session(project_root, *, generation_csvs=None, text_records_csv=None,
                   gillam_csv="data/inputs/gillam/external_story_records.csv",
                   protocol_path="configs/evaluation/protocol.json", resume_dir=None,
                   ratios=(1, 10), output_root=None):
    """Select explicit inputs before creating a run; resume verifies every input."""
    root = Path(project_root).expanduser().resolve()
    if resume_dir is not None:
        session = _path(root, resume_dir) / "notebook_session.json"
        config = json.loads(session.read_text())
        if config.get("schema_version") != 1:
            raise ValueError("Unsupported notebook evaluation session.")
        old_root = Path(config["project_root"])
        if old_root != root or Path(config["output_dir"]) != session.parent:
            def relocate(name):
                path = Path(name)
                try:
                    return str(root / path.relative_to(old_root))
                except ValueError:
                    return str(path)
            updated = {**config, "project_root": str(root), "output_dir": str(session.parent)}
            for field in ("text_records_csv", "gillam_csv", "protocol_path", "participants_csv", "stories_csv", "experiment_path", "roster_csv"):
                if updated.get(field):
                    updated[field] = relocate(updated[field])
            for field in ("inputs", "implementation"):
                updated[field] = {relocate(name): expected for name, expected in config[field].items()}
            updated["generation_csvs"] = {model: [relocate(name) for name in names] for model, names in config["generation_csvs"].items()}
            _verify_files(updated)
            record = session.parent / "session_relocations.json"
            events = json.loads(record.read_text()) if record.exists() else []
            events.append({"previous_session_sha256": sha256(session), "old_project_root": str(old_root),
                           "old_output_dir": config["output_dir"], "new_project_root": str(root),
                           "new_output_dir": str(session.parent)})
            _json(record, events)
            _json(session, updated)
        load_session(session)
        return session
    if bool(generation_csvs) == bool(text_records_csv):
        raise ValueError("Select either generation_csvs or text_records_csv, with no automatic fallback.")
    if not ratios or any(type(r) is not int or r < 1 for r in ratios) or len(set(ratios)) != len(ratios):
        raise ValueError("Ratios must be unique positive integers.")
    protocol = _path(root, protocol_path)
    experiment = root / "configs/generation/experiment.json"
    generation = json.loads(experiment.read_text())
    roster = experiment.parent / generation["roster_csv"]
    participants = _path(root, generation["provenance"]["participant_manifest"])
    stories = _path(root, generation["inputs"]["narratives_csv"])
    catalog = _path(root, json.loads(protocol.read_text())["generator_catalog"])
    selected = {}
    for model, values in (generation_csvs or {}).items():
        if isinstance(values, (str, Path)):
            values = [values]
        if not values:
            raise ValueError("Select saved generation CSVs for " + model)
        selected[model] = [str(_path(root, value)) for value in values]
    configured = json.loads(protocol.read_text())["generator_keys"]
    if selected and set(selected) != set(configured):
        raise ValueError("Select saved CSVs for every configured generator separately.")
    paths = [Path(value) for values in selected.values() for value in values]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate generation CSV paths; select aggregate or cohort files, not both.")
    records = _path(root, text_records_csv) if text_records_csv else None
    external = _path(root, gillam_csv)
    paths += [protocol, catalog, experiment, roster, participants, stories, external]
    if records:
        paths.append(records)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError("Evaluation input is unavailable: " + str(path))
    from .runs import create_framework_evaluation_run
    output = create_framework_evaluation_run(root, output_root=output_root)
    session = output / "notebook_session.json"
    _json(session, {
        "schema_version": 1, "project_root": str(root), "output_dir": str(output),
        "generation_csvs": selected, "text_records_csv": str(records) if records else None,
        "gillam_csv": str(external), "protocol_path": str(protocol),
        "participants_csv": str(participants), "stories_csv": str(stories),
        "experiment_path": str(experiment), "roster_csv": str(roster), "ratios": sorted(ratios),
        "inputs": {str(path): sha256(path) for path in paths},
        "implementation": {str(path): sha256(path) for path in sorted((root / "src/icl_pilot/evaluation").glob("*.py"))},
    })
    return session


def load_session(session):
    config = json.loads(Path(session).read_text())
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported notebook evaluation session.")
    _verify_files(config)
    return config


def _verify_files(config):
    for field in ("inputs", "implementation"):
        for name, expected in config[field].items():
            if not Path(name).is_file() or sha256(name) != expected:
                raise ValueError("Evaluation input or implementation changed; use a fresh run: " + name)


def prepare_session(session):
    """Build narrative inputs once, then use the existing audited split machinery."""
    import pandas as pd
    from .framework_data import load_protocol, prepare_evaluation, child_id
    from .story_text_exports import _generated_records, _real_records, RECORD_COLUMNS

    config = load_session(session)
    root, output = Path(config["project_root"]), Path(config["output_dir"])
    protocol = load_protocol(root, config["protocol_path"])
    real, missing = _real_records(
        measures_path=Path(config["participants_csv"]), stories_path=Path(config["stories_csv"]),
        project_root=root, minimum_age_months=min(protocol["cohorts"]) * 12,
        maximum_age_months=(max(protocol["cohorts"]) + 1) * 12,
    )
    if missing:
        raise ValueError("Canonical ENNI participants lack narrative text.")
    if config["text_records_csv"]:
        records = pd.read_csv(config["text_records_csv"], dtype={"source_child_id": str, "td_child_id": str, "sli_child_id": str})
    else:
        generated = []
        for model, paths in config["generation_csvs"].items():
            for name in paths:
                rows = _generated_records(source_path=Path(name), cohort_year=None, project_root=root)
                if not rows or any(row["model_key"] != model for row in rows):
                    raise ValueError("Selected generation CSV disagrees with its generator label: " + name)
                generated.extend(rows)
        records = pd.DataFrame(real + generated, columns=list(RECORD_COLUMNS))
    # A prebuilt export must contain the same canonical real stories and text.
    actual = records.loc[records["source"].eq("real")]
    def real_inventory(frame):
        return sorted((child_id(row.source_child_id), row.group, int(row.cohort_year), row.story_id, row.story_text)
                      for row in frame.itertuples())
    if real_inventory(actual) != real_inventory(pd.DataFrame(real)):
        raise ValueError("Evaluation ENNI reference differs from the canonical participant/story inputs.")
    external = pd.read_csv(config["gillam_csv"], dtype={"source_child_id": str})
    prepared = prepare_evaluation(records, external, protocol, output)
    # Equal generator schedules alone do not establish full study completeness.
    # Check all frozen pairs/replicates and both stages against the study design.
    roster = pd.read_csv(config["roster_csv"], dtype=str)
    experiment = json.loads(Path(config["experiment_path"]).read_text())
    expected = set()
    for row in roster.itertuples():
        age = int(row.cohort.split("-", 1)[0])
        if age in protocol["cohorts"]:
            for rep in range(1, experiment["n_replicates"] + 1):
                for story in protocol["story_ids"]:
                    for stage in ("stage1_td", "stage2_sli"):
                        expected.add((age, row.pair_id, rep, story, stage, child_id(row.td_child_id), child_id(row.sli_child_id)))
    fields = ["cohort_year", "pair_id", "replicate_id", "story_id", "stage", "td_child_id", "sli_child_id"]
    for model in protocol["generator_keys"]:
        observed = set(map(tuple, prepared.synthetic.loc[prepared.synthetic["model_key"].eq(model), fields].to_numpy()))
        if observed != expected:
            raise ValueError("Incomplete or conflicting frozen study schedule for " + model)
    combined = output / "evaluation_text_records.csv"
    if combined.exists() and sha256(combined) != json.loads((output / "prepared_text_receipt.json").read_text())["sha256"]:
        raise ValueError("Prepared evaluation text changed; existing file preserved.")
    if not combined.exists():
        records.to_csv(combined, index=False)
        _json(output / "prepared_text_receipt.json", {"sha256": sha256(combined)})
    # Generation QC is review-only, including prompt-like meta responses.
    mask = records["source"].eq("synthetic") & records["story_text"].str.contains(
        "I replaced the main characters to match the target story", case=False, regex=False, na=False)
    records.loc[mask].to_csv(output / "review_only_meta_generations.csv", index=False)
    return prepared


def run_action(session, action, *, backend=None):
    """Run just the requested block; classifier fitting is never implicit."""
    import pandas as pd
    from .framework_classifier import write_training_plan, run_classifiers
    from .framework_ratios import prepare_ratio_experiment, run_ratio_experiment, summarize_ratio_experiment
    from .framework_metrics import summarize_predictions

    config = load_session(session)
    output = Path(config["output_dir"])
    prepared = prepare_session(session)
    classification = output / "classification"
    ratios = output / "ratio_experiment"
    if action == "inputs":
        plan = write_training_plan(prepared, classification)
        print("Planned classifier fits:", len(plan))
    elif action == "descriptive":
        from .framework_descriptive import run_descriptive
        run_descriptive(prepared)
    elif action == "quality":
        from .framework_quality import run_text_quality
        run_text_quality(prepared)
    elif action == "classifiers":
        run_classifiers(prepared, classification, backend=backend)
    elif action == "summary":
        predictions = classification / "all_story_predictions.csv"
        _require_prediction_receipt(output, "classifiers", predictions)
        summarize_predictions(pd.read_csv(predictions, dtype={"source_child_id": str}), prepared.protocol, classification / "summary")
    elif action in {"ratio-plan", "ratios", "ratio-summary"}:
        experiment = prepare_ratio_experiment(prepared, ratios, ratios=config["ratios"])
        if action == "ratios":
            run_ratio_experiment(experiment, backend=backend, reuse_from=(classification,))
        elif action == "ratio-summary":
            predictions = ratios / "all_ratio_story_predictions.csv"
            _require_prediction_receipt(output, "ratios", predictions)
            results = summarize_ratio_experiment(pd.read_csv(predictions, dtype={"source_child_id": str}), prepared.protocol, ratios / "summary")
            _ratio_plot(results["summary"], prepared.protocol, config["ratios"], ratios)
        print("Planned ratio-experiment fits:", len(experiment.plan))
    else:
        raise ValueError("Unknown evaluation action: " + action)
    receipts = output / "block_receipts"
    receipts.mkdir(exist_ok=True)
    predictions = {"classifiers": classification / "all_story_predictions.csv",
                   "ratios": ratios / "all_ratio_story_predictions.csv"}.get(action)
    _json(receipts / (action + ".json"), {"action": action, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_sha256": sha256(predictions) if predictions else None})
    print("Completed", action, "— results:", output)


def _require_prediction_receipt(output, action, predictions):
    receipt = output / "block_receipts" / (action + ".json")
    if not receipt.is_file() or not predictions.is_file():
        raise RuntimeError("Run the " + action + " block first, or resume its saved evaluation session.")
    if sha256(predictions) != json.loads(receipt.read_text())["prediction_sha256"]:
        raise ValueError("Saved prediction table changed; summary refused.")


def _ratio_plot(summary, protocol, ratios, output):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    children = summary.loc[summary["unit"].eq("child")]
    fig = Figure(figsize=(10, 4))
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2, sharey=True)
    for axis, dataset in zip(axes, ("enni", "gillam")):
        scope = children.loc[children["evaluation_dataset"].eq(dataset)]
        baseline = scope.loc[scope["training_setting"].eq("real-only"), "balanced_accuracy_mean"].iloc[0]
        axis.axhline(baseline, color="black", linestyle="--", alpha=.5, label="Real only")
        for generator in protocol["generator_keys"]:
            points = scope.loc[scope["generator_key"].eq(generator) & scope["training_setting"].eq("synthetic-train-augment")].sort_values("ratio_level")
            axis.plot([0] + points["ratio_level"].tolist(), [baseline] + points["balanced_accuracy_mean"].tolist(), marker="o", label=generator)
        axis.set(title=dataset.upper(), xlabel="Sampled synthetic stories per real training story", ylim=(0, 1))
        axis.set_xticks([0] + sorted(ratios))
    axes[0].set_ylabel("Mean child-level balanced accuracy")
    axes[1].legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(output / "child_balanced_accuracy_by_ratio.png", dpi=180)


def execute_block(session, action, runtime_python):
    """Stream worker progress in Colab without changing the notebook environment."""
    from .notebook_runtime import verify_runtime
    config = load_session(session)
    runtime = verify_runtime(runtime_python, toolkit=action in {"quality", "classifiers", "ratios"})
    _json(Path(config["output_dir"]) / ("runtime_" + action + ".json"), runtime)
    env = dict(os.environ, PYTHONPATH=str(Path(config["project_root"]) / "src"), PYTHONDONTWRITEBYTECODE="1")
    subprocess.run([str(runtime_python), "-m", "icl_pilot.evaluation.notebook", "--session", str(session), "--action", action], env=env, check=True)


def export_session(session):
    """Create a verified compact ZIP; model/checkpoint directories stay on disk."""
    config = load_session(session)
    output = Path(config["output_dir"])
    excluded = {"exports", "__pycache__", "model", "trainer", "tester", "external_gillam_tester"}
    paths = [p for p in sorted(output.rglob("*")) if p.is_file() and not (set(p.relative_to(output).parts) & excluded)]
    inventory = {p.relative_to(output).as_posix(): sha256(p) for p in paths}
    exports = output / "exports"
    exports.mkdir(exist_ok=True)
    archive = exports / (output.name + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".zip")
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as handle:
        for path in paths:
            handle.write(path, path.relative_to(output).as_posix())
        handle.writestr("portable_file_manifest.json", json.dumps(inventory, indent=2) + "\n")
    with zipfile.ZipFile(archive) as handle:
        if handle.testzip() is not None:
            raise OSError("Evaluation export ZIP failed CRC verification.")
        for name, expected in inventory.items():
            if hashlib.sha256(handle.read(name)).hexdigest() != expected:
                raise OSError("Evaluation export checksum mismatch: " + name)
    _json(archive.with_suffix(".json"), {"archive": str(archive), "sha256": sha256(archive), "files": len(inventory)})
    return archive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--action", required=True, choices=["inputs", "descriptive", "quality", "classifiers", "summary", "ratio-plan", "ratios", "ratio-summary"])
    args = parser.parse_args()
    run_action(args.session, args.action)


if __name__ == "__main__":
    main()
