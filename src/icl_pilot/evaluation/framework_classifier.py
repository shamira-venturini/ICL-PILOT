"""Audited orchestration around SynthTextEval's pinned BERT classifier helpers.

The package's CSV reader, model loader and tokenization are retained. Explicit
training frames and update budgets replace implicit source mixing. Probabilities
are exported once per real story for both ENNI and test-only Gillam.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from .framework_data import digest, frame_digest, _write_json


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def update_budget(n_rows, classifier):
    if "diagnostic_max_steps" in classifier:
        steps = classifier["diagnostic_max_steps"]
        if classifier.get("test_only") is not True or type(steps) is not int or not 1 <= steps <= 2:
            raise ValueError("A diagnostic update budget requires test_only=True and one or two steps.")
        return steps
    batches = math.ceil(n_rows / classifier["batch_size"])
    return max(1, math.ceil(classifier["epochs"] * math.ceil(batches / classifier["gradient_accumulation_steps"])))


def training_jobs(prepared):
    """One shared baseline/control and two conditions per generator, per round/seed."""
    classifier = prepared.protocol["classifier"]
    implementation_hash = digest({path.name: file_digest(path) for path in Path(__file__).parent.glob("framework_*.py")})
    jobs = []
    for round_id, frames in prepared.rounds.items():
        real_counts = frames["train"]["Label"].value_counts().to_dict()
        augmented_sizes = {len(frames["train"]) + len(frame) for frame in frames["synthetic"].values()}
        if len(augmented_sizes) != 1:
            raise ValueError("Generators must have equal augmentation sizes for the shared effort control.")
        augmented_steps = update_budget(augmented_sizes.pop(), classifier)
        conditions = [("shared", "real-only", frames["train"], update_budget(len(frames["train"]), classifier))]
        if classifier["include_update_matched_real_control"]:
            conditions.append(("shared", "real-only-update-matched", frames["train"], augmented_steps))
        for generator in prepared.protocol["generator_keys"]:
            synthetic = frames["synthetic"][generator]
            conditions.extend([
                (generator, "synthetic-train-only", synthetic, update_budget(len(synthetic), classifier)),
                (generator, "synthetic-train-augment", pd.concat([frames["train"], synthetic], ignore_index=True), augmented_steps),
            ])
        for seed in classifier["seeds"]:
            for generator, condition, train, steps in conditions:
                counts = train["Label"].value_counts().to_dict()
                if set(counts) != {0, 1} or any(counts[label]*len(frames["train"]) != real_counts[label]*len(train) for label in (0, 1)):
                    raise ValueError("Every training condition must preserve the real training fold's clinical proportions.")
                source_counts = train["source"].value_counts().to_dict()
                job = {"generator_key": generator, "training_setting": condition,
                       "training_round": round_id, "classifier_seed": seed, "max_steps": steps,
                       "n_train": len(train), "n_real_train": source_counts.get("real", 0),
                       "n_synthetic_train": source_counts.get("synthetic", 0),
                       "n_unique_synthetic_train": int(train.loc[train["source"].eq("synthetic"), "sample_id"].nunique()),
                       "real_reference_TD": real_counts[1], "real_reference_SLI": real_counts[0],
                       "synthetic_sampling": "stratified_bundle_bootstrap_with_replacement",
                       "n_TD_train": counts[1], "n_SLI_train": counts[0],
                       "train_sha256": frame_digest(train), "validation_sha256": frame_digest(frames["validation"]),
                       "test_sha256": frame_digest(frames["test"]), "external_sha256": frame_digest(prepared.external),
                       "protocol_sha256": prepared.protocol["protocol_sha256"], "implementation_sha256": implementation_hash}
                job["effective_synthetic_to_real_ratio"] = (job["n_synthetic_train"] / job["n_real_train"]
                                                            if job["n_real_train"] else None)
                job["classifier_run_id"] = digest(job)[:24]
                jobs.append((job, train.copy(), frames["validation"].copy(), frames["test"].copy()))
    return jobs


def tokenize_audit(tokenizer, frame, max_length, context):
    rows = []
    for row in frame.itertuples():
        tokens = tokenizer(str(row.output_text), add_special_tokens=True, truncation=False)["input_ids"]
        rows.append({**context, "sample_id": row.sample_id, "source": row.source,
                     "group": row.group, "cohort_year": row.cohort_year, "source_child_id": row.source_child_id,
                     "original_tokens_including_special": len(tokens), "used_tokens": min(len(tokens), max_length),
                     "was_truncated": len(tokens) > max_length, "max_length": max_length})
    return pd.DataFrame(rows)


def _torch_probabilities(model, tokenizer, frame, classifier):
    import torch
    model.eval()
    chunks = []
    for start in range(0, len(frame), classifier["eval_batch_size"]):
        batch = frame.iloc[start:start + classifier["eval_batch_size"]]
        encoded = tokenizer(batch["output_text"].astype(str).tolist(), padding=True, truncation=True,
                            max_length=classifier["max_length"], return_tensors="pt")
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.no_grad():
            chunks.append(torch.softmax(model(**encoded).logits, dim=-1).cpu().numpy())
    return np.vstack(chunks)


def package_backend(job, train, validation, tests, protocol, output):
    """Scientific execution requires the configured single CUDA GPU."""
    return _fit_package_classifier(job, train, validation, tests, protocol, output, device="cuda")


def cpu_diagnostic_backend(job, train, validation, tests, protocol, output):
    """Actual bounded CPU training; never an alternative scientific backend."""
    if protocol.get("test_only") is not True or protocol["classifier"].get("test_only") is not True:
        raise ValueError("CPU diagnostic fitting requires explicitly test-only inputs and protocol.")
    if type(job["max_steps"]) is not int or not 1 <= job["max_steps"] <= 2:
        raise ValueError("CPU diagnostics are limited to one or two optimizer updates per fit.")
    return _fit_package_classifier(job, train, validation, tests, protocol, output, device="cpu")


def _write_classifier_dataset(directory, train, validation):
    """Give the toolkit a shared feature/label schema across both CSV splits.

    Real folds and synthetic bootstrap frames have different provenance columns.
    The HF CSV builder infers one schema across splits; passing those metadata
    columns to it can reject synthetic-only or augmented fits. Full identities
    and membership remain in the input/split/bootstrap/tokenization audits.
    """
    for name, frame in (("train", train), ("validation", validation)):
        frame.loc[:, ["output_text", "Label"]].to_csv(directory / (name + ".csv"), index=False)


def _fit_package_classifier(job, train, validation, tests, protocol, output, *, device):
    """Shared package loading/tokenization/training/prediction implementation."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import set_seed
    from synthtexteval.eval.downstream.classify import train_classifier as toolkit

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Classifier fitting requires a CUDA-enabled Colab runtime.")
    if device == "cuda" and torch.cuda.device_count() != 1:
        raise RuntimeError("Use one visible Colab GPU so update budgets retain their recorded meaning.")
    config = protocol["classifier"]
    checkpoint = snapshot_download(repo_id=config["model_id"], revision=config["revision"],
        allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                        "vocab.txt", "special_tokens_map.json"])
    # Seed BEFORE the randomly initialized classification head is loaded.
    set_seed(job["classifier_seed"])
    dataset_dir = output / "dataset"
    dataset_dir.mkdir()
    _write_classifier_dataset(dataset_dir, train, validation)
    args = toolkit.TrainingArguments(
        output_dir=str(output / "trainer"), num_train_epochs=config["epochs"], max_steps=job["max_steps"],
        per_device_train_batch_size=config["batch_size"], per_device_eval_batch_size=config["eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"], weight_decay=config["weight_decay"], warmup_ratio=config["warmup_ratio"],
        seed=job["classifier_seed"], data_seed=job["classifier_seed"], optim="adamw_torch",
        eval_strategy="epoch", save_strategy="no", logging_strategy="steps", logging_steps=10,
        report_to=[], fp16=False, bf16=False, dataloader_num_workers=0, load_best_model_at_end=False,
        use_cpu=device == "cpu",
    )
    model_args = toolkit.ModelArguments(model_name=checkpoint, path_to_dataset=str(dataset_dir),
        path_to_model=str(output / "model"), n_labels=2, problem_type="single_label_classification",
        text_field="output_text", label_field="Label", synthetic_usage="real-only", is_train=True)
    # The explicit train CSV already contains the requested source mix. Disable
    # the package's implicit concatenation to avoid adding synthetic rows twice.
    classifier = toolkit.Classifier(toolkit.Arguments(train=args, model=model_args))
    classifier.model.config.label2id = {"SLI": 0, "TD": 1}
    classifier.model.config.id2label = {0: "SLI", 1: "TD"}
    audits = []
    for name, frame in {"train": train, "validation": validation, **tests}.items():
        audits.append(tokenize_audit(classifier.tokenizer, frame, config["max_length"],
                                     {"classifier_run_id": job["classifier_run_id"], "split": name}))
    pd.concat(audits, ignore_index=True).to_csv(output / "tokenization_audit.csv", index=False)
    trainer = toolkit.Trainer(model=classifier.model.to(device), args=args,
        train_dataset=classifier.process_map(classifier.dataset["train"]),
        eval_dataset=classifier.process_map(classifier.dataset["validation"]),
        compute_metrics=classifier.compute_metrics)
    start = time.monotonic()
    trainer.train()
    trainer.state.save_to_json(str(output / "trainer_state.json"))
    if trainer.state.global_step != job["max_steps"]:
        raise RuntimeError("Observed optimizer updates differ from the planned training budget.")
    if config["retain_model_checkpoints"]:
        trainer.save_model(str(output / "model"))
        classifier.tokenizer.save_pretrained(output / "model")
    predictions = {name: _torch_probabilities(trainer.model, classifier.tokenizer, frame, config)
                   for name, frame in tests.items()}
    _write_json(output / "training_audit.json", {
        **job, "checkpoint_id": config["model_id"], "checkpoint_revision": config["revision"],
        "resolved_checkpoint": checkpoint, "actual_optimizer_steps": trainer.state.global_step,
        "training_seconds": time.monotonic() - start,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "test_only": protocol.get("test_only", False),
        "decision_threshold_sli": config["decision_threshold_sli"], "validation_use": "monitoring_only; fixed final update budget",
        "class_weighting": "none", "label_mapping": {"SLI": 0, "TD": 1},
        "model_checkpoints_retained": config["retain_model_checkpoints"],
        "package_input_columns": ["output_text", "Label"],
        "training_arguments": args.to_dict(),
    })
    del trainer, classifier
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return predictions


def write_training_plan(prepared, output_dir, *, jobs=None):
    """Write the auditable plan only when it agrees with any existing snapshot."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    jobs = training_jobs(prepared) if jobs is None else jobs
    if not jobs or len({job["classifier_run_id"] for job, *_ in jobs}) != len(jobs):
        raise ValueError("A nonempty classifier plan with unique job identities is required.")
    plan_path = output / "training_plan.json"
    plan = [job for job, *_ in jobs]
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Existing classifier training plan differs; use a fresh evaluation run.")
    _write_json(plan_path, plan)
    pd.DataFrame(plan).to_csv(output / "training_plan.csv", index=False)
    return jobs


def _reuse_completed(job, relative, target, roots, backend_identity):
    """Copy small, checksum-verified fit records from an identical completed job."""
    for root in roots:
        cached = Path(root) / relative
        receipt_path = cached / "completed.json"
        if not receipt_path.exists():
            continue
        receipt = json.loads(receipt_path.read_text())
        if receipt["job"] != job or receipt.get("execution_backend") != backend_identity:
            continue
        prediction_path = cached / "story_predictions.csv"
        if not prediction_path.exists() or receipt["predictions_sha256"] != file_digest(prediction_path):
            raise ValueError("Reusable completed classifier outputs failed verification: " + str(cached))
        target.mkdir(parents=True)
        for name in ("requested_job.json", "story_predictions.csv", "training_audit.json",
                     "tokenization_audit.csv", "trainer_state.json"):
            if (cached / name).exists():
                shutil.copy2(cached / name, target / name)
        for name in ("dataset", "model"):
            if (cached / name).is_dir():
                shutil.copytree(cached / name, target / name)
        _write_json(target / "reused_from.json", {"source": str(cached.resolve()),
            "completed_receipt_sha256": file_digest(receipt_path),
            "classifier_run_id": job["classifier_run_id"], "model_refitted": False})
        _write_json(target / "completed.json", receipt)
        return


def run_classifiers(prepared, output_dir=None, *, backend=None, jobs=None, reuse_from=()):
    """Execute or verify completed jobs without overwriting partial model fits."""
    output = Path(output_dir) if output_dir else prepared.output_dir / "classification"
    output.mkdir(parents=True, exist_ok=True)
    jobs = write_training_plan(prepared, output, jobs=jobs)
    backend = package_backend if backend is None else backend
    backend_identity = backend.__module__ + "." + backend.__qualname__
    parts = []
    for index, (job, train, validation, internal) in enumerate(jobs, start=1):
        relative = Path(job["training_round"]) / ("seed_" + str(job["classifier_seed"])) / job["training_setting"] / job["generator_key"]
        target = output / relative
        if not target.exists():
            _reuse_completed(job, relative, target, reuse_from, backend_identity)
        complete = target / "completed.json"
        prediction_path = target / "story_predictions.csv"
        if complete.exists():
            receipt = json.loads(complete.read_text())
            if receipt["job"] != job or receipt.get("execution_backend") != backend_identity or not prediction_path.exists() or receipt["predictions_sha256"] != file_digest(prediction_path):
                raise ValueError("Completed classifier job differs from its inputs/configuration or outputs: " + str(target))
            parts.append(pd.read_csv(prediction_path, dtype={"source_child_id": str, "pair_id": str,
                "sample_id": str, "child_key": str, "classifier_run_id": str}))
            print(f"Verified completed classifier {index}/{len(jobs)}: {job['classifier_run_id']}")
            continue
        if target.exists():
            raise FileExistsError("Partial classifier fit preserved; choose a fresh evaluation run or explicitly recover: " + str(target))
        target.mkdir(parents=True)
        _write_json(target / "requested_job.json", job)
        tests = {"enni": internal, "gillam": prepared.external}
        print(f"Classifier {index}/{len(jobs)}: {job['training_round']} seed={job['classifier_seed']} {job['generator_key']} {job['training_setting']}")
        probabilities = backend(job, train, validation, tests, prepared.protocol, target)
        if set(probabilities) != set(tests):
            raise ValueError("Every classifier must predict both internal ENNI and test-only Gillam.")
        outputs = []
        for dataset, frame in tests.items():
            values = np.asarray(probabilities[dataset], dtype=float)
            if values.shape != (len(frame), 2) or not np.isfinite(values).all() or not np.allclose(values.sum(axis=1), 1) or (values < 0).any():
                raise ValueError("Invalid classifier probabilities or lost test rows.")
            prediction = frame.copy()
            prediction["p_SLI"], prediction["p_TD"] = values[:, 0], values[:, 1]
            for field in ("generator_key", "training_setting", "training_round", "classifier_seed", "classifier_run_id"):
                prediction[field] = job[field]
            prediction["evaluation_dataset"] = dataset
            if "pair_id" not in prediction:
                prediction["pair_id"] = ""
            outputs.append(prediction)
        predictions = pd.concat(outputs, ignore_index=True)
        predictions.to_csv(prediction_path, index=False)
        _write_json(complete, {"job": job, "execution_backend": backend_identity,
                              "predictions_sha256": file_digest(prediction_path)})
        parts.append(predictions)
    result = pd.concat(parts, ignore_index=True)
    result.to_csv(output / "all_story_predictions.csv", index=False)
    return result
