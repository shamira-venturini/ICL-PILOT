"""Sequential two-stage execution with run-level random-stream continuation."""
from __future__ import annotations

import json
import re
from pathlib import Path
import uuid

import pandas as pd

from .prompt_records import PromptJournal, text_sha256
from .randomness import RandomStreams
from .recovery import (CHECKPOINT, JOURNAL, atomic_bytes, atomic_json, execution_order,
                       order_sha256, read_checkpoint, recover_journal, single_writer, timestamp, write_checkpoint)


def cohort_slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def record_runtime_session(manifest_path: Path, metadata: dict, *, resume: bool = False) -> str:
    """Preserve the original runtime; append actual environment for each load."""
    manifest = json.loads(manifest_path.read_text())
    session_id = uuid.uuid4().hex
    manifest.setdefault("model_runtime", metadata)
    manifest.setdefault("runtime_sessions", []).append({"session_id": session_id, "resumed": resume,
                                                       "recorded_at_utc": timestamp(), "runtime": metadata})
    manifest["active_runtime_session_id"] = session_id
    atomic_json(manifest_path, manifest)
    return session_id


@single_writer
def run_generation(prepared, generate, *, run_id: str, save_every: int = 25,
                   on_cohort_complete=None, progress=None, resume: bool = False,
                   on_checkpoint=None, random_streams=None) -> list[Path]:
    """Keep one random stream. Restore only after the caller loads the model.

    Completed calls are committed with RNG state before another call is allowed.
    CSVs are derived caches; the append-only journal and latest checkpoint are
    authoritative. This function never seeds or enables deterministic GPU modes.
    """
    if type(save_every) is not int or save_every < 1:
        raise ValueError("save_every must be a positive integer")
    out = prepared.run_manifest_path.parent
    exists = (out / JOURNAL).exists() or (out / CHECKPOINT).exists() or any(out.glob("*story_generations.csv"))
    if exists and not resume:
        raise FileExistsError("Generation artifacts exist. Use explicit resume=True with a valid RNG checkpoint, or choose a fresh run.")
    manifest = json.loads(prepared.run_manifest_path.read_text())
    master_seed = prepared.config.get("master_seed", 42)
    if master_seed != 42 or prepared.config.get("seed_policy", "run_level") != "run_level":
        raise ValueError("This experiment requires master_seed=42 and seed_policy=run_level.")
    calls = execution_order(prepared.run_df)
    order_hash = order_sha256(calls)
    streams = random_streams if random_streams is not None else RandomStreams()
    checkpoint = None
    if resume:
        checkpoint = read_checkpoint(out)
        if (checkpoint["run_id"] != run_id or checkpoint["execution_order_sha256"] != order_hash
                or checkpoint["input_sha256"] != manifest.get("input_sha256", {})
                or checkpoint["master_seed"] != master_seed or checkpoint["seed_policy"] != "run_level"):
            raise ValueError("Checkpoint run, inputs, seed policy or execution order differ; artifacts preserved.")
        count = checkpoint["completed_calls"]
        if not 0 <= count <= len(calls) or checkpoint["next_call"] != (calls[count] if count < len(calls) else None):
            raise ValueError("Checkpoint does not identify the next scheduled call.")
        recover_journal(out, checkpoint)

    def commit_output(event, journal):
        nonlocal checkpoint
        count = len(journal.output_records)
        if count >= len(calls) or event["execution_call"] != calls[count]:
            raise ValueError("Output is not the next scheduled generation call.")
        checkpoint = write_checkpoint(out, {
            "schema_version": 1, "run_id": run_id, "master_seed": master_seed, "seed_policy": "run_level",
            "recorded_at_utc": timestamp(), "execution_order_sha256": order_hash,
            "input_sha256": manifest.get("input_sha256", {}), "completed_calls": count + 1,
            "last_completed": {**calls[count], "record_id": event["record_id"], "output_text_sha256": event["output_text_sha256"]},
            "next_call": calls[count + 1] if count + 1 < len(calls) else None,
            "journal_prefix_bytes": journal.byte_count, "journal_prefix_sha256": journal.sha256,
            "output_event": event, "rng_state": streams.capture(),
            "runtime_session_id": journal.runtime_session_id,
        })

    journal = PromptJournal(out / JOURNAL, run_id=run_id, builder=prepared.prompt_builder,
                            run_manifest=manifest, resume=resume, commit_output=commit_output,
                            runtime_session_id=manifest.get("active_runtime_session_id"))
    if resume:
        if len(journal.output_records) != checkpoint["completed_calls"]:
            raise ValueError("Journal completed-call count differs from checkpoint.")
        for call, (record_id, output) in zip(calls, journal.output_records.items()):
            prompt = journal.prompt_events[record_id]
            if output["execution_call"] != call or prompt["execution_call"] != call:
                raise ValueError("Journal execution order differs from schedule.")
            if call["stage"] == "stage2_sli":
                parent = journal.completed.get((call["job_id"], "stage1_td"))
                if prompt["parent_record_id"] != parent or prompt["parent_text_sha256"] != journal._outputs[parent]:
                    raise ValueError("Saved Stage 2 ancestry differs from its Stage 1 parent.")
        if checkpoint["completed_calls"]:
            last = next(reversed(journal.output_records.values()))
            if last != checkpoint["output_event"]:
                raise ValueError("Last completed output differs from checkpoint.")
        journal.abandon_unfinished()
    else:
        checkpoint = write_checkpoint(out, {
            "schema_version": 1, "run_id": run_id, "master_seed": master_seed, "seed_policy": "run_level",
            "recorded_at_utc": timestamp(), "execution_order_sha256": order_hash,
            "input_sha256": manifest.get("input_sha256", {}), "completed_calls": 0,
            "last_completed": None, "next_call": calls[0] if calls else None,
            "journal_prefix_bytes": journal.byte_count, "journal_prefix_sha256": journal.sha256,
            "output_event": None, "rng_state": streams.capture(),
            "runtime_session_id": journal.runtime_session_id,
        })

    aggregate = out / "all_cohorts_rotating_story_generations.csv"
    paths, all_rows = [], []
    builder, model_settings = prepared.prompt_builder, prepared.config["model"]
    restore_pending = resume

    def status(value):
        manifest.update(status=value, completed_generation_jobs=checkpoint["completed_calls"] // 2,
                        completed_generation_calls=checkpoint["completed_calls"],
                        master_seed=master_seed, seed_policy="run_level",
                        recovery={"checkpoint": CHECKPOINT, "next_call": checkpoint["next_call"]})
        atomic_json(prepared.run_manifest_path, manifest)

    def save_csv(path, rows):
        if rows:
            atomic_bytes(path, pd.DataFrame(rows).to_csv(index=False).encode("utf-8"))

    def call(row, stage, job_index, parent=None, parent_id=None):
        nonlocal restore_pending
        scheduled = calls[2 * job_index + (stage == "stage2_sli")]
        existing = journal.completed.get((scheduled["job_id"], stage))
        if existing is not None:
            return journal.output_records[existing]["output_text"], existing
        journal.active_call = scheduled
        messages = builder.stage1_messages(row) if parent is None else builder.stage2_messages(row, parent)
        if restore_pending:
            streams.restore(checkpoint["rng_state"])
            restore_pending = False
        kwargs = {} if parent is None else {"parent_record_id": parent_id, "stage1_text": parent}
        before = len(journal.output_records)
        result = generate(messages, max_new_tokens=model_settings["max_new_tokens_stage1" if parent is None else "max_new_tokens_stage2"],
                          row=row, stage=stage, journal=journal, **kwargs)
        if (len(journal.output_records) != before + 1 or journal.completed.get((scheduled["job_id"], stage)) != result[1]
                or journal.output_records[result[1]]["output_text"] != result[0]):
            raise ValueError("Inference must return exactly its committed journal output.")
        if on_checkpoint is not None:
            on_checkpoint(out)
        return result

    status("generation_in_progress")
    job_index = 0
    try:
        if on_checkpoint is not None:
            on_checkpoint(out)
        for label, frame in prepared.run_df.groupby("cohort_label", sort=False):
            path = out / f"{cohort_slug(label)}_rotating_story_generations.csv"
            cohort_rows = []
            rows = frame.itertuples(index=False)
            if progress is not None:
                rows = progress(rows, total=len(frame), desc=f"Generate {label}")
            for row in rows:
                first, first_id = call(row, "stage1_td", job_index)
                second, second_id = call(row, "stage2_sli", job_index, first, first_id)
                names = ("cohort", "cohort_label", "round_id", "bundle_id", "pair_id", "replicate_id",
                         "story_slot_order", "target_story", "story_title", "sli_child_id", "td_child_id",
                         "sli_age", "td_age", "prompt_td_story_e1", "prompt_sli_story_e2", "story_units_text")
                record = {key: getattr(row, key) for key in names}
                record.update(stage1_group="TD", stage2_group="SLI", evaluation_td_group="TD",
                              evaluation_sli_group="SLI", stage1_td_story=first, stage2_sli_story=second,
                              stage1_prompt_record_id=first_id, stage2_prompt_record_id=second_id,
                              stage1_text_sha256=text_sha256(first), stage2_parent_text_sha256=text_sha256(first),
                              test_only=manifest.get("execution", {}).get("test_only", False))
                record.update(run_id=run_id, model_key=model_settings.get("key"), model_id=model_settings.get("id"),
                              model_revision=model_settings.get("revision"), master_seed=master_seed, seed_policy="run_level",
                              execution_job_index=job_index + 1, stage1_call_index=2 * job_index + 1, stage2_call_index=2 * job_index + 2)
                cohort_rows.append(record)
                all_rows.append(record)
                job_index += 1
                if len(cohort_rows) % save_every == 0:
                    save_csv(path, cohort_rows)
            save_csv(path, cohort_rows)
            save_csv(aggregate, all_rows)
            paths.append(path)
            status("generation_in_progress")
            if on_cohort_complete is not None:
                on_cohort_complete(path)
    except BaseException:
        status("generation_interrupted; resume from completed-call checkpoint")
        raise
    status("generation_complete")
    return paths
