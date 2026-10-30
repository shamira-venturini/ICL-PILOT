"""Append-only records of actual inference prompts, outputs and stage ancestry."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import uuid

from .prompts import PromptBuilder, STAGES, row_value
from .models import model_messages as prepare_model_messages
from .recovery import event_bytes, timestamp


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _metadata(row) -> dict:
    names = ("cohort", "round_id", "bundle_id", "pair_id", "target_story", "td_age", "sli_age",
             "prompt_td_story_e1", "prompt_sli_story_e2")
    metadata = {name: str(row_value(row, name)) for name in names}
    for name in ("replicate_id", "story_slot_order", "td_child_id", "sli_child_id"):
        metadata[name] = int(row_value(row, name))
    metadata["job_id"] = "/".join(metadata[name] for name in ("cohort", "round_id", "bundle_id", "target_story"))
    return metadata


class PromptJournal:
    def __init__(self, path: str | Path, *, run_id: str, builder: PromptBuilder, run_manifest: dict,
                 resume: bool = False, commit_output=None, runtime_session_id=None):
        self.path, self.run_id, self.builder = Path(path), run_id, builder
        self._prompts, self._outputs, self._jobs = {}, {}, set()
        self._model_contexts = {}
        self.output_records, self.prompt_events = {}, {}
        self.completed = {}
        self._abandoned = set()
        self.commit_output, self.runtime_session_id = commit_output, runtime_session_id
        self.active_call = None
        self.master_seed = run_manifest.get("config", {}).get("master_seed", 42)
        self.model_settings = run_manifest.get("config", {}).get("model", {})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._hash, self.byte_count = hashlib.sha256(), 0
        if resume:
            with self.path.open("rb") as handle:
                for line in handle:
                    record = json.loads(line)
                    if record.get("run_id") != run_id:
                        raise ValueError("Journal belongs to another run.")
                    self._hash.update(line)
                    self.byte_count += len(line)
                    kind = record["kind"]
                    if kind == "run":
                        if self.byte_count != len(line) or record["template_sha256"] != builder.template_hashes:
                            raise ValueError("Journal header/templates differ.")
                    elif kind == "prompt":
                        record_id = record["record_id"]
                        metadata = {name: record[name] for name in _metadata(record)}
                        key = (metadata["job_id"], record["stage"])
                        if record_id in self._prompts or key in self._jobs:
                            raise ValueError("Duplicate prompt/job in journal.")
                        self._prompts[record_id] = (metadata, record["stage"])
                        self._model_contexts[record_id] = record.get("model_context")
                        self.prompt_events[record_id] = record
                        self._jobs.add(key)
                    elif kind == "output":
                        record_id = record["record_id"]
                        if record_id not in self._prompts or record_id in self._outputs:
                            raise ValueError("Duplicate or orphan output in journal.")
                        if text_sha256(record["output_text"]) != record["output_text_sha256"]:
                            raise ValueError("Journal output checksum failed.")
                        self._accept_output(record)
                    elif kind == "attempt_abandoned":
                        prompt = self._prompts[record["record_id"]]
                        if record["record_id"] in self._outputs:
                            raise ValueError("A completed output cannot be abandoned.")
                        self._jobs.discard((prompt[0]["job_id"], prompt[1]))
                        self._abandoned.add(record["record_id"])
                    else:
                        raise ValueError("Unknown journal event.")
            return
        with self.path.open("x", encoding="utf-8"):
            pass
        self._append({"kind": "run", "schema_version": 2, "run_id": run_id,
                      "master_seed": self.master_seed, "seed_policy": "run_level",
                      "run_manifest": run_manifest, "template_sha256": builder.template_hashes})

    def _append(self, record: dict) -> dict:
        record = {"recorded_at_utc": timestamp(), **record}
        value = event_bytes(record)
        with self.path.open("ab") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        self._hash.update(value)
        self.byte_count += len(value)
        return record

    @property
    def sha256(self):
        return self._hash.hexdigest()

    def abandon_unfinished(self) -> None:
        for record_id, (metadata, stage) in self._prompts.items():
            key = (metadata["job_id"], stage)
            if record_id not in self._outputs and record_id not in self._abandoned:
                self._append({"kind": "attempt_abandoned", "run_id": self.run_id, "record_id": record_id,
                              "reason": "interrupted before a committed output; retry from saved RNG state"})
                self._jobs.discard(key)
                self._abandoned.add(record_id)

    def _accept_output(self, record):
        record_id = record["record_id"]
        metadata, stage = self._prompts[record_id]
        self._outputs[record_id] = record["output_text_sha256"]
        self.output_records[record_id] = record
        self.completed[(metadata["job_id"], stage)] = record_id

    def record_prompt(self, *, row, stage: str, messages: list, rendered_prompt: str,
                      input_token_ids: list[int], generation_parameters: dict,
                      parent_record_id: str | None = None, stage1_text: str | None = None,
                      model_messages: list | None = None, model_context: dict | None = None) -> str:
        if stage not in STAGES:
            raise ValueError("Unknown prompt stage.")
        metadata = _metadata(row)
        key = (metadata["job_id"], stage)
        if key in self._jobs:
            raise ValueError("A prompt is already recorded for this job/stage.")
        if stage == "stage1_td":
            if parent_record_id is not None or stage1_text is not None:
                raise ValueError("Stage 1 cannot have a Stage 1 parent.")
            expected = self.builder.stage1_messages(row)
        else:
            parent = self._prompts.get(parent_record_id)
            output = self._outputs.get(parent_record_id)
            if parent != (metadata, "stage1_td") or output is None:
                raise ValueError("Stage 2 requires a completed Stage 1 parent from the same job.")
            expected = self.builder.stage2_messages(row, stage1_text)
            if text_sha256(stage1_text) != output:
                raise ValueError("Stage 2 parent text differs from the recorded Stage 1 output.")
        if messages != expected:
            raise ValueError("Messages differ from the configured prompt builder.")
        if model_messages is not None and model_messages != prepare_model_messages(messages):
            raise ValueError("Model-facing messages differ from the shared instruction packaging.")
        if model_context is not None:
            for name in ("key", "id", "revision"):
                if name in self.model_settings and model_context.get(name) != self.model_settings[name]:
                    raise ValueError("Inference model differs from the run's selected checkpoint.")
        if stage == "stage2_sli" and model_context != self._model_contexts[parent_record_id]:
            raise ValueError("Stage 2 must use its Stage 1 parent's model context.")
        if not isinstance(rendered_prompt, str) or not rendered_prompt:
            raise ValueError("Record the actual nonempty chat-template text sent to the tokenizer.")
        if not input_token_ids or any(type(value) is not int for value in input_token_ids):
            raise ValueError("Record a nonempty sequence of actual input token IDs.")
        record_id = uuid.uuid4().hex
        token_json = json.dumps(input_token_ids, separators=(",", ":"))
        prompt = self._append({"kind": "prompt", "run_id": self.run_id, "record_id": record_id, "stage": stage,
                      "master_seed": self.master_seed, "seed_policy": "run_level",
                      "execution_call": self.active_call, "runtime_session_id": self.runtime_session_id,
                      **metadata, "parent_record_id": parent_record_id,
                      "parent_text_sha256": text_sha256(stage1_text) if stage1_text is not None else None,
                      "messages": messages, "rendered_prompt": rendered_prompt,
                      "model_messages": model_messages, "model_context": model_context,
                      "rendered_prompt_sha256": text_sha256(rendered_prompt),
                      "input_token_count": len(input_token_ids), "input_token_ids_sha256": text_sha256(token_json),
                      "generation_parameters": generation_parameters,
                      "template_sha256": self.builder.template_hashes[stage]})
        self.prompt_events[record_id] = prompt
        self._prompts[record_id] = (metadata, stage)
        self._model_contexts[record_id] = copy.deepcopy(model_context)
        self._jobs.add(key)
        return record_id

    def record_output(self, record_id: str, text: str, *, generated_token_count: int,
                      finish_reason: str | None = None, terminal_token_id: int | None = None,
                      reached_token_ceiling: bool | None = None) -> None:
        if record_id not in self._prompts or record_id in self._outputs:
            raise ValueError("Output requires one previously recorded, unfinished prompt.")
        if not isinstance(text, str) or generated_token_count < 0:
            raise ValueError("Output text and generated-token count are invalid.")
        output_hash = text_sha256(text)
        record = {"recorded_at_utc": timestamp(), "kind": "output", "run_id": self.run_id, "record_id": record_id,
                      "execution_call": self.active_call, "runtime_session_id": self.runtime_session_id,
                      "master_seed": self.master_seed, "seed_policy": "run_level",
                      "output_text": text, "output_text_sha256": output_hash,
                      "finish_reason": finish_reason, "terminal_token_id": terminal_token_id,
                      "reached_token_ceiling": reached_token_ceiling,
                      "generated_token_count": generated_token_count, "empty_output": not bool(text.strip())}
        if self.commit_output is not None:
            self.commit_output(record, self)
        self._append(record)
        self._accept_output(record)
