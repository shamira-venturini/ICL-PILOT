"""Append-only records of actual inference prompts and stage ancestry.

This records progress; loading an existing journal to resume is a later step.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid

from .prompts import PromptBuilder, STAGES, row_value


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
    def __init__(self, path: str | Path, *, run_id: str, builder: PromptBuilder, run_manifest: dict):
        self.path, self.run_id, self.builder = Path(path), run_id, builder
        self._prompts, self._outputs, self._jobs = {}, {}, set()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("x", encoding="utf-8") as handle:
            self._write(handle, {"kind": "run", "schema_version": 1, "run_id": run_id,
                                 "run_manifest": run_manifest, "template_sha256": builder.template_hashes})

    @staticmethod
    def _write(handle, record: dict) -> None:
        record = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), **record}
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    def _append(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            self._write(handle, record)

    def record_prompt(self, *, row, stage: str, messages: list, rendered_prompt: str,
                      input_token_ids: list[int], generation_parameters: dict,
                      parent_record_id: str | None = None, stage1_text: str | None = None) -> str:
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
        if not isinstance(rendered_prompt, str) or not rendered_prompt:
            raise ValueError("Record the actual nonempty chat-template text sent to the tokenizer.")
        if not input_token_ids or any(type(value) is not int for value in input_token_ids):
            raise ValueError("Record a nonempty sequence of actual input token IDs.")
        record_id = uuid.uuid4().hex
        token_json = json.dumps(input_token_ids, separators=(",", ":"))
        self._append({"kind": "prompt", "run_id": self.run_id, "record_id": record_id, "stage": stage,
                      **metadata, "parent_record_id": parent_record_id,
                      "parent_text_sha256": text_sha256(stage1_text) if stage1_text is not None else None,
                      "messages": messages, "rendered_prompt": rendered_prompt,
                      "rendered_prompt_sha256": text_sha256(rendered_prompt),
                      "input_token_count": len(input_token_ids), "input_token_ids_sha256": text_sha256(token_json),
                      "generation_parameters": generation_parameters,
                      "template_sha256": self.builder.template_hashes[stage]})
        self._prompts[record_id] = (metadata, stage)
        self._jobs.add(key)
        return record_id

    def record_output(self, record_id: str, text: str, *, generated_token_count: int) -> None:
        if record_id not in self._prompts or record_id in self._outputs:
            raise ValueError("Output requires one previously recorded, unfinished prompt.")
        if not isinstance(text, str) or generated_token_count < 0:
            raise ValueError("Output text and generated-token count are invalid.")
        output_hash = text_sha256(text)
        self._append({"kind": "output", "run_id": self.run_id, "record_id": record_id,
                      "output_text": text, "output_text_sha256": output_hash,
                      "generated_token_count": generated_token_count, "empty_output": not bool(text.strip())})
        self._outputs[record_id] = output_hash
