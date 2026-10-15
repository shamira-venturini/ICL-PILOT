"""Frozen, inspectable message templates for the staged ENNI protocol."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from string import Formatter


STAGES = ("stage1_td", "stage2_sli")
_STORIES = {"A1", "A2", "A3", "B1", "B2", "B3"}
_FIELDS = {
    "stage1_td": {"target_story", "story_title", "replicate_id", "prompt_td_story_e1",
                  "story_units_text", "content_invariants_text", "optional_content_text",
                  "prompt_constraints_text", "prompt_td_exemplar_story"},
    "stage2_sli": {"target_story", "story_title", "prompt_sli_story_e2", "story_units_text",
                   "content_invariants_text", "prompt_sli_exemplar_story", "stage1_text"},
}


def row_value(row, name: str):
    try:
        return row[name] if isinstance(row, Mapping) else getattr(row, name)
    except (KeyError, AttributeError) as error:
        raise ValueError(f"Missing prompt input field: {name}") from error


def validate_story_slots(row) -> None:
    target = row_value(row, "target_story")
    td, sli = row_value(row, "prompt_td_story_e1"), row_value(row, "prompt_sli_story_e2")
    if target not in _STORIES or td not in _STORIES or sli not in _STORIES:
        raise ValueError("Prompt stories must be valid ENNI slots.")
    if target[0] == td[0] or target[0] == sli[0] or td == sli:
        raise ValueError("Prompt exemplars must be distinct slots from the opposite story set.")


@dataclass(frozen=True)
class PromptBuilder:
    # Immutable tuples prevent accidental template mutation after preflight.
    templates: tuple
    source_bytes: tuple
    source_paths: tuple

    @property
    def template_hashes(self) -> dict[str, str]:
        return {stage: hashlib.sha256(raw).hexdigest() for stage, raw in self.source_bytes}

    def snapshot(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for stage, raw in self.source_bytes:
            (directory / (stage + ".json")).write_bytes(raw)

    def _messages(self, stage: str, row, stage1_text: str | None = None) -> list[dict[str, str]]:
        validate_story_slots(row)
        values = {}
        for field in _FIELDS[stage]:
            value = stage1_text if field == "stage1_text" else row_value(row, field)
            if value is None or not str(value).strip():
                raise ValueError(f"Missing or empty prompt input: {field}")
            values[field] = value
        messages = dict(self.templates)[stage]
        return [{"role": role, "content": text.format_map(values)} for role, text in messages]

    def stage1_messages(self, row) -> list[dict[str, str]]:
        return self._messages("stage1_td", row)

    def stage2_messages(self, row, stage1_text: str) -> list[dict[str, str]]:
        if not isinstance(stage1_text, str) or not stage1_text.strip():
            raise ValueError("Stage 2 requires its actual nonempty Stage 1 output.")
        return self._messages("stage2_sli", row, stage1_text)


def load_prompt_builder(config_path: str | Path, config: dict | None = None) -> PromptBuilder:
    config_path = Path(config_path).resolve()
    config = config if config is not None else json.loads(config_path.read_text())
    if set(config.get("prompt_templates", {})) != set(STAGES):
        raise ValueError("Configure exactly one template for each generation stage.")
    templates, sources, paths = [], [], []
    for stage in STAGES:
        relative = Path(config["prompt_templates"][stage])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Prompt template paths must stay inside the configuration directory.")
        path = config_path.parent / relative
        raw = path.read_bytes()
        template = json.loads(raw.decode("utf-8"))
        messages = template.get("messages", [])
        if template.get("schema_version") != 1 or template.get("stage") != stage:
            raise ValueError(f"Invalid prompt template schema/stage: {path}")
        if [message.get("role") for message in messages] != ["system", "user"]:
            raise ValueError("Prompt templates require system and user messages in that order.")
        compiled, fields = [], set()
        for message in messages:
            lines = message.get("lines")
            if not isinstance(lines, list) or not lines or not all(isinstance(line, str) for line in lines):
                raise ValueError("Prompt message lines must be a nonempty list of strings.")
            text = "\n".join(lines)
            if not text.strip():
                raise ValueError("Prompt message text must be nonempty.")
            for _literal, field, spec, conversion in Formatter().parse(text):
                if field is not None:
                    if field not in _FIELDS[stage] or spec or conversion:
                        raise ValueError(f"Unsupported prompt placeholder: {field}")
                    fields.add(field)
            compiled.append((message["role"], text))
        if fields != _FIELDS[stage]:
            raise ValueError(f"Prompt template is missing required fields: {sorted(_FIELDS[stage] - fields)}")
        templates.append((stage, tuple(compiled)))
        sources.append((stage, raw))
        paths.append((stage, str(relative)))
    return PromptBuilder(tuple(templates), tuple(sources), tuple(paths))
