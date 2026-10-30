"""Pinned text-model selection and common, content-preserving chat packaging."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re


INSTRUCTION_PACKAGING = "system-in-user-v1"
TRANSFORMERS_VERSION = "4.57.1"


def load_model_catalog(path: str | Path) -> dict:
    catalog = json.loads(Path(path).read_text(encoding="utf-8"))
    if catalog.get("schema_version") != 1 or not catalog.get("models"):
        raise ValueError("Unsupported or empty generation model catalog.")
    for key, spec in catalog["models"].items():
        if not re.fullmatch(r"[a-z0-9_]+", key):
            raise ValueError("Model keys must be lowercase letters, numbers and underscores.")
        if not spec.get("id") or not re.fullmatch(r"[a-f0-9]{40}", spec.get("revision", "")):
            raise ValueError(f"Model {key} needs an official checkpoint and pinned revision.")
        if spec.get("family") not in {"llama", "qwen3", "gemma2"} or not spec.get("turn_end_token"):
            raise ValueError(f"Unsupported model family or missing turn-end token: {key}")
        expected = {"enable_thinking": False} if spec["family"] == "qwen3" else {}
        if spec.get("chat_template_kwargs") != expected:
            raise ValueError(f"Unexpected chat-template options for {key}; Qwen must disable thinking.")
    return catalog["models"]


def select_model(settings: dict, catalog: dict, key: str | None = None) -> dict:
    default = settings.get("key")
    if default not in catalog or any(settings.get(name) != catalog[default][name] for name in ("id", "revision")):
        raise ValueError("Default model key/id/revision disagree with the pinned model catalog.")
    chosen = default if key is None else key
    if chosen not in catalog:
        raise ValueError(f"Unknown model key {chosen!r}; choose from {', '.join(catalog)}.")
    return {**copy.deepcopy(settings), **copy.deepcopy(catalog[chosen]), "key": chosen,
            "instruction_packaging": INSTRUCTION_PACKAGING}


def model_messages(messages: list) -> list:
    """Move the literal system sentence into the user turn for every family."""
    if len(messages) != 2 or [m.get("role") for m in messages] != ["system", "user"]:
        raise ValueError("Expected the canonical system/user prompt pair.")
    if any(not isinstance(m.get("content"), str) or not m["content"] for m in messages):
        raise ValueError("Canonical prompt contents must be nonempty strings.")
    return [{"role": "user", "content": messages[0]["content"] + "\n\n" + messages[1]["content"]}]


def render_prompt(tokenizer, messages: list, settings: dict) -> tuple[list, str]:
    if settings.get("instruction_packaging") != INSTRUCTION_PACKAGING:
        raise ValueError("Use the shared system-in-user instruction packaging.")
    packed = model_messages(messages)
    prompt = tokenizer.apply_chat_template(packed, tokenize=False, add_generation_prompt=True,
                                          **settings["chat_template_kwargs"])
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("The native chat template returned an empty prompt.")
    return packed, prompt


def stop_token_ids(model, tokenizer, settings: dict) -> list[int]:
    """Preserve checkpoint stop IDs and include its native end-of-turn token."""
    result = []
    for value in (getattr(model.generation_config, "eos_token_id", None), tokenizer.eos_token_id):
        values = value if isinstance(value, (list, tuple)) else [value]
        for token_id in values:
            if token_id is not None:
                if type(token_id) is not int or token_id < 0:
                    raise ValueError("Invalid native stopping-token ID.")
                if token_id not in result:
                    result.append(token_id)
    token = settings["turn_end_token"]
    if token not in tokenizer.get_vocab():
        raise ValueError(f"Native tokenizer is missing its turn-end token: {token}")
    turn_id = tokenizer.convert_tokens_to_ids(token)
    if type(turn_id) is not int or turn_id < 0 or turn_id == tokenizer.unk_token_id:
        raise ValueError("Native turn-end token resolves to an invalid/unknown ID.")
    if turn_id not in result:
        result.append(turn_id)
    return result
