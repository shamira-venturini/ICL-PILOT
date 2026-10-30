"""Single-GPU loading and traceable two-stage inference for three text families."""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
import platform

from .models import TRANSFORMERS_VERSION, render_prompt, stop_token_ids


class GenerationRuntime:
    def __init__(self, model, tokenizer, settings: dict, *, torch):
        self.model, self.tokenizer, self.torch = model, tokenizer, torch
        self.settings = copy.deepcopy(settings)
        self.stop_ids = stop_token_ids(model, tokenizer, settings)
        self.base_generation_config = copy.deepcopy(model.generation_config.to_dict())
        self.context = {name: settings[name] for name in ("key", "id", "revision", "instruction_packaging")}
        self.context["chat_template_kwargs"] = copy.deepcopy(settings["chat_template_kwargs"])
        self.context["checkpoint_generation_config_sha256"] = hashlib.sha256(
            json.dumps(self.base_generation_config, sort_keys=True).encode()).hexdigest()

    def render(self, messages: list) -> tuple[list, str]:
        return render_prompt(self.tokenizer, messages, self.settings)

    def generation_parameters(self, max_new_tokens):
        parameters = {name: self.settings[name] for name in ("do_sample", "temperature", "top_p", "top_k")}
        parameters.update(max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.pad_token_id,
                          eos_token_id=self.stop_ids)
        return parameters

    def generate(self, messages, *, max_new_tokens, row, stage, journal,
                 parent_record_id=None, stage1_text=None):
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("Supply a positive output-token ceiling.")
        packed, prompt = self.render(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False,
                                truncation=False)
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        count = inputs["input_ids"].shape[1]
        context_limit = getattr(self.model.config, "max_position_embeddings", None)
        if context_limit is not None and count + max_new_tokens > context_limit:
            raise ValueError("Prompt plus output ceiling exceeds this model's context window; no text was truncated.")
        parameters = self.generation_parameters(max_new_tokens)
        record_id = journal.record_prompt(
            row=row, stage=stage, messages=messages, model_messages=packed,
            model_context=self.context, rendered_prompt=prompt,
            input_token_ids=inputs["input_ids"][0].detach().cpu().tolist(),
            generation_parameters=parameters, parent_record_id=parent_record_id,
            stage1_text=stage1_text,
        )
        with self.torch.no_grad():
            outputs = self.model.generate(**inputs, **parameters)
        ids = outputs[0][count:]
        ids_list = ids.detach().cpu().tolist()
        stopped = bool(ids_list and ids_list[-1] in self.stop_ids)
        finish_reason = ("end_token" if stopped else
                         "token_ceiling" if len(ids_list) >= max_new_tokens else "other")
        text = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
        journal.record_output(record_id, text, generated_token_count=len(ids_list),
                              finish_reason=finish_reason,
                              terminal_token_id=ids_list[-1] if ids_list else None,
                              reached_token_ceiling=len(ids_list) >= max_new_tokens)
        return text, record_id


def load_generation_runtime(settings: dict) -> tuple[GenerationRuntime, dict]:
    """Load one pinned official checkpoint; never download three models together."""
    if importlib.metadata.version("transformers") != TRANSFORMERS_VERSION:
        raise RuntimeError(f"Install transformers=={TRANSFORMERS_VERSION}, restart the kernel and rerun setup.")
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError("The kernel still has an older Transformers import; restart it and rerun setup.")
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a GPU runtime in Colab before loading a generation model.")
    quantization = settings["quantization"]
    requested_dtype = quantization["compute_dtype"]
    effective_dtype = requested_dtype
    if requested_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        effective_dtype = "float16"
        print("This GPU does not support bfloat16; using float16 computation (recorded in the manifest).")
    dtype = getattr(torch, effective_dtype)
    auth = {"token": os.environ["HF_TOKEN"]} if os.environ.get("HF_TOKEN") else {}
    try:
        tokenizer = AutoTokenizer.from_pretrained(settings["id"], revision=settings["revision"], **auth)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer has no usable padding/EOS token.")
        kwargs = {"revision": settings["revision"], "device_map": {"": 0},
                  "torch_dtype": dtype, **auth}
        if quantization["load_in_4bit"]:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type=quantization["quant_type"],
                bnb_4bit_compute_dtype=dtype)
        model = AutoModelForCausalLM.from_pretrained(settings["id"], **kwargs)
    except OSError as error:
        raise RuntimeError(f"Cannot load pinned {settings['id']}. For Llama/Gemma, accept the official "
                           "Hugging Face access terms and authenticate in this Colab runtime; "
                           "also check network access and available disk space.") from error
    model.eval()
    runtime = GenerationRuntime(model, tokenizer, settings, torch=torch)
    metadata = {
        "model_key": settings["key"], "model_id": settings["id"],
        "requested_revision": settings["revision"],
        "tokenizer_id": settings["id"], "requested_tokenizer_revision": settings["revision"],
        "tokenizer_class": type(tokenizer).__name__,
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
        "resolved_tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"),
        "instruction_packaging": settings["instruction_packaging"],
        "chat_template_kwargs": settings["chat_template_kwargs"],
        "native_chat_template_sha256": hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest(),
        "native_stop_token_ids": runtime.stop_ids, "pad_token_id": tokenizer.pad_token_id,
        "checkpoint_generation_config": runtime.base_generation_config,
        "effective_generation_config_by_stage": {
            stage: {**runtime.base_generation_config, **runtime.generation_parameters(settings[ceiling])}
            for stage, ceiling in (("stage1_td", "max_new_tokens_stage1"), ("stage2_sli", "max_new_tokens_stage2"))},
        "decoding_overrides": {name: settings[name] for name in ("do_sample", "temperature", "top_p", "top_k")},
        "other_decoding_controls": "checkpoint/library defaults, recorded above; common policy still pending",
        "requested_compute_dtype": requested_dtype, "effective_compute_dtype": effective_dtype,
        "loaded_model_dtype": str(getattr(model, "dtype", dtype)),
        "python_version": platform.python_version(), "platform": platform.platform(),
        "numpy_version": importlib.metadata.version("numpy"),
        "cuda_runtime_version": getattr(getattr(torch, "version", None), "cuda", None),
        "cudnn_version": torch.backends.cudnn.version() if hasattr(torch, "backends") else None,
        "quantization": quantization, "device_map": {"": 0},
        "attention_implementation": getattr(model.config, "_attn_implementation", None),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "tokenizers", "accelerate", "bitsandbytes", "huggingface_hub")},
    }
    return runtime, metadata
