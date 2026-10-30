from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from icl_pilot.generation.config import compile_experiment_design, load_experiment_config, prepare_colab_generation
from icl_pilot.generation.inference import GenerationRuntime, load_generation_runtime
from icl_pilot.generation.models import load_model_catalog, model_messages, render_prompt, stop_token_ids
from icl_pilot.generation.prompt_records import PromptJournal
from icl_pilot.generation.execution import run_generation


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/generation/experiment.json"
KEYS = ["llama3_1_8b", "qwen3_8b", "gemma2_9b"]
MESSAGES = [{"role": "system", "content": "Shared instruction."},
            {"role": "user", "content": "Literal exemplar xxx and original wording."}]


class Tokens(list):
    def detach(self): return self
    def cpu(self): return self
    def tolist(self): return list(self)
    def __getitem__(self, key):
        value = super().__getitem__(key)
        return Tokens(value) if isinstance(key, slice) else value


class Inputs:
    shape = (1, 3)
    def to(self, device): return self
    def __getitem__(self, index): return Tokens([2, 10, 11])


class NativeTokenizer:
    pad_token_id, eos_token_id, unk_token_id = 0, 1, 99
    init_kwargs = {}
    def __init__(self): self.calls, self.templates = [], []
    def get_vocab(self): return {"<|eot_id|>": 9, "<|im_end|>": 8, "<end_of_turn>": 7}
    def convert_tokens_to_ids(self, token): return self.get_vocab().get(token, 99)
    def apply_chat_template(self, messages, **kwargs):
        assert [m["role"] for m in messages] == ["user"]
        self.templates.append((messages, kwargs))
        return "<bos>" + messages[0]["content"] + "<assistant>"
    def get_chat_template(self): return "native fixture template"
    def __call__(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return {"input_ids": Inputs()}
    def decode(self, ids, **kwargs): return "a fixture narrative ."


def fake_model():
    calls = []
    def generate(**kwargs):
        calls.append(kwargs)
        return [Tokens([2, 10, 11, 20, kwargs["eos_token_id"][-1]])]
    model = SimpleNamespace(device="cuda:0", config=SimpleNamespace(max_position_embeddings=8192),
        generation_config=SimpleNamespace(eos_token_id=[1, 6],
            to_dict=lambda: {"eos_token_id": [1, 6], "top_k": 50, "repetition_penalty": 1.0}),
        generate=generate, calls=calls, eval=lambda: None)
    return model


def test_common_packaging_preserves_all_canonical_instruction_text():
    before = json.dumps(MESSAGES)
    assert model_messages(MESSAGES) == [{"role": "user", "content":
        "Shared instruction.\n\nLiteral exemplar xxx and original wording."}]
    assert json.dumps(MESSAGES) == before


@pytest.mark.parametrize("key", KEYS)
def test_selection_is_pinned_without_changing_shared_sampling(key):
    config = load_experiment_config(CONFIG, model_key=key)
    settings = config["model"]
    assert settings["key"] == key and len(settings["revision"]) == 40
    assert (settings["temperature"], settings["top_p"], settings["top_k"]) == (0.8, 0.9, 0)
    packed, prompt = render_prompt(NativeTokenizer(), MESSAGES, settings)
    assert packed[0]["content"] in prompt
    assert settings["chat_template_kwargs"] == ({"enable_thinking": False} if key == "qwen3_8b" else {})
    stops = stop_token_ids(fake_model(), NativeTokenizer(), settings)
    assert stops[:2] == [1, 6]
    assert stops[-1] == NativeTokenizer().get_vocab()[settings["turn_end_token"]]


def test_unknown_model_rejected_before_output(tmp_path):
    with pytest.raises(ValueError, match="Unknown model key"):
        prepare_colab_generation(CONFIG, ROOT, tmp_path / "run", model_key="unknown")
    assert not (tmp_path / "run").exists()


def test_model_selection_preserves_the_complete_schedule_and_evaluation_split():
    import pandas as pd
    baseline_config, baseline_schedule, baseline_eval, _ = compile_experiment_design(CONFIG)
    for key in KEYS[1:]:
        config, schedule, evaluation, _ = compile_experiment_design(CONFIG, model_key=key)
        pd.testing.assert_frame_equal(schedule, baseline_schedule)
        pd.testing.assert_frame_equal(evaluation, baseline_eval)
        for field in ("temperature", "top_p", "top_k", "quantization", "max_new_tokens_stage1", "max_new_tokens_stage2"):
            assert config["model"][field] == baseline_config["model"][field]


def test_missing_native_turn_end_is_not_silently_mapped_to_unknown():
    settings = load_experiment_config(CONFIG)["model"]
    settings["turn_end_token"] = "missing"
    with pytest.raises(ValueError, match="missing its turn-end"):
        stop_token_ids(fake_model(), NativeTokenizer(), settings)


def test_catalog_rejects_qwen_thinking_mode(tmp_path):
    catalog = json.loads((CONFIG.parent / "models.json").read_text())
    catalog["models"]["qwen3_8b"]["chat_template_kwargs"]["enable_thinking"] = True
    path = tmp_path / "models.json"
    path.write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="disable thinking"):
        load_model_catalog(path)


@pytest.mark.parametrize("key", KEYS)
def test_each_model_runs_its_own_stages_with_explicit_tokens_and_provenance(tmp_path, key):
    import pandas as pd
    out = tmp_path / key
    prepared = prepare_colab_generation(CONFIG, ROOT, out, smoke_test=True, smoke_test_jobs=1, model_key=key)
    model, tokenizer = fake_model(), NativeTokenizer()
    runtime = GenerationRuntime(model, tokenizer, prepared.config["model"], torch=SimpleNamespace(no_grad=nullcontext))
    paths = run_generation(prepared, runtime.generate, run_id=key)
    events = [json.loads(line) for line in (out / "prompt_records.jsonl").read_text().splitlines()]
    prompts = [e for e in events if e["kind"] == "prompt"]
    outputs = [e for e in events if e["kind"] == "output"]
    assert len(model.calls) == 2 and len(prompts) == 2
    assert [p["model_context"]["key"] for p in prompts] == [key, key]
    assert prompts[1]["parent_record_id"] == prompts[0]["record_id"]
    assert prompts[1]["parent_text_sha256"] == outputs[0]["output_text_sha256"]
    assert outputs[0]["output_text"] in prompts[1]["model_messages"][0]["content"]
    assert all(p["generation_parameters"]["top_k"] == 0 for p in prompts)
    assert all(call["max_new_tokens"] == 640 and call["top_k"] == 0 for call in model.calls)
    assert all(call["eos_token_id"] == runtime.stop_ids for call in model.calls)
    assert all(kwargs == {"return_tensors": "pt", "add_special_tokens": False, "truncation": False}
               for _prompt, kwargs in tokenizer.calls)
    assert all(e["finish_reason"] == "end_token" and not e["reached_token_ceiling"] for e in outputs)
    assert [p["rendered_prompt"] for p in prompts] == [prompt for prompt, _kwargs in tokenizer.calls]
    if key == "qwen3_8b":
        assert all(kwargs.get("enable_thinking") is False for _messages, kwargs in tokenizer.templates)
    saved = pd.read_csv(paths[0])
    assert saved["model_key"].tolist() == [key] and saved["model_id"].tolist() == [prepared.config["model"]["id"]]


def test_model_change_refuses_existing_run_without_rewriting_it(tmp_path):
    out = tmp_path / "run"
    prepared = prepare_colab_generation(CONFIG, ROOT, out, smoke_test=True, smoke_test_jobs=1)
    original = prepared.run_manifest_path.read_bytes()
    with pytest.raises(ValueError, match="Run inputs changed"):
        prepare_colab_generation(CONFIG, ROOT, out, smoke_test=True, smoke_test_jobs=1, model_key="qwen3_8b")
    assert prepared.run_manifest_path.read_bytes() == original


def test_context_overflow_fails_without_truncating_or_recording_a_prompt(tmp_path):
    prepared = prepare_colab_generation(CONFIG, ROOT, tmp_path / "run", smoke_test=True, smoke_test_jobs=1)
    row = prepared.run_df.iloc[0]
    model = fake_model()
    model.config.max_position_embeddings = 10
    runtime = GenerationRuntime(model, NativeTokenizer(), prepared.config["model"], torch=SimpleNamespace(no_grad=nullcontext))
    journal = PromptJournal(tmp_path / "test.jsonl", run_id="fixture", builder=prepared.prompt_builder, run_manifest={})
    with pytest.raises(ValueError, match="exceeds this model's context"):
        runtime.generate(prepared.prompt_builder.stage1_messages(row), row=row, stage="stage1_td", journal=journal, max_new_tokens=640)
    assert not model.calls
    assert len(journal.path.read_text().splitlines()) == 1


@pytest.mark.parametrize("ending,reason", [(5, "token_ceiling"), (9, "end_token")])
def test_termination_at_ceiling_records_end_token_separately(tmp_path, ending, reason):
    prepared = prepare_colab_generation(CONFIG, ROOT, tmp_path / "run", smoke_test=True, smoke_test_jobs=1)
    row = prepared.run_df.iloc[0]
    model = fake_model()
    model.generate = lambda **kwargs: [Tokens([2, 10, 11, 20, ending])]
    runtime = GenerationRuntime(model, NativeTokenizer(), prepared.config["model"], torch=SimpleNamespace(no_grad=nullcontext))
    journal = PromptJournal(tmp_path / "test.jsonl", run_id="fixture", builder=prepared.prompt_builder, run_manifest={})
    runtime.generate(prepared.prompt_builder.stage1_messages(row), row=row, stage="stage1_td", journal=journal, max_new_tokens=2)
    output = json.loads(journal.path.read_text().splitlines()[-1])
    assert output["finish_reason"] == reason and output["terminal_token_id"] == ending
    assert output["reached_token_ceiling"] is True


def test_stage2_cannot_change_model_context(tmp_path):
    prepared = prepare_colab_generation(CONFIG, ROOT, tmp_path / "run", smoke_test=True, smoke_test_jobs=1)
    row = prepared.run_df.iloc[0]
    journal = PromptJournal(tmp_path / "test.jsonl", run_id="fixture", builder=prepared.prompt_builder, run_manifest={})
    runtime = GenerationRuntime(fake_model(), NativeTokenizer(), prepared.config["model"], torch=SimpleNamespace(no_grad=nullcontext))
    text, first = runtime.generate(prepared.prompt_builder.stage1_messages(row), row=row, stage="stage1_td", journal=journal, max_new_tokens=640)
    runtime.context["id"] = "different/model"
    with pytest.raises(ValueError, match="parent's model context"):
        runtime.generate(prepared.prompt_builder.stage2_messages(row, text), row=row, stage="stage2_sli", journal=journal,
                         max_new_tokens=640, parent_record_id=first, stage1_text=text)


@pytest.mark.parametrize("key,bf16", [(k, b) for k in KEYS for b in (True, False)])
def test_loader_uses_pinned_official_models_and_single_gpu_nf4(monkeypatch, key, bf16):
    import icl_pilot.generation.inference as inference
    settings = load_experiment_config(CONFIG, model_key=key)["model"]
    calls = []
    tokenizer, model = NativeTokenizer(), fake_model()
    def load_tokenizer(name, **kwargs):
        calls.append(("tokenizer", name, kwargs))
        return tokenizer
    def load_model(name, **kwargs):
        calls.append(("model", name, kwargs))
        return model
    cuda = SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: bf16,
        get_device_name=lambda i: "fixture GPU", get_device_properties=lambda i: SimpleNamespace(total_memory=16000000000))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda, bfloat16="bf16", float16="fp16"))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        __version__="4.57.1",
        AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model), BitsAndBytesConfig=lambda **kw: kw))
    monkeypatch.setattr(inference.importlib.metadata, "version", lambda name: "4.57.1")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    runtime, metadata = load_generation_runtime(settings)
    assert [c[1] for c in calls] == [settings["id"], settings["id"]]
    assert all(c[2]["revision"] == settings["revision"] for c in calls)
    kwargs = calls[1][2]
    assert kwargs["device_map"] == {"": 0}
    assert kwargs["quantization_config"]["bnb_4bit_quant_type"] == "nf4"
    assert kwargs["torch_dtype"] == ("bf16" if bf16 else "fp16")
    assert metadata["effective_compute_dtype"] == ("bfloat16" if bf16 else "float16")
    assert metadata["native_stop_token_ids"] == runtime.stop_ids
    assert metadata["checkpoint_generation_config"]["top_k"] == 50
    assert metadata["decoding_overrides"]["top_k"] == 0
