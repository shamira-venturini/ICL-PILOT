from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from icl_pilot.generation.prompts import load_prompt_builder
from icl_pilot.generation.prompt_records import PromptJournal


ROOT = Path(__file__).resolve().parents[1]
ROW = {
    "cohort": "4-year-old", "round_id": "round_01", "bundle_id": "pair_sli444_td410_rep01",
    "pair_id": "pair_sli444_td410", "replicate_id": 1, "story_slot_order": 1,
    "target_story": "A1", "story_title": "Ball in the pool", "td_child_id": 410, "sli_child_id": 444,
    "td_age": "4;01.27", "sli_age": "4;02.04", "prompt_td_story_e1": "B1", "prompt_sli_story_e2": "B3",
    "story_units_text": "1. A ball falls.\n2. It is retrieved.",
    "content_invariants_text": "- pool\n- ball", "optional_content_text": "None",
    "prompt_constraints_text": "- Keep the events in order.",
    "prompt_td_exemplar_story": "the dog said {hello} .", "prompt_sli_exemplar_story": "her runned (.) home .",
}
PARENT = "he got the ball .\n{unaltered parent}"


@pytest.fixture
def builder():
    return load_prompt_builder(ROOT / "configs/generation/experiment.json")


@pytest.fixture
def journal(builder, tmp_path: Path):
    return PromptJournal(tmp_path / "prompts.jsonl", run_id="fixture", builder=builder, run_manifest={})


def record(journal, stage="stage1_td", row=None, parent_record_id=None, parent_text=None, messages=None):
    row = ROW if row is None else row
    if messages is None:
        messages = (journal.builder.stage1_messages(row) if stage == "stage1_td"
                    else journal.builder.stage2_messages(row, parent_text))
    return journal.record_prompt(row=row, stage=stage, messages=messages,
                                 rendered_prompt="exact fixture chat text", input_token_ids=[1, 42, 3],
                                 generation_parameters={"max_new_tokens": 180},
                                 parent_record_id=parent_record_id, stage1_text=parent_text)


def events(journal):
    return [json.loads(line) for line in journal.path.read_text().splitlines()]


def test_templates_match_pre_cleanup_notebook_messages_byte_for_byte(builder) -> None:
    # Frozen hashes come from the functions in notebook revision 78a5cada.
    # Matching source does not establish the actual historical execution.
    for messages, expected in ((builder.stage1_messages(ROW), "8b0ca2a2665ced6b0d44833c970ee393fe6c59800a00e7a5da56432663b83c54"),
                               (builder.stage2_messages(ROW, PARENT), "d31377f55a5810774b93392e285d852e1de1e2e9f55036b7595d6015bf4248d9")):
        serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
        assert hashlib.sha256(serialized.encode()).hexdigest() == expected
    assert builder.stage1_messages(SimpleNamespace(**ROW)) == builder.stage1_messages(ROW)


def test_story_reference_matches_pre_cleanup_scaffold_byte_for_byte():
    # Includes the inserted style constraints, which are part of the prompt.
    raw = (ROOT / "configs/generation/story_unit_reference.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "e1cd483eb249e5ce80169cd73b43dee6456a104b0e0c1789af75c874647b353e")


@pytest.mark.parametrize("story", ["A1", "A2", "A3", "B1", "B2", "B3"])
def test_prompts_keep_actual_story_content_exemplars_and_stage1_parent_literal(builder, story):
    reference = json.loads((ROOT / "configs/generation/story_unit_reference.json").read_text())[story]
    slots = ("B1", "B3") if story.startswith("A") else ("A1", "A2")
    row = {**ROW, "target_story": story, "story_title": reference["title"],
           "prompt_td_story_e1": slots[0], "prompt_sli_story_e2": slots[1],
           "story_units_text": "\n".join(f"{i + 1}. {text}" for i, text in enumerate(reference["units"])),
           "prompt_constraints_text": "\n".join("- " + text for text in reference["prompt_constraints"]),
           "prompt_td_exemplar_story": "the child goed to an adult . {literal example}",
           "prompt_sli_exemplar_story": "the adult help the child . {literal example}"}
    first, second = builder.stage1_messages(row), builder.stage2_messages(row, PARENT)
    assert row["story_units_text"] in first[1]["content"] and row["story_units_text"] in second[1]["content"]
    assert row["prompt_constraints_text"] in first[1]["content"]
    assert row["prompt_td_exemplar_story"] in first[1]["content"]
    assert row["prompt_sli_exemplar_story"] in second[1]["content"]
    assert PARENT in second[1]["content"]
    assert "Return only the transformed narrative text." in second[1]["content"]


def test_prompt_restoration_preserves_all_canonical_story_content():
    reference = json.loads((ROOT / "configs/generation/story_unit_reference.json").read_text())
    content = {story: {key: value for key, value in record.items() if key not in {"notes", "prompt_constraints"}}
               for story, record in reference.items()}
    assert hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest() == (
        "26a12b10750b949c4a205e50782624039c8a624c524aea6a993746287d1405dd")


@pytest.mark.parametrize("parent", [None, "", "   "])
def test_stage2_requires_a_real_nonempty_parent(builder, parent) -> None:
    with pytest.raises(ValueError, match="actual nonempty Stage 1"):
        builder.stage2_messages(ROW, parent)


def test_builder_rejects_same_set_exemplars(builder) -> None:
    with pytest.raises(ValueError, match="opposite story set"):
        builder.stage1_messages({**ROW, "prompt_td_story_e1": "A2"})


@pytest.mark.parametrize("change", ["unknown_field", "missing_parent", "path_escape"])
def test_invalid_template_definitions_are_rejected(tmp_path: Path, change: str) -> None:
    source = ROOT / "configs/generation"
    config = json.loads((source / "experiment.json").read_text())
    (tmp_path / "prompts").mkdir()
    for stage in ("stage1", "stage2"):
        (tmp_path / "prompts" / (stage + ".json")).write_bytes((source / "prompts" / (stage + ".json")).read_bytes())
    path = tmp_path / "prompts/stage2.json"
    template = json.loads(path.read_text())
    if change == "unknown_field":
        template["messages"][1]["lines"].append("{target_story.__class__}")
    elif change == "missing_parent":
        template["messages"][1]["lines"].remove("{stage1_text}")
    else:
        config["prompt_templates"]["stage2_sli"] = "../stage2.json"
    path.write_text(json.dumps(template))
    config_path = tmp_path / "experiment.json"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_prompt_builder(config_path)


def test_actual_prompt_output_and_parent_link_are_recorded(journal) -> None:
    first = record(journal)
    journal.record_output(first, PARENT, generated_token_count=8)
    second = record(journal, "stage2_sli", parent_record_id=first, parent_text=PARENT)
    journal.record_output(second, "her got ball .", generated_token_count=4)
    saved = events(journal)
    assert [r["kind"] for r in saved] == ["run", "prompt", "output", "prompt", "output"]
    prompt = saved[3]
    assert prompt["record_id"] == second
    assert prompt["parent_record_id"] == first
    assert prompt["parent_text_sha256"] == hashlib.sha256(PARENT.encode()).hexdigest()
    assert prompt["messages"] == journal.builder.stage2_messages(ROW, PARENT)
    assert prompt["rendered_prompt"] == "exact fixture chat text"
    assert prompt["input_token_count"] == 3
    assert prompt["td_age"] == ROW["td_age"]
    assert saved[4]["output_text"] == "her got ball ."


def test_stage2_rejects_an_unfinished_parent(journal) -> None:
    first = record(journal)
    with pytest.raises(ValueError, match="completed Stage 1 parent"):
        record(journal, "stage2_sli", parent_record_id=first, parent_text=PARENT)
    assert len(events(journal)) == 2


@pytest.mark.parametrize("change", ["wrong_text", "wrong_story", "wrong_replicate", "wrong_round"])
def test_stage2_rejects_mismatched_parent_ancestry(journal, change: str) -> None:
    first = record(journal)
    journal.record_output(first, PARENT, generated_token_count=8)
    row, parent_text = dict(ROW), PARENT
    if change == "wrong_text":
        parent_text = "another generated story ."
    elif change == "wrong_story":
        row["target_story"] = "A2"
    elif change == "wrong_round":
        row["round_id"] = "round_02"
    else:
        row["replicate_id"], row["bundle_id"] = 2, "pair_sli444_td410_rep02"
    with pytest.raises(ValueError):
        record(journal, "stage2_sli", row=row, parent_record_id=first, parent_text=parent_text)
    assert len(events(journal)) == 3


def test_round_is_part_of_job_identity(journal) -> None:
    first = record(journal)
    second = record(journal, row={**ROW, "round_id": "round_02"})
    prompts = [r for r in events(journal) if r["kind"] == "prompt"]
    assert first != second
    assert prompts[0]["job_id"] != prompts[1]["job_id"]


def test_duplicate_or_altered_prompt_records_are_rejected(journal) -> None:
    altered = journal.builder.stage1_messages(ROW)
    altered[1]["content"] += "changed instruction"
    with pytest.raises(ValueError, match="configured prompt builder"):
        record(journal, messages=altered)
    first = record(journal)
    with pytest.raises(ValueError, match="already recorded"):
        record(journal)
    journal.record_output(first, "story .", generated_token_count=2)
    with pytest.raises(ValueError, match="unfinished prompt"):
        journal.record_output(first, "replacement .", generated_token_count=2)
    assert len(events(journal)) == 3


def test_existing_prompt_journal_is_preserved(journal) -> None:
    original = journal.path.read_bytes()
    with pytest.raises(FileExistsError):
        PromptJournal(journal.path, run_id="replacement", builder=journal.builder, run_manifest={})
    assert journal.path.read_bytes() == original


class Tokens(list):
    def detach(self): return self
    def cpu(self): return self
    def tolist(self): return list(self)
    def numel(self): return len(self)
    def __getitem__(self, value):
        result = super().__getitem__(value)
        return Tokens(result) if isinstance(value, slice) else result


class Batch:
    shape = (1, 3)
    def to(self, device): return self
    def __getitem__(self, index): return Tokens([1, 42, 3])


class Tokenizer:
    pad_token_id, eos_token_id = 0, 3
    def __init__(self): self.used_prompts = []
    def apply_chat_template(self, messages, **kwargs): return json.dumps(messages) + "<assistant>"
    def __call__(self, prompt, **kwargs):
        self.used_prompts.append(prompt)
        return {"input_ids": Batch()}
    def decode(self, ids, **kwargs): return "  fixture story .\n"


def inference_namespace(journal):
    model = SimpleNamespace(device="cpu", generate=lambda **kwargs: [Tokens([1, 42, 3, 10, 11])])
    namespace = {"prompt_journal": journal, "torch": SimpleNamespace(no_grad=nullcontext),
                 "tokenizer": Tokenizer(), "model": model, "DO_SAMPLE": True,
                 "TEMPERATURE": 0.7, "TOP_P": 0.95}
    notebook = json.loads((ROOT / "notebooks/colab_direct_generation_pipeline.ipynb").read_text())
    source = next(cell for cell in notebook["cells"] if cell.get("id") == "generation-function")
    exec("".join(source["source"]), namespace)
    return namespace


def test_notebook_records_the_chat_text_it_actually_tokenizes(journal) -> None:
    namespace = inference_namespace(journal)
    generate = namespace["generate_from_messages"]
    text, first = generate(journal.builder.stage1_messages(ROW), row=ROW, stage="stage1_td", journal=journal)
    assert text == "fixture story ."
    _text, second = generate(journal.builder.stage2_messages(ROW, text), row=ROW, stage="stage2_sli",
                             parent_record_id=first, stage1_text=text, journal=journal)
    prompts = [r for r in events(journal) if r["kind"] == "prompt"]
    assert [r["rendered_prompt"] for r in prompts] == namespace["tokenizer"].used_prompts
    assert prompts[1]["record_id"] == second and prompts[1]["parent_record_id"] == first
    assert prompts[0]["generation_parameters"]["top_p"] == 0.95


def test_model_failure_keeps_the_attempted_prompt_without_fabricating_an_output(journal) -> None:
    namespace = inference_namespace(journal)
    def fail(**kwargs): raise RuntimeError("fixture model failure")
    namespace["model"].generate = fail
    with pytest.raises(RuntimeError, match="fixture model failure"):
        namespace["generate_from_messages"](journal.builder.stage1_messages(ROW), row=ROW, stage="stage1_td", journal=journal)
    assert [r["kind"] for r in events(journal)] == ["run", "prompt"]


def test_notebook_loop_exports_prompt_links_and_includes_records_in_the_archive(builder, tmp_path: Path, monkeypatch) -> None:
    import pandas as pd
    import zipfile

    # Model/tokenizer are fixtures; no CUDA, Drive or downloads.
    out = tmp_path / "run"
    out.mkdir()
    manifest = out / "run_manifest.json"
    manifest.write_text("{}")
    (out / "pipeline_manifest.json").write_text('{"stages": {}}')
    builder.snapshot(out / "generation_design/prompts")
    namespace = inference_namespace(None)
    namespace.update({
        "RUN_OUTPUT_DIR": out, "RUN_ID": "fixture", "json": json, "SAVE_EVERY": 1,
        "RUN_GENERATION": True, "persist_run": lambda: None, "MAKE_ARCHIVE": True,
        "DOWNLOAD_ARCHIVE": False, "RELEASE_COLAB_RUNTIME": False, "IS_COLAB": False,
        "prepared": SimpleNamespace(run_manifest_path=manifest, prompt_builder=builder,
            config={"model": {"max_new_tokens_stage1": 180, "max_new_tokens_stage2": 180}},
            run_df=pd.DataFrame([{**ROW, "cohort_label": ROW["cohort"]}])),
    })
    from icl_pilot.pipeline import record_stage, package_run
    namespace.update(record_stage=record_stage, package_run=package_run)
    notebook = json.loads((ROOT / "notebooks/colab_direct_generation_pipeline.ipynb").read_text())
    source = "".join(next(c for c in notebook["cells"] if c.get("id") == "generation-run")["source"])
    exec(source, namespace)
    saved = pd.read_csv(next(out.glob("4*story_generations.csv"))).iloc[0]
    prompts = [r for r in events(SimpleNamespace(path=out / "prompt_records.jsonl")) if r["kind"] == "prompt"]
    assert saved["stage1_prompt_record_id"] == prompts[0]["record_id"]
    assert saved["stage2_prompt_record_id"] == prompts[1]["record_id"]
    assert prompts[1]["parent_record_id"] == saved["stage1_prompt_record_id"]
    export = next(c for c in notebook["cells"] if c.get("id") == "export")
    exec("".join(export["source"]), namespace)
    with zipfile.ZipFile(namespace["archive_path"]) as archive:
        assert "prompt_records.jsonl" in archive.namelist()
        assert "generation_design/prompts/stage1_td.json" in archive.namelist()
        assert all(not name.startswith("/") for name in archive.namelist())


def test_generation_interruption_retains_stage1_parent_and_requires_a_fresh_run(builder, tmp_path):
    import pandas as pd
    from icl_pilot.generation.execution import run_generation
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text("{}")
    prepared = SimpleNamespace(run_manifest_path=manifest, prompt_builder=builder,
        config={"model": {"max_new_tokens_stage1": 180, "max_new_tokens_stage2": 180}},
        run_df=pd.DataFrame([{**ROW, "cohort_label": ROW["cohort"]}]))
    namespace = inference_namespace(None)
    count = [0]
    def generate(**kwargs):
        count[0] += 1
        if count[0] == 2:
            raise RuntimeError("fixture Stage 2 failure")
        return [Tokens([1, 42, 3, 10, 11])]
    namespace["model"].generate = generate
    with pytest.raises(RuntimeError, match="Stage 2 failure"):
        run_generation(prepared, namespace["generate_from_messages"], run_id="interrupted")
    records = events(SimpleNamespace(path=tmp_path / "prompt_records.jsonl"))
    assert [r["kind"] for r in records] == ["run", "prompt", "output", "prompt"]
    assert records[-1]["parent_record_id"] == records[1]["record_id"]
    state = json.loads(manifest.read_text())
    assert state["status"].startswith("generation_interrupted")
    assert state["completed_generation_jobs"] == 0
    with pytest.raises(FileExistsError, match="resume is not implemented"):
        run_generation(prepared, namespace["generate_from_messages"], run_id="interrupted")
