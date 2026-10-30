"""Small stochastic interruption checks; no model weights or GPU are needed."""
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from icl_pilot.generation.execution import record_runtime_session, run_generation
from icl_pilot.generation.prompt_records import PromptJournal
from icl_pilot.generation.prompts import load_prompt_builder
from icl_pilot.generation.randomness import RandomStreams, initialise_run_randomness
from icl_pilot.generation.recovery import (CHECKPOINT, JOURNAL, read_checkpoint, recover_journal,
                                          sync_generation_checkpoint)
from test_generation_prompts import ROW, record
from test_generation_config import experiment

ROOT = Path(__file__).resolve().parents[1]


class FakeCUDA:
    """Exercise the all-device RNG API using an independent CPU generator."""
    def __init__(self):
        self.generator = torch.Generator()
    def is_available(self): return True
    def device_count(self): return 1
    def manual_seed_all(self, seed): self.generator.manual_seed(seed)
    def get_rng_state_all(self): return [self.generator.get_state()]
    def set_rng_state_all(self, states): self.generator.set_state(states[0])


@pytest.fixture
def streams():
    backend = SimpleNamespace(cuda=FakeCUDA(), manual_seed=torch.manual_seed, get_rng_state=torch.get_rng_state,
                              set_rng_state=torch.set_rng_state, tensor=torch.tensor, uint8=torch.uint8)
    initialise_run_randomness(torch=backend)
    return RandomStreams(torch=backend)


def draw(streams):
    return [random.random(), float(np.random.random()), float(torch.rand(1)),
            float(torch.rand(1, generator=streams.torch.cuda.generator))]


def prepared_run(folder, jobs=2):
    folder.mkdir()
    manifest = folder / "run_manifest.json"
    manifest.write_text('{"input_sha256": {"fixture": "fixed-inputs"}}')
    return SimpleNamespace(run_manifest_path=manifest,
        prompt_builder=load_prompt_builder(ROOT / "configs/generation/experiment.json"),
        config={"master_seed": 42, "seed_policy": "run_level",
                "model": {"max_new_tokens_stage1": 640, "max_new_tokens_stage2": 640}},
        run_df=pd.DataFrame([{**ROW, "cohort_label": ROW["cohort"], "target_story": "A" + str(i + 1),
                              "story_slot_order": i + 1} for i in range(jobs)]))


def generator(streams, *, fail_call=None, seen=None):
    def generate(messages, *, row, stage, journal, parent_record_id=None, stage1_text=None, **kwargs):
        if seen is not None:
            seen.append((row.target_story, stage, stage1_text))
        record_id = record(journal, stage=stage, row=row, parent_record_id=parent_record_id,
                           parent_text=stage1_text, messages=messages)
        numbers = draw(streams)
        if journal.active_call["call_index"] == fail_call:
            raise RuntimeError("simulated interruption during generation")
        text = "literal narrative\n" + json.dumps(numbers)
        journal.record_output(record_id, text, generated_token_count=20)
        return text, record_id
    return generate


def outputs(folder):
    return [event for event in events(folder) if event["kind"] == "output"]


def events(folder):
    return [json.loads(line) for line in (folder / JOURNAL).read_text().splitlines()]


def test_json_round_trip_restores_python_numpy_cpu_and_cuda(streams):
    random.gauss(0, 1)
    np.random.normal()
    saved = json.loads(json.dumps(streams.capture()))
    expected = draw(streams)
    draw(streams)
    streams.restore(saved)
    assert draw(streams) == expected
    assert len(saved["torch_cuda"]) == 1


@pytest.mark.parametrize("fail_call", [1, 2, 3, 4])
def test_resume_keeps_completed_outputs_order_and_exact_parent(tmp_path, streams, fail_call):
    reference = prepared_run(tmp_path / "reference")
    run_generation(reference, generator(streams), run_id="reference", random_streams=streams)
    expected = [event["output_text"] for event in outputs(tmp_path / "reference")]

    initialise_run_randomness(torch=streams.torch)  # A different fresh model run.
    interrupted = prepared_run(tmp_path / "interrupted")
    folder = interrupted.run_manifest_path.parent
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_generation(interrupted, generator(streams, fail_call=fail_call), run_id="interrupted", random_streams=streams)
    prefix = (folder / JOURNAL).read_bytes()
    completed = outputs(folder)
    assert len(completed) == fail_call - 1
    checkpoint = read_checkpoint(folder)
    assert checkpoint["next_call"]["call_index"] == fail_call
    assert checkpoint["master_seed"] == 42 and checkpoint["seed_policy"] == "run_level"

    # Model loading may consume all the global streams. Restoration is inside
    # run_generation, after that load and immediately before the pending call.
    for _ in range(4): draw(streams)
    seen = []
    run_generation(interrupted, generator(streams, seen=seen), run_id="interrupted", resume=True, random_streams=streams)
    assert (folder / JOURNAL).read_bytes().startswith(prefix)
    actual = outputs(folder)
    assert [event["output_text"] for event in actual] == expected
    assert actual[:len(completed)] == completed
    assert [event["execution_call"]["call_index"] for event in actual] == [1, 2, 3, 4]
    assert len({event["record_id"] for event in actual}) == 4
    assert len(seen) == 5 - fail_call
    if fail_call == 2:
        assert seen[0] == ("A1", "stage2_sli", completed[0]["output_text"])
    saved = pd.read_csv(folder / "all_cohorts_rotating_story_generations.csv")
    assert len(saved) == 2 and saved["target_story"].tolist() == ["A1", "A2"]
    assert "generation_seed" not in saved and saved["master_seed"].eq(42).all()
    assert saved["seed_policy"].eq("run_level").all()
    assert saved["stage1_text_sha256"].equals(saved["stage2_parent_text_sha256"])
    # A completed resume rebuilds caches without model calls or duplicate output.
    def forbidden(*args, **kwargs): raise AssertionError("no new inference allowed")
    run_generation(interrupted, forbidden, run_id="interrupted", resume=True, random_streams=streams)
    assert outputs(folder) == actual


@pytest.mark.parametrize("torn", [False, True])
def test_checkpoint_committed_before_output_line_recovers_without_regeneration(tmp_path, streams, monkeypatch, torn):
    prepared = prepared_run(tmp_path / "run")
    folder = prepared.run_manifest_path.parent
    append = PromptJournal._append
    def crash(self, event):
        if event["kind"] == "output" and event["execution_call"]["call_index"] == 1:
            if torn:
                with self.path.open("ab") as handle: handle.write(b'{"recorded_at_utc":')
            raise OSError("simulated output append failure")
        return append(self, event)
    monkeypatch.setattr(PromptJournal, "_append", crash)
    with pytest.raises(OSError):
        run_generation(prepared, generator(streams), run_id="run", random_streams=streams)
    assert read_checkpoint(folder)["completed_calls"] == 1
    monkeypatch.setattr(PromptJournal, "_append", append)
    seen = []
    run_generation(prepared, generator(streams, seen=seen), run_id="run", resume=True, random_streams=streams)
    assert len(seen) == 3 and seen[0][1] == "stage2_sli"
    assert len(outputs(folder)) == 4
    assert bool(list((folder / "journal_recovery").glob("*.partial"))) == torn


def test_checkpoint_write_failure_does_not_commit_an_output(tmp_path, streams, monkeypatch):
    from icl_pilot.generation import recovery
    prepared = prepared_run(tmp_path / "run")
    atomic = recovery.atomic_json
    def fail(path, value):
        if path.name == CHECKPOINT and value["completed_calls"] == 2:
            raise OSError("checkpoint storage failed")
        return atomic(path, value)
    monkeypatch.setattr(recovery, "atomic_json", fail)
    with pytest.raises(OSError):
        run_generation(prepared, generator(streams), run_id="run", random_streams=streams)
    assert read_checkpoint(prepared.run_manifest_path.parent)["completed_calls"] == 1
    monkeypatch.setattr(recovery, "atomic_json", atomic)
    run_generation(prepared, generator(streams), run_id="run", resume=True, random_streams=streams)
    assert len(outputs(prepared.run_manifest_path.parent)) == 4


def test_mirror_is_recoverable_at_each_boundary_and_on_torn_output(tmp_path, streams):
    prepared = prepared_run(tmp_path / "run")
    target = tmp_path / "drive" / "run"
    counts = []
    def mirror(source):
        sync_generation_checkpoint(source, target)
        saved = read_checkpoint(target)
        recover_journal(target, saved)
        counts.append(saved["completed_calls"])
    run_generation(prepared, generator(streams), run_id="run", random_streams=streams, on_checkpoint=mirror)
    assert counts == [0, 1, 2, 3, 4]
    assert outputs(target) == outputs(prepared.run_manifest_path.parent)
    checkpoint = read_checkpoint(target)
    with (target / JOURNAL).open("r+b") as handle: handle.truncate(checkpoint["journal_prefix_bytes"] + 10)
    sync_generation_checkpoint(prepared.run_manifest_path.parent, target)
    recover_journal(target, read_checkpoint(target))
    assert len(outputs(target)) == 4


@pytest.mark.parametrize("change", ["order", "input", "run_id", "checksum"])
def test_resume_rejects_conflicting_identity_without_overwriting(tmp_path, streams, change):
    prepared = prepared_run(tmp_path / "run")
    folder = prepared.run_manifest_path.parent
    with pytest.raises(RuntimeError):
        run_generation(prepared, generator(streams, fail_call=2), run_id="run", random_streams=streams)
    if change == "order": prepared.run_df = prepared.run_df.iloc[::-1]
    if change == "input":
        manifest = json.loads(prepared.run_manifest_path.read_text())
        manifest["input_sha256"]["fixture"] = "changed"
        prepared.run_manifest_path.write_text(json.dumps(manifest))
    if change == "checksum":
        checkpoint = json.loads((folder / CHECKPOINT).read_text())
        checkpoint["completed_calls"] = 999
        (folder / CHECKPOINT).write_text(json.dumps(checkpoint))
    before = {name: (folder / name).read_bytes() for name in (CHECKPOINT, JOURNAL)}
    with pytest.raises(ValueError):
        run_generation(prepared, generator(streams), run_id="other" if change == "run_id" else "run", resume=True, random_streams=streams)
    assert before == {name: (folder / name).read_bytes() for name in before}


def test_legacy_outputs_are_not_given_fabricated_random_states(tmp_path, streams):
    prepared = prepared_run(tmp_path / "run")
    folder = prepared.run_manifest_path.parent
    (folder / JOURNAL).write_text('{"legacy": true}\n')
    with pytest.raises(FileNotFoundError, match="No RNG checkpoint"):
        run_generation(prepared, generator(streams), run_id="run", resume=True, random_streams=streams)
    assert (folder / JOURNAL).read_text() == '{"legacy": true}\n'
    assert not (folder / CHECKPOINT).exists()


def test_runtime_sessions_preserve_original_environment(tmp_path):
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text("{}")
    first = record_runtime_session(manifest, {"gpu": "A100", "precision": "bfloat16"})
    second = record_runtime_session(manifest, {"gpu": "T4", "precision": "float16"}, resume=True)
    saved = json.loads(manifest.read_text())
    assert saved["model_runtime"]["gpu"] == "A100"
    assert [session["session_id"] for session in saved["runtime_sessions"]] == [first, second]
    assert saved["runtime_sessions"][1]["resumed"]


def test_changed_cuda_device_count_is_rejected_before_restoration(streams):
    saved = streams.capture()
    saved["torch_cuda"] = []
    with pytest.raises(ValueError, match="CUDA device count"):
        streams.restore(saved)


@pytest.mark.parametrize("change", [None, "design", "model", "selection", "decoding"])
def test_portable_setup_checks_original_design_before_resuming(experiment, tmp_path, streams, change):
    from icl_pilot.generation.config import prepare_colab_generation
    folder = tmp_path / "run"
    prepared = prepare_colab_generation(experiment, tmp_path, folder, smoke_test=True, smoke_test_jobs=2)
    with pytest.raises(RuntimeError):
        run_generation(prepared, generator(streams, fail_call=2), run_id="run", random_streams=streams)
    original = prepared.run_manifest_path.read_bytes()
    kwargs = {"smoke_test": True, "smoke_test_jobs": 2, "resume": True}
    if change == "design":
        (folder / "generation_design/execution_call_order.csv").write_text("changed")
    if change == "model": kwargs["model_key"] = "qwen3_8b"
    if change == "selection": kwargs["smoke_test_jobs"] = 1
    if change == "decoding":
        config = json.loads(experiment.read_text())
        config["model"]["temperature"] = 0.81
        experiment.write_text(json.dumps(config))
    if change is not None:
        with pytest.raises(ValueError): prepare_colab_generation(experiment, tmp_path, folder, **kwargs)
        assert prepared.run_manifest_path.read_bytes() == original
    else:
        restored = prepare_colab_generation(experiment, tmp_path, folder, **kwargs)
        assert restored.run_manifest_path.read_bytes() == original
        assert restored.artifacts.cohort_packages == prepared.artifacts.cohort_packages
        run_generation(restored, generator(streams), run_id="run", resume=True, random_streams=streams)
        assert len(outputs(folder)) == 4
        order = pd.read_csv(folder / "generation_design/execution_call_order.csv")
        assert order["call_index"].tolist() == [1, 2, 3, 4]
        schedule = pd.read_csv(folder / "generation_design/execution_schedule.csv")
        assert "generation_seed" not in schedule and schedule["replicate_id"].tolist() == [1, 1]
        assert schedule["master_seed"].eq(42).all() and schedule["seed_policy"].eq("run_level").all()


def test_no_reseeding_inside_generation_or_resume(tmp_path, streams, monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError("per-call reseeding is forbidden")
    monkeypatch.setattr(random, "seed", forbidden)
    monkeypatch.setattr(np.random, "seed", forbidden)
    monkeypatch.setattr(torch, "manual_seed", forbidden)
    monkeypatch.setattr(streams.torch.cuda, "manual_seed_all", forbidden)
    prepared = prepared_run(tmp_path / "run")
    with pytest.raises(RuntimeError):
        run_generation(prepared, generator(streams, fail_call=2), run_id="run", random_streams=streams)
    run_generation(prepared, generator(streams), run_id="run", resume=True, random_streams=streams)


def test_real_cpu_state_without_cuda_round_trips():
    backend = RandomStreams()
    saved = backend.capture()
    expected = [random.random(), float(np.random.random()), float(torch.rand(1))]
    backend.restore(saved)
    assert [random.random(), float(np.random.random()), float(torch.rand(1))] == expected


def test_notebook_seeds_only_fresh_run_once_and_resume_setup_does_not_seed(experiment, tmp_path, streams, monkeypatch):
    from icl_pilot.generation import randomness
    notebook = json.loads((ROOT / "notebooks/colab_direct_generation_pipeline.ipynb").read_text())
    source = "".join(next(cell for cell in notebook["cells"] if cell.get("id") == "generation-setup")["source"])
    seeded = []
    original = randomness.initialise_run_randomness
    def initialise(seed, **kwargs):
        seeded.append(seed)
        original(seed, **kwargs)
    monkeypatch.setattr(randomness, "initialise_run_randomness", initialise)
    namespace = {"json": json, "RUN_GENERATION": True, "CONFIG_PATH": experiment, "DATA_ROOT": tmp_path,
                 "RUN_OUTPUT_DIR": tmp_path / "run", "RUN_ID": "run", "RESUME_GENERATION": False,
                 "REPLICATES": None, "SMOKE_TEST": True, "SMOKE_TEST_JOBS": 2,
                 "MODEL_KEY": "llama3_1_8b", "persist_run": lambda: None}
    exec(source, namespace)
    exec(source, namespace)
    assert seeded == [42]
    with pytest.raises(RuntimeError):
        run_generation(namespace["prepared"], generator(streams, fail_call=2), run_id="run", random_streams=streams)
    namespace["RESUME_GENERATION"] = True
    namespace.pop("_SEEDED_RUN_ID")
    exec(source, namespace)
    assert seeded == [42]


def test_remote_checkpoint_commit_recovers_even_if_mirror_output_append_is_interrupted(tmp_path, streams, monkeypatch):
    from icl_pilot.generation import recovery
    prepared = prepared_run(tmp_path / "run")
    target = tmp_path / "drive" / "run"
    atomic = recovery.atomic_json
    def fail_after_remote_commit(path, value):
        atomic(path, value)
        if path == target / CHECKPOINT and value["completed_calls"] == 2:
            raise OSError("mirror interrupted after atomic checkpoint commit")
    monkeypatch.setattr(recovery, "atomic_json", fail_after_remote_commit)
    with pytest.raises(OSError, match="mirror interrupted"):
        run_generation(prepared, generator(streams), run_id="run", random_streams=streams,
                       on_checkpoint=lambda source: sync_generation_checkpoint(source, target))
    assert read_checkpoint(target)["completed_calls"] == 2
    assert len(outputs(target)) == 1
    recover_journal(target, read_checkpoint(target))
    assert outputs(target) == outputs(prepared.run_manifest_path.parent)
    monkeypatch.setattr(recovery, "atomic_json", atomic)
    sync_generation_checkpoint(prepared.run_manifest_path.parent, target)
    assert len(outputs(target)) == 2
