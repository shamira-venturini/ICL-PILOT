"""Learned-boundary contract, word preservation and export provenance."""
import csv
import hashlib
import json
import os
from pathlib import Path

import pytest

from icl_pilot.chat import segmentation
from icl_pilot.chat.generated_chat import convert_generated_story_csv_to_cha
from icl_pilot.pipeline import convert_csv_inputs
from segmentation_stub import StubSegmenter


def segment(text, actions):
    return segmentation.render_segmented_narrative(segmentation.prepare_narrative(text), actions)


def test_whole_story_boundaries_can_merge_punctuation_and_split_run_on():
    # Six words: model ignores the period and splits before the last word.
    result = segment("he goed. she run and fall", [0, 0, 0, 0, 2, 0])
    assert result["utterances"] == ["he goed she run and .", "fall ."]
    assert result["word_count"] == 6
    assert result["boundaries"][-1]["final_fragment_default"] is True


def test_model_question_and_exclamation_labels_become_chat_terminators():
    assert segment("he go home she go too", [0, 0, 3, 0, 0, 4])["utterances"] == [
        "he go home ?", "she go too !"]


@pytest.mark.parametrize("actions", [[0], [0, 9], [False, 2], [0, "2"]])
def test_invalid_predictions_fail_closed(actions):
    with pytest.raises(ValueError, match="mismatch"):
        segment("he goed", actions)


@pytest.mark.parametrize("text", ["he [*", "he ]", "(.)", "he \x00 goed"])
def test_malformed_or_wordless_narratives_are_refused(text):
    with pytest.raises(ValueError):
        segmentation.prepare_narrative(text)


def test_case_repetition_morphology_annotations_and_pauses_are_preserved():
    text = "He (.) go [* m:0ed] go [/] (be)cause she &=whispers want it"
    result = segment(text, [0, 0, 0, 0, 0, 0, 2])
    assert result["utterances"] == [text + " ."]
    for word, (start, end) in zip(segmentation.prepare_narrative(text)["words"], result["word_offsets"]):
        assert text[start:end] == word


def test_quotations_do_not_add_utterance_boundaries_or_reporting_words():
    result = segment('He said "no go!" then he goed.', [0, 0, 0, 0, 0, 0, 2])
    assert result["utterances"] == ['He said “no go” then he goed .']
    assert len(result["punctuation_changes"]) == 4


def test_explicit_quote_controls_are_logged_and_do_not_force_boundaries():
    result = segment('he said +"/. +" no go!', [0, 0, 0, 4])
    assert result["utterances"] == ['he said “no go” !']
    assert {c["before"] for c in result["punctuation_changes"]} == {'+"/.', '+"', '!', ''}


def test_quoted_speech_is_balanced_across_predicted_boundaries_without_extra_tiers():
    result = segment('He said "no go and no stay"', [0, 0, 0, 2, 0, 0, 2])
    assert result["utterances"] == ['He said “no go” .', '“and no stay” .']
    assert {change.get("kind") for change in result["punctuation_changes"]} >= {
        "quote_boundary_close", "quote_boundary_reopen"}


def test_scope_conflict_is_refused_rather_than_dropping_or_rewriting_words():
    with pytest.raises(ValueError, match="inside a CHAT scope"):
        segment("<he go> [/] he goed", [2, 0, 0, 2])
    assert segment("<he go> [/] he goed", [0, 0, 0, 2])["utterances"] == ["<he go> [/] he goed ."]
    assert segment("first <he go> [/] he goed", [2, 0, 0, 0, 2])["utterances"] == [
        "first .", "<he go> [/] he goed ."]


def test_explicit_interruption_requires_matching_boundary():
    assert segment("he go +... she goed", [0, 2, 0, 2])["utterances"] == ["he go +...", "she goed ."]
    with pytest.raises(ValueError, match="conflicts"):
        segment("he go +... she goed", [0, 0, 0, 2])
    assert segment('"he go +..."', [0, 2])["utterances"] == ['“he go” +...']
    assert segment("he go +...", [0, 0])["boundaries"][-1]["final_fragment_default"] is False


def saved_csv(path, suffix="1", text="He goed. she run"):
    row = {"bundle_id": "bundle_" + suffix, "pair_id": "pair_" + suffix, "target_story": "A1",
           "td_child_id": "410", "sli_child_id": "444", "td_age": "4;01.27", "sli_age": "4;02.04",
           "stage1_td_story": text, "stage2_sli_story": text,
           "stage1_prompt_record_id": "td_" + suffix, "stage2_prompt_record_id": "sli_" + suffix}
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return path


def test_cohorts_and_stages_use_one_predictor_call_and_exact_parent_metadata(tmp_path):
    one, two = saved_csv(tmp_path / "one.csv"), saved_csv(tmp_path / "two.csv", "2")
    before = [one.read_bytes(), two.read_bytes()]
    stub = StubSegmenter()
    roots = convert_csv_inputs({"one": one, "two": two}, tmp_path / "run", roster_csv=None, segmenter=stub)
    assert len(stub.calls) == 1 and len(stub.calls[0]) == 4
    for index, root in enumerate(roots.values(), 1):
        records = [json.loads(line) for line in (root / "segmentation_records.jsonl").read_text().splitlines()]
        assert records[1]["parent_prompt_record_id"] == "td_" + str(index)
        assert records[0]["policy"] == segmentation.POLICY
        with (root / "chat_manifest.csv").open() as handle:
            manifest = list(csv.DictReader(handle))
        for entry, record in zip(manifest, records):
            assert entry["segmentation_record_sha256"] == hashlib.sha256(json.dumps(
                record, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            assert entry["chat_sha256"] == hashlib.sha256(Path(entry["output_path"]).read_bytes()).hexdigest()
    assert [one.read_bytes(), two.read_bytes()] == before


def test_chat_apostrophe_glyph_conversion_is_audited_before_segmentation(tmp_path):
    text = "they’re playing and he ‘goed’ and they’re happy."
    source = saved_csv(tmp_path / "source.csv", text=text)
    before = source.read_bytes()
    stub = StubSegmenter()
    roots = convert_csv_inputs({"one": source}, tmp_path / "run", roster_csv=None, segmenter=stub)
    expected = text.replace("‘", "'").replace("’", "'")
    assert stub.calls == [[expected, expected]]
    with (roots["one"] / "chat_manifest.csv").open() as handle:
        entries = list(csv.DictReader(handle))
    for entry in entries:
        changes = json.loads(entry["formatting_changes"])
        assert changes == [
            {"kind": "ascii_apostrophe", "before": char, "after": "'"}
            for char in text if char in "‘’"
        ]
        assert entry["source_text_sha256"] == hashlib.sha256(text.encode()).hexdigest()
        chat = Path(entry["output_path"]).read_text()
        assert "they're playing and he 'goed' and they're happy ." in chat
        assert "‘" not in chat and "’" not in chat
    assert source.read_bytes() == before


def test_late_invalid_csv_stops_before_inference_or_any_export(tmp_path):
    one, two = saved_csv(tmp_path / "one.csv"), saved_csv(tmp_path / "two.csv", "2", text="")
    stub = StubSegmenter()
    with pytest.raises(ValueError, match="Empty"):
        convert_csv_inputs({"one": one, "two": two}, tmp_path / "run", roster_csv=None, segmenter=stub)
    assert not stub.calls and not (tmp_path / "run").exists()


def test_tampered_or_incomplete_segmentation_is_rejected_before_chat_writing(tmp_path):
    source = saved_csv(tmp_path / "source.csv")
    class Tampered(StubSegmenter):
        def segment_many(self, *args):
            results, metadata = super().segment_many(*args)
            results[0]["utterances"][0] = "invented correct grammar ."
            return results, metadata
    with pytest.raises(ValueError, match="changed words"):
        convert_generated_story_csv_to_cha(input_csv=source, output_dir=tmp_path / "out",
                                            roster_csv=None, segmenter=Tampered())
    assert not list((tmp_path / "out").rglob("*.cha"))


def test_config_is_pinned_and_no_stanza_or_legacy_fallback(tmp_path):
    config = segmentation.load_segmentation_config()
    assert config["version"] == "0.10.7" and config["device"] == "cpu"
    config["model_revision"] = "main"
    file = tmp_path / "bad.json"
    file.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="pinned"):
        segmentation.load_segmentation_config(file)


def test_missing_runtime_never_falls_back_to_punctuation(tmp_path):
    runtime = segmentation.BatchalignSegmenter(tmp_path / "missing/python")
    with pytest.raises(FileNotFoundError, match="runtime missing"):
        runtime.segment_many(["he goed. she run"], tmp_path / "report")
    assert not (tmp_path / "report").exists()


@pytest.mark.skipif(not os.environ.get("ICL_PILOT_SEGMENTATION_PYTHON"),
                    reason="Opt-in actual BERT weights/runtime; ordinary tests never download models.")
def test_actual_batchalign_model_csv_to_chat_and_native_chatter(tmp_path, monkeypatch):
    import random
    import numpy as np
    from icl_pilot.chat.validation import ensure_chatter, validate_chat_directories
    from test_chat_validation import fake_talktag
    from icl_pilot.chat import annotation
    fixtures = [
        "The elephant ran to the pool. He got the ball and gave it back!",
        "he goed to pool and he he get ball then she fall down and him was sad",
        'She said "where is my ball?" Then he said "I got it!" and gave it back.',
        "he (.) go [/] go (be)cause she want it and she &=whispers goed home .",
        " ".join(["The dog goed to the park and he saw a bunny who was running"] * 55),
    ]
    source = tmp_path / "actual_model_fixtures.csv"
    rows = []
    for index, text in enumerate(fixtures):
        file = saved_csv(tmp_path / (str(index) + ".csv"), str(index), text=text)
        with file.open() as handle:
            rows.extend(csv.DictReader(handle))
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    before = source.read_bytes()
    python_state, numpy_state = random.getstate(), np.random.get_state()
    output = convert_generated_story_csv_to_cha(input_csv=source, output_dir=tmp_path / "chat",
        roster_csv=None, segmenter=segmentation.BatchalignSegmenter(os.environ["ICL_PILOT_SEGMENTATION_PYTHON"]))
    records = [json.loads(line) for line in output["segmentation_records"].read_text().splitlines()]
    assert len(records) == len(fixtures) * 2
    assert all(record["word_preservation"] == "original_tokens_in_original_order" for record in records)
    assert records[1]["parent_prompt_record_id"] == "td_0"
    provenance = json.loads((output["output_dir"] / "segmentation_provenance.json").read_text())
    runtime = provenance["runtime"]["runtime"]
    assert runtime["packages"]["batchalign"] == "0.10.7"
    assert runtime["model_revision"] == segmentation.load_segmentation_config()["model_revision"]
    assert runtime["legacy_cleanup"] is runtime["stanza_fallback"] is False
    validator = ensure_chatter(install=False)
    checked = validate_chat_directories({"fixtures": output["output_dir"]}, tmp_path / "chatter", validator=validator)
    assert json.loads(checked.read_text())["files_checked"] == len(fixtures) * 2
    assert source.read_bytes() == before
    assert random.getstate() == python_state
    after_numpy = np.random.get_state()
    assert after_numpy[0] == numpy_state[0] and (after_numpy[1] == numpy_state[1]).all()
    assert after_numpy[2:] == numpy_state[2:]
    # Real BERT + native Chatter + simulated TalkTag proves the stage wiring;
    # this does not claim execution/accuracy of the full TalkTag model.
    calls = fake_talktag(monkeypatch)
    config = annotation.load_annotation_config(segmentation.ROOT / "configs/annotation/talktag.json")
    annotated = annotation.annotate_chat_directory(output["output_dir"], tmp_path / "annotation",
        config=config, python="test-fixture", runtime={"source_commit": config["revision"]}, validator=validator)
    record = json.loads(annotated.read_text())
    assert len(calls) == 1 and record["status"] == "annotation_complete"
    assert record["expected_target_utterances"] == record["reported_target_utterances"]
