"""Annotation drift is repaired from immutable source, without guessing labels."""
import json
import random

import pytest

from icl_pilot.chat.annotation import postprocess_annotated_chat
from icl_pilot.chat.preservation import require_text_preserved, text_fingerprint


def chat(*bodies):
    return "@Begin\n" + "".join("*CHI:\t" + body + "\n" for body in bodies) + "@End\n"


@pytest.mark.parametrize("source,raw,restored", [
    ("He goed home .", "He went [* m:over] home.", "He goed home ."),
    ("he go to a park .", "he go [* m:03s:a] to park.", "he go [* m:03s:a] to a park ."),
    ("he go home .", "he really [* m:over] go [* m:03s:a] home.", "he go [* m:03s:a] home ."),
    ("he he go .", "he [* s:r:gc:pro] go.", "he he go ."),
    ("he don't goed .", "he doesn't [* m:03s:a] goed [* m:over].", "he don't goed [* m:over] ."),
    ("he (.) go [/] go home .", "he go [/] go home.", "he (.) go [/] go home ."),
])
def test_restores_changed_deleted_added_and_repeated_words(source, raw, restored):
    audit = {}
    result = postprocess_annotated_chat(chat(source), chat(raw), audit=audit)
    assert result == chat(restored)
    require_text_preserved(text_fingerprint(chat(source)), result)
    assert audit["utterances"][0]["edit_distance"] > 0
    if source != "he (.) go [/] go home .":
        assert audit["annotation_coverage"] == "partial"


def test_restored_word_is_recorded_as_unassessed_instead_of_error_free():
    audit = {}
    result = postprocess_annotated_chat(chat("he go home ."), chat("he goes [* m:03s:a] home ."), audit=audit)
    assert result == chat("he go home .")
    report = audit["utterances"][0]
    assert [item["token"] for item in report["unassessed_source_words"]] == ["go"]
    assert [item["tag"] for item in report["unmapped_annotations"]] == ["[* m:03s:a]"]
    assert report["annotation_coverage"] == "partial"


def test_repeated_word_alignment_does_not_move_a_tag_arbitrarily():
    audit = {}
    result = postprocess_annotated_chat(chat("he he go ."), chat("he [* s:r:gc:pro] go ."), audit=audit)
    assert result == chat("he he go .")
    report = audit["utterances"][0]
    assert len(report["unassessed_source_words"]) == 2
    assert report["unmapped_annotations"][0]["reason"] == "changed_inserted_or_ambiguous_anchor"


@pytest.mark.parametrize("raw", ["he really go .", "he go [* m:over .", "he go [*] ."])
def test_inserted_words_or_malformed_tags_do_not_look_like_complete_error_assessment(raw):
    audit = {}
    assert postprocess_annotated_chat(chat("he go ."), chat(raw), audit=audit) == chat("he go .")
    assert audit["annotation_coverage"] == "partial"


@pytest.mark.parametrize("raw", [chat("he go [* m:03s:a] . she run [* m:03s:a] ."),
                                 chat("he go [* m:03s:a] .", "she", "run [* m:03s:a] .")])
def test_automatically_restores_original_utterance_boundaries(raw):
    source = chat("he go .", "she run .")
    audit = {}
    result = postprocess_annotated_chat(source, raw, audit=audit)
    assert result == chat("he go [* m:03s:a] .", "she run [* m:03s:a] .")
    assert audit["boundaries_restored"] is True
    require_text_preserved(text_fingerprint(source), result)


def test_empty_annotator_output_restores_the_entire_original_file():
    source = chat("he go .", "she run .")
    audit = {}
    assert postprocess_annotated_chat(source, "", audit=audit) == source
    assert audit["annotation_coverage"] == "partial" and audit["raw_utterances"] == 0
    assert len(audit["utterances"][0]["unassessed_source_words"]) == 4


def test_edit_and_boundary_variations_always_preserve_original_speech():
    rng = random.Random(913)
    vocabulary = ["he", "he", "goed", "home", "don't", "want", "a", "ball"]
    for _ in range(100):
        source_words = [[rng.choice(vocabulary) for _ in range(6)] for _ in range(3)]
        source = chat(*( " ".join(words) + " ." for words in source_words))
        raw_words = []
        for words in source_words:
            for word in words:
                operation = rng.randrange(4)
                if operation == 0:
                    continue
                raw_words.append(rng.choice(vocabulary) if operation == 1 else word)
                if rng.randrange(3) == 0:
                    raw_words.append("[* m:over]")
                if operation == 2:
                    raw_words.append("extra")
        raw_bodies = []
        while raw_words:
            count = rng.randint(1, 10)
            raw_bodies.append(" ".join(raw_words[:count]) + " .")
            raw_words = raw_words[count:]
        result = postprocess_annotated_chat(source, chat(*raw_bodies))
        require_text_preserved(text_fingerprint(source), result)


def test_headers_other_speakers_and_existing_tiers_always_come_from_source():
    source = "@Begin\n@Comment:\tsource header\n*EXA:\twhy ?\n*CHI:\the go .\n%foo:\tkeep\n@End\n"
    raw = "@Begin\n@Comment:\trewritten header\n*CHI:\the go [* m:03s:a] .\n@End\n"
    audit = {}
    result = postprocess_annotated_chat(source, raw, audit=audit)
    assert result == source.replace("he go .", "he go [* m:03s:a] .")
    assert audit["raw_non_target_content_ignored"] is True
    require_text_preserved(text_fingerprint(source), result)


@pytest.mark.parametrize("mode", ["text_drift", "tier_loss"])
def test_wrapper_continues_after_automatic_restoration_and_retains_coverage_audit(tmp_path, monkeypatch, mode):
    from test_annotation_pipeline import annotate, fake_tool, CHAT, ROOT
    from icl_pilot.chat.annotation import load_annotation_config
    fake_tool(monkeypatch, mode=mode)
    settings = load_annotation_config(ROOT / "configs/annotation/talktag.json")
    path = annotate(tmp_path, settings)
    report = json.loads(path.read_text())
    assert report["status"] == "annotation_complete" and report["qc_issues"] == []
    assert report["postprocessing"]["version"] == 3
    record = report["postprocessing"]["files"][0]["source_restoration"]
    if mode == "text_drift":
        assert record["annotation_coverage"] == "partial"
        assert record["utterances"][0]["unassessed_source_words"][0]["token"] == "home"
    else:
        assert record["raw_non_target_content_ignored"] is True
    assert (tmp_path / "source/sample.cha").read_text() == CHAT
