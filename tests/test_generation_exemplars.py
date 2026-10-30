from __future__ import annotations

import json

import pytest

from icl_pilot.chat.story_unit_semrp import _iter_story_utterances
from icl_pilot.chat.text_cleaning import prepare_extracted_story as prepare_exemplar
from icl_pilot.generation.exemplars import validate_exemplar


@pytest.mark.parametrize("marker", ["(.)", "(..)", "(...)", "(0.7)", "(2)",
                                   "xxx", "XXX", "yyy", "www", "&=whispers", "&=high_pitched"])
def test_markers_are_deleted_without_guessing_missing_words(marker):
    result = prepare_exemplar(f"her {marker} runned .")
    assert result.text == "her runned ."
    assert sum(result.removed.values()) == 1


def test_preparation_preserves_repetitions_grammar_and_partial_word_notation():
    original = "her her runned (be)cause the dog goed back ."
    result = prepare_exemplar(original)
    assert result.text == original
    assert not any(result.removed.values())


def test_unknown_markers_do_not_remove_parts_of_other_words_or_terminal_punctuation():
    assert prepare_exemplar("xxxylophone wwword xxx?").text == "xxxylophone wwword ?"
    assert prepare_exemplar("go &=laughs, now .").text == "go , now ."


def test_extraction_and_preparation_do_not_use_annotated_replacement_words():
    chat = "@G: A1\n*CHI: her xxx [: dog] runned [* m:ed] (.) &=whispers .\n"
    units = _iter_story_utterances(chat)["A1"]
    result = prepare_exemplar(" ".join(units), json.dumps(units))
    assert result.text == "her runned ."
    assert "dog" not in result.text


def test_marker_only_utterances_are_dropped_without_stray_sentence_boundaries():
    units = ["xxx .", "her runned .", "(.) &=laughs !"]
    result = prepare_exemplar(" ".join(units), json.dumps(units))
    assert result.text == "her runned ."
    assert result.utterances == ("her runned .",)
    assert result.original_utterances == 3 and result.dropped_utterances == 2
    assert result.removed == {"annotator_annotations": 0, "pause_markers": 1, "unintelligible_markers": 1, "non_speech_codes": 1}


@pytest.mark.parametrize("text", ["xxx .", "(.) &=laughs !", "www", "..."])
def test_empty_speech_fails_instead_of_inventing_an_exemplar(text):
    with pytest.raises(ValueError, match="no spoken text"):
        prepare_exemplar(text)


@pytest.mark.parametrize("units", ['"not an array"', '[1]', '["different text"]'])
def test_inconsistent_utterance_arrays_are_rejected(units):
    with pytest.raises(ValueError, match="utterance array"):
        prepare_exemplar("original .", units)


def test_preparation_is_idempotent():
    first = prepare_exemplar("he xxx (.) runned &=laughs .")
    second = prepare_exemplar(first.text, json.dumps(first.utterances))
    assert first.text == second.text
    assert not any(second.removed.values())


def test_generation_validation_preserves_literal_prepared_text():
    text = "her her runned (be)cause he goed ."
    result = validate_exemplar(text, json.dumps([text]))
    assert result.text is text
    assert result.utterances == (text,)


@pytest.mark.parametrize("marker", ["(.)", "xxx", "yyy", "www", "&=whispers", "[: went]", "[= comment]", "[* m:ed]"])
def test_generation_rejects_unfinished_extraction_instead_of_cleaning_it(marker):
    text = f"he {marker} goed ."
    with pytest.raises(ValueError, match="CHAT-to-CSV extraction"):
        validate_exemplar(text, json.dumps([text]))


def test_generation_rejects_stray_non_speech_boundaries():
    with pytest.raises(ValueError, match="non-speech utterance"):
        validate_exemplar(". her runned .", json.dumps([".", "her runned ."]))
