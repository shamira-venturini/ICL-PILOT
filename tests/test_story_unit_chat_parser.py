from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from icl_pilot.story_unit_semrp import (
    _iter_story_utterances, _parse_age_group, extract_story_unit_narratives,
)


@pytest.mark.parametrize("age,months", [
    ("4;08.18", "56"), ("6;04.", "76"), ("4;08", "56"),
    ("0;11.31", "11"), ("4;", ""), ("", ""),
    ("4;12.01", ""), ("4;08.99", ""), ("4;08;18", ""), ("unknown", ""),
])
def test_chat_age_format_preserves_raw_age_and_unknown_months(age: str, months: str) -> None:
    text = f"@ID:\teng|ENNI|CHI|{age}|female|SLI||Target_Child|||\n"
    assert _parse_age_group(text) == (age, months, "SLI")


def test_metadata_uses_literal_child_header_and_group() -> None:
    text = "@ID: eng|OtherCorpus|EXA|35;00.00|female|investigator|||\n"
    text += "@ID: eng|OtherCorpus|CHI|5;03.12||literal_group|||\n"
    assert _parse_age_group(text) == ("5;03.12", "63", "literal_group")
    assert _parse_age_group("@ID:\teng|ENNI|CHI|4;08.18|female|||||\n") == ("4;08.18", "56", "")


def test_conflicting_child_headers_fail_instead_of_selecting_one() -> None:
    text = "@ID: eng|ENNI|CHI|4;08.18|female|TD|||\n"
    text += "@ID: eng|ENNI|CHI|4;09.01|female|SLI|||\n"
    with pytest.raises(ValueError, match="Conflicting CHI"):
        _parse_age_group(text)


def test_wrapped_child_tiers_join_before_cleaning_and_keep_story_boundaries() -> None:
    text = "\n".join([
        "\ufeff@Begin", "*CHI: this introduction is outside a story .",
        "@G: A1", "*CHI: he he runned [* m:ed] [= ran",
        "\tyesterday] to the", "\tpool . \x15100_200\x15",
        "%mor: pro:sub|he v|run-PAST prep|to",
        "\tdet:art|the n|pool .", "%gra: 1|2|SUBJ",
        "\t2|0|ROOT 3|2|JCT", "*EXA: an adult utterance",
        "\twhich continues here .", "*CHI: then the ball",
        "\tfell in .", "@G:\tB1", "*CHI:\tthey built a castle",
        "\t.", "@End",
    ])
    issues = []
    stories = _iter_story_utterances(text, diagnostics=issues)
    assert stories == {"A1": ["he he runned to the pool .", "then the ball fell in ."],
                       "B1": ["they built a castle ."]}
    assert issues == []


def test_orphan_annotation_blocks_cannot_enter_child_narrative() -> None:
    text = "\n".join([
        "@G: A1", "*CHI:\ther runned back .",
        "\tpro:poss|her v|run-PAST adv|back .", "\t1|2|SUBJ 2|0|ROOT",
        "\t.", "*CHI:\the hurted her leg .", "\t.",
        "*CHI:\tand went home .", "@End",
    ])
    issues = []
    assert _iter_story_utterances(text, diagnostics=issues) == {
        "A1": ["her runned back .", "he hurted her leg .", "and went home ."]}
    assert [r["kind"] for r in issues] == ["orphan_annotation_continuation"] * 3 + ["redundant_punctuation_continuation"]
    assert [r["line_number"] for r in issues] == [3, 4, 5, 7]
    with pytest.warns(UserWarning, match="Excluded 4"):
        _iter_story_utterances(text)


def test_legacy_surface_policy_is_preserved_instead_of_correcting_child_grammar() -> None:
    text = "@G: A1\n*CHI: <her> [//] her runned [: ran] [* m:ed] (.) xxx &=whispers .\n"
    assert _iter_story_utterances(text) == {"A1": ["her her runned (.) xxx &=whispers ."]}


@pytest.mark.parametrize("delimiter,expected", [
    ('+"/.', 'he said +"/.'), ("+//.", "he said"), ("+...", "he said"), ("+/.", "he said"),
])
def test_orphan_chat_terminators_do_not_duplicate_main_tier_punctuation(delimiter: str, expected: str) -> None:
    issues = []
    text = f"@G: A1\n*CHI: he said {delimiter}\n\t{delimiter}\n*CHI: no toys .\n"
    assert _iter_story_utterances(text, diagnostics=issues) == {"A1": [expected, "no toys ."]}
    assert [r["kind"] for r in issues] == ["redundant_punctuation_continuation"]


@pytest.mark.parametrize("annotation", [
    "un#adj|do&PASTP .", "un#n:gerund|load-GERUND .",
    "+n|air+n|plane .", "1|2|SUBJ 2|0|ROOT", "word|VERB .|PUNCT",
])
def test_prefixed_compound_and_ud_annotation_remnants_are_excluded(annotation: str) -> None:
    issues = []
    assert _iter_story_utterances(f"@G: A1\n*CHI: it was undone .\n\t{annotation}\n",
                                  diagnostics=issues) == {"A1": ["it was undone ."]}
    assert [r["kind"] for r in issues] == ["orphan_annotation_continuation"]


def test_extraction_fixes_metadata_and_reports_source_hashes_without_editing_chat(tmp_path: Path) -> None:
    root = tmp_path / "chat"
    root.mkdir()
    transcript = root / "474.cha"
    raw = "@ID:\teng|ENNI|CHI|4;08.18|female|SLI||Target_Child|||\n"
    raw += "@G: A1\n*CHI: her runned\n\tto the pool .\n%mor: pro:poss|her\n\tv|run-PAST .\n@End\n"
    transcript.write_text(raw)
    before = transcript.read_bytes()
    output = tmp_path / "new" / "narratives.csv"
    assert extract_story_unit_narratives(str(root), str(output)) == 0
    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["Age_Raw"] == "4;08.18"
    assert rows[0]["Age_Month"] == "56"
    assert rows[0]["Group"] == "SLI"
    assert rows[0]["story_text"] == "her runned to the pool ."
    qc = json.loads(output.with_suffix(".parsing_qc.json").read_text())
    assert qc["sources"][0]["transcript_path"] == "474.cha"
    assert qc["sources"][0]["sha256"] == hashlib.sha256(before).hexdigest()
    assert qc["issue_counts"] == {}
    assert transcript.read_bytes() == before


def test_unknown_metadata_remains_blank_and_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "chat"
    root.mkdir()
    (root / "523.cha").write_text("@ID: eng|ENNI|CHI|4;|female|||||\n@G: A1\n*CHI: a ball .\n")
    output = tmp_path / "new.csv"
    extract_story_unit_narratives(str(root), str(output))
    with output.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["Age_Raw"] == "4;"
    assert row["Age_Month"] == row["Group"] == ""
    qc = json.loads(output.with_suffix(".parsing_qc.json").read_text())
    assert qc["issue_counts"] == {"unknown_or_invalid_age_months": 1, "missing_group": 1}


@pytest.mark.parametrize("existing", ["csv", "qc"])
def test_existing_research_outputs_are_not_overwritten(tmp_path: Path, existing: str) -> None:
    root = tmp_path / "chat"
    root.mkdir()
    (root / "410.cha").write_text("@G: A1\n*CHI: a ball .\n")
    output = tmp_path / "narratives.csv"
    preserved = output if existing == "csv" else output.with_suffix(".parsing_qc.json")
    preserved.write_text("existing research artifact\n")
    with pytest.raises(FileExistsError, match="choose fresh output"):
        extract_story_unit_narratives(str(root), str(output))
    assert preserved.read_text() == "existing research artifact\n"


def test_missing_transcript_root_does_not_write_an_empty_table(tmp_path: Path) -> None:
    output = tmp_path / "narratives.csv"
    with pytest.raises(FileNotFoundError, match="transcript root"):
        extract_story_unit_narratives(str(tmp_path / "missing"), str(output))
    assert not output.exists()
