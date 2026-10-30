from __future__ import annotations

import json

import pytest

from icl_pilot.maintenance.real_story_tokenizer_audit import audit, representations, summarize


def test_tokenizer_audit_preserves_partial_words_in_the_definitive_preparation():
    text = "her xxx (be)cause (.) runned &=laughs ."
    result = representations({"story_text": text, "story_units_json": json.dumps([text])})
    assert result == {"prepared_text": "her (be)cause runned ."}


def test_unusable_prepared_story_has_no_invented_zero_token_count():
    result = representations({"story_text": "xxx .", "story_units_json": '["xxx ."]'})
    assert result["prepared_text"] is None


def test_summary_counts_ceiling_boundaries_and_preserves_missingness():
    rows = [{"File_ID": str(i), "Group": "TD", "Age_Raw": "4;01.00", "story_id": "A1", "n": n}
            for i, n in enumerate([None, 500, 640, 650])]
    summary = summarize(rows, "n", 640)
    assert summary["counted"] == 3 and summary["unusable"] == 1
    assert summary["median_tokens"] == 640 and summary["p95_nearest_rank"] == 650
    assert summary["max_tokens"] == 650 and summary["ceiling_minus_max"] == -10
    assert summary["above_ceiling"] == 1 and summary["at_or_above_ceiling"] == 2
    assert summary["maxima"][0]["File_ID"] == "3"


def test_summary_preserves_all_missing_values():
    summary = summarize([{"n": None}], "n", 640)
    assert summary["counted"] == 0 and summary["unusable"] == 1
    assert summary["max_tokens"] is None and summary["median_tokens"] is None
    assert summary["p95_nearest_rank"] is None and summary["ceiling_minus_max"] is None
    assert summary["maxima"] == []


def test_existing_output_is_preserved(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "comparison.json"
    marker.write_text("preserve this evidence")
    with pytest.raises(FileExistsError):
        audit(tmp_path / "missing.csv", tmp_path / "missing_roster.csv", {}, output)
    assert marker.read_text() == "preserve this evidence"


def test_frozen_csv_hash_mismatch_stops_before_output(tmp_path):
    source = tmp_path / "input.csv"
    source.write_text("changed corpus")
    source.with_suffix(".parsing_qc.json").write_text(json.dumps({"output_csv_sha256": "wrong"}))
    output = tmp_path / "audit"
    with pytest.raises(ValueError, match="frozen extraction QC"):
        audit(source, tmp_path / "missing_roster.csv", {"gemma2_9b": tmp_path / "missing.json"}, output)
    assert not output.exists()
