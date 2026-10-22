from __future__ import annotations

import csv
from pathlib import Path

import pytest

from icl_pilot.maintenance.generation_endings_audit import audit_csv, main, screen_text, summarize


@pytest.mark.parametrize("text,terminal", [
    ('She said "the end."', True), ("They went home…", True),
    ("They went home.)", True), ("The mommy", False), ("and", False),
])
def test_punctuation_is_only_a_descriptive_indicator(text, terminal):
    result = screen_text(text)
    assert result["has_terminal_punctuation"] is terminal
    assert result["termination_status"] == "unknown_from_saved_text"
    assert result["retokenized_saved_text_tokens"] is None


def test_boundary_count_does_not_fabricate_a_stop_reason():
    result = screen_text("They went home.", token_counter=lambda text: 180)
    assert result["proxy_equals_reference_limit"] is True
    assert result["has_terminal_punctuation"] is True
    assert result["termination_status"] == "unknown_from_saved_text"
    assert screen_text("The mommy", token_counter=lambda text: 178)["proxy_within_two_below_limit"] is True
    assert screen_text("The mommy", token_counter=lambda text: 181)["proxy_above_reference_limit"] is True


def test_repetition_measure_is_case_insensitive_without_rewriting_text():
    result = screen_text("Ball ball BALL! Then they went home.")
    assert result["max_consecutive_identical_words"] == 3
    assert result["tail_text"] == "Ball ball BALL! Then they went home."


def source_csv(path: Path, duplicate=False):
    row = {"bundle_id": "pair_rep01", "pair_id": "pair", "round_id": "round_01",
           "replicate_id": "1", "target_story": "A1", "stage1_group": "TD",
           "stage2_group": "SLI", "stage1_td_story": "The ball fell.\nThen it came back.",
           "stage2_sli_story": "The ball fall.", "td_child_id": "410", "sli_child_id": "444"}
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerows([row, row] if duplicate else [row])


def test_audit_preserves_sources_metadata_and_literal_text(tmp_path):
    path = tmp_path / "source.csv"
    source_csv(path)
    before = path.read_bytes()
    rows = audit_csv(path, cohort_year=4, token_counter=lambda text: 180)
    assert path.read_bytes() == before
    assert [row["conditioning_group"] for row in rows] == ["TD", "SLI"]
    assert [row["source_child_id"] for row in rows] == ["410", "444"]
    assert rows[0]["tail_text"] == "The ball fell.\nThen it came back."
    assert all(row["recorded_cohort"] == "" for row in rows)
    assert summarize(rows, ("stage",))[0]["proxy_equals_reference_limit"] == 1


def test_duplicate_jobs_are_rejected(tmp_path):
    path = tmp_path / "source.csv"
    source_csv(path, duplicate=True)
    with pytest.raises(ValueError, match="Duplicate generation job"):
        audit_csv(path, cohort_year=4)


def test_missing_source_columns_are_rejected_before_reports_are_created(tmp_path):
    path = tmp_path / "source.csv"
    path.write_text("bundle_id\npair\n")
    output = tmp_path / "audit"
    with pytest.raises(ValueError, match="Missing generation columns"):
        main(["--source", "4=" + str(path), "--output-dir", str(output)])
    assert not output.exists()


def test_existing_report_directory_is_protected(tmp_path):
    output = tmp_path / "audit"
    output.mkdir()
    saved = output / "summary.json"
    saved.write_text("Existing research record")
    with pytest.raises(SystemExit):
        main(["--source", "4=unused.csv", "--output-dir", str(output)])
    assert saved.read_text() == "Existing research record"
