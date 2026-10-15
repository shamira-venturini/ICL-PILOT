from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from icl_pilot.generated_chat import convert_generated_story_csv_to_cha


def write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def row() -> dict[str, str]:
    return {
        "cohort": "4-year-old", "round_id": "round_01",
        "bundle_id": "pair_sli474_td459_rep01", "pair_id": "pair_sli474_td459",
        "replicate_id": "1", "target_story": "A1",
        "td_child_id": "459", "sli_child_id": "474",
        "td_age": "4;08.04", "sli_age": "4;08.18",
        "stage1_group": "TD", "stage2_group": "SLI",
        "stage1_td_story": "The elephant ran. He got the ball!",
        "stage2_sli_story": "Elephant run. He get ball.",
        "stage1_prompt_record_id": "td_prompt_1",
        "stage2_prompt_record_id": "sli_prompt_1",
    }


def convert(tmp_path: Path, rows: list[dict[str, str]], **kwargs):
    source = write_csv(tmp_path / "source.csv", rows)
    return convert_generated_story_csv_to_cha(
        input_csv=source, output_dir=tmp_path / "chat", **kwargs,
    )


def test_export_preserves_text_and_records_actual_parents_and_hashes(tmp_path, row):
    output = convert(tmp_path, [row])
    with output["manifest_csv"].open() as handle:
        records = list(csv.DictReader(handle))
    td, sli = records
    assert td["prompt_record_id"] == "td_prompt_1"
    assert td["parent_prompt_record_id"] == ""
    assert sli["prompt_record_id"] == "sli_prompt_1"
    assert sli["parent_prompt_record_id"] == "td_prompt_1"
    assert td["age_source"] == td["source_child_id_source"] == "generation_csv"
    assert sli["cohort"] == "4-year-old"
    for record in records:
        path = Path(record["output_path"])
        assert record["chat_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert record["source_csv_sha256"] == hashlib.sha256((tmp_path / "source.csv").read_bytes()).hexdigest()
    td_text = Path(td["output_path"]).read_text()
    assert [line for line in td_text.splitlines() if line.startswith("*CHI:")] == [
        "*CHI:\tThe elephant ran .", "*CHI:\tHe got the ball !",
    ]
    assert "|CHI|4;08.04||TD|" in td_text
    assert "@Comment:\tparent_prompt_record_id=td_prompt_1" in Path(sli["output_path"]).read_text()


@pytest.mark.parametrize("field", ["stage1_td_story", "stage2_sli_story"])
@pytest.mark.parametrize("value", ["", "   ", "..."])
def test_missing_or_nonlexical_story_rejected_before_writing(tmp_path, row, field, value):
    row[field] = value
    with pytest.raises(ValueError, match="Empty or nonlexical"):
        convert(tmp_path, [row])
    assert not (tmp_path / "chat").exists()


def test_literal_na_is_not_reinterpreted_as_a_missing_cell(tmp_path, row):
    row["stage1_td_story"] = "NA"
    output = convert(tmp_path, [row])
    assert "*CHI:\tNA ." in next(output["stage1_td_dir"].glob("*.cha")).read_text()


def test_blank_age_and_child_cells_use_checked_roster(tmp_path, row):
    for field in ("td_age", "sli_age", "td_child_id", "sli_child_id"):
        row[field] = ""
    output = convert(tmp_path, [row])
    with output["manifest_csv"].open() as handle:
        td, sli = csv.DictReader(handle)
    assert (td["age"], td["source_child_id"]) == ("4;08.04", "459")
    assert (sli["age"], sli["source_child_id"]) == ("4;08.18", "474")
    assert td["age_source"] == sli["source_child_id_source"] == "roster"


@pytest.mark.parametrize("field,value", [
    ("td_age", "4;09.04"), ("sli_child_id", "475"), ("cohort", "5-year-old"),
])
def test_conflicting_csv_metadata_rejected(tmp_path, row, field, value):
    row[field] = value
    with pytest.raises(ValueError, match="disagrees with frozen roster"):
        convert(tmp_path, [row])
    assert not (tmp_path / "chat").exists()


@pytest.mark.parametrize("field,value", [
    ("td_age", ""), ("sli_age", "4;12.00"), ("td_age", "4;08.99"),
    ("sli_child_id", ""), ("td_child_id", "nan"), ("td_age", "5;08.04"),
])
def test_self_contained_csv_requires_valid_age_and_child_metadata(tmp_path, row, field, value):
    row[field] = value
    with pytest.raises(ValueError):
        convert(tmp_path, [row], roster_csv=None)
    assert not (tmp_path / "chat").exists()


def test_explicit_missing_roster_is_not_silently_ignored(tmp_path, row):
    with pytest.raises(FileNotFoundError):
        convert(tmp_path, [row], roster_csv=tmp_path / "missing_roster.csv")
    assert not (tmp_path / "chat").exists()


def test_duplicate_roster_pair_is_rejected(tmp_path, row):
    roster_row = {k: row[k] for k in ("pair_id", "td_age", "sli_age")}
    roster = write_csv(tmp_path / "roster.csv", [roster_row, roster_row])
    with pytest.raises(ValueError, match="Duplicate roster pair_id"):
        convert(tmp_path, [row], roster_csv=roster)


def test_unknown_roster_pair_is_rejected(tmp_path, row):
    row["pair_id"] = "unknown_pair"
    with pytest.raises(ValueError, match="Unknown pair_id"):
        convert(tmp_path, [row])


def test_duplicate_destination_across_rounds_cannot_overwrite(tmp_path, row):
    for key in ("stage1_prompt_record_id", "stage2_prompt_record_id"):
        row.pop(key)
    second = dict(row, round_id="round_02")
    with pytest.raises(ValueError, match="Duplicate CHAT destination"):
        convert(tmp_path, [row, second])
    assert not (tmp_path / "chat").exists()


def test_existing_directory_and_transcripts_are_preserved(tmp_path, row):
    output = convert(tmp_path, [row])
    existing = {p: p.read_bytes() for p in output["output_dir"].rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError, match="fresh directory"):
        convert(tmp_path, [dict(row, stage1_td_story="Changed text.")])
    assert {p: p.read_bytes() for p in existing} == existing


def test_late_validation_failure_leaves_no_partial_export(tmp_path, row):
    second = dict(row, target_story="A2", stage2_sli_story="",
                  stage1_prompt_record_id="td_prompt_2", stage2_prompt_record_id="sli_prompt_2")
    with pytest.raises(ValueError, match="Empty or nonlexical"):
        convert(tmp_path, [row, second])
    assert not (tmp_path / "chat").exists()


@pytest.mark.parametrize("field,value", [
    ("bundle_id", "../outside"), ("pair_id", ""), ("target_story", "Z1"),
    ("round_id", "round\n@End"), ("replicate_id", "0"), ("stage2_group", "TD"),
    ("stage1_prompt_record_id", ""), ("stage2_prompt_record_id", "td_prompt_1"),
])
def test_invalid_identifiers_groups_and_incomplete_ancestry_rejected(tmp_path, row, field, value):
    row[field] = value
    with pytest.raises(ValueError):
        convert(tmp_path, [row])
    assert not (tmp_path / "chat").exists()


def test_prompt_ids_cannot_be_reused_for_other_stories(tmp_path, row):
    with pytest.raises(ValueError, match="reused across CSV records"):
        convert(tmp_path, [row, dict(row, target_story="A2")])


def test_historical_rows_without_prompt_ids_remain_supported(tmp_path, row):
    row.pop("stage1_prompt_record_id")
    row.pop("stage2_prompt_record_id")
    output = convert(tmp_path, [row])
    with output["manifest_csv"].open() as handle:
        records = list(csv.DictReader(handle))
    assert all(record["prompt_record_id"] == "" for record in records)
    assert all(record["parent_prompt_record_id"] == "" for record in records)


@pytest.mark.parametrize("content", [
    "bundle_id,bundle_id\nx,y\n",
    "bundle_id,pair_id\nx,y\n",
    "bundle_id,pair_id,target_story,stage1_td_story,stage2_sli_story\nx,y\n",
    "bundle_id,pair_id,target_story,stage1_td_story,stage2_sli_story\nx,y,A1,story,story,extra\n",
    "bundle_id,pair_id,target_story,stage1_td_story,stage2_sli_story\n",
])
def test_malformed_or_incomplete_csv_is_rejected(tmp_path, content):
    source = tmp_path / "source.csv"
    source.write_text(content)
    with pytest.raises(ValueError):
        convert_generated_story_csv_to_cha(input_csv=source, output_dir=tmp_path / "chat")
    assert not (tmp_path / "chat").exists()
