import json

import pytest

from icl_pilot.evaluation.inputs import load_evaluation_inputs


def config_at(root, **values):
    path = root / "configs/evaluation/inputs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, **values}))
    return path


@pytest.mark.parametrize("value", [None, "", "  ", 12])
def test_pending_inputs_have_no_historical_fallback(tmp_path, value):
    config_at(tmp_path, text_records_csv=value, gillam_external_text_csv=value)
    with pytest.raises(RuntimeError, match="inputs are pending"):
        load_evaluation_inputs(tmp_path)
    assert not (tmp_path / "results").exists()


def test_paths_resolve_against_checkout_and_accept_explicit_external_files(tmp_path):
    story = tmp_path / "selected run.csv"
    external = tmp_path / "external.csv"
    story.write_text("fixture\n")
    external.write_text("fixture\n")
    config_at(tmp_path, text_records_csv=story.name, gillam_external_text_csv=str(external))
    resolved = load_evaluation_inputs(tmp_path)
    assert resolved == {"text_records_csv": story, "gillam_external_text_csv": external}


@pytest.mark.parametrize("source", ["missing.csv", "."])
def test_unavailable_inputs_do_not_create_output_directories(tmp_path, source):
    config_at(tmp_path, text_records_csv=source, gillam_external_text_csv=source)
    with pytest.raises(FileNotFoundError, match="missing or is a directory"):
        load_evaluation_inputs(tmp_path)
    assert not (tmp_path / "results").exists()
