from pathlib import Path
import json

import pytest

from icl_pilot.evaluation.runs import create_framework_evaluation_run
from icl_pilot.evaluation.inputs import load_evaluation_inputs


ROOT = Path(__file__).resolve().parents[1]


def test_new_framework_runs_have_distinct_directories(tmp_path):
    first = create_framework_evaluation_run(tmp_path)
    (first / "result.json").write_text('{"preserve": true}\n')
    second = create_framework_evaluation_run(tmp_path)
    assert first != second
    assert first.parent == second.parent == tmp_path / "results/framework_based_evaluation"
    assert (first / "result.json").read_text() == '{"preserve": true}\n'


def test_existing_run_is_refused_without_changing_results(tmp_path):
    first = create_framework_evaluation_run(tmp_path, run_id="run_20260827T153330Z")
    result = first / "result.json"
    result.write_bytes(b"historical result\n")
    with pytest.raises(FileExistsError):
        create_framework_evaluation_run(tmp_path, run_id=first.name)
    assert result.read_bytes() == b"historical result\n"


@pytest.mark.parametrize("run_id", ["", ".", "..", "../old", "/absolute", "nested/run", 12])
def test_invalid_run_names_do_not_create_results(tmp_path, run_id):
    with pytest.raises(ValueError, match="run_id"):
        create_framework_evaluation_run(tmp_path, run_id=run_id)
    assert not (tmp_path / "results").exists()


def test_notebook_configuration_creates_new_runs_beside_preserved_results(tmp_path):
    for relative in [
        "data/new_run/text_records.csv",
        "data/reference/external_text_records.csv",
    ]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture input\n")
    config = tmp_path / "configs/evaluation/inputs.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"schema_version": 1,
        "text_records_csv": "data/new_run/text_records.csv",
        "gillam_external_text_csv": "data/reference/external_text_records.csv"}))
    historical = create_framework_evaluation_run(tmp_path, run_id="run_20260827T153330Z")
    (historical / "result.json").write_bytes(b"historical result\n")
    notebook = json.loads((ROOT / "notebooks/synthtexteval_demo.ipynb").read_text())
    cell = next(c for c in notebook["cells"] if c.get("id") == "8YPXv35RHGG4")
    # Run the real path/settings block without executing IPython checkout setup.
    source = "".join(cell["source"])
    settings = source[source.index("EVALUATION_INPUT_CONFIG ="):]
    namespace = {"PROJECT_ROOT": tmp_path, "Path": Path,
                 "create_framework_evaluation_run": create_framework_evaluation_run,
                 "load_evaluation_inputs": load_evaluation_inputs}
    exec(settings, namespace)
    first = namespace["RESULTS_DIR"]
    exec(settings, namespace)
    assert namespace["RESULTS_DIR"] != first != historical
    assert namespace["RESULTS_DIR"].parent == historical.parent
    assert (historical / "result.json").read_bytes() == b"historical result\n"


def test_pending_notebook_inputs_stop_before_creating_results(tmp_path):
    config = tmp_path / "configs/evaluation/inputs.json"
    config.parent.mkdir(parents=True)
    config.write_bytes((ROOT / "configs/evaluation/inputs.json").read_bytes())
    notebook = json.loads((ROOT / "notebooks/synthtexteval_demo.ipynb").read_text())
    source = "".join(next(c for c in notebook["cells"] if c.get("id") == "8YPXv35RHGG4")["source"])
    namespace = {"PROJECT_ROOT": tmp_path, "Path": Path,
                 "create_framework_evaluation_run": create_framework_evaluation_run,
                 "load_evaluation_inputs": load_evaluation_inputs}
    with pytest.raises(RuntimeError, match="inputs are pending"):
        exec(source[source.index("EVALUATION_INPUT_CONFIG ="):], namespace)
    assert not (tmp_path / "results").exists()
