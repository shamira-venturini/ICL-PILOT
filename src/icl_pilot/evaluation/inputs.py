"""Explicit input paths for framework evaluation of a selected experiment."""
from __future__ import annotations

import json
from pathlib import Path


def load_evaluation_inputs(project_root: str | Path, *, config_path: str | Path | None = None) -> dict[str, Path]:
    """Resolve both required CSVs before creating any evaluation outputs."""
    root = Path(project_root).expanduser().resolve()
    path = Path(config_path) if config_path is not None else root / "configs/evaluation/inputs.json"
    if not path.is_absolute():
        path = root / path
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported framework input configuration schema.")
    fields = ("text_records_csv", "gillam_external_text_csv")
    pending = [field for field in fields if not isinstance(config.get(field), str) or not config[field].strip()]
    if pending:
        raise RuntimeError("Revised-study evaluation inputs are pending. Update configs/evaluation/inputs.json: "
                           + ", ".join(pending) + ". Historical Phase 2 inputs are not selected automatically.")
    resolved = {}
    for field in fields:
        source = Path(config[field]).expanduser()
        source = (source if source.is_absolute() else root / source).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Configured evaluation input {field} is missing or is a directory: {source}")
        resolved[field] = source
    return resolved
