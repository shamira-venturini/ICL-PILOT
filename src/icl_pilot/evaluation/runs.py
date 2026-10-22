"""Separate framework-evaluation runs from preserved result snapshots."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re


def create_framework_evaluation_run(
    project_root: str | Path, *, run_id: str | None = None,
) -> Path:
    """Create a fresh run directory, refusing existing names and path traversal."""
    if run_id is None:
        run_id = "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("Evaluation run_id must be a nonempty directory name without path separators.")
    root = Path(project_root).expanduser().resolve()
    destination = root / "results/framework_based_evaluation" / run_id
    destination.mkdir(parents=True, exist_ok=False)
    return destination
