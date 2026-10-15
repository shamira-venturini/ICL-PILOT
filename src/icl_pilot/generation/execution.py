"""Sequential two-stage execution shared by the Colab interface and checks."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .prompt_records import PromptJournal


def cohort_slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def run_generation(prepared, generate, *, run_id: str, save_every: int = 25,
                   on_cohort_complete=None, progress=None) -> list[Path]:
    """Generate in the frozen schedule order; never resume or replace a run.

    ``generate`` supplies the notebook's unchanged model/tokenizer call and
    records prompts/outputs in the supplied journal. No per-job reseeding occurs.
    """
    if type(save_every) is not int or save_every < 1:
        raise ValueError("save_every must be a positive integer")
    out = prepared.run_manifest_path.parent
    if any(out.glob("*story_generations.csv")) or (out / "prompt_records.jsonl").exists():
        raise FileExistsError("Generation artifacts exist. Choose a fresh run; resume is not implemented.")
    manifest = json.loads(prepared.run_manifest_path.read_text())
    journal = PromptJournal(out / "prompt_records.jsonl", run_id=run_id,
                            builder=prepared.prompt_builder, run_manifest=manifest)
    aggregate = out / "all_cohorts_rotating_story_generations.csv"
    paths, all_rows = [], []
    builder, model_settings = prepared.prompt_builder, prepared.config["model"]

    def status(value):
        manifest.update(status=value, completed_generation_jobs=len(all_rows))
        prepared.run_manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    status("generation_in_progress")
    try:
        for label, frame in prepared.run_df.groupby("cohort_label", sort=False):
            path = out / f"{cohort_slug(label)}_rotating_story_generations.csv"
            cohort_rows = []
            rows = frame.itertuples(index=False)
            if progress is not None:
                rows = progress(rows, total=len(frame), desc=f"Generate {label}")
            for row in rows:
                first, first_id = generate(
                    builder.stage1_messages(row), max_new_tokens=model_settings["max_new_tokens_stage1"],
                    row=row, stage="stage1_td", journal=journal,
                )
                second, second_id = generate(
                    builder.stage2_messages(row, first), max_new_tokens=model_settings["max_new_tokens_stage2"],
                    row=row, stage="stage2_sli", parent_record_id=first_id,
                    stage1_text=first, journal=journal,
                )
                names = ("cohort", "cohort_label", "round_id", "bundle_id", "pair_id", "replicate_id",
                         "story_slot_order", "target_story", "story_title", "sli_child_id", "td_child_id",
                         "sli_age", "td_age", "prompt_td_story_e1", "prompt_sli_story_e2", "story_units_text")
                record = {key: getattr(row, key) for key in names}
                record.update(stage1_group="TD", stage2_group="SLI", evaluation_td_group="TD",
                              evaluation_sli_group="SLI", stage1_td_story=first, stage2_sli_story=second,
                              stage1_prompt_record_id=first_id, stage2_prompt_record_id=second_id,
                              test_only=manifest.get("execution", {}).get("test_only", False))
                cohort_rows.append(record)
                all_rows.append(record)
                if len(cohort_rows) % save_every == 0:
                    pd.DataFrame(cohort_rows).to_csv(path, index=False)
            pd.DataFrame(cohort_rows).to_csv(path, index=False)
            pd.DataFrame(all_rows).to_csv(aggregate, index=False)
            paths.append(path)
            status("generation_in_progress")
            print(f"Saved {len(cohort_rows)} jobs for {label}: {path}")
            if on_cohort_complete is not None:
                on_cohort_complete(path)
    except BaseException:
        status("generation_interrupted; fresh run required")
        raise
    status("generation_complete")
    return paths
