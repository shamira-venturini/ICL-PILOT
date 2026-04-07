from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


_STORY_ORDER = ["A1", "A2", "A3", "B1", "B2", "B3"]
_GROUP_DOMAIN_COLUMNS = [
    "severity_structural",
    "severity_lexical",
    "severity_disruption",
    "severity_morphosyntax_burden",
]
_STORY_RE = re.compile(r"^@G:\s+(\S+)\s*$", re.MULTILINE)


def _cohort_label(age_months: float) -> str:
    return f"{int(age_months) // 12}-year-old"


def _profile_label(row: pd.Series, margin: float = 0.35) -> str:
    values = {column: float(row[column]) for column in _GROUP_DOMAIN_COLUMNS}
    ranked = sorted(values.items(), key=lambda item: item[1], reverse=True)
    top_name, top_value = ranked[0]
    second_value = ranked[1][1]
    if top_value - second_value < margin:
        return "balanced"
    mapping = {
        "severity_structural": "structural",
        "severity_lexical": "lexical",
        "severity_disruption": "disruption",
        "severity_morphosyntax_burden": "morphosyntax_burden",
    }
    return mapping[top_name]


def _read_story_ids(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    seen: list[str] = []
    for story in _STORY_RE.findall(text):
        if story in _STORY_ORDER and story not in seen:
            seen.append(story)
    return seen


def _resolve_transcript_path(
    transcript_root: Path,
    group: str,
    file_kideval: str,
) -> Path:
    direct = transcript_root / group / file_kideval
    if direct.exists():
        return direct

    a_fallback = transcript_root / group / "A" / file_kideval
    if a_fallback.exists():
        return a_fallback

    b_fallback = transcript_root / group / "B" / file_kideval
    if b_fallback.exists():
        return b_fallback

    return direct


def _assign_prompt_story(group: pd.DataFrame) -> dict[str, str]:
    counts = {story: 0 for story in _STORY_ORDER}
    assignments: dict[str, str] = {}
    for _, row in group.sort_values(["File_ID", "File_kideval"]).iterrows():
        file_key = str(row["File_kideval"])
        available = [story for story in row["available_stories"] if story in _STORY_ORDER]
        if not available:
            continue
        chosen = min(available, key=lambda story: (counts[story], _STORY_ORDER.index(story)))
        assignments[file_key] = chosen
        counts[chosen] += 1
    return assignments


def build_story_generation_design(
    severity_banded_csv: str,
    counterbalance_csv: str,
    transcript_root: str,
    output_dir: str,
    min_prompt_candidates: int = 2,
    min_eval_candidates: int = 3,
) -> int:
    severity_path = Path(severity_banded_csv).expanduser().resolve()
    counterbalance_path = Path(counterbalance_csv).expanduser().resolve()
    transcript_root_path = Path(transcript_root).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    severity = pd.read_csv(severity_path)
    counterbalance = pd.read_csv(counterbalance_path)

    design = severity.copy()
    design = design[design["Group"].isin(["SLI", "TD"])].copy()
    design["Age_cohort"] = pd.to_numeric(design["Age(Month)"], errors="coerce").map(_cohort_label)
    design["severity_band_for_design"] = design["severity_band_sli_tertile_sli_only"].fillna("")
    design.loc[design["Group"] == "TD", "severity_band_for_design"] = "typical"
    design["profile_label_for_design"] = design.apply(
        lambda row: "typical" if row["Group"] == "TD" else _profile_label(row),
        axis=1,
    )
    design["transcript_path"] = design.apply(
        lambda row: _resolve_transcript_path(
            transcript_root=transcript_root_path,
            group=row["Group"],
            file_kideval=str(row["File_kideval"]),
        ),
        axis=1,
    )
    design["available_stories"] = design["transcript_path"].map(_read_story_ids)

    assignment_rows: list[dict[str, object]] = []
    grouped = design.groupby(
        ["Age_cohort", "Group", "severity_band_for_design", "profile_label_for_design"],
        dropna=False,
    )
    prompt_assignments: dict[str, str] = {}
    for _, group in grouped:
        prompt_assignments.update(_assign_prompt_story(group))

    for _, row in design.iterrows():
        prompt_story = prompt_assignments.get(str(row["File_kideval"]), "")
        for story in row["available_stories"]:
            assignment_rows.append(
                {
                    "File_ID": int(row["File_ID"]),
                    "File_kideval": row["File_kideval"],
                    "Group": row["Group"],
                    "Age(Month)": int(row["Age(Month)"]),
                    "Age_cohort": row["Age_cohort"],
                    "severity_band": row["severity_band_for_design"],
                    "profile_label": row["profile_label_for_design"],
                    "story_id": story,
                    "pool_role": "prompt" if story == prompt_story else "eval",
                    "severity_core_composite": row["severity_core_composite"],
                    "severity_domain_composite": row["severity_domain_composite"],
                    "severity_structural": row["severity_structural"],
                    "severity_lexical": row["severity_lexical"],
                    "severity_disruption": row["severity_disruption"],
                    "severity_morphosyntax_burden": row["severity_morphosyntax_burden"],
                }
            )

    assignment_df = pd.DataFrame(assignment_rows)
    assignment_df.to_csv(out_dir / "story_pool_assignment.csv", index=False)

    prompt_df = assignment_df[assignment_df["pool_role"] == "prompt"].copy()
    eval_df = assignment_df[assignment_df["pool_role"] == "eval"].copy()

    cell_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    config_counter = 1
    for _, rule in counterbalance.iterrows():
        target_story = rule["target_story"]
        e1_story = rule["e1_story"]
        e2_story = rule["e2_story"]

        eligible = assignment_df[assignment_df["story_id"] == target_story]
        strata = (
            eligible[
                ["Age_cohort", "Group", "severity_band", "profile_label"]
            ]
            .drop_duplicates()
            .sort_values(["Age_cohort", "Group", "severity_band", "profile_label"])
        )

        for _, stratum in strata.iterrows():
            age_cohort = stratum["Age_cohort"]
            group = stratum["Group"]
            severity_band = stratum["severity_band"]
            profile_label = stratum["profile_label"]

            prompt_matches = prompt_df[
                (prompt_df["story_id"] == target_story)
                & (prompt_df["Age_cohort"] == age_cohort)
                & (prompt_df["Group"] == group)
                & (prompt_df["severity_band"] == severity_band)
                & (prompt_df["profile_label"] == profile_label)
            ]
            eval_matches = eval_df[
                (eval_df["story_id"] == target_story)
                & (eval_df["Age_cohort"] == age_cohort)
                & (eval_df["Group"] == group)
                & (eval_df["severity_band"] == severity_band)
                & (eval_df["profile_label"] == profile_label)
            ]

            prompt_ids = sorted(prompt_matches["File_ID"].astype(int).astype(str).unique().tolist())
            eval_ids = sorted(eval_matches["File_ID"].astype(int).astype(str).unique().tolist())
            n_prompt = len(prompt_ids)
            n_eval = len(eval_ids)
            cell_status = (
                "usable"
                if n_prompt >= min_prompt_candidates and n_eval >= min_eval_candidates
                else "sparse"
            )

            cell_row = {
                "Age_cohort": age_cohort,
                "target_story": target_story,
                "target_group": group,
                "target_severity_band": severity_band,
                "target_profile_label": profile_label,
                "e1_story": e1_story,
                "e2_story": e2_story,
                "n_prompt_candidates": n_prompt,
                "n_eval_candidates": n_eval,
                "prompt_subject_ids": "/".join(prompt_ids),
                "eval_subject_ids": "/".join(eval_ids),
                "cell_status": cell_status,
            }
            cell_rows.append(cell_row)

            manifest_rows.append(
                {
                    "config_id": f"cfg_{config_counter:04d}",
                    **cell_row,
                    "default_n_outputs": 100 if group == "SLI" else 10,
                }
            )
            config_counter += 1

    cells_df = pd.DataFrame(cell_rows)
    cells_df.to_csv(out_dir / "story_generation_cells.csv", index=False)

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(out_dir / "story_generation_manifest_v2.csv", index=False)

    summary = {
        "severity_banded_csv": str(severity_path),
        "counterbalance_csv": str(counterbalance_path),
        "transcript_root": str(transcript_root_path),
        "min_prompt_candidates": min_prompt_candidates,
        "min_eval_candidates": min_eval_candidates,
        "children": int(design["File_ID"].nunique()),
        "story_pool_rows": int(len(assignment_df)),
        "generation_cells": int(len(cells_df)),
        "usable_cells": int((cells_df["cell_status"] == "usable").sum()),
        "sparse_cells": int((cells_df["cell_status"] == "sparse").sum()),
        "stories_per_child_summary": (
            assignment_df.groupby("File_ID")["story_id"].nunique().describe().to_dict()
            if not assignment_df.empty
            else {}
        ),
    }
    (out_dir / "story_generation_design_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote story pool assignment to {out_dir / 'story_pool_assignment.csv'}")
    print(f"Wrote story generation cells to {out_dir / 'story_generation_cells.csv'}")
    print(f"Wrote story generation manifest to {out_dir / 'story_generation_manifest_v2.csv'}")
    print(f"Wrote story generation design summary to {out_dir / 'story_generation_design_summary.json'}")
    return 0
