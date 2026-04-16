from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .severity_features import _COLUMN_DECISIONS, _KEEP_ORDER


_GENERATED_METADATA_COLUMNS = [
    "bundle_id",
    "pair_id",
    "round_id",
    "replicate_id",
    "source",
    "stage",
    "comparison_target_group",
    "evaluation_pool_policy",
    "prompt_exclusion_policy",
    "source_child_id",
    "prompt_td_child_id",
    "prompt_sli_child_id",
    "td_child_id",
    "sli_child_id",
    "sli_age_generated_from",
    "sli_severity_band_generated_from",
    "sli_profile_label_generated_from",
    "story_slots",
    "n_stories",
]


_GENERATED_FALLBACKS = {
    "*-3S": "*-S3",
    "irr-3S": "irr-S3",
}


def build_generated_severity_feature_table(
    input_csv: str,
    output_csv: str,
    output_spec_csv: str,
    output_summary_json: str,
) -> int:
    input_path = Path(input_csv).expanduser().resolve()
    out_csv = Path(output_csv).expanduser().resolve()
    out_spec = Path(output_spec_csv).expanduser().resolve()
    out_summary = Path(output_summary_json).expanduser().resolve()

    df = pd.read_csv(input_path)

    # Generated bundle measures omit a few archival real-corpus columns and
    # sometimes rename morphology columns in the CLAN export.
    for canonical, fallback in _GENERATED_FALLBACKS.items():
        if canonical not in df.columns and fallback in df.columns:
            df[canonical] = df[fallback]

    drop_columns = [column for column in df.columns if column.endswith("_mlt")]
    for fallback in _GENERATED_FALLBACKS.values():
        if fallback in df.columns:
            drop_columns.append(fallback)
    if drop_columns:
        df = df.drop(columns=sorted(set(drop_columns)))

    optional_missing = {"IPSyn_Utts", "IPSyn_Total", "irr-PAST"}
    for column in optional_missing:
        if column not in df.columns:
            df[column] = pd.NA

    missing_keep = [column for column in _KEEP_ORDER if column not in df.columns]
    if missing_keep:
        raise ValueError(f"Missing required keep columns for generated severity features: {', '.join(missing_keep)}")

    missing_decisions = [column for column in df.columns if column not in _COLUMN_DECISIONS and column not in _GENERATED_METADATA_COLUMNS]
    if missing_decisions:
        raise ValueError(f"Columns missing spec decisions: {', '.join(missing_decisions)}")

    keep_columns = [column for column in _GENERATED_METADATA_COLUMNS if column in df.columns] + _KEEP_ORDER
    filtered = df[keep_columns].copy()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(out_csv, index=False)

    spec_rows: list[dict[str, str]] = []
    for column in df.columns:
        if column in _GENERATED_METADATA_COLUMNS:
            spec_rows.append(
                {
                    "column": column,
                    "action": "keep",
                    "category": "generated_metadata",
                    "included_in_severity_features_raw": "1",
                    "reason": "Preserved for round-aware prompt exclusion and bundle-level evaluation.",
                }
            )
            continue
        action, category, reason = _COLUMN_DECISIONS[column]
        spec_rows.append(
            {
                "column": column,
                "action": action,
                "category": category,
                "included_in_severity_features_raw": "1" if column in keep_columns else "0",
                "reason": reason,
            }
        )

    spec_df = pd.DataFrame(spec_rows)
    spec_df.to_csv(out_spec, index=False)

    summary = {
        "input_csv": str(input_path),
        "output_csv": str(out_csv),
        "rows": int(len(filtered)),
        "columns_kept": int(len(keep_columns)),
        "kept_columns": keep_columns,
        "generated_metadata_columns": [column for column in _GENERATED_METADATA_COLUMNS if column in df.columns],
        "dropped_columns": [column for column in df.columns if column not in keep_columns],
    }
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote generated severity feature table to {out_csv}")
    print(f"Wrote generated severity feature spec to {out_spec}")
    print(f"Wrote generated severity feature summary to {out_summary}")
    return 0
