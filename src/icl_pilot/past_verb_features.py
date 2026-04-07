from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


_UMOR_RE = re.compile(r"^%umor:\t(.*)$")

_PAST_OVERGENERALIZATION_CODES = [
    "err_m_c_eq_ed",
    "err_m_c_eq_ed_c_i",
    "err_m_c_plus_ed",
    "err_m_c_plus_plus_ed",
    "err_m_c_plus_plus_ed_c_i",
]

_BARE_PAST_CODES = [
    "err_m_c_0ed",
    "err_m_c_base_c_ed",
]


def _count_overt_past_finite_verbs(path: Path) -> int:
    count = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _UMOR_RE.match(line)
        if not match:
            continue
        for token in match.group(1).split():
            if not token.startswith("verb|"):
                continue
            if "-Fin-" in token and "-Past-" in token:
                count += 1
    return count


def build_past_verb_feature_table(
    transcript_root: str,
    error_code_csv: str,
    output_csv: str,
    output_summary_json: str,
) -> int:
    transcript_root_path = Path(transcript_root).expanduser().resolve()
    error_code_path = Path(error_code_csv).expanduser().resolve()
    out_csv = Path(output_csv).expanduser().resolve()
    out_summary = Path(output_summary_json).expanduser().resolve()

    error_df = pd.read_csv(error_code_path)
    required = ["File_ID", "error_source", "error_relpath", *_PAST_OVERGENERALIZATION_CODES, *_BARE_PAST_CODES]
    missing = [column for column in required if column not in error_df.columns]
    if missing:
        raise ValueError(f"Missing required error-code columns: {', '.join(missing)}")

    rows: list[dict[str, object]] = []
    for row in error_df.to_dict("records"):
        transcript_path = Path(str(row["error_source"]))
        if not transcript_path.is_absolute():
            transcript_path = transcript_root_path / transcript_path
        overt_past = _count_overt_past_finite_verbs(transcript_path)
        bare_past = sum(int(row.get(column, 0) or 0) for column in _BARE_PAST_CODES)
        opportunities = overt_past + bare_past
        overgen = sum(int(row.get(column, 0) or 0) for column in _PAST_OVERGENERALIZATION_CODES)
        rate = (overgen / opportunities) if opportunities else 0.0
        rows.append(
            {
                "File_ID": int(row["File_ID"]),
                "error_source": str(row["error_source"]),
                "error_relpath": str(row["error_relpath"]),
                "overt_past_finite_verbs": overt_past,
                "bare_past_error_tokens": bare_past,
                "past_verb_opportunities": opportunities,
                "past_overgeneralization_errors": overgen,
                "past_overgeneralization_errors_per_past_opp": rate,
            }
        )

    feature_df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(out_csv, index=False)

    summary = {
        "transcript_root": str(transcript_root_path),
        "error_code_csv": str(error_code_path),
        "output_csv": str(out_csv),
        "rows": int(len(feature_df)),
        "past_overgeneralization_codes": _PAST_OVERGENERALIZATION_CODES,
        "bare_past_codes": _BARE_PAST_CODES,
        "totals": {
            "overt_past_finite_verbs": int(feature_df["overt_past_finite_verbs"].sum()),
            "bare_past_error_tokens": int(feature_df["bare_past_error_tokens"].sum()),
            "past_verb_opportunities": int(feature_df["past_verb_opportunities"].sum()),
            "past_overgeneralization_errors": int(feature_df["past_overgeneralization_errors"].sum()),
        },
    }
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote past-verb feature table to {out_csv}")
    print(f"Wrote past-verb feature summary to {out_summary}")
    return 0


def merge_past_verb_features_into_master(
    master_csv: str,
    past_feature_csv: str,
    output_csv: str | None = None,
) -> int:
    master_path = Path(master_csv).expanduser().resolve()
    feature_path = Path(past_feature_csv).expanduser().resolve()
    output_path = Path(output_csv).expanduser().resolve() if output_csv else master_path

    master = pd.read_csv(master_path)
    feature_df = pd.read_csv(feature_path)

    feature_columns = [
        "overt_past_finite_verbs",
        "bare_past_error_tokens",
        "past_verb_opportunities",
        "past_overgeneralization_errors",
        "past_overgeneralization_errors_per_past_opp",
    ]
    missing = [column for column in ["File_ID", "error_relpath", *feature_columns] if column not in feature_df.columns]
    if missing:
        raise ValueError(f"Missing required past-verb columns: {', '.join(missing)}")

    for column in feature_columns:
        if column in master.columns:
            master = master.drop(columns=[column])

    features = feature_df.to_dict("records")
    merged_rows: list[dict[str, object]] = []
    matched = 0
    unmatched = 0
    ambiguous = 0

    for row in master.to_dict("records"):
        file_id = int(row.get("File_ID")) if pd.notna(row.get("File_ID")) else None
        file_kideval = str(row.get("File_kideval", "") or "")
        candidates = [item for item in features if int(item["File_ID"]) == file_id]
        selected: dict[str, object] | None = None
        if len(candidates) == 1:
            selected = candidates[0]
        elif len(candidates) > 1:
            suffix_matches = [item for item in candidates if str(item["error_relpath"]).endswith(file_kideval)]
            if len(suffix_matches) == 1:
                selected = suffix_matches[0]
            elif len(suffix_matches) > 1:
                ambiguous += 1
            else:
                ambiguous += 1

        if selected is not None:
            matched += 1
            for column in feature_columns:
                row[column] = selected[column]
        else:
            unmatched += 1
            for column in feature_columns:
                row[column] = 0
        merged_rows.append(row)

    pd.DataFrame(merged_rows).to_csv(output_path, index=False)
    print(f"Wrote master with past-verb features to {output_path}")
    print(f"Matched rows: {matched}")
    print(f"Unmatched rows: {unmatched}")
    print(f"Ambiguous rows: {ambiguous}")
    return 0
