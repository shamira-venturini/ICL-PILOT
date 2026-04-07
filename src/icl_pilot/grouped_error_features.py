from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd


_GROUP_TO_CODES = {
    "bare_marking_errors": [
        "err_m_c_03s_c_a",
        "err_m_c_vsg_c_a",
        "err_m_c_vun_c_a",
        "err_m_c_0ed",
        "err_m_c_base_c_ed",
        "err_m_c_0_apos_s",
        "err_m_c_0s",
        "err_m_c_0s_c_a",
        "err_m_c_0es",
    ],
    "determiner_errors": [
        "err_s_c_r_c_gc_c_det",
    ],
    "pronoun_errors": [
        "err_s_c_r_c_gc_c_pro",
    ],
    "rule_generalization_errors": [
        "err_m_c_plus_3s",
        "err_m_c_plus_3s_c_a",
        "err_m_c_plus_plus_3s",
        "err_m_c_eq_ed",
        "err_m_c_eq_ed_c_i",
        "err_m_c_eq_en",
        "err_m_c_plus_ed",
        "err_m_c_plus_plus_ed",
        "err_m_c_plus_plus_ed_c_i",
        "err_m_c_plus_plus_en",
        "err_m_c_plus_plus_en_c_i",
        "err_m_c_plus_es",
    ],
}


_DISPLAY_CODE_MAP = {
    "err_m_c_03s_c_a": "m:03s:a",
    "err_m_c_0_apos_s": "m:0's",
    "err_m_c_0ed": "m:0ed",
    "err_m_c_0es": "m:0es",
    "err_m_c_0s": "m:0s",
    "err_m_c_0s_c_a": "m:0s:a",
    "err_m_c_plus_es": "m:+es",
    "err_m_c_plus_3s": "m:+3s",
    "err_m_c_plus_3s_c_a": "m:+3s:a",
    "err_m_c_plus_plus_3s": "m:++3s",
    "err_m_c_allo": "m:allo",
    "err_m_c_base_c_ed": "m:base:ed",
    "err_m_c_eq_ed": "m:=ed",
    "err_m_c_eq_ed_c_i": "m:=ed:i",
    "err_m_c_eq_en": "m:=en",
    "err_m_c_eq_s": "m:=s",
    "err_m_c_irr_c_ed": "m:irr:ed",
    "err_m_c_irr_c_s": "m:irr:s",
    "err_m_c_plus_ed": "m:+ed",
    "err_m_c_plus_plus_apos_s": "m:++'s",
    "err_m_c_plus_plus_ed": "m:++ed",
    "err_m_c_plus_plus_ed_c_i": "m:++ed:i",
    "err_m_c_plus_plus_en": "m:++en",
    "err_m_c_plus_plus_en_c_i": "m:++en:i",
    "err_m_c_plus_plus_est": "m:++est",
    "err_m_c_plus_plus_s": "m:++s",
    "err_m_c_plus_plus_s_c_i": "m:++s:i",
    "err_m_c_plus_s": "m:+s",
    "err_m_c_plus_s_c_a": "m:+s:a",
    "err_m_c_sub_c_ed": "m:sub:ed",
    "err_m_c_vsg_c_a": "m:vsg:a",
    "err_m_c_vun_c_a": "m:vun:a",
    "err_m_dash_ed": "m-ed",
    "err_s_c_r_c_gc_c_det": "s:r:gc:det",
    "err_s_c_r_c_gc_c_pro": "s:r:gc:pro",
}


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_grouped_error_feature_table(
    input_csv: str,
    output_csv: str,
    output_summary_json: str,
) -> int:
    input_path = Path(input_csv).expanduser().resolve()
    out_csv = Path(output_csv).expanduser().resolve()
    out_summary = Path(output_summary_json).expanduser().resolve()

    _, rows = _read_csv(input_path)

    grouped_rows: list[dict[str, str]] = []
    nonzero_files: dict[str, int] = {group: 0 for group in _GROUP_TO_CODES}
    total_counts: dict[str, int] = {group: 0 for group in _GROUP_TO_CODES}

    for row in rows:
        chi_utterances = float(row.get("chi_utterances", "0") or 0)
        grouped = {
            "File_ID": row.get("File_ID", ""),
            "error_source": row.get("error_source", ""),
            "error_relpath": row.get("error_relpath", ""),
            "chi_utterances": row.get("chi_utterances", ""),
            "chi_utterances_with_error_code": row.get("chi_utterances_with_error_code", ""),
            "error_total_annotations": row.get("error_total_annotations", ""),
        }
        for group, columns in _GROUP_TO_CODES.items():
            total = sum(int(row.get(column, "0") or 0) for column in columns)
            grouped[group] = str(total)
            grouped[f"{group}_per_utt"] = f"{(total / chi_utterances) if chi_utterances else 0.0:.6f}"
            total_counts[group] += total
            if total > 0:
                nonzero_files[group] += 1
        grouped_rows.append(grouped)

    fieldnames = [
        "File_ID",
        "error_source",
        "error_relpath",
        "chi_utterances",
        "chi_utterances_with_error_code",
        "error_total_annotations",
    ]
    for group in _GROUP_TO_CODES:
        fieldnames.extend([group, f"{group}_per_utt"])
    _write_csv(out_csv, fieldnames, grouped_rows)

    summary = {
        "input_csv": str(input_path),
        "output_csv": str(out_csv),
        "rows": len(grouped_rows),
        "group_to_codes": {
            group: [
                {"feature_column": column, "source_codes": _DISPLAY_CODE_MAP.get(column, column)}
                for column in columns
            ]
            for group, columns in _GROUP_TO_CODES.items()
        },
        "nonzero_file_counts": nonzero_files,
        "nonzero_file_proportions": {
            group: (count / len(grouped_rows)) if grouped_rows else 0.0
            for group, count in nonzero_files.items()
        },
        "total_counts": total_counts,
    }
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote grouped error features to {out_csv}")
    print(f"Wrote grouped error summary to {out_summary}")
    return 0


def merge_grouped_error_features_into_master(
    master_csv: str,
    grouped_csv: str,
    output_csv: str | None = None,
) -> int:
    master_path = Path(master_csv).expanduser().resolve()
    grouped_path = Path(grouped_csv).expanduser().resolve()
    output_path = Path(output_csv).expanduser().resolve() if output_csv else master_path

    master = pd.read_csv(master_path)
    grouped = pd.read_csv(grouped_path)

    feature_columns = [
        "chi_utterances_with_error_code",
        "error_total_annotations",
        "bare_marking_errors",
        "bare_marking_errors_per_utt",
        "determiner_errors",
        "determiner_errors_per_utt",
        "pronoun_errors",
        "pronoun_errors_per_utt",
        "rule_generalization_errors",
        "rule_generalization_errors_per_utt",
    ]
    missing = [column for column in ["File_ID", "error_relpath", *feature_columns] if column not in grouped.columns]
    if missing:
        raise ValueError(f"Missing required grouped-error columns: {', '.join(missing)}")

    for column in feature_columns:
        if column in master.columns:
            master = master.drop(columns=[column])

    grouped_rows = grouped.to_dict("records")

    merged_rows: list[dict[str, object]] = []
    matched = 0
    unmatched = 0
    ambiguous = 0

    for row in master.to_dict("records"):
        file_id = int(row.get("File_ID")) if pd.notna(row.get("File_ID")) else None
        file_kideval = str(row.get("File_kideval", "") or "")
        candidates = [item for item in grouped_rows if int(item["File_ID"]) == file_id]
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
    print(f"Wrote master with grouped error features to {output_path}")
    print(f"Matched rows: {matched}")
    print(f"Unmatched rows: {unmatched}")
    print(f"Ambiguous rows: {ambiguous}")
    return 0
