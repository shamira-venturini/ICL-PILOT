from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


_META_COLUMNS = [
    "File_ID",
    "File_kideval",
    "Group",
    "Sex",
    "Age(Month)",
    "#utterances",
    "#words",
    "DSS_tbl_warning_50",
]

_ABILITY_COLUMNS = [
    "MLU_Morphemes_td_z",
    "VOCD_D_optimum_average_td_z",
    "IPSyn_Total_td_z",
    "DSS_tbl_score_td_z",
    "DSS_tbl_IP_td_z",
    "DSS_tbl_PP_td_z",
    "DSS_tbl_MV_td_z",
    "DSS_tbl_SV_td_z",
    "DSS_tbl_NG_td_z",
    "DSS_tbl_CNJ_td_z",
    "DSS_tbl_IR_td_z",
    "DSS_tbl_WHQ_td_z",
]

_BURDEN_COLUMNS = [
    "Word_Errors_per_utt_td_z",
    "retracing_per_utt_td_z",
    "repetition_per_utt_td_z",
    "bare_marking_errors_per_word_td_z",
    "pronoun_errors_per_word_td_z",
    "past_overgeneralization_errors_per_past_opp_td_z",
]

_DOMAIN_SPEC = {
    "severity_structural": [
        "MLU_Morphemes_severity_z",
        "IPSyn_Total_severity_z",
        "DSS_tbl_score_severity_z",
        "DSS_tbl_MV_severity_z",
    ],
    "severity_lexical": [
        "VOCD_D_optimum_average_severity_z",
    ],
    "severity_disruption": [
        "Word_Errors_per_utt_severity_z",
        "retracing_per_utt_severity_z",
        "repetition_per_utt_severity_z",
    ],
    "severity_morphosyntax_burden": [
        "bare_marking_errors_per_word_severity_z",
        "pronoun_errors_per_word_severity_z",
        "past_overgeneralization_errors_per_past_opp_severity_z",
    ],
    "profile_dss_subscales": [
        "DSS_tbl_IP_severity_z",
        "DSS_tbl_PP_severity_z",
        "DSS_tbl_MV_severity_z",
        "DSS_tbl_SV_severity_z",
        "DSS_tbl_NG_severity_z",
        "DSS_tbl_CNJ_severity_z",
        "DSS_tbl_IR_severity_z",
        "DSS_tbl_WHQ_severity_z",
    ],
}

_SEVERITY_CORE_COLUMNS = [
    "MLU_Morphemes_severity_z",
    "VOCD_D_optimum_average_severity_z",
    "IPSyn_Total_severity_z",
    "DSS_tbl_score_severity_z",
    "Word_Errors_per_utt_severity_z",
    "bare_marking_errors_per_word_severity_z",
]


def _strip_suffix(value: str, suffix: str) -> str:
    if value.endswith(suffix):
        return value[: -len(suffix)]
    return value


def build_severity_profile_table(
    input_csv: str,
    output_csv: str,
    output_summary_json: str,
) -> int:
    input_path = Path(input_csv).expanduser().resolve()
    out_csv = Path(output_csv).expanduser().resolve()
    out_summary = Path(output_summary_json).expanduser().resolve()

    df = pd.read_csv(input_path)

    missing = [
        column
        for column in _META_COLUMNS + _ABILITY_COLUMNS + _BURDEN_COLUMNS
        if column not in df.columns
    ]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    out = df[_META_COLUMNS].copy()

    for column in _ABILITY_COLUMNS:
        base = _strip_suffix(column, "_td_z")
        out[f"{base}_severity_z"] = -pd.to_numeric(df[column], errors="coerce")

    for column in _BURDEN_COLUMNS:
        base = _strip_suffix(column, "_td_z")
        out[f"{base}_severity_z"] = pd.to_numeric(df[column], errors="coerce")

    out["severity_structural"] = out[_DOMAIN_SPEC["severity_structural"]].mean(axis=1)
    out["severity_lexical"] = out[_DOMAIN_SPEC["severity_lexical"]].mean(axis=1)
    out["severity_disruption"] = out[_DOMAIN_SPEC["severity_disruption"]].mean(axis=1)
    out["severity_morphosyntax_burden"] = out[_DOMAIN_SPEC["severity_morphosyntax_burden"]].mean(axis=1)
    out["profile_dss_subscales"] = out[_DOMAIN_SPEC["profile_dss_subscales"]].mean(axis=1)

    out["severity_core_composite"] = out[_SEVERITY_CORE_COLUMNS].mean(axis=1)
    out["severity_domain_composite"] = out[
        [
            "severity_structural",
            "severity_lexical",
            "severity_disruption",
            "severity_morphosyntax_burden",
        ]
    ].mean(axis=1)
    out["severity_use_with_caution"] = out["DSS_tbl_warning_50"].astype(int)

    out.to_csv(out_csv, index=False)

    summary = {
        "input_csv": str(input_path),
        "output_csv": str(out_csv),
        "rows": len(out),
        "meta_columns": _META_COLUMNS,
        "ability_columns_flipped_to_severity_direction": _ABILITY_COLUMNS,
        "burden_columns_kept_in_severity_direction": _BURDEN_COLUMNS,
        "domain_spec": _DOMAIN_SPEC,
        "severity_core_columns": _SEVERITY_CORE_COLUMNS,
        "notes": [
            "Severity-direction z-scores are coded so higher values indicate more severe deviation from TD expectations.",
            "Ability measures were multiplied by -1 after TD-based age normalization.",
            "Burden/error measures were left in their original direction after TD-based age normalization.",
            "DSS_tbl_warning_50 is retained as a caution flag rather than used directly in the composite.",
        ],
    }
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote severity/profile table to {out_csv}")
    print(f"Wrote severity/profile summary to {out_summary}")
    return 0
