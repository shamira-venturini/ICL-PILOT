from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from patsy import dmatrix


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


def _safe_name(column: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", column).strip("_")
    return cleaned or "feature"


def _ols_fit(design: pd.DataFrame, target: pd.Series) -> tuple[np.ndarray, pd.Series, float]:
    x = np.asarray(design, dtype=float)
    y = np.asarray(target, dtype=float)
    coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ coef
    residuals = y - fitted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return coef, pd.Series(fitted, index=target.index), r2


def _predict_from_design(design: pd.DataFrame, coef: np.ndarray, index: pd.Index) -> pd.Series:
    x = np.asarray(design, dtype=float)
    return pd.Series(x @ coef, index=index)


def _linear_design(age: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"const": 1.0, "age": age.astype(float)}, index=age.index)


def _spline_design(
    age: pd.Series,
    lower_bound: float,
    upper_bound: float,
    df: int = 4,
) -> pd.DataFrame:
    spline = dmatrix(
        f"bs(age, df={df}, degree=3, include_intercept=False, "
        f"lower_bound={lower_bound}, upper_bound={upper_bound})",
        {"age": age.astype(float)},
        return_type="dataframe",
    )
    spline.insert(0, "const", 1.0)
    return spline


def build_age_adjusted_severity_table(
    input_csv: str,
    output_csv: str,
    output_model_csv: str,
    output_summary_json: str,
) -> int:
    input_path = Path(input_csv).expanduser().resolve()
    out_csv = Path(output_csv).expanduser().resolve()
    out_model = Path(output_model_csv).expanduser().resolve()
    out_summary = Path(output_summary_json).expanduser().resolve()

    df = pd.read_csv(input_path)
    missing_meta = [column for column in _META_COLUMNS if column not in df.columns]
    if missing_meta:
        raise ValueError(f"Missing required metadata columns: {', '.join(missing_meta)}")

    feature_columns = [column for column in df.columns if column not in _META_COLUMNS]
    age = pd.to_numeric(df["Age(Month)"], errors="coerce")
    td_mask = df["Group"].eq("TD") & age.notna()

    if int(td_mask.sum()) < 20:
        raise ValueError("Not enough TD rows to perform age adjustment.")

    lower_bound = float(age.min())
    upper_bound = float(age.max())
    adjusted = df.copy()
    model_rows: list[dict[str, object]] = []

    for feature in feature_columns:
        value = pd.to_numeric(df[feature], errors="coerce")
        valid_td = td_mask & value.notna()
        td_n = int(valid_td.sum())
        if td_n < 20:
            model_rows.append(
                {
                    "feature": feature,
                    "feature_safe": _safe_name(feature),
                    "td_n": td_n,
                    "method": "skipped",
                    "reason": "fewer_than_20_td_rows",
                }
            )
            continue

        td_age = age.loc[valid_td]
        td_value = value.loc[valid_td]

        spline_design_td = _spline_design(
            age=td_age,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        spline_coef, spline_fitted_td, spline_r2 = _ols_fit(
            design=spline_design_td,
            target=td_value,
        )
        linear_design_td = _linear_design(td_age)
        linear_coef, linear_fitted_td, linear_r2 = _ols_fit(
            design=linear_design_td,
            target=td_value,
        )

        spline_expected_all = _predict_from_design(
            design=_spline_design(
                age=age,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            ),
            coef=spline_coef,
            index=age.index,
        )
        linear_expected_all = _predict_from_design(
            design=_linear_design(age),
            coef=linear_coef,
            index=age.index,
        )

        spline_resid_td = td_value - spline_fitted_td
        linear_resid_td = td_value - linear_fitted_td
        spline_resid_sd = float(spline_resid_td.std(ddof=1))
        linear_resid_sd = float(linear_resid_td.std(ddof=1))

        safe = _safe_name(feature)
        adjusted[f"{safe}_td_expected"] = spline_expected_all
        adjusted[f"{safe}_td_z"] = (value - spline_expected_all) / spline_resid_sd
        adjusted[f"{safe}_td_linear_expected"] = linear_expected_all
        adjusted[f"{safe}_td_linear_z"] = (value - linear_expected_all) / linear_resid_sd

        available = value.notna()
        corr_z = adjusted.loc[available, f"{safe}_td_z"].corr(
            adjusted.loc[available, f"{safe}_td_linear_z"]
        )

        model_rows.append(
            {
                "feature": feature,
                "feature_safe": safe,
                "td_n": td_n,
                "td_age_min": float(td_age.min()),
                "td_age_max": float(td_age.max()),
                "spline_resid_sd": spline_resid_sd,
                "spline_r2": spline_r2,
                "linear_resid_sd": linear_resid_sd,
                "linear_r2": linear_r2,
                "corr_spline_vs_linear_z": float(corr_z) if pd.notna(corr_z) else "",
                "method": "td_cubic_b_spline_df4",
                "reason": "",
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    adjusted.to_csv(out_csv, index=False)

    model_df = pd.DataFrame(model_rows)
    model_df.to_csv(out_model, index=False)

    summary = {
        "input_csv": str(input_path),
        "output_csv": str(out_csv),
        "output_model_csv": str(out_model),
        "reference_group": "TD",
        "reference_group_n": int(td_mask.sum()),
        "non_td_rows": int((~df["Group"].eq("TD")).sum()),
        "age_column": "Age(Month)",
        "feature_count": len(feature_columns),
        "method": "TD-only cubic B-spline regression with df=4; z-scores use TD residual SD.",
        "linear_sensitivity": "Linear age model fit in parallel for sensitivity comparison.",
        "meta_columns": _META_COLUMNS,
        "feature_columns": feature_columns,
    }
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote age-adjusted severity table to {out_csv}")
    print(f"Wrote per-feature model summary to {out_model}")
    print(f"Wrote normalization summary to {out_summary}")
    return 0
