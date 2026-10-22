"""Retained bundle-level model helpers, separate from generation setup.

These Python diagnostics are separate from the integrated R Markdown analysis.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ..generation.workflow import bundle_feature_columns


def _prepare_mixedlm_frame(
    df: pd.DataFrame,
    outcome: str,
    source_col: str,
    cluster_col: str,
    age_col: str | None,
    group_col: str | None,
    synthetic_label: str,
) -> tuple[pd.DataFrame, str]:
    required = [outcome, source_col, cluster_col]
    if age_col is not None:
        required.append(age_col)
    if group_col is not None:
        required.append(group_col)

    working = df[required].copy()
    working = working.dropna(subset=[outcome, source_col, cluster_col])
    working[outcome] = pd.to_numeric(working[outcome], errors="coerce")
    working = working.dropna(subset=[outcome])
    working["source_is_synthetic"] = (working[source_col].astype(str) == synthetic_label).astype(int)
    formula_terms = ["source_is_synthetic"]
    if age_col is not None:
        formula_terms.append(age_col)
        working[age_col] = pd.to_numeric(working[age_col], errors="coerce")
        working = working.dropna(subset=[age_col])
    if group_col is not None:
        formula_terms.append(f"C({group_col})")
    formula = f"{outcome} ~ " + " + ".join(formula_terms)
    return working, formula


def fit_primary_mixed_models(
    df: pd.DataFrame,
    outcomes: Sequence[str],
    source_col: str = "source",
    cluster_col: str = "cluster_id",
    age_col: str | None = None,
    group_col: str | None = None,
    synthetic_label: str = "synthetic",
) -> pd.DataFrame:
    try:
        import statsmodels.formula.api as smf
    except Exception as exc:  # pragma: no cover - notebook convenience
        raise ImportError(
            "statsmodels is required for mixed-effects fitting. Install it in the notebook environment."
        ) from exc

    rows: list[dict[str, object]] = []
    for outcome in outcomes:
        try:
            working, formula = _prepare_mixedlm_frame(
                df=df,
                outcome=outcome,
                source_col=source_col,
                cluster_col=cluster_col,
                age_col=age_col,
                group_col=group_col,
                synthetic_label=synthetic_label,
            )
            if working.empty:
                raise ValueError(f"No usable rows available for outcome {outcome!r}")

            model = smf.mixedlm(formula, working, groups=working[cluster_col])
            result = model.fit(reml=False, method="lbfgs", disp=False)
            term = "source_is_synthetic"
            params = result.params
            bse = result.bse
            pvalues = result.pvalues
            conf = result.conf_int()
            ci_low, ci_high = (np.nan, np.nan)
            if term in conf.index:
                ci_low = float(conf.loc[term, 0])
                ci_high = float(conf.loc[term, 1])

            rows.append(
                {
                    "outcome": outcome,
                    "formula": formula,
                    "n_obs": int(result.nobs),
                    "n_clusters": int(working[cluster_col].nunique()),
                    "estimate": float(params.get(term, np.nan)),
                    "std_err": float(bse.get(term, np.nan)),
                    "p_value": float(pvalues.get(term, np.nan)),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "converged": bool(getattr(result, "converged", True)),
                    "llf": float(getattr(result, "llf", np.nan)),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "outcome": outcome,
                    "formula": "",
                    "n_obs": 0,
                    "n_clusters": 0,
                    "estimate": np.nan,
                    "std_err": np.nan,
                    "p_value": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "converged": False,
                    "llf": np.nan,
                    "error": str(exc),
                }
            )

    return pd.DataFrame(rows)


def fit_anomaly_qc(
    reference_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_cols: Sequence[str] | None = None,
    contamination: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover - notebook convenience
        raise ImportError(
            "scikit-learn is required for anomaly QC. Install it in the notebook environment."
        ) from exc

    cols = list(feature_cols or bundle_feature_columns())
    ref = reference_df.copy()
    tgt = target_df.copy()
    ref_numeric = ref[cols].apply(pd.to_numeric, errors="coerce")
    tgt_numeric = tgt[cols].apply(pd.to_numeric, errors="coerce")

    pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        IsolationForest(contamination=contamination, random_state=random_state),
    )
    pipeline.fit(ref_numeric)

    target_scores = -pipeline.named_steps["isolationforest"].decision_function(
        pipeline.named_steps["standardscaler"].transform(
            pipeline.named_steps["simpleimputer"].transform(tgt_numeric)
        )
    )
    target_flags = pipeline.named_steps["isolationforest"].predict(
        pipeline.named_steps["standardscaler"].transform(
            pipeline.named_steps["simpleimputer"].transform(tgt_numeric)
        )
    )

    out = tgt.copy()
    out["qc_anomaly_score"] = target_scores
    out["qc_anomaly_flag"] = target_flags == -1
    out["qc_anomaly_flag"] = out["qc_anomaly_flag"].astype(bool)
    return out
