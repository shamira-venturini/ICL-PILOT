from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def build_severity_bands(
    input_csv: str,
    output_csv: str,
    output_summary_json: str,
) -> int:
    input_path = Path(input_csv).expanduser().resolve()
    out_csv = Path(output_csv).expanduser().resolve()
    out_summary = Path(output_summary_json).expanduser().resolve()

    df = pd.read_csv(input_path)
    if "Group" not in df.columns or "severity_core_composite" not in df.columns:
        raise ValueError("Input table must contain `Group` and `severity_core_composite`.")

    sli = pd.to_numeric(
        df.loc[df["Group"] == "SLI", "severity_core_composite"],
        errors="coerce",
    ).dropna()
    if len(sli) < 9:
        raise ValueError("Not enough SLI rows to derive tertile bands.")

    q1 = float(sli.quantile(1 / 3))
    q2 = float(sli.quantile(2 / 3))

    def band_all(value: float) -> str:
        if pd.isna(value):
            return ""
        if value <= q1:
            return "low"
        if value <= q2:
            return "moderate"
        return "high"

    out = df.copy()
    score = pd.to_numeric(out["severity_core_composite"], errors="coerce")
    out["severity_band_sli_tertile_all"] = score.map(band_all)
    out["severity_band_sli_tertile_sli_only"] = ""
    sli_mask = out["Group"].eq("SLI") & score.notna()
    out.loc[sli_mask, "severity_band_sli_tertile_sli_only"] = score.loc[sli_mask].map(band_all)

    out["severity_percentile_in_sli_reference"] = score.apply(
        lambda value: float((sli <= value).mean()) if pd.notna(value) else None
    )

    out.to_csv(out_csv, index=False)

    summary = {
        "input_csv": str(input_path),
        "output_csv": str(out_csv),
        "reference_group": "SLI",
        "reference_measure": "severity_core_composite",
        "band_method": "SLI tertiles",
        "sli_n": int(len(sli)),
        "cutpoints": {
            "low_to_moderate": q1,
            "moderate_to_high": q2,
        },
        "sli_band_counts": {
            label: int((out.loc[sli_mask, "severity_band_sli_tertile_sli_only"] == label).sum())
            for label in ["low", "moderate", "high"]
        },
    }
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote banded severity table to {out_csv}")
    print(f"Wrote severity band summary to {out_summary}")
    return 0
