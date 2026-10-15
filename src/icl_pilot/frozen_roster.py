from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .paths import FROZEN_ROSTER_CSV, ROOT


def _default_cohort_label(age_min_months: int, age_max_months: int) -> str:
    min_year = age_min_months // 12
    max_year = age_max_months // 12
    if min_year == max_year and age_min_months % 12 == 0 and age_max_months % 12 == 11:
        return f"{min_year}-year-old"
    return f"{age_min_months}to{age_max_months}_months"


def _filter_age_bin(frame: pd.DataFrame, age_min_months: int, age_max_months: int) -> pd.DataFrame:
    age_months = pd.to_numeric(frame["Age(Month)"], errors="coerce")
    return frame.loc[age_months.between(age_min_months, age_max_months)].copy()


def _load_roster_frame(
    dev_measures_csv: Path,
    age_min_months: int,
    age_max_months: int,
) -> pd.DataFrame:
    dev = pd.read_csv(dev_measures_csv)
    dev = dev[dev["Group"].isin(["SLI", "TD"])].copy()
    roster = _filter_age_bin(dev, age_min_months=age_min_months, age_max_months=age_max_months)
    if roster.duplicated(["File_ID", "Group"]).any():
        raise ValueError("Developmental metadata contains duplicate child/group records.")
    roster["age_months_approx"] = pd.to_numeric(roster["Age_Month_approx"], errors="coerce")
    roster = roster.loc[roster["age_months_approx"].notna()].copy()
    return roster


def _solve_pairing(roster: pd.DataFrame, cohort_label: str) -> pd.DataFrame:
    sli = roster[roster["Group"] == "SLI"].sort_values(["age_months_approx", "File_ID"]).reset_index(drop=True)
    td = roster[roster["Group"] == "TD"].sort_values(["age_months_approx", "File_ID"]).reset_index(drop=True)

    if sli.empty:
        raise ValueError("No SLI children found in the requested age bin")
    if td.empty:
        raise ValueError("No TD children found in the requested age bin")
    if len(td) < len(sli):
        raise ValueError("Not enough TD children to build a one-to-one frozen roster")

    cost = np.abs(sli["age_months_approx"].to_numpy()[:, None] - td["age_months_approx"].to_numpy()[None, :])
    # Deterministic tie-breaker: prefer the lower TD file id when age gaps are identical.
    cost = cost + 1e-9 * td["File_ID"].to_numpy()[None, :]

    row_ind, col_ind = linear_sum_assignment(cost)
    assignments = sorted(
        zip(row_ind, col_ind),
        key=lambda rc: (float(sli.iloc[rc[0]]["age_months_approx"]), int(sli.iloc[rc[0]]["File_ID"])),
    )

    rows: list[dict[str, object]] = []
    for pair_order, (row_idx, col_idx) in enumerate(assignments, start=1):
        sli_row = sli.iloc[row_idx]
        td_row = td.iloc[col_idx]
        age_gap = abs(float(sli_row["age_months_approx"]) - float(td_row["age_months_approx"]))
        rows.append(
            {
                "pair_order": pair_order,
                "pair_id": f"pair_sli{int(sli_row['File_ID'])}_td{int(td_row['File_ID'])}",
                "cohort": cohort_label,
                "sli_child_id": int(sli_row["File_ID"]),
                "sli_file_kideval": sli_row["File_kideval"],
                "sli_age": sli_row["Age"],
                "sli_age_months_approx": round(float(sli_row["age_months_approx"]), 3),
                # Legacy schema fields remain empty; no profile is estimated.
                "sli_severity_band": "",
                "sli_profile_label": "",
                "td_child_id": int(td_row["File_ID"]),
                "td_file_kideval": td_row["File_kideval"],
                "td_age": td_row["Age"],
                "td_age_months_approx": round(float(td_row["age_months_approx"]), 3),
                "age_gap_months": round(age_gap, 3),
                "match_rule": "global minimum absolute age gap with one-to-one TD capacity",
                "notes": "TD child used once; pairing uses age only",
            }
        )

    return pd.DataFrame(rows)


def build_frozen_roster_manifest(
    dev_measures_csv: str,
    output_csv: str,
    age_min_months: int = 48,
    age_max_months: int = 59,
    cohort_label: str | None = None,
) -> int:
    dev_path = Path(dev_measures_csv).expanduser().resolve()
    output_path = Path(output_csv).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError("Frozen roster already exists; choose a new output path for a comparison build.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    roster = _load_roster_frame(
        dev_measures_csv=dev_path,
        age_min_months=age_min_months,
        age_max_months=age_max_months,
    )
    resolved_cohort_label = cohort_label or _default_cohort_label(age_min_months, age_max_months)
    roster_df = _solve_pairing(roster, cohort_label=resolved_cohort_label)
    roster_df.to_csv(output_path, index=False)

    total_gap = float(roster_df["age_gap_months"].sum())
    max_gap = float(roster_df["age_gap_months"].max())
    print(f"Wrote frozen roster to {output_path}")
    print(f"Paired {len(roster_df)} SLI children to {len(roster_df)} unique TD children")
    print(f"Total age gap: {total_gap:.3f} months; max gap: {max_gap:.3f} months")
    return 0


def build_default_frozen_roster_manifest() -> int:
    return build_frozen_roster_manifest(
        dev_measures_csv=str(ROOT / "phase2" / "ENNI" / "measures" / "dev_measures.csv"),
        output_csv=str(FROZEN_ROSTER_CSV),
    )
