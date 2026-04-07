from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .paths import FROZEN_ROSTER_CSV, ROOT
from .story_generation_design import _profile_label


_SEVERITY_COLUMNS = [
    "severity_band_sli_tertile_sli_only",
    "severity_structural",
    "severity_lexical",
    "severity_disruption",
    "severity_morphosyntax_burden",
]


def _filter_age_bin(frame: pd.DataFrame, age_min_months: int, age_max_months: int) -> pd.DataFrame:
    age_months = pd.to_numeric(frame["Age(Month)"], errors="coerce")
    return frame.loc[age_months.between(age_min_months, age_max_months)].copy()


def _load_roster_frame(
    dev_measures_csv: Path,
    severity_profile_csv: Path,
    age_min_months: int,
    age_max_months: int,
) -> pd.DataFrame:
    dev = pd.read_csv(dev_measures_csv)
    sev = pd.read_csv(severity_profile_csv)

    dev = dev[dev["Group"].isin(["SLI", "TD"])].copy()
    sev = sev[sev["Group"].isin(["SLI", "TD"])].copy()
    dev = _filter_age_bin(dev, age_min_months=age_min_months, age_max_months=age_max_months)
    sev = _filter_age_bin(sev, age_min_months=age_min_months, age_max_months=age_max_months)

    merge_cols = ["File_ID", "Group", *_SEVERITY_COLUMNS]
    roster = dev.merge(
        sev[merge_cols],
        on=["File_ID", "Group"],
        how="inner",
        validate="one_to_one",
    )
    roster["age_months_approx"] = pd.to_numeric(roster["Age_Month_approx"], errors="coerce")
    roster["profile_label"] = roster.apply(
        lambda row: "typical" if row["Group"] == "TD" else _profile_label(row),
        axis=1,
    )
    return roster


def _solve_pairing(roster: pd.DataFrame) -> pd.DataFrame:
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
                "cohort": "4-year-old",
                "sli_child_id": int(sli_row["File_ID"]),
                "sli_file_kideval": sli_row["File_kideval"],
                "sli_age": sli_row["Age"],
                "sli_age_months_approx": round(float(sli_row["age_months_approx"]), 3),
                "sli_severity_band": sli_row["severity_band_sli_tertile_sli_only"],
                "sli_profile_label": sli_row["profile_label"],
                "td_child_id": int(td_row["File_ID"]),
                "td_file_kideval": td_row["File_kideval"],
                "td_age": td_row["Age"],
                "td_age_months_approx": round(float(td_row["age_months_approx"]), 3),
                "age_gap_months": round(age_gap, 3),
                "match_rule": "global minimum absolute age gap with one-to-one TD capacity",
                "notes": "TD child used once; severity and profile are metadata only",
            }
        )

    return pd.DataFrame(rows)


def build_frozen_roster_manifest(
    dev_measures_csv: str,
    severity_profile_csv: str,
    output_csv: str,
    age_min_months: int = 48,
    age_max_months: int = 59,
) -> int:
    dev_path = Path(dev_measures_csv).expanduser().resolve()
    severity_path = Path(severity_profile_csv).expanduser().resolve()
    output_path = Path(output_csv).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    roster = _load_roster_frame(
        dev_measures_csv=dev_path,
        severity_profile_csv=severity_path,
        age_min_months=age_min_months,
        age_max_months=age_max_months,
    )
    roster_df = _solve_pairing(roster)
    roster_df.to_csv(output_path, index=False)

    total_gap = float(roster_df["age_gap_months"].sum())
    max_gap = float(roster_df["age_gap_months"].max())
    print(f"Wrote frozen roster to {output_path}")
    print(f"Paired {len(roster_df)} SLI children to {len(roster_df)} unique TD children")
    print(f"Total age gap: {total_gap:.3f} months; max gap: {max_gap:.3f} months")
    return 0


def build_default_frozen_roster_manifest() -> int:
    return build_frozen_roster_manifest(
        dev_measures_csv=str(ROOT / "phase2" / "measures" / "dev_measures.csv"),
        severity_profile_csv=str(ROOT / "phase2" / "measures" / "severity_profile_banded_table.csv"),
        output_csv=str(FROZEN_ROSTER_CSV),
    )
