from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .generation.workflow import build_rotating_eval_manifest
from .severity_features import _KEEP_ORDER


_ID_COLUMNS = {"File_ID", "File_kideval", "Group", "Sex"}
_EXPECTED_GROUPS = {"TD", "SLI"}


def _coerce_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _normalize_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        numeric = float(text)
    except ValueError:
        return text
    if numeric.is_integer():
        return str(int(numeric))
    return text


def _shared_feature_columns(real_df: pd.DataFrame, generated_df: pd.DataFrame) -> list[str]:
    shared = [
        column
        for column in _KEEP_ORDER
        if column not in _ID_COLUMNS and column in real_df.columns and column in generated_df.columns
    ]
    usable: list[str] = []
    for column in shared:
        real_values = pd.to_numeric(real_df[column], errors="coerce")
        generated_values = pd.to_numeric(generated_df[column], errors="coerce")
        if real_values.notna().any() and generated_values.notna().any():
            usable.append(column)
    if not usable:
        raise ValueError("No shared numeric severity features are available for evaluation.")
    return usable


def _build_reference_table(
    real_pool: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, tuple[float, float]]]]:
    reference_rows: list[dict[str, object]] = []
    reference: dict[str, dict[str, tuple[float, float]]] = {}

    for group, group_df in real_pool.groupby("comparison_target_group", sort=True):
        reference[group] = {}
        unique_group = group_df.drop_duplicates(subset=["File_ID", "Group"]).copy()
        for feature in feature_columns:
            values = pd.to_numeric(unique_group[feature], errors="coerce").dropna()
            mean = float(values.mean()) if not values.empty else 0.0
            std = float(values.std(ddof=0)) if len(values) > 1 else 0.0
            safe_std = std if np.isfinite(std) and std > 0 else 1.0
            reference[group][feature] = (mean, safe_std)
            reference_rows.append(
                {
                    "comparison_target_group": group,
                    "feature": feature,
                    "n_real_children_with_value": int(values.shape[0]),
                    "mean": mean,
                    "std": std,
                    "safe_std_used_for_scoring": safe_std,
                }
            )

    return pd.DataFrame(reference_rows), reference


def build_generated_bundle_evaluation(
    *,
    generated_severity_csv: str | Path,
    real_severity_csv: str | Path,
    frozen_roster_csv: str | Path,
    output_dir: str | Path,
    pairs_per_round: int = 3,
) -> int:
    generated_path = _coerce_path(generated_severity_csv)
    real_path = _coerce_path(real_severity_csv)
    roster_path = _coerce_path(frozen_roster_csv)
    out_dir = _coerce_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_manifest_csv = out_dir / "rotating_eval_manifest.csv"
    heldout_real_pool_csv = out_dir / "heldout_real_pool.csv"
    feature_reference_csv = out_dir / "feature_zscore_reference.csv"
    pairwise_csv = out_dir / "pairwise_feature_distances.csv"
    nearest_csv = out_dir / "nearest_neighbor_summary.csv"
    summary_json = out_dir / "evaluation_summary.json"

    generated = pd.read_csv(generated_path)
    real = pd.read_csv(real_path)

    generated = generated[generated["comparison_target_group"].isin(_EXPECTED_GROUPS)].copy()
    generated = generated[generated["round_id"].notna() & generated["round_id"].astype(str).ne("")].copy()
    real = real[real["Group"].isin(_EXPECTED_GROUPS)].copy()

    generated["File_ID_norm"] = generated["File_ID"].map(_normalize_id)
    generated["prompt_td_child_id_norm"] = generated["prompt_td_child_id"].map(_normalize_id)
    generated["prompt_sli_child_id_norm"] = generated["prompt_sli_child_id"].map(_normalize_id)
    real["File_ID_norm"] = real["File_ID"].map(_normalize_id)

    eval_manifest = build_rotating_eval_manifest(
        frozen_roster_csv=roster_path,
        pairs_per_round=pairs_per_round,
        output_csv=eval_manifest_csv,
    )
    eval_manifest["eval_child_id_norm"] = eval_manifest["eval_child_id"].map(_normalize_id)

    real_pool = eval_manifest.merge(
        real,
        left_on=["eval_child_id_norm", "eval_group"],
        right_on=["File_ID_norm", "Group"],
        how="left",
        validate="many_to_one",
    )

    missing_real = real_pool["File_ID"].isna()
    if missing_real.any():
        missing_rows = real_pool.loc[missing_real, ["round_id", "eval_group", "eval_child_id"]]
        raise ValueError(
            "Could not match some held-out real evaluation children to the real severity table: "
            f"{missing_rows.to_dict('records')[:10]}"
        )

    feature_columns = _shared_feature_columns(real_df=real_pool, generated_df=generated)
    feature_reference_df, feature_reference = _build_reference_table(real_pool=real_pool, feature_columns=feature_columns)

    real_pool.to_csv(heldout_real_pool_csv, index=False)
    feature_reference_df.to_csv(feature_reference_csv, index=False)

    pairwise_rows: list[dict[str, object]] = []
    for synthetic_row in generated.to_dict("records"):
        round_id = str(synthetic_row["round_id"])
        group = str(synthetic_row["comparison_target_group"])
        prompt_child_id = (
            synthetic_row["prompt_td_child_id_norm"] if group == "TD" else synthetic_row["prompt_sli_child_id_norm"]
        )
        candidate_pool = real_pool[
            (real_pool["round_id"].astype(str) == round_id)
            & (real_pool["comparison_target_group"].astype(str) == group)
        ].copy()
        if prompt_child_id:
            candidate_pool = candidate_pool[
                candidate_pool["eval_child_id"].map(_normalize_id) != prompt_child_id
            ].copy()
        if candidate_pool.empty:
            raise ValueError(f"No held-out real pool found for synthetic row {synthetic_row['File_ID']} in {round_id}/{group}")
        synthetic_z: dict[str, float] = {}
        for feature in feature_columns:
            value = pd.to_numeric(pd.Series([synthetic_row.get(feature)]), errors="coerce").iloc[0]
            if pd.notna(value):
                mean, std = feature_reference[group][feature]
                synthetic_z[feature] = (float(value) - mean) / std

        for candidate in candidate_pool.to_dict("records"):
            diffs: list[float] = []
            for feature in feature_columns:
                if feature not in synthetic_z:
                    continue
                real_value = pd.to_numeric(pd.Series([candidate.get(feature)]), errors="coerce").iloc[0]
                if pd.isna(real_value):
                    continue
                mean, std = feature_reference[group][feature]
                real_z = (float(real_value) - mean) / std
                diffs.append(synthetic_z[feature] - real_z)

            if not diffs:
                raise ValueError(
                    f"No overlapping non-missing features for synthetic row {synthetic_row['File_ID']} "
                    f"and real child {candidate['File_ID']}."
                )

            diffs_array = np.asarray(diffs, dtype=float)
            pairwise_rows.append(
                {
                    "synthetic_file_id": synthetic_row["File_ID"],
                    "synthetic_bundle_id": synthetic_row.get("bundle_id", ""),
                    "pair_id": synthetic_row.get("pair_id", ""),
                    "round_id": round_id,
                    "replicate_id": synthetic_row.get("replicate_id", ""),
                    "stage": synthetic_row.get("stage", ""),
                    "comparison_target_group": group,
                    "source_child_id": synthetic_row.get("source_child_id", ""),
                    "prompt_td_child_id": synthetic_row.get("prompt_td_child_id", ""),
                    "prompt_sli_child_id": synthetic_row.get("prompt_sli_child_id", ""),
                    "real_file_id": candidate["File_ID"],
                    "real_bundle_id": candidate["bundle_id"],
                    "real_child_id": candidate["eval_child_id"],
                    "real_pair_id": candidate["eval_pair_id"],
                    "real_age_months_approx": candidate["eval_age_months_approx"],
                    "real_severity_band": candidate["eval_severity_band"],
                    "real_profile_label": candidate["eval_profile_label"],
                    "candidate_count_in_round_group": int(len(candidate_pool)),
                    "n_features_used": int(diffs_array.size),
                    "euclidean_z_distance": float(np.sqrt(np.square(diffs_array).sum())),
                    "rms_z_distance": float(np.sqrt(np.square(diffs_array).mean())),
                    "mean_abs_z_distance": float(np.abs(diffs_array).mean()),
                    "max_abs_z_distance": float(np.abs(diffs_array).max()),
                    "prompt_child_overlap_flag": int(_normalize_id(candidate["eval_child_id"]) == prompt_child_id),
                }
            )

    pairwise = pd.DataFrame(pairwise_rows).sort_values(
        ["synthetic_file_id", "rms_z_distance", "euclidean_z_distance", "real_file_id"]
    ).reset_index(drop=True)
    pairwise["rank_in_round_group"] = pairwise.groupby("synthetic_file_id").cumcount() + 1
    pairwise.to_csv(pairwise_csv, index=False)

    nearest = pairwise.groupby("synthetic_file_id", as_index=False).first()
    second_best = (
        pairwise[pairwise["rank_in_round_group"] == 2][["synthetic_file_id", "rms_z_distance"]]
        .rename(columns={"rms_z_distance": "second_best_rms_z_distance"})
    )
    nearest = nearest.merge(second_best, on="synthetic_file_id", how="left")
    nearest["margin_to_second_best"] = nearest["second_best_rms_z_distance"] - nearest["rms_z_distance"]
    nearest.to_csv(nearest_csv, index=False)

    prompt_overlap_rows = int(pairwise["prompt_child_overlap_flag"].sum())
    candidate_counts = pairwise[["synthetic_file_id", "candidate_count_in_round_group"]].drop_duplicates()

    summary = {
        "generated_severity_csv": str(generated_path),
        "real_severity_csv": str(real_path),
        "frozen_roster_csv": str(roster_path),
        "output_dir": str(out_dir),
        "pairs_per_round": int(pairs_per_round),
        "n_generated_rows": int(len(generated)),
        "n_real_rows": int(len(real)),
        "n_rounds": int(generated["round_id"].nunique()),
        "rounds": sorted(generated["round_id"].astype(str).unique().tolist()),
        "groups": sorted(generated["comparison_target_group"].astype(str).unique().tolist()),
        "feature_columns_used_for_distance": feature_columns,
        "n_feature_columns_used_for_distance": int(len(feature_columns)),
        "n_pairwise_rows": int(len(pairwise)),
        "n_nearest_neighbor_rows": int(len(nearest)),
        "candidate_count_per_synthetic_row": {
            "min": int(candidate_counts["candidate_count_in_round_group"].min()),
            "max": int(candidate_counts["candidate_count_in_round_group"].max()),
            "mean": float(candidate_counts["candidate_count_in_round_group"].mean()),
        },
        "candidate_pool_policy_applied": "same round, same target group, exclude prompt pairs and prompt child ids",
        "prompt_overlap_rows": prompt_overlap_rows,
        "prompt_overlap_check_passed": prompt_overlap_rows == 0,
        "median_best_rms_z_distance": float(nearest["rms_z_distance"].median()),
        "median_best_mean_abs_z_distance": float(nearest["mean_abs_z_distance"].median()),
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote rotating evaluation manifest to {eval_manifest_csv}")
    print(f"Wrote held-out real pool to {heldout_real_pool_csv}")
    print(f"Wrote feature z-score reference to {feature_reference_csv}")
    print(f"Wrote pairwise feature distances to {pairwise_csv}")
    print(f"Wrote nearest-neighbor summary to {nearest_csv}")
    print(f"Wrote evaluation summary to {summary_json}")
    return 0
