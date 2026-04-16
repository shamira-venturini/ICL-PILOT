from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .error_rate_features import add_error_rate_features_to_master


def _coerce_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _file_id_from_name(value: str) -> str:
    name = Path(str(value)).name
    if name.endswith(".cha"):
        return name[:-4]
    return Path(name).stem


def _bundle_metadata(story_generation_csv: Path) -> pd.DataFrame:
    story_df = pd.read_csv(story_generation_csv)
    required = {"bundle_id", "pair_id", "round_id", "replicate_id", "target_story", "sli_child_id", "td_child_id"}
    missing = required.difference(story_df.columns)
    if missing:
        raise ValueError(f"Story-generation CSV is missing required columns: {sorted(missing)}")

    grouped = (
        story_df.sort_values(["bundle_id", "story_slot_order"])
        .groupby("bundle_id", as_index=False)
        .agg(
            pair_id=("pair_id", "first"),
            round_id=("round_id", "first"),
            replicate_id=("replicate_id", "first"),
            sli_child_id=("sli_child_id", "first"),
            td_child_id=("td_child_id", "first"),
            sli_age=("sli_age", "first"),
            sli_severity_band=("sli_severity_band", "first"),
            sli_profile_label=("sli_profile_label", "first"),
            story_slots=("target_story", lambda values: "|".join(str(value) for value in values)),
        )
    )

    rows: list[dict[str, object]] = []
    for row in grouped.to_dict("records"):
        bundle_id = str(row["bundle_id"])
        pair_id = str(row["pair_id"])
        round_id = str(row["round_id"])
        replicate_id = str(row["replicate_id"])
        sli_child_id = str(row["sli_child_id"])
        td_child_id = str(row["td_child_id"])
        story_slots = str(row["story_slots"])

        rows.append(
            {
                "File_ID": f"{bundle_id}_TD_bundle",
                "bundle_id": bundle_id,
                "pair_id": pair_id,
                "round_id": round_id,
                "replicate_id": replicate_id,
                "source": "synthetic",
                "stage": "stage1_td",
                "comparison_target_group": "TD",
                "evaluation_pool_policy": "all non-prompt pairs in this round",
                "prompt_exclusion_policy": "exclude prompt children from evaluation in the same round",
                "source_child_id": td_child_id,
                "prompt_td_child_id": td_child_id,
                "prompt_sli_child_id": sli_child_id,
                "td_child_id": td_child_id,
                "sli_child_id": sli_child_id,
                "sli_age_generated_from": row["sli_age"],
                "sli_severity_band_generated_from": row["sli_severity_band"],
                "sli_profile_label_generated_from": row["sli_profile_label"],
                "story_slots": story_slots,
                "n_stories": 6,
            }
        )
        rows.append(
            {
                "File_ID": f"{bundle_id}_SLI_bundle",
                "bundle_id": bundle_id,
                "pair_id": pair_id,
                "round_id": round_id,
                "replicate_id": replicate_id,
                "source": "synthetic",
                "stage": "stage2_sli",
                "comparison_target_group": "SLI",
                "evaluation_pool_policy": "all non-prompt pairs in this round",
                "prompt_exclusion_policy": "exclude prompt children from evaluation in the same round",
                "source_child_id": sli_child_id,
                "prompt_td_child_id": td_child_id,
                "prompt_sli_child_id": sli_child_id,
                "td_child_id": td_child_id,
                "sli_child_id": sli_child_id,
                "sli_age_generated_from": row["sli_age"],
                "sli_severity_band_generated_from": row["sli_severity_band"],
                "sli_profile_label_generated_from": row["sli_profile_label"],
                "story_slots": story_slots,
                "n_stories": 6,
            }
        )

    return pd.DataFrame(rows)


def build_generated_dev_measures(
    *,
    story_generation_csv: str | Path,
    kideval_clean_csv: str | Path,
    mlt_clean_csv: str | Path,
    dss_csv: str | Path,
    grouped_error_csv: str | Path,
    past_verb_csv: str | Path,
    output_csv: str | Path,
    output_summary_json: str | Path,
) -> int:
    story_generation_path = _coerce_path(story_generation_csv)
    kideval_path = _coerce_path(kideval_clean_csv)
    mlt_path = _coerce_path(mlt_clean_csv)
    dss_path = _coerce_path(dss_csv)
    grouped_path = _coerce_path(grouped_error_csv)
    past_path = _coerce_path(past_verb_csv)
    output_path = _coerce_path(output_csv)
    summary_path = _coerce_path(output_summary_json)

    kideval = pd.read_csv(kideval_path, dtype=str).fillna("")
    mlt = pd.read_csv(mlt_path, dtype=str).fillna("")
    dss = pd.read_csv(dss_path, dtype=str).fillna("")
    grouped = pd.read_csv(grouped_path, dtype=str).fillna("")
    past = pd.read_csv(past_path, dtype=str).fillna("")
    metadata = _bundle_metadata(story_generation_path).fillna("")

    kideval = kideval.rename(columns={"File": "File_kideval"})
    kideval["File_ID"] = kideval["File_kideval"].map(_file_id_from_name)

    mlt = mlt.rename(columns={"File": "File_mlt"})
    if "File_ID" not in mlt.columns:
        mlt["File_ID"] = mlt["File_mlt"].map(_file_id_from_name)

    # Preserve the same approximate layout as the real-kid master while
    # carrying synthetic-bundle metadata needed for round-based exclusion.
    dev = kideval.merge(mlt, on="File_ID", how="left", validate="one_to_one", suffixes=("", "_mlt"))
    dev = dev.merge(dss, on="File_ID", how="left", validate="one_to_one")
    dev = dev.merge(
        grouped[
            [
                "File_ID",
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
        ],
        on="File_ID",
        how="left",
        validate="one_to_one",
    )
    dev = dev.merge(
        past[
            [
                "File_ID",
                "overt_past_finite_verbs",
                "bare_past_error_tokens",
                "past_verb_opportunities",
                "past_overgeneralization_errors",
                "past_overgeneralization_errors_per_past_opp",
            ]
        ],
        on="File_ID",
        how="left",
        validate="one_to_one",
    )
    dev = dev.merge(metadata, on="File_ID", how="left", validate="one_to_one")

    # Keep a trace of the MLT-side file id just like the real-kid master.
    if "File_mlt" not in dev.columns and "File_mlt_mlt" in dev.columns:
        dev = dev.rename(columns={"File_mlt_mlt": "File_mlt"})

    if "Group_2" not in dev.columns and "Group_2_kideval" in dev.columns:
        dev = dev.rename(columns={"Group_2_kideval": "Group_2"})

    for column in [
        "File_mlt",
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
        "overt_past_finite_verbs",
        "bare_past_error_tokens",
        "past_verb_opportunities",
        "past_overgeneralization_errors",
        "past_overgeneralization_errors_per_past_opp",
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
    ]:
        if column not in dev.columns:
            dev[column] = ""

    dev.to_csv(output_path, index=False)
    add_error_rate_features_to_master(str(output_path), str(output_path))

    final_df = pd.read_csv(output_path)
    summary = {
        "story_generation_csv": str(story_generation_path),
        "kideval_clean_csv": str(kideval_path),
        "mlt_clean_csv": str(mlt_path),
        "dss_csv": str(dss_path),
        "grouped_error_csv": str(grouped_path),
        "past_verb_csv": str(past_path),
        "output_csv": str(output_path),
        "rows": int(len(final_df)),
        "groups": sorted(final_df["Group"].dropna().astype(str).unique().tolist()) if "Group" in final_df.columns else [],
        "rounds": sorted(final_df["round_id"].dropna().astype(str).unique().tolist()) if "round_id" in final_df.columns else [],
        "n_bundle_ids": int(final_df["bundle_id"].nunique()) if "bundle_id" in final_df.columns else 0,
        "n_prompt_td_child_ids": int(final_df["prompt_td_child_id"].replace("", pd.NA).dropna().nunique()) if "prompt_td_child_id" in final_df.columns else 0,
        "n_prompt_sli_child_ids": int(final_df["prompt_sli_child_id"].replace("", pd.NA).dropna().nunique()) if "prompt_sli_child_id" in final_df.columns else 0,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0
