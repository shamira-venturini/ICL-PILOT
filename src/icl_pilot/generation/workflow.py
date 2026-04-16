from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ..frozen_roster import build_frozen_roster_manifest
from ..story_generation_design import _profile_label


_STORY_ORDER = ["A1", "A2", "A3", "B1", "B2", "B3"]
_FEATURE_COLUMNS = [
    "n_utterances",
    "n_words",
    "mlu_morphemes",
    "mlu_words",
    "dss",
    "ipsyn_total",
    "vocd_d",
    "word_errors_per_utt",
    "bare_marking_errors_per_word",
    "pronoun_errors_per_word",
    "rule_generalization_errors_per_word",
    "past_overgeneralization_errors_per_past_opp",
    "severity_core_composite",
    "severity_domain_composite",
    "qc_anomaly_score",
    "qc_anomaly_flag",
]


@dataclass(frozen=True)
class GenerationPackageArtifacts:
    output_dir: Path
    roster_csv: Path
    schedule_csv: Path
    bundle_index_csv: Path
    synthetic_template_csv: Path
    real_template_csv: Path
    summary_json: Path


@dataclass(frozen=True)
class RotatingGenerationPackageArtifacts:
    output_dir: Path
    roster_csv: Path
    prompt_rounds_csv: Path
    schedule_csv: Path
    bundle_index_csv: Path
    synthetic_template_csv: Path
    eval_manifest_csv: Path
    summary_json: Path


@dataclass(frozen=True)
class MultiAgeRotatingGenerationArtifacts:
    output_dir: Path
    cohort_packages: dict[str, RotatingGenerationPackageArtifacts]
    summary_json: Path


def _coerce_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _resolve_cohort_label(age_min_months: int, age_max_months: int, cohort_label: str | None = None) -> str:
    if cohort_label:
        return cohort_label
    min_year = age_min_months // 12
    max_year = age_max_months // 12
    if min_year == max_year and age_min_months % 12 == 0 and age_max_months % 12 == 11:
        return f"{min_year}-year-old"
    return f"{age_min_months}to{age_max_months}_months"


def _cohort_slug(label: str) -> str:
    text = str(label).strip().lower()
    year_match = re.fullmatch(r"(\d+)-year-old", text)
    if year_match:
        number = int(year_match.group(1))
        number_words = {
            0: "zero",
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine",
            10: "ten",
            11: "eleven",
            12: "twelve",
        }
        if number in number_words:
            return f"{number_words[number]}_year_old"
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return slug.strip("_") or "cohort"


def load_frozen_roster(path: str | Path) -> pd.DataFrame:
    roster_path = _coerce_path(path)
    roster = pd.read_csv(roster_path)
    required = {
        "pair_order",
        "pair_id",
        "cohort",
        "sli_child_id",
        "sli_file_kideval",
        "sli_age",
        "sli_age_months_approx",
        "sli_severity_band",
        "sli_profile_label",
        "td_child_id",
        "td_file_kideval",
        "td_age",
        "td_age_months_approx",
        "age_gap_months",
        "match_rule",
        "notes",
    }
    missing = required.difference(roster.columns)
    if missing:
        raise ValueError(f"Frozen roster is missing columns: {sorted(missing)}")
    return roster.sort_values(["pair_order", "sli_child_id", "td_child_id"]).reset_index(drop=True)


def load_counterbalance_table(path: str | Path) -> pd.DataFrame:
    counterbalance_path = _coerce_path(path)
    counterbalance = pd.read_csv(counterbalance_path)
    required = {"target_set", "target_story", "e1_story", "e2_story"}
    missing = required.difference(counterbalance.columns)
    if missing:
        raise ValueError(f"Counterbalance table is missing columns: {sorted(missing)}")
    return counterbalance


def bundle_feature_columns() -> list[str]:
    return list(_FEATURE_COLUMNS)


def _counterbalance_lookup(counterbalance: pd.DataFrame) -> dict[tuple[str, str], list[tuple[str, str]]]:
    lookup: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for _, row in counterbalance.iterrows():
        key = (str(row["target_set"]), str(row["target_story"]))
        lookup.setdefault(key, []).append((str(row["e1_story"]), str(row["e2_story"])))
    if not lookup:
        raise ValueError("Counterbalance table is empty")
    return lookup


def _target_set_for_story(story_id: str) -> str:
    if story_id not in _STORY_ORDER:
        raise ValueError(f"Unsupported story id: {story_id}")
    return story_id[0]


def _build_story_generation_schedule_from_roster(
    roster: pd.DataFrame,
    counterbalance: pd.DataFrame,
    n_replicates: int,
) -> pd.DataFrame:
    lookup = _counterbalance_lookup(counterbalance)

    rows: list[dict[str, object]] = []
    for _, pair in roster.sort_values("pair_order").iterrows():
        pair_order = int(pair["pair_order"])
        for replicate_id in range(1, n_replicates + 1):
            bundle_id = f"{pair['pair_id']}_rep{replicate_id:02d}"
            for story_slot_order, target_story in enumerate(_STORY_ORDER, start=1):
                target_set = _target_set_for_story(target_story)
                variants = lookup[(target_set, target_story)]
                variant_total = len(variants)
                if variant_total == 0:
                    raise ValueError(f"No counterbalance variants found for {target_set} / {target_story}")

                variant_index = (replicate_id - 1 + pair_order - 1) % variant_total
                prompt_td_story_e1, prompt_sli_story_e2 = variants[variant_index]
                rows.append(
                    {
                        "pair_order": pair_order,
                        "pair_id": pair["pair_id"],
                        "cohort": pair["cohort"],
                        "sli_child_id": int(pair["sli_child_id"]),
                        "sli_file_kideval": pair["sli_file_kideval"],
                        "sli_age": pair["sli_age"],
                        "sli_age_months_approx": float(pair["sli_age_months_approx"]),
                        "sli_severity_band": pair["sli_severity_band"],
                        "sli_profile_label": pair["sli_profile_label"],
                        "td_child_id": int(pair["td_child_id"]),
                        "td_file_kideval": pair["td_file_kideval"],
                        "td_age": pair["td_age"],
                        "td_age_months_approx": float(pair["td_age_months_approx"]),
                        "age_gap_months": float(pair["age_gap_months"]),
                        "bundle_id": bundle_id,
                        "replicate_id": replicate_id,
                        "story_slot_order": story_slot_order,
                        "target_story_set": target_set,
                        "target_story": target_story,
                        "prompt_td_story_target": target_story,
                        "prompt_td_story_e1": prompt_td_story_e1,
                        "prompt_sli_story_e2": prompt_sli_story_e2,
                        "counterbalance_variant_index": variant_index + 1,
                        "counterbalance_variant_total": variant_total,
                        "generation_seed": replicate_id,
                        "holdout_policy": "exclude prompt pairs from evaluation in the same round",
                        "notes": "story-level prompt schedule for a fixed frozen child pair",
                    }
                )

    return pd.DataFrame(rows).sort_values(
        ["pair_order", "replicate_id", "story_slot_order"]
    ).reset_index(drop=True)


def build_story_generation_schedule(
    frozen_roster_csv: str | Path,
    counterbalance_csv: str | Path,
    n_replicates: int = 10,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    roster = load_frozen_roster(frozen_roster_csv)
    counterbalance = load_counterbalance_table(counterbalance_csv)
    schedule = _build_story_generation_schedule_from_roster(
        roster=roster,
        counterbalance=counterbalance,
        n_replicates=n_replicates,
    )

    if output_csv is not None:
        output_path = _coerce_path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        schedule.to_csv(output_path, index=False)

    return schedule


def build_bundle_index(schedule_df: pd.DataFrame, output_csv: str | Path | None = None) -> pd.DataFrame:
    required = {"bundle_id", "pair_id", "pair_order", "replicate_id", "story_slot_order", "target_story"}
    missing = required.difference(schedule_df.columns)
    if missing:
        raise ValueError(f"Schedule is missing required columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    grouped = schedule_df.groupby(["bundle_id", "pair_id", "pair_order", "replicate_id"], sort=False)
    for (bundle_id, pair_id, pair_order, replicate_id), group in grouped:
        ordered = group.sort_values("story_slot_order")
        story_slots = ordered["target_story"].tolist()
        if story_slots != _STORY_ORDER:
            raise ValueError(
                f"Bundle {bundle_id} does not contain the canonical six story slots: {story_slots}"
            )
        first = ordered.iloc[0]
        rows.append(
            {
                "bundle_id": bundle_id,
                "pair_id": pair_id,
                "pair_order": pair_order,
                "replicate_id": replicate_id,
                "round_id": first["round_id"] if "round_id" in ordered.columns else "",
                "cohort": first["cohort"],
                "group": "SLI",
                "source": "synthetic",
                "cluster_id": pair_id,
                "age": first["sli_age"],
                "age_months_approx": float(first["sli_age_months_approx"]),
                "sli_child_id": int(first["sli_child_id"]),
                "sli_file_kideval": first["sli_file_kideval"],
                "sli_age": first["sli_age"],
                "sli_age_months_approx": float(first["sli_age_months_approx"]),
                "sli_severity_band": first["sli_severity_band"],
                "sli_profile_label": first["sli_profile_label"],
                "td_child_id": int(first["td_child_id"]),
                "td_file_kideval": first["td_file_kideval"],
                "td_age": first["td_age"],
                "td_age_months_approx": float(first["td_age_months_approx"]),
                "age_gap_months": float(first["age_gap_months"]),
                "n_stories": len(ordered),
                "story_slots": "|".join(story_slots),
                "target_story_order": "|".join(story_slots),
                "prompt_td_child_id": int(first["td_child_id"]),
                "prompt_sli_child_id": int(first["sli_child_id"]),
            }
        )

    bundle_index = pd.DataFrame(rows).sort_values(["pair_order", "replicate_id"]).reset_index(drop=True)

    if output_csv is not None:
        output_path = _coerce_path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        bundle_index.to_csv(output_path, index=False)

    return bundle_index


def build_rotating_prompt_rounds(
    frozen_roster_csv: str | Path,
    pairs_per_round: int = 3,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    roster = load_frozen_roster(frozen_roster_csv)
    if pairs_per_round <= 0:
        raise ValueError("pairs_per_round must be positive")

    rows: list[dict[str, object]] = []
    for round_index, start in enumerate(range(0, len(roster), pairs_per_round), start=1):
        round_id = f"round_{round_index:02d}"
        prompt_pairs = roster.iloc[start : start + pairs_per_round].copy()
        for _, pair in prompt_pairs.iterrows():
            rows.append(
                {
                    "round_id": round_id,
                    "round_index": round_index,
                    "prompt_pair_order": int(pair["pair_order"]),
                    "pair_id": pair["pair_id"],
                    "cohort": pair["cohort"],
                    "sli_child_id": int(pair["sli_child_id"]),
                    "sli_file_kideval": pair["sli_file_kideval"],
                    "sli_age": pair["sli_age"],
                    "sli_age_months_approx": float(pair["sli_age_months_approx"]),
                    "sli_severity_band": pair["sli_severity_band"],
                    "sli_profile_label": pair["sli_profile_label"],
                    "td_child_id": int(pair["td_child_id"]),
                    "td_file_kideval": pair["td_file_kideval"],
                    "td_age": pair["td_age"],
                    "td_age_months_approx": float(pair["td_age_months_approx"]),
                    "age_gap_months": float(pair["age_gap_months"]),
                    "pairs_per_round": pairs_per_round,
                    "prompt_pair_count_in_round": int(len(prompt_pairs)),
                    "eval_pair_count_in_round": int(len(roster) - len(prompt_pairs)),
                    "round_prompt_policy": "pairs in this round are excluded from evaluation only in this round",
                }
            )

    rounds_df = pd.DataFrame(rows).sort_values(["round_index", "prompt_pair_order"]).reset_index(drop=True)
    if output_csv is not None:
        output_path = _coerce_path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rounds_df.to_csv(output_path, index=False)
    return rounds_df


def build_rotating_generation_schedule(
    frozen_roster_csv: str | Path,
    counterbalance_csv: str | Path,
    n_replicates: int = 10,
    pairs_per_round: int = 3,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    roster = load_frozen_roster(frozen_roster_csv)
    counterbalance = load_counterbalance_table(counterbalance_csv)
    round_rows = build_rotating_prompt_rounds(
        frozen_roster_csv=frozen_roster_csv,
        pairs_per_round=pairs_per_round,
    )

    roster_by_pair = {str(row["pair_id"]): row for _, row in roster.iterrows()}
    schedule_parts: list[pd.DataFrame] = []
    for round_id, round_df in round_rows.groupby("round_id", sort=True):
        pair_ids = round_df["pair_id"].astype(str).tolist()
        prompt_roster = pd.DataFrame([roster_by_pair[pair_id] for pair_id in pair_ids])
        part = _build_story_generation_schedule_from_roster(
            roster=prompt_roster,
            counterbalance=counterbalance,
            n_replicates=n_replicates,
        )
        part["round_id"] = round_id
        part["prompt_pair_count_in_round"] = int(round_df["prompt_pair_count_in_round"].iloc[0])
        part["eval_pair_count_in_round"] = int(round_df["eval_pair_count_in_round"].iloc[0])
        part["evaluation_scope"] = "compare stage1 TD outputs to held-out TD bundles and stage2 SLI outputs to held-out SLI bundles"
        schedule_parts.append(part)

    schedule = pd.concat(schedule_parts, ignore_index=True).sort_values(
        ["round_id", "pair_order", "replicate_id", "story_slot_order"]
    ).reset_index(drop=True)
    if output_csv is not None:
        output_path = _coerce_path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        schedule.to_csv(output_path, index=False)
    return schedule


def build_rotating_eval_manifest(
    frozen_roster_csv: str | Path,
    pairs_per_round: int = 3,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    roster = load_frozen_roster(frozen_roster_csv)
    round_rows = build_rotating_prompt_rounds(
        frozen_roster_csv=frozen_roster_csv,
        pairs_per_round=pairs_per_round,
    )
    prompt_pairs_by_round = {
        round_id: set(group["pair_id"].astype(str).tolist())
        for round_id, group in round_rows.groupby("round_id", sort=True)
    }

    rows: list[dict[str, object]] = []
    for round_id, prompt_pairs in prompt_pairs_by_round.items():
        round_index = int(round_rows.loc[round_rows["round_id"] == round_id, "round_index"].iloc[0])
        for _, pair in roster.iterrows():
            pair_id = str(pair["pair_id"])
            if pair_id in prompt_pairs:
                continue
            rows.append(
                {
                    "round_id": round_id,
                    "round_index": round_index,
                    "eval_pair_id": pair_id,
                    "eval_pair_order": int(pair["pair_order"]),
                    "eval_group": "SLI",
                    "comparison_target_group": "SLI",
                    "eval_child_id": int(pair["sli_child_id"]),
                    "eval_file_kideval": pair["sli_file_kideval"],
                    "eval_age": pair["sli_age"],
                    "eval_age_months_approx": float(pair["sli_age_months_approx"]),
                    "eval_severity_band": pair["sli_severity_band"],
                    "eval_profile_label": pair["sli_profile_label"],
                    "bundle_id": f"real_{int(pair['sli_child_id'])}",
                    "story_slots": "|".join(_STORY_ORDER),
                    "evaluation_pool_policy": "all non-prompt pairs in this round",
                }
            )
            rows.append(
                {
                    "round_id": round_id,
                    "round_index": round_index,
                    "eval_pair_id": pair_id,
                    "eval_pair_order": int(pair["pair_order"]),
                    "eval_group": "TD",
                    "comparison_target_group": "TD",
                    "eval_child_id": int(pair["td_child_id"]),
                    "eval_file_kideval": pair["td_file_kideval"],
                    "eval_age": pair["td_age"],
                    "eval_age_months_approx": float(pair["td_age_months_approx"]),
                    "eval_severity_band": "typical",
                    "eval_profile_label": "typical",
                    "bundle_id": f"real_{int(pair['td_child_id'])}",
                    "story_slots": "|".join(_STORY_ORDER),
                    "evaluation_pool_policy": "all non-prompt pairs in this round",
                }
            )

    eval_df = pd.DataFrame(rows).sort_values(
        ["round_id", "eval_group", "eval_pair_order"]
    ).reset_index(drop=True)
    if output_csv is not None:
        output_path = _coerce_path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        eval_df.to_csv(output_path, index=False)
    return eval_df


def _append_feature_placeholders(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in _FEATURE_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    return out


def build_bundle_feature_template(
    metadata_df: pd.DataFrame,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    required = {"bundle_id", "source", "group", "cohort", "cluster_id"}
    missing = required.difference(metadata_df.columns)
    if missing:
        raise ValueError(f"Metadata frame is missing required columns: {sorted(missing)}")

    template = _append_feature_placeholders(metadata_df)
    ordered_columns = [
        "bundle_id",
        "source",
        "group",
        "cohort",
        "age",
        "age_months_approx",
        "cluster_id",
        "pair_id",
        "replicate_id",
        "round_id",
        "sli_child_id",
        "sli_file_kideval",
        "sli_age",
        "sli_age_months_approx",
        "sli_severity_band",
        "sli_profile_label",
        "td_child_id",
        "td_file_kideval",
        "td_age",
        "td_age_months_approx",
        "age_gap_months",
        "n_stories",
        "story_slots",
        "target_story_order",
        "prompt_td_child_id",
        "prompt_sli_child_id",
        *_FEATURE_COLUMNS,
    ]
    columns = [column for column in ordered_columns if column in template.columns]
    template = template[columns].copy()

    if output_csv is not None:
        output_path = _coerce_path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        template.to_csv(output_path, index=False)

    return template


def build_real_bundle_template(
    dev_measures_csv: str | Path,
    severity_profile_csv: str | Path,
    age_min_months: int = 48,
    age_max_months: int = 59,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    dev = pd.read_csv(_coerce_path(dev_measures_csv))
    sev = pd.read_csv(_coerce_path(severity_profile_csv))

    dev = dev[dev["Group"].isin(["SLI", "TD"])].copy()
    sev = sev[sev["Group"].isin(["SLI", "TD"])].copy()
    age_mask = pd.to_numeric(dev["Age(Month)"], errors="coerce").between(age_min_months, age_max_months)
    dev = dev.loc[age_mask].copy()
    sev = sev.loc[pd.to_numeric(sev["Age(Month)"], errors="coerce").between(age_min_months, age_max_months)].copy()

    merged = dev.merge(
        sev[
            [
                "File_ID",
                "Group",
                "severity_band_sli_tertile_sli_only",
                "severity_core_composite",
                "severity_domain_composite",
                "severity_structural",
                "severity_lexical",
                "severity_disruption",
                "severity_morphosyntax_burden",
            ]
        ],
        on=["File_ID", "Group"],
        how="inner",
        validate="one_to_one",
    )

    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        child_id = int(row["File_ID"])
        group = str(row["Group"])
        age = str(row["Age"])
        age_months = float(pd.to_numeric(row["Age_Month_approx"], errors="coerce"))
        rows.append(
            {
                "bundle_id": f"real_{child_id}",
                "source": "real",
                "group": group,
                "cohort": f"{int(pd.to_numeric(row['Age(Month)'], errors='coerce')) // 12}-year-old",
                "age": age,
                "age_months_approx": age_months,
                "cluster_id": str(child_id),
                "pair_id": "",
                "replicate_id": "",
                "sli_child_id": child_id if group == "SLI" else "",
                "sli_file_kideval": row["File_kideval"] if group == "SLI" else "",
                "sli_age": age if group == "SLI" else "",
                "sli_age_months_approx": age_months if group == "SLI" else np.nan,
                "sli_severity_band": row["severity_band_sli_tertile_sli_only"] if group == "SLI" else "typical",
                "sli_profile_label": _profile_label(row) if group == "SLI" else "typical",
                "td_child_id": child_id if group == "TD" else "",
                "td_file_kideval": row["File_kideval"] if group == "TD" else "",
                "td_age": age if group == "TD" else "",
                "td_age_months_approx": age_months if group == "TD" else np.nan,
                "age_gap_months": np.nan,
                "n_stories": 6,
                "story_slots": "|".join(_STORY_ORDER),
                "target_story_order": "|".join(_STORY_ORDER),
                "prompt_td_child_id": "",
                "prompt_sli_child_id": "",
                "severity_core_composite": row["severity_core_composite"],
                "severity_domain_composite": row["severity_domain_composite"],
                "severity_structural": row["severity_structural"],
                "severity_lexical": row["severity_lexical"],
                "severity_disruption": row["severity_disruption"],
                "severity_morphosyntax_burden": row["severity_morphosyntax_burden"],
            }
        )

    real_template = pd.DataFrame(rows)
    real_template = build_bundle_feature_template(real_template, output_csv=output_csv)
    return real_template


def build_generation_package(
    dev_measures_csv: str | Path,
    severity_profile_csv: str | Path,
    counterbalance_csv: str | Path,
    output_dir: str | Path,
    n_replicates: int = 10,
    age_min_months: int = 48,
    age_max_months: int = 59,
    cohort_label: str | None = None,
) -> GenerationPackageArtifacts:
    out_dir = _coerce_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_cohort_label = _resolve_cohort_label(
        age_min_months=age_min_months,
        age_max_months=age_max_months,
        cohort_label=cohort_label,
    )
    cohort_slug = _cohort_slug(resolved_cohort_label)

    roster_csv = out_dir / f"{cohort_slug}_frozen_roster.csv"
    schedule_csv = out_dir / f"{cohort_slug}_generation_schedule.csv"
    bundle_index_csv = out_dir / f"{cohort_slug}_bundle_index.csv"
    synthetic_template_csv = out_dir / f"{cohort_slug}_synthetic_bundle_template.csv"
    real_template_csv = out_dir / f"{cohort_slug}_real_bundle_template.csv"
    summary_json = out_dir / f"{cohort_slug}_generation_package_summary.json"

    build_frozen_roster_manifest(
        dev_measures_csv=str(dev_measures_csv),
        severity_profile_csv=str(severity_profile_csv),
        output_csv=str(roster_csv),
        age_min_months=age_min_months,
        age_max_months=age_max_months,
        cohort_label=resolved_cohort_label,
    )

    schedule = build_story_generation_schedule(
        frozen_roster_csv=roster_csv,
        counterbalance_csv=counterbalance_csv,
        n_replicates=n_replicates,
        output_csv=schedule_csv,
    )
    bundle_index = build_bundle_index(schedule, output_csv=bundle_index_csv)
    build_bundle_feature_template(bundle_index, output_csv=synthetic_template_csv)
    build_real_bundle_template(
        dev_measures_csv=dev_measures_csv,
        severity_profile_csv=severity_profile_csv,
        age_min_months=age_min_months,
        age_max_months=age_max_months,
        output_csv=real_template_csv,
    )

    summary = {
        "dev_measures_csv": str(_coerce_path(dev_measures_csv)),
        "severity_profile_csv": str(_coerce_path(severity_profile_csv)),
        "counterbalance_csv": str(_coerce_path(counterbalance_csv)),
        "output_dir": str(out_dir),
        "cohort_label": resolved_cohort_label,
        "cohort_slug": cohort_slug,
        "n_replicates": n_replicates,
        "age_min_months": age_min_months,
        "age_max_months": age_max_months,
        "n_pairs": int(bundle_index["pair_id"].nunique()),
        "n_bundles": int(bundle_index["bundle_id"].nunique()),
        "n_story_rows": int(len(schedule)),
        "feature_columns": bundle_feature_columns(),
        "story_order": _STORY_ORDER,
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return GenerationPackageArtifacts(
        output_dir=out_dir,
        roster_csv=roster_csv,
        schedule_csv=schedule_csv,
        bundle_index_csv=bundle_index_csv,
        synthetic_template_csv=synthetic_template_csv,
        real_template_csv=real_template_csv,
        summary_json=summary_json,
    )


def build_rotating_generation_package(
    dev_measures_csv: str | Path,
    severity_profile_csv: str | Path,
    counterbalance_csv: str | Path,
    output_dir: str | Path,
    n_replicates: int = 10,
    age_min_months: int = 48,
    age_max_months: int = 59,
    pairs_per_round: int = 3,
    cohort_label: str | None = None,
) -> RotatingGenerationPackageArtifacts:
    out_dir = _coerce_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_cohort_label = _resolve_cohort_label(
        age_min_months=age_min_months,
        age_max_months=age_max_months,
        cohort_label=cohort_label,
    )
    cohort_slug = _cohort_slug(resolved_cohort_label)

    roster_csv = out_dir / f"{cohort_slug}_frozen_roster.csv"
    prompt_rounds_csv = out_dir / f"{cohort_slug}_prompt_rounds.csv"
    schedule_csv = out_dir / f"{cohort_slug}_rotating_generation_schedule.csv"
    bundle_index_csv = out_dir / f"{cohort_slug}_rotating_bundle_index.csv"
    synthetic_template_csv = out_dir / f"{cohort_slug}_rotating_synthetic_bundle_template.csv"
    eval_manifest_csv = out_dir / f"{cohort_slug}_rotating_eval_manifest.csv"
    summary_json = out_dir / f"{cohort_slug}_rotating_generation_package_summary.json"

    build_frozen_roster_manifest(
        dev_measures_csv=str(dev_measures_csv),
        severity_profile_csv=str(severity_profile_csv),
        output_csv=str(roster_csv),
        age_min_months=age_min_months,
        age_max_months=age_max_months,
        cohort_label=resolved_cohort_label,
    )

    prompt_rounds = build_rotating_prompt_rounds(
        frozen_roster_csv=roster_csv,
        pairs_per_round=pairs_per_round,
        output_csv=prompt_rounds_csv,
    )
    schedule = build_rotating_generation_schedule(
        frozen_roster_csv=roster_csv,
        counterbalance_csv=counterbalance_csv,
        n_replicates=n_replicates,
        pairs_per_round=pairs_per_round,
        output_csv=schedule_csv,
    )
    bundle_index = build_bundle_index(schedule, output_csv=bundle_index_csv)
    build_bundle_feature_template(bundle_index, output_csv=synthetic_template_csv)
    eval_manifest = build_rotating_eval_manifest(
        frozen_roster_csv=roster_csv,
        pairs_per_round=pairs_per_round,
        output_csv=eval_manifest_csv,
    )

    summary = {
        "dev_measures_csv": str(_coerce_path(dev_measures_csv)),
        "severity_profile_csv": str(_coerce_path(severity_profile_csv)),
        "counterbalance_csv": str(_coerce_path(counterbalance_csv)),
        "output_dir": str(out_dir),
        "cohort_label": resolved_cohort_label,
        "cohort_slug": cohort_slug,
        "n_replicates": n_replicates,
        "age_min_months": age_min_months,
        "age_max_months": age_max_months,
        "pairs_per_round": pairs_per_round,
        "n_pairs": int(prompt_rounds["pair_id"].nunique()),
        "n_rounds": int(prompt_rounds["round_id"].nunique()),
        "n_prompt_pairs_total": int(len(prompt_rounds)),
        "n_synthetic_bundles": int(bundle_index["bundle_id"].nunique()),
        "n_story_rows": int(len(schedule)),
        "n_eval_rows": int(len(eval_manifest)),
        "n_eval_td_rows": int((eval_manifest["eval_group"] == "TD").sum()),
        "n_eval_sli_rows": int((eval_manifest["eval_group"] == "SLI").sum()),
        "feature_columns": bundle_feature_columns(),
        "story_order": _STORY_ORDER,
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return RotatingGenerationPackageArtifacts(
        output_dir=out_dir,
        roster_csv=roster_csv,
        prompt_rounds_csv=prompt_rounds_csv,
        schedule_csv=schedule_csv,
        bundle_index_csv=bundle_index_csv,
        synthetic_template_csv=synthetic_template_csv,
        eval_manifest_csv=eval_manifest_csv,
        summary_json=summary_json,
    )


def build_multi_age_rotating_generation_packages(
    dev_measures_csv: str | Path,
    severity_profile_csv: str | Path,
    counterbalance_csv: str | Path,
    output_dir: str | Path,
    age_bins: Sequence[tuple[str, int, int]] | None = None,
    n_replicates: int = 10,
    pairs_per_round: int = 3,
) -> MultiAgeRotatingGenerationArtifacts:
    out_dir = _coerce_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bins = list(age_bins) if age_bins is not None else [
        ("5-year-old", 60, 71),
        ("6-year-old", 72, 83),
        ("7-year-old", 84, 95),
        ("8-year-old", 96, 107),
        ("9-year-old", 108, 119),
    ]
    if not bins:
        raise ValueError("age_bins is empty")

    artifacts_by_label: dict[str, RotatingGenerationPackageArtifacts] = {}
    summary_rows: list[dict[str, object]] = []
    for cohort_label, age_min_months, age_max_months in bins:
        cohort_slug = _cohort_slug(cohort_label)
        cohort_dir = out_dir / cohort_slug
        cohort_dir.mkdir(parents=True, exist_ok=True)

        artifacts = build_rotating_generation_package(
            dev_measures_csv=dev_measures_csv,
            severity_profile_csv=severity_profile_csv,
            counterbalance_csv=counterbalance_csv,
            output_dir=cohort_dir,
            n_replicates=n_replicates,
            age_min_months=age_min_months,
            age_max_months=age_max_months,
            pairs_per_round=pairs_per_round,
            cohort_label=cohort_label,
        )
        artifacts_by_label[cohort_label] = artifacts

        summary_rows.append(
            {
                "cohort_label": cohort_label,
                "cohort_slug": cohort_slug,
                "age_min_months": int(age_min_months),
                "age_max_months": int(age_max_months),
                "output_dir": str(cohort_dir),
                "roster_csv": str(artifacts.roster_csv),
                "prompt_rounds_csv": str(artifacts.prompt_rounds_csv),
                "schedule_csv": str(artifacts.schedule_csv),
                "bundle_index_csv": str(artifacts.bundle_index_csv),
                "synthetic_template_csv": str(artifacts.synthetic_template_csv),
                "eval_manifest_csv": str(artifacts.eval_manifest_csv),
                "summary_json": str(artifacts.summary_json),
            }
        )

    summary = {
        "output_dir": str(out_dir),
        "dev_measures_csv": str(_coerce_path(dev_measures_csv)),
        "severity_profile_csv": str(_coerce_path(severity_profile_csv)),
        "counterbalance_csv": str(_coerce_path(counterbalance_csv)),
        "n_replicates": int(n_replicates),
        "pairs_per_round": int(pairs_per_round),
        "age_bins": [
            {
                "cohort_label": label,
                "age_min_months": int(min_m),
                "age_max_months": int(max_m),
            }
            for label, min_m, max_m in bins
        ],
        "cohort_packages": summary_rows,
    }
    summary_json = out_dir / "multi_age_rotating_generation_packages_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return MultiAgeRotatingGenerationArtifacts(
        output_dir=out_dir,
        cohort_packages=artifacts_by_label,
        summary_json=summary_json,
    )


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
