from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd


_KEEP_ORDER = [
    "File_ID",
    "File_kideval",
    "Group",
    "Sex",
    "Age(Month)",
    "#utterances",
    "#words",
    "DSS_tbl_warning_50",
    "MLU_Morphemes",
    "VOCD_D_optimum_average",
    "IPSyn_Total",
    "DSS_tbl_score",
    "Word_Errors_per_utt",
    "retracing_per_utt",
    "repetition_per_utt",
    "DSS_tbl_IP",
    "DSS_tbl_PP",
    "DSS_tbl_MV",
    "DSS_tbl_SV",
    "DSS_tbl_NG",
    "DSS_tbl_CNJ",
    "DSS_tbl_IR",
    "DSS_tbl_WHQ",
    "bare_marking_errors_per_word",
    "pronoun_errors_per_word",
    "past_overgeneralization_errors_per_past_opp",
]


_COLUMN_DECISIONS: dict[str, tuple[str, str, str]] = {
    "File_ID": ("keep", "key", "Stable merge key for child/file."),
    "File_kideval": ("keep", "key", "Transcript identifier for traceability back to the corpus."),
    "Group": ("keep", "grouping", "Clinical group label needed for TD-based normalization and DLD comparisons."),
    "Sex": ("keep", "grouping", "Useful demographic covariate if later needed for sensitivity checks."),
    "Age(Month)": ("keep", "covariate", "Primary age covariate for TD-based developmental normalization."),
    "#utterances": ("keep", "qc_exposure", "Single retained utterance-count variable for sample-size QC and optional normalization."),
    "#words": ("keep", "qc_exposure", "Single retained word-count variable for sample-size QC and optional normalization."),
    "DSS_tbl_warning_50": ("keep", "qc_flag", "Flags transcripts that did not meet DSS 50-sentence criterion."),
    "MLU_Morphemes": ("keep", "feature", "Chosen MLU representative; keep morpheme-based rather than word-based MLU."),
    "VOCD_D_optimum_average": ("keep", "feature", "Primary lexical diversity measure."),
    "IPSyn_Total": ("keep", "feature", "Global syntactic complexity measure."),
    "DSS_tbl_score": ("keep", "feature", "Primary DSS score from the direct DSS table output."),
    "Word_Errors": ("drop", "raw_count_archival", "Raw count retained in the master archive, but severity modeling uses the per-utterance rate."),
    "retracing": ("drop", "raw_count_archival", "Raw count retained in the master archive, but severity modeling uses the per-utterance rate."),
    "repetition": ("drop", "raw_count_archival", "Raw count retained in the master archive, but severity modeling uses the per-utterance rate."),
    "Word_Errors_per_utt": ("keep", "feature_rate", "General error burden normalized by utterance count."),
    "retracing_per_utt": ("keep", "feature_rate", "Retracing/disruption rate normalized by utterance count."),
    "repetition_per_utt": ("keep", "feature_rate", "Repetition/disruption rate normalized by utterance count."),
    "DSS_tbl_IP": ("keep", "feature_subscale", "DSS subscale: indefinite pronouns / noun modifiers."),
    "DSS_tbl_PP": ("keep", "feature_subscale", "DSS subscale: personal pronouns."),
    "DSS_tbl_MV": ("keep", "feature_subscale", "DSS subscale: main verbs."),
    "DSS_tbl_SV": ("keep", "feature_subscale", "DSS subscale: secondary verbs."),
    "DSS_tbl_NG": ("keep", "feature_subscale", "DSS subscale: negatives."),
    "DSS_tbl_CNJ": ("keep", "feature_subscale", "DSS subscale: conjunctions."),
    "DSS_tbl_IR": ("keep", "feature_subscale", "DSS subscale: interrogative reversals."),
    "DSS_tbl_WHQ": ("keep", "feature_subscale", "DSS subscale: wh-questions."),
    "chi_utterances_with_error_code": ("drop", "secondary_qc", "Useful descriptive count, but not retained in the first-pass severity feature set."),
    "error_total_annotations": ("drop", "secondary_qc", "Useful descriptive count, but not retained in the first-pass severity feature set."),
    "bare_marking_errors": ("drop", "raw_count_archival", "Raw grouped count retained in the master archive; severity modeling uses the per-word rate."),
    "bare_marking_errors_per_utt": ("drop", "rate_not_selected", "Per-utterance version not selected; morphosyntactic grouped errors are normalized per word."),
    "bare_marking_errors_per_word": ("keep", "feature_rate", "Grouped bare-marking error rate normalized by word count."),
    "determiner_errors": ("drop", "raw_count_archival", "Raw grouped count retained in the master archive; excluded from profiling after cleaning because the remaining feature is too sparse."),
    "determiner_errors_per_utt": ("drop", "rate_not_selected", "Per-utterance version not selected; excluded from profiling after cleaning because the remaining feature is too sparse."),
    "determiner_errors_per_word": ("drop", "too_sparse", "Excluded from the severity/profile table after removing m:allo contamination left the feature too sparse."),
    "pronoun_errors": ("drop", "raw_count_archival", "Raw grouped count retained in the master archive; severity modeling uses the per-word rate."),
    "pronoun_errors_per_utt": ("drop", "rate_not_selected", "Per-utterance version not selected; morphosyntactic grouped errors are normalized per word."),
    "pronoun_errors_per_word": ("keep", "feature_rate", "Grouped pronoun error rate normalized by word count."),
    "rule_generalization_errors": ("drop", "raw_count_archival", "Legacy broad grouped count retained in the master archive; replaced in profiling by a past-specific overgeneralization measure."),
    "rule_generalization_errors_per_utt": ("drop", "legacy_rate", "Legacy broad grouped rate retained in the archive table only."),
    "rule_generalization_errors_per_word": ("drop", "legacy_rate", "Legacy broad grouped rate retained in the archive table only; replaced in profiling by a past-specific opportunity-based measure."),
    "overt_past_finite_verbs": ("drop", "denominator_component", "Component of the past overgeneralization opportunity denominator; retained in the master archive only."),
    "bare_past_error_tokens": ("drop", "denominator_component", "Component of the past overgeneralization opportunity denominator; retained in the master archive only."),
    "past_verb_opportunities": ("drop", "denominator_component", "Total denominator for the past overgeneralization rate; retained in the master archive only."),
    "past_overgeneralization_errors": ("drop", "raw_count_archival", "Raw count retained in the master archive; severity modeling uses the opportunity-based rate."),
    "past_overgeneralization_errors_per_past_opp": ("keep", "feature_rate", "Past overgeneralization normalized by overt past finite verbs plus explicit bare-past error tokens."),
    "Language": ("drop", "constant", "Constant provenance column; no analytical value."),
    "Corpus": ("drop", "constant", "Constant provenance column; no analytical value."),
    "Code": ("drop", "constant", "Constant provenance column; no analytical value."),
    "Race": ("drop", "constant_placeholder", "Empty placeholder metadata column."),
    "SES": ("drop", "constant_placeholder", "Empty placeholder metadata column."),
    "Role": ("drop", "constant", "Constant provenance column; no analytical value."),
    "Education": ("drop", "constant_placeholder", "Empty placeholder metadata column."),
    "Custom_field": ("drop", "constant_placeholder", "Empty placeholder metadata column."),
    "Design": ("drop", "constant", "Constant provenance column; no analytical value."),
    "Activity": ("drop", "constant", "Constant provenance column; no analytical value."),
    "Group_2": ("drop", "redundant_group", "Redundant alternate group column; `Group` is the primary label."),
    "Total_Utts": ("drop", "redundant_count", "Near-duplicate utterance count; keep `#utterances` only."),
    "MLU_Utts": ("drop", "redundant_count", "Near-duplicate utterance count; keep `#utterances` only."),
    "MLU_Words": ("drop", "redundant_mlu", "Word-based MLU variant; keep `MLU_Morphemes` only."),
    "MLU50_Utts": ("drop", "constant", "Constant column; no analytical value."),
    "MLU50_Words": ("drop", "redundant_mlu", "Highly correlated MLU-50 variant not needed in the first-pass severity table."),
    "MLU50_Morphemes": ("drop", "redundant_mlu", "Highly correlated MLU-50 variant not needed in the first-pass severity table."),
    "FREQ_types": ("drop", "redundant_lexical", "Raw type count is sample-length sensitive; use `VOCD_D_optimum_average` instead."),
    "FREQ_tokens": ("drop", "redundant_count", "Exact duplicate of `mor_Words` and redundant with retained `#words`."),
    "NDW_100": ("drop", "redundant_lexical", "Alternative lexical diversity measure; keep VOCD as the primary lexical index."),
    "Verbs_Utt": ("drop", "secondary_feature", "Potentially useful later, but excluded from the first-pass core severity set."),
    "TD_Words": ("drop", "redundant_count", "Near-duplicate word count; keep `#words` only."),
    "TD_Utts": ("drop", "redundant_count", "Near-duplicate utterance count; keep `#utterances` only."),
    "TD_Time_(secs)": ("drop", "secondary_qc", "Timing variable not needed in the first-pass severity table."),
    "TD_Words_Time": ("drop", "secondary_qc", "Timing-normalized word rate not needed in the first-pass severity table."),
    "TD_Utts_Time": ("drop", "secondary_qc", "Timing-normalized utterance rate not needed in the first-pass severity table."),
    "Utt_Errors": ("drop", "constant", "Constant column; no analytical value."),
    "DSS_Utts": ("drop", "secondary_qc", "Sentence-count support variable; excluded from first-pass feature table."),
    "DSS": ("drop", "redundant_dss", "Near-duplicate of `DSS_tbl_score`; keep table-derived DSS only."),
    "IPSyn_Utts": ("drop", "secondary_qc", "Support variable for IPSyn, not a retained core feature."),
    "mor_Words": ("drop", "duplicate", "Exact duplicate of `FREQ_tokens`; retain neither because `#words` is kept."),
    "*-PRESP": ("drop", "secondary_morphology", "Morph marker count left out of the first-pass compact severity set."),
    "in": ("drop", "secondary_morphology", "Morph marker count left out of the first-pass compact severity set."),
    "on": ("drop", "secondary_morphology", "Morph marker count left out of the first-pass compact severity set."),
    "*-PL": ("drop", "secondary_morphology", "Morph marker count left out of the first-pass compact severity set."),
    "irr-PAST": ("drop", "secondary_morphology", "Morph marker count left out of the first-pass compact severity set."),
    "*-POSS": ("drop", "secondary_morphology", "Morph marker count left out of the first-pass compact severity set."),
    "u-cop": ("drop", "secondary_morphology", "Morph marker count left out of the first-pass compact severity set."),
    "det:art": ("drop", "secondary_morphology", "Morph marker count left out of the first-pass compact severity set."),
    "*-PAST": ("drop", "secondary_morphology", "Morph marker count left out of the first-pass compact severity set."),
    "*-3S": ("drop", "secondary_morphology", "Morph marker count left out of the first-pass compact severity set."),
    "irr-3S": ("drop", "secondary_morphology", "Morph marker count left out of the first-pass compact severity set."),
    "u-aux": ("drop", "secondary_morphology", "Morph marker count left out of the first-pass compact severity set."),
    "c-cop": ("drop", "secondary_morphology", "Morph marker count left out of the first-pass compact severity set."),
    "c-aux": ("drop", "secondary_morphology", "Morph marker count left out of the first-pass compact severity set."),
    "Total_non_zero_mors": ("drop", "secondary_morphology", "Summary morphology count left out of the first-pass compact severity set."),
    "File_mlt": ("drop", "provenance", "Source-specific path column; retained in the archive table only."),
    "Age": ("drop", "redundant_age", "String age representation; keep numeric `Age(Month)` only."),
    "#turns": ("drop", "secondary_qc", "Turn count not required in the first-pass severity table."),
    "words/turns": ("drop", "redundant_rate", "Turn-based rate excluded from first-pass compact severity set."),
    "utterances/turns": ("drop", "redundant_rate", "Turn-based rate excluded from first-pass compact severity set."),
    "words/utterances": ("drop", "redundant_mlu", "Near-duplicate of word-based MLU measures; keep `MLU_Morphemes` only."),
    "Standard deviation": ("drop", "secondary_qc", "MLT dispersion measure not included in the first-pass compact severity set."),
    "Age_Month_approx": ("drop", "redundant_age", "Approximate month age; keep exact `Age(Month)` only."),
    "DSS_tbl_source": ("drop", "provenance", "File-path provenance column; keep only in archive table."),
    "DSS_tbl_relpath": ("drop", "provenance", "File-path provenance column; keep only in archive table."),
    "DSS_tbl_transcript": ("drop", "provenance", "File-path provenance column; keep only in archive table."),
    "DSS_tbl_transcript_relpath": ("drop", "provenance", "File-path provenance column; keep only in archive table."),
    "DSS_tbl_command": ("drop", "constant", "Constant command provenance; no analytical value."),
    "DSS_tbl_score_is_na": ("drop", "constant", "Constant QC column in the current DSS batch."),
    "DSS_tbl_S": ("drop", "redundant_dss", "Sentence-point subscore is tightly coupled to overall DSS scoring and omitted from the compact set."),
    "DSS_tbl_TOT": ("drop", "redundant_dss", "Raw total underlying DSS score; keep `DSS_tbl_score` only."),
}


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_severity_feature_table(
    input_csv: str,
    output_csv: str,
    output_spec_csv: str,
    output_summary_json: str,
) -> int:
    input_path = Path(input_csv).expanduser().resolve()
    out_csv = Path(output_csv).expanduser().resolve()
    out_spec = Path(output_spec_csv).expanduser().resolve()
    out_summary = Path(output_summary_json).expanduser().resolve()

    df = pd.read_csv(input_path)

    missing_keep = [column for column in _KEEP_ORDER if column not in df.columns]
    if missing_keep:
        raise ValueError(f"Missing required keep columns: {', '.join(missing_keep)}")

    missing_decisions = [column for column in df.columns if column not in _COLUMN_DECISIONS]
    if missing_decisions:
        raise ValueError(f"Columns missing spec decisions: {', '.join(missing_decisions)}")

    filtered = df[_KEEP_ORDER].copy()
    filtered.to_csv(out_csv, index=False)

    spec_rows: list[dict[str, str]] = []
    for column in df.columns:
        action, category, reason = _COLUMN_DECISIONS[column]
        spec_rows.append(
            {
                "column": column,
                "action": action,
                "category": category,
                "included_in_severity_features_raw": "1" if column in _KEEP_ORDER else "0",
                "reason": reason,
            }
        )
    _write_csv(
        out_spec,
        ["column", "action", "category", "included_in_severity_features_raw", "reason"],
        spec_rows,
    )

    summary = {
        "input_csv": str(input_path),
        "output_csv": str(out_csv),
        "rows": len(filtered),
        "columns_kept": len(_KEEP_ORDER),
        "kept_columns": _KEEP_ORDER,
        "dropped_columns": [column for column in df.columns if column not in _KEEP_ORDER],
    }
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote filtered severity feature table to {out_csv}")
    print(f"Wrote feature spec to {out_spec}")
    print(f"Wrote summary to {out_summary}")
    return 0
