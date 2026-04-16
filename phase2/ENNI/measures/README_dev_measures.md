# `dev_measures.csv`

This file is the current child-level developmental-measures table for the ENNI severity profiling workflow.

Path:

- [`phase2/measures/dev_measures.csv`](/Users/shamiraventurini/PycharmProjects/ICL-PILOT/phase2/measures/dev_measures.csv)

## What One Row Represents

Each row corresponds to one ENNI child/file.

- total rows: `360`
- unique `File_ID`: `360`

`File_ID` is the stable numeric child identifier used to merge outputs across CLAN analyses.

## What The File Contains

The table is a merge of three analysis sources:

1. `kideval` measures
2. `mlt` measures
3. DSS table-derived category features

### 1. KIDEVAL columns

These are the original developmental and morphosyntactic summary fields from KIDEVAL, including:

- metadata: `Language`, `Corpus`, `Code`, `Age(Month)`, `Sex`, `Group`, `Design`, `Activity`
- length/diversity measures: `MLU_Morphemes`, `NDW_100`, `VOCD_D_optimum_average`
- fluency/error fields: `Word_Errors`, `Utt_Errors`, `retracing`, `repetition`
- syntactic fields: `DSS`, `IPSyn_Total`
- morph marker counts such as `*-PAST`, `*-3S`, `u-aux`, `c-cop`

Provenance:

- [`phase2/measures/kideval/kideval_clean.csv`](/Users/shamiraventurini/PycharmProjects/ICL-PILOT/phase2/measures/kideval/kideval_clean.csv)

### 2. MLT columns

These are the utterance/turn length summaries from MLT:

- `#utterances`
- `#turns`
- `#words`
- `words/turns`
- `utterances/turns`
- `words/utterances`
- `Standard deviation`
- `Age`
- `Age_Month_approx`

Provenance:

- [`phase2/measures/mlt/mlt_clean.csv`](/Users/shamiraventurini/PycharmProjects/ICL-PILOT/phase2/measures/mlt/mlt_clean.csv)

### 3. DSS table-derived columns

These were parsed from CLAN `*.tbl.cex` outputs and added with the `DSS_tbl_` prefix.

Source/path fields:

- `DSS_tbl_source`
- `DSS_tbl_relpath`
- `DSS_tbl_transcript`
- `DSS_tbl_transcript_relpath`
- `DSS_tbl_command`

Score/QC fields:

- `DSS_tbl_score`
- `DSS_tbl_score_is_na`
- `DSS_tbl_warning_50`

Category totals:

- `DSS_tbl_IP`
- `DSS_tbl_PP`
- `DSS_tbl_MV`
- `DSS_tbl_SV`
- `DSS_tbl_NG`
- `DSS_tbl_CNJ`
- `DSS_tbl_IR`
- `DSS_tbl_WHQ`
- `DSS_tbl_S`
- `DSS_tbl_TOT`

Provenance:

- [`phase2/measures/dss/dss_category_features.csv`](/Users/shamiraventurini/PycharmProjects/ICL-PILOT/phase2/measures/dss/dss_category_features.csv)
- [`phase2/measures/dss/dss_qc_summary.json`](/Users/shamiraventurini/PycharmProjects/ICL-PILOT/phase2/measures/dss/dss_qc_summary.json)

## Important Notes

- `DSS` and `DSS_tbl_score` are not the same field.
  - `DSS` comes from KIDEVAL.
  - `DSS_tbl_score` comes from the parsed CLAN DSS table output.
- `DSS_tbl_warning_50 = 1` means CLAN flagged that transcript for not meeting the 50-sentence DSS criterion.
  - current count: `14` rows
  - these rows are still present in the table and should be flagged in downstream analysis
- `DSS_tbl_score_is_na` is currently `0` for all rows in this version of the file.

## Merge Logic

The file was built by merging on `File_ID`.

- `kideval` and `mlt` were joined by numeric child/file ID
- DSS table features were added by `File_ID`
- duplicate numeric ID `523` in the corpus was resolved by matching transcript path, so the correct DSS row was attached to the correct master row

## Intended Use

This is the current base table for severity profiling.

Recommended next use:

- compute age-adjusted TD-based z-scores
- derive domain scores
- derive a composite severity score
- optionally exclude or sensitivity-flag rows with `DSS_tbl_warning_50 = 1`
