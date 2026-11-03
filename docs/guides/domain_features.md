# KIDEVAL features in the main notebook

The main pipeline and the existing smoke notebook now contain a linked
**Domain-specific features — KIDEVAL English UD** section. It is disabled by
default. R analyses remain in their separate notebook; this block extracts
features without running statistical comparisons.

## Run from saved annotations

1. Run the repository cell near the top of the notebook.
2. Jump to **domain features** in the clickable outline.
3. Set `RUN_DOMAIN_FEATURES=True`. List completed final-UD manifests or
   processing ZIPs in `FEATURE_INPUTS`. Include the real ENNI run and each
   synthetic generator run once. Gillam remains external classifier data.
4. Run support, runtime, extraction, preview and export. In the original smoke
   project, upload `ICL-PILOT_domain_features_support_20261102.zip` from the
   handoff folder first. The support cell verifies the archive and its members
   before installing the reviewed files; unknown changes are preserved.

After a new annotation run, leave `FEATURE_INPUTS=[]` to use its completed
`morphosyntax_manifest` automatically. This route writes under
`RUN_OUTPUT_DIR/domain_features/`, which the regular processing archive includes.
For independent saved-input runs, Colab defaults to
`/content/drive/MyDrive/ICL-PILOT/domain_features/`; local runs default to
`data/generated/domain_features/`. Each extraction creates a fresh directory.
`FEATURE_OUTPUT_DIR` may select a different fresh directory. Failed outputs and
logs remain available; reruns never overwrite an earlier directory.

No generation, TalkTag, Batchalign or GPU inference is required for saved-input
extraction. Python needs pandas; macOS or Linux/Colab needs `make` and a C++
compiler. The native tools are built once, then verified and reused.

## Units and preservation

Real units contain one child's six ENNI stories. Synthetic units contain one
model/run/pair/replicate/stage's six stories. Canonical order is A1, A2, A3, B1,
B2, B3. The inventory retains child identities, canonical ages and sex, model
revisions, generation lineage and pilot status. Duplicate identities fail.
Incomplete bundles remain rows with missing features; generation-review flags
do not exclude stories or bundles.

Final annotated sources are read only and checked against their manifests.
The block also verifies their hashes after extraction. New analysis bundles
preserve utterances and annotations. They omit only `@Media` links and timing
bullets, because canonical story ordering need not match original recording
order. Removed items are counted in the feature manifest; time-based measures
are excluded. The original annotated transcripts retain their audio metadata.
Chatter validates the analysis bundles before measurement.

## Native settings and missing values

The [official CLAN manual](https://talkbank.org/0info/manuals/CLAN.html) documents
English UD support. The pinned official Unix source is CLAN V 21-Sep-2026 11:00,
archive SHA-256
`a9dabc78eb74b153e5e459b92592d4a310dda54af11cf4de71745c7bd0697706`.
The original-smoke support ZIP includes this source archive; an updated checkout
uses `data/inputs/tools/unix-clan_2026-09-21.zip` when available, otherwise the
installer downloads and verifies the configured archive. A changed upstream
archive is rejected. Binaries and language resources are verified on reuse.

The commands are `kideval +0 +lengu +t*CHI +L<library> unit_*.cha` and the
corresponding `dss` command for category tables. `+0` selects file input in a
noninteractive Python subprocess. The native English UD script enables UD DSS
and disables IPSyn. No normative comparison is requested. Default raw counts
are retained; `+o4` is not used because it requests percentages. The default
50-sentence DSS requirement is unchanged. Native VOCD sampling is stochastic;
the manifest records that fact rather than claiming a fixed seed.

The selected table includes MLU, lexical diversity, morphology, retracing,
repetition, DSS and DSS category totals. Retracing and repetition additionally
have per-utterance rates. DSS category totals retain their developmental-score
meaning. Native `NA` values remain missing. TalkTag coverage is complete,
partial or unknown; DSS sentence scores remain missing when coverage is not
complete. The raw report retains CLAN's original scores for inspection. Short
samples that fail native DSS eligibility do not receive an invented score.

Native `Word_Errors` and `Utt_Errors` remain in the raw report only. They are not
the study's error measures, which must exclude preposition-error codes. Custom
error/pronoun/bare-marking/past-opportunity extraction and the statistical
revisions in R remain separate work. Do not pass the raw error counts into R
as substitutes for those measures. Pilot rows are explicitly `test_only` and
must not be mixed into study results.

## Outputs

- `kideval_features.csv`: combined selected features with identities and flags.
- `real_kideval_features.csv`, `synthetic_kideval_features.csv`: separate views.
- `unit_inventory.csv`: all complete and incomplete units.
- `kideval_all_columns.csv` and native `.kideval.xls`: unmodified tool columns.
- `bundles/`: derived CHAT units and native DSS reports/category tables.
- `clan_runtime.json`, `feature_manifest.json`, logs and validation receipts:
  pinned runtime, input/software/output hashes, commands and coverage.
- `exports/`: a ZIP of the feature artifacts, also copied to Colab Contents
  for VS Code download when running in Colab.

The software check used six real pilot children and four synthetic bundles.
It establishes that the documented execution/export route works on these saved
annotations. Full-study extraction and R inference remain pending.
