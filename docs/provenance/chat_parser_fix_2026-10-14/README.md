# CHAT parser repair — 14 October 2026

The user requested a fix to CHAT parsing during generation preparation.
The parser now reads CHAT ages as `years;months.days`, preserves raw ages and
unknown metadata, joins wrapped CHI main tiers before normalization, and keeps
dependent-tier content separate. Recognizable orphan annotation blocks and
redundant delimiter continuations in processed input are excluded and reported.

The extraction command writes a new CSV with a source-hashed parsing QC sidecar
and refuses to overwrite either output. The pre-existing surface normalization
policy is preserved, including repetitions and children's grammatical forms.
The tier-removal utility was also repaired to remove complete morphology and
dependency tiers, including their continuations, and to run on Python 3.8.
That mutating utility was tested on fixtures only; no corpus tiers were removed.

## Active generation input

`data/processed/generation_inputs/story_unit_narratives_chat_v2.csv`

SHA-256: `89643c6ca494e525b47eb3863a36bea3d702402c9dd1e3e5d42b401fa9828cbc`.

Its `*.parsing_qc.json` sidecar records 361 source files, relative paths, source
hashes and all parsing issues. `configs/generation/experiment.json` points new
runs to this version. The original CSV remains unchanged for historical
analyses and original prompt provenance.

## Verification

- 53 focused tests passed on Python 3.8.5, with three pre-existing dependency warnings.
- All 361 source CHAT files are byte-for-byte unchanged.
- All 2,166 story texts and utterance arrays match the old CSV. Only
  `Age_Month` changed, becoming populated in 2,160 rows.
- All 564 selected exemplar ages match the master roster.
- All 2,820 complete generation input rows and the evaluation manifest retain
  their pre-repair fingerprints when using the new CSV.
- 6,036 protected research files and all three R source files are unchanged.
- Python syntax, the extraction CLI/help entry point and `git diff --check` passed.

`TD/B/523.cha` has missing CHI age/group metadata and a filename-ID collision
with an SLI-labelled source. Its six story rows remain unknown; the directory
and `@Types` label were not used to fill them. These rows are not selected as
TD prompts. The QC sidecar reports 33,437 orphan annotation continuation lines
and 948 redundant delimiter continuations across all source files. These are
parsing/QC counts, not clinical error counts.

`corpus_validation.json`, `csv_changes.json`, `final_verification.json`, and
`source_chat_inventory.json` contain the comparisons and source hashes.
Reproduction and parsing policy are documented in `docs/chat_narrative_parsing.md`.
The official syntax reference is the
[TalkBank CHAT manual](https://talkbank.org/0info/manuals/CHAT.html).

## Backup and intermediate checks

Verified external directory:

`/Volumes/TranscendBU/ICL-PILOT_78a5cadad659_20261014T131515Z/CHATParsing/`

`chat_parser_snapshot.tar.gz` preserves the original parser, CLI, input CSV,
configuration and relevant documentation. Its SHA-256 is
`9ef042d5c13c60cd1a9c9391be07653157b6452d47609fa0d1fbea1a075c9c83`.
`tier_strip_code_snapshot.tar.gz` preserves that utility and its original tests;
its SHA-256 is `64a37a973c2c8a3be11af80f9cdd4e1d59870614ef54f9a5de0bf928ba7cc3a8`.
All archived members were decompressed and checked against their sources.

Two intermediate corpus comparisons caught orphan quoted-speech terminators
and prefixed morphology fragments. Both candidates were externally archived
and verified before replacement; neither was configured for generation or
used in scientific analyses. Regression tests now cover those formats.

No model inference, scientific analysis, raw-data repair, annotation regeneration,
commit or publication was performed. Existing unrelated worktree changes remain.
