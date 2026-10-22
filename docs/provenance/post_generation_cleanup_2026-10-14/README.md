# First post-generation cleanup — 14 October 2026

The user deferred Colab execution work and requested polishing and cleanup of
other repository areas. This pass addresses the existing CSV-to-CHAT exporter
and workflow documentation. It does not change model inference, annotation,
feature extraction, R models or results. No pre-existing project files were
deleted, staged, committed or published.

## Changes

- `src/icl_pilot/generated_chat.py` reads CSV fields as literal strings, checks
  the entire input before writing, and requires a fresh output directory.
  Blank narratives cannot become `nan`; duplicate destinations cannot overwrite
  transcripts. Age/child/cohort conflicts with the specified roster, missing
  metadata, unsafe identifiers and incomplete prompt links fail explicitly.
- Missing age/child fields can use known frozen-roster values. Historical CSVs
  without prompt IDs remain supported; no ancestry is fabricated for them.
- New exports retain prompt/parent IDs, metadata sources and source/output
  SHA-256 hashes in `chat_manifest.csv`; prompt links also appear in CHAT comments.
- Surface normalization, punctuation and utterance segmentation are unchanged.
- `docs/post_generation_processing.md` documents the exporter and the remaining
  annotation work. README/handoff link the current procedure.
- `.gitignore` now allows five maintained workflow guides. Other documents
  retain the prior default exclusion. The guides are eligible for tracking;
  this pass does not stage or publish them.

## Verification

- The initial combined run passed 112 tests: 38 exporter cases and the 74
  existing generation/parser/retirement cases. Three existing environment
  dependency warnings remain. A final additional malformed-CSV case was added;
  all 39 exporter tests then passed. This covers 113 distinct test cases.
- `validate_exports.py` compares old/new temporary exports for every one of
  the 2,820 historical jobs. All 5,640 CHAT files match the previous converter
  after removing the added cohort comment; existing manifest values and names
  also match. Temporary exports are removed after comparison.
- `protected_before.json` records 29,602 files across data, phase2, results,
  configurations, notebooks, generation helpers and the three R sources.
  All remain byte-identical. `validation.json` contains cohort counts and hashes.
- CLI `validate-layout`, source syntax, `git diff --check`, and selective
  documentation ignore-rule checks pass. `validation_commands.json` records
  the final exporter test and whitespace-check logs.
- No actual TalkTag, CLAN or GPU inference was run.

`legacy_generated_chat.py.txt` is the pre-change comparison oracle, not active
pipeline code. `backup_verification.json` identifies the pre-change snapshot
and each verified member on TranscendBU. The post-change snapshot and copied
audit are recorded in `completed_work_backup.json`.

## Next bounded areas

1. Consolidate TalkTag execution. `notebooks/TalkTag.ipynb` has fixed Drive paths
   and an unpinned installation; the generation notebook has a separate optional
   command and defaults to annotating Stage 2 only. Review source preservation,
   coverage and existing repaired cases, then prepare a parameterized workflow.
2. Document CLAN inputs/options/versions and trace feature tables to their
   annotation inputs. Existing annotation repair tools have distinct purposes
   and can mutate text/tiers; do not apply them as a generic cleanup sequence.
3. Separately inventory analysis drafts and rendered reports before proposing
   external archival. The original/current R Markdown and rendered HTML files
   are not proven interchangeable by this pass.
4. Disk-use review found an approximately 1.4 GB `.venv` referenced by PyCharm
   project settings. It is not classified as unused and was retained. R workspace
   files and PDF-render inspection images also remain pending a separate audit.

The Colab checkpoint/restart, software/model pins and GPU trial remain deferred.
