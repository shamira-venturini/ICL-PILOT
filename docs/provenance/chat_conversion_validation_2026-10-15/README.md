# CSV-to-CHAT conversion audit — 15 October 2026

On 16 October, the validated exports were moved to their maintained home at
`data/generated/<age>_year_old/pre_annotation/`. All 5,640 transcript hashes
were checked against the original audit copies and the verified external
cleaned-data snapshot before the duplicate `final/exports/` tree was removed.
The original audit CSV manifests are saved at
`/Volumes/TranscendBU/ICL-PILOT_af80b106b6dd_20261016T085254Z/Data/pre_annotation_audit_manifests/`.
Reports below retain their original paths as historical evidence. See
`data/README.md` and `data/qc/` for the current retained dataset.

Scope: the maintained converter and all four original ages-4–7 paired-stage
generation CSVs. This is a formatting/lexical-preservation audit, not a new
generation, annotation, feature-extraction, statistical or manuscript task.
The author's newest Colab smoke run was not accessed directly.

## Result

- 2,820 source rows produced 5,640 fresh, pre-annotation CHAT files.
- Before the fixes, Chatter rejected 649 files under standard checks and 658
  under strict cross-utterance quotation-link validation.
- All 5,640 final files pass Chatter 0.30.0 with `--force --strict-linkers`.
  No checks were suppressed and no internal validator failures occurred.
- All normalized lexical sequences agree with the exported main tiers.
- 392 files have recorded changes in lexical encoding (principally numeral
  and currency spelling); 729 have logged nontrivial formatting operations.
- Headers, age/group metadata, story identifiers, provenance and source CSV
  hashes are preserved. No existing research transcript or result was replaced.
- 229 focused tests passed. Seventeen source-format regression examples also
  passed the real Chatter executable with strict quotation checks enabled.
- The notebook's executable cells, restored generation prompts/settings and
  R source remain unchanged. No GPU or model inference was used.

## Evidence and reproduction

`validator_source.json` records the official release URL, archive/binary
SHA-256, platform and executable version. The download digest was verified.
Chatter was extracted into this audit's tools directory, not installed globally.

`before/` records the original exporter; `final/` records the corrected one.
Each has the converter SHA-256, source hashes, per-file inventory, preservation
report and complete machine-readable validator output. Intermediate named
phases retain their own reports and code hashes; they are diagnostic steps.
Archives of intermediate export trees preserve relative paths and file bytes.
`archived_exports.json` records the verified archive inventories when present.
The uncompressed corrected files are under `final/exports/`.

`before_after_inventory.csv` compares every initial/final file, including
hashes and utterance counts. Formatting/quotation fixes alter boundaries in
42 files, so the new exports must not silently replace historical analysis
inputs. No linguistic measurements were recomputed.

The main audit can be repeated into an unused phase name:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python \
  outputs/maintenance/chat_conversion_validation_2026-10-15/audit_conversion.py NEW_PHASE
```

The source CSVs, frozen roster and validated Chatter executable must be present.
The phase directory must not exist. The script exits unsuccessfully for a
validator failure or an unexplained lexical sequence change. Numeric spelling
and other declared normalizations are checked against their recorded operations;
they are not misreported as literal preservation of the original numeric tokens.

Policy and primary references: `docs/post_generation_processing.md`.
Chatter checks syntax and linkage; an independent linguistic assessment of
C-unit segmentation and narrative/annotation validity remains outside this audit.
