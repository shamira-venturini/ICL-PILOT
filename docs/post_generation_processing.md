# Post-generation processing

The generation notebook calls
`icl_pilot.generated_chat.convert_generated_story_csv_to_cha` after each cohort.
This exports Stage 1 TD-like and Stage 2 SLI-conditioned narratives to separate
CHAT directories. It does not annotate errors, run CLAN, or construct features.

## CSV to CHAT

From the project root, the same converter can process an existing generation
CSV without loading a model:

```python
from icl_pilot.generated_chat import convert_generated_story_csv_to_cha

artifacts = convert_generated_story_csv_to_cha(
    input_csv="data/generated/runs/MY_RUN/4_year_old_rotating_story_generations.csv",
    output_dir="data/generated/runs/MY_RUN/4_year_old_chat_export",
    roster_csv="configs/generation/master_frozen_roster_4_to_7_year_old.csv",
)
```

Run with `src` on `PYTHONPATH`, or install the accompanying project revision.
Replace `MY_RUN` with the intended run directory. The output directory must
be new, including when repeating a conversion. Existing exports are preserved.

The CSV must contain `bundle_id`, `pair_id`, `target_story`,
`stage1_td_story` and `stage2_sli_story`. Both stages require nonempty lexical
text, a source child ID and a valid CHAT age. Missing ages or child IDs are
resolved only from the specified frozen roster; recorded values must agree
with it. `roster_csv=None` permits a self-contained CSV with its own metadata.
A missing specified roster, unknown pair, metadata conflict, duplicate filename,
unsafe identifier or malformed record stops conversion before files are written.
An I/O failure after preflight can leave a partial new directory for inspection.

Empty cells remain empty during parsing; they cannot become the narrative
`nan`. Literal strings such as `NA` remain text. Existing surface normalization,
punctuation spacing, quotation handling and utterance segmentation are preserved.
Generated grammar is not corrected. These utterance boundaries are a conversion
rule for generated prose, not an independent manual segmentation assessment.

`stage1_td/` and `stage2_sli/` contain the transcripts. `chat_manifest.csv`
links each file to its source CSV, pair, round, replicate, story, group and child.
It records the exported age, whether age/child metadata came from the generation
CSV or roster, and SHA-256 hashes of the source CSV and CHAT file. Keep the
roster snapshot and generation run manifest with this export.

New generation CSVs contain `stage1_prompt_record_id` and
`stage2_prompt_record_id`. Conversion requires both when either column exists,
rejects reused IDs, and retains them in CHAT comments and the manifest. The
Stage 2 entry points to the same row's Stage 1 record. Historical CSVs without
these columns remain supported; their missing record IDs are not reconstructed.
This validates the CSV links, not the contents of a separate prompt journal.

## Annotation and CLAN

The combined notebook now has a separate TalkTag block. `full` and `from_csv`
routes annotate both converted stages by default; `from_chat` starts from saved
CHAT directories without generation. `generation_only` and `convert_only` stop
earlier. The old standalone TalkTag notebook is externally backed up and retired.
See `colab_generation_package.md` for controls, source pin and current limitations.

The reusable wrapper is `icl_pilot.annotation.annotate_chat_directory`. It operates
on a working input snapshot, records source/output hashes and runtime metadata,
and checks output coverage and preservation of source text/tiers. Wrapped CHI
main tiers are joined only in the snapshot. It retains failures for review and
does not certify linguistic tag accuracy. Real and synthetic comparisons require
comparable documented annotation and measurement procedures; the new interface
does not establish that equivalence.

CLAN commands, tool versions, input tier expectations and feature lineage still
need a later dedicated guide. CLAN measurements and scientific evaluation are
not run in the generation/annotation notebook.

Existing repair utilities have different purposes and can rewrite text or
dependent tiers in place. They are not a universal automatic preprocessing
sequence. Their use and existing repaired cases need separate review before
they are incorporated into a reusable annotation workflow.

## Verification

The 14 October 2026 audit compares temporary old/new exports for all 2,820
historical jobs (5,640 CHAT files, ages four through seven). After removing the
new cohort provenance comment, CHAT contents and existing manifest fields
match the previous converter. Existing research transcripts and analyses are
not regenerated. Audit: `outputs/maintenance/post_generation_cleanup_2026-10-14/`.
