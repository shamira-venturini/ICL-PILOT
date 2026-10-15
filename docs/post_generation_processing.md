# Post-generation processing

The generation notebook calls
`icl_pilot.generated_chat.convert_generated_story_csv_to_cha` in its conversion
block, after generation or from explicitly selected saved CSVs.
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
`nan`. Literal strings such as `NA` remain text. Formatting now follows the
Chatter-checked policy below. Generated grammar is not corrected. Utterance
boundaries still depend on the model's punctuation and quotation notation;
they have not received an independent linguistic C-unit segmentation review.

New exports use the three-field CHAT header `@Types: cross, narrative, TD`
or `SLI`. The separate `@Comment: data_source=generated` records synthetic
provenance; the group field remains the literal conditioning label. This
corrects the earlier four-field `cross, narrative, generated, <group>` header.
See the [CHAT header specification](https://talkbank.org/0info/manuals/CHAT.html).

`stage1_td/` and `stage2_sli/` contain the transcripts. `chat_manifest.csv`
links each file to its source CSV, pair, round, replicate, story, group and child.
It records the exported age, whether age/child metadata came from the generation
CSV or roster, and SHA-256 hashes of the source CSV and CHAT file. Keep the
roster snapshot and generation run manifest with this export.

### Formatting policy and validation — 15 October 2026

The converter protects existing CHAT pauses, word shortenings, square-bracket
codes, lengthening colons and interposed-speaker notation while spacing prose
punctuation. Pauses such as `(.)` no longer create sentence breaks or become
`( .)`. Explicit CHAT quotation markers and ordinary quotation marks are handled
separately; quotation linkers are paired, and standalone quoted material without
a reporting frame uses Unicode quotation marks. No reporting words are invented.

English numerals use a deterministic cardinal spelling (decimals use `point`
and individual digits; leading-zero strings use individual digits). Dollar
amounts expand to number words and `dollar(s)`. Literal story identifiers such
as `B2` become `B@l two`; ampersands used as conjunctions become `and`. These
are declared orthographic choices for generated text, not recovered pronunciation.
Known written nonverbal-event forms receive `&=` notation. Standalone prose
parentheses retain their enclosed words; embedded CHAT shortenings remain intact,
and separated omitted suffixes are reattached. Uncoded standalone `+`, `/` and
`@` are removed as prose symbols without assigning a pause, retracing or error.

Every export row now records `source_text_sha256` and a JSON `formatting_changes`
list with the original and replacement forms. A CHAT comment records the number
of logged normalizations. Original CSV text is preserved, including model
commentary; syntax validation does not establish narrative quality. The
normalization can change word counts and utterance boundaries, so adopting new
exports in an analysis requires a deliberate measurement rerun.

The full saved ages-4–7 generation CSVs were exported to a fresh audit tree.
TalkBank Chatter **0.30.0**, with `--force --strict-linkers` and no suppressed
checks, validated **all 5,640 files**. Before correction, standard validation
rejected 649 files; strict quotation-link validation rejected 658. The final
audit found no lexical sequence changes beyond the recorded orthographic
normalization. Headers, child ages/groups and story/provenance fields were
preserved. Existing research CHAT, CSVs, annotations and results were not replaced.

To check a fresh export with an installed Chatter CLI:

```bash
chatter validate PATH_TO_CHAT_EXPORT --force --strict-linkers --format json
```

Use the returned exit code and final summary; exit zero must cover every file.
This checks the files actually supplied to Chatter. The latest remote Colab
smoke export has not been inspected directly by this local audit. Reconvert a
saved CSV with `MODE="convert_only"` into a new run to use the corrected converter;
model inference and annotation are unnecessary for that step.

References: [CHAT manual](https://talkbank.org/0info/manuals/CHAT.html),
[Chatter CLI](https://talkbank.github.io/chatter/chatter/user-guide/cli-reference.html).
Before/after inventories, replacement logs, pinned binary hashes and validator
reports: `outputs/maintenance/chat_conversion_validation_2026-10-15/`.
The focused generation/conversion/annotation suite passed 229 tests, including
17 formatting regressions checked with the actual Chatter executable. Notebook
executable cells, generation templates/settings and the R analysis are unchanged.

New generation CSVs contain `stage1_prompt_record_id` and
`stage2_prompt_record_id`. Conversion requires both when either column exists,
rejects reused IDs, and retains them in CHAT comments and the manifest. The
Stage 2 entry points to the same row's Stage 1 record. Historical CSVs without
these columns remain supported; their missing record IDs are not reconstructed.
This validates the CSV links, not the contents of a separate prompt journal.

## Correcting saved generated CHAT headers

Use `repair_generated_chat_types` on the reviewed generated CHAT collection to
correct the known four-field header without rerunning generation or annotation:

```python
import json
from pathlib import Path
from icl_pilot.generated_chat import repair_generated_chat_types

run = Path("/content/ICL-PILOT/data/generated/direct_generation/"
           "enni_smoke_20261015T094156357444Z")
manifest = repair_generated_chat_types(
    run / "annotation/postprocessed", run / "annotation/formatted")
report = json.loads(manifest.read_text())
print(report["status"], report["files_checked"], report["files_changed"])
```

The hierarchy inside `formatted/` matches the source collection. Only the
`@Types` header and the separate generated-provenance comment are changed;
utterances, error tags, dependent tiers and other metadata remain byte-for-byte
intact, including line endings. Already-correct generated files are copied
unchanged. Missing/duplicate headers, unsupported formats, missing generated
provenance or a group inconsistent with the CHI `@ID` stop the entire collection
before any output is written. Real corpus files are outside this repair.
Existing output directories are refused.

`formatting_manifest.json` inventories before/after file hashes and the number
changed, identifies the source directory and records the implementation hash.
Keep the original annotation manifest and files alongside this derived copy;
their recorded hashes still describe those unchanged originals. Use the
formatted copy for subsequent format validation and CLAN processing. A
`formatting_complete` status refers only to this header correction; full CHAT
syntax and linguistic annotation accuracy have not been validated by this helper.

To include the correction in the saved pipeline record and Drive export:

```python
from icl_pilot.pipeline import record_stage, sync_run
record_stage(run, "chat_formatting", "complete", manifest=str(manifest))
if IS_COLAB:
    sync_run(run, DRIVE_EXPORT_ROOT / run.name)
```

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
That historical preservation check predates the intentional formatting fixes
and Chatter validation documented above.
