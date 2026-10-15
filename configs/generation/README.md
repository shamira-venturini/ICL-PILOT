# Generation configuration

Start with `notebooks/colab_direct_generation_pipeline.ipynb`. It provides the
settings, input checks, CUDA model loading, two-stage generation, CHAT conversion,
TalkTag error annotation and exports in separate notebook blocks. Use a checkout containing this configuration
and make the required ENNI inputs available before running it.

## Authoritative inputs

| File | Role |
| --- | --- |
| `experiment.json` | Shared model/decoding settings, replicates, round size, input locations and ages four–seven. |
| `cross_story_counterbalancing.csv` | Opposite-set TD/SLI exemplar choices for each target story. |
| `story_unit_reference.json` | Fixed project-authored scaffolds for A1–B3: ordered events, content invariants, optional details and prompt constraints. |
| `prompts/stage1.json` | Frozen system/user messages for TD-like generation, with readable template lines and named inputs. |
| `prompts/stage2.json` | Frozen system/user messages for the SLI-conditioned transformation of the actual Stage 1 output. |
| `master_frozen_roster_4_to_7_year_old.csv` | All 47 recorded prompt pairs: 11 at age four, 12 at five, 11 at six and 13 at seven. Includes the reconciled four-year-old 474/459 and 475/417 pairs. |

The author confirmed on 14 October 2026 that age four used the same generation
configuration as ages five–seven. `experiment.json` records that confirmation;
historical model revisions were not present in the saved CSVs. A new run records
the revision actually resolved by model loading. The original narrative text is
not regenerated or changed by this configuration cleanup.

Each master row fixes one SLI-labelled child and one TD child. Pair IDs and order
come from the original generation records, not a new matching optimization.
`Age` values and KIDEVAL identifiers are joined by child ID from developmental
metadata. Approximate-month fields retain that metadata's existing precision;
raw CHAT age strings remain available. Severity/profile fields are absent.
Changing a roster means creating a new experimental design, not rerunning this
one. Pairing builders refuse to overwrite an existing frozen roster.
Setup reads the master once and selects rows by `cohort`. `pair_order` restarts
at one within each cohort and determines that cohort's round assignments.

## Inputs and run outputs

Configuration assets are supplied in this folder. The roster, counterbalancing
and `story_reference_json` paths in `experiment.json` are relative to its own
directory, so the scaffolds travel with the generation configuration.
The scaffolds are project-authored outlines derived from ENNI pictures;
they are not official adult verbal retells. Changes to their content change
the generation protocol. Their hash is recorded in every run manifest.

The external corpus input is relative to the notebook's `DATA_ROOT`:

- `data/processed/generation_inputs/story_unit_narratives_chat_v2.csv`: exemplar text
  keyed by child ID, literal corpus group, and story slot.

This version fixes CHAT age parsing and includes a `*.parsing_qc.json` sidecar
with source hashes and parsing issues. Story text and utterance arrays match
the historical table exactly. The original table under
`phase2/ENNI/measures/story_units/` remains available to existing analyses.
See `docs/chat_narrative_parsing.md` for the reproducible extraction command.

The setup validates every exemplar and checks child-level prompt/evaluation
separation within each age-specific round before loading the model. Generation
does not require the archived profiling tables or access to analysis results.
Selected exemplar raw ages must also match their frozen roster entries.

`experiment.json` schema 4 names both prompt templates. Their wording,
whitespace and replicate-ID content were preserved from the original notebook.
The shared builder rejects missing inputs, invalid story-slot combinations,
and Stage 2 calls without a nonempty Stage 1 parent. Age matching is supplied
by the selected exemplars; no new age instruction was added to the messages.

Per-cohort roster snapshots, schedules, round assignments, bundle indexes, templates and summaries are
derived into a new run's `generation_design/` directory. They are not additional
editable sources of truth in this folder. The default design has 47 pairs,
ten replicates and six stories: **2,820 target-story jobs, with two outputs per
job**. The notebook's `REPLICATES` override can change this count for a new run:
each replicate contributes 282 narratives per stage. The effective settings are
hashed and recorded without editing this study configuration. Round size changes
the prompt/evaluation design and is not a generation-volume setting.
Every run gets input hashes and a settings snapshot in `run_manifest.json`.
The notebook now starts with `SMOKE_TEST=True` for a one-target-story pipeline
check (two stage outputs). Set it to false for the full configured schedule.
This execution limit does not alter the frozen study configuration. The selected
job list and `test_only` labels are recorded separately from the complete design.
Setup and generation reject reuse that would overwrite existing narratives.

During inference, `prompt_records.jsonl` records each attempted prompt before
model generation and its output afterward. It includes system/user messages,
the actual chat-template string passed to the tokenizer, input-token count/hash,
generation parameters, exemplar identities/ages and job metadata. Each Stage 2
record links to a completed Stage 1 record from the same job and checks its
parent-text hash. Output CSVs include both prompt-record IDs.
Template snapshots and hashes are retained with each run, and the prompt
journal is included in Drive exports and the final archive. This journal records
progress but does not yet implement restart from existing outputs.

The `full` route continues through conversion and error annotation after releasing
the generation model. `from_csv` and `from_chat` routes use saved inputs without
loading that model or the exemplar corpus; `generation_only` and `convert_only`
stop after the selected step. TalkTag options live separately in
`configs/annotation/talktag.json`. Python notebook and R Markdown evaluations
remain separate. See `docs/colab_generation_package.md`.

The split manifest produced during setup covers the matched roster children.
The current R analysis independently uses its ENNI reference pool and excludes
actual prompt children from synthetic metadata. Those are different reference
pool sizes; both must maintain the same within-round exclusion rule.

## Archived configurations

The four separate cohort inputs and the obsolete combined roster were backed up
and removed after consolidation. The current master preserves all rows from
the reconciled cohort inputs, including the two corrected four-year-old pairs.
R files now reference its new location; their analysis logic is unchanged.
The R split setup derives exclusions from actual synthetic ancestry; this was
checked across all 17 age-specific rounds.

Verified pre-consolidation backup:
`/Volumes/TranscendBU/ICL-PILOT_78a5cadad659_20261014T123617Z/GenerationRoster/`.
The consolidation audit is under
`outputs/maintenance/master_roster_consolidation_2026-10-14/`.

Older report/pilot/LOOCV/profile-grid configurations and saved one-replicate
packages are preserved in the verified external backup:

`/Volumes/TranscendBU/ICL-PILOT_78a5cadad659_20261014T115428Z/Generation/`.

The local change audit is under `outputs/maintenance/generation_cleanup_2026-10-14/`.
No raw corpus data, generated narratives, annotations, analysis inputs, or
scientific results are removed by this cleanup.
