# Generation configuration

**Status:** on 20 October 2026, story counterbalancing was revised and the
generator was corrected to an official Meta checkpoint. On 21 October, the
author adopted a 640-new-token ceiling, temperature 0.8 and top-p 0.9 for both stages,
then removed age/status and directional length cues and adopted the supplied
matched generation/rewriting prompt pair.
The working configuration is `enni_staged_4_to_7_counterbalance_v2`; the remaining
scientific choices are pending. The previous study and pre-revision configuration
are frozen on TranscendBU. Child pairs, canonical story content and exemplar text
remain the earlier baseline.

Start with `notebooks/colab_direct_generation_pipeline.ipynb`. It provides the
settings, input checks, CUDA model loading, two-stage generation, CHAT conversion,
TalkTag error annotation and exports in separate notebook blocks. Use a checkout
containing this configuration and its frozen exemplar inputs.

## Authoritative inputs

| File | Role |
| --- | --- |
| `experiment.json` | Shared model/decoding settings, replicates, round size, input locations and ages four–seven. |
| `cross_story_counterbalancing.csv` | Opposite-set TD/SLI exemplar choices for each target story. |
| `story_unit_reference.json` | Fixed project-authored scaffolds for A1–B3: ordered events, content invariants, optional details and prompt constraints. |
| `prompts/stage1.json` | Versioned system/user messages for generation with the selected TD exemplar, without age/group instructions. |
| `prompts/stage2.json` | Versioned system/user messages for transformation with the selected SLI-labelled exemplar and actual Stage 1 output, without age/group instructions. |
| `master_frozen_roster_4_to_7_year_old.csv` | All 47 recorded prompt pairs: 11 at age four, 12 at five, 11 at six and 13 at seven. Includes the reconciled four-year-old 474/459 and 475/417 pairs. |
| `inputs/story_unit_narratives_chat_v2.csv` | Frozen real ENNI story text and metadata used to select in-context exemplars. |
| `inputs/story_unit_narratives_chat_v2.parsing_qc.json` | Source hashes, parser settings and extraction issues for the frozen CSV; retained as provenance. |

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

## Official generator — corrected 20 October 2026

`experiment.json` now selects `meta-llama/Llama-3.1-8B-Instruct`, pinned to revision
`0e9e39f249a16976918f6564b8830bc894c89659` for both model and tokenizer. The previous
community checkpoint retains its original attribution in the external freeze;
this correction changes the model used by new runs. Approved access to Meta's
Hugging Face repository and runtime authentication are required. The existing
notebook login block uses `HF_TOKEN` from the environment or a saved login.

See [model selection](../../docs/guides/model_selection.md) for the source audit,
single-GPU constraint, choice rationale and proposed comparison conditions.
Qwen3 and vision conditions are not yet configured. GPU loading/inference with
the official checkpoint still needs a fresh Colab check.

## Sampling — adopted 21 October 2026

New runs use **temperature 0.8 and top-p 0.9**, shared across stages and age
cohorts. The author selected these after checking previous research. The earlier
values (0.7 and 0.95) and the decision's attribution are retained in
`experiment.json` provenance. Setup reads and prints the shared configuration;
each inference prompt record retains the values actually passed to the model.
Changing either value requires a fresh run directory.

This change does not resolve the inherited top-k setting. Under the notebook's
declared Transformers 4.x runtime, the library default is 50; an explicit
top-k policy remains pending. No top-k override was introduced here.

## Output ceiling — adopted 21 October 2026

`max_new_tokens_stage1` and `max_new_tokens_stage2` are both **640**, for every
cohort. This upper bound excludes input tokens and permits earlier
end-of-sequence stopping. It does not impose a minimum or ask for a 640-token
story. Setup prints both effective values; the notebook helper requires an
explicit value supplied by the shared runner, without a separate default.

The [real-story audit](../../docs/provenance/real_story_tokenizers_2026-10-21/README.md)
checked all 1,428 real stories at ages four through seven. The longest used 568
tokens with saved CHAT notation, or 556 after light cleanup, under the pinned
official Llama tokenizer, historical reference and both proposed Qwen tokenizers.
The ceiling leaves 72 tokens above the stored-text maximum. This is a practical
buffer for the study's age range, not a validated maximum for child language or
a guarantee that generated outputs will never reach it. The nine-year-old
maximum in the full snapshot is outside the active age range.

Run manifests snapshot and hash the effective settings. The prompt journal
records each call's limit and actual generated-token count for review of ceiling
hits; explicit stopping-token/finish-reason logging remains pending. A ceiling
hit does not trigger automatic exclusion, resampling or text repair. Changed
limits require a fresh run directory. The previous limits (180 per stage) and
the audit reference are recorded in `experiment.json` provenance.

## Story counterbalancing — adopted 20 October 2026

`cross_story_counterbalancing.csv` defines two assignments per target. Within
each target, CSV row order identifies assignment 1 and assignment 2. E1 is the
TD exemplar used for Stage 1; E2 is the SLI-labelled exemplar used for Stage 2.

| Target | Assignment 1: E1 / E2 | Assignment 2: E1 / E2 |
| --- | --- | --- |
| A1 | B1 / B2 | B2 / B1 |
| A2 | B1 / B3 | B3 / B1 |
| A3 | B2 / B3 | B3 / B2 |
| B1 | A1 / A2 | A2 / A1 |
| B2 | A1 / A3 | A3 / A1 |
| B3 | A2 / A3 | A3 / A2 |

For each prompt pair and target, the default ten replicates allocate five to
each assignment. The existing scheduler alternates assignments with an offset
by `pair_order`; five consecutive replicates per assignment are not required.
Exemplars are always distinct stories from the opposite set. Within each
target, each selected story appears equally often in E1 and E2. Across all six
targets, each story appears ten times per position per pair, and each complexity
level (1–3) appears twenty times per position per pair.

The five/five guarantee applies to the full ten-replicate schedule. An odd
replicate override or a truncated smoke test need not have exact balance.
This revision balances exemplar assignment. The generation-versus-rewriting
asymmetry remains an open design issue. The previous schedule's hash and the
adoption date are recorded in `experiment.json`; original configurations remain
recoverable from the study freeze. New runs record the revised CSV hash and
schedule automatically and require a fresh output directory.

## Inputs and run outputs

Configuration assets are supplied in this folder. The roster, counterbalancing
and `story_reference_json` paths in `experiment.json` are relative to its own
directory, so the scaffolds travel with the generation configuration.
The scaffolds are project-authored outlines derived from ENNI pictures;
they are not official adult verbal retells. Changes to their content change
the generation protocol. Their hash is recorded in every run manifest.

The frozen corpus input is supplied alongside the configuration. Its path in
`experiment.json` is relative to the notebook's `DATA_ROOT`:

- `configs/generation/inputs/story_unit_narratives_chat_v2.csv`: exemplar text
  keyed by child ID, literal corpus group, and story slot.

`DATA_ROOT` defaults to the project checkout in both Colab and local Jupyter.
No separate CSV upload is needed for an accompanying checkout containing these
files. An explicit `ICL_PILOT_DATA_ROOT` override selects a different input tree
with this same relative path; optional upload staging still verifies hashes and
refuses conflicting inputs. Drive remains the default Colab output destination.

This version fixes CHAT age parsing and includes a `*.parsing_qc.json` sidecar
with source hashes and parsing issues. Story text and utterance arrays match
the historical table exactly. The original table under
`phase2/ENNI/measures/story_units/` is preserved in the external study freeze.
The two files were moved byte-for-byte from `data/processed/generation_inputs/`
on 16 October 2026. Embedded source paths and extraction records remain historical
provenance. See `inputs/README.md` and `docs/guides/chat_narrative_parsing.md` for details
and the extraction command. Generation reads the CSV, not the source CHAT files
or QC sidecar.

The setup validates every exemplar and checks child-level prompt/evaluation
separation within each age-specific round before loading the model. Generation
does not require the archived profiling tables or access to analysis results.
Selected exemplar raw ages must also match their frozen roster entries.

`experiment.json` schema 4 names both prompt templates. On 21 October 2026, the
author adopted the supplied reviewed generation/rewriting pair after removing
age/status and directional length cues. Both stages now share the same system
message, four story-information fields (events, required content, optional
content, additional constraints) and full instruction list. Exemplar guidance
covers wording, sentence construction, narrative style and level of detail in
both stages. Stage 2 additionally receives the actual Stage 1 parent to rewrite
with its own selected exemplar. All six additional-constraint lists specify only
event order and no new plot events; do not add linguistic or length instructions.

Canonical story content, literal exemplar text and the Stage 1 parent are
preserved, including any children or adults mentioned as characters. Templates
contain no ENNI/task labels, story/exemplar IDs, story titles, replicate IDs or
age/status instructions. IDs and literal corpus labels remain in logs and
validation. The 640-new-token operational ceiling is retained. This reviewed
pair is distinct from the complete 15 October neutral variant.

Age matching and corpus group labels remain in selection and provenance; these
metadata are not inserted into model messages. The shared builder rejects
missing inputs, metadata placeholders, invalid story-slot combinations and Stage 2
calls without a nonempty Stage 1 parent. Template and scaffold changes require a fresh run
directory. After updating an existing notebook checkout, restart the kernel and
run setup again to load the revised templates. Previous hashes are recorded in
`experiment.json` provenance; the study freeze preserves the earlier wording.

Per-cohort roster snapshots, schedules, round assignments, bundle indexes, templates and summaries are
derived into a new run's `generation_design/` directory. They are not additional
editable sources of truth in this folder. The default design has 47 pairs,
ten replicates and six stories: **2,820 target-story jobs, with two outputs per
job**. The notebook's `REPLICATES` override can change this count for a new run:
each replicate contributes 282 narratives per stage. The effective settings are
hashed and recorded without editing this study configuration. Round size changes
the prompt/evaluation design and is not a generation-volume setting.
Every run gets input hashes and a settings snapshot in `run_manifest.json`.
The notebook starts with `SMOKE_TEST=True` and a configurable `SMOKE_TEST_JOBS`
limit (currently 12 jobs, or 24 narratives; use 1 for the one-story check). This
uses the first jobs in the configured schedule, preserving seed and prompt templates.
Set `SMOKE_TEST=False` for the full configured schedule.
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
remain separate. See `docs/guides/colab_generation_package.md`.

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
`docs/provenance/master_roster_consolidation_2026-10-14/`.

Older report/pilot/LOOCV/profile-grid configurations and saved one-replicate
packages are preserved in the verified external backup:

`/Volumes/TranscendBU/ICL-PILOT_78a5cadad659_20261014T115428Z/Generation/`.

Selected local audit reports are under `docs/provenance/generation_cleanup_2026-10-14/`;
the complete historical audit is externally archived as described in
`docs/provenance/README.md`.
No raw corpus data, generated narratives, annotations, analysis inputs, or
scientific results are removed by this cleanup.
