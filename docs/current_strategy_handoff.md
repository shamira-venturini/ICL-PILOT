# Project Status and Study Handoff

**Current status — 21 October 2026:** the author authorized a complete external
freeze of the previous study and a fresh working repository for a revised
experiment. The revised scientific protocol is pending. See
[`study_freeze_2026-10-16.md`](provenance/study_freeze_2026-10-16.md) for the
verified snapshot and restoration index, and
[`revised_study_setup.md`](guides/revised_study_setup.md) for pending decisions.

The following dated records preserve earlier decisions, findings and cleanup
history. Their old data paths may now exist only in the archive. The author
adopted revised story counterbalancing and requested an official Llama checkpoint
on 20 October 2026, then adopted a 640-new-token ceiling for both stages on
21 October. The author then selected temperature 0.8 and top-p 0.9, shared by
both stages and all cohorts. The author subsequently removed age and clinical-status
cues and directional story-length instructions, then adopted the supplied
matched generation/rewriting prompt pair. Child pairs remain the earlier baseline. The retained
R and framework implementations require explicit revised inputs and further
protocol review; no revised scientific results have been produced.

### Matched generation/rewriting prompts — 21 October 2026

The author supplied a reviewed pair and requested its implementation. Both
templates now match that pair exactly. They share the system message, ordered
events, required content, optional content, additional content constraints and
the full instruction list. Both use the exemplar to guide wording, sentence
construction, narrative style and level of detail. This shared level-of-detail
guidance supersedes its earlier removal; neither stage asks for a shorter story.
Stage 1 generates a narrative. Stage 2 rewrites the exact Stage 1 narrative with
its own selected exemplar. The generation-versus-rewriting distinction remains.

ENNI/task labels, story/exemplar IDs, story titles and replicate IDs are removed
from the model-facing templates. IDs and literal corpus labels remain in
validation, schedules and prompt records; age-matched exemplar selection is
unchanged. The builder permits the same four story-information fields in both
stages and rejects metadata placeholders in messages. All six packaged
additional-constraint lists contain only event-order/no-new-event rules.
Canonical story content, literal exemplar/parent text, child pairs, assignment
rules, the official model, sampling and the 640-token ceilings are unchanged.
`experiment.json` records the adoption and immediately preceding template hashes.
New prompts require a fresh run after reloading setup; no revised outputs have
been generated.

Validation: 91 focused configuration/prompt tests passed with the three existing
dependency warnings. Independent transcription checks confirm exact equality to
the supplied JSON pair. All 5,640 stage-message lists for the 2,820 scheduled jobs
rendered with identical system messages, story-information fields and instruction
lists across stages. Metadata/age/status/directional-length cue checks passed
outside preserved literal content. Schedule/evaluation fingerprints, scaffold and
exemplar bytes, model/decoding settings, execution/journal code and all 16 notebook
code cells are unchanged. The builder's permitted fields were updated for the
reviewed pair. Python 3.8/notebook syntax, layout, 36 local links and whitespace
checks passed. Temporary checks were removed; no GPU inference, annotation or
scientific analysis was run.

### Story-length instructions removed — 21 October 2026

The author subsequently requested removal of story-length references from both
prompts. Stage 1 no longer requests short narratives. Stage 2 no longer asks
the model to match the exemplar's amount of detail or permits clause shortening
and omission of optional details. Exemplar-based style/grammatical-level guidance,
event-order/content constraints and the removal of age/status cues are retained.
The 640-new-token operational ceiling remains unchanged; no verbal word/token
target is supplied to the model. Scaffolds, literal exemplars, Stage 1 parent
text, model/decoding settings, child pairs and scheduling are preserved.
`experiment.json` records the decision and immediately preceding template hashes.
Use a fresh run after reloading notebook setup.

Validation: all 79 focused configuration/prompt tests passed with the three
existing dependency warnings. All 5,640 scheduled stage-message lists rendered
without age/status or length cues in the instruction shell; literal content was
preserved. Notebook syntax, local documentation links and whitespace checks
passed. Temporary test/preflight outputs were removed. No inference was run.

### Prompt instruction revision — 21 October 2026

The author requested removal of all references to narrator age and clinical or
non-clinical status from the prompts. Both templates now omit TD labels,
child-like/adult-style narration instructions and clinical-plausibility framing.
The inserted constraints for all six scaffolds also omit child-like/simple
narration and adult-style avoidance. ENNI/story identifiers, event-order and
content constraints, exemplar-based style/grammatical-level guidance and the
two-stage generation/transformation flow are retained. Other prompt decisions
remain pending; this is not the complete neutral-wording variant tested on
15 October.

Age matching and literal TD/SLI corpus labels remain in exemplar selection and
run provenance, outside the model-facing instructions. Story content, including
children/adults as characters, human exemplar text and the exact Stage 1 parent
are preserved. Model/decoding settings, child pairs, counterbalancing and
prompt/evaluation separation are unchanged. `experiment.json` records the
revision date and previous template/scaffold hashes; the prior wording remains
in the verified study freeze. New prompts require a fresh run and reloading
setup in an existing notebook session. No revised outputs have been generated.

Validation: 77 focused configuration/prompt tests passed, with the three existing
dependency warnings. All 5,640 stage-message lists for the 2,820 scheduled jobs
rendered without age/status cues in the instruction shell; literal story and
exemplar content was excluded from that cue check. Schedule/evaluation
fingerprints, selection inputs, generation code, all 16 notebook code cells and
model/decoding settings are unchanged. Temporary preflight outputs were removed.
Notebook syntax, maintained local links and layout checks passed. No GPU
inference, annotation or scientific analysis was run.

### Sampling parameter update — 21 October 2026

The author instructed that new generation use **temperature 0.8 and top-p 0.9**,
after checking previous research. The previous values were 0.7 and 0.95. A
specific research citation was not supplied in that instruction; the provenance
records the author's report rather than inventing a reference or asserting
task-specific validation. Both stages and all age cohorts use the same values
from `configs/generation/experiment.json`. Setup prints them before model
loading; both model calls and the prompt journal receive those settings.

The 640-new-token ceilings, seed, checkpoint, quantization, prompts, rosters,
story scaffolds and counterbalancing are unchanged by this parameter update.
New settings require a fresh run directory. The
[update audit](provenance/generation_sampling_update_2026-10-21.json) records
before/after settings, preservation and validation.

Top-k is still inherited, not explicitly selected in the configuration or
notebook call. The declared Transformers 4.x runtime defaults to 50, so its
effective filtering must be considered when describing the decoding method.
An explicit top-k policy remains pending; no override was added in this update.
See the [Transformers generation configuration](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/text_generation).

### Real-story tokenizer comparison — 21 October 2026

The author requested a real-story basis for the output ceiling and a comparison
across tokenizers. After the official Llama tokenizer check, the author adopted
**640 new tokens for both stages and all age cohorts**. The previous limits were
180. `configs/generation/experiment.json` is the executable source of truth;
its provenance records the adoption and real-story basis. The notebook helper
requires the caller's configured limit, and setup prints both stage limits.
The [comparison report](provenance/real_story_tokenizers_2026-10-21/README.md)
records pinned tokenizers, checked source hashes and all per-story counts.
The author preferred the existing Colab login over local authentication, then
asked for another route. A public tokenizer-only copy was subsequently verified
against the pinned official repository's Git blob hash and size. Those exact
bytes enabled local execution; the earlier Colab request is superseded, and the
active model remains the official Meta one.

The official Llama 3.1, historical reference, Qwen3-8B and Qwen3-VL-8B-Instruct
text tokenizers all place the maximum among 1,428 real stories at ages four
through seven at 568
tokens in saved CHAT surface form, or 556 after the same light cleanup used in
the initial length estimate. Across all 2,166 saved records, the maxima are 643
and 604 respectively, both from a nine-year-old. Thus 640 clears all study-age
stories in the completed checks; it does not clear that older stored CHAT text.
Equal maxima do not imply identical tokenization: Qwen differs on some stories.
The official and historical count fingerprints match on all stories in both
representations. These counts exclude prompts and special tokens, and no claim
about natural termination follows from them. Qwen models remain proposed
conditions. The tokenizer comparison itself downloaded no weights, edited no
source data and changed no generation settings. The subsequent ceiling update
changes only these two decoding limits; prompt text, other sampling settings,
child pairs and story assignments are preserved.

The shared runner passes the configured values to each model call and the
prompt journal records those limits and actual generated-token counts. New runs
must use fresh directories; a limit change cannot reuse an earlier manifest.
Explicit stopping-token/finish-reason logging and GPU verification remain
pending. The ceiling does not imply that future outputs cannot hit it, nor does
it introduce automatic retries, exclusions or a requested narrative length.
The [update audit](provenance/generation_ceiling_update_2026-10-21.json) records
configuration hashes, preservation checks and focused validation.

### Historical generation endings audit — 21 October 2026

At the author's request, all 5,640 archived raw stage outputs (2,820 paired jobs,
ages four through seven) were screened before choosing a revised token ceiling.
The [audit report](provenance/generation_termination_audit_2026-10-21/README.md)
identifies the verified earlier data archive, exact source members and outputs.
Each CSV hash matches its archived manifest; all text hashes and job metadata
match the 14 October generation baseline. Historical source files are unchanged.

Re-encoding saved decoded text with a pinned tokenizer from the previously
configured community model gives 57 texts exactly at the 180-token reference:
10 TD-like and 47 SLI-conditioned, out of 2,820 each. Seven more are at 178–179;
none exceed 180. Nine of the ten Stage 1 boundary cases are B3. All 64 boundary
tails were inspected, revealing unfinished-looking phrases, complete-looking
clauses, long repeated-word runs and response-format artifacts.

Original token IDs, stopping reasons and per-run limits were not saved in these
CSVs; the historical installed tokenizer revision is unknown. These are proxy
counts, not a confirmed truncation rate. The 180 reference comes from preserved
notebook source. Punctuation alone cannot classify termination. Replicated outputs
were not treated as independent children and no inferential analysis was run.

Eleven focused audit tests passed; no generation, annotation or existing
linguistic analysis was rerun. The generation ceiling and prompt wording remain
unchanged. A balanced pilot with the official model should establish a generous
ceiling and record EOS versus budget termination before a new numeric limit is
adopted. Length instructions in the prompts remain a separate pending decision.

### Model source and comparison scope — 20 October 2026

The author requested correction to an official Llama model and a justified model
comparison, and confirmed that it must run on a single Colab GPU.
`configs/generation/experiment.json` now selects Meta's official
`meta-llama/Llama-3.1-8B-Instruct`, with revision
`0e9e39f249a16976918f6564b8830bc894c89659` pinned for model and tokenizer. The
previous `avans06/Meta-Llama-3.2-8B-Instruct` publisher describes a community
conversion of Llama 3.2 11B Vision with vision components removed. This is a
generator change. Historical outputs retain their original model attribution;
the historical installed revision remains unknown. The prior configuration is
in the verified freeze and the counterbalancing revision audit.

The [model-selection note](guides/model_selection.md) distinguishes the implemented
source correction from proposed comparisons: Qwen3-8B in non-thinking mode as a
similarly sized text comparator, and a Qwen3-VL-8B-Instruct pilot with a matched
text-input control. The comparator, vision contrast, adapters and full comparison
protocol remain undecided/unimplemented. Reasons are source transparency,
reproducibility, cross-family comparison and single-GPU feasibility; developmental
and clinical fidelity still require evaluation. No claim of superior DLD modelling
follows from general benchmarks.

The existing notebook authentication block supports the official repository's
gated access. A fresh Colab loading/inference check remains pending. No weights
were downloaded or narratives generated locally. Prompt wording, decoding
settings, child roster, scaffolds and revised counterbalancing are unchanged by
this model-source correction. See the model-selection note for primary sources.

All 66 focused configuration/prompt tests passed with the existing three
dependency warnings. Both notebooks' code cells parse after IPython syntax
transformation, and outputs remain empty. The
[model-source audit](provenance/model_source_correction_2026-10-20.json) records
the old/new config hashes, public repository metadata, pin and validation scope.

### Revised story counterbalancing — adopted 20 October 2026

After verifying both the full study freeze and compact retained-source checkpoint,
the author authorized the proposed symmetric schedule. The working configuration
is `enni_staged_4_to_7_counterbalance_v2`. The executable source of truth is
[`cross_story_counterbalancing.csv`](../configs/generation/cross_story_counterbalancing.csv);
the [generation README](../configs/generation/README.md) lists all assignments.
A1 assignment 1 changes from B1/B3 to B1/B2; B3 assignments change from A1/A2 and
A2/A1 to A2/A3 and A3/A2. The remaining rows are unchanged.

Each target has two assignments that reverse E1 (TD exemplar) and E2
(SLI-labelled exemplar). With the ten-replicate default, each prompt pair and
target uses each assignment five times through the existing alternating
scheduler. Exemplars remain distinct and from the opposite story set. Across
all six targets for each pair, each story appears ten times in each position,
and each complexity level appears twenty times in each position. Truncated
smoke tests or odd replicate overrides need not have exact balance.

`experiment.json` records revision 2, the approval date and previous CSV hash.
The full and compact archives preserve the old schedule. Prompts, roster,
scaffolds, model/decoding settings, seed, round size and generation code are
unchanged by this revision. Other protocol decisions remain pending, including
the generation-versus-rewriting asymmetry. No narratives or results were rerun.

Validation passed for all 47 prompt pairs and 2,820 scheduled story jobs: 5/5
allocation for every pair/target, story and complexity balance in each exemplar
position, full exemplar/age preflight and matched-roster within-round child
separation. All 5,640 stage-message lists rendered without inference. The 73
focused generation/configuration/prompt/roster tests passed, including regression
checks for balance and refusal to reuse a run after counterbalancing changes.
The [revision audit](provenance/counterbalancing_update_2026-10-20.json) records
the exact row changes, archive reference, hashes and validation scope.

### Study freeze and repository restructuring — completed 20 October 2026

The full pre-restructuring working tree was archived and verified on 16 October,
before historical local data and results were retired. Its archive SHA-256 was
checked again after reconnecting TranscendBU on 20 October and is unchanged.
The freeze index identifies both that snapshot and the compact checkpoint of
the retained repository. Restore historical work into a separate workspace.

The maintained interfaces are now together under `notebooks/`: generation/CHAT/
TalkTag, framework evaluation, and `clinical_analyses_reordered.Rmd`. The five
R inputs in `configs/analysis/inputs.R` and two framework inputs in
`configs/evaluation/inputs.json` are pending. Both evaluation interfaces stop
before data processing while their inputs are absent. CSV exporters also
require explicit paths. These changes remove automatic selection of retired
Phase 2 inputs; they do not settle the revised analysis protocol.

Preservation checks against the external baseline confirmed:

- all ten existing non-README configuration payloads are byte-identical,
  including prompts, roster, counterbalancing, scaffolds and exemplar inputs;
- all 16 generation-notebook code cells are unchanged; only the framework
  notebook's input/settings cell changed;
- all 2,820 scheduled jobs, the evaluation manifest and 5,640 rendered
  stage-message lists have identical fingerprints, using a fixed Stage 1 parent
  for Stage 2 verification, without inference;
- 346 existing Python function/class definitions are unchanged; the remaining
  changes require explicit CSV paths or remove the unused default-roster wrapper;
  a new framework input resolver was added;
- all 526 R expressions outside the input-setup chunk are preserved apart from
  three output-path expressions. Root discovery supports knitting from
  `notebooks/`, and figures use dated `results/linguistic_evaluation/` folders.

The 312 Python tests and base-R input/root-discovery tests passed on 19 October.
Python 3.8 syntax, R Markdown syntax, layout and maintained local documentation
links also passed. Historical notebook outputs were preserved externally and
cleared locally. No GPU generation, TalkTag inference or scientific analyses
were rerun, and no staging, commit or push was performed.

At completion of the freeze, the next step was to settle the scientific decisions in
[`revised_study_setup.md`](guides/revised_study_setup.md), then validate the new
inputs and processing lineage. The earlier prompts, model settings and child
pairs remained a software baseline; the counterbalancing revision was adopted
subsequently, as recorded above. The historical sections below retain their
original context and may refer to paths that now exist only in external archives.

This file records the current agreed generation, annotation and evaluation decisions.
On 14 October 2026 the user retired the exploratory severity/profile analysis
from the local repository after verified external backup. Profiling is not
part of the current paper's generation or RStudio analysis.

## 1. Generation Unit

- Generation stays at the **story level**.
- We are **not** moving to synthetic child-level bundles for the main augmentation pipeline.
- One generated sample corresponds to one ENNI target story.

### Generation configuration cleanup scope — 14 October 2026

- The user confirmed that the four-year-old cohort used the same generation
  configuration as ages five through seven. The portable setup applies one
  shared protocol to **ages four through seven** and retains age four.
  This confirmation is recorded as author-provided provenance; original CSVs
  did not contain model revisions or full decoding metadata.
- Colab remains the primary interface for CUDA generation. The notebook reads
  `configs/generation/experiment.json` and the single frozen master
  `configs/generation/master_frozen_roster_4_to_7_year_old.csv`.
  Cohort snapshots and schedules are derived into each new run folder.
- Work remains limited to generation and identity/round separation checks.
  Scientific analysis logic and results are unchanged. R file references to
  the roster location were updated during the subsequent consolidation.
- The [generation baseline](provenance/generation_baseline_2026-10-14/README.md)
  records two four-year-old pair discrepancies: saved generations used
  SLI 474 / TD 459 and SLI 475 / TD 417, whereas the archived older roster had
  the TD children reversed. Recorded child/story/round identities remain
  traceable. The 120 corresponding Stage 1 CHAT exports lack ages, but the
  TD ages are available in their original ENNI transcripts. These findings
  are reconciled in the four-year-old rows of the portable master using recorded
  pair identities and child metadata. The original roster is externally
  backed up; existing CHAT files and analysis tables are not repaired or rewritten.
- The membership audit under
  `docs/provenance/generation_separation_2026-10-14/` executed only the
  current R data-assembly code. Its main and developmental pools have zero
  prompt/evaluation child overlap across all 17 age-specific rounds. All 940
  synthetic metadata rows match recorded generation ancestry.
- Historical Python outputs under
  `phase2/generated/4_year_old/evaluation/bundle_round_holdout/` violate the
  within-round rule: TD 459 occurs in round 2's evaluation pool and TD 417 in
  round 3's, producing 40 pairwise comparisons with prompt children. These
  historical outputs remain untouched and must not be presented as valid
  held-out evaluation without a separate correction/rerun.
- The separation pass applies to the round-stacked main/developmental R pools.
  The R PCA/correlation (`pca_base`) and distributional (`profile_df`) sections instead
  assemble unique full-reference real and synthetic rows without round-specific
  prompt exclusions. Those are descriptive reference comparisons, not certified
  held-out evaluations. Their code and results remain unchanged for the later
  analysis review; do not generalize the separation pass to the whole R report.
- Verified pre-cleanup backup:
  `/Volumes/TranscendBU/ICL-PILOT_78a5cadad659_20261014T115428Z/Generation/`.
- Subsequent consolidation, authorized on the same date, leaves one 47-pair
  master as the maintained roster input. The four cohort CSVs and the obsolete
  old master are backed up and removed; no compatibility copies are retained.
  `experiment.json` schema 2 names the master once. Cohort labels, orders,
  child identities and raw ages are preserved. Schedule builders reject
  mixing cohorts into one round. Only roster path references changed in
  `clinical_analyses_reordered.Rmd`, `clinical_analyses.Rmd`, and the existing
  `clinical_analyses_reordered_purl.R`; scientific code was not regenerated.
  Backup: `/Volumes/TranscendBU/ICL-PILOT_78a5cadad659_20261014T123617Z/GenerationRoster/`.
  Audit: `docs/provenance/master_roster_consolidation_2026-10-14/`.
- The user subsequently confirmed that fixed story scaffolds belong to the
  generation configuration. `story_unit_reference.json` was moved unchanged
  from `phase2/ENNI/measures/story_units/` to `configs/generation/`.
  `experiment.json` schema 3 names this asset at the top level, relative to
  its configuration directory; only the human exemplar CSV is resolved
  through `DATA_ROOT`. The scaffold hash and full job/prompt inputs are
  preserved. Corpus data, scientific analysis code and results are unchanged.
  Backup and move verification are recorded in
  `docs/provenance/generation_scaffold_config_2026-10-14/`.
- CHAT parsing was subsequently repaired in `src/icl_pilot/story_unit_semrp.py`:
  ages use `years;months.days`, CHI main-tier continuations are joined before
  normalization, and dependent tiers or recognizable orphan annotation blocks
  cannot enter narrative text. Extraction records source hashes and line-level
  QC and refuses to overwrite existing outputs. Surface normalization is
  otherwise unchanged; no grammatical forms, clinical labels or missing ages
  were invented or corrected.
- The new generation input is
  `configs/generation/inputs/story_unit_narratives_chat_v2.csv` (originally
  exported under `data/processed/generation_inputs/`, relocated on 16 October).
  Across 361 source CHAT files and 2,166 story rows, text and utterance arrays
  match the historical CSV exactly. The only populated field change is
  `Age_Month` in 2,160 rows. All 564 selected exemplars have ages matching the
  frozen roster. The old CSV remains unchanged for existing analyses.
- One file, `data/annotated/ENNI_batchalign/TD/B/523.cha`, has empty CHI age/group
  fields and shares the filename ID with an SLI-labelled file. Its six rows
  retain unknown age/group; no TD label is inferred from directory or `@Types`.
  It is not selected as a TD prompt. The corrected parser flags the missing
  metadata and all annotation remnants in its QC sidecar.
- The tier-removal utility now removes complete morphosyntactic tiers,
  including their continuations, and works on the declared Python 3.8 minimum.
  It was tested only on fixtures; no corpus tiers were removed or regenerated.
  Backup: `/Volumes/TranscendBU/ICL-PILOT_78a5cadad659_20261014T131515Z/CHATParsing/`.
  Parsing decisions and audit:
  `docs/guides/chat_narrative_parsing.md` and `docs/provenance/chat_parser_fix_2026-10-14/`.
- Prompt preparation uses the versioned message templates in
  `configs/generation/prompts/stage1.json` and `stage2.json`, selected by
  `experiment.json` schema 4. One `PromptBuilder` serves Colab previews and
  inference. The initial extraction preserved the pre-cleanup instructions and
  whitespace. The neutral wording test and subsequent restoration to the
  pre-cleanup baseline on 15 October are recorded below. Replicate-ID content
  and age-through-exemplar design are preserved.
  Preflight validates each selected
  exemplar's raw age against the roster and renders both stage templates.
- New runs write `prompt_records.jsonl` with attempted prompts before model
  calls and output records after decoding. The journal stores exact messages,
  rendered chat text, input-token hashes/counts, sampling parameters, exemplar
  identities/ages and job lineage. Stage 2 requires its completed same-job
  Stage 1 record and matching parent text. Output CSVs store both prompt IDs.
  Template bytes/code hashes are captured, and exports include the journal,
  manifest and generation-design snapshots. Failure records are not completions.
- Before the neutral wording update, all 2,820 job prompt pairs were checked against the pre-cleanup notebook;
  Stage 2 checks use actual saved historical Stage 1 outputs. This verifies
  message preservation, not fresh GPU inference. Checkpoint/restart and frequent
  persistent Drive saving remain the next execution work; journals currently
  require a fresh run directory. Runtime/model version pins and a real Colab GPU
  trial remain pending. No corpus data, scientific analysis or results changed.
  Backup: `/Volumes/TranscendBU/ICL-PILOT_78a5cadad659_20261014T134815Z/GenerationPrompts/`.
  Audit: `docs/provenance/prompt_builder_2026-10-14/`.
- The user subsequently deferred Colab execution work and requested cleanup of
  other repository areas. The first post-generation pass hardens the existing
  CSV-to-CHAT converter without changing narrative normalization, punctuation
  or utterance segmentation. It validates all rows before writing, requires a
  fresh output directory, keeps empty CSV cells distinct from text, and rejects
  duplicate destinations, conflicting age/child/cohort metadata and malformed
  identifiers. Known missing metadata can come only from the specified roster;
  absent specified rosters and unknown pairs now fail explicitly.
- New CHAT exports and their manifests retain available prompt-record IDs and
  same-row Stage 1 parent links. Historical CSVs without journal IDs remain
  supported. Manifest additions record metadata sources and source/output hashes.
  Maintained workflow Markdown files are now explicitly allowed by `.gitignore`;
  this does not stage, commit or publish them. Other documents retain their
  previous ignore behavior. See `docs/guides/post_generation_processing.md`.
- Existing transcripts, model settings, prompts, notebooks, R sources and
  scientific results are unchanged. Annotation is not rerun. The next separate
  cleanup is the TalkTag/CLAN workflow: the standalone TalkTag notebook has fixed
  Drive paths and an unpinned installation, and the generation notebook's optional
  annotation currently defaults to Stage 2 only. Source-preserving annotation,
  comparable real/synthetic processing and repair-tool lineage need review.
  Backup and verification: `docs/provenance/post_generation_cleanup_2026-10-14/`.
- The user then requested one generation-plus-annotation notebook, with
  independently runnable blocks and evaluation kept separately in a Python
  Colab/Jupyter notebook and R Markdown. The maintained interface remains
  `notebooks/colab_direct_generation_pipeline.ipynb` (retained path, updated title).
  Its modes are `full`, `generation_only`, `from_csv`, `convert_only`, and
  `from_chat`. Saved-input modes do not load the generator or exemplar corpus.
  The existing Python evaluation notebook and all R sources/results are unchanged.
- `REPLICATES=None` keeps the frozen ten-replicate study default. A positive
  override affects only the new run, is hashed in its effective configuration,
  and previews 282 jobs per replicate across 47 pairs/six stories. Each job has
  TD-like and SLI-conditioned outputs. Child pairs, input/exemplar/scaffold text,
  message templates, schedule order, global seed and model sampling calls remain
  unchanged. The runner now records completion/interruption status and explicitly
  retains `td_age` in new CSVs. It does not implement per-job reseeding or resume.
- Conversion and annotation are separate blocks after generation; the generator
  is released before TalkTag starts. Both synthetic stages are selected by
  default, and all selected cohorts/stages share one annotation invocation.
  New `configs/annotation/talktag.json` pins upstream source commit
  `a9df736acf4efaa0c6ec79240a2d36348de983ab`. This is a new portable-run source
  selection, not recovered historical annotation provenance. The annotation
  environment is separate; its installed source commit and package versions
  are recorded. Full software/model locks and actual GPU validation remain pending.
- `src/icl_pilot/annotation.py` copies CHAT inputs, joins wrapped CHI main tiers
  only in the copy, and retains raw TalkTag CHAT separately. A deterministic
  postprocessor restores the snapshot's case/spacing while retaining tags at
  their token boundaries; changed words, punctuation, order, boundaries or
  pre-existing annotations are rejected. Final QC compares surface tokens
  case-sensitively and checks file/utterance coverage and non-target/header/tier
  preservation. Hashes, commands, runtime, logs/reports and per-file restoration
  flags are recorded; failures are retained and debug-limited annotation is
  labelled partial. These checks do not certify linguistic tag accuracy. No
  existing repair scripts or CLAN/feature builders run automatically. Missing
  selected stages and overlapping CSV/input collections fail explicitly.
- `pipeline_manifest.json` records routes and block states. Drive copies are
  hash-verified, preserve the run hierarchy and reject another run's destination;
  exports include all local processing artifacts. Downloads/runtime release are
  optional and off by default. Generation Drive copies remain at cohort boundaries;
  frequent durable checkpoints and sampling-state recovery remain deferred.
- The standalone `notebooks/TalkTag.ipynb` was verified against the external
  backup and removed after the combined mocked workflow passed. The audit records
  143 passing focused tests, all 5,640 default stage prompts unchanged, unchanged
  default job/evaluation tables, and 29,598 protected files unchanged. No actual
  GPU inference, annotation, CLAN or scientific analyses were run locally.
  Backup: `/Volumes/TranscendBU/ICL-PILOT_78a5cadad659_20261014T143952Z/GenerationAnnotation/`.
  Audit: `docs/provenance/combined_notebook_2026-10-14/`.
- On 15 October the user requested a one-narrative Colab check. The notebook now
  defaults to `SMOKE_TEST=True`, executing one canonical target-story job through
  both generation stages, CHAT conversion and annotation. It previews one TD-like
  and one SLI-conditioned narrative; the test run ID uses `enni_smoke_`. Manifests
  and CSVs explicitly mark `test_only`, and the selected execution schedule is
  recorded separately from the unchanged complete study design. These outputs
  are not study data or complete six-story bundles. Set `SMOKE_TEST=False` for
  a full generation run. Replicate count alone is not a one-job limit.
  The switch applies only to new generation; saved-input routes retain their
  explicit input selection. Source exemplars and both models are still required.
- The Colab/VS Code setup now defaults to a persistent input tree under
  `MyDrive/icl_pilot_inputs`; local Jupyter still defaults to the checkout and
  `ICL_PILOT_DATA_ROOT` overrides both. A first upload of the exemplar CSV to
  `/content` is copied into the configured relative path on mounted Drive,
  SHA-256 verified, and recorded in `pipeline_manifest.json`. Later sessions
  reuse that file. Conflicting uploads stop without replacing existing inputs;
  saved-input routes skip exemplar staging. Outputs still use the existing
  Drive export folder. The author handles Drive authorization/upload in VS Code;
  no desktop control, Drive upload or GPU inference was performed by the agent.
  Validation: 159 focused checks passed, followed by all 36 pipeline checks
  after adding an additional Drive-mount test and integrating input staging
  into the mocked end-to-end route. Actual Colab inference remains pending.
- Live Colab troubleshooting on 15 October found a failure creating the TalkTag
  environment with `/usr/bin/python3` (Python 3.13). The VS Code traceback did
  not contain the child process output, so missing `ensurepip` is a suspected
  cause, not a verified diagnosis. The installer now uses `venv --without-pip`
  and kernel pip >=22.3 with `--python` targeting the annotation interpreter.
  It repairs partial environment configuration without clearing the directory
  and includes venv stdout/stderr in setup failures. TalkTag revision, annotation
  options and the generation protocol are unchanged. The remote checkout must
  receive this helper update or the current session must use the documented
  environment-recovery commands before retrying annotation.
- The author subsequently reported a completed smoke test and pasted the two
  annotated CHAT exports from `enni_smoke_20261015T090021775255Z`: first pair
  SLI 444 / TD 410, target A1, replicate 1. Their nine CHI utterances are identical
  and contain no inline error tags. The Stage 2 header links to the supplied
  Stage 1 prompt-record ID. This is one observed unchanged transformation, not
  evidence of clinical fidelity or a measured failure rate across the study.
  The checked Stage 2 prompt requests style/grammatical-level transfer while
  preserving events; it does not require explicit error insertion or reject
  unchanged text. The notebook now has a read-only whitespace-normalized copy
  check for newly generated or selected saved CSVs. It flags identity without
  changing prompts, outputs, sampling, selection or annotation behavior.
- The author explicitly confirmed that the prompts must remain unchanged and
  requested a further very small test. `SMOKE_TEST_JOBS` now controls the size
  of the canonical schedule prefix used with `SMOKE_TEST=True`. The notebook
  defaults to 12 jobs: all six stories, replicates 1 and 2, for the first frozen
  four-year-old pair SLI 444 / TD 410 (24 narratives total). The Python setup API
  retains one job as its default. Full study mode and its schedule are unchanged.
  Each test uses a fresh directory, records the exact selection and remains
  `test_only`. Seed 42 and the sequential sampling stream are retained, so the
  first job may reproduce the previous test; the later jobs provide additional
  conditions/draws. This is one-pair diagnostics, not a cohort-wide estimate.
  The local seven-file focused suite passed 176 tests. A direct comparison with
  Git commit `01cace79` checks all 5,640 prompts using historical Stage 1 parents,
  tokenization/chat-template/decoding and the model generation call, alongside
  the unchanged complete-input/evaluation fingerprints. Detailed verification
  is under `docs/provenance/small_batch_2026-10-15/`. Code reorganization,
  parsing/age metadata, roster reconciliation, provenance, I/O and annotation
  runtime changes must not be described as "no code changed". Historical model
  revisions and complete dependency versions remain incompletely recorded.
- The subsequent Colab run `enni_smoke_20261015T094156357444Z` reached annotation
  QC and stopped with `Annotation QC needs review`. TalkTag returned zero.
  The author supplied the manifest summary: expected and reported coverage both
  equal 287 utterances, with exactly one
  `target_text_or_boundaries_changed` issue for
  `4_year_old_stage2_sli/pair_sli444_td410_rep02_A2_SLI.cha`.
  The supplied before/after diff shows `me` changed to `Me`, a pronoun error tag
  added after that token, and spaces before final periods removed. The existing
  checker already ignores punctuation spacing and recognized tags; the initial
  case change caused this flag. Tag correctness has not been adjudicated.
  `RUN_OUTPUT_DIR` had meanwhile changed to a newer run, so diagnostics/recovery
  must use the failing run's explicit path.
- The author requested proper postprocessing rather than accepting changed
  capitalization in QC. New runs retain exact output under `raw_talktag/` and
  write restored copies under `annotated/`; the case-sensitive checker remains
  unchanged. `postprocess_annotation_run()` can recover retained older batches
  into fresh `postprocessed/` and `postprocessed_manifest.json` artifacts without
  model inference, replacing the old manifest or altering raw files. It verifies
  recorded source/snapshot/raw hashes and repeats the coverage/preservation QC.
  The recovery manifest retains the original issues and source-manifest hash.
  Prompts, sampling, generation CSVs and scientific analyses are unchanged.
  All 192 focused tests passed, covering the pasted case, tag anchors, existing
  annotations, real textual changes and both fresh/recovery workflows. Python
  compilation and diff checks passed; all notebook code cells and the final
  case-sensitive comparator match the preceding revision. The local environment
  still reports its existing Pandas/NumExpr compatibility/deprecation warnings.
  Actual recovery of the remote 287-utterance
  batch must still be executed by the author; it has not been run by the agent.
- On 15 October, review against the CHAT header specification identified a
  four-field `@Types: cross, narrative, generated, <group>` defect in the
  converter. New exports now use `cross, narrative, TD` or `SLI`, with
  `@Comment: data_source=generated` separately preserving synthetic provenance.
  No narratives, prompts, model settings, child metadata or analyses change.
  `repair_generated_chat_types()` can produce a fresh corrected copy of an
  existing generated CHAT collection, including annotated files, without
  inference. It checks generated provenance and agreement with the CHI `@ID`
  group, validates the whole collection before writing, preserves other bytes,
  and inventories source/output hashes in `formatting_manifest.json`.
  Original transcripts and annotation manifests remain unchanged; the repair
  manifest identifies the derived files. This is a narrow header correction,
  not full CHAT validation. See `docs/guides/post_generation_processing.md` for the
  saved-batch procedure. All 203 focused generation/annotation/parsing tests,
  Python compilation and diff checks passed. Tests cover TD/SLI, LF/CRLF line
  endings, byte-preserved tagged/dependent tiers, unchanged original manifests,
  already-correct files and rejection before writing of inconsistent headers.
  Actual remote batch repair and CLAN CHECK remain pending.
- The author subsequently reasserted that generation prompts should contain no
  TD/SLI or similar diagnostic references. An audit found two literal TD headings
  in `prompts/stage1.json` and `clinically plausible` in Stage 2's system message.
  These are model-visible; internal stage/group metadata are separate. Direct
  rendering checks confirmed that the templates at the time of that audit matched Git notebook
  versions `f2e61695`, `78a5cada` and `01cace79`. The 7 April `f2e61695` update
  removed Stage 2's explicit SLI/profile instructions but left those TD/clinical
  strings. Previous preservation checks therefore establish equality to the
  pre-cleanup baseline, not compliance with the intended label-free design.
  The exact intended original wording remains unidentified. No templates or
  generated artifacts were changed during this audit. Do not claim that prompts
  or historical generations are label-free on the basis of the prior audit;
  use the actual run journal where available. See `docs/guides/generation_prompt_records.md`.
- The author then approved the neutral wording previously shown in chat for
  the next smoke test, without an additional evaluation experiment. Both stages
  now use the exemplar's language/narrative style as their linguistic guide.
  Model-facing instructions omit ENNI/TD/SLI/DLD, child-like/simple-language
  requests, adult-style avoidance, clinical framing and prescribed shortening.
  All six scaffold `prompt_constraints` were updated to event order/no new
  events only. Event units, titles, invariants, optional content, exemplars,
  rosters, counterbalancing and sampling settings remain unchanged. Internal
  provenance labels are retained. Templates are still hashed and snapshotted
  per run; historical templates remain in Git and existing run snapshots.
  The notebook preview now prints the complete Stage 1 prompt and both template
  hashes. An existing Colab session must update its checkout and rerun from a
  restarted kernel into a fresh run. No original generated/annotated files or
  analysis results were rewritten. Historical execution equivalence and the
  cause of the poorer smoke outputs remain unresolved.
  Validation: 211 focused tests passed. All 5,640 assembled stage messages were
  checked for neutral instruction/constraint wording while preserving exemplar
  text and historical Stage 1 parents literally. Restoring only the old
  constraint column reproduces the preceding full-input fingerprint, and the
  evaluation-manifest fingerprint is unchanged. All 5,640 saved historical
  narrative texts still match the pre-cleanup inventory. Seventeen historical
  Stage 1 parents themselves contain inherited style words; these were retained
  in the rendering audit, never used as new generated outputs or silently
  filtered. No GPU inference was performed. Audit:
  `docs/provenance/neutral_prompt_wording_2026-10-15/validation.json`.
- The author reported another smoke test without grammatical errors and then
  requested a return to the prompts in the original notebook. The active
  templates and complete scaffold reference have been restored verbatim to
  `7dbfc657`, whose messages match the pre-cleanup notebook at `78a5cada`.
  This restores the TD/ENNI headings, child-like/simple-language constraints,
  clinical framing and Stage 2 guidance on grammatical level/shortening. The
  neutral version remains in Git commit `9a59fa21`. No severity/profile
  conditioning is restored; no narrative, model setting, roster, evaluation
  rule, annotation helper or scientific result is changed. The newest smoke
  output and runtime have not been inspected directly. Earlier smoke failures
  predate the neutral edit, so the edit cannot explain the entire discrepancy.
  Original raw CSVs already contain fragmented clauses (four-year-old first
  pair A1) and nonstandard agreement (six-year-old first pair A1) before
  annotation. Historical executed prompts and runtime remain incompletely
  recorded; restoring known notebook source does not establish historical
  inference equivalence or guarantee error-bearing output.
  Validation: 211 focused tests passed; all 5,640 assembled stage messages match
  the actual functions extracted from Git notebook `78a5cada`, using saved
  Stage 1 outputs only as rendering inputs. The complete run-input and evaluation
  fingerprints match the pre-neutral baseline, and all 5,640 saved narrative
  text hashes still match the original inventory. Notebook executable cells
  are unchanged. Compilation and diff checks passed; no GPU inference was run.
  Audit: `docs/provenance/prompt_restoration_2026-10-15/validation.json`.
- The author subsequently reported improved smoke narratives after prompt
  restoration and deferred runtime/resume work. They explicitly clarified that
  the next task is CSV-to-CHAT correctness, not manuscript writing or R changes.
  A fresh export of all 2,820 saved jobs (5,640 files) exposed 649 standard
  Chatter failures (658 with strict quotation-link checks), despite preservation
  of the original lexical sequence. The converter damaged existing pause and
  quotation codes and did not normalize written numerals or prose symbols.
  It now protects CHAT notation, repairs quotation layout and records explicit
  orthographic normalization in `chat_manifest.csv` (`source_text_sha256` and
  `formatting_changes`). See `docs/guides/post_generation_processing.md` for the exact
  number, symbol, parentheses and quotation policy. All 5,640 fresh outputs pass
  Chatter 0.30.0 with `--force --strict-linkers`, without suppressed checks.
  The original source CSV hashes are unchanged; no original CHAT, annotation,
  measurement table, prompt, R code, paper text or result was overwritten.
  These are format checks, not validation of C-unit segmentation or clinical
  fidelity. The latest remote Colab smoke export remains to be checked directly.
  Validation: 229 focused tests passed, including 17 regression examples run
  through the pinned Chatter executable. All final normalized lexical sequences
  match the emitted main tiers, with zero unexplained changes. Before/after
  inventories include file and utterance counts; no measurements were rerun.
  Audit: `docs/provenance/chat_conversion_validation_2026-10-15/`.
- On 16 October the author authorized a complete `data/` backup and cleanup,
  retaining the corrected pre-annotation CHAT and usable TalkTag files. They
  confirmed that real ENNI transcripts and the portable prompt inputs must also
  remain. The verified full pre-cleanup snapshot is
  `/Volumes/TranscendBU/ICL-PILOT_af80b106b6dd_20261016T085254Z/Data/`:
  20,330 original file hashes and all 20,411 archive members were checked before
  removal of 19,960 obsolete local files (45,019,837 file-content bytes).
- The maintained historical generation data now live at
  `data/generated/<age>_year_old/pre_annotation/{stage1_td,stage2_sli}/` and
  `data/generated/<age>_year_old/talktag/{stage1_td,stage2_sli}/`.
  All 5,640 pre-annotation files are byte-identical to the previous successful
  audit exports; their copied CSV manifests now point to the retained paths.
  All 5,578 retained TalkTag files pass Chatter 0.30.0 with strict linkers,
  alignment enabled and no suppressions. They also match the corresponding
  pre-annotation CHI word sequences and utterance boundaries under the explicit
  written-variant comparison. The data guide is `data/README.md`; reports,
  per-file hashes, provenance, repairs and dispositions are under `data/qc/`.
- `src/icl_pilot/chat_structure.py` formats generated tagged copies while
  conserving their main-tier counts and every bracketed code/lexical anchor.
  It repairs the known generated header, explicit notation, spacing, quotation
  links and commas that prevent repetition-code attachment. It refuses body
  changes with existing morphosyntactic tiers. Missing recorded ages in 120
  historical TD files are restored from the already verified pre-annotation
  exports; conflicting metadata are rejected. No words are reconstructed,
  grammatical forms repaired or missing error codes inferred.
- Primary selection retains the existing curated per-story copies, with
  age-6 TD taken from the processed copy carrying its TalkTag report. Fifty
  complete existing alternative copies replace primaries that fail QC; tags
  from different versions are never combined. Their exact original source
  hashes and paths remain in `data/qc/retained_inventory.json`. Original
  TalkTag reports are preserved as historical provenance, not as checksums
  of subsequent edits or proof of an installed source revision.
- Annotation coverage remains incomplete: 62 corrected pre-annotation files
  are queued in `data/qc/needs_reannotation.json`. Reasons include 20 unresolved
  structural failures, 16 surface-text discrepancies and 27 boundary
  discrepancies (one file has two reasons). The rejected historical inputs
  and formatted repair attempts are externally backed up. These cases require
  fresh annotation/review, not scientific exclusion; all corresponding
  pre-annotation files remain. No classifier or statistical analysis was rerun.
- All 361 retained real ENNI transcripts/metadata and both portable prompt-input
  files are unchanged. Real-reference CHAT validation was not part of this
  generated-data cleanup; the directory name `ENNI_batchalign` is historical.
  Original CSVs, old CHAT/annotation copies, bundles, picture-derived story
  packets and old data-folder measurements are available in the verified
  snapshot. `phase2/`, existing analysis artifacts, prompts, rosters and model
  settings remain untouched. Fresh CLAN annotation/measurements and subsequent
  analyses remain pending. New notebook runs still use their own run folders.
  Validation: 248 focused tests passed (three existing Pandas/NumExpr warnings),
  the repository layout check passed, and every retained generated CHAT file
  passed fresh Chatter validation. Tests and corpus checks are recorded in
  `data/qc/`; the full cleanup audit is under
  `docs/provenance/data_cleanup_2026-10-16/`.
  The cleaned dataset also has a separately verified
  `retained_data_snapshot.tar.gz` in the same external backup directory
  (11,614 files). After checking all 5,640 transcript hashes against both the
  retained copies and that snapshot, the duplicate former audit-export tree
  was removed. Its original CSV manifests remain in the external backup;
  historical audit reports are preserved with a relocation notice.
- The author then explicitly requested source-guided repair of those 62
  annotated files: match the corrected pre-annotation versions, retain correct
  existing annotation and remove incorrect annotation. This supersedes the
  earlier decision to hold the 62 out of the local annotation set. New
  `src/icl_pilot/annotation_recovery.py` restores all speech, case, punctuation,
  headers and main-tier boundaries directly from the corrected reference.
  Only existing, uniquely aligned bracketed codes can be inserted. Changed
  adjacent words require documented context review; ambiguous anchors cannot
  be forced. Existing morphological tiers or differing dependent tiers fail
  explicitly rather than being silently reused or dropped.
- All 62 repaired outputs reproduce their reference byte-for-byte after removal
  of the recorded insertions and pass Chatter. Of 124 existing error tags, 90
  are retained and 34 removed after selective review against the CHAT scheme
  and restored context. Examples include incorrect missing-progressive labels
  on reduced -in forms, derivational labels on spelling/nonword contrasts,
  past-error labels on grammatical past lay, an irregular-past label on regular
  happen, and pronoun-case labels unsupported in a possessive phrase or fragment.
  Three other codes cannot be attached safely and are removed. No substitute
  or new error label is inferred. Restored speech comes only from the corrected
  pre-annotation reference; no new generation was performed.
- The current local TalkTag set again contains all 5,640 story files, all passing
  Chatter 0.30.0 with strict linkers, alignment enabled and no suppressions.
  All normalized word sequences and utterance boundaries match their corrected
  pre-annotation parents. The existing 5,578 TalkTag files, all 5,640 reference
  files, real ENNI transcripts and prompt inputs are hash-verified unchanged.
  `data/qc/annotation_restoration_summary.json`, `annotation_restoration.jsonl`
  and `annotation_decisions.csv` document scope, hashes, kept/removed codes and
  reasons. The previous 62-file queue and validation reports remain in QC history;
  the active queue is empty. File coverage is complete; linguistic annotation
  completeness is not established, since restored spans and removed labels do
  not receive new inferred tags. This selective review is not independent gold
  annotation. Fresh CLAN tiers/measurements and analyses remain pending.
  All 265 focused tests pass (three existing Pandas/NumExpr warnings). Original
  annotations, the repaired dataset and the exact code/review records are
  separately archived and hash-verified on TranscendBU; temporary local repair
  transcript copies are removed after verification.

### Subsequent real ENNI copy removal — 16 October 2026

- The author explicitly superseded the earlier decision to retain
  `data/annotated/ENNI_batchalign/` and authorized its removal without another
  backup, because source corpus files can be obtained from CHILDES.
- The removed directory contained 361 historical processed CHAT files
  (286 TD-labelled and 75 SLI-labelled transcript files, not verified participant
  counts) and six supporting files, totaling 3,620,393 content bytes. Inspection
  found morphology/dependency-like continuations without labelled MOR/GRA or UD
  tiers, conflicting with its historical Batchalign notes. No real-reference
  Chatter validation or annotation repair was claimed.
- Generated pre-annotation and TalkTag collections and the portable prompt inputs
  remain unchanged. The generation notebook reads the retained prompt CSV;
  fresh corpus extraction requires source transcripts supplied separately.
  Historical source paths and preservation reports retain their original meaning
  as records of earlier steps, not claims that these files are still local.
- No additional archive was created. Earlier verified data snapshots contain
  the removed processed copies; a fresh corpus download should not be assumed
  byte-identical to historical processed annotations. The removal and preservation
  checks are recorded in `data/qc/real_enni_removal.json`.

### Frozen generation inputs moved into configuration — 16 October 2026

- The author requested that the important exemplar CSV and parsing-QC sidecar
  move into the configuration folder. Both now live at
  `configs/generation/inputs/`, moved byte-for-byte with their source hashes and
  embedded historical paths intact. The empty former input and processed
  directories were removed; no duplicate input copy is kept under `data/`.
- `experiment.json` keeps schema 4 and now points `inputs.narratives_csv` at
  `configs/generation/inputs/story_unit_narratives_chat_v2.csv`, relative to
  `DATA_ROOT` as before. Other configuration-relative assets and the Python setup
  API retain their existing path semantics.
- The notebook defaults `DATA_ROOT` to the checkout for both Colab and Jupyter,
  so a checkout containing the inputs needs no separate CSV upload. Explicit
  `ICL_PILOT_DATA_ROOT` overrides and safe upload staging remain supported.
  Existing external input trees must use the new configured relative path;
  they are not silently migrated. Drive output persistence is unchanged.
- The move changes input locations and setup defaults, not exemplar content,
  frozen prompts, child pairs, decoding settings, annotation or analyses.
  The configuration/input READMEs, extraction guide and notebook guide identify
  the new sources of truth. Historical preservation/removal records retain the
  earlier paths as provenance. Relocation hashes and before/after design checks
  are recorded under `docs/provenance/generation_input_relocation_2026-10-16/`.
- Validation: all 222 targeted tests pass (three existing Pandas/NumExpr warnings).
  CPU preparation before and after the move produces identical 2,820-job and
  evaluation tables and all 5,640 rendered stage-message objects, using a fixed
  verification parent for Stage 2. Both moved files are byte-identical, and all
  11,292 existing generated-directory files are hash-verified unchanged. Only the
  notebook settings/input-staging code cells changed; no GPU generation,
  annotation, CLAN processing or scientific analysis was rerun.

### Outputs maintenance cleanup — 16 October 2026

- The author requested removal of the old presentation folder and temporary
  maintenance files. Inspection found useful provenance and the local Chatter
  executable mixed with test fixtures, caches and intermediate export archives.
  The useful material was preserved before removing the old directories.
- The complete `outputs/maintenance/` and
  `outputs/manual-20260601-icl-update/` trees are archived at
  `/Volumes/TranscendBU/ICL-PILOT_af80b106b6dd_20261016T113750Z/Outputs/maintenance_and_manual_snapshot.tar.gz`.
  All 6,196 regular-file hashes and 748 symlink targets were verified, followed
  by a source-inventory recheck, before local removal. The archive SHA-256 is
  `662d59a7b7541b1eb5f54500527121eef5d02ff888daf5f759ca3f6af13d3671`.
- `docs/provenance/` retains 119 selected historical reports (approximately
  5 MiB), including the generation baseline, split audits, input-move checks
  and backup/repair summaries. Their contents remain unchanged; their original
  paths and descriptions are historical, and omitted scripts/inventories remain
  in the complete external archive. The provenance README and
  `outputs_cleanup_2026-10-16.json` identify the archive and retained-file hashes.
  Audit locations earlier in this handoff now point to these retained reports.
- The verified Chatter 0.30.0 executable moved unchanged to
  `tools/chatter/chatter`, with its release/hash record in `source.json`.
  The binary is ignored by Git. Active documentation points to this location.
  Existing `outputs/synthtexteval_demo/` and `outputs/annotation_qc/` contents,
  generated transcripts and generation inputs are hash-verified unchanged.
  No model inference, corpus annotation, CLAN measurements or analyses were rerun.
- The relocated executable passes all 17 native CHAT-formatting regressions;
  temporary test/cache files were removed afterward. The cleanup reduces the
  local file footprint by approximately 201 MiB after retaining reports and
  Chatter. `outputs/` now occupies approximately 36 MiB and contains the existing
  evaluation and annotation-QC folders. Git whitespace and preserved-file hash
  checks pass; detailed verification is in the provenance cleanup record.

### Framework-evaluation results organization — 16 October 2026

- The author agreed to preserve the framework result bundle under a clearly
  named results area. All 180 files formerly in `outputs/synthtexteval_demo/`
  now live in `results/framework_based_evaluation/run_20260827T153330Z/`, with
  every file hash unchanged and all 22 post-hoc output checksums verified.
  The run identifier comes from the recorded 27 August export timestamp; the
  bundle includes the later 28 August post-hoc analyses. Historical absolute
  Colab paths and the archived post-hoc script remain unchanged as provenance.
- The maintained evaluation notebook now creates a fresh UTC-dated directory
  under `results/framework_based_evaluation/`. An explicit `EVALUATION_RUN_ID`
  must name an unused directory. `src/icl_pilot/evaluation_runs.py` supplies the
  shared creation helper and refuses invalid names or existing runs. It is
  imported from the accompanying checkout after repository discovery.
  Evaluation methods, model/metric settings and saved notebook outputs remain
  unchanged; only setup/export naming and run identity metadata changed.
- Future Colab exports use `MyDrive/ICL-PILOT/framework_based_evaluation/`.
  Existing Drive artifacts were not accessed or moved. No evaluation, model
  training, annotation or R analysis was rerun. The pre-existing univariate
  result table remains at its original location under `results/`.
  Folder guides, active path references and ignore rules identify the new home;
  result payloads stay ignored by Git, while the guides are versionable.
  Relocation hashes are in
  `docs/provenance/framework_results_relocation_2026-10-16.json`.
- Validation: 10 focused checks pass, covering fresh run creation, refusal of
  existing names and traversal paths, and the actual notebook settings block.
  All 180 historical result files and 11,308 other protected files retain their
  hashes; saved notebook outputs are unchanged. Git whitespace checks pass.

### Documentation organization — 16 October 2026

- At the author's request, 11 loose documents were grouped into two minimal
  subfolders: `docs/guides/` for five workflow/reference guides and
  `docs/background/` for the original report and five earlier/exploratory plans.
  The current strategy handoff remains at its stable top-level path, and the
  existing `docs/provenance/` collection remains in place.
- `docs/README.md` indexes the retained documents and distinguishes maintained
  instructions from historical/proposed procedures. Document references, relative
  links and ignore rules were updated. No document was discarded or restored,
  and no experimental method, executable notebook cell or result was changed.

### Phase2 backup and first cleanup — 16 October 2026

- The author requested a full `phase2/` backup before starting cleanup. The
  complete directory, including source transcripts, CLAN outputs, features,
  scripts, input snapshots and historical results, is archived at
  `/Volumes/TranscendBU/ICL-PILOT_af80b106b6dd_20261016T122903Z/Phase2/phase2_snapshot.tar.gz`.
  All 9,282 regular-file hashes were verified by archive read-back and a second
  local-inventory check before removal. The archive SHA-256 is
  `f778b7355c42acaf82d79f632377829715566f0dc395c101941ed91a4834f267`.
- The first pass removes only three byte-identical partial prediction checkpoint
  tables, retaining their complete counterparts, plus 28 disposable Finder/Python
  cache files. Removed file content totals 249,027,566 bytes (approximately
  237.5 MiB). The other 9,251 original phase2 files and 11,488 protected files
  outside phase2 are hash-verified unchanged. No feature, measurement, transcript,
  code or unique scientific result was removed or regenerated.
- `phase2/README.md` maps the current consumers and distinguishes the two Gillam
  selections and legacy raw-feature names from retired profiling. Exact
  removals, duplicate replacements, source hashes and the verified archive
  location are recorded in `docs/provenance/phase2_cleanup_2026-10-16.json`.
- A preserved framework-text export still assigns TD to the six real story
  rows of child 523 via its transcript directory, whereas the current prompt
  parser leaves the missing source group unresolved. This missing-label rule
  differs between exports and needs review before a fresh evaluation; this
  cleanup did not alter clinical labels or existing results. Historical
  leakage/supersession issues remain documented below. Further reorganization
  must preserve the active R and framework text inputs.

### Replication workspace removed after verified backup — 16 October 2026

- The author explicitly requested removal of the entire local
  `phase2/replications/` directory. All 545 remaining files were checked against
  the full phase2 backup manifest and freshly read back from the archive above;
  the archive SHA-256 was also rechecked before deletion.
- The directory is now absent locally, freeing 1,464,197,120 allocated bytes
  (approximately 1.36 GiB). Its historical scripts, input snapshots, unique and
  superseded results, and presentations remain in the external archive.
- The 20,335 protected files covering retained phase2 data, source, configs,
  notebooks, data, results, tests and selected root files were hash-verified
  unchanged. The active generation, R analysis and framework notebook have no
  executable references to the removed workspace in the checked sources. No
  analysis or annotation was rerun.
- `docs/provenance/replications_removal_2026-10-16.json` records every removed
  file and hash, the backup location and verification. A byte-identical copy is
  stored beside the archive as `REPLICATIONS_REMOVAL.json`. Historical references
  below describe archived artifacts and do not imply their local availability.

### Python source organized by responsibility — 16 October 2026

- The author authorized source reorganization after the folder review. The
  package now groups implementation under `generation/`, `chat/`, `features/`,
  `evaluation/` and `maintenance/`; `src/README.md` is the source map. The root
  retains notebook-stage coordination, config paths/manifests and the CLI.
  Maintained notebook, test and guide imports were updated together. Historical
  audit references retain their original paths. Use notebooks and source from
  the same updated checkout; previously imported modules in a running kernel
  require a restart before switching to the new package paths.
- Thirty-three modules were moved. The raw linguistic feature builders now live
  in `features/raw_features.py` and `features/generated_raw_features.py`, retaining
  their legacy function names, output names and scientific definitions. The
  mixed-model/anomaly helpers were moved unchanged from generation workflow into
  `evaluation/bundle_models.py`; the integrated R analysis was not changed.
  Historical/case-specific repairs remain available under `maintenance/`, with
  their older layout and analysis scope documented. No substantive research
  implementation was discarded.
- The CLI's optional imports, hidden parser branches and two unreachable handlers
  for already-absent Batchalign runners were removed; all 33 previously available
  commands remain available. The historical TalkTag repair entry point now reads
  the shared project-root constant, preserving its root after the module move.
- `evaluation/story_text_exports.py` requires explicit `(cohort_year, CSV path)`
  inputs, or repeated `--generation-source AGE=CSV` arguments. Obsolete hardcoded
  generation CSV defaults were removed. The existing text selection, normalization,
  group-label rules, age window and real-reference defaults were preserved, with
  ten new tests for explicit source selection, provenance and the separate Gillam
  export. CSV exports do not incorporate later CHAT repairs; the previously
  flagged missing-label fallback remains unresolved pending evaluation review.
- A verified pre-edit snapshot of 68 source, test, notebook, guide and project
  files is stored at
  `/Volumes/TranscendBU/ICL-PILOT_af80b106b6dd_20261016T125209Z/Source/source_before_reorganization.tar.gz`.
  All archived file hashes were checked by read-back before edits. The detailed
  module map and verification are in
  `docs/provenance/source_reorganization_2026-10-16.json`.
- Before/after checks preserve all 2,820 planned jobs, evaluation manifests and
  5,640 rendered stage-message lists (Stage 2 checked with a fixed parent fixture).
  The 20,220 protected data/config/result/R/project files are hash-identical, and
  the notebook changes are limited to module paths, preserving saved outputs,
  metadata and execution counts. AST comparisons preserve 345 function/class
  definitions, including the relocated statistical helpers; four definitions
  changed only for the explicit input/root/CLI housekeeping above and two dead
  handlers were removed. All 303 tests pass on Python 3.8, including native
  Chatter regressions, with the same three pre-existing dependency warnings.
  This was local validation with mocked inference; no new GPU generation,
  TalkTag inference, CLAN processing or scientific analysis was run.

### R input dependencies replaced by a pending-input signpost — 16 October 2026

- While considering a fresh study after reviewer feedback, the author requested
  removing R file dependencies on the historical dataset locations and adding a
  signpost to update once the revised files and paths are known. No new protocol,
  counterbalancing table, corpus selection or statistical specification was
  adopted by this change.
- `configs/analysis/inputs.R` is now the shared configuration for
  `clinical_analyses_reordered.Rmd`, `clinical_analyses.Rmd` and the retained
  `clinical_analyses_reordered_purl.R`. Its five input paths are deliberately
  blank: real/synthetic raw linguistic features, the matching frozen roster,
  and real/synthetic developmental measures. Every R entry point checks all
  inputs before loading packages/data and stops clearly while paths are pending
  or unavailable. There is no automatic Phase 2 or historical-roster fallback.
- Paths resolve relative to the checkout or can be absolute. Optional
  `ICL_PILOT_*_CSV` overrides are documented in `configs/analysis/README.md`;
  `ICL_PILOT_REAL_CSV` retains its existing override role without an old-data
  default. `ICL_PILOT_PROJECT_ROOT` supports execution outside the checkout.
  The existing analysis-tag behavior and output code are unchanged.
- The older R Markdown and extracted script previously defaulted to the pooled
  real table, unlike the main document's ENNI-only default. All now require
  explicit inputs; neither historical corpus choice is silently inherited by
  the revised study. Statistical code and interpretations remain available for
  protocol review. The extracted script was edited only for setup/paths, rather
  than regenerated from the differently organized main document.
- Before/after parsed-expression checks preserve 461, 525 and 460 non-setup R
  expressions respectively, apart from their five CSV path arguments. Base-R
  regression checks pass for pending inputs, explicit files, relative/absolute
  paths, spaces, environment overrides, missing files/directories and all three
  actual setup blocks when run outside the checkout. Full R syntax parses.
  The installed knitr/xfun combination emits an existing `xfun::attr()`
  deprecation during code extraction; it occurs in the pre-edit documents too.
  No models, analyses or rendered outputs were regenerated.
- The three pre-edit R files and handoff are preserved in a hash-verified
  external snapshot at
  `/Volumes/TranscendBU/ICL-PILOT_af80b106b6dd_20261016T133744Z/RAnalysisInputs/r_analysis_before_input_refactor.tar.gz`.
  `docs/provenance/r_input_signpost_2026-10-16.json` records file hashes and
  validation. Phase 2 has not been removed by this task; the framework notebook
  and historical exporter defaults still have separate Phase 2 dependencies.

## 2. Profiling Is Archived Exploratory Work

- Severity/profile scores were computed in earlier exploratory work, including
  some preliminary four-year-old severity-adjusted models.
- They did not condition the active prompts, determine age matching, or enter
  the current integrated R Markdown models.
- The active code no longer loads profiling tables to construct rosters or
  generation packages. New runs leave legacy profile metadata unknown; existing
  records retain their historical fields for lineage.
- Raw linguistic feature tables remain active inputs despite their legacy
  `severity_features_raw_*` filenames. They must not be deleted as profiling.
- Backup and restoration inventory:
  `/Volumes/TranscendBU/ICL-PILOT_78a5cadad659_20261014T092052Z/Profiling/`.
  The local audit is under `docs/provenance/profiling_cleanup_2026-10-14/`.

## 3. Historical Profiling Observations

The following observations arose from the archived exploratory profiling.
They are not a description of active severity scoring in the current paper.

- The strongest severity signal in the current corpus is **bare marking / missing required morphology**.
- Classic structural weakness also clearly contributes:
  - lower MLU,
  - lower DSS,
  - lower IPSyn,
  - lower lexical diversity.
- Past overgeneralization is present, but looks more like a developmental/profile feature than the main severity driver.
- Pronoun problems are not SLI-only; they also appear in TD, though less often.

## 4. Error Grouping Currently In Use

Current grouped morphosyntactic error features:

- `bare_marking_errors`
- `pronoun_errors`
- `past_overgeneralization_errors`

Important note:

- `determiner_errors` was removed from the core profiling pipeline after cleaning showed that the surviving determiner feature was too sparse.
- broad `rule_generalization_errors` was replaced in the core profiling pipeline by a narrower **past overgeneralization** feature.

## 5. Rate Decisions

Use per-utterance rates for disruption/error burden:

- `Word_Errors_per_utt`
- `retracing_per_utt`
- `repetition_per_utt`

Use per-word / opportunity-based rates for grouped morphosyntactic burden:

- `bare_marking_errors_per_word`
- `pronoun_errors_per_word`
- `past_overgeneralization_errors_per_past_opp`

The current past-overgeneralization denominator is:

- overt past finite lexical verbs from `%umor`
- plus explicit bare-past error tokens (`m:0ed`, `m:base:ed`)

So:

- `past_overgeneralization_errors_per_past_opp`
  =
  `past_overgeneralization_errors / (overt past finite verbs + bare past error tokens)`

DSS category totals remain developmental scores, not rates. TD-referenced
age adjustment was part of the archived profiling; the active R analysis
models age directly.

## 6. Historical Age Adjustment

- Retained features were age-adjusted using the **TD group only**.
- The adjustment is TD-referenced and age-based.
- Features were then aligned so that **higher values always mean greater severity**.

## 7. Active Features And Archived Profiles

Active inputs include `phase2/ENNI/measures/dev_measures.csv`,
`phase2/ENNI/measures/severity_features_raw_real.csv`, and
`phase2/generated/measures/severity_features_raw_syn.csv`.

The age-adjusted tables, profile composites, severity bands, dedicated
profiling documentation, and their builders are preserved in the external
backup. Its manifest records the original repository-relative paths and
checksums. The historical SLI severity bands were corpus-internal tertiles,
not clinical categories. Frozen rosters, schedules and existing generated
metadata remain local and unchanged.

## 8. Prompt Design Decision

The agreed conclusion is:

- the **core prompt architecture remains the same as in the PDF report**
- this means:
  - story-level prompting,
  - age matching,
  - cross-story counterbalancing,
  - two-step TD-to-DLD style prompting

What changed is our understanding of what the corpus can support:

- full `age x severity x profile` prompt stratification is too sparse for SLI
- therefore the pilot should remain close to the original PDF design
- the earlier profiling was exploratory and is now archived; do not describe
  it as an implemented conditioning or active evaluation step

## 9. ENNI Story Content Scaffold

- ENNI does **not** appear to provide an official canonical adult verbal story script.
- The official materials provide the picture sequences and administration instructions, but not a normative adult retell to use as a source text.
- Therefore, the project should **not** invent an "adult ENNI version" and treat it as if it were part of the instrument.
- Instead, each target story should have a project-internal **canonical story content representation** derived from the official ENNI picture sequence.
- This canonical representation should be concise and structured, for example:
  - page-by-page events,
  - main referents,
  - key actions,
  - outcomes / story grammar units.
- That representation should be used as the stable content scaffold for stage 1 prompting.
- In other words:
  - official ENNI picture story = content source,
  - internal canonical outline = prompt scaffold,
  - generated TD-like narrative = stage 1 output.

## 10. Evaluation Design Decision

- We want to preserve natural data for evaluation.
- A single fixed prompt/eval split wastes too much data.
- The agreed solution is a **rotating held-out design**:
  - in each round, one subset of children is reserved for evaluation,
  - the remaining children are used for prompt construction,
  - no child appears in both roles within the same round,
  - across rounds, children may change role.

This should be described in the paper as a **stratified rotating held-out evaluation design** or a **cross-validated prompt/evaluation design**, not as a custom ad hoc workaround.

## 11. Current Analysis Scope

The domain-specific paper analysis is `clinical_analyses_reordered.Rmd`, using
ENNI as its real reference corpus. Its linguistic measures, mixed models,
distributional comparisons, PCA/correlation analyses and paired stage changes
are separate from severity/profile modelling. Gillam is used for the
framework text-classifier external evaluation. Additional Python feature-
classification methods are deferred to a future manuscript revision.

## 12. Key Paper-Safe Summary

If a future chat needs the shortest possible summary:

> Generation remains story-level, age-matched, cross-story counterbalanced,
> and staged from a picture-derived content scaffold to TD-like and then
> DLD-like text. Profiling is archived exploratory work. Linguistic fidelity
> is evaluated against ENNI in rotating held-out rounds, separating prompt
> and evaluation children within each round. Framework text classifiers use
> a separate external Gillam sample.

## 13. Gillam Lineage And ENNI-Only Sensitivity Audit (August 2026)

The replication scripts and result paths referenced in this historical audit
were externally archived and removed locally on 16 October 2026; see the
replication-workspace removal record above. Restore the archived workspace
before attempting to rerun its commands.

The pooled real feature table at
`phase2/generated/measures/severity_features_raw_real.csv` is an exact
row-wise concatenation of the ENNI and Gillam real feature tables. In the
49.9--95 month Python evaluation window, it contains 351 children (212 TD,
139 SLI). The ENNI-only table contains 235 (188 TD, 47 SLI), and the 116
Gillam rows contain 24 TD and 92 SLI children.

Gillam therefore supplied most of the pooled SLI evaluation sample. Removing
Gillam is a useful corpus-sensitivity analysis, but a naive pooled ENNI-only
augmentation test is not prompt-independent: all 47 ENNI SLI children in the
evaluation window, and 47 of 188 ENNI TD children, served as prompt children
in at least one rotating round. There are zero globally prompt-free ENNI SLI
children. ENNI-only augmentation utility must therefore be evaluated
round-by-round, using no stories from a held-out child in that round and only
synthetic descendants from compatible rounds. Pooled ENNI-only classifier
sweeps are exploratory sensitivity analyses, not confirmatory leakage-free
tests.

The saved external-Gillam classifier result is also not an independent
external evaluation: all 116 Gillam holdout File_IDs are already present in
the pooled real training table. Do not cite that result as external
generalization.

The operational failure is traceable in the runner: its default `REAL_CSV`
was the pooled ENNI+Gillam table, while `--external-holdout-csv` added a test
table without subtracting those children from `real_rows`. The old output also
labelled its threshold source as ENNI even though the out-of-fold predictions
came from the pooled table, and there was no train--holdout ID assertion or
recorded real-input path. Thus the reasonable goals of increasing the SLI
reference sample and creating a separate Gillam test were accidentally
combined in one execution path.

A corrected external-Gillam evaluation was subsequently run without
overwriting the historical artifacts. It fits models on the 235-child
ENNI-only evaluation window, selects thresholds from ENNI-only out-of-fold
predictions, and tests on the same 116 Gillam children; the training/test
File_ID intersection is zero. The same 940-row synthetic table, panels, model
family, folds, and seed were retained. Corrected augmented-minus-real-only AUC
contrasts are -0.003 (all shared), +0.009 (balanced literature), -0.009
(classic clinical), -0.048 (morphosyntax errors), and -0.086 (syntax
complexity). Balanced-accuracy contrasts are small and positive (+0.006 to
+0.032) but threshold-dependent and lack uncertainty intervals. This does not
support a reliable augmentation benefit. The runner now stops if any external
holdout File_ID occurs in its real training input. Corrected outputs and the
invalid-versus-valid comparison are stored under the
`allages_overlap_utility_panels_valid_enni_train_external_gillam*` and
`allages_overlap_utility_panels_gillam_invalid_vs_valid_comparison*` result
prefixes.

The ratio-sensitivity extension is implemented in
`phase2/replications/guo_schneider_6yo_ablation/scripts/run_external_synthetic_ratio_sweep.py`.
It keeps Gillam entirely test-only, makes the ENNI-to-synthetic ratio explicit
within TD and SLI groups, retunes thresholds from ENNI-only out-of-fold
predictions for each ratio, and writes a draw-level synthetic-source manifest.
Ratios above the available source-child count use replacement and must be
interpreted as repeated training exposure rather than extra independent
synthetic children. Its outputs must be given a new result prefix and reported
as retrospective external sensitivity results, not as a prospectively blinded
confirmatory sweep.

The completed external-Gillam sweep used ratios 0, 0.25, 0.5, 1, 2, and 5,
with the ratio defined independently in TD and SLI relative to the ENNI
training count. The zero-ratio baseline reproduced the prior corrected
external estimates exactly. Results were mixed at 0.25: the balanced-literature
panel changed from AUC 0.9022 to 0.9108 and balanced accuracy 0.6857 to 0.7337,
and the classic-clinical panel changed from AUC 0.8881 to 0.8986, but the
all-shared, morphosyntax, and syntax-complexity panels all lost AUC (−0.0091,
−0.0471, and −0.1051, respectively). Higher ratios did not produce a
consistent AUC benefit and ratio 5 reduced AUC in every panel. The full
machine-readable sweep, including ENNI out-of-fold threshold selection,
Gillam predictions, and the draw-level source manifest, has prefix
`allages_external_gillam_synthetic_ratio_sweep` under the replication results
directory. It was executed with:

```bash
python phase2/replications/guo_schneider_6yo_ablation/scripts/run_external_synthetic_ratio_sweep.py \
  --output-prefix allages_external_gillam_synthetic_ratio_sweep
```

This pattern does not establish a reliable augmentation benefit.

The active `clinical_analyses_reordered.Rmd` now defaults to
`phase2/ENNI/measures/severity_features_raw_real.csv`. A pooled-corpus
sensitivity render must be requested explicitly through
`ICL_PILOT_REAL_CSV`; it is no longer the default paper-facing analysis.

The ENNI-only rerun materially changed reference geometry and several
paper-facing estimates. Apparent same-group overlap and coverage often
increased because nearest-neighbor thresholds widened after the SLI reference
sample fell from 139 to 47; these increases are not evidence of improved
generation. The identical-code integrated R rerun also changed PCA feature
retention and source-by-group conclusions for MLU, VOCD-D, and DSS. Full
commands, audits, deltas, warnings, and reports are under
`phase2/replications/guo_schneider_6yo_ablation/results/enni_only_rerun/`.

The completed rerun covers 23 current non-smoke Python evaluations plus paired
pooled and ENNI-only renders of the integrated R analysis. Exact matching
produced 8,468 classifier-configuration comparisons with no unmatched rows.
The ENNI-only scarcity and quantity surfaces show apparent positive
augmentation AUC deltas across all real-data levels, unlike the pooled
surfaces at moderate and full real-data sizes. These deltas are not valid
utility evidence under the pooled Python design because of the prompt-ancestry
problem above. The mechanism audit also changed substantively: tree-model
importance shifted from IPSyn under the pooled reference to bare-marking
errors under ENNI only, and concordance between real-only and augmented model
importance weakened. Use the rerun as evidence that Gillam materially affects
the estimated reference phenotype and evaluation conclusions, not as evidence
that ENNI-only augmentation has been validated.
