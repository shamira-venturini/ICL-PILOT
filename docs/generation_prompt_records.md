# Generation prompts and inference records

`configs/generation/prompts/stage1.json` and `stage2.json` define the maintained
two-stage protocol. Each file contains system/user messages as lists of text
lines; joining those lines with newline characters fixes the exact message
whitespace, including the trailing newline. The initial extraction preserved
the pre-cleanup notebook at Git commit `78a5cada`. The author subsequently
approved the neutral wording below for the next smoke test on 15 October 2026.

## Approved neutral wording — 15 October 2026

Stage 1 asks for a new narrative from the supplied events, guided by the example's
language and narrative style. Stage 2 asks for a rewrite guided by its example,
preserving the same main events and their order. The model-facing instructions
contain no ENNI/TD/SLI/DLD labels or requests for child-like/simple language,
shorter clauses, adult-style avoidance or clinical plausibility.

All six stories' inserted `prompt_constraints` now specify event order and no
new plot events. The event units, titles, content invariants, optional content
and human exemplar transcripts are unchanged. References to children/adults as
story characters remain. Internal stage names, source groups and journal
metadata retain their provenance labels; placeholder names resolve to story IDs
or text before inference and are not sent as names to the model.

Templates are hashed and snapshotted per run. The scaffold configuration is
hashed, and its rendered content is retained in the actual prompt records.
Existing runs, their template snapshots and generated/annotated files remain
unchanged. After updating an existing Colab checkout, restart the kernel and run
the notebook from the top with a fresh run ID. The preview now shows the complete
Stage 1 prompt and both template hashes. No additional evaluation experiment is
introduced by this wording update.

## Historical preservation audit and its limits

Before this wording update, direct rendering checks matched the notebook
versions at `f2e61695`, `78a5cada` and `01cace79`. The 7 April `f2e61695`
prompt update removed explicit SLI/profile references from Stage 2 and some TD
wording from Stage 1, but retained two TD headings and `clinically plausible`.
The prior preservation audit establishes equality to that historical notebook;
it does not establish the exact wording executed for the evaluated data, and it
does not describe the current neutral templates. Do not describe historical
generations as label-free without evidence of the actual messages used; new
runs record those messages in `prompt_records.jsonl`.

`experiment.json` schema 4 identifies these files. `PromptBuilder` loads them
once during preflight, validates their named fields, and supplies messages to
both preview and inference cells. It inserts exemplar text and Stage 1 parent
text literally; braces inside those texts are not interpreted as templates.
It rejects same-set exemplar slots, missing inputs and empty Stage 1 parents.
Preflight also verifies each selected exemplar's raw age against the frozen roster.

## What is preserved

Story content, exemplar choices, replicate ID in Stage 1, child pairs and
cross-story rules are preserved. Age remains implicit in exemplar
selection. Tokenizer chat-template arguments, tokenization defaults, sampling
parameters and decoding behavior are unchanged. No model was run locally to
claim numerical or output equivalence across GPU/software environments.

The initial compatibility audit compared all 2,820 job message pairs with the
pre-cleanup notebook functions, using saved historical Stage 1 parents for
Stage 2. These historical checks predate the approved wording change. Hashes
and verification are stored under
`outputs/maintenance/prompt_builder_2026-10-14/`.

## Records from a new run

`prompt_records.jsonl` is an append-only UTF-8 JSON Lines file:

- A `run` record captures the settings/input-hash manifest, runtime information
  available at journal creation, and template hashes.
- A `prompt` record is flushed before a model call. It contains its record ID,
  job/round/pair/replicate/story identities, exemplar IDs and ages, system/user
  messages, exact chat-template text passed to the tokenizer, input token count
  and token-ID hash, generation parameters, and template hash.
- An `output` record links to that prompt and stores the returned narrative,
  text hash, generated-token count and an empty-output flag.

Input token IDs are hashed using their compact JSON list representation. The
full input token list is not duplicated in the file; messages and rendered chat
text are retained. Reconstruction requires the matching tokenizer/runtime;
resolved revisions are recorded when provided by the loader. The reconstructed
token sequence can then be checked against the hash.

For Stage 2, the recorder requires a completed Stage 1 record with identical job
metadata and parent text. It stores the Stage 1 record ID and parent-text hash.
An empty Stage 1 result remains an output artifact but cannot be transformed.
A failed model call leaves its attempted prompt recorded without a fabricated
completion. Duplicate prompt/output writes are rejected.

The generated CSV includes `stage1_prompt_record_id` and `stage2_prompt_record_id`.
It also records `test_only`, which is true for smoke tests. Its run
manifest distinguishes the full planned study design from the selected execution
schedule; smoke outputs are not study samples.
`run_manifest.json` identifies the journal and hashes the templates and relevant
source code. Exact template bytes are copied into `generation_design/prompts/`
as `stage1_td.json` and `stage2_sli.json`.

## Export and current limits

The combined generation/annotation notebook exports the growing journal, run
manifest and generation design to the run's Drive root at cohort boundaries.
Conversion and annotation run as separate blocks and can start from saved
inputs in a fresh session. The final archive includes all local run artifacts,
including CHAT files, annotation logs/reports and QC, with relative paths.
The generation CSV now also carries the TD source age explicitly.

Local journal writes are flushed and synced, but this is not yet an inference
resume mechanism or a guarantee of recovery after loss of a Colab runtime.
Existing journals and generation CSVs require a fresh run directory. Persistent
checkpoint timing, restart/random-state recovery, dependency/model pins and
complete GPU validation remain separate execution work. The combined interface
uses one builder for preview and inference and records the actual messages.
See `colab_generation_package.md` for current processing controls.
