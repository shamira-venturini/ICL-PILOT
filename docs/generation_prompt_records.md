# Frozen prompts and inference records

`configs/generation/prompts/stage1.json` and `stage2.json` define the existing
two-stage protocol. Each file contains system/user messages as lists of text
lines; joining those lines with newline characters preserves the original
message whitespace, including the trailing newline. The texts were extracted
from the original notebook, rather than rewritten.

`experiment.json` schema 4 identifies these files. `PromptBuilder` loads them
once during preflight, validates their named fields, and supplies messages to
both preview and inference cells. It inserts exemplar text and Stage 1 parent
text literally; braces inside those texts are not interpreted as templates.
It rejects same-set exemplar slots, missing inputs and empty Stage 1 parents.
Preflight also verifies each selected exemplar's raw age against the frozen roster.

## What is preserved

The original instructions, scaffold fields, exemplar choices, replicate ID in
Stage 1, and exact whitespace are preserved. Age remains implicit in exemplar
selection. Tokenizer chat-template arguments, tokenization defaults, sampling
parameters and decoding behavior are unchanged. No model was run locally to
claim numerical or output equivalence across GPU/software environments.

All 2,820 job message pairs are compared with the original notebook functions.
Stage 2 comparisons use the actual saved historical Stage 1 outputs. The check
therefore covers both stages with real prompt inputs, not only a placeholder
parent. Hashes and verification are stored under
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
checkpoint timing, restart/random-state recovery, dependency/model pins and a
real GPU trial remain separate execution work. Extracting the prompts did not
change their content; the later combined interface uses the same builder and
records. See `colab_generation_package.md` for current processing controls.
