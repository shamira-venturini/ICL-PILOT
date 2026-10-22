# Frozen prompt builder and inference records — 14 October 2026

The user authorized moving the existing Stage 1/Stage 2 templates into
configuration, using one prompt builder in Colab, validating all jobs, and
recording the actual inference prompts with their output ancestry.

## Sources of truth

- `configs/generation/prompts/stage1.json` and `stage2.json`: the original
  system/user message text, stored as readable lists of lines.
- `configs/generation/experiment.json` schema 4: template locations and the
  unchanged model, decoding, roster, scaffold and corpus-input settings.
- `src/icl_pilot/generation/prompts.py`: validated, immutable loaded templates
  and the shared builder used by Colab previews and inference.
- `src/icl_pilot/generation/prompt_records.py`: an append-only journal of
  prompts, actual rendered chat text, token hashes/counts, outputs and parents.

Template text, whitespace, the Stage 1 replicate ID, age-through-exemplar
conditioning, and the existing tokenizer/generation/decoding calls are preserved.
Preflight now also checks selected exemplar raw ages against the frozen roster.
Template bytes and code hashes are included in the run metadata. Template
snapshots are copied to each run's `generation_design/prompts/` directory.

Each new inference attempt writes its prompt before calling the model, then
records the returned text after decoding. Stage 2 requires a completed Stage 1
record with matching cohort, round, pair, replicate, story, exemplar metadata
and parent text. Output CSVs store the two prompt-record IDs. Empty outputs
remain explicit artifacts; an empty Stage 1 output cannot be transformed.

The notebook includes `prompt_records.jsonl`, `run_manifest.json` and the
generation design in exports. Drive copies use the run root for shared metadata
and cohort folders for cohort outputs; final archive entries use relative paths.

## Verification

- 74 focused tests passed on Python 3.8, with three pre-existing dependency warnings.
- All 2,820 jobs, or 5,640 stage prompts, match the original notebook functions
  byte for byte. Stage 2 comparisons use actual saved historical Stage 1 texts.
- Full generation input rows and the rotating evaluation manifest retain their
  pre-change fingerprints. No new child pairing or prompt design was introduced.
- Golden-message tests cover literal inserted braces and message whitespace;
  negative tests cover malformed templates, same-set exemplars, wrong/unfinished
  parents, changed parent text, round/replicate mismatches and duplicate writes.
- Mock inference checks that the recorded chat text is exactly what the notebook
  passes to the tokenizer. A mocked notebook loop checks CSV record links and
  that the final archive contains the journal and template snapshots.
- A failed model-call fixture records an attempted prompt without inventing output.
- 6,036 protected research files, the corrected generation-input CSV and all
  three R source files were rehashed unchanged.
- Notebook syntax, CLI layout validation and `git diff --check` passed.

`all_job_prompt_hashes.csv`, `validation.json` and `validation_commands.json`
contain the complete comparisons and test logs. The original prompt functions
are retained as a historical oracle in this audit, not as active generation code.

## Backup and remaining execution work

Verified pre-change backup:

`/Volumes/TranscendBU/ICL-PILOT_78a5cadad659_20261014T134815Z/GenerationPrompts/generation_prompts_snapshot.tar.gz`

SHA-256: `338180a02e42711e07eea0c9b9ad266f3bede33c4957d712838686fd02df6a09`.
All members were decompressed and hashed against their source files.

The recorder is not an inference resume mechanism. Fresh run directories are
still required; frequent persistent checkpoints, recovery of sampling state,
restart logic, tested dependency/model pins and an actual Colab GPU trial are
the next execution tasks. Runtime completion-status bookkeeping also remains
part of that later execution review. No actual model inference, annotation
regeneration or scientific analysis was run, and no data/results were rewritten.
No changes were staged, committed or published.
