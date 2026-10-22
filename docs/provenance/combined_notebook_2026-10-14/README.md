# Combined generation and annotation notebook — 14 October 2026

The user requested one processing notebook with separate blocks for generation
and annotation, while keeping Python notebook evaluation and R Markdown separate.

## Maintained implementation

- `notebooks/colab_direct_generation_pipeline.ipynb`: one 20-cell interface with
  common setup, generation, conversion, error annotation and QC/export blocks.
  Modes: `full`, `generation_only`, `from_csv`, `convert_only`, `from_chat`.
- `src/icl_pilot/generation/config.py`: optional positive replicate override,
  recorded/hashed effective settings and execution-source hash. Frozen default
  study settings, child pairs, scaffolds and prompt templates are unchanged.
- `src/icl_pilot/generation/execution.py`: sequential frozen-order generation,
  exact prompt/output journal, cohort/aggregate CSVs and status bookkeeping.
  It preserves the global random stream; no per-job seeds/resume are introduced.
  New CSVs also retain the TD age explicitly.
- `src/icl_pilot/pipeline.py`: route selection, saved-input conversion, input
  overlap checks, block records, verified persistent copying and complete exports.
- `src/icl_pilot/annotation.py`: one TalkTag process for all selected inputs,
  copied/unfolded working CHAT, runtime/command/source/output provenance and QC.
- `configs/annotation/talktag.json`: immutable upstream source commit/options
  for new runs. Annotation dependencies use a separate environment; existing
  generation dependencies are not upgraded by the annotation block.

The old standalone TalkTag notebook is removed after verifying it matches the
pre-change archive. Its original SHA-256 is
`b208c79ce1e80fdaae783e02e6be04f7372a101d1a948eae6f04df7e06dd5d7c`.
No other existing project files were removed. Reusable helpers remain Python 3.8
compatible; executing TalkTag requires its Python 3.10+ runtime.

## Validation and limits

- 143 focused tests passed, with three existing dependency warnings. The full
  notebook sequence and saved-CSV/saved-CHAT routes were run in fresh namespaces
  with mocked inference/CLI. Annotation is invoked once per selected collection.
- Negative checks cover textual drift, lost tiers, missing/unexpected outputs,
  insufficient coverage, source revision mismatch, existing outputs, duplicate
  inputs, destination ownership and interrupted Stage 2 lineage.
- Annotation debug limits remain explicitly partial. Tool `annotated_lines`
  counts changed utterances, so coverage instead uses target-line counts.
- `validate_design.py` compares 5,640 default messages against the prior audit,
  using actual historical Stage 1 parents. Default input schedule and evaluation
  manifest hashes match. The inference function's Python AST matches the previous
  function after normalizing the new explicit journal parameter.
- One/two replicate overrides produce 282/564 jobs without modifying study inputs.
- 29,598 protected files remain byte-identical: corpus/generated data, operational
  configuration assets, Python evaluation notebook, generation helpers outside
  this task, and all three R sources. The initial broad protected inventory also
  contained `configs/generation/README.md`; the validator explicitly classifies
  its intended guide update separately. It does not exempt operational CSV/JSON.
- CLI layout, notebook schema/Python syntax and whitespace checks pass.

Actual CUDA generation, TalkTag model inference, CLAN and scientific analyses
were not run. No dependencies/models were installed locally. The source pin is
not evidence of the historical TalkTag version or a tested complete dependency/
model lock. GPU integration, frequent persistent generation checkpoints, sampling
state recovery and per-job resume remain pending. Starting from saved inputs is
supported independently of inference resume. QC does not validate clinical tags.

## Audit and backup

`backup_verification.json` records the 14-file pre-change archive on TranscendBU.
`guide_backup_verification.json` adds the configuration guide's pre-change bytes,
verified against their inventory hash and the earlier completed prompt snapshot.
`original_notebook.json` is a historical comparison oracle, not another active
notebook. `upstream_talktag/` records inspected source at the selected commit.
`build_notebook.py` emits an apply_patch; edits were applied through that tool.

`validation.json` contains design/hash results; `validation_commands.json`
contains focused tests and CLI/whitespace logs. `completed_work_backup.json`
identifies the verified post-change snapshot and copied audit. No changes were
staged, committed, pushed or published.
