# Framework actual-data diagnostic, 2 November 2026

All nine diagnostic blocks passed using the saved Qwen pilot and actual real
reference text. This is a bounded software test, with no scientific performance
interpretation. Full-study generation/evaluation and CUDA execution remain
separate work.

## Inputs and executed paths

- The unmodified reviewed Qwen archive supplies all 24 synthetic narratives
  (12 per stage, one four-year-old pair and two replicates). Original `test_only`
  flags, text and ancestry hashes remain present.
- All 1,428 canonical ENNI stories from 238 children and all 543 selected Gillam
  stories from 182 children were read and checked. The actual all-child grouped
  ENNI folds were constructed from the definitive 48-pair roster.
- The diagnostic uses one fold in which both pilot ancestors are in training.
  Its four-year-old training subset contains 198 narratives from 33 real children
  (162 TD and 36 SLI-labelled stories). Synthetic bootstrap counts preserve this
  real training distribution.
- Bounded validation and ENNI test each use one distinct held-out prompt pair:
  two children and 12 narratives per partition. The Gillam prediction check uses
  two children and six narratives, with no external text entering training.
- Corpus diagnostics use all 24 synthetic narratives and 24 real narratives
  sampled by the existing age/group/story matching function.

The exact scientific notebook input route was also exercised. Canonical real
checks passed, then it correctly rejected the pilot with
`Software-fixture generation cannot enter scientific evaluation.` No pilot flag
was removed to make a scientific run pass.

## Actual computations

Passed actions: inputs, descriptive, quality, classifiers, summary, ratio-plan,
ratios, ratio-summary and export.

Length, lexical diversity, Jaccard and shared-real TF–IDF metrics completed.
The pinned GPT-2 checkpoint produced perplexity values, and the pinned MiniLM
checkpoint produced embeddings and Gaussian Fréchet distances. Values were
checked for finiteness. All source narratives were retained.

Five unique BERT classifiers actually trained on CPU using the pinned BERT
checkpoint, one seed and exactly one optimizer update per fit. This exercises
the same toolkit model loading, tokenization, Trainer, validation, probability
prediction and audit implementation used by the CUDA backend. Diagnostic
controls explicitly restrict shortened budgets to test-only runs; scientific
budgets and the CUDA requirement remain unchanged.

Main-condition execution generated 72 story predictions. The separate 1:1 and
10:1 ratio experiment generated 90 prediction rows, including its baseline and
verified fit reuse. Story/child aggregation, age recall, conditional paired
bootstrap summaries and the ratio plot completed. Identical one-update real-only
and update-matched controls produced matching probabilities, as expected.

The portable export passed ZIP CRC and all 164 result-file checksums. All input
and implementation hashes in the successful diagnostic manifest still match.
Original corpus archives, raw generations and annotation outputs were unchanged.

## Errors found and corrected

1. **Incompatible model-runtime dependencies.** Unconstrained upstream
   requirements selected datasets 5.1.0 and sentence-transformers 6.2.0, requiring
   newer Hub/Transformers releases. Pinning Transformers afterward left an
   incompatible environment and caused a `BucketNotFoundError` import failure.
   The installer now resolves Transformers 4.57.6, datasets 4.0.0,
   sentence-transformers 5.2.2 and huggingface-hub 0.36.2 in one transaction.
   Installation/import verification and `pip check` passed.
2. **Different CSV schemas across classifier splits.** Real validation and
   synthetic bootstrap frames have different provenance columns. Passing all
   columns to the toolkit's HF CSV reader caused `DatasetGenerationCastError` in
   synthetic-only training. Package input CSVs now contain just `output_text`
   and `Label` in both splits. Complete provenance remains in source, split,
   bootstrap, job and tokenization audits. Every actual training condition then
   completed.

Earlier failed diagnostic outputs/logs are retained separately from the successful
`framework_actual_data_v3` run. The support ZIP and both existing notebooks were
updated with the fixes. Upgrade from the previous reviewed support ZIP was
verified; unknown modifications are still preserved.

## Truncation and limits

The MiniLM audit records two truncated real comparison stories and one synthetic
story at its 256-token limit. The BERT training audit contains one unique
synthetic story above 512 tokens (643 including special tokens); it was truncated
under the existing policy. No bounded validation/ENNI-test/Gillam-test story
exceeded BERT's limit. The tokenizer's uncapped length-audit warning was expected;
model inference itself completed.

The diagnostic has one generator, one prompt pair, one classifier seed and one
update per fit. It cannot estimate classification utility, annotation accuracy,
distributional fidelity or uncertainty for the study. Its small bootstrap
intervals and metric values must not enter the paper as results. Local CPU
training does not verify Colab CUDA availability, GPU memory or long-run fitting.

The regression suite passed 121 checks. Its one noninteractive plotting warning
comes from an existing standalone notebook test. Machine-readable identities,
hashes, counts and artifacts are in [the execution record](framework_actual_data_20261102.json).
