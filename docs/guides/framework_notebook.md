# Framework evaluation in the main notebook

Use section **6. Framework-based evaluation** of
`notebooks/colab_direct_generation_pipeline.ipynb`. The same section is present
in the existing smoke notebook. Its outline links to settings, corpus metrics,
text quality, classification, the separate ratio experiment and export.
R analyses remain in their own notebook.

## Start from saved text

1. Run the repository cell near the top, then jump directly to section 6.
   For the original smoke upload, run its bootstrap first.
2. Set `RUN_FRAMEWORK_EVALUATION=True`. Choose either
   `EVAL_GENERATION_CSVS` or `EVAL_TEXT_RECORDS_CSV`.
   With generation CSVs, supply one selection for each of `llama3_1_8b`,
   `qwen3_8b` and `gemma2_9b`. Each selection can contain an aggregate CSV or
   the four cohort CSVs; do not select both. Paths may be outside the checkout,
   including on mounted Drive. The prebuilt-export alternative must use the
   existing common framework text policy and full model/parent provenance.
3. Check `EVAL_GILLAM_CSV`. The default is the verified 182-child, 543-story
   external export. Generation and annotation are not prerequisites for reading
   this existing export. Gillam stays external test-only.
4. Run the support, session, runtime and input-check cells. For the original
   smoke upload, also upload `ICL-PILOT_framework_support_20261102.zip` from the
   smoke handoff folder. Its checksum-pinned support cell installs the current
   evaluation sources and derived reference text inputs, while preserving
   unknown modifications. The original smoke ZIP remains unchanged.
5. Inspect the split audit and training plan, then run the desired blocks.

The session validates the canonical ENNI reference text, frozen prompt pairs,
all six targets, all configured replicates, both stages, model identities,
parent ancestry and shared generator schedules. Pilot/smoke generations are
refused. Review flags remain review-only, including prompt-like meta responses.
No stories or bundles are removed by evaluation setup.

## Independent blocks

| Block | Controls | Execution |
|---|---|---|
| Corpus metrics | `RUN_EVAL_DESCRIPTIVE` | Length, lexical diversity, Jaccard and shared-real TF–IDF cosine; no model inference. |
| Text quality | `RUN_EVAL_TEXT_QUALITY` | Pinned GPT-2 perplexity and Gaussian MiniLM Fréchet distance. |
| Main classifiers | `RUN_EVAL_CLASSIFICATION` | Existing five-fold, three-seed BERT plan, with shared controls and training-only synthetic bootstrap. |
| Main summaries | `RUN_EVAL_SUMMARIES` | Summarise completed main fits without fitting; also runs after enabled main fitting. |
| Ratio plan | `PREPARE_EVAL_RATIO_PLAN` | Plan the prespecified ratio levels without fitting. |
| Ratio fitting | `RUN_EVAL_RATIOS` | Separate ratio experiment; reuse only exactly matching verified main fits. |
| Ratio summaries | `RUN_EVAL_RATIO_SUMMARIES` | Tables, paired conditional intervals and exported child-accuracy curve; also runs after enabled ratio fitting. |

The entire section is disabled by default. BERT, quality-model inference and
ratio fitting are each disabled separately. When evaluation is enabled, the
input checks and main training plan precede analysis. Set
`INSTALL_FRAMEWORK_TOOLKIT=True` before model-based blocks. A separate virtual
environment retains the existing pinned SynthTextEval revision and tokenizer
compatibility repairs, with Transformers 4.57.6. The notebook's generation and
annotation environments remain available without a kernel restart. Colab
inherits its CUDA-enabled PyTorch; local base runtimes use isolated packages.
Package versions, source hashes and runtime checks are retained with results.
Model setup resolves Transformers 4.57.6, datasets 4.0.0,
sentence-transformers 5.2.2 and huggingface-hub 0.36.2 together. This prevents
newer datasets/encoder packages from requiring Hub 1.x alongside Transformers
4.x, which requires Hub <1. The original and previously reviewed support uploads
can receive this exact versioned update; unknown source edits remain protected.

## Persistence, resume and export

Colab defaults to writing results incrementally under
`/content/drive/MyDrive/ICL-PILOT/framework_based_evaluation/`.
Local runs default to `results/framework_based_evaluation/` in the checkout.
`EVAL_OUTPUT_ROOT` selects another parent directory. Each new session has a fresh
timestamped directory and `notebook_session.json` containing selected paths,
ratios and input/source checksums. Generation and annotation run directories are
independent of it.

To return later, set `EVAL_RESUME_DIR` to that directory, rerun the section's
setup/input cells and jump to the required block. Resume uses the saved input
selection and ratios. Recreate the same checkout/support revision and make the
original inputs available; changed data or implementations require a fresh run.
Verified completed classifier fits are reused. An interrupted partial fit is
preserved and requires the existing explicit recovery procedure or a fresh run;
the integration does not claim optimizer-checkpoint continuation.

An extracted portable export can also be resumed in a new directory. Session
relocation maps checkout-owned input paths to the current checkout, verifies
their original hashes, records old/new roots and uses the restored output
directory. External input paths still need to be accessible.

The final cell creates a compact ZIP, verifies every member against its checksum
manifest, and optionally copies it to Drive/downloads it. Prepared text, plans,
splits, bootstrap membership, completed predictions, metrics, figures and
provenance are included. Model/trainer directories remain on disk and are
omitted from the ZIP. No runtime disconnection is automatic.

The original integration tests used controlled classifier outputs. A subsequent
[actual-data diagnostic](../reproducibility/framework_actual_data_20261102.md)
completed GPT-2/MiniLM inference, five actual bounded BERT CPU fits, ratio reuse,
summaries and verified export using the saved Qwen pilot and real ENNI/Gillam.
This tests software execution with one update per fit, not model performance.
The full scientific mode continues to reject pilot data. Full-study CUDA fitting
remains pending. The scientific protocol is described in
[the framework guide](framework_evaluation.md).

The repeatable diagnostic entry point is `icl_pilot.evaluation.real_data_smoke`.
It requires the exact reviewed pilot archive and a separate output directory;
its actions are inputs, descriptive, quality, classifiers, summary, ratio-plan,
ratios, ratio-summary and export. Explicit test-only flags are required for its
bounded CPU backend and shortened budgets. The production backend retains its
single-CUDA-GPU requirement and normal training budgets.
