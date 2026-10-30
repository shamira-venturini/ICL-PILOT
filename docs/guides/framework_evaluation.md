# Framework and classifier evaluation

The study treats descriptive corpus similarity, linguistic fidelity and classification utility as distinct outcomes. Use configs/evaluation/protocol.json and the framework_* modules. The notebook evaluates all three configured generators separately.

## Inputs and common reference

The text exporter requires explicit new-generation CSVs, the canonical ENNI participant/story CSVs and a fresh destination. Gillam is a separate external-evaluation file. Both paths in configs/evaluation/inputs.json must exist before results are created.

The verified Gillam export contains 182 children (91 TD/91 SLI), aged 60--95
completed months. Repeated visits use one normalized child ID and the earliest
eligible recording, checked against sex and date of birth. In the original archive,
a-1-68pre and a-1-68 are visits by the same child; retain a-1-68pre and exclude the
later session from the derived export. Raw sessions remain unchanged. All eligible
unique SLI children are retained; complete TD transcripts
are selected by fixed hash ranking (seed 20260827) within one-year age-cohort by
sex strata. Age and sex come from literal CHI metadata; clinical group comes from
explicit @Types verified against the corpus directory. Empty trailing story markers
are logged and ignored. Retain the available canonical stories for all selected
children: 543 narratives (273 TD, 270 SLI), with three SLI children contributing
two stories. Child predictions therefore average available stories. Source and
selection provenance are in docs/reproducibility/gillam.json.

Validate model/revision/run identities, exact paired-stage hashes, child/story identities and complete schedules across generators. Smoke fixtures are refused. Text preparation is shared across ENNI, synthetic and Gillam: remove notation while retaining grammatical forms, repetitions, case and partial words.

Corpus diagnostics use identical real rows and matched age/group/story sample counts for each generator. They are descriptive reference comparisons, not held-out estimates of clinical fidelity.

Generation [QC review](generation_review.md) is descriptive and review-only.
Flags for token-limit stops, unchanged Stage 2, possible prompt leakage or loops
do not remove stories, break bundles or change classifier sampling. Human notes
are separate artifacts and are not applied as eligibility decisions.

## Classifier splits and budgets

All 238 eligible real children (190 TD, 48 SLI) participate in five age-stratified
classifier folds independent of generation batches. Prompt-pair members are one
group; the 142 unmatched TD children are individual groups. Each fold is used once
for test, the next fold for real validation, and the other three for training.
Every child therefore enters test and validation once and training three times.
There is one ordinary ENNI test set per fold. No separate expanded-test experiment
is implemented. Neither a child nor a prompt pair crosses partitions within a fit.

Eligible synthetic source pools contain only descendants of pairs whose two
ancestors are in real training. Bootstrap complete story bundles with replacement
within age and literal clinical group, targeting the corresponding real training
child counts multiplied by the requested synthetic:real ratio. Complete bundles
preserve the six story slots as well. All three generators use the same source
bundle draws; the same seeded draw stream gives nested prefixes across ratios.
TD/DLD proportions follow the real training fold, rather than 50:50 or 7%.
Different clinical classes can be sampled at different frequencies; original
paired stages and raw generation artifacts remain complete and unchanged.

The primary synthetic:real ratio is 10:1 sampled training stories. Repeated draws
are training exposures, not new narratives or independent children. Save original
sample/bundle identities, bootstrap draw IDs/seeds and unique versus drawn counts.
Use unweighted cross-entropy because resampling already preserves the target
clinical distribution. Validation, ENNI test and Gillam are not training-resampled.

One shared real-only baseline and one shared augmentation-update-matched control
are fitted per fold/seed, plus each generator's synthetic-only and augmented
conditions. Five folds and three seeds imply 120 planned fits. Generation still
uses 18 age-specific batches; classifier folds do not select a single batch.

BERT is the pinned bert-base-uncased checkpoint. The protocol uses 512-token story inputs, batch size 8, AdamW at 2e-5 with weight decay 0.01, unweighted cross-entropy and threshold 0.5 for SLI. Fixed update budgets derive from three epochs of the condition dataset. Seed before loading the random head. Validation monitors training without selecting a best checkpoint. Record truncation, updates and every argument.

The saved BERT length audit in docs/reproducibility/bert_tokenizer_audit/ counts
special tokens and uses the exact framework-prepared story text. Only child 725's
B3 story exceeds 512 tokens among the 1,428 ENNI study stories and the nested
576-story prompt-pair subset (604 tokens, 92 removed). All 543 selected Gillam
stories fit; maximum 460. Retain the existing right-truncation policy and report
this isolated case. Audit definitive synthetic story lengths before fitting; the
real-data check does not establish their truncation rates. Classifier runs also
write tokenization_audit.csv for every train/validation/test input.

## Separate synthetic-to-real ratio experiment

The notebook's optional ratio block uses framework_ratios.py and defaults to
synthetic:real ratios (1, 10), plus a shared real-only baseline. Set
RUN_RATIO_EXPERIMENT=True in that block to fit; its default False writes plans
without starting BERT. It requires the setup/install/import and input-preparation
cells, but not the main classifier, descriptive or text-quality cells.

Keep real training, validation, ENNI test and Gillam rows fixed. Synthetic draws
are nested prefixes of the training-only age/group bundle bootstrap. Every ratio
preserves the real training age/group/story proportions and achieves its exact
requested number of sampled stories. Sampling with replacement permits ratios
such as 10:1 even when the number of unique eligible synthetic stories is lower.
Report unique counts and multiplicities explicitly. Optional integer levels such
as (1, 5, 10) can be prespecified; these are sampled-exposure ratios.

Use the epoch-derived update budget at each ratio and one shared real-only effort
control per ratio/fold/seed. Defaults plan 135 fits in the five-fold, three-seed,
three-generator design. Identical completed main-experiment fits can be reused
after job/backend/checksum verification; after a matching main experiment only
60 additional 1:1 fits remain. Synthetic-only replacement is retained in the main
experiment and is not repeated at every ratio.

All outputs are separate under RESULTS_DIR/ratio_experiment/: an immutable
experiment manifest, combined training plan, selected synthetic identities,
per-fit predictions/audits, story/child summaries, paired differences against the
shared baseline and ratio-specific effort controls, and direct higher-minus-lower
ratio contrasts with the existing conditional cluster bootstrap. A child-level
balanced-accuracy curve shows means across rounds/seeds. Ratios are reported as
prespecified conditions; ENNI test and Gillam do not choose a best ratio.

The block resumes verified completed fits and preserves partial/conflicting
outputs. It reuses the main experiment's folds and sampling policy and saves its
own ratio outputs. Check generated-story BERT lengths before enabling fitting. The normal
Drive/ZIP export includes this subdirectory and its ratio configuration.

## Outcomes and uncertainty

Evaluate held-out real ENNI and independent Gillam at story and child levels. Child probabilities average available story probabilities. Primary child outcome is balanced accuracy; retain AUC, both group recalls, accuracy, macro-F1 and predicted-SLI rate. Report age recall only where that age exists.

Use paired contrasts and conditional 95% cluster-bootstrap intervals with the configured 2,000 draws. Resample ENNI prompt pairs jointly and unmatched TD children individually in separate age-by-unit-type strata; resample Gillam children within age/group. Retain linked repeated predictions under one weight. Intervals condition on fixed fitted models and splits; no bootstrap retraining occurs. Across-round/seed SDs are descriptive. No automatic significance or equivalence claim follows from these intervals.

## Descriptive metrics

- Story word/type counts and type-token ratios use whitespace units.
- Jaccard averages all cross-document whitespace-set comparisons.
- Cosine uses a TF-IDF vocabulary/IDF fitted to the shared real reference; report synthetic out-of-vocabulary behavior.
- Perplexity uses the pinned GPT-2 scorer, its 1,024-token limit and arithmetic mean of story perplexities.
- Fréchet distance uses Gaussian means and covariances of pinned MiniLM embeddings, including the covariance/Bures term. Save embedding arrays and encoder truncation audits.

The local Fréchet implementation follows the Gaussian moment definition; implementation details belong in the supplement/code. IG, MAUVE, topic modeling and privacy analysis are excluded.

## Reproducibility

Write fresh result directories. Save protocol/code/backend/data hashes, common samples, split/count audits, budgets, predictions, bootstrap validity and provenance. Completed jobs are reused only after hash verification; changed plans and partial fits are refused rather than overwritten. Full scientific fitting and results remain pending.
