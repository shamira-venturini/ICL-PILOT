# Definitive study status

This file describes the current study only. Configuration, source data and implementation must agree with it.

## Data and generation

The verified ENNI reference comprises 238 children aged 4–7: 190 TD and 48 SLI-labelled. The canonical participant table, prepared story CSV and pair roster are identified in [generation configuration](../configs/generation/README.md). Original CHAT and the supplied ZIP remain unchanged.

There are 48 within-cohort age-matched pairs, 11/13/11/13 across ages four–seven, and 18 generation batches. Each pair supplies six target stories and ten replicates. Each of the three models produces 2,880 stories per stage and 480 six-story bundles per stage; total generation is 17,280 individual stories.

The three pinned models are Llama-3.1-8B-Instruct, Qwen3-8B with thinking disabled and Gemma-2-9B-IT. Each Stage 2 rewrites its own model's exact Stage 1 output. Opposite-set exemplars and reversed assignments balance story roles. Both stages use temperature 0.8, top-p 0.9, top-k 0 and 640 new tokens. Seed 42 initializes each model run once before loading.

This is one practical generation procedure. Stage effects combine rewriting and clinical exemplar conditioning. No extra generation-control experiments are in scope.

## Implemented

- Verified ZIP intake, demographic cross-check, complete participant identities and completed-month ages.
- One authoritative prepared exemplar CSV and roster, with source hashes and extraction QC.
- Native model adapters, full prompt records, parent hashes, run-level RNG recovery and immutable run directories.
- Review-only generation QC with an offline searchable page, full paired-stage
  text, flag evidence, CSV/JSON reports and separately downloadable notes.
  The review_only route uses saved generation CSVs without model loading.
  Token-limit stops use verified journal records; missing stopping reasons remain
  unknown. All stories, bundles and downstream sampling stay unchanged.
- Whole-narrative Batchalign segmentation, pre/post annotation Chatter gates and preservation checks.
- Common deletion-only preparation before fresh TalkTag for real ENNI and all
  synthetic collections together. A labelled from_chat selection includes real
  ENNI once; prepare_only runs cleaning/validation without annotation models.
  Temporary original snapshots are used for synthetic/unspecified collections. Real source
  files are referenced by file identity, checksum and available CHAT PID without
  duplicate original backups. Per-file deletion audits, prepared working files,
  retained transcription codes and collection metadata remain checked through
  the TalkTag-to-UD handoff. Keep one checksum-pinned original ZIP per real corpus
  for recovery, without expanding or copying it into every run. The final notebook
  retention block verifies the complete chain and source-to-final utterance/header
  fingerprints before removing verified successful-run intermediate CHAT.
  Final annotated CHAT, fingerprints, audits, manifests and diagnostics remain;
  failed/incomplete runs retain their working files. The final receipt supports
  relocated-run verification and exact original-member recovery. Selected ENNI
  study inputs remain available for fresh processing. ZIP-backed ENNI view/Gillam
  export reproduction avoids redundant expanded originals.
- Independent Batchalign 3 English UD annotation block after TalkTag, plus a
  morphosyntax_only route for saved real/synthetic CHAT. Pinned package/model
  resources, preserved tokenization/boundaries, fresh %mor/%gra for all speakers,
  separate CHI coverage, immutable snapshots and final Chatter/preservation gates.
  Native smoke checks do not constitute full-corpus or Colab execution.
- Shared framework text preparation and all-child classifier cross-validation:
  five age-stratified folds across 190 TD and 48 SLI children, grouping prompt-pair
  members and treating unmatched TD as individual child groups. Each fold is test
  once, next-fold validation once, and training in the other three folds.
- Training-only synthetic bundle bootstrap matched to each real fold's age/group
  proportions. All generators share draws; lower-ratio draws are nested prefixes.
  Repeated exposures and unique narratives are logged separately. Loss is unweighted.
- Fixed-budget BERT jobs, unchanged Gillam evaluation, child aggregation and
  conditional uncertainty resampling of prompt pairs/unmatched TD children.
- Separate opt-in ratio block in synthtexteval_demo.ipynb: 1:1 and 10:1 augmentation,
  shared real baseline and ratio-specific update-matched controls, fixed real splits,
  nested synthetic draw prefixes, paired ratio contrasts and a child-accuracy curve.
  Its default is plan-only; completed identical main fits can be reused. The
  standalone plan has 135 fits, or 60 extra fits after a matching 120-fit main experiment.
- Descriptive lexical metrics, pinned perplexity and Gaussian Fréchet distance.
- Pinned BERT token-length audit: after framework preparation, one of 1,428 real
  ENNI stories (also one of 576 in the prompt-pair subset) exceeds 512 tokens;
  all 543 selected Gillam stories fit. Child 725/B3 has 604 tokens and would lose
  92 content tokens. Counts include special tokens. Synthetic length audits remain
  pending generation; no classifier fitting was performed by this audit.
- Canonical ENNI identity/age guards for the R feature inputs.
- A separate input directory containing the 238 verified study transcripts.
- Verified original Gillam intake and external-only export: 182 children (91 TD,
  91 SLI), aged 60--95 completed months, matched within one-year age-cohort by
  sex. Repeated visits a-1-68pre/a-1-68 count as one child, retaining the earliest
  eligible recording. The 543 available narratives include three SLI children with two stories.
  The external input is configured; ENNI/generated combined text remains pending.

The definitive study excludes IG, MAUVE, exploratory topic modeling and privacy analysis.

## Remaining scientific work

1. Complete full-model GPU execution checks and generation for all three models; inspect effective decoding settings and stopping/truncation records.
2. Review generated utterance boundaries against the intended ENNI unit convention. Real CHAT boundaries remain the corpus reference.
3. Complete TalkTag/Batchalign UD processing with source-preservation QC and
   construct fresh compatible linguistic feature tables. Traditional MOR/MEGRASP
   and IPSyn are not run by the annotation notebook. R analyses remain separate.
   Original quotation/completion markers and boundaries are preserved. Optional
   strict cross-utterance linker checks are outside the study outcomes and are
   not enabled; standard structural and alignment checks remain required.
   All 238 original and all 238 prepared ENNI files pass those required checks.
4. Implement the domain-specific statistical revisions, including dependence from unique real children and repeated synthetic ancestry, generator separation and paired stages.
5. Run the framework classifiers and corpus metrics on the definitive inputs.
6. Update the manuscript Methods, results, tables and supplement from the implemented analyses.

The R notebook contains analysis code without empirical interpretation text. Feature inputs and statistical review remain pending. No definitive scientific results have been produced by software checks.

## Sources of truth

- [Generation](../configs/generation/experiment.json), [model pins](../configs/generation/models.json), prompts and story scaffolds.
- [Framework protocol](../configs/evaluation/protocol.json).
- [Canonical R metadata guard](../configs/analysis/enni_metadata.R).
- [Reproducibility records](reproducibility/README.md).

The active repository contains only this study's inputs, source, guides and reproducibility records. A recovery snapshot is stored externally; its location is in docs/reproducibility/external_archive.json.
