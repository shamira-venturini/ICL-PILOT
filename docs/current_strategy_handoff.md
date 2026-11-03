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
- Independent framework-evaluation section 6 in the main notebook and existing
  smoke notebook, with clickable settings/corpus/quality/classification/ratio/export
  blocks. Saved narrative CSVs or prebuilt exports enter the existing shared
  protocol; annotation is not required for text evaluation. All model inference
  and fitting switches default off. The separate evaluation runtime avoids
  changing generation/annotation dependencies. Colab writes incremental results
  to Drive; saved sessions verify source/input hashes and reuse completed fits.
  The original smoke upload has a checksum-pinned support ZIP; original pilot
  generations remain ineligible for scientific evaluation. Local checks use
  controlled predictions; full-study BERT/GPT-2/MiniLM execution remains pending.
  See the [notebook guide](guides/framework_notebook.md).
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

The bounded Qwen3-8B Colab run on 2026-10-30 generated 12 paired-stage jobs
(24 stories) and completed actual Batchalign segmentation. Initial Chatter
validation passed 20 files and rejected four for 15 curly apostrophes (E256).
The converter now records curly-to-ASCII apostrophe formatting before
segmentation, retaining raw generation bytes and hashes. Local replay of the
saved boundary actions, followed by fresh native Chatter validation and shared
preparation, passed all 24 files. This replay does not constitute a new boundary
prediction run. Both the official notebook and the existing smoke notebook now
check the formatting rule before conversion and audit actual CHAT files before
Chatter. The exact old uploaded converter can receive the checksum-pinned
four-line correction, recorded in the run; the smoke bootstrap accepts the
checksum-pinned reviewed source corrections. The same notebook and original bundle remain usable.
See the
[diagnostic record](reproducibility/qwen_smoke_apostrophe_fix.json).

The subsequent Colab smoke run completed TalkTag, with one preservation failure:
TalkTag removed double quotes around “no running”; all words and all 16 main-tier
boundaries were unchanged. Postprocessing version 2 restores original double
quotes alongside case/spacing while retaining strict checks on all other tokens
and final exact source-surface preservation. The saved diff was reproduced
locally and the restored file matched its source exactly and passed native
Chatter. Both notebooks include the source-preservation policy and can recheck
the retained completed TalkTag outputs without inference in a new derived
directory/manifest. Actual Colab rechecking, UD and retention/export remain
pending. See the [quote-restoration record](reproducibility/talktag_quote_restoration.json).

The current user-directed policy is automatic source restoration (version 3).
TalkTag-side word edits, deletions, insertions and changed boundaries no longer
require per-file manual decisions: derived CHAT is rebuilt from the authoritative
pre-annotation source. Tags are copied only from unchanged, uniquely aligned
tokens. Unmapped tags and unassessed source words are logged with partial
annotation coverage; error-feature extraction must carry this missingness,
not interpret it as no errors. All stories/bundles remain present. Both notebooks
include this policy and checksum-pinned upgrades from the original/v2 Colab
helpers. Original CHAT/generation text and raw TalkTag outputs remain unchanged.
See the [automatic-restoration checks](reproducibility/automatic_text_restoration.json).

The 2026-10-31 Colab smoke run completed generation and TalkTag with preservation
QC (24 files, 298 CHI utterances, no unavailable annotation anchors), then stopped
at UD setup because the moving upstream Stanza catalog no longer matched the
reviewed checksum. Catalog and model downloads now use immutable commits matching
the original catalog and all eight original model hashes. The worker directs
Batchalign's startup catalog refresh to that same revision. Both notebooks can
upgrade the original uploaded helpers and resume at UD setup without repeating
generation or TalkTag. Completion of actual Colab UD and final export remains
pending. See the [resource-pin check](reproducibility/ud_resource_pins.json).

The 2026-11-02 Qwen Colab smoke run subsequently completed the entire software
path, including UD, Chatter, retention and ZIP export. Local inspection of the
downloaded archive verified all 24 final CHAT files, 298 tagged utterances, the
retention receipt, exact fingerprints and 57 retained audits. The run has two
unchanged Stage 2 stories and one repetitive Stage 1 story that reached 640
tokens; all remain retained. Concrete segmentation and tagging examples show
that linguistic accuracy still requires attention before interpreting the
domain measures. These are bounded smoke-test findings from one prompt pair,
not study results. See the [inspection report](reproducibility/qwen_smoke_review_20261102.md).

A real ENNI annotation pilot is prepared in the same notebooks. The active smoke
notebook defaults to `RUN_PROFILE="real_enni_pilot"`: six children from the first
three four-year-old roster pairs, 36 narratives and 550 CHI utterances. A separate
verified input upload includes those existing study working files and one
original ENNI ZIP. The route skips generation/segmentation and marks the run
test-only. Local intake, shared cleaning, native Chatter, original-member recovery
and regressions passed; actual fresh TalkTag/UD execution for these six children
is pending on Colab. Gillam remains external classification-only. Preposition
error tags are excluded from the planned analysed error counts; feature-table
implementation is still pending. See the [pilot record](reproducibility/real_annotation_pilot.json).

The six-child real ENNI Colab pilot completed TalkTag and all six UD inference
outputs, then stopped because Batchalign replaced an older source morphosyntax
version comment. Comparison of the saved failed-run archive found all 712
speaker main tiers identical; each file differed only by its tool-version stamp.
The wrapper now restores that specific historical source comment during the
source-preserving merge and records current provenance separately. All six raw
outputs were recovered locally without model inference and passed exact source
fingerprints and fresh native Chatter validation. UD covers 547 of 550 CHI
utterances; the other three are nonlexical. TalkTag restored three changed words
in three utterances and records those three words as unassessed for error labels.
Both notebooks include the reviewed helper upgrade. Historical failed manifests
remain unchanged; the local recovery has its own manifest and outputs. See the
[recovery record](reproducibility/enni_ud_provenance_recovery_20261102.json).

A preliminary real–synthetic annotation comparison inspected a fixed sample of
144 canonical-story utterances, 36 per real TD/DLD and synthetic stage condition.
The review was performed by the AI assistant, with no blinded independent human
gold annotation; it is not an accuracy benchmark. Clear POS/finiteness issues
were identified in three real TD cases and one synthetic Stage 2 case, plus one
synthetic Stage 1 AUX/COP relation issue. Uncertain cases were more frequent in
the fragmented Stage 2 sample. The evidence supports retaining the automated
pipeline with cautious feature interpretation, not a claim of equivalent
annotation accuracy. The existing syntax reader was independently checked and
would skip 100 valid clitic-bearing utterances because it does not expand `~`;
fresh feature extraction must correct that compatibility issue. No annotation
or production feature code was changed by this comparison. See the
[comparison report](reproducibility/annotation_comparison_20261102.md).

The syntax reader now expands `~`-joined morphology components in dependency
order before alignment. Verification on all 30 saved real/synthetic pilot CHAT
files recovers the 100 falsely skipped utterances: all 845 tagged utterances are
analysed, with zero token mismatches. The three nonlexical real utterances remain
without UD analysis. Twelve regression checks cover contractions, finite verbs,
punctuation, graph metrics, unchanged source bytes and genuine invalid/missing
tiers. No text, annotation labels or dependency indices were changed, and no
model inference was repeated. Fresh study feature tables and R analyses remain
pending. See the [reader-fix verification](reproducibility/syntax_reader_clitics_20261102.json).

A bounded framework diagnostic subsequently used the actual saved 24-story
Qwen pilot, all 1,428 ENNI and 543 Gillam stories for input/fold checks, and small
held-out real subsets for actual predictions. GPT-2/MiniLM, five actual one-update
BERT CPU fits, main and ratio summaries, verified completed-fit reuse and compact
export passed. It found and corrected a model-runtime dependency conflict and
the toolkit's requirement for matching train/validation CSV schemas. The package
now receives only text/label columns; metadata remains in provenance audits.
All pilot flags and source hashes remain preserved, and scientific preparation
still rejects pilot generations. These are software checks, not scientific
performance results; full-study GPU execution remains pending. See the
[actual-data report](reproducibility/framework_actual_data_20261102.md).

The main and existing smoke notebooks now also contain independently runnable
KIDEVAL English UD feature extraction. It uses completed final-UD manifests or
processing ZIPs, builds the pinned native CLAN tools, assembles comparable
six-story units and exports combined/separate tables with lineage and coverage.
The supported +lengu script disables IPSyn; no normative comparisons are run.
Native raw counts and the default DSS requirement are retained. Audio links and
timing bullets are removed only from derived canonical-order analysis bundles;
the original final annotations remain hash-verified. Incomplete units remain
flagged rows with missing features. Native error columns are raw diagnostics,
not the study's preposition-excluding error measures. DSS sentence scores are
missing when TalkTag coverage is partial/unknown. Custom error features and R
statistics remain pending. See the [feature guide](guides/domain_features.md).

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
