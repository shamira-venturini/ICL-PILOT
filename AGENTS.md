# ICL-PILOT working instructions

## Scope and authority

This repository contains one definitive ENNI study. Superseded datasets, rosters, draft reports and comparison logs belong on the external archive, not in the active repository or paper. The recovery location is recorded in docs/reproducibility/external_archive.json. Never restore an archived variant as a study default.

Instruction order:
1. The human user's current directions.
2. docs/current_strategy_handoff.md.
3. The model, prompt, scaffold, roster and input files named by configs/generation/experiment.json.
4. Current implementation and run manifests.
5. Workflow guides and current source/validation records.

Preserve unrelated working-tree edits. Inspect git status before changes. Do not stage, commit, push or rewrite Git history unless requested.

## Scientific purpose

Assess generation fidelity, developmental and clinical pattern preservation, and classification utility separately. A plausible story does not establish a clinical phenotype; classifier accuracy does not establish linguistic fidelity.

The staged procedure is one practical method, not a demonstrated improvement over direct generation. Stage 1–Stage 2 differences combine rewriting and exemplar conditioning. Additional direct-DLD and TD-to-TD generation experiments are outside the current scope.

Use DLD for the research construct, SLI for literal corpus labels, and TD for controls. Generated outputs are TD-conditioned or SLI-conditioned narratives, not diagnosed children. Do not infer clinical labels from linguistic measures, directory names or model behavior.

## Definitive data and design

- Raw source: data/raw/enni/2026-10-28/ENNI.zip and its unmodified members.
- Participant authority: configs/generation/inputs/enni_participants_2026-10-28.csv.
- Prepared text: configs/generation/inputs/story_unit_narratives_enni_2026-10-28.csv and its parsing QC.
- Roster: configs/generation/master_frozen_roster_enni_2026-10-28.csv.
- Annotation inputs: data/inputs/enni_verified_2026-10-28/chat_4_to_7/.

The ZIP contains 361 CHAT files. Complete age/group metadata and unique usable IDs yield 360 participant records. TD/B/523.cha lacks CHI age/group and is excluded; fully specified SLI 523 is included and independently identified in the original demographics table. The study window is 48 <= completed months < 96: 238 children, 190 TD and 48 SLI.

Age cohorts use completed CHAT years/months; retain raw days and never round into a new age year. Days/30 is used only for fine within-cohort matching. Sex follows the original demographic table, with literal Sex_CHI retained. Working-copy sex-header adjustments are recorded; raw files remain unchanged.

All 48 study-age SLI children are paired once with unique TD children by within-cohort Hungarian age matching. Cohort pair counts are 11/13/11/13, yielding 18 batches of at most three pairs. Six stories and ten replicates produce 2,880 stories per stage per model, 480 bundles per stage per model, and 17,280 stories across three models. Six-story bundles and stochastic repetitions are not independent human participants.

## Generation protocol

Use the three exact checkpoints/revisions in configs/generation/models.json. Qwen thinking is disabled. Apply native chat templates, fold the shared system sentence into the user message for every model, and use each model's own Stage 1 parent for Stage 2.

Keep temperature 0.8, top-p 0.9, top-k 0 and a 640-new-token ceiling for both stages. Initialize Python, NumPy and PyTorch CPU/CUDA once per fresh model run with seed 42 before model loading. Replicate IDs are identifiers, not seeds. Resume only from verified completed-call checkpoints after model reload and RNG-state restoration.

Prompts contain the full selected exemplar and canonical story scaffold without narrator-age or clinical-status labels, prescribed error patterns or verbal length targets. Preserve cross-story counterbalancing: opposite story sets, two reversed assignments per target, five uses per assignment at ten replicates. No vision condition is included.

Log checkpoint identity, native template options, effective decoding settings, token counts, finish reasons, prompts, outputs and exact parent hashes. Do not invent missing provenance or imply GPU/model validation from local fixtures.

Generation QC is review-only. Use configs/generation/review.json and the notebook
review block for possible cutoffs, unchanged Stage 2, prompt leakage and word or
phrase loops. Retain every story and complete bundle; flags and downloaded notes
must not change generation, conversion, annotation, classifier sampling or domain
analysis eligibility. Stopping information comes from verified prompt journals;
mark missing information unknown. Review thresholds are heuristics, not clinical
norms. Review-only saved-CSV mode loads no models. Keep full paired text, reasons,
source hashes and provenance in the report; any future exclusions require a
separate explicit decision about analysis units and coverage.

## Text, CHAT and annotations

Never edit raw corpus files or original generated text in place. Do not repair grammar, infer unknown speech or reconstruct parenthesized word fragments. Extraction removes annotation comments, pauses, unintelligibility markers and non-speech codes; it preserves grammatical forms and repetitions.

For generated prose: whole narrative -> pinned Batchalign boundaries -> CHAT -> shared annotation preparation and Chatter -> TalkTag -> preservation QC and Chatter -> Batchalign 3 English UD -> preservation QC and Chatter -> compatible measurement. Do not resegment after annotation. Real ENNI boundaries are retained. Structural validity and word preservation do not establish linguistic boundary or tag accuracy; manual boundary review remains pending.

Use configs/annotation/preparation.json for the same deletion-only preparation
of real and synthetic CHAT before fresh TalkTag. Remove all old dependent tiers,
error judgments, replacement/correction hints and inline annotator comments;
retain produced words and transcription structure, including bch/trn bookkeeping.
Never substitute corrected words. In shared preparation, keep original snapshots
temporarily for synthetic or unspecified collections. Do not duplicate original files for
collections explicitly marked source_type=real; retain file identities, source
checksums, available CHAT PIDs and deletion audits instead. Real originals need
not remain locally available after preparation; verify their checksums whenever
they are available. Keep one original checksum-pinned ZIP per real corpus for
recovery, referenced without copying it into each run. With LEAN_ANNOTATION=True,
the final notebook block verifies the complete shared preparation/TalkTag/UD
chain and source-to-final text fingerprints before removing verified run-owned
intermediate CHAT. Keep final annotations, the checksum-anchored retention
receipt, deletion audits, runtime/manifests and validation diagnostics. Failed,
incomplete and standalone UD runs retain working files. After compaction use
verify_retained_annotations, rather than live intermediate-stage checks; historical
manifests stay immutable. Drive may remove only explicitly recorded intermediate
files with matching checksums. Never remove external inputs, original ZIPs or
generation CSVs/journals through annotation retention. Expanded downloaded
originals may be removed only after all members match their retained ZIPs;
ZIP-backed readers and exact-member recovery must remain usable.
Keep explicit collection provenance. Select real ENNI once alongside all
generator collections in from_chat; prepare_only performs just cleaning and
Chatter. Use standard Chatter structural/alignment checks for real and synthetic
CHAT. Optional strict cross-utterance quotation/completion linker checks are
outside the study outcomes and are not enabled. Preserve original markers and
boundaries rather than repairing their relationships. UD-only on already annotated CHAT
preserves inline codes and intentionally does not repeat the fresh-cleaning pass.

Use configs/annotation/batchalign.json for fresh UD tagging of both real and
synthetic CHAT. Preserve words, codes, headers and main-tier boundaries. Replace
old morphology only in derived copies, retaining snapshots and replacement
counts. Canonical new UD tiers are %mor/%gra for all speakers, with CHI coverage
recorded separately. Do not leave stale %umor/%ugra as preferred feature inputs.
Traditional MOR/MEGRASP and IPSyn are not run by this annotation notebook; R
analyses remain separate. Full corpus execution and manual review remain pending.

Use the pinned validation, segmentation and TalkTag settings. Require complete fresh validation results. Retain failures for review rather than silently skipping files, suppressing errors or replacing words. Annotation works on copies and preserves headers, words, boundaries and existing tiers except explicitly documented preparation.

## Linguistic features

Retain interpretable structural, lexical, disruption, syntax and morphosyntactic measures. Table names containing severity denote raw features, not a severity/profile model.

- General error burden, retracing and repetition use per-utterance rates.
- Bare-marking and pronoun errors use per-word rates.
- Past-overgeneralization opportunities are overt past finite lexical verbs from UD morphology plus explicit bare-past tokens m:0ed and m:base:ed. New annotation uses canonical %mor; older %umor inputs require explicit lineage mapping when fresh feature tables are built.
- Zero opportunities remain missing, not zero rates.
- DSS category totals retain their developmental-score meaning.
- Do not replace narrow past-overgeneralization with a broader error category.

The R notebook's feature inputs and reviewer-requested statistical revisions are pending. Do not report its unexecuted code as results. Domain inference must address unique-child reference observations, repeated synthetic ancestry, generator differences and paired stages. Changing those specifications requires explicit implementation and validation.

## Framework evaluation

Use configs/evaluation/protocol.json and the framework_* modules. Keep generators separate and real splits shared. All eligible real children enter age-stratified classifier folds independent of generation batches. Prompt-pair members remain in one fold; unmatched TD children are individual groups. Use one fold for test, the next for real validation and the others for training. Only synthetic descendants whose two prompt ancestors are both in training are eligible for that fit.

Bootstrap complete synthetic story bundles with replacement within age and clinical group to match each real training fold's counts at the requested synthetic:real ratio. Keep unweighted loss: the target is the observed real training distribution, not 50:50 or population prevalence. Synthetic selections are common across generators and nested across ratio levels. Log sampled rows and unique narratives/bundles/ancestors separately. Preserve the complete original paired-stage data for linguistic analyses.

BERT uses story inputs, fixed update budgets and three classifier seeds; child predictions average story probabilities. Gillam's verified 182-child external sample remains unchanged. Maintain the shared real-only baseline and update-matched controls. Confidence intervals resample ENNI prompt pairs jointly and unmatched TD children individually in separate age/unit-type strata, retaining repeated predictions. Gillam uncertainty resamples children within age/group. This conditional uncertainty bootstrap is distinct from synthetic training resampling.

Corpus metrics are descriptive. Use common text preparation, shared-real TF-IDF, all-cross-document whitespace Jaccard, pinned GPT-2 perplexity and Gaussian mean/covariance Fréchet distance in pinned MiniLM embeddings. IG, MAUVE, exploratory topic modeling and privacy analysis are not included.

## Repository workflow

Use reusable source modules and explicit paths. Keep generated outputs out of the root. Preserve raw/source hashes, child/story identities, ages, model/run/pair/round/replicate IDs and parent ancestry. Fail on missing, conflicting or duplicated identities.

Use apply_patch for hand-edited source. Add meaningful checks for parsing, pairing, split separation, normalization and annotation preservation. Maintain Python compatibility declared by pyproject.toml. Test the changed paths; do not run expensive scientific jobs merely to check imports.

Update current documentation after implementation. Keep reproducibility records factual and limited to this study. External backups require archive/member checksum verification before local removal. Report what was changed, checked and still pending; never fabricate empirical findings or claim that stale outputs were corrected by metadata changes alone.
