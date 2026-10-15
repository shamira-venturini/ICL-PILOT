# Current Agreed Strategy

This file records the current agreed generation, annotation and evaluation decisions.
On 14 October 2026 the user retired the exploratory severity/profile analysis
from the local repository after verified external backup. Profiling is not
part of the current paper's generation or RStudio analysis.

## 1. Generation Unit

- Generation stays at the **story level**.
- We are **not** moving to synthetic child-level bundles for the main augmentation pipeline.
- One generated sample corresponds to one ENNI target story.

### Generation configuration cleanup scope — 14 October 2026

- The user confirmed that the four-year-old cohort used the same generation
  configuration as ages five through seven. The portable setup applies one
  shared protocol to **ages four through seven** and retains age four.
  This confirmation is recorded as author-provided provenance; original CSVs
  did not contain model revisions or full decoding metadata.
- Colab remains the primary interface for CUDA generation. The notebook reads
  `configs/generation/experiment.json` and the single frozen master
  `configs/generation/master_frozen_roster_4_to_7_year_old.csv`.
  Cohort snapshots and schedules are derived into each new run folder.
- Work remains limited to generation and identity/round separation checks.
  Scientific analysis logic and results are unchanged. R file references to
  the roster location were updated during the subsequent consolidation.
- The [generation baseline](../outputs/maintenance/generation_baseline_2026-10-14/README.md)
  records two four-year-old pair discrepancies: saved generations used
  SLI 474 / TD 459 and SLI 475 / TD 417, whereas the archived older roster had
  the TD children reversed. Recorded child/story/round identities remain
  traceable. The 120 corresponding Stage 1 CHAT exports lack ages, but the
  TD ages are available in their original ENNI transcripts. These findings
  are reconciled in the four-year-old rows of the portable master using recorded
  pair identities and child metadata. The original roster is externally
  backed up; existing CHAT files and analysis tables are not repaired or rewritten.
- The membership audit under
  `outputs/maintenance/generation_separation_2026-10-14/` executed only the
  current R data-assembly code. Its main and developmental pools have zero
  prompt/evaluation child overlap across all 17 age-specific rounds. All 940
  synthetic metadata rows match recorded generation ancestry.
- Historical Python outputs under
  `phase2/generated/4_year_old/evaluation/bundle_round_holdout/` violate the
  within-round rule: TD 459 occurs in round 2's evaluation pool and TD 417 in
  round 3's, producing 40 pairwise comparisons with prompt children. These
  historical outputs remain untouched and must not be presented as valid
  held-out evaluation without a separate correction/rerun.
- The separation pass applies to the round-stacked main/developmental R pools.
  The R PCA/correlation (`pca_base`) and distributional (`profile_df`) sections instead
  assemble unique full-reference real and synthetic rows without round-specific
  prompt exclusions. Those are descriptive reference comparisons, not certified
  held-out evaluations. Their code and results remain unchanged for the later
  analysis review; do not generalize the separation pass to the whole R report.
- Verified pre-cleanup backup:
  `/Volumes/TranscendBU/ICL-PILOT_78a5cadad659_20261014T115428Z/Generation/`.
- Subsequent consolidation, authorized on the same date, leaves one 47-pair
  master as the maintained roster input. The four cohort CSVs and the obsolete
  old master are backed up and removed; no compatibility copies are retained.
  `experiment.json` schema 2 names the master once. Cohort labels, orders,
  child identities and raw ages are preserved. Schedule builders reject
  mixing cohorts into one round. Only roster path references changed in
  `clinical_analyses_reordered.Rmd`, `clinical_analyses.Rmd`, and the existing
  `clinical_analyses_reordered_purl.R`; scientific code was not regenerated.
  Backup: `/Volumes/TranscendBU/ICL-PILOT_78a5cadad659_20261014T123617Z/GenerationRoster/`.
  Audit: `outputs/maintenance/master_roster_consolidation_2026-10-14/`.
- The user subsequently confirmed that fixed story scaffolds belong to the
  generation configuration. `story_unit_reference.json` was moved unchanged
  from `phase2/ENNI/measures/story_units/` to `configs/generation/`.
  `experiment.json` schema 3 names this asset at the top level, relative to
  its configuration directory; only the human exemplar CSV is resolved
  through `DATA_ROOT`. The scaffold hash and full job/prompt inputs are
  preserved. Corpus data, scientific analysis code and results are unchanged.
  Backup and move verification are recorded in
  `outputs/maintenance/generation_scaffold_config_2026-10-14/`.
- CHAT parsing was subsequently repaired in `src/icl_pilot/story_unit_semrp.py`:
  ages use `years;months.days`, CHI main-tier continuations are joined before
  normalization, and dependent tiers or recognizable orphan annotation blocks
  cannot enter narrative text. Extraction records source hashes and line-level
  QC and refuses to overwrite existing outputs. Surface normalization is
  otherwise unchanged; no grammatical forms, clinical labels or missing ages
  were invented or corrected.
- The new generation input is
  `data/processed/generation_inputs/story_unit_narratives_chat_v2.csv`.
  Across 361 source CHAT files and 2,166 story rows, text and utterance arrays
  match the historical CSV exactly. The only populated field change is
  `Age_Month` in 2,160 rows. All 564 selected exemplars have ages matching the
  frozen roster. The old CSV remains unchanged for existing analyses.
- One file, `data/annotated/ENNI_batchalign/TD/B/523.cha`, has empty CHI age/group
  fields and shares the filename ID with an SLI-labelled file. Its six rows
  retain unknown age/group; no TD label is inferred from directory or `@Types`.
  It is not selected as a TD prompt. The corrected parser flags the missing
  metadata and all annotation remnants in its QC sidecar.
- The tier-removal utility now removes complete morphosyntactic tiers,
  including their continuations, and works on the declared Python 3.8 minimum.
  It was tested only on fixtures; no corpus tiers were removed or regenerated.
  Backup: `/Volumes/TranscendBU/ICL-PILOT_78a5cadad659_20261014T131515Z/CHATParsing/`.
  Parsing decisions and audit:
  `docs/chat_narrative_parsing.md` and `outputs/maintenance/chat_parser_fix_2026-10-14/`.
- Prompt preparation now uses the frozen, verbatim message templates in
  `configs/generation/prompts/stage1.json` and `stage2.json`, selected by
  `experiment.json` schema 4. One `PromptBuilder` serves Colab previews and
  inference. The exact instructions, whitespace, replicate-ID content and
  age-through-exemplar design are preserved. Preflight validates each selected
  exemplar's raw age against the roster and renders both stage templates.
- New runs write `prompt_records.jsonl` with attempted prompts before model
  calls and output records after decoding. The journal stores exact messages,
  rendered chat text, input-token hashes/counts, sampling parameters, exemplar
  identities/ages and job lineage. Stage 2 requires its completed same-job
  Stage 1 record and matching parent text. Output CSVs store both prompt IDs.
  Template bytes/code hashes are captured, and exports include the journal,
  manifest and generation-design snapshots. Failure records are not completions.
- All 2,820 job prompt pairs were checked against the original notebook;
  Stage 2 checks use actual saved historical Stage 1 outputs. This verifies
  message preservation, not fresh GPU inference. Checkpoint/restart and frequent
  persistent Drive saving remain the next execution work; journals currently
  require a fresh run directory. Runtime/model version pins and a real Colab GPU
  trial remain pending. No corpus data, scientific analysis or results changed.
  Backup: `/Volumes/TranscendBU/ICL-PILOT_78a5cadad659_20261014T134815Z/GenerationPrompts/`.
  Audit: `outputs/maintenance/prompt_builder_2026-10-14/`.
- The user subsequently deferred Colab execution work and requested cleanup of
  other repository areas. The first post-generation pass hardens the existing
  CSV-to-CHAT converter without changing narrative normalization, punctuation
  or utterance segmentation. It validates all rows before writing, requires a
  fresh output directory, keeps empty CSV cells distinct from text, and rejects
  duplicate destinations, conflicting age/child/cohort metadata and malformed
  identifiers. Known missing metadata can come only from the specified roster;
  absent specified rosters and unknown pairs now fail explicitly.
- New CHAT exports and their manifests retain available prompt-record IDs and
  same-row Stage 1 parent links. Historical CSVs without journal IDs remain
  supported. Manifest additions record metadata sources and source/output hashes.
  Maintained workflow Markdown files are now explicitly allowed by `.gitignore`;
  this does not stage, commit or publish them. Other documents retain their
  previous ignore behavior. See `docs/post_generation_processing.md`.
- Existing transcripts, model settings, prompts, notebooks, R sources and
  scientific results are unchanged. Annotation is not rerun. The next separate
  cleanup is the TalkTag/CLAN workflow: the standalone TalkTag notebook has fixed
  Drive paths and an unpinned installation, and the generation notebook's optional
  annotation currently defaults to Stage 2 only. Source-preserving annotation,
  comparable real/synthetic processing and repair-tool lineage need review.
  Backup and verification: `outputs/maintenance/post_generation_cleanup_2026-10-14/`.
- The user then requested one generation-plus-annotation notebook, with
  independently runnable blocks and evaluation kept separately in a Python
  Colab/Jupyter notebook and R Markdown. The maintained interface remains
  `notebooks/colab_direct_generation_pipeline.ipynb` (retained path, updated title).
  Its modes are `full`, `generation_only`, `from_csv`, `convert_only`, and
  `from_chat`. Saved-input modes do not load the generator or exemplar corpus.
  The existing Python evaluation notebook and all R sources/results are unchanged.
- `REPLICATES=None` keeps the frozen ten-replicate study default. A positive
  override affects only the new run, is hashed in its effective configuration,
  and previews 282 jobs per replicate across 47 pairs/six stories. Each job has
  TD-like and SLI-conditioned outputs. Child pairs, input/exemplar/scaffold text,
  message templates, schedule order, global seed and model sampling calls remain
  unchanged. The runner now records completion/interruption status and explicitly
  retains `td_age` in new CSVs. It does not implement per-job reseeding or resume.
- Conversion and annotation are separate blocks after generation; the generator
  is released before TalkTag starts. Both synthetic stages are selected by
  default, and all selected cohorts/stages share one annotation invocation.
  New `configs/annotation/talktag.json` pins upstream source commit
  `a9df736acf4efaa0c6ec79240a2d36348de983ab`. This is a new portable-run source
  selection, not recovered historical annotation provenance. The annotation
  environment is separate; its installed source commit and package versions
  are recorded. Full software/model locks and actual GPU validation remain pending.
- `src/icl_pilot/annotation.py` copies CHAT inputs, joins wrapped CHI main tiers
  only in the copy, and checks file/utterance coverage, source surface/order and
  non-target/header/tier preservation. It records hashes, commands, runtime,
  logs/reports and QC, retains failures and labels debug-limited annotation
  partial. These checks do not certify linguistic tag accuracy. No existing
  repair scripts or CLAN/feature builders run automatically. Missing selected
  stages and overlapping CSV/input collections fail explicitly.
- `pipeline_manifest.json` records routes and block states. Drive copies are
  hash-verified, preserve the run hierarchy and reject another run's destination;
  exports include all local processing artifacts. Downloads/runtime release are
  optional and off by default. Generation Drive copies remain at cohort boundaries;
  frequent durable checkpoints and sampling-state recovery remain deferred.
- The standalone `notebooks/TalkTag.ipynb` was verified against the external
  backup and removed after the combined mocked workflow passed. The audit records
  143 passing focused tests, all 5,640 default stage prompts unchanged, unchanged
  default job/evaluation tables, and 29,598 protected files unchanged. No actual
  GPU inference, annotation, CLAN or scientific analyses were run locally.
  Backup: `/Volumes/TranscendBU/ICL-PILOT_78a5cadad659_20261014T143952Z/GenerationAnnotation/`.
  Audit: `outputs/maintenance/combined_notebook_2026-10-14/`.
- On 15 October the user requested a one-narrative Colab check. The notebook now
  defaults to `SMOKE_TEST=True`, executing one canonical target-story job through
  both generation stages, CHAT conversion and annotation. It previews one TD-like
  and one SLI-conditioned narrative; the test run ID uses `enni_smoke_`. Manifests
  and CSVs explicitly mark `test_only`, and the selected execution schedule is
  recorded separately from the unchanged complete study design. These outputs
  are not study data or complete six-story bundles. Set `SMOKE_TEST=False` for
  a full generation run. Replicate count alone is not a one-job limit.
  The switch applies only to new generation; saved-input routes retain their
  explicit input selection. Source exemplars and both models are still required.
- The Colab/VS Code setup now defaults to a persistent input tree under
  `MyDrive/icl_pilot_inputs`; local Jupyter still defaults to the checkout and
  `ICL_PILOT_DATA_ROOT` overrides both. A first upload of the exemplar CSV to
  `/content` is copied into the configured relative path on mounted Drive,
  SHA-256 verified, and recorded in `pipeline_manifest.json`. Later sessions
  reuse that file. Conflicting uploads stop without replacing existing inputs;
  saved-input routes skip exemplar staging. Outputs still use the existing
  Drive export folder. The author handles Drive authorization/upload in VS Code;
  no desktop control, Drive upload or GPU inference was performed by the agent.
  Validation: 159 focused checks passed, followed by all 36 pipeline checks
  after adding an additional Drive-mount test and integrating input staging
  into the mocked end-to-end route. Actual Colab inference remains pending.
- Live Colab troubleshooting on 15 October found a failure creating the TalkTag
  environment with `/usr/bin/python3` (Python 3.13). The VS Code traceback did
  not contain the child process output, so missing `ensurepip` is a suspected
  cause, not a verified diagnosis. The installer now uses `venv --without-pip`
  and kernel pip >=22.3 with `--python` targeting the annotation interpreter.
  It repairs partial environment configuration without clearing the directory
  and includes venv stdout/stderr in setup failures. TalkTag revision, annotation
  options and the generation protocol are unchanged. The remote checkout must
  receive this helper update or the current session must use the documented
  environment-recovery commands before retrying annotation.
- The author subsequently reported a completed smoke test and pasted the two
  annotated CHAT exports from `enni_smoke_20261015T090021775255Z`: first pair
  SLI 444 / TD 410, target A1, replicate 1. Their nine CHI utterances are identical
  and contain no inline error tags. The Stage 2 header links to the supplied
  Stage 1 prompt-record ID. This is one observed unchanged transformation, not
  evidence of clinical fidelity or a measured failure rate across the study.
  The checked Stage 2 prompt requests style/grammatical-level transfer while
  preserving events; it does not require explicit error insertion or reject
  unchanged text. The notebook now has a read-only whitespace-normalized copy
  check for newly generated or selected saved CSVs. It flags identity without
  changing prompts, outputs, sampling, selection or annotation behavior.
- The author explicitly confirmed that the prompts must remain unchanged and
  requested a further very small test. `SMOKE_TEST_JOBS` now controls the size
  of the canonical schedule prefix used with `SMOKE_TEST=True`. The notebook
  defaults to 12 jobs: all six stories, replicates 1 and 2, for the first frozen
  four-year-old pair SLI 444 / TD 410 (24 narratives total). The Python setup API
  retains one job as its default. Full study mode and its schedule are unchanged.
  Each test uses a fresh directory, records the exact selection and remains
  `test_only`. Seed 42 and the sequential sampling stream are retained, so the
  first job may reproduce the previous test; the later jobs provide additional
  conditions/draws. This is one-pair diagnostics, not a cohort-wide estimate.
  The local seven-file focused suite passed 176 tests. A direct comparison with
  Git commit `01cace79` checks all 5,640 prompts using historical Stage 1 parents,
  tokenization/chat-template/decoding and the model generation call, alongside
  the unchanged complete-input/evaluation fingerprints. Detailed verification
  is under `outputs/maintenance/small_batch_2026-10-15/`. Code reorganization,
  parsing/age metadata, roster reconciliation, provenance, I/O and annotation
  runtime changes must not be described as "no code changed". Historical model
  revisions and complete dependency versions remain incompletely recorded.

## 2. Profiling Is Archived Exploratory Work

- Severity/profile scores were computed in earlier exploratory work, including
  some preliminary four-year-old severity-adjusted models.
- They did not condition the active prompts, determine age matching, or enter
  the current integrated R Markdown models.
- The active code no longer loads profiling tables to construct rosters or
  generation packages. New runs leave legacy profile metadata unknown; existing
  records retain their historical fields for lineage.
- Raw linguistic feature tables remain active inputs despite their legacy
  `severity_features_raw_*` filenames. They must not be deleted as profiling.
- Backup and restoration inventory:
  `/Volumes/TranscendBU/ICL-PILOT_78a5cadad659_20261014T092052Z/Profiling/`.
  The local audit is under `outputs/maintenance/profiling_cleanup_2026-10-14/`.

## 3. Historical Profiling Observations

The following observations arose from the archived exploratory profiling.
They are not a description of active severity scoring in the current paper.

- The strongest severity signal in the current corpus is **bare marking / missing required morphology**.
- Classic structural weakness also clearly contributes:
  - lower MLU,
  - lower DSS,
  - lower IPSyn,
  - lower lexical diversity.
- Past overgeneralization is present, but looks more like a developmental/profile feature than the main severity driver.
- Pronoun problems are not SLI-only; they also appear in TD, though less often.

## 4. Error Grouping Currently In Use

Current grouped morphosyntactic error features:

- `bare_marking_errors`
- `pronoun_errors`
- `past_overgeneralization_errors`

Important note:

- `determiner_errors` was removed from the core profiling pipeline after cleaning showed that the surviving determiner feature was too sparse.
- broad `rule_generalization_errors` was replaced in the core profiling pipeline by a narrower **past overgeneralization** feature.

## 5. Rate Decisions

Use per-utterance rates for disruption/error burden:

- `Word_Errors_per_utt`
- `retracing_per_utt`
- `repetition_per_utt`

Use per-word / opportunity-based rates for grouped morphosyntactic burden:

- `bare_marking_errors_per_word`
- `pronoun_errors_per_word`
- `past_overgeneralization_errors_per_past_opp`

The current past-overgeneralization denominator is:

- overt past finite lexical verbs from `%umor`
- plus explicit bare-past error tokens (`m:0ed`, `m:base:ed`)

So:

- `past_overgeneralization_errors_per_past_opp`
  =
  `past_overgeneralization_errors / (overt past finite verbs + bare past error tokens)`

DSS category totals remain developmental scores, not rates. TD-referenced
age adjustment was part of the archived profiling; the active R analysis
models age directly.

## 6. Historical Age Adjustment

- Retained features were age-adjusted using the **TD group only**.
- The adjustment is TD-referenced and age-based.
- Features were then aligned so that **higher values always mean greater severity**.

## 7. Active Features And Archived Profiles

Active inputs include `phase2/ENNI/measures/dev_measures.csv`,
`phase2/ENNI/measures/severity_features_raw_real.csv`, and
`phase2/generated/measures/severity_features_raw_syn.csv`.

The age-adjusted tables, profile composites, severity bands, dedicated
profiling documentation, and their builders are preserved in the external
backup. Its manifest records the original repository-relative paths and
checksums. The historical SLI severity bands were corpus-internal tertiles,
not clinical categories. Frozen rosters, schedules and existing generated
metadata remain local and unchanged.

## 8. Prompt Design Decision

The agreed conclusion is:

- the **core prompt architecture remains the same as in the PDF report**
- this means:
  - story-level prompting,
  - age matching,
  - cross-story counterbalancing,
  - two-step TD-to-DLD style prompting

What changed is our understanding of what the corpus can support:

- full `age x severity x profile` prompt stratification is too sparse for SLI
- therefore the pilot should remain close to the original PDF design
- the earlier profiling was exploratory and is now archived; do not describe
  it as an implemented conditioning or active evaluation step

## 9. ENNI Story Content Scaffold

- ENNI does **not** appear to provide an official canonical adult verbal story script.
- The official materials provide the picture sequences and administration instructions, but not a normative adult retell to use as a source text.
- Therefore, the project should **not** invent an "adult ENNI version" and treat it as if it were part of the instrument.
- Instead, each target story should have a project-internal **canonical story content representation** derived from the official ENNI picture sequence.
- This canonical representation should be concise and structured, for example:
  - page-by-page events,
  - main referents,
  - key actions,
  - outcomes / story grammar units.
- That representation should be used as the stable content scaffold for stage 1 prompting.
- In other words:
  - official ENNI picture story = content source,
  - internal canonical outline = prompt scaffold,
  - generated TD-like narrative = stage 1 output.

## 10. Evaluation Design Decision

- We want to preserve natural data for evaluation.
- A single fixed prompt/eval split wastes too much data.
- The agreed solution is a **rotating held-out design**:
  - in each round, one subset of children is reserved for evaluation,
  - the remaining children are used for prompt construction,
  - no child appears in both roles within the same round,
  - across rounds, children may change role.

This should be described in the paper as a **stratified rotating held-out evaluation design** or a **cross-validated prompt/evaluation design**, not as a custom ad hoc workaround.

## 11. Current Analysis Scope

The domain-specific paper analysis is `clinical_analyses_reordered.Rmd`, using
ENNI as its real reference corpus. Its linguistic measures, mixed models,
distributional comparisons, PCA/correlation analyses and paired stage changes
are separate from severity/profile modelling. Gillam is used for the
framework text-classifier external evaluation. Additional Python feature-
classification methods are deferred to a future manuscript revision.

## 12. Key Paper-Safe Summary

If a future chat needs the shortest possible summary:

> Generation remains story-level, age-matched, cross-story counterbalanced,
> and staged from a picture-derived content scaffold to TD-like and then
> DLD-like text. Profiling is archived exploratory work. Linguistic fidelity
> is evaluated against ENNI in rotating held-out rounds, separating prompt
> and evaluation children within each round. Framework text classifiers use
> a separate external Gillam sample.

## 13. Gillam Lineage And ENNI-Only Sensitivity Audit (August 2026)

The pooled real feature table at
`phase2/generated/measures/severity_features_raw_real.csv` is an exact
row-wise concatenation of the ENNI and Gillam real feature tables. In the
49.9--95 month Python evaluation window, it contains 351 children (212 TD,
139 SLI). The ENNI-only table contains 235 (188 TD, 47 SLI), and the 116
Gillam rows contain 24 TD and 92 SLI children.

Gillam therefore supplied most of the pooled SLI evaluation sample. Removing
Gillam is a useful corpus-sensitivity analysis, but a naive pooled ENNI-only
augmentation test is not prompt-independent: all 47 ENNI SLI children in the
evaluation window, and 47 of 188 ENNI TD children, served as prompt children
in at least one rotating round. There are zero globally prompt-free ENNI SLI
children. ENNI-only augmentation utility must therefore be evaluated
round-by-round, using no stories from a held-out child in that round and only
synthetic descendants from compatible rounds. Pooled ENNI-only classifier
sweeps are exploratory sensitivity analyses, not confirmatory leakage-free
tests.

The saved external-Gillam classifier result is also not an independent
external evaluation: all 116 Gillam holdout File_IDs are already present in
the pooled real training table. Do not cite that result as external
generalization.

The operational failure is traceable in the runner: its default `REAL_CSV`
was the pooled ENNI+Gillam table, while `--external-holdout-csv` added a test
table without subtracting those children from `real_rows`. The old output also
labelled its threshold source as ENNI even though the out-of-fold predictions
came from the pooled table, and there was no train--holdout ID assertion or
recorded real-input path. Thus the reasonable goals of increasing the SLI
reference sample and creating a separate Gillam test were accidentally
combined in one execution path.

A corrected external-Gillam evaluation was subsequently run without
overwriting the historical artifacts. It fits models on the 235-child
ENNI-only evaluation window, selects thresholds from ENNI-only out-of-fold
predictions, and tests on the same 116 Gillam children; the training/test
File_ID intersection is zero. The same 940-row synthetic table, panels, model
family, folds, and seed were retained. Corrected augmented-minus-real-only AUC
contrasts are -0.003 (all shared), +0.009 (balanced literature), -0.009
(classic clinical), -0.048 (morphosyntax errors), and -0.086 (syntax
complexity). Balanced-accuracy contrasts are small and positive (+0.006 to
+0.032) but threshold-dependent and lack uncertainty intervals. This does not
support a reliable augmentation benefit. The runner now stops if any external
holdout File_ID occurs in its real training input. Corrected outputs and the
invalid-versus-valid comparison are stored under the
`allages_overlap_utility_panels_valid_enni_train_external_gillam*` and
`allages_overlap_utility_panels_gillam_invalid_vs_valid_comparison*` result
prefixes.

The ratio-sensitivity extension is implemented in
`phase2/replications/guo_schneider_6yo_ablation/scripts/run_external_synthetic_ratio_sweep.py`.
It keeps Gillam entirely test-only, makes the ENNI-to-synthetic ratio explicit
within TD and SLI groups, retunes thresholds from ENNI-only out-of-fold
predictions for each ratio, and writes a draw-level synthetic-source manifest.
Ratios above the available source-child count use replacement and must be
interpreted as repeated training exposure rather than extra independent
synthetic children. Its outputs must be given a new result prefix and reported
as retrospective external sensitivity results, not as a prospectively blinded
confirmatory sweep.

The completed external-Gillam sweep used ratios 0, 0.25, 0.5, 1, 2, and 5,
with the ratio defined independently in TD and SLI relative to the ENNI
training count. The zero-ratio baseline reproduced the prior corrected
external estimates exactly. Results were mixed at 0.25: the balanced-literature
panel changed from AUC 0.9022 to 0.9108 and balanced accuracy 0.6857 to 0.7337,
and the classic-clinical panel changed from AUC 0.8881 to 0.8986, but the
all-shared, morphosyntax, and syntax-complexity panels all lost AUC (−0.0091,
−0.0471, and −0.1051, respectively). Higher ratios did not produce a
consistent AUC benefit and ratio 5 reduced AUC in every panel. The full
machine-readable sweep, including ENNI out-of-fold threshold selection,
Gillam predictions, and the draw-level source manifest, has prefix
`allages_external_gillam_synthetic_ratio_sweep` under the replication results
directory. It was executed with:

```bash
python phase2/replications/guo_schneider_6yo_ablation/scripts/run_external_synthetic_ratio_sweep.py \
  --output-prefix allages_external_gillam_synthetic_ratio_sweep
```

This pattern does not establish a reliable augmentation benefit.

The active `clinical_analyses_reordered.Rmd` now defaults to
`phase2/ENNI/measures/severity_features_raw_real.csv`. A pooled-corpus
sensitivity render must be requested explicitly through
`ICL_PILOT_REAL_CSV`; it is no longer the default paper-facing analysis.

The ENNI-only rerun materially changed reference geometry and several
paper-facing estimates. Apparent same-group overlap and coverage often
increased because nearest-neighbor thresholds widened after the SLI reference
sample fell from 139 to 47; these increases are not evidence of improved
generation. The identical-code integrated R rerun also changed PCA feature
retention and source-by-group conclusions for MLU, VOCD-D, and DSS. Full
commands, audits, deltas, warnings, and reports are under
`phase2/replications/guo_schneider_6yo_ablation/results/enni_only_rerun/`.

The completed rerun covers 23 current non-smoke Python evaluations plus paired
pooled and ENNI-only renders of the integrated R analysis. Exact matching
produced 8,468 classifier-configuration comparisons with no unmatched rows.
The ENNI-only scarcity and quantity surfaces show apparent positive
augmentation AUC deltas across all real-data levels, unlike the pooled
surfaces at moderate and full real-data sizes. These deltas are not valid
utility evidence under the pooled Python design because of the prompt-ancestry
problem above. The mechanism audit also changed substantively: tree-model
importance shifted from IPSyn under the pooled reference to bare-marking
errors under ENNI only, and concordance between real-only and augmented model
importance weakened. Use the rerun as evidence that Gillam materially affects
the estimated reference phenotype and evaluation conclusions, not as evidence
that ENNI-only augmentation has been validated.
