# AGENTS.md

This repository is the accumulated working memory and implementation surface for the ICL-PILOT research program. Work here should compound the project’s evidence without blurring the distinction between a plausible synthetic narrative, a faithful model of developmental language, and a clinically useful augmentation sample.

## Mission

Build, evaluate, and document a reproducible pipeline for generating synthetic ENNI-style child narratives with large language models, with particular attention to Developmental Language Disorder (DLD).

The project asks whether in-context learning can reduce the scarcity of naturalistic clinical language data while preserving the developmental, morphosyntactic, lexical, and distributional properties of real child language. Its practical endpoint is whether synthetic data improve models evaluated on held-out real children.

The project is not:

- a generic child-story generator;
- a replacement for clinical assessment;
- evidence that an LLM has acquired DLD;
- permission to infer diagnoses from generated or corpus-derived language measures;
- a generic model-editing or mechanistic-interpretability project.

Frame work as a research peer. Prioritize source fidelity, developmental and clinical precision, leakage control, reproducibility, and careful separation between what the data show and what the project can responsibly claim.

## Canonical research questions

Use these as the current paper-facing questions unless the user explicitly revises them:

1. **Generation fidelity:** Can in-context learning generate age-appropriate ENNI narratives that plausibly represent typically developing (TD) and DLD/SLI language while retaining natural variability?
2. **Architecture:** Does the staged story-scaffold -> TD -> DLD pipeline produce more faithful clinical language than direct generation, and does Stage 2 introduce appropriate DLD-like changes without distorting story content or caricaturing impairment?
3. **Phenotype preservation:** Do synthetic narratives preserve real TD-DLD differences, developmental trajectories, severity heterogeneity, feature distributions, and the covariance or latent structure of the natural population?
4. **Linguistic mechanism:** Which structural, lexical, disruption, and morphosyntactic features explain synthetic fidelity or non-conformity? In particular, can generation reproduce missing obligatory morphology, pronoun patterns, and past-tense behavior at natural rates?
5. **Downstream utility:** Does adding synthetic data improve TD-vs-DLD classification on held-out real children, and how do augmentation ratio, filtering, and feature selection affect that result?
6. **Design feasibility:** With the available clinical corpus, how much age, severity, profile, and story stratification can be supported while retaining a valid held-out evaluation pool?

Fidelity and utility are separate questions. A synthetic sample can appear close to real data in one feature space without improving a classifier. Conversely, a classifier result alone does not establish ecological or clinical fidelity.

## Stable design decisions

Preserve these decisions unless the user explicitly changes the design:

- The primary augmentation unit is one ENNI target story.
- Child-level profiling is used to characterize the natural population, estimate severity and profile shape, test design feasibility, and interpret results.
- Do not impose a full `age x severity x profile` prompt grid by default. The SLI corpus is too sparse at that resolution after evaluation data are reserved.
- Keep age matching central.
- Keep cross-story counterbalancing central: when the target comes from ENNI set A, prompt exemplars come from set B, and vice versa.
- Keep the two-stage generation logic:
  1. realize the target story as an age-appropriate TD-like narrative;
  2. transform that baseline into a DLD/SLI-like narrative.
- Use a project-internal canonical story-content representation derived from the official ENNI picture sequence as the stable content scaffold.
- Do not invent an “official adult ENNI story.” ENNI supplies picture sequences and administration materials, not a normative adult verbal retell.
- Separate prompt children from evaluation children within every evaluation round.
- Prefer a stratified rotating held-out or cross-validated prompt/evaluation design over a single permanent split.
- A child held out in a round contributes none of that child’s stories to prompts in that round.
- Across rounds, a child may change role, but never within the same round.

The primary generation unit and the evaluation unit need not be identical. Story-level samples may be assembled into six-story synthetic child bundles for bundle-level analyses, but this aggregation does not redefine the main augmentation unit or make repeated synthetic bundles independent children.

## Terminology

- Use **DLD** for the contemporary clinical construct when discussing the research target.
- Preserve **SLI** when it is the literal corpus label, directory name, variable value, or historical study terminology.
- When necessary, write “SLI-labelled corpus data used as the DLD reference population” rather than silently relabelling records.
- Use **TD** for typically developing controls.
- Treat generated outputs as **DLD-like** or **SLI-conditioned**, not as children with a diagnosis.
- Treat severity bands as corpus-internal strata, not validated clinical subtypes.

## Source-of-truth hierarchy

When documents conflict, use this order:

1. Explicit instructions from the user in the current task.
2. `docs/current_strategy_handoff.md` for current agreed generation, profiling, and evaluation decisions.
3. Checked operational manifests and rosters under `configs/generation/`.
4. Current implementation under `src/icl_pilot/` and the metadata attached to generated artifacts.
5. `docs/generation_and_evaluation_process.tex` and focused current design notes under `docs/`.
6. `clinical_analyses_reordered.Rmd` for the active integrated analysis when it exists and is current.
7. `docs/data-augmentation_project-report.pdf` for the original pilot framing and historical results.
8. Older draft plans and archived phase-1 material.

Do not assume that a table copied from the PDF report is an operational manifest. Some report-era IDs and mappings were later found to be invalid or unavailable. Use the checked manifests and frozen rosters for executable work.

When the repository and documentation disagree, do not silently choose whichever is convenient. Report the discrepancy, identify the likely authoritative source, and preserve traceability.

## Repository map

- `src/icl_pilot/`: reusable Python package for manifests, generation design, transcript processing, feature construction, and evaluation.
- `configs/generation/`: checked generation manifests, counterbalancing tables, frozen rosters, schedules, and evaluation splits.
- `data/`: source materials, normalized inputs, annotated CHAT transcripts, ENNI picture materials, and prompt-ready data.
- `phase2/ENNI/`: real ENNI reference data and derived measures.
- `phase2/Gillam/`: Gillam reference material and derived artifacts where used.
- `phase2/generated/<age>_year_old/`: age-cohort-specific generated transcripts, measures, and evaluation artifacts.
- `phase2/generated/measures/`: cross-cohort or merged generated-data feature tables.
- `docs/`: strategy, evaluation, profiling, story-content, and handoff documentation.
- `notebooks/`: generation notebooks and exploratory execution surfaces.
- `clinical_analyses_reordered.Rmd`: current integrated R analysis organized around multivariate, distributional, mixed-model, and operator-validity analyses.
- `outputs/`, `images/`, and `results/`: derived presentation, figure, and result artifacts.
- `archive/phase1_prompting/`: historical prompting experiments. Preserve for provenance; do not treat as the active pipeline.

Do not write new generated data or analysis artifacts at the repository root. Use the existing age-, phase-, and purpose-specific directories.

## Data boundaries and provenance

- Never edit raw corpus data, official ENNI materials, or source CHAT transcripts in place.
- Keep processed data reproducible from code, manifests, and explicit commands.
- Preserve child IDs, story IDs, source groups, ages, round IDs, pair IDs, replicate IDs, and stage metadata.
- Never fabricate missing ages, clinical labels, source stories, error annotations, or prompt exemplars.
- Never move a child between TD and SLI/DLD based on derived measures.
- Do not use generated samples as if they were independent human participants.
- Preserve prompt ancestry. A Stage 2 sample must remain traceable to its Stage 1 parent and prompt pair.
- Preserve the distinction between official ENNI content, project-authored canonical story outlines, human transcripts, and model-generated text.
- Treat `.cha`, `.cex`, `.xls`, `.xlsx`, `.csv`, `.json`, and model output files as research artifacts, not disposable build products.
- Large mechanical changes to annotations or tiers require before/after inventories and explicit QC summaries.

## CHAT, CLAN, and annotation rules

- CHAT is the canonical transcript format for linguistic processing.
- CLAN/KIDEVAL, DSS, MLT, IPSyn-related outputs, morphological tiers, grammatical-relation tiers, and manual error codes have distinct meanings. Do not collapse them into a single generic “language score.”
- Preserve `%mor`, `%gra`, `%err`, and other dependent tiers unless a documented normalization or repair step explicitly targets them.
- Do not globally regenerate `%mor` or `%gra` without checking known repaired cases and downstream compatibility.
- Do not interpret absence of an error code as proof that an error was absent unless annotation coverage is known to be complete.
- Apply the same annotation and feature-extraction pipeline to real and synthetic data whenever making source comparisons.
- Record tool versions, command options, and any exclusions that can alter derived measures.

## Profiling and feature conventions

Child-level profiling currently uses structural, lexical, disruption, and morphosyntactic information. Preserve the following operational choices.

### Core grouped error features

- `bare_marking_errors`
- `pronoun_errors`
- `past_overgeneralization_errors`

Do not restore `determiner_errors` to the core profile without a new sparsity and validity review. Do not replace the narrow past-overgeneralization feature with the older broad `rule_generalization_errors` category without documenting the scientific reason.

### Normalization

Use per-utterance rates for general disruption or error burden:

- `Word_Errors_per_utt`
- `retracing_per_utt`
- `repetition_per_utt`

Use the current burden or opportunity-based normalizations for grouped morphology:

- `bare_marking_errors_per_word`
- `pronoun_errors_per_word`
- `past_overgeneralization_errors_per_past_opp`

The current past-opportunity denominator is:

```text
overt past finite lexical verbs from %umor
+ explicit bare-past error tokens (m:0ed, m:base:ed)
```

Therefore:

```text
past_overgeneralization_errors_per_past_opp
= past_overgeneralization_errors
  / (overt past finite verbs + explicit bare-past error tokens)
```

Do not substitute all words or all verbs as the denominator without treating that as a new analysis decision.

DSS category totals remain developmental scores; do not convert them to rates merely for scale uniformity.

### Age adjustment and severity

- Fit developmental expectations using the TD group only.
- Compare each child with the TD expectation for that age.
- Scale deviations using the TD residual variability defined by the active pipeline.
- Orient retained features so higher adjusted values indicate greater severity.
- Keep the distinction between overall severity and profile shape.
- Treat low/moderate/high SLI severity bands as within-corpus tertiles unless a different, validated specification is explicitly introduced.

The strongest current severity signal is bare marking or missing required morphology. Lower MLU, DSS, IPSyn, and lexical diversity also contribute. Past overgeneralization currently behaves more like a developmental or profile feature than the primary severity driver. Pronoun problems are not exclusive to SLI.

These are empirical project findings, not universal diagnostic rules.

## Evaluation framework

Evaluation should address at least four distinct layers.

### 1. Content fidelity

Ask whether a generated narrative realizes the intended ENNI event structure without copying a human transcript.

- Use canonical story units or picture-derived content representations.
- Treat story-unit SemR-p and semantic similarity as supplementary until validated against manual macrostructure judgments.
- Do not allow content similarity alone to stand in for clinical fidelity.

### 2. Linguistic and developmental fidelity

Compare real and synthetic data on pre-specified, interpretable measures such as:

- MLU in morphemes and words;
- DSS;
- IPSyn or syntax-derived measures;
- VOCD-D or other lexical-diversity measures;
- bare-marking, pronoun, and past-overgeneralization rates;
- agreement, auxiliary, determiner, subordination, complex-NP, subject-realization, and dependency-depth measures when justified;
- disfluency and general error burden.

Evaluate more than marginal means. Where the sample supports it, assess:

- distributional distance;
- variance and tail coverage;
- TD-SLI group separation;
- age trajectories;
- covariance and correlation preservation;
- PCA or latent-domain geometry;
- source-pair concentration;
- Stage 2 minus Stage 1 operator effects.

Anomaly detection is a secondary plausibility or QC layer, not the headline clinical result.

### 3. Detectability and synthetic artifacts

Test whether real and synthetic narratives are distinguishable under properly held-out evaluation. High detectability can reveal source artifacts even when selected clinical measures overlap.

Keep detectability separate from TD-vs-SLI classification. They answer different questions.

### 4. Downstream augmentation utility

The confirmatory endpoint is performance on held-out real children.

- Compare against a real-only baseline.
- Use real-only validation/test folds for primary claims.
- Fit feature selection inside each training fold.
- Sweep augmentation ratios without using the held-out test fold for selection.
- Report AUC and balanced accuracy together; include sensitivity, specificity, and uncertainty when available.
- Include synthetic-only performance as a diagnostic, not as evidence of clinical utility.
- Treat overlap-filtered or anomaly-filtered augmentation as a separate analysis.
- Do not describe a tiny AUC increase as useful when balanced accuracy, confidence intervals, or paired-fold contrasts do not support the claim.

## Statistical rules

- State the unit of analysis for every model: story, transcript, bundle, child, prompt pair, or fold.
- Do not treat repeated generations from one prompt pair as independent children.
- Use `cluster_id`, pair-level aggregation, random effects, or another justified dependence correction for replicated synthetic outputs.
- Include age when cohorts span meaningful developmental ranges.
- Account for evaluation round or `round_batch` when fold-stacked data reuse real children across rounds.
- Test source (`real` vs `synthetic`), clinical group (`TD` vs `SLI`), and their interaction when asking whether synthetic-real shifts differ by group.
- Distinguish confirmatory pre-specified measures from exploratory feature sweeps.
- Report effect estimates and uncertainty, not p-values alone.
- Preserve zero denominators and missingness explicitly. Do not silently turn undefined rates into zero.
- Keep real-reference standardization fold-local when it could otherwise leak evaluation information.
- If a model is singular, unstable, separation-prone, or sensitive to scaling choices, report that limitation rather than selecting a favorable specification silently.

## Current evidence state

Treat the following as provisional project findings that must remain traceable to current result files and analysis code:

- Some synthetic cases overlap local regions of the real clinical feature space, but overlap is limited and concentrated in a subset of source pairs.
- Synthetic data can preserve some clinical signal while shifting marginal distributions and distorting feature relationships.
- Current analyses indicate substantial synthetic-real mismatch in core syntactic and lexical measures and weak preservation of SLI latent/correlation structure.
- Feature selection has contributed more to classifier performance than synthetic augmentation in the analyses run so far.
- Current augmentation sweeps show little or no practical improvement over strong real-only classification baselines.
- Filtering to the most plausible-looking synthetic cases has not yet produced a reliable augmentation gain.

Do not hard-code these as permanent conclusions. Re-check them when data, generation models, annotation, features, folds, or analysis specifications change. Preserve negative results: they constrain the next generation design.

## Research-gap taxonomy

Use these labels when recording limitations or open questions:

1. **Generation-to-clinic gap** — generated text exhibits selected DLD-like markers but has not been shown to represent the natural clinical population.
2. **Fidelity-to-utility gap** — synthetic samples look plausible under one or more fidelity tests but do not improve held-out real-data performance.
3. **Distributional-structure gap** — marginal measures overlap while covariance, latent geometry, variability, or profile relationships remain distorted.
4. **Developmental-validity gap** — age matching or age trajectories are inadequate, unstable, or unsupported across cohorts.
5. **Prompt-evaluation leakage gap** — prompt children, source stories, feature fitting, filtering, or model selection contaminate evaluation.
6. **Task-anchoring gap** — a verbal scaffold, exemplar, or retelling setup constrains outputs in ways that differ from picture-elicited ENNI narration.
7. **Measurement-and-annotation gap** — automated tiers, error codes, derived measures, or denominators do not capture the intended linguistic construct consistently.
8. **Corpus-support gap** — the available SLI/DLD data are too sparse for the proposed age, story, severity, and profile stratification.
9. **Model-to-child gap** — similar surface behavior does not imply a shared developmental process or impairment mechanism.

## Source and claim fidelity

- Never invent citations, quotes, sample sizes, model settings, diagnoses, results, or conclusions.
- When evidence is incomplete, mark the claim as provisional or needing verification.
- State the evidence basis: full PDF, source transcript, manifest, generated output, R analysis, result table, presentation summary, or secondary note.
- Preserve exact formulas, thresholds, feature definitions, and denominators from their source. If unreadable or ambiguous, mark them for verification rather than reconstructing them from memory.
- Keep authors’ interpretations separate from project interpretations and both separate from new hypotheses.
- Do not turn a project-internal QC threshold into a universal clinical cutoff.
- Do not collapse disagreement between analyses into false consensus. Record why results differ: unit, cohort, feature set, fold, stage, annotation version, or model specification.
- When drafting paper text, make sample sizes, age ranges, source corpora, model versions, prompt architecture, evaluation unit, and leakage controls explicit.
- New speculative explanations must be labelled as hypotheses and accompanied by the evidence that would confirm or refute them.

## Coding and repository behavior

- Inspect `git status` before edits. This worktree may contain substantial human-authored changes and moved generated artifacts.
- Preserve unrelated modifications and untracked research outputs.
- Never use destructive Git commands to simplify a dirty worktree.
- Prefer reusable code under `src/icl_pilot/` over one-off root scripts.
- Use `apply_patch` for hand-edited source changes.
- Keep manifests and configs versioned and human-inspectable.
- Do not silently rewrite frozen rosters, counterbalancing rules, child IDs, or evaluation folds.
- Add targeted tests or reproducible validation checks when changing parsing, pairing, tier handling, feature construction, denominators, bundle assembly, or evaluation logic.
- Keep paper-facing tables and figures traceable to machine-readable result files and the code that generated them.
- Avoid committing large rendered or regenerated artifacts unless they are intentional project outputs.
- Do not edit generated HTML as the source of an analysis. Edit the `.Rmd` and regenerate deliberately.
- Keep Python support compatible with the version declared in `pyproject.toml` unless the project explicitly raises the minimum.

## Routine commands

Run package commands from the repository root with `src` on `PYTHONPATH`:

```bash
PYTHONPATH=src python -m icl_pilot validate-layout
PYTHONPATH=src python -m icl_pilot show-layout
PYTHONPATH=src python -m icl_pilot show-counterbalance
PYTHONPATH=src python -m icl_pilot show-manifest --limit 5
```

Useful lightweight validation after Python edits:

```bash
python -m compileall -q src
```

The CLI also contains focused builders for DSS features, severity features, age adjustment, error-code features, grouped errors, past-verb features, severity profiles, story-generation designs, frozen rosters, generated-data evaluation, story packets, and story-unit SemR-p. Inspect `PYTHONPATH=src python -m icl_pilot --help` and the relevant implementation before running a mutating builder.

Do not run broad regeneration commands merely to test an import. Many commands write research artifacts and may overwrite or supersede derived outputs.

## Session workflow

At the start of a task:

1. Inspect `git status` and preserve the user’s existing changes.
2. Read `docs/current_strategy_handoff.md` for project decisions relevant to the task.
3. Identify the exact cohort, stage, story unit, prompt/evaluation round, and source files in scope.
4. Inspect the checked manifest or artifact metadata rather than inferring lineage from filenames alone.

During work:

1. Keep scientific assumptions explicit.
2. Make small, reviewable changes.
3. Validate at the level appropriate to the risk.
4. Record reusable design decisions in the relevant Markdown or TeX handoff document.
5. Preserve negative and contradictory results.

At the end of a task:

1. Summarize what changed and what was verified.
2. Name the files that are now sources of truth.
3. Report tests or checks run and any checks that could not be run.
4. Flag stale documentation, data lineage uncertainty, leakage risks, or unresolved contradictions.

## What to flag immediately

Report these conditions instead of silently normalizing them away:

- prompt/evaluation child overlap within a round;
- the same story or child used in a way that violates counterbalancing;
- invalid, missing, duplicated, or inconsistently formatted child IDs;
- age mismatches or cohort labels inconsistent with age in months;
- Stage 2 outputs with missing or ambiguous Stage 1 parents;
- synthetic replicates treated as independent human participants;
- SLI corpus labels silently rewritten as verified DLD diagnoses;
- fabricated adult story scripts presented as official ENNI material;
- manual or automated annotation versions mixed without provenance;
- real and synthetic features built with different pipelines;
- changed rate denominators or severity direction without documentation;
- severity tertiles described as clinical categories;
- missing fold-local fitting for feature selection, scaling, filtering, or thresholding;
- optimization on held-out real test children;
- claims of augmentation benefit supported only by a negligible metric increase;
- documentation that conflicts with operational manifests or current code;
- a request that would overwrite raw data, frozen manifests, or human-authored conclusions.

## Persistent project memory

The original February 2026 project report established the data-scarcity motivation, the first prompting comparison, conformal-anomaly evaluation, and the sample-wise multi-step proposal. Its early pilot focused on ages four and five.

The active repository now contains generated and evaluated cohorts spanning ages four through seven. Always state the actual cohort used by an analysis; do not generalize a four-year-old or four-to-five-year-old result to the full age range.

The current working architecture remains age-matched, cross-story, and staged from a canonical story scaffold to TD-like text and then DLD/SLI-like text. Profiling informs interpretation and feasibility but does not justify a maximally fine clinical prompt grid.

The current evaluation program combines:

- held-out real-vs-synthetic linguistic comparisons;
- age- and group-aware mixed-effects models;
- distributional and latent-structure audits;
- content, semantic, perplexity, and detectability diagnostics;
- paired Stage 2 minus Stage 1 operator-validity analyses;
- downstream TD-vs-SLI classification with real-only held-out evaluation.

The most important unresolved issue is not whether the generated data contain any clinical signal. It is whether they preserve the natural developmental phenotype well enough to provide reproducible value beyond strong real-only baselines.
