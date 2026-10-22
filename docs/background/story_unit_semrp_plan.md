# Story-Unit SemR-p Plan

This note records the agreed plan for exploring whether Semantic R-Precision (SemR-p) can be used as a supplementary macrostructure evaluation layer for ENNI story generation.

## Core Decision

- We do **not** want to redefine the SemR-p metric itself.
- The camera-ready paper defines SemR-p as:
  - a rank-aware top-\(R\) metric,
  - with exact stem-match credit,
  - and embedding-based similarity for non-exact matches,
  - averaged over the top-\(R\) predicted units.
- Therefore, if we change the unit of comparison from **keyphrases** to **ordered story units**, but keep the scoring computation unchanged, this remains an **application of SemR-p**, not a new metric.

Safe name:

- `story-unit SemR-p`
- or `SemR-p applied to story units`

What we should **not** do at this stage:

- add explicit story-grammar weights,
- add causal-graph scoring,
- add type-specific penalties or rewards,
- change the top-\(R\) logic,
- change the aggregation formula.

Those changes would turn the method into a new task-specific metric inspired by SemR-p rather than SemR-p itself.

## Motivation

The current generation evaluation stack already covers:

- microstructure / clinical realism,
- error burden,
- developmental severity and profile.

What it does **not** yet cover well is:

- story content coverage,
- rough event ordering,
- macrostructure fidelity.

The idea is therefore to test whether SemR-p can provide a **supplementary automatic macrostructure signal** without replacing manual story-grammar scoring.

## Planned Use

The intended role of story-unit SemR-p is:

- a supplementary automatic metric,
- applied to natural and generated narratives,
- validated first against manual macrostructure judgments.

It is **not** intended, at least initially, to replace:

- ENNI story grammar scoring sheets,
- or human macrostructure annotation.

## Work Plan

### Phase 1. Build Canonical Story-Unit References

For each ENNI story:

- derive a concise ordered list of story units from the official picture sequence,
- keep the units concrete and event-based,
- keep them short enough to function as semantic comparison targets.

Possible unit content:

- setting / characters,
- initiating event,
- attempts,
- outcomes,
- reactions,
- other key story events.

Deliverable:

- `configs/generation/story_unit_reference.json`: the existing fixed reference
  shared with generation; corpus narrative inputs and derived scores stay
  under their data/analysis paths.

Important:

- these are **project-internal canonical content outlines**
- they are not official ENNI verbal scripts

## Phase 2. Build Narrative-to-Unit Extraction

For each natural or generated narrative:

- segment the story into ordered proposition-like units,
- preserve narrative order,
- produce one list of predicted story units.

The first version should be simple:

- sentence or clause segmentation,
- light proposition extraction / normalization,
- no full event graph yet.

Deliverable:

- `story_unit_predictions.jsonl`

## Phase 3. Apply Unchanged SemR-p

Use the current SemR-p implementation with:

- `references` = canonical ordered story-unit list for the target story,
- `predictions` = extracted ordered story units from the narrative.

This yields:

- one `story_unit_semrp` score per narrative.

The key methodological rule is:

- **keep the metric computation unchanged**
- change only the compared representation

Deliverable:

- `story_unit_semrp_scores.csv`

## Phase 4. Build a Gold Validation Subset

Select a manageable subset of natural ENNI narratives:

- balanced across ages where possible,
- balanced across TD and SLI,
- including weaker and stronger macrostructure performance.

Annotate this subset manually for:

- ENNI story grammar / macrostructure score,
- or at minimum:
  - content coverage,
  - event ordering adequacy,
  - overall macrostructure quality.

Deliverable:

- `macrostructure_gold_subset.csv`

## Phase 5. Validate Story-Unit SemR-p

Test whether `story_unit_semrp` correlates with:

- manual story grammar score,
- manual content coverage judgments,
- manual macrostructure quality judgments.

Also compare it against simpler baselines such as:

- exact unit overlap,
- bag-of-events cosine similarity,
- BERTScore or similar semantic baselines if useful.

Deliverables:

- correlation table,
- plots,
- qualitative failure analysis.

## Phase 6. Decide Whether To Use It In Generation Evaluation

If validation is promising:

- include story-unit SemR-p as a supplementary macrostructure metric for generated narratives.

If validation is weak:

- keep it exploratory,
- do not use it for core evaluation claims.

## Optional Later Extension

Only after initial validation:

- test typed story-unit variants,
- test event-role-aware matching,
- or add a separate coherence/order layer.

However, these should be framed as:

- **derived task-specific extensions**
- not as the original SemR-p metric

## Paper-Safe Framing

Safe framing for future writing:

> We explored whether Semantic R-Precision, originally proposed for ranked semantic overlap of keyphrase predictions, could be applied to ordered story-content units as a supplementary macrostructure evaluation signal.

Unsafe framing to avoid:

> We turned SemR-p into a story-grammar scoring metric.

## Final Summary

The agreed plan is:

1. keep SemR-p computationally unchanged,
2. represent each ENNI story as an ordered list of canonical story units,
3. extract ordered story units from narratives,
4. score narrative-to-story alignment with SemR-p,
5. validate this score against manual macrostructure judgments before using it in generation evaluation.
