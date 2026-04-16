# Bundle Perplexity Analysis

## Status

This note records the bundle-level text perplexity analysis for synthetic child narratives.

- The corrected `TD` analysis is the primary result.
- The earlier `TD` analysis that left source children in the real pool is superseded.
- `SLI` results are exploratory only and should be interpreted cautiously because the real reference pool is small.

## Goal

Estimate whether bundled synthetic child narratives are more surprising than real 4-year-old child narratives under an in-domain language model.

The main quantity of interest is not absolute perplexity by itself, but the contrast between:

- held-out real child bundles
- synthetic child bundles

## Data

### Synthetic TD bundles

- Source directory: `data/generated/test2/reannotated/bundled_synthetic_children/stage1_td`
- Bundles scored: `110`

### Real TD bundles

- Source directory: `archive/phase1_prompting/data/enni_files_4-5/TD`
- Eligible real 4-year-old TD child-level bundle files: `45`
- Corrected real evaluation pool after excluding TD source children used in generation: `34`

Excluded real TD source children:

- `406`
- `410`
- `415`
- `417`
- `419`
- `439`
- `450`
- `459`
- `460`
- `466`
- `471`

## Preprocessing

The analysis keeps only the child's spoken surface and removes CHAT analysis markup before modeling.

Kept:

- `*CHI:` utterances only

Removed:

- `@` metadata lines
- `%` analysis tiers
- timing markers
- bracketed CHAT annotations
- angle-bracket grouping markup
- CHAT control symbols such as retracing markers and `+/`
- placeholder tokens such as `xxx`, `yyy`, `www`

Preserved when possible:

- lexical fillers such as `um`, `uh`, `er`, `ah`, `oh`

## Model

- Model type: in-domain smoothed trigram language model
- N-gram order: `3`
- Additive smoothing alpha: `0.5`
- Real-child evaluation: `5`-fold cross-validation
- Synthetic-child evaluation: each synthetic bundle scored under each fold-trained model, then averaged across folds

## Why The TD Analysis Was Corrected

The first TD run used the full set of real 4-year-old TD bundles, but the synthetic TD bundles were generated from a roster of `11` TD source children that were still present in that real evaluation pool.

That is not good evaluation practice, because it risks making the real reference distribution too close to the generation source distribution.

The corrected TD analysis excludes those `11` source TD children from the real training/evaluation pool before fitting the language model.

## Corrected TD Results

Output directory:

- `phase2/generated/evaluation/td_bundle_perplexity_4yo_excluding_source_children`

Key files:

- [summary.json](/Users/shamiraventurini/PycharmProjects/ICL-PILOT/phase2/generated/evaluation/td_bundle_perplexity_4yo_excluding_source_children/summary.json)
- [real_td_bundle_perplexity.csv](/Users/shamiraventurini/PycharmProjects/ICL-PILOT/phase2/generated/evaluation/td_bundle_perplexity_4yo_excluding_source_children/real_td_bundle_perplexity.csv)
- [synthetic_td_bundle_perplexity.csv](/Users/shamiraventurini/PycharmProjects/ICL-PILOT/phase2/generated/evaluation/td_bundle_perplexity_4yo_excluding_source_children/synthetic_td_bundle_perplexity.csv)

Point estimates:

- real median perplexity: `217.23`
- synthetic median perplexity: `454.36`
- median difference: `+237.13`
- median ratio: `2.09x`
- share of synthetic bundles within the real TD IQR: `0.0`
- share of synthetic bundles above the real TD 75th percentile: `1.0`
- Mann-Whitney p-value: `5.89e-15`

Bootstrap uncertainty, `10,000` transcript-level resamples:

- median real perplexity CI: `[190.31, 245.28]`
- median synthetic perplexity CI: `[436.80, 472.17]`
- median difference CI: `[203.54, 273.20]`
- median ratio CI: `[1.84x, 2.42x]`
- synthetic share within real IQR CI: `[0.0, 0.0]`

Interpretation:

- The corrected TD result remains strong after excluding the TD children used in generation.
- Synthetic TD bundles are still much more surprising than real held-out TD bundles under the in-domain TD trigram model.

## Superseded TD Result

The earlier TD run without source-child exclusion is kept on disk for reference only:

- `phase2/generated/evaluation/td_bundle_perplexity_4yo`

It should not be used as the primary result.

## SLI Note

The same style of analysis was run for `SLI`, but it relies on only `11` real 4-year-old SLI bundle transcripts. The result is directionally informative, but much less stable than the corrected TD result. For now, the TD result is the cleaner one to report.

## Recommendation On More Data

Adding more independent 4-year-old child narrative data could strengthen the credibility of the perplexity analysis, especially for `TD`, provided the extra data are reasonably matched in:

- age
- elicitation task
- narrative style
- transcription conventions

Best practice would be:

- keep the corrected in-domain TD analysis as the primary result
- add one or more independent external 4-year-old child corpora as sensitivity analyses
- avoid mixing in badly mismatched conversational or differently transcribed data without checking domain fit
