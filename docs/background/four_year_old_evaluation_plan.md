# Four-Year-Old Evaluation Plan

This note turns the bundle idea into a concrete evaluation workflow.

The key principle is simple:

- evaluate at the **bundle** level for the main clinical claims,
- keep **story-level** analysis as a secondary diagnostic view,
- use **anomaly detection** only as a quality-control layer, not as the headline result.

## 1. What We Evaluate

The synthetic output should be organized into one row per synthetic child bundle.

Each bundle is the six-story transcript assembled from:

- one fixed SLI-TD prompt pair,
- one fixed replicate index,
- the six canonical ENNI story slots.

The corresponding human comparison unit is one real child transcript, because each real child also contributes six stories.

## 2. Recommended Tables

### 2.1 Bundle Feature Table

One row per bundle, real or synthetic.

Suggested columns:

| Column | Meaning |
| --- | --- |
| `bundle_id` | Unique identifier for the bundle |
| `source` | `real` or `synthetic` |
| `group` | `SLI` or `TD` |
| `cohort` | `4-year-old` |
| `age_months_approx` | Approximate chronological age |
| `age_bin` | Optional coarse age bin if we later widen beyond 4-year-olds |
| `cluster_id` | `pair_id` for synthetic, child id for real |
| `pair_id` | Frozen prompt-pair id for synthetic bundles |
| `replicate_id` | `1..10` for repeated synthetic bundles |
| `prompt_td_child_id` | Fixed TD prompt child for the pair |
| `prompt_sli_child_id` | Fixed SLI prompt child for the pair |
| `target_story_order` | `A1|A2|A3|B1|B2|B3` |
| `n_stories` | Usually `6` |
| `n_utterances` | Bundle-level utterance count |
| `n_words` | Bundle-level word count |
| `mlu_morphemes` | Full-bundle CLAN/KIDEVAL output |
| `mlu_words` | Full-bundle CLAN/KIDEVAL output |
| `dss` | Full-bundle CLAN/KIDEVAL output |
| `ipsyn_total` | Full-bundle CLAN/KIDEVAL output |
| `vocd_d` | Full-bundle lexical diversity |
| `word_errors_per_utt` | Error burden rate |
| `bare_marking_errors_per_word` | Morphosyntactic burden rate |
| `pronoun_errors_per_word` | Morphosyntactic burden rate |
| `rule_generalization_errors_per_word` | Morphosyntactic burden rate |
| `past_overgeneralization_errors_per_past_opp` | Specialized past-tense burden |
| `severity_core_composite` | Optional derived composite for comparison |
| `severity_domain_composite` | Optional derived composite for comparison |
| `qc_anomaly_score` | Optional QC score from a secondary anomaly model |

### 2.2 Reference Bundle Table

One row per real child bundle.

Use the same feature columns as above, except:

- `source = real`
- `pair_id` and `replicate_id` are empty
- `cluster_id = child_id`

## 3. Main Statistical Test

The first formal comparison should be a mixed-effects model on each pre-specified feature.

For a 4-year-old-only comparison, the simplest model is:

```text
feature ~ source + (1 | cluster_id)
```

Where:

- `source = 0` for real bundles
- `source = 1` for synthetic bundles
- `cluster_id` captures repeated synthetic bundles generated from the same frozen prompt pair

Interpretation:

- the `source` coefficient asks whether synthetic bundles systematically differ from real bundles
- the random intercept prevents the 10 synthetic replicates from being treated as fully independent

If we later expand beyond a single age bin, add age as a fixed effect:

```text
feature ~ source + age_months_approx + (1 | cluster_id)
```

If we want to compare both SLI and TD simultaneously:

```text
feature ~ source * group + (1 | cluster_id)
```

That interaction tells us whether the synthetic-vs-real shift differs between SLI and TD.

## 4. Suggested Primary Metrics

Start with a short pre-registered set:

- `mlu_morphemes`
- `vocd_d`
- `dss`
- `ipsyn_total`
- `bare_marking_errors_per_word`
- `pronoun_errors_per_word`
- `rule_generalization_errors_per_word`
- `past_overgeneralization_errors_per_past_opp`

Those are the most interpretable bundle-level measures for the current corpus.

## 5. How To Report Results

For each primary metric, report:

- the fixed-effect estimate for `source`
- a confidence interval
- the p-value or likelihood-ratio test result
- an effect size

Also report:

- the proportion of synthetic bundles that fall outside the central real-child range
- the same summary stratified by `pair_id`

Because synthetic bundles are repeated draws from the same roster pair, do not present the 10 replicates as if they were 10 independent children.

As a sensitivity check, it is also reasonable to average the synthetic replicates within each `pair_id` and refit the same model on pair-level means.

## 6. Where Anomaly Detection Fits

The report PDF used anomaly detection as a plausibility screen for short generated stories.

That idea is still useful, but here it should be secondary:

- train the anomaly model on **real bundle-level features only**
- score the synthetic bundles after the main mixed-model analysis
- use the anomaly score to flag caricatured or unstable bundles for inspection

Do **not** make anomaly detection the main headline result.

## 7. Practical Workflow

1. Build the synthetic bundle transcripts.
2. Annotate them with the same CLAN/KIDEVAL and error pipeline used for real children.
3. Assemble the bundle feature table.
4. Assemble the real-child reference table with the same columns.
5. Fit the mixed models for the primary metrics.
6. Run anomaly detection as a QC pass.
7. Summarize the results at the bundle level, with cluster-aware uncertainty.
8. Repeat the main comparison on pair-level means as a robustness check.

## 8. Short Version

The main comparison should be:

```text
synthetic bundle features vs real child bundle features
```

using mixed-effects models, with anomaly detection kept as a secondary plausibility check.
