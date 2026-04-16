# TD Evaluation Report

## Scope

This report summarizes the current core-text evaluation for the synthetic `TD` bundled narratives.

Included:

- corrected bundle-level perplexity analysis
- corrected bundle-level semantic similarity analysis

Set aside for now:

- `SLI`
- child-language feature analyses

## Correct Evaluation Procedure

Synthetic TD bundles were evaluated against real 4-year-old ENNI TD bundle transcripts.

Because the synthetic TD bundles were generated from a roster of `11` TD source children, those real TD children were excluded from the real evaluation pool before fitting the evaluation models.

Excluded TD source children:

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

Counts after correction:

- real TD bundles before exclusion: `45`
- real TD bundles after exclusion: `34`
- synthetic TD bundles: `110`

## Preprocessing

For both analyses, the CHAT transcripts were cleaned before scoring:

- kept `*CHI:` utterances only
- removed `@` metadata
- removed `%` analysis tiers
- removed timing markers and CHAT control symbols
- removed bracketed and angle-bracket annotations
- removed placeholders such as `xxx`, `yyy`, `www`
- preserved lexical fillers such as `um` and `uh` when possible

## 1. Perplexity

Model:

- in-domain smoothed trigram language model
- additive smoothing `alpha = 0.5`
- trained only on real 4-year-old TD bundles from the corrected pool
- real bundles scored with `5`-fold cross-validation
- synthetic bundles scored under each fold-trained model and averaged across folds

Output directory:

- `phase2/generated/evaluation/td_bundle_perplexity_4yo_excluding_source_children`

Key results:

- real median perplexity: `217.23`
- synthetic median perplexity: `454.36`
- median difference: `+237.13`
- median ratio: `2.09x`
- synthetic share within real TD IQR: `0.0`
- Mann-Whitney `p = 5.89e-15`

Bootstrap uncertainty, `10,000` transcript-level resamples:

- median difference CI: `[203.54, 273.20]`
- median ratio CI: `[1.84x, 2.42x]`

Interpretation:

- The synthetic TD bundles are much more surprising than real held-out TD bundles under an in-domain TD language model.
- This result remains strong after excluding the TD children used in generation.

## 2. Semantic Similarity

Model:

- local Sentence-BERT-style embedding model based on `sentence-transformers/all-MiniLM-L6-v2`
- embeddings computed by attention-masked mean pooling over the final hidden state
- embeddings L2-normalized before cosine similarity

Evaluation setup:

- same corrected TD pool as above
- for each real TD bundle: nearest-neighbor cosine similarity to another real TD bundle
- for each synthetic TD bundle: nearest-neighbor cosine similarity to the corrected real TD pool

Output directory:

- `phase2/generated/evaluation/td_bundle_semantic_4yo_excluding_source_children`

Key results:

- real median nearest-real cosine similarity: `0.878`
- synthetic median nearest-real cosine similarity: `0.840`
- median difference: `-0.038`
- median ratio: `0.957`
- synthetic share within real TD IQR: `0.391`
- Mann-Whitney `p = 4.49e-05`

Bootstrap uncertainty, `10,000` transcript-level resamples:

- median difference CI: `[-0.051, -0.017]`
- median ratio CI: `[0.943, 0.980]`

Interpretation:

- Semantically, the synthetic TD bundles are fairly close to real TD bundles.
- However, they are still consistently below the real-to-real nearest-neighbor similarity baseline.
- So the semantic mismatch is present, but much smaller than the perplexity mismatch.

## Overall Interpretation

The corrected TD evaluation gives a mixed but coherent picture:

- `Perplexity`: strong mismatch between synthetic and real TD narrative text
- `Semantic similarity`: synthetic TD bundles are reasonably close in content space, but still not quite as close as real bundles are to each other

In short:

- the synthetic TD bundles seem to preserve a fair amount of broad narrative content
- but they still do not look distributionally natural as text when compared with real 4-year-old TD narratives

## Reporting Recommendation

The corrected TD perplexity analysis should be treated as the primary result.

The semantic evaluation is a useful complement because it shows that the problem is not simply "wrong story content." Instead, the synthetic bundles appear to be semantically plausible while still differing from real TD child narrative language in broader textual distribution.
