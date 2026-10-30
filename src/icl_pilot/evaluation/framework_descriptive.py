"""Small, explicit corpus descriptions on identical real reference samples."""
from __future__ import annotations

from collections import Counter
import math
import re

import numpy as np
import pandas as pd

from .framework_data import _write_json


def _tfidf_tokens(text):
    # Equivalent token pattern/lowercasing to default sklearn TfidfVectorizer.
    return re.findall(r"(?u)\b\w\w+\b", str(text).lower())


def lexical_comparison(reference, generated):
    """All-cross-document means, never arbitrary row-wise semantic pairing.

    TF-IDF vocabulary/IDF are fit once on the shared real reference. This avoids
    changing the feature space with each generator's vocabulary. Novel synthetic
    tokens are measured explicitly through out-of-vocabulary token share.
    """
    ref_tokens = [_tfidf_tokens(text) for text in reference]
    syn_tokens = [_tfidf_tokens(text) for text in generated]
    frequency = Counter(token for tokens in ref_tokens for token in set(tokens))
    if not frequency:
        raise ValueError("The real-reference TF-IDF vocabulary is empty.")
    idf = {token: math.log((1 + len(reference)) / (1 + count)) + 1 for token, count in frequency.items()}
    def mean_vector(documents):
        mean = Counter()
        for tokens in documents:
            vector = {word: count * idf[word] for word, count in Counter(tokens).items() if word in idf}
            norm = math.sqrt(sum(value * value for value in vector.values()))
            if norm:
                for word, value in vector.items():mean[word] += value / norm / len(documents)
        return mean
    r, s = mean_vector(ref_tokens), mean_vector(syn_tokens)
    cosine = sum(value * r[word] for word, value in s.items())
    # Cache repeated document sets while preserving their empirical frequencies.
    ref_sets = Counter(frozenset(str(text).split()) for text in reference)
    syn_sets = Counter(frozenset(str(text).split()) for text in generated)
    jaccard = sum(rc * sc * len(a & b) / len(a | b)
                  for a, rc in ref_sets.items() for b, sc in syn_sets.items() if a | b) / (len(reference) * len(generated))
    total = sum(len(tokens) for tokens in syn_tokens)
    unknown = sum(sum(token not in idf for token in tokens) for tokens in syn_tokens)
    return {"jaccard_whitespace_all_cross_pairs": jaccard,
            "cosine_real_reference_tfidf_all_cross_pairs": cosine,
            "synthetic_tfidf_oov_token_share": unknown / total if total else np.nan}


def run_descriptive(prepared):
    rows, similarities = [], []
    for model, sample in prepared.samples.items():
        for source in ("real", "synthetic"):
            frame = sample[source]
            for (cohort, group), subset in frame.groupby(["cohort_year", "group"]):
                tokens = [str(text).split() for text in subset["output_text"]]
                lengths = np.array([len(words) for words in tokens])
                types = np.array([len(set(words)) for words in tokens])
                rows.append({"generator_key": model, "source": source, "cohort_year": cohort, "group": group,
                    "n_stories": len(subset), "length_mean": lengths.mean(), "length_median": np.median(lengths),
                    "length_min": lengths.min(), "length_max": lengths.max(), "unique_words_mean": types.mean(),
                    "mean_story_type_token_ratio": np.mean(types / lengths)})
        scores = lexical_comparison(sample["real"]["output_text"].tolist(), sample["synthetic"]["output_text"].tolist())
        similarities.append({"generator_key": model, **scores})
        _write_json(sample["path"] / "descriptive_metric_definitions.json", {
            "scope": "descriptive full ENNI reference; not held-out clinical fidelity",
            "length_and_types": "case-sensitive whitespace tokens; per-story summaries",
            "jaccard": "mean set intersection/union over every real/synthetic document pair; case-sensitive whitespace words",
            "cosine": "mean all-cross-document L2 TF-IDF cosine; vocabulary and smooth IDF fit to the identical real reference",
            "tfidf_tokenization": "lowercase; Unicode word tokens of length >=2; raw term counts; smooth IDF log((1+n)/(1+df))+1",
            "oov": "synthetic TF-IDF tokens outside the real vocabulary, divided by all synthetic TF-IDF tokens",
            "type_token_ratio": "mean of each story's types/tokens; not ratio of corpus means",
            "semantic_pairs": False,
        })
    distributions, overlaps = pd.DataFrame(rows), pd.DataFrame(similarities)
    distributions.to_csv(prepared.output_dir / "descriptive_by_generator_age_group.csv", index=False)
    overlaps.to_csv(prepared.output_dir / "lexical_overlap_by_generator.csv", index=False)
    return distributions, overlaps
