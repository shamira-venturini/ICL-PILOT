from __future__ import annotations

from pathlib import Path

import pandas as pd


def add_error_rate_features_to_master(
    input_csv: str,
    output_csv: str | None = None,
) -> int:
    input_path = Path(input_csv).expanduser().resolve()
    output_path = Path(output_csv).expanduser().resolve() if output_csv else input_path

    df = pd.read_csv(input_path)

    required = [
        "#utterances",
        "#words",
        "Word_Errors",
        "retracing",
        "repetition",
        "bare_marking_errors",
        "determiner_errors",
        "pronoun_errors",
        "rule_generalization_errors",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    utterances = pd.to_numeric(df["#utterances"], errors="coerce")
    words = pd.to_numeric(df["#words"], errors="coerce")

    per_utt_pairs = {
        "Word_Errors": "Word_Errors_per_utt",
        "retracing": "retracing_per_utt",
        "repetition": "repetition_per_utt",
    }
    per_word_pairs = {
        "bare_marking_errors": "bare_marking_errors_per_word",
        "determiner_errors": "determiner_errors_per_word",
        "pronoun_errors": "pronoun_errors_per_word",
        "rule_generalization_errors": "rule_generalization_errors_per_word",
    }

    for source, target in per_utt_pairs.items():
        numerator = pd.to_numeric(df[source], errors="coerce")
        df[target] = numerator.div(utterances).where(utterances > 0)

    for source, target in per_word_pairs.items():
        numerator = pd.to_numeric(df[source], errors="coerce")
        df[target] = numerator.div(words).where(words > 0)

    df.to_csv(output_path, index=False)
    print(f"Wrote master with rate features to {output_path}")
    return 0
