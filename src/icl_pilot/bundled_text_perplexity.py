from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.model_selection import KFold


_AGE_RE = re.compile(r"@ID:\s+eng\|[^|]+\|CHI\|([^|]*)\|([^|]*)\|([^|]+)\|")
_REAL_BUNDLE_RE = re.compile(r"(?P<child_id>\d{3})\.cha$")
_SYNTHETIC_BUNDLE_RE = re.compile(
    r"pair_sli(?P<sli_child_id>\d+)_td(?P<td_child_id>\d+)_rep(?P<replicate_id>\d+)_(?P<group>TD|SLI)_bundle\.cha$"
)
_TIMING_RE = re.compile(r"\x15[^\x15]*\x15")
_ANGLE_RE = re.compile(r"<([^<>]*)>")
_BRACKET_RE = re.compile(r"\[[^\]]*\]")
_CHAT_CODE_RE = re.compile(r"\+[/.!?]+")
_ZERO_PREFIX_RE = re.compile(r"\b0([A-Za-z']+)")
_WORD_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_PLACEHOLDER_TOKENS = {"xxx", "xx", "yyy", "yy", "www"}
_VALID_GROUPS = {"TD", "SLI"}


@dataclass
class BundleTranscript:
    source: str
    transcript_id: str
    child_id: str
    transcript_path: str
    age_raw: str
    group: str
    replicate_id: int | None
    pair_id: str
    source_child_id: str
    utterance_tokens: list[tuple[str, ...]]

    @property
    def utterance_count(self) -> int:
        return len(self.utterance_tokens)

    @property
    def token_count(self) -> int:
        return int(sum(len(tokens) for tokens in self.utterance_tokens))

    @property
    def cleaned_text(self) -> str:
        return " ".join(token for utterance in self.utterance_tokens for token in utterance)

    def to_manifest_row(self) -> dict[str, object]:
        return {
            "source": self.source,
            "transcript_id": self.transcript_id,
            "child_id": self.child_id,
            "transcript_path": self.transcript_path,
            "age_raw": self.age_raw,
            "group": self.group,
            "replicate_id": self.replicate_id,
            "pair_id": self.pair_id,
            "source_child_id": self.source_child_id,
            "utterance_count": self.utterance_count,
            "token_count": self.token_count,
            "cleaned_text": self.cleaned_text,
        }


class SmoothedNgramLanguageModel:
    def __init__(self, sequences: list[tuple[str, ...]], order: int = 3, alpha: float = 0.5) -> None:
        if order < 1:
            raise ValueError("N-gram order must be at least 1.")
        if not sequences:
            raise ValueError("At least one token sequence is required.")

        self.order = int(order)
        self.alpha = float(alpha)
        self.end_token = "</s>"
        self.unk_token = "<unk>"
        self.start_tokens = tuple(f"<s{i}>" for i in range(self.order - 1))

        self.vocab: set[str] = {self.end_token, self.unk_token, *self.start_tokens}
        for sequence in sequences:
            self.vocab.update(sequence)
        self.vocab_size = len(self.vocab)

        self.ngram_counts: Counter[tuple[str, ...]] = Counter()
        self.context_counts: Counter[tuple[str, ...]] = Counter()

        for sequence in sequences:
            self._add_sequence(sequence)

    def _map_token(self, token: str) -> str:
        return token if token in self.vocab else self.unk_token

    def _add_sequence(self, sequence: tuple[str, ...]) -> None:
        mapped = tuple(self._map_token(token) for token in sequence) + (self.end_token,)
        padded = self.start_tokens + mapped
        for index in range(self.order - 1, len(padded)):
            context = padded[index - (self.order - 1) : index]
            ngram = context + (padded[index],)
            self.ngram_counts[ngram] += 1
            self.context_counts[context] += 1

    def sequence_log_probability(self, sequence: tuple[str, ...]) -> tuple[float, int]:
        mapped = tuple(self._map_token(token) for token in sequence) + (self.end_token,)
        padded = self.start_tokens + mapped
        total_log_probability = 0.0
        steps = 0
        for index in range(self.order - 1, len(padded)):
            context = padded[index - (self.order - 1) : index]
            ngram = context + (padded[index],)
            numerator = self.ngram_counts[ngram] + self.alpha
            denominator = self.context_counts[context] + self.alpha * self.vocab_size
            total_log_probability += math.log(numerator / denominator)
            steps += 1
        return total_log_probability, steps

    def transcript_perplexity(self, utterances: list[tuple[str, ...]]) -> float:
        total_log_probability = 0.0
        total_steps = 0
        for utterance in utterances:
            log_probability, steps = self.sequence_log_probability(utterance)
            total_log_probability += log_probability
            total_steps += steps
        return math.exp(-total_log_probability / max(total_steps, 1))


def _coerce_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _parse_header_age_group(text: str) -> tuple[str, str]:
    match = _AGE_RE.search(text)
    if not match:
        return "", ""
    age_raw, _sex, group = match.groups()
    return age_raw, group


def _age_year(age_raw: str) -> int | None:
    if ";" not in age_raw:
        return None
    try:
        return int(age_raw.split(";", 1)[0])
    except ValueError:
        return None


def _normalize_chat_utterance(content: str) -> tuple[str, ...]:
    text = content
    text = _TIMING_RE.sub(" ", text)
    text = _ANGLE_RE.sub(r"\1", text)
    text = _BRACKET_RE.sub(" ", text)
    text = text.replace("\u0015", " ")
    text = _ZERO_PREFIX_RE.sub(r"\1", text)

    # Turn lexicalized CHAT fillers into words before removing the remaining CHAT codes.
    text = re.sub(r"&-(um|uh|er|ah|oh)\b", r" \1 ", text, flags=re.IGNORECASE)
    text = re.sub(r"&=(\w+)", " ", text)
    text = re.sub(r"&[+-][A-Za-z:']+", " ", text)
    text = _CHAT_CODE_RE.sub(" ", text)
    text = re.sub(r"[<>]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [
        token.lower()
        for token in _WORD_TOKEN_RE.findall(text)
        if token.lower() not in _PLACEHOLDER_TOKENS
    ]
    return tuple(tokens)


def _extract_chi_utterances(text: str) -> list[tuple[str, ...]]:
    utterances: list[tuple[str, ...]] = []
    for raw_line in text.splitlines():
        if not raw_line.startswith("*CHI:\t"):
            continue
        content = raw_line.split("\t", 1)[1]
        tokens = _normalize_chat_utterance(content)
        if tokens:
            utterances.append(tokens)
    return utterances


def _load_real_bundle_transcripts(real_root: Path, *, age_years: int, group: str) -> list[BundleTranscript]:
    transcripts: list[BundleTranscript] = []
    for path in sorted(real_root.rglob("*.cha")):
        match = _REAL_BUNDLE_RE.search(path.name)
        if not match:
            continue

        text = path.read_text(encoding="utf-8", errors="replace")
        age_raw, transcript_group = _parse_header_age_group(text)
        if transcript_group != group or _age_year(age_raw) != age_years:
            continue

        utterances = _extract_chi_utterances(text)
        if not utterances:
            continue

        child_id = match.group("child_id")
        transcripts.append(
            BundleTranscript(
                source="real",
                transcript_id=child_id,
                child_id=child_id,
                transcript_path=str(path),
                age_raw=age_raw,
                group=transcript_group,
                replicate_id=None,
                pair_id="",
                source_child_id=child_id,
                utterance_tokens=utterances,
            )
        )
    return transcripts


def _load_synthetic_bundle_transcripts(synthetic_root: Path, *, group: str) -> list[BundleTranscript]:
    transcripts: list[BundleTranscript] = []
    for path in sorted(synthetic_root.rglob("*.cha")):
        match = _SYNTHETIC_BUNDLE_RE.search(path.name)
        if not match:
            continue

        text = path.read_text(encoding="utf-8", errors="replace")
        age_raw, transcript_group = _parse_header_age_group(text)
        file_group = match.group("group")
        effective_group = transcript_group or file_group
        if effective_group != group:
            continue

        utterances = _extract_chi_utterances(text)
        if not utterances:
            continue

        pair_id = f"pair_sli{match.group('sli_child_id')}_td{match.group('td_child_id')}"
        transcript_id = path.stem
        transcripts.append(
            BundleTranscript(
                source="synthetic",
                transcript_id=transcript_id,
                child_id=match.group("td_child_id") if group == "TD" else match.group("sli_child_id"),
                transcript_path=str(path),
                age_raw=age_raw,
                group=effective_group,
                replicate_id=int(match.group("replicate_id")),
                pair_id=pair_id,
                source_child_id=match.group("td_child_id") if group == "TD" else match.group("sli_child_id"),
                utterance_tokens=utterances,
            )
        )
    return transcripts


def _median(values: pd.Series) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.median()) if not clean.empty else None


def _iqr(values: pd.Series) -> tuple[float | None, float | None]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return None, None
    return float(clean.quantile(0.25)), float(clean.quantile(0.75))


def _percentile_interval(values: np.ndarray, confidence: float = 0.95) -> list[float] | None:
    if values.size == 0:
        return None
    alpha = max(0.0, min(1.0, 1.0 - confidence))
    lower = float(np.quantile(values, alpha / 2.0))
    upper = float(np.quantile(values, 1.0 - alpha / 2.0))
    return [lower, upper]


def _bootstrap_comparison_summary(
    real_values: pd.Series,
    synthetic_values: pd.Series,
    *,
    reps: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, object]:
    real = pd.to_numeric(real_values, errors="coerce").dropna().to_numpy(dtype=float)
    synthetic = pd.to_numeric(synthetic_values, errors="coerce").dropna().to_numpy(dtype=float)
    if real.size == 0 or synthetic.size == 0:
        return {
            "reps": int(reps),
            "seed": int(seed),
            "confidence": float(confidence),
            "ran": False,
            "reason": "Need non-empty real and synthetic score arrays.",
        }

    rng = np.random.default_rng(seed)
    median_real = np.empty(reps, dtype=float)
    median_synthetic = np.empty(reps, dtype=float)
    median_difference = np.empty(reps, dtype=float)
    median_ratio = np.empty(reps, dtype=float)
    synthetic_share_within_real_iqr = np.empty(reps, dtype=float)

    for rep in range(reps):
        real_sample = rng.choice(real, size=real.size, replace=True)
        synthetic_sample = rng.choice(synthetic, size=synthetic.size, replace=True)
        real_median = float(np.median(real_sample))
        synthetic_median = float(np.median(synthetic_sample))
        real_q1 = float(np.quantile(real_sample, 0.25))
        real_q3 = float(np.quantile(real_sample, 0.75))

        median_real[rep] = real_median
        median_synthetic[rep] = synthetic_median
        median_difference[rep] = synthetic_median - real_median
        median_ratio[rep] = synthetic_median / real_median if real_median != 0 else np.nan
        synthetic_share_within_real_iqr[rep] = float(
            ((synthetic_sample >= real_q1) & (synthetic_sample <= real_q3)).mean()
        )

    ratio_clean = median_ratio[np.isfinite(median_ratio)]
    share_clean = synthetic_share_within_real_iqr[np.isfinite(synthetic_share_within_real_iqr)]
    return {
        "reps": int(reps),
        "seed": int(seed),
        "confidence": float(confidence),
        "ran": True,
        "median_real_perplexity_ci": _percentile_interval(median_real, confidence=confidence),
        "median_synthetic_perplexity_ci": _percentile_interval(median_synthetic, confidence=confidence),
        "median_difference_ci": _percentile_interval(median_difference, confidence=confidence),
        "median_ratio_ci": _percentile_interval(ratio_clean, confidence=confidence),
        "synthetic_share_within_real_iqr_ci": _percentile_interval(share_clean, confidence=confidence),
    }


def build_bundled_text_perplexity(
    *,
    synthetic_root: str | Path,
    real_root: str | Path,
    output_dir: str | Path,
    group: str,
    age_years: int = 4,
    folds: int = 5,
    order: int = 3,
    alpha: float = 0.5,
    random_seed: int = 0,
    bootstrap_reps: int = 10000,
    bootstrap_seed: int = 0,
    exclude_synthetic_source_children_from_real: bool = False,
) -> int:
    group = str(group).upper()
    if group not in _VALID_GROUPS:
        raise ValueError(f"group must be one of {sorted(_VALID_GROUPS)}, got {group!r}")

    synthetic_path = _coerce_path(synthetic_root)
    real_path = _coerce_path(real_root)
    out_dir = _coerce_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    group_slug = group.lower()

    transcript_manifest_csv = out_dir / "transcript_manifest.csv"
    real_scores_csv = out_dir / f"real_{group_slug}_bundle_perplexity.csv"
    synthetic_scores_csv = out_dir / f"synthetic_{group_slug}_bundle_perplexity.csv"
    synthetic_fold_scores_csv = out_dir / f"synthetic_{group_slug}_bundle_perplexity_by_fold.csv"
    summary_json = out_dir / "summary.json"

    real_transcripts = _load_real_bundle_transcripts(real_path, age_years=age_years, group=group)
    synthetic_transcripts = _load_synthetic_bundle_transcripts(synthetic_path, group=group)
    if not real_transcripts:
        raise ValueError(f"No real {group} bundle transcripts found under {real_path}")
    if not synthetic_transcripts:
        raise ValueError(f"No synthetic {group} bundle transcripts found under {synthetic_path}")

    initial_real_count = len(real_transcripts)
    excluded_real_child_ids: list[str] = []
    if exclude_synthetic_source_children_from_real:
        excluded_real_child_ids = sorted({str(transcript.source_child_id) for transcript in synthetic_transcripts})
        excluded_set = set(excluded_real_child_ids)
        real_transcripts = [transcript for transcript in real_transcripts if transcript.child_id not in excluded_set]
        if not real_transcripts:
            raise ValueError(
                f"Excluding synthetic source children removed all real {group} transcripts from the evaluation pool."
            )

    manifest_rows = [transcript.to_manifest_row() for transcript in real_transcripts + synthetic_transcripts]
    pd.DataFrame(manifest_rows).sort_values(["source", "transcript_id"]).to_csv(transcript_manifest_csv, index=False)

    n_splits = min(max(2, folds), len(real_transcripts))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    real_rows: list[dict[str, object]] = []
    synthetic_fold_rows: list[dict[str, object]] = []

    real_array = np.asarray(real_transcripts, dtype=object)
    for fold_id, (train_index, test_index) in enumerate(splitter.split(real_array), start=1):
        train_sequences = [
            utterance
            for transcript in real_array[train_index]
            for utterance in transcript.utterance_tokens
        ]
        model = SmoothedNgramLanguageModel(sequences=train_sequences, order=order, alpha=alpha)

        for transcript in real_array[test_index]:
            perplexity = model.transcript_perplexity(transcript.utterance_tokens)
            real_rows.append(
                {
                    "fold_id": fold_id,
                    "transcript_id": transcript.transcript_id,
                    "child_id": transcript.child_id,
                    "transcript_path": transcript.transcript_path,
                    "utterance_count": transcript.utterance_count,
                    "token_count": transcript.token_count,
                    "perplexity": perplexity,
                }
            )

        for transcript in synthetic_transcripts:
            perplexity = model.transcript_perplexity(transcript.utterance_tokens)
            synthetic_fold_rows.append(
                {
                    "fold_id": fold_id,
                    "transcript_id": transcript.transcript_id,
                    "child_id": transcript.child_id,
                    "pair_id": transcript.pair_id,
                    "replicate_id": transcript.replicate_id,
                    "transcript_path": transcript.transcript_path,
                    "utterance_count": transcript.utterance_count,
                    "token_count": transcript.token_count,
                    "perplexity": perplexity,
                }
            )

    real_scores = pd.DataFrame(real_rows).sort_values("transcript_id").reset_index(drop=True)
    synthetic_fold_scores = (
        pd.DataFrame(synthetic_fold_rows).sort_values(["transcript_id", "fold_id"]).reset_index(drop=True)
    )
    synthetic_scores = (
        synthetic_fold_scores.groupby(
            ["transcript_id", "child_id", "pair_id", "replicate_id", "transcript_path", "utterance_count", "token_count"],
            as_index=False,
        )["perplexity"]
        .agg(["mean", "median", "std", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "mean": "perplexity_mean_across_folds",
                "median": "perplexity_median_across_folds",
                "std": "perplexity_std_across_folds",
                "min": "perplexity_min_across_folds",
                "max": "perplexity_max_across_folds",
            }
        )
        .sort_values("transcript_id")
        .reset_index(drop=True)
    )

    real_scores.to_csv(real_scores_csv, index=False)
    synthetic_fold_scores.to_csv(synthetic_fold_scores_csv, index=False)
    synthetic_scores.to_csv(synthetic_scores_csv, index=False)

    real_ppl = real_scores["perplexity"]
    synthetic_ppl = synthetic_scores["perplexity_mean_across_folds"]
    real_q1, real_q3 = _iqr(real_ppl)
    mann_whitney = mannwhitneyu(synthetic_ppl, real_ppl, alternative="two-sided")
    synthetic_within_real_iqr = (
        float(((synthetic_ppl >= real_q1) & (synthetic_ppl <= real_q3)).mean())
        if real_q1 is not None and real_q3 is not None
        else None
    )
    bootstrap_summary = _bootstrap_comparison_summary(
        real_ppl,
        synthetic_ppl,
        reps=max(1000, int(bootstrap_reps)),
        seed=int(bootstrap_seed),
    )

    summary = {
        "synthetic_root": str(synthetic_path),
        "real_root": str(real_path),
        "age_years": int(age_years),
        "model": {
            "type": "in_domain_smoothed_ngram_lm",
            "order": int(order),
            "alpha": float(alpha),
            "folds": int(n_splits),
            "random_seed": int(random_seed),
            "training_data": f"real {group} {age_years}-year-old bundled CHAT transcripts only",
            "evaluation_unit": "bundled child transcript",
        },
        "preprocessing": {
            "kept_lines": ["*CHI utterances only"],
            "removed": [
                "@ metadata lines",
                "% analysis tiers",
                "timing markers",
                "bracketed CHAT annotations",
                "angle-bracket grouping markup",
                "CHAT control codes such as +/ and retracing symbols",
                "placeholder tokens like xxx/yyy/www",
            ],
            "lexical_fillers_preserved_when_possible": ["um", "uh", "er", "ah", "oh"],
        },
        "counts": {
            f"n_real_{group_slug}_bundles": int(len(real_transcripts)),
            f"n_real_{group_slug}_bundles_before_exclusion": int(initial_real_count),
            f"n_synthetic_{group_slug}_bundles": int(len(synthetic_transcripts)),
        },
        "exclusion_policy": {
            "exclude_synthetic_source_children_from_real": bool(exclude_synthetic_source_children_from_real),
            "excluded_real_child_ids": excluded_real_child_ids,
        },
        "real_distribution": {
            "median_perplexity": _median(real_ppl),
            "mean_perplexity": float(real_ppl.mean()),
            "q1_perplexity": real_q1,
            "q3_perplexity": real_q3,
            "min_perplexity": float(real_ppl.min()),
            "max_perplexity": float(real_ppl.max()),
        },
        "synthetic_distribution": {
            "median_perplexity": _median(synthetic_ppl),
            "mean_perplexity": float(synthetic_ppl.mean()),
            "q1_perplexity": _iqr(synthetic_ppl)[0],
            "q3_perplexity": _iqr(synthetic_ppl)[1],
            "min_perplexity": float(synthetic_ppl.min()),
            "max_perplexity": float(synthetic_ppl.max()),
        },
        "comparison": {
            "median_ratio_synthetic_over_real": (
                float(_median(synthetic_ppl) / _median(real_ppl)) if _median(real_ppl) not in (None, 0) else None
            ),
            "median_difference_synthetic_minus_real": (
                float(_median(synthetic_ppl) - _median(real_ppl))
                if _median(real_ppl) is not None and _median(synthetic_ppl) is not None
                else None
            ),
            "synthetic_share_within_real_iqr": synthetic_within_real_iqr,
            "mann_whitney_u": float(mann_whitney.statistic),
            "mann_whitney_p_value": float(mann_whitney.pvalue),
        },
        "bootstrap_uncertainty": bootstrap_summary,
        "outputs": {
            "transcript_manifest_csv": str(transcript_manifest_csv),
            "real_scores_csv": str(real_scores_csv),
            "synthetic_scores_csv": str(synthetic_scores_csv),
            "synthetic_fold_scores_csv": str(synthetic_fold_scores_csv),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote transcript manifest to {transcript_manifest_csv}")
    print(f"Wrote real bundle perplexity scores to {real_scores_csv}")
    print(f"Wrote synthetic bundle perplexity scores to {synthetic_scores_csv}")
    print(f"Wrote synthetic fold-wise perplexity scores to {synthetic_fold_scores_csv}")
    print(f"Wrote summary to {summary_json}")
    return 0


def build_bundled_td_text_perplexity(
    *,
    synthetic_root: str | Path,
    real_root: str | Path,
    output_dir: str | Path,
    age_years: int = 4,
    folds: int = 5,
    order: int = 3,
    alpha: float = 0.5,
    random_seed: int = 0,
    bootstrap_reps: int = 10000,
    bootstrap_seed: int = 0,
    exclude_synthetic_source_children_from_real: bool = False,
) -> int:
    return build_bundled_text_perplexity(
        synthetic_root=synthetic_root,
        real_root=real_root,
        output_dir=output_dir,
        group="TD",
        age_years=age_years,
        folds=folds,
        order=order,
        alpha=alpha,
        random_seed=random_seed,
        bootstrap_reps=bootstrap_reps,
        bootstrap_seed=bootstrap_seed,
        exclude_synthetic_source_children_from_real=exclude_synthetic_source_children_from_real,
    )
