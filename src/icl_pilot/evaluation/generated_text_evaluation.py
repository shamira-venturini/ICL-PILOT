from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold


_AGE_RE = re.compile(r"@ID:\s+eng\|[^|]+\|CHI\|([^|]+)\|([^|]*)\|([^|]+)\|")
_STORY_RE = re.compile(r"^@G:\s+(\S+)\s*$")
_TIMING_RE = re.compile(r"\x15[^\x15]*\x15")
_ANGLE_RE = re.compile(r"<([^<>]*)>")
_BRACKET_RE = re.compile(r"\[[^\]]*\]")
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
_REAL_PATH_RE = re.compile(r"(?P<child_id>\d+)(?:_(?P<story_id>[AB]\d))?\.cha$")
_SYNTHETIC_STORY_PATH_RE = re.compile(
    r"pair_sli(?P<sli_child_id>\d+)_td(?P<td_child_id>\d+)_rep(?P<replicate_id>\d+)_(?P<story_id>[AB]\d)_(?P<group>SLI|TD)\.cha$"
)
_PLACEHOLDER_TOKENS = {"xxx", "yyy", "www", "xx", "yy"}
_CANONICAL_STORY_IDS = {"A1", "A2", "A3", "B1", "B2", "B3"}


@dataclass
class NarrativeRecord:
    source: str
    group: str
    story_id: str
    child_id: str
    transcript_path: str
    age_raw: str
    age_months: float | None
    stage: str
    bundle_id: str
    pair_id: str
    replicate_id: int | None
    round_id: str
    source_child_id: str
    raw_story_text: str
    normalized_text: str
    tokens: tuple[str, ...]
    token_count: int
    type_token_ratio: float

    def to_csv_row(self) -> dict[str, object]:
        return {
            "source": self.source,
            "group": self.group,
            "story_id": self.story_id,
            "child_id": self.child_id,
            "transcript_path": self.transcript_path,
            "age_raw": self.age_raw,
            "age_months": self.age_months,
            "stage": self.stage,
            "bundle_id": self.bundle_id,
            "pair_id": self.pair_id,
            "replicate_id": self.replicate_id,
            "round_id": self.round_id,
            "source_child_id": self.source_child_id,
            "raw_story_text": self.raw_story_text,
            "normalized_text": self.normalized_text,
            "token_count": self.token_count,
            "type_token_ratio": self.type_token_ratio,
        }


class AdditiveBigramLanguageModel:
    def __init__(self, token_sequences: list[tuple[str, ...]], alpha: float = 0.5) -> None:
        if not token_sequences:
            raise ValueError("At least one reference sequence is required to build a language model.")

        self.alpha = float(alpha)
        self.unk = "<unk>"
        self.end = "</s>"
        self.vocab: set[str] = {self.unk, self.end}
        for sequence in token_sequences:
            self.vocab.update(sequence)
        self.vocab_size = len(self.vocab)

        self.unigram_counts: Counter[str] = Counter()
        self.bigram_counts: Counter[tuple[str, str]] = Counter()
        self.context_counts: Counter[str] = Counter()
        self.total_tokens = 0

        for sequence in token_sequences:
            mapped = [self._map_token(token) for token in sequence]
            unigram_sequence = mapped + [self.end]
            self.unigram_counts.update(unigram_sequence)
            self.total_tokens += len(unigram_sequence)

            bigram_sequence = ["<s>"] + mapped + [self.end]
            for previous, current in zip(bigram_sequence, bigram_sequence[1:]):
                self.bigram_counts[(previous, current)] += 1
                self.context_counts[previous] += 1

    def _map_token(self, token: str) -> str:
        return token if token in self.vocab else self.unk

    def unigram_perplexity(self, tokens: tuple[str, ...]) -> float:
        mapped = [self._map_token(token) for token in tokens] + [self.end]
        log_probability = 0.0
        denominator = self.total_tokens + self.alpha * self.vocab_size
        for token in mapped:
            probability = (self.unigram_counts[token] + self.alpha) / denominator
            log_probability += math.log(probability)
        return math.exp(-log_probability / max(len(mapped), 1))

    def bigram_perplexity(self, tokens: tuple[str, ...]) -> float:
        mapped = ["<s>"] + [self._map_token(token) for token in tokens] + [self.end]
        log_probability = 0.0
        steps = 0
        for previous, current in zip(mapped, mapped[1:]):
            numerator = self.bigram_counts[(previous, current)] + self.alpha
            denominator = self.context_counts[previous] + self.alpha * self.vocab_size
            probability = numerator / denominator
            log_probability += math.log(probability)
            steps += 1
        return math.exp(-log_probability / max(steps, 1))


def _coerce_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _parse_age_group(text: str) -> tuple[str, str]:
    match = _AGE_RE.search(text)
    if not match:
        return "", ""
    age_raw, _sex, group = match.groups()
    return age_raw, group


def _age_year(age_raw: str) -> int | None:
    if ";" not in age_raw:
        return None
    year_text = age_raw.split(";", 1)[0]
    try:
        return int(year_text)
    except ValueError:
        return None


def _age_months(age_raw: str) -> float | None:
    if ";" not in age_raw:
        return None
    try:
        years_text, months_text = age_raw.split(";", 1)
        months_token = months_text.split(".", 1)[0]
        return float(int(years_text) * 12 + int(months_token))
    except ValueError:
        return None


def _clean_chat_surface(text: str) -> str:
    text = _TIMING_RE.sub(" ", text)
    text = _ANGLE_RE.sub(r"\1", text)
    text = _BRACKET_RE.sub(" ", text)
    text = text.replace("\u0015", " ")
    text = text.replace("\t", " ")
    text = re.sub(r"\+\.\.\.", " ", text)
    text = re.sub(r"\+/{1,2}\.?", " ", text)
    text = re.sub(r"\+\s*,", " ", text)
    text = re.sub(r"&[+-][A-Za-z:']+", " ", text)
    text = re.sub(r"\b0[A-Za-z']+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_story_utterances(text: str) -> dict[str, list[str]]:
    stories: dict[str, list[str]] = {}
    current_story = ""
    for raw_line in text.splitlines():
        story_match = _STORY_RE.match(raw_line)
        if story_match:
            current_story = story_match.group(1)
            stories.setdefault(current_story, [])
            continue
        if not current_story or not raw_line.startswith("*CHI:\t"):
            continue
        content = raw_line.split("\t", 1)[1]
        cleaned = _clean_chat_surface(content)
        if cleaned:
            stories[current_story].append(cleaned)
    return stories


def _extract_comment_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw_line in text.splitlines():
        if not raw_line.startswith("@Comment:\t"):
            continue
        payload = raw_line.split("\t", 1)[1].strip()
        if "=" not in payload:
            continue
        key, value = payload.split("=", 1)
        fields[key.strip()] = value.strip()
    return fields


def _normalize_tokens(text: str) -> tuple[str, tuple[str, ...], int, float]:
    tokens = [
        token.lower()
        for token in _TOKEN_RE.findall(text)
        if not token.isdigit() and token.lower() not in _PLACEHOLDER_TOKENS
    ]
    normalized = " ".join(tokens)
    token_count = len(tokens)
    type_token_ratio = (len(set(tokens)) / token_count) if token_count else 0.0
    return normalized, tuple(tokens), token_count, type_token_ratio


def _median(values: pd.Series) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.median())


def _quantile(values: pd.Series, q: float) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.quantile(q))


def _share_within_interval(values: pd.Series, lower: float | None, upper: float | None) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty or lower is None or upper is None:
        return None
    return float(((clean >= lower) & (clean <= upper)).mean())


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator / denominator)


def _lcs_length(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def _rouge_l_f1(prediction: tuple[str, ...], reference: tuple[str, ...]) -> float:
    if not prediction or not reference:
        return 0.0
    lcs = _lcs_length(prediction, reference)
    precision = lcs / len(prediction)
    recall = lcs / len(reference)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _load_real_story_records(real_root: Path, age_years: int) -> list[NarrativeRecord]:
    records_by_child_story: dict[tuple[str, str], NarrativeRecord] = {}
    for path in sorted(real_root.rglob("*.cha")):
        match = _REAL_PATH_RE.search(path.name)
        if not match:
            continue

        text = path.read_text(encoding="utf-8", errors="replace")
        age_raw, group = _parse_age_group(text)
        if _age_year(age_raw) != age_years:
            continue

        child_id = match.group("child_id")
        story_map = _extract_story_utterances(text)
        path_story_id = match.group("story_id")
        if path_story_id:
            candidate_story_ids = [path_story_id] if path_story_id in _CANONICAL_STORY_IDS else []
        else:
            candidate_story_ids = sorted(story_id for story_id in story_map if story_id in _CANONICAL_STORY_IDS)

        for story_id in candidate_story_ids:
            utterances = story_map.get(story_id)
            if not utterances:
                if path_story_id and len(story_map) == 1:
                    utterances = next(iter(story_map.values()))
                else:
                    continue

            raw_story_text = " ".join(utterances).strip()
            normalized_text, tokens, token_count, type_token_ratio = _normalize_tokens(raw_story_text)
            if not tokens:
                continue

            record = NarrativeRecord(
                source="real",
                group=group,
                story_id=story_id,
                child_id=child_id,
                transcript_path=str(path.resolve()),
                age_raw=age_raw,
                age_months=_age_months(age_raw),
                stage="",
                bundle_id="",
                pair_id="",
                replicate_id=None,
                round_id="",
                source_child_id=child_id,
                raw_story_text=raw_story_text,
                normalized_text=normalized_text,
                tokens=tokens,
                token_count=token_count,
                type_token_ratio=type_token_ratio,
            )

            key = (child_id, story_id)
            existing = records_by_child_story.get(key)
            # Prefer explicit story-split files over bundle files when both exist.
            if existing is None or path_story_id is not None:
                records_by_child_story[key] = record

    return list(records_by_child_story.values())


def _load_synthetic_story_records(synthetic_root: Path) -> list[NarrativeRecord]:
    records: list[NarrativeRecord] = []
    for path in sorted(synthetic_root.rglob("*.cha")):
        match = _SYNTHETIC_STORY_PATH_RE.search(path.name)
        if not match:
            continue

        text = path.read_text(encoding="utf-8", errors="replace")
        comment_fields = _extract_comment_fields(text)
        age_raw, header_group = _parse_age_group(text)
        story_map = _extract_story_utterances(text)

        story_id = match.group("story_id")
        if story_id not in _CANONICAL_STORY_IDS:
            continue
        group = header_group or match.group("group")
        utterances = story_map.get(story_id)
        if not utterances and len(story_map) == 1:
            utterances = next(iter(story_map.values()))
        if not utterances:
            continue

        raw_story_text = " ".join(utterances).strip()
        normalized_text, tokens, token_count, type_token_ratio = _normalize_tokens(raw_story_text)
        if not tokens:
            continue

        pair_id = comment_fields.get("pair_id", f"pair_sli{match.group('sli_child_id')}_td{match.group('td_child_id')}")
        bundle_id = comment_fields.get("bundle_id", path.stem.rsplit(f"_{story_id}_{group}", 1)[0])
        replicate_text = comment_fields.get("replicate_id", match.group("replicate_id"))
        source_child_id = comment_fields.get(
            "source_child_id",
            match.group("td_child_id") if group == "TD" else match.group("sli_child_id"),
        )

        try:
            replicate_id = int(replicate_text)
        except ValueError:
            replicate_id = None

        records.append(
            NarrativeRecord(
                source="synthetic",
                group=group,
                story_id=story_id,
                child_id=bundle_id,
                transcript_path=str(path.resolve()),
                age_raw=age_raw,
                age_months=_age_months(age_raw),
                stage=comment_fields.get("stage", path.parent.name),
                bundle_id=bundle_id,
                pair_id=pair_id,
                replicate_id=replicate_id,
                round_id=comment_fields.get("round_id", ""),
                source_child_id=source_child_id,
                raw_story_text=raw_story_text,
                normalized_text=normalized_text,
                tokens=tokens,
                token_count=token_count,
                type_token_ratio=type_token_ratio,
            )
        )
    return records


def _build_metric_rows(
    *,
    synthetic_df: pd.DataFrame,
    real_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    synthetic_rows: list[dict[str, object]] = []
    real_rows: list[dict[str, object]] = []

    for (group, story_id), synthetic_slice in synthetic_df.groupby(["group", "story_id"], sort=True):
        real_slice = real_df[(real_df["group"] == group) & (real_df["story_id"] == story_id)].copy()
        if synthetic_slice.empty or real_slice.empty:
            continue

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        real_matrix = vectorizer.fit_transform(real_slice["normalized_text"].tolist())
        synthetic_matrix = vectorizer.transform(synthetic_slice["normalized_text"].tolist())
        synthetic_to_real = cosine_similarity(synthetic_matrix, real_matrix)
        real_to_real = cosine_similarity(real_matrix, real_matrix)

        real_token_lists = real_slice["tokens"].tolist()
        real_reference_model = AdditiveBigramLanguageModel(real_token_lists)

        real_baseline_nearest_tfidf: list[float] = []
        real_baseline_nearest_rouge: list[float] = []
        real_baseline_bigram_ppl: list[float] = []
        real_baseline_unigram_ppl: list[float] = []

        for real_index, real_row in enumerate(real_slice.to_dict("records")):
            similarities = np.asarray(real_to_real[real_index], dtype=float).copy()
            similarities[real_index] = -1.0
            nearest_index = int(np.argmax(similarities))
            valid_similarities = similarities[similarities >= 0.0]

            candidate_rouge = [
                _rouge_l_f1(real_row["tokens"], reference_tokens)
                for candidate_index, reference_tokens in enumerate(real_token_lists)
                if candidate_index != real_index
            ]
            leave_one_out_tokens = [
                reference_tokens
                for candidate_index, reference_tokens in enumerate(real_token_lists)
                if candidate_index != real_index
            ]
            leave_one_out_model = AdditiveBigramLanguageModel(leave_one_out_tokens)

            nearest_tfidf = float(similarities[nearest_index]) if valid_similarities.size else None
            mean_tfidf = float(valid_similarities.mean()) if valid_similarities.size else None
            nearest_rouge = max(candidate_rouge) if candidate_rouge else None
            mean_rouge = float(np.mean(candidate_rouge)) if candidate_rouge else None
            unigram_ppl = leave_one_out_model.unigram_perplexity(real_row["tokens"])
            bigram_ppl = leave_one_out_model.bigram_perplexity(real_row["tokens"])

            real_baseline_nearest_tfidf.append(nearest_tfidf if nearest_tfidf is not None else np.nan)
            real_baseline_nearest_rouge.append(nearest_rouge if nearest_rouge is not None else np.nan)
            real_baseline_unigram_ppl.append(unigram_ppl)
            real_baseline_bigram_ppl.append(bigram_ppl)

            real_rows.append(
                {
                    "source": "real_baseline",
                    "group": group,
                    "story_id": story_id,
                    "child_id": real_row["child_id"],
                    "transcript_path": real_row["transcript_path"],
                    "token_count": int(real_row["token_count"]),
                    "type_token_ratio": float(real_row["type_token_ratio"]),
                    "n_reference_real_texts": int(len(real_slice) - 1),
                    "nearest_real_child_id": real_slice.iloc[nearest_index]["child_id"] if valid_similarities.size else "",
                    "nearest_real_path": real_slice.iloc[nearest_index]["transcript_path"] if valid_similarities.size else "",
                    "nearest_tfidf_cosine": nearest_tfidf,
                    "mean_tfidf_cosine_to_real": mean_tfidf,
                    "nearest_rouge_l_f1": nearest_rouge,
                    "mean_rouge_l_f1_to_real": mean_rouge,
                    "unigram_perplexity_vs_real": unigram_ppl,
                    "bigram_perplexity_vs_real": bigram_ppl,
                }
            )

        baseline_nearest_tfidf_median = _median(pd.Series(real_baseline_nearest_tfidf))
        baseline_nearest_rouge_median = _median(pd.Series(real_baseline_nearest_rouge))
        baseline_bigram_ppl_median = _median(pd.Series(real_baseline_bigram_ppl))
        baseline_unigram_ppl_median = _median(pd.Series(real_baseline_unigram_ppl))
        real_token_count_mean = float(real_slice["token_count"].mean())
        real_ttr_mean = float(real_slice["type_token_ratio"].mean())

        for synthetic_index, synthetic_row in enumerate(synthetic_slice.to_dict("records")):
            similarities = np.asarray(synthetic_to_real[synthetic_index], dtype=float)
            nearest_index = int(np.argmax(similarities))
            rouge_scores = [
                _rouge_l_f1(synthetic_row["tokens"], reference_tokens)
                for reference_tokens in real_token_lists
            ]
            unigram_ppl = real_reference_model.unigram_perplexity(synthetic_row["tokens"])
            bigram_ppl = real_reference_model.bigram_perplexity(synthetic_row["tokens"])

            synthetic_rows.append(
                {
                    "source": "synthetic",
                    "group": group,
                    "story_id": story_id,
                    "child_id": synthetic_row["child_id"],
                    "bundle_id": synthetic_row["bundle_id"],
                    "pair_id": synthetic_row["pair_id"],
                    "replicate_id": synthetic_row["replicate_id"],
                    "round_id": synthetic_row["round_id"],
                    "stage": synthetic_row["stage"],
                    "source_child_id": synthetic_row["source_child_id"],
                    "transcript_path": synthetic_row["transcript_path"],
                    "token_count": int(synthetic_row["token_count"]),
                    "type_token_ratio": float(synthetic_row["type_token_ratio"]),
                    "n_reference_real_texts": int(len(real_slice)),
                    "nearest_real_child_id": real_slice.iloc[nearest_index]["child_id"],
                    "nearest_real_path": real_slice.iloc[nearest_index]["transcript_path"],
                    "nearest_tfidf_cosine": float(similarities[nearest_index]),
                    "mean_tfidf_cosine_to_real": float(similarities.mean()),
                    "nearest_rouge_l_f1": float(max(rouge_scores)),
                    "mean_rouge_l_f1_to_real": float(np.mean(rouge_scores)),
                    "unigram_perplexity_vs_real": unigram_ppl,
                    "bigram_perplexity_vs_real": bigram_ppl,
                    "nearest_tfidf_minus_real_baseline_median": (
                        float(similarities[nearest_index]) - baseline_nearest_tfidf_median
                        if baseline_nearest_tfidf_median is not None
                        else None
                    ),
                    "nearest_rouge_minus_real_baseline_median": (
                        float(max(rouge_scores)) - baseline_nearest_rouge_median
                        if baseline_nearest_rouge_median is not None
                        else None
                    ),
                    "bigram_perplexity_over_real_baseline_median": _safe_ratio(
                        bigram_ppl,
                        baseline_bigram_ppl_median,
                    ),
                    "unigram_perplexity_over_real_baseline_median": _safe_ratio(
                        unigram_ppl,
                        baseline_unigram_ppl_median,
                    ),
                    "token_count_over_real_mean": _safe_ratio(float(synthetic_row["token_count"]), real_token_count_mean),
                    "type_token_ratio_over_real_mean": _safe_ratio(
                        float(synthetic_row["type_token_ratio"]),
                        real_ttr_mean,
                    ),
                }
            )

    synthetic_metrics = pd.DataFrame(synthetic_rows)
    real_baseline = pd.DataFrame(real_rows)
    return synthetic_metrics, real_baseline


def _build_slice_summary(synthetic_metrics: pd.DataFrame, real_baseline: pd.DataFrame) -> pd.DataFrame:
    if synthetic_metrics.empty or real_baseline.empty:
        return pd.DataFrame()

    grouping_specs = [
        (["group", "story_id"], "story"),
        (["group"], "group"),
        ([], "overall"),
    ]
    summary_rows: list[dict[str, object]] = []

    for group_columns, level in grouping_specs:
        if group_columns:
            synthetic_groups = synthetic_metrics.groupby(group_columns, sort=True)
        else:
            synthetic_groups = [(tuple(), synthetic_metrics)]

        for group_key, synthetic_subset in synthetic_groups:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            filters = dict(zip(group_columns, group_key))
            real_subset = real_baseline.copy()
            for column, value in filters.items():
                real_subset = real_subset[real_subset[column] == value]
            if synthetic_subset.empty or real_subset.empty:
                continue

            real_tfidf_q25 = _quantile(real_subset["nearest_tfidf_cosine"], 0.25)
            real_tfidf_q75 = _quantile(real_subset["nearest_tfidf_cosine"], 0.75)
            real_rouge_q25 = _quantile(real_subset["nearest_rouge_l_f1"], 0.25)
            real_rouge_q75 = _quantile(real_subset["nearest_rouge_l_f1"], 0.75)
            real_bigram_q25 = _quantile(real_subset["bigram_perplexity_vs_real"], 0.25)
            real_bigram_q75 = _quantile(real_subset["bigram_perplexity_vs_real"], 0.75)

            summary_rows.append(
                {
                    "summary_level": level,
                    "group": filters.get("group", "ALL"),
                    "story_id": filters.get("story_id", "ALL"),
                    "n_synthetic": int(len(synthetic_subset)),
                    "n_real_baseline": int(len(real_subset)),
                    "synthetic_token_count_median": _median(synthetic_subset["token_count"]),
                    "real_token_count_median": _median(real_subset["token_count"]),
                    "synthetic_type_token_ratio_median": _median(synthetic_subset["type_token_ratio"]),
                    "real_type_token_ratio_median": _median(real_subset["type_token_ratio"]),
                    "synthetic_nearest_tfidf_median": _median(synthetic_subset["nearest_tfidf_cosine"]),
                    "real_nearest_tfidf_median": _median(real_subset["nearest_tfidf_cosine"]),
                    "synthetic_nearest_tfidf_within_real_iqr_share": _share_within_interval(
                        synthetic_subset["nearest_tfidf_cosine"],
                        real_tfidf_q25,
                        real_tfidf_q75,
                    ),
                    "synthetic_nearest_rouge_l_median": _median(synthetic_subset["nearest_rouge_l_f1"]),
                    "real_nearest_rouge_l_median": _median(real_subset["nearest_rouge_l_f1"]),
                    "synthetic_nearest_rouge_l_within_real_iqr_share": _share_within_interval(
                        synthetic_subset["nearest_rouge_l_f1"],
                        real_rouge_q25,
                        real_rouge_q75,
                    ),
                    "synthetic_bigram_perplexity_median": _median(synthetic_subset["bigram_perplexity_vs_real"]),
                    "real_bigram_perplexity_median": _median(real_subset["bigram_perplexity_vs_real"]),
                    "synthetic_bigram_perplexity_within_real_iqr_share": _share_within_interval(
                        synthetic_subset["bigram_perplexity_vs_real"],
                        real_bigram_q25,
                        real_bigram_q75,
                    ),
                    "synthetic_bigram_perplexity_over_real_median": _safe_ratio(
                        _median(synthetic_subset["bigram_perplexity_vs_real"]),
                        _median(real_subset["bigram_perplexity_vs_real"]),
                    ),
                }
            )

    return pd.DataFrame(summary_rows)


def _run_detectability_cv(frame: pd.DataFrame, scope: str) -> tuple[pd.DataFrame, dict[str, object]]:
    if frame.empty or frame["source"].nunique() < 2:
        return pd.DataFrame(), {"scope": scope, "ran": False, "reason": "Need both real and synthetic texts."}

    strata = (
        frame["group"].astype(str)
        + "|"
        + frame["story_id"].astype(str)
        + "|"
        + frame["source"].astype(str)
    )
    min_stratum = int(strata.value_counts().min())
    n_splits = min(5, min_stratum)
    if n_splits < 2:
        return pd.DataFrame(), {
            "scope": scope,
            "ran": False,
            "reason": "Not enough examples per stratum for cross-validation.",
        }

    predictions = np.zeros(len(frame), dtype=float)
    folds = np.full(len(frame), -1, dtype=int)
    labels = (frame["source"].astype(str) == "synthetic").astype(int).to_numpy()
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    for fold_id, (train_index, test_index) in enumerate(splitter.split(frame["normalized_text"], strata)):
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
        train_text = frame.iloc[train_index]["normalized_text"].tolist()
        test_text = frame.iloc[test_index]["normalized_text"].tolist()
        x_train = vectorizer.fit_transform(train_text)
        x_test = vectorizer.transform(test_text)

        model = LogisticRegression(max_iter=2000, class_weight="balanced")
        model.fit(x_train, labels[train_index])
        predictions[test_index] = model.predict_proba(x_test)[:, 1]
        folds[test_index] = fold_id

    prediction_frame = frame[
        [
            "source",
            "group",
            "story_id",
            "child_id",
            "bundle_id",
            "pair_id",
            "replicate_id",
            "round_id",
            "transcript_path",
        ]
    ].copy()
    prediction_frame["scope"] = scope
    prediction_frame["cv_fold"] = folds
    prediction_frame["synthetic_probability"] = predictions

    summary = {
        "scope": scope,
        "ran": True,
        "n_rows": int(len(frame)),
        "n_splits": int(n_splits),
        "roc_auc": float(roc_auc_score(labels, predictions)),
        "accuracy_at_0_5": float(accuracy_score(labels, predictions >= 0.5)),
        "mean_predicted_synthetic_prob_for_real": float(predictions[labels == 0].mean()),
        "mean_predicted_synthetic_prob_for_synthetic": float(predictions[labels == 1].mean()),
    }
    return prediction_frame, summary


def build_generated_text_evaluation(
    *,
    synthetic_root: str | Path,
    real_root: str | Path,
    output_dir: str | Path,
    age_years: int = 4,
) -> int:
    synthetic_path = _coerce_path(synthetic_root)
    real_path = _coerce_path(real_root)
    out_dir = _coerce_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records_csv = out_dir / "story_text_records.csv"
    synthetic_metrics_csv = out_dir / "synthetic_text_metrics.csv"
    real_baseline_csv = out_dir / "real_text_baseline.csv"
    slice_summary_csv = out_dir / "slice_metric_summary.csv"
    detectability_predictions_csv = out_dir / "detectability_cv_predictions.csv"
    detectability_summary_csv = out_dir / "detectability_summary.csv"
    summary_json = out_dir / "evaluation_summary.json"

    real_records = _load_real_story_records(real_path, age_years=age_years)
    synthetic_records = _load_synthetic_story_records(synthetic_path)
    if not real_records:
        raise ValueError(f"No real {age_years}-year-old story-level CHAT files were found under {real_path}")
    if not synthetic_records:
        raise ValueError(f"No synthetic story-level CHAT files were found under {synthetic_path}")

    records = real_records + synthetic_records
    records_df = pd.DataFrame([record.to_csv_row() for record in records]).sort_values(
        ["source", "group", "story_id", "child_id", "replicate_id"],
        kind="stable",
    ).reset_index(drop=True)
    records_df.to_csv(records_csv, index=False)

    analysis_df = pd.DataFrame(
        [
            {
                **record.to_csv_row(),
                "tokens": record.tokens,
            }
            for record in records
        ]
    )
    real_df = analysis_df[analysis_df["source"] == "real"].copy().reset_index(drop=True)
    synthetic_df = analysis_df[analysis_df["source"] == "synthetic"].copy().reset_index(drop=True)

    synthetic_metrics, real_baseline = _build_metric_rows(synthetic_df=synthetic_df, real_df=real_df)
    synthetic_metrics.to_csv(synthetic_metrics_csv, index=False)
    real_baseline.to_csv(real_baseline_csv, index=False)

    slice_summary = _build_slice_summary(synthetic_metrics=synthetic_metrics, real_baseline=real_baseline)
    slice_summary.to_csv(slice_summary_csv, index=False)

    detectability_frames: list[pd.DataFrame] = []
    detectability_summaries: list[dict[str, object]] = []
    for scope, frame in [
        ("overall", records_df),
        ("TD", records_df[records_df["group"] == "TD"].copy()),
        ("SLI", records_df[records_df["group"] == "SLI"].copy()),
    ]:
        prediction_frame, summary = _run_detectability_cv(frame=frame, scope=scope)
        if not prediction_frame.empty:
            detectability_frames.append(prediction_frame)
        detectability_summaries.append(summary)

    detectability_predictions = (
        pd.concat(detectability_frames, ignore_index=True) if detectability_frames else pd.DataFrame()
    )
    detectability_predictions.to_csv(detectability_predictions_csv, index=False)
    pd.DataFrame(detectability_summaries).to_csv(detectability_summary_csv, index=False)

    summary = {
        "synthetic_root": str(synthetic_path),
        "real_root": str(real_path),
        "age_years": int(age_years),
        "n_story_text_rows_total": int(len(records_df)),
        "n_story_text_rows_real": int((records_df["source"] == "real").sum()),
        "n_story_text_rows_synthetic": int((records_df["source"] == "synthetic").sum()),
        "groups": sorted(records_df["group"].astype(str).unique().tolist()),
        "stories": sorted(records_df["story_id"].astype(str).unique().tolist()),
        "story_counts_by_source_group": (
            records_df.groupby(["source", "group"]).size().rename("n_rows").reset_index().to_dict("records")
        ),
        "real_story_counts_by_group_story": (
            records_df[records_df["source"] == "real"]
            .groupby(["group", "story_id"])
            .size()
            .rename("n_rows")
            .reset_index()
            .to_dict("records")
        ),
        "synthetic_story_counts_by_group_story": (
            records_df[records_df["source"] == "synthetic"]
            .groupby(["group", "story_id"])
            .size()
            .rename("n_rows")
            .reset_index()
            .to_dict("records")
        ),
        "outputs": {
            "story_text_records_csv": str(records_csv),
            "synthetic_text_metrics_csv": str(synthetic_metrics_csv),
            "real_text_baseline_csv": str(real_baseline_csv),
            "slice_metric_summary_csv": str(slice_summary_csv),
            "detectability_predictions_csv": str(detectability_predictions_csv),
            "detectability_summary_csv": str(detectability_summary_csv),
        },
        "detectability": detectability_summaries,
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote story text records to {records_csv}")
    print(f"Wrote synthetic text metrics to {synthetic_metrics_csv}")
    print(f"Wrote real text baseline metrics to {real_baseline_csv}")
    print(f"Wrote slice metric summary to {slice_summary_csv}")
    print(f"Wrote detectability predictions to {detectability_predictions_csv}")
    print(f"Wrote detectability summary to {detectability_summary_csv}")
    print(f"Wrote evaluation summary to {summary_json}")
    return 0
