from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from scipy.stats import mannwhitneyu
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .bundled_text_perplexity import (
    _coerce_path,
    _iqr,
    _load_real_bundle_transcripts,
    _load_synthetic_bundle_transcripts,
    _median,
)


def _percentile_interval(values: np.ndarray, confidence: float = 0.95) -> list[float] | None:
    if values.size == 0:
        return None
    alpha = max(0.0, min(1.0, 1.0 - confidence))
    lower = float(np.quantile(values, alpha / 2.0))
    upper = float(np.quantile(values, 1.0 - alpha / 2.0))
    return [lower, upper]


def _bootstrap_similarity_summary(
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
        "median_real_similarity_ci": _percentile_interval(median_real, confidence=confidence),
        "median_synthetic_similarity_ci": _percentile_interval(median_synthetic, confidence=confidence),
        "median_difference_ci": _percentile_interval(median_difference, confidence=confidence),
        "median_ratio_ci": _percentile_interval(ratio_clean, confidence=confidence),
        "synthetic_share_within_real_iqr_ci": _percentile_interval(share_clean, confidence=confidence),
    }


def _resolve_sentence_bert_snapshot(model_name: str) -> Path:
    model_slug = model_name.replace("/", "--")
    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_slug}"
    snapshots_dir = cache_root / "snapshots"
    if not snapshots_dir.exists():
        raise FileNotFoundError(f"No cached snapshots found for {model_name} under {snapshots_dir}")

    candidates = [
        snapshot
        for snapshot in sorted(snapshots_dir.iterdir())
        if snapshot.is_dir() and (snapshot / "model.safetensors").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No snapshot with model.safetensors found for {model_name} under {snapshots_dir}")
    return candidates[-1]


class LocalSentenceBertEncoder:
    def __init__(self, model_name: str, batch_size: int = 16) -> None:
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.model_path = _resolve_sentence_bert_snapshot(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True)
        config = AutoConfig.from_pretrained(str(self.model_path), local_files_only=True)
        self.model = AutoModel.from_config(config)
        state = load_file(str(self.model_path / "model.safetensors"))
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ValueError(
                f"Unexpected state dict mismatch for {model_name}: missing={len(missing)} unexpected={len(unexpected)}"
            )
        self.model.eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        embeddings: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                summed = (outputs.last_hidden_state * attention_mask).sum(dim=1)
                counts = attention_mask.sum(dim=1).clamp(min=1)
                pooled = summed / counts
                normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.append(normalized.cpu().numpy())
        return np.vstack(embeddings) if embeddings else np.empty((0, 0), dtype=float)


def build_bundled_text_semantic_evaluation(
    *,
    synthetic_root: str | Path,
    real_root: str | Path,
    output_dir: str | Path,
    group: str,
    age_years: int = 4,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 16,
    bootstrap_reps: int = 10000,
    bootstrap_seed: int = 0,
    exclude_synthetic_source_children_from_real: bool = False,
) -> int:
    group = str(group).upper()
    synthetic_path = _coerce_path(synthetic_root)
    real_path = _coerce_path(real_root)
    out_dir = _coerce_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    group_slug = group.lower()

    transcript_manifest_csv = out_dir / "transcript_manifest.csv"
    real_baseline_csv = out_dir / f"real_{group_slug}_bundle_semantic_baseline.csv"
    synthetic_scores_csv = out_dir / f"synthetic_{group_slug}_bundle_semantic_scores.csv"
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

    encoder = LocalSentenceBertEncoder(model_name=model_name, batch_size=batch_size)
    real_texts = [transcript.cleaned_text for transcript in real_transcripts]
    synthetic_texts = [transcript.cleaned_text for transcript in synthetic_transcripts]
    real_embeddings = encoder.encode(real_texts)
    synthetic_embeddings = encoder.encode(synthetic_texts)

    real_similarity = cosine_similarity(real_embeddings, real_embeddings)
    synthetic_to_real_similarity = cosine_similarity(synthetic_embeddings, real_embeddings)
    real_centroid = real_embeddings.mean(axis=0, keepdims=True)
    real_centroid = real_centroid / np.linalg.norm(real_centroid, axis=1, keepdims=True)

    real_rows: list[dict[str, object]] = []
    for index, transcript in enumerate(real_transcripts):
        similarities = np.asarray(real_similarity[index], dtype=float).copy()
        similarities[index] = -1.0
        nearest_index = int(np.argmax(similarities))
        valid = similarities[similarities >= 0.0]
        centroid_similarity = float(np.dot(real_embeddings[index], real_centroid[0]))
        real_rows.append(
            {
                "transcript_id": transcript.transcript_id,
                "child_id": transcript.child_id,
                "transcript_path": transcript.transcript_path,
                "token_count": transcript.token_count,
                "utterance_count": transcript.utterance_count,
                "nearest_real_child_id": real_transcripts[nearest_index].child_id if valid.size else "",
                "nearest_real_path": real_transcripts[nearest_index].transcript_path if valid.size else "",
                "nearest_real_cosine_similarity": float(similarities[nearest_index]) if valid.size else None,
                "mean_cosine_similarity_to_real": float(valid.mean()) if valid.size else None,
                "cosine_similarity_to_real_centroid": centroid_similarity,
            }
        )

    synthetic_rows: list[dict[str, object]] = []
    for index, transcript in enumerate(synthetic_transcripts):
        similarities = np.asarray(synthetic_to_real_similarity[index], dtype=float)
        nearest_index = int(np.argmax(similarities))
        centroid_similarity = float(np.dot(synthetic_embeddings[index], real_centroid[0]))
        synthetic_rows.append(
            {
                "transcript_id": transcript.transcript_id,
                "child_id": transcript.child_id,
                "pair_id": transcript.pair_id,
                "replicate_id": transcript.replicate_id,
                "source_child_id": transcript.source_child_id,
                "transcript_path": transcript.transcript_path,
                "token_count": transcript.token_count,
                "utterance_count": transcript.utterance_count,
                "nearest_real_child_id": real_transcripts[nearest_index].child_id,
                "nearest_real_path": real_transcripts[nearest_index].transcript_path,
                "nearest_real_cosine_similarity": float(similarities[nearest_index]),
                "mean_cosine_similarity_to_real": float(similarities.mean()),
                "cosine_similarity_to_real_centroid": centroid_similarity,
            }
        )

    real_baseline = pd.DataFrame(real_rows).sort_values("transcript_id").reset_index(drop=True)
    synthetic_scores = pd.DataFrame(synthetic_rows).sort_values("transcript_id").reset_index(drop=True)
    real_baseline.to_csv(real_baseline_csv, index=False)
    synthetic_scores.to_csv(synthetic_scores_csv, index=False)

    real_nearest = real_baseline["nearest_real_cosine_similarity"]
    synthetic_nearest = synthetic_scores["nearest_real_cosine_similarity"]
    real_q1, real_q3 = _iqr(real_nearest)
    mann_whitney = mannwhitneyu(synthetic_nearest, real_nearest, alternative="two-sided")
    synthetic_within_real_iqr = (
        float(((synthetic_nearest >= real_q1) & (synthetic_nearest <= real_q3)).mean())
        if real_q1 is not None and real_q3 is not None
        else None
    )
    bootstrap_summary = _bootstrap_similarity_summary(
        real_nearest,
        synthetic_nearest,
        reps=max(1000, int(bootstrap_reps)),
        seed=int(bootstrap_seed),
    )

    summary = {
        "synthetic_root": str(synthetic_path),
        "real_root": str(real_path),
        "age_years": int(age_years),
        "embedding_model": {
            "name": model_name,
            "resolved_snapshot": str(encoder.model_path),
            "pooling": "attention-masked mean pooling over last hidden state",
            "normalized_embeddings": True,
            "batch_size": int(batch_size),
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
            "median_nearest_real_cosine_similarity": _median(real_nearest),
            "mean_nearest_real_cosine_similarity": float(real_nearest.mean()),
            "q1_nearest_real_cosine_similarity": real_q1,
            "q3_nearest_real_cosine_similarity": real_q3,
            "min_nearest_real_cosine_similarity": float(real_nearest.min()),
            "max_nearest_real_cosine_similarity": float(real_nearest.max()),
        },
        "synthetic_distribution": {
            "median_nearest_real_cosine_similarity": _median(synthetic_nearest),
            "mean_nearest_real_cosine_similarity": float(synthetic_nearest.mean()),
            "q1_nearest_real_cosine_similarity": _iqr(synthetic_nearest)[0],
            "q3_nearest_real_cosine_similarity": _iqr(synthetic_nearest)[1],
            "min_nearest_real_cosine_similarity": float(synthetic_nearest.min()),
            "max_nearest_real_cosine_similarity": float(synthetic_nearest.max()),
        },
        "comparison": {
            "median_difference_synthetic_minus_real": (
                float(_median(synthetic_nearest) - _median(real_nearest))
                if _median(synthetic_nearest) is not None and _median(real_nearest) is not None
                else None
            ),
            "median_ratio_synthetic_over_real": (
                float(_median(synthetic_nearest) / _median(real_nearest))
                if _median(real_nearest) not in (None, 0)
                else None
            ),
            "synthetic_share_within_real_iqr": synthetic_within_real_iqr,
            "mann_whitney_u": float(mann_whitney.statistic),
            "mann_whitney_p_value": float(mann_whitney.pvalue),
        },
        "bootstrap_uncertainty": bootstrap_summary,
        "outputs": {
            "transcript_manifest_csv": str(transcript_manifest_csv),
            "real_baseline_csv": str(real_baseline_csv),
            "synthetic_scores_csv": str(synthetic_scores_csv),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote transcript manifest to {transcript_manifest_csv}")
    print(f"Wrote real semantic baseline scores to {real_baseline_csv}")
    print(f"Wrote synthetic semantic scores to {synthetic_scores_csv}")
    print(f"Wrote summary to {summary_json}")
    return 0
