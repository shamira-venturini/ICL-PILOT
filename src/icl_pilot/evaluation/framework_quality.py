"""Traceable corpus diagnostics with a correctly defined Fréchet distance.

The pinned toolkit's so-called FID is a centroid-cosine score. That function is
not used here. The retained distance compares Gaussian first/second moments in
one fixed sentence-embedding space; it is exploratory, not a clinical measure.
"""
from __future__ import annotations

import dataclasses
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .framework_classifier import file_digest
from .framework_data import _write_json


def frechet_distance(real_features, synthetic_features):
    """Squared mean distance + Bures covariance distance, with ddof=1 moments."""
    real = np.asarray(real_features, dtype=np.float64)
    synthetic = np.asarray(synthetic_features, dtype=np.float64)
    if real.ndim != 2 or synthetic.ndim != 2 or real.shape[1] != synthetic.shape[1] or min(len(real), len(synthetic)) < 2:
        raise ValueError("Fréchet distance requires >=2 samples per source in the same embedding space.")
    if not np.isfinite(real).all() or not np.isfinite(synthetic).all():
        raise ValueError("Nonfinite embedding features.")
    mu_r, mu_s = real.mean(axis=0), synthetic.mean(axis=0)
    cov_r, cov_s = np.atleast_2d(np.cov(real, rowvar=False)), np.atleast_2d(np.cov(synthetic, rowvar=False))
    eigenvalues, eigenvectors = np.linalg.eigh((cov_r + cov_r.T) / 2)
    root = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))) @ eigenvectors.T
    middle = root @ cov_s @ root
    middle_eigenvalues = np.linalg.eigvalsh((middle + middle.T) / 2)
    result = np.sum((mu_r - mu_s) ** 2) + np.trace(cov_r) + np.trace(cov_s) - 2 * np.sqrt(np.maximum(middle_eigenvalues, 0)).sum()
    tolerance = 1e-7 * max(1.0, np.trace(cov_r) + np.trace(cov_s))
    if result < -tolerance:
        raise ArithmeticError("Numerically unstable Fréchet covariance calculation.")
    return float(max(0.0, result))


def run_text_quality(prepared):
    """Run pinned GPT-2 perplexity and fixed-encoder Fréchet per generator."""
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    from synthtexteval.eval.text_quality import metrics as toolkit
    from synthtexteval.eval.text_quality.arguments import LMArgs
    from synthtexteval.eval.text_quality import perplexity as perplexity_module

    settings = prepared.protocol["text_quality"]
    perplexity_path = snapshot_download(repo_id=settings["perplexity_model_id"], revision=settings["perplexity_revision"],
        allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"])
    embedding_path = snapshot_download(repo_id=settings["embedding_model_id"], revision=settings["embedding_revision"],
        ignore_patterns=["*.onnx", "onnx/*", "openvino/*", "*.h5", "*.msgpack", "pytorch_model.bin"])
    encoder = SentenceTransformer(embedding_path)
    evaluator_class = getattr(toolkit, "QualEval", None) or getattr(toolkit, "TextQualityEval", None)
    if evaluator_class is None:
        raise ImportError("Pinned SynthTextEval has no supported text-quality evaluator.")
    real_features = None
    summaries = []
    for model, sample in prepared.samples.items():
        target = sample["path"]
        comparison = sample["comparison"]
        arguments = SimpleNamespace(perplexity=LMArgs(source_text_column="source", ref_text_column="reference", model_name=perplexity_path))
        evaluator = evaluator_class(arguments)
        evaluator.calculate_perplexity(comparison)
        results = {key: value for key, value in evaluator.return_results().items() if value}
        if real_features is None:
            real_features = encoder.encode(sample["real"]["output_text"].tolist(), show_progress_bar=True)
        synthetic_features = encoder.encode(sample["synthetic"]["output_text"].tolist(), show_progress_bar=True)
        results["frechet_embedding_distance"] = frechet_distance(real_features, synthetic_features)
        # Keep exact representations so moment calculations can be repeated
        # without re-downloading or re-running the encoder.
        np.save(target / "real_embeddings.npy", real_features)
        np.save(target / "synthetic_embeddings.npy", synthetic_features)
        rows = []
        for source, frame in (("real", sample["real"]), ("synthetic", sample["synthetic"])):
            for row in frame.itertuples():
                n_tokens = len(encoder.tokenizer(row.output_text, add_special_tokens=True, truncation=False)["input_ids"])
                rows.append({"sample_id": row.sample_id, "source": source, "group": row.group,
                    "cohort_year": row.cohort_year, "embedding_token_count": n_tokens,
                    "embedding_max_length": encoder.max_seq_length, "embedding_was_truncated": n_tokens > encoder.max_seq_length})
        pd.DataFrame(rows).to_csv(target / "embedding_tokenization_audit.csv", index=False)
        _write_json(target / "text_quality_configuration.json", {
            "model_key": model, "scope": "descriptive full-corpus comparison; no clinical-validity inference",
            "checkpoints": settings, "perplexity_arguments": dataclasses.asdict(arguments.perplexity),
            "perplexity_max_length": 1024, "perplexity_summary": "toolkit arithmetic mean of per-story perplexities",
            "embedding_max_length": encoder.max_seq_length, "embedding_dimensions": int(np.asarray(real_features).shape[1]),
            "frechet_definition": "squared mean distance + covariance Bures distance; unbiased covariances (ddof=1)",
            "frechet_n_real": len(real_features), "frechet_n_synthetic": len(synthetic_features),
            "legacy_toolkit_fid": "not used: pinned implementation is centroid cosine, not Fréchet distance",
            "perplexity_module_sha256": file_digest(perplexity_module.__file__),
        })
        (target / "perplexity_wrapper_source.txt").write_text(inspect.getsource(perplexity_module.calculate_perplexity))
        _write_json(target / "text_quality_results.json", results)
        details = getattr(evaluator, "detailed_results", {})
        _write_json(target / "perplexity_detailed_results.json", {key: value for key, value in details.items() if value})
        summaries.append({"generator_key": model, "frechet_embedding_distance": results["frechet_embedding_distance"],
            "synthetic_perplexity": results.get("synth perplexity", [None])[-1],
            "real_perplexity": results.get("real perplexity", [None])[-1]})
    result = pd.DataFrame(summaries)
    result.to_csv(prepared.output_dir / "text_quality_by_generator.csv", index=False)
    return result
