"""Offline BERT length audit of definitive real stories; no classifier fitting."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import statistics
import math

from ..evaluation.story_text_exports import _real_records
from ..evaluation.framework_text import TEXT_POLICY
from .real_story_tokenizer_audit import representations


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class VocabularyReference:
    """Reconstruct the WordPiece pipeline from vocab.txt, without tokenizer.json."""
    def __init__(self, path):
        from tokenizers import BertWordPieceTokenizer
        self.backend = BertWordPieceTokenizer(str(path), lowercase=True)

    def __call__(self, text, **kwargs):
        return {"input_ids": self.backend.encode(text, add_special_tokens=kwargs["add_special_tokens"]).ids}


def measure(text, tokenizer, reference, limit):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Cannot audit empty or missing narrative text.")
    original = tokenizer(text, add_special_tokens=True, truncation=False, verbose=False)["input_ids"]
    independent = reference(text, add_special_tokens=True, truncation=False, verbose=False)["input_ids"]
    if original != independent:
        raise ValueError("BERT tokenizer and vocabulary-reconstructed token IDs disagree.")
    retained = tokenizer(text, add_special_tokens=True, truncation=True, max_length=limit, verbose=False)["input_ids"]
    if len(retained) != min(len(original), limit):
        raise ValueError("Observed BERT truncation length differs from the configured limit.")
    if tokenizer.num_special_tokens_to_add(pair=False) != 2:
        raise ValueError("Expected [CLS] and [SEP] for a single BERT story.")
    expected = original if len(original) <= limit else original[:limit-1] + [tokenizer.sep_token_id]
    if retained != expected:
        raise ValueError("BERT did not apply the recorded right-truncation policy.")
    removed = len(original) - len(retained)
    return {"original_tokens_including_special": len(original),
            "retained_tokens_including_special": len(retained),
            "removed_content_tokens": removed,
            "removed_content_fraction": removed / (len(original)-2),
            "would_truncate": len(original) > limit}


def summarize(rows):
    if not rows:
        raise ValueError("Cannot summarize an empty audit stratum.")
    lengths = sorted(r["original_tokens_including_special"] for r in rows)
    truncated = sum(r["would_truncate"] for r in rows)
    removed = sum(r["removed_content_tokens"] for r in rows)
    return {"stories": len(rows), "children": len({r["source_child_id"] for r in rows}),
            "median_tokens": statistics.median(lengths),
            "p95_nearest_rank": lengths[math.ceil(.95*len(lengths))-1],
            "max_tokens": max(lengths), "stories_above_limit": truncated,
            "percent_stories_above_limit": 100*truncated/len(rows),
            "removed_content_tokens": removed,
            "percent_content_tokens_removed": 100*removed/sum(n-2 for n in lengths)}


def audit(project_root, tokenizer_dir, output_dir):
    import pandas as pd
    from transformers import BertTokenizerFast

    root, tokenizer_dir, output_dir = map(lambda p: Path(p).resolve(), (project_root, tokenizer_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError("BERT audit output exists; choose a fresh directory.")
    gen_rel = "configs/generation/experiment.json"
    protocol_rel = "configs/evaluation/protocol.json"
    inputs_rel = "configs/evaluation/inputs.json"
    gen = json.loads((root/gen_rel).read_text())
    protocol = json.loads((root/protocol_rel).read_text())
    inputs = json.loads((root/inputs_rel).read_text())
    paths = {"generation_config": gen_rel, "framework_protocol": protocol_rel, "framework_inputs": inputs_rel,
             "enni_stories": gen["inputs"]["narratives_csv"],
             "enni_participants": gen["provenance"]["participant_manifest"],
             "roster": "configs/generation/"+gen["roster_csv"],
             "gillam": inputs["gillam_external_text_csv"], "gillam_audit": "docs/reproducibility/gillam.json"}
    hashes = {label: {"path": rel, "sha256": sha(root/rel)} for label, rel in paths.items()}
    source = root/paths["enni_stories"]
    qc = json.loads(source.with_suffix(".parsing_qc.json").read_text())
    if sha(source) != qc["output_csv_sha256"]:
        raise ValueError("ENNI stories differ from their extraction QC.")
    if sha(root/paths["gillam"]) != json.loads((root/paths["gillam_audit"]).read_text())["output_csv_sha256"]:
        raise ValueError("Gillam stories differ from their verified selection audit.")
    provenance = json.loads((tokenizer_dir/"download_manifest.json").read_text())
    settings = protocol["classifier"]
    if (provenance["model_id"], provenance["revision"]) != (settings["model_id"], settings["revision"]):
        raise ValueError("Tokenizer provenance differs from the pinned BERT classifier.")
    for item in provenance["files"]:
        if sha(tokenizer_dir/item["file"]) != item["sha256"]:
            raise ValueError("Tokenizer source hash mismatch.")
    tokenizer = BertTokenizerFast.from_pretrained(str(tokenizer_dir), local_files_only=True)
    reference = VocabularyReference(tokenizer_dir/"vocab.txt")
    limit = settings["max_length"]
    if not tokenizer.is_fast or tokenizer.truncation_side != "right":
        raise ValueError("Expected the pinned fast tokenizer and right truncation.")
    if limit != json.loads((tokenizer_dir/"config.json").read_text())["max_position_embeddings"]:
        raise ValueError("Classifier limit differs from pinned BERT position capacity.")
    roster = pd.read_csv(root/paths["roster"], dtype=str)
    prompt = {(row[field], group) for row in roster.to_dict("records")
              for field, group in (("td_child_id", "TD"), ("sli_child_id", "SLI"))}
    with source.open(newline="", encoding="utf-8") as handle:
        prepared = {(r["File_ID"], r["Group"], r["story_id"]): r for r in csv.DictReader(handle)}
    real, missing = _real_records(measures_path=root/paths["enni_participants"], stories_path=source,
                                  project_root=root, minimum_age_months=0, maximum_age_months=1000)
    if missing:
        raise ValueError("A canonical ENNI participant lacks stories.")
    texts = []
    for row in real:
        child = str(row["source_child_id"])
        identity = {"corpus": "ENNI", "source_child_id": child, "group": row["group"],
                    "story_id": row["story_id"], "age_months": int(row["age_months"]),
                    "cohort_year": row["cohort_year"], "prompt_child": (child, row["group"]) in prompt}
        example = representations(prepared[(child, row["group"], row["story_id"])])["prepared_text"]
        texts.extend([(identity, "generation_prepared", example), (identity, "classifier_prepared", row["story_text"])])
    external = pd.read_csv(root/paths["gillam"], dtype={"source_child_id": str})
    if not external["text_normalization"].eq(TEXT_POLICY).all() or external.duplicated(["source_child_id", "story_id"]).any():
        raise ValueError("Gillam has conflicting story identities or text preparation.")
    for row in external.to_dict("records"):
        identity = {"corpus": "Gillam", **{k: row[k] for k in ("source_child_id", "group", "story_id", "age_months", "cohort_year")},
                    "prompt_child": False}
        texts.append((identity, "classifier_prepared", row["story_text"]))
    records = [{**identity, "representation": representation,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(), **measure(text, tokenizer, reference, limit)}
               for identity, representation, text in texts]
    summaries = []
    for representation in ("generation_prepared", "classifier_prepared"):
        scopes = {"enni_all_saved": [r for r in records if r["representation"] == representation and r["corpus"] == "ENNI"],
                  "enni_study_4_to_7": [r for r in records if r["representation"] == representation and r["corpus"] == "ENNI" and 48 <= r["age_months"] < 96],
                  "enni_prompt_pair_pool": [r for r in records if r["representation"] == representation and r["corpus"] == "ENNI" and r["prompt_child"]]}
        if representation == "classifier_prepared":
            scopes["gillam_external"] = [r for r in records if r["corpus"] == "Gillam"]
        for scope, rows in scopes.items():
            strata = [("all", "all", rows)]
            strata += [(g, "all", [r for r in rows if r["group"] == g]) for g in ("TD", "SLI")]
            for age in sorted({r["cohort_year"] for r in rows}):
                strata.append(("all", age, [r for r in rows if r["cohort_year"] == age]))
                strata.extend((g, age, [r for r in rows if r["group"] == g and r["cohort_year"] == age]) for g in ("TD", "SLI"))
            summaries.extend({"scope": scope, "representation": representation, "group": group,
                              "cohort_year": age, **summarize(subset)} for group, age, subset in strata if subset)
    for item in hashes.values():
        if sha(root/item["path"]) != item["sha256"]:
            raise ValueError("Audit input changed during counting.")
    result = {"inputs": hashes, "tokenizer": provenance, "limit_including_special_tokens": limit,
              "special_tokens_per_story": 2, "truncation_side": tokenizer.truncation_side,
              "text_policy": TEXT_POLICY, "vocabulary_reconstruction_full_token_id_agreements": len(records),
              "cross_check": "same tokenizers engine; pinned tokenizer JSON versus WordPiece reconstructed from vocab.txt",
              "counting_used_full_untruncated_text": True, "model_weights_loaded": False,
              "classifier_fitting_performed": False, "generation_performed": False,
              "synthetic_stories_audited": 0, "synthetic_audit_status": "definitive generation remains pending",
              "software": {name: version(name) for name in ("transformers", "tokenizers", "pandas")},
              "implementation_sha256": sha(Path(__file__)), "summary": summaries}
    output_dir.mkdir(parents=True)
    for name, rows in (("per_story.csv", records), ("summary.csv", summaries),
                       ("over_limit.csv", [r for r in records if r["would_truncate"]])):
        fields = list((summaries if name == "summary.csv" else records)[0])
        with (output_dir/name).open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    (output_dir/"comparison.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.project_root, args.tokenizer_dir, args.output_dir)
    print(json.dumps([r for r in result["summary"] if r["representation"] == "classifier_prepared"
                      and r["group"] == "all" and r["cohort_year"] == "all"], indent=2))


if __name__ == "__main__":
    main()
