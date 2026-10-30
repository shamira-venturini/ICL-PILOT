"""Offline, read-only token-length audit using explicitly supplied tokenizer files."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from ..chat.text_cleaning import prepare_extracted_story


REPRESENTATIONS = ("prepared_text",)


def representations(row: dict) -> dict:
    text = row["story_text"]
    try:
        prepared = prepare_extracted_story(text, row["story_units_json"]).text
    except ValueError as error:
        if "no spoken text" not in str(error):
            raise
        prepared = None
    return {"prepared_text": prepared}


def summarize(rows: list[dict], column: str, ceiling: int) -> dict:
    valid = [row for row in rows if row[column] is not None]
    values = sorted(row[column] for row in valid)
    maximum = max(values) if values else None
    return {
        "stories": len(rows), "counted": len(values), "unusable": len(rows) - len(values),
        "median_tokens": statistics.median(values) if values else None,
        "p95_nearest_rank": values[math.ceil(.95 * len(values)) - 1] if values else None,
        "max_tokens": maximum,
        "ceiling_minus_max": ceiling - maximum if maximum is not None else None,
        "above_ceiling": sum(value > ceiling for value in values),
        "at_or_above_ceiling": sum(value >= ceiling for value in values),
        "maxima": [{key: row[key] for key in ("File_ID", "Group", "Age_Raw", "story_id")}
                   for row in valid if row[column] == maximum],
    }


def audit(source_csv: Path, roster_csv: Path, tokenizer_files: dict, output_dir: Path,
          *, ceiling: int = 640, gemma_sentencepiece: Path | None = None) -> dict:
    from tokenizers import Tokenizer
    if output_dir.exists():
        raise FileExistsError("Tokenizer audit output already exists; choose a fresh directory.")
    if type(ceiling) is not int or ceiling < 1 or not tokenizer_files:
        raise ValueError("Supply tokenizer files and a positive ceiling.")
    source_hash = hashlib.sha256(source_csv.read_bytes()).hexdigest()
    qc = json.loads(source_csv.with_suffix(".parsing_qc.json").read_text())
    if source_hash != qc["output_csv_sha256"]:
        raise ValueError("Source CSV does not match its frozen extraction QC.")
    with source_csv.open(newline="", encoding="utf-8") as handle:
        sources = list(csv.DictReader(handle))
    with roster_csv.open(newline="", encoding="utf-8") as handle:
        roster = list(csv.DictReader(handle))
    prompt_children = {(row[field], group) for row in roster
                       for field, group in (("td_child_id", "TD"), ("sli_child_id", "SLI"))}
    tokenizers = {}
    for label, path in tokenizer_files.items():
        tokenizer = Tokenizer.from_file(str(path))
        tokenizer.no_truncation()
        tokenizer.no_padding()
        tokenizers[label] = tokenizer
    independent = None
    if gemma_sentencepiece is not None:
        import sentencepiece
        independent = sentencepiece.SentencePieceProcessor(model_file=str(gemma_sentencepiece))
    records, agreements = [], 0
    for row in sources:
        record = {key: row[key] for key in ("File_ID", "Group", "Age_Raw", "Age_Month", "story_id")}
        months = int(row["Age_Month"]) if row["Age_Month"] else None
        record["study_age"] = months is not None and 48 <= months <= 95
        record["prompt_child"] = (row["File_ID"], row["Group"]) in prompt_children
        record["source_text_sha256"] = hashlib.sha256(row["story_text"].encode()).hexdigest()
        texts = representations(row)
        record["prepared_text_sha256"] = (hashlib.sha256(texts["prepared_text"].encode()).hexdigest()
                                           if texts["prepared_text"] is not None else "")
        for label, tokenizer in tokenizers.items():
            for representation, text in texts.items():
                ids = tokenizer.encode(text, add_special_tokens=False).ids if text is not None else None
                record[label + "__" + representation] = len(ids) if ids is not None else None
                if independent is not None and label == "gemma2_9b" and text is not None:
                    if ids != independent.encode(text, out_type=int):
                        raise ValueError(f"Gemma tokenizer implementations disagree: {record}")
                    agreements += 1
        records.append(record)
    summaries = []
    for label in tokenizers:
        for representation in REPRESENTATIONS:
            column = label + "__" + representation
            for scope, subset in (("all_saved", records),
                                  ("ages_4_to_7", [row for row in records if row["study_age"]]),
                                  ("configured_prompt_pool", [row for row in records if row["prompt_child"]])):
                summaries.append({"tokenizer": label, "representation": representation,
                                  "scope": scope, **summarize(subset, column, ceiling)})
    result = {
        "source_csv": str(source_csv), "source_sha256": source_hash,
        "roster_sha256": hashlib.sha256(roster_csv.read_bytes()).hexdigest(),
        "tokenizer_sha256": {label: hashlib.sha256(path.read_bytes()).hexdigest()
                             for label, path in tokenizer_files.items()},
        "output_ceiling_checked": ceiling, "special_tokens_added": False,
        "padding_or_truncation": False, "ceilings_changed": False,
        "model_weights_loaded": False, "generation_performed": False,
        "gemma_independent_full_token_id_agreements": agreements,
        "summary": summaries,
    }
    if hashlib.sha256(source_csv.read_bytes()).hexdigest() != source_hash:
        raise ValueError("Source changed during audit.")
    output_dir.mkdir(parents=True)
    with (output_dir / "per_story.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--roster-csv", type=Path, required=True)
    parser.add_argument("--tokenizer", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--gemma-sentencepiece", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ceiling", type=int, default=640)
    args = parser.parse_args()
    paths = {label: Path(path) for label, path in (item.split("=", 1) for item in args.tokenizer)}
    if len(paths) != len(args.tokenizer):
        parser.error("Tokenizer labels must be unique.")
    audit(args.source_csv, args.roster_csv, paths, args.output_dir,
          ceiling=args.ceiling, gemma_sentencepiece=args.gemma_sentencepiece)
    print(f"Audit written to {args.output_dir}; no generation settings changed.")


if __name__ == "__main__":
    main()
