"""Read-only screening of saved generation text; never infer a confirmed stop reason."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics


STAGES = ("stage1_td", "stage2_sli")
_TERMINAL = re.compile(r"[.!?\u2026][\"'\u2019\u201d)\]}]*\s*$")
_WORDS = re.compile(r"[^\W\d_]+(?:['\u2019][^\W\d_]+)*", re.UNICODE)


def screen_text(text: str, *, token_counter=None, reference_limit: int = 180) -> dict:
    """Keep token counts explicitly labelled as a re-encoding proxy.

    Decoding, stripping whitespace and removing special tokens are irreversible;
    even a count equal to the reference limit is not a recovered stopping reason.
    Punctuation and repeated words are descriptive review aids, not clinical QC.
    """
    if type(reference_limit) is not int or reference_limit < 1:
        raise ValueError("reference_limit must be a positive integer")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Expected nonempty saved generation text")
    words = _WORDS.findall(text)
    proxy = token_counter(text) if token_counter is not None else None
    if proxy is not None and (type(proxy) is not int or proxy < 0):
        raise ValueError("token_counter must return a nonnegative integer")
    max_run = max((sum(1 for _ in group) for _, group in
                   itertools.groupby(word.casefold() for word in words)), default=0)
    return {
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "characters": len(text),
        "whitespace_words": len(text.split()),
        "retokenized_saved_text_tokens": proxy,
        "proxy_equals_reference_limit": proxy == reference_limit if proxy is not None else None,
        "proxy_within_two_below_limit": reference_limit - 2 <= proxy < reference_limit if proxy is not None else None,
        "proxy_above_reference_limit": proxy > reference_limit if proxy is not None else None,
        "has_terminal_punctuation": bool(_TERMINAL.search(text)),
        "last_word": words[-1].casefold() if words else "",
        "max_consecutive_identical_words": max_run,
        "tail_text": text[-220:],
        "termination_status": "unknown_from_saved_text",
    }


def audit_csv(path: str | Path, *, cohort_year: int, token_counter=None,
              reference_limit: int = 180) -> list[dict]:
    """Read both stages without changing text or reconstructing missing labels."""
    source = Path(path)
    required = {"bundle_id", "pair_id", "round_id", "replicate_id", "target_story",
                "stage1_td_story", "stage2_sli_story", "stage1_group", "stage2_group"}
    records, seen = [], set()
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing generation columns: {sorted(missing)}")
        for line, row in enumerate(reader, start=2):
            key = (row["bundle_id"], row["target_story"])
            if key in seen:
                raise ValueError(f"Duplicate generation job at CSV row {line}: {key}")
            seen.add(key)
            for stage, group_column, child_column in (
                ("stage1_td", "stage1_group", "td_child_id"),
                ("stage2_sli", "stage2_group", "sli_child_id"),
            ):
                record = {name: row[name] for name in
                          ("bundle_id", "pair_id", "round_id", "replicate_id", "target_story")}
                record.update(cohort_year=cohort_year, stage=stage,
                              recorded_cohort=row.get("cohort", ""),
                              conditioning_group=row[group_column],
                              source_child_id=row.get(child_column, ""),
                              source_csv=str(source), csv_row_number=line)
                record.update(screen_text(row[stage + "_story"], token_counter=token_counter,
                                          reference_limit=reference_limit))
                records.append(record)
    return records


def summarize(records: list[dict], by: tuple[str, ...]) -> list[dict]:
    groups = defaultdict(list)
    for row in records:
        groups[tuple(row[name] for name in by)].append(row)
    summaries = []
    for key, rows in sorted(groups.items()):
        counts = sorted(row["retokenized_saved_text_tokens"] for row in rows
                        if row["retokenized_saved_text_tokens"] is not None)
        summary = dict(zip(by, key))
        summary.update(
            narratives=len(rows), token_proxy_available=len(counts),
            median_token_proxy=statistics.median(counts) if counts else None,
            p95_token_proxy_nearest_rank=counts[math.ceil(.95 * len(counts)) - 1] if counts else None,
            max_token_proxy=max(counts) if counts else None,
            proxy_equals_reference_limit=sum(row["proxy_equals_reference_limit"] is True for row in rows),
            proxy_within_two_below_limit=sum(row["proxy_within_two_below_limit"] is True for row in rows),
            proxy_above_reference_limit=sum(row["proxy_above_reference_limit"] is True for row in rows),
            missing_terminal_punctuation=sum(not row["has_terminal_punctuation"] for row in rows),
        )
        summaries.append(summary)
    return summaries


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No audit rows to write")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, metavar="AGE=CSV")
    parser.add_argument("--tokenizer-json", type=Path)
    parser.add_argument("--reference-limit", type=int, default=180)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("Choose a fresh output directory; existing reports are not overwritten")
    counter = None
    if args.tokenizer_json:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
        counter = lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)
    records, sources, seen = [], [], set()
    for specification in args.source:
        age, separator, filename = specification.partition("=")
        if not separator or not age.isdigit() or not filename:
            parser.error("Each source must be AGE=CSV")
        path = Path(filename)
        part = audit_csv(path, cohort_year=int(age), token_counter=counter,
                         reference_limit=args.reference_limit)
        for row in part:
            key = (row["cohort_year"], row["bundle_id"], row["target_story"], row["stage"])
            if key in seen:
                parser.error("Overlapping source specifications would duplicate generation jobs")
            seen.add(key)
        sources.append({"cohort_year": int(age), "path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "stage_outputs": len(part)})
        records.extend(part)
    if not records:
        parser.error("No generation outputs found")
    args.output_dir.mkdir(parents=True)
    write_rows(args.output_dir / "per_output.csv", records)
    for name, columns in (("by_age_stage", ("cohort_year", "stage")),
                          ("by_story_stage", ("target_story", "stage")),
                          ("by_age_story_stage", ("cohort_year", "target_story", "stage"))):
        write_rows(args.output_dir / (name + ".csv"), summarize(records, columns))
    review = [row for row in records if row["proxy_equals_reference_limit"]
              or row["proxy_within_two_below_limit"] or row["proxy_above_reference_limit"]]
    if review:
        write_rows(args.output_dir / "boundary_review.csv", review)
    summary = {"sources": sources, "stage_outputs": len(records),
               "reference_limit": args.reference_limit,
               "recorded_stop_reasons_used": False,
               "scope": "Decoded-text screening only; re-tokenized counts are proxies, not recovered generation token IDs or finish reasons",
               "by_stage": summarize(records, ("stage",)), "boundary_review_rows": len(review)}
    if args.tokenizer_json:
        summary["tokenizer_json_sha256"] = hashlib.sha256(args.tokenizer_json.read_bytes()).hexdigest()
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
