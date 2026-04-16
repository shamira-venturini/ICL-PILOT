from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path


_FILE_ID_RE = re.compile(r"(\d+)(?:\.cha|\.tbl\.cex)?$")
_FROM_FILE_RE = re.compile(r"^From file <(.+)>$")
_HEADER_RE = re.compile(r"^\s+Sentence\s+\|")
_TOTAL_RE = re.compile(r"^TOTAL\s+\|")
_SCORE_RE = re.compile(r"^Developmental sentence score:\s+(.+)$")


def _extract_file_id(value: str) -> str:
    match = _FILE_ID_RE.search(value)
    if match:
        return match.group(1)

    synthetic_id = value
    for suffix in (".tbl.cex", ".dss.cex", ".cha"):
        if synthetic_id.endswith(suffix):
            synthetic_id = synthetic_id[: -len(suffix)]
            break
    return synthetic_id


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_pipe_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.split("|") if cell.strip()]


def parse_dss_tbl_file(path: Path, repo_root: Path | None = None) -> dict[str, str]:
    repo_root = repo_root or path.resolve().parents[2]
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    transcript_path = ""
    command = lines[0].strip() if lines else ""
    for line in lines:
        file_match = _FROM_FILE_RE.match(line)
        if file_match:
            transcript_path = file_match.group(1)
            break

    categories: list[str] = []
    totals: list[str] = []
    score = ""
    for line in lines:
        if not categories and _HEADER_RE.match(line):
            categories = _parse_pipe_row(line)[1:]
        elif categories and _TOTAL_RE.match(line):
            totals = _parse_pipe_row(line)[1:]
        else:
            score_match = _SCORE_RE.match(line)
            if score_match:
                score = score_match.group(1).strip()

    transcript_rel = ""
    if transcript_path:
        try:
            transcript_rel = str(Path(transcript_path).resolve().relative_to(repo_root))
        except ValueError:
            transcript_rel = transcript_path

    row: dict[str, str] = {
        "File_ID": _extract_file_id(path.name),
        "DSS_tbl_source": str(path.resolve()),
        "DSS_tbl_relpath": str(path.resolve().relative_to(repo_root)),
        "DSS_tbl_transcript": transcript_path,
        "DSS_tbl_transcript_relpath": transcript_rel,
        "DSS_tbl_command": command,
        "DSS_tbl_score": score,
        "DSS_tbl_score_is_na": "1" if score == "NA" else "0",
        "DSS_tbl_warning_50": "1" if "DSS requires 50 complete sentences" in text else "0",
    }

    for category, total in zip(categories, totals):
        row[f"DSS_tbl_{category}"] = total

    return row


def build_dss_feature_table(
    dss_root: str,
    output_csv: str,
    output_qc_json: str,
) -> int:
    root = Path(dss_root).expanduser().resolve()
    out_csv = Path(output_csv).expanduser().resolve()
    out_qc = Path(output_qc_json).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[2]

    files = sorted(root.rglob("*.tbl.cex"))
    rows = [parse_dss_tbl_file(path, repo_root=repo_root) for path in files]

    category_columns = sorted(
        {
            key
            for row in rows
            for key in row
            if key.startswith("DSS_tbl_") and key.count("_") == 2 and key.rsplit("_", 1)[1].isupper()
        }
    )
    base_columns = [
        "File_ID",
        "DSS_tbl_source",
        "DSS_tbl_relpath",
        "DSS_tbl_transcript",
        "DSS_tbl_transcript_relpath",
        "DSS_tbl_command",
        "DSS_tbl_score",
        "DSS_tbl_score_is_na",
        "DSS_tbl_warning_50",
    ]
    fieldnames = base_columns + category_columns

    normalized_rows: list[dict[str, str]] = []
    for row in rows:
        normalized_rows.append({column: row.get(column, "") for column in fieldnames})

    duplicate_ids = {
        file_id: sorted(records)
        for file_id, records in defaultdict(list, {
            row["File_ID"]: []
            for row in normalized_rows
        }).items()
    }
    # Rebuild duplicates map without relying on mutation in a comprehension.
    duplicate_buckets: dict[str, list[str]] = defaultdict(list)
    for row in normalized_rows:
        duplicate_buckets[row["File_ID"]].append(row["DSS_tbl_transcript_relpath"])
    duplicate_ids = {
        file_id: sorted(paths)
        for file_id, paths in duplicate_buckets.items()
        if file_id and len(paths) > 1
    }

    _write_csv(out_csv, fieldnames, normalized_rows)

    qc_summary = {
        "dss_tbl_files": len(files),
        "dss_tbl_unique_ids": len({row["File_ID"] for row in normalized_rows if row["File_ID"]}),
        "dss_tbl_na_scores": sum(row["DSS_tbl_score_is_na"] == "1" for row in normalized_rows),
        "dss_tbl_warning_50_count": sum(row["DSS_tbl_warning_50"] == "1" for row in normalized_rows),
        "duplicate_file_ids": duplicate_ids,
        "category_columns": category_columns,
    }
    out_qc.parent.mkdir(parents=True, exist_ok=True)
    out_qc.write_text(json.dumps(qc_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote {len(normalized_rows)} DSS rows to {out_csv}")
    print(f"Wrote DSS QC summary to {out_qc}")
    return 0


def merge_dss_into_master(
    master_csv: str,
    dss_csv: str,
    output_csv: str | None = None,
) -> int:
    master_path = Path(master_csv).expanduser().resolve()
    dss_path = Path(dss_csv).expanduser().resolve()
    out_path = Path(output_csv).expanduser().resolve() if output_csv else master_path

    master_fields, master_rows = _read_csv(master_path)
    dss_fields, dss_rows = _read_csv(dss_path)

    dss_columns = [field for field in dss_fields if field != "File_ID"]
    by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in dss_rows:
        by_id[row.get("File_ID", "")].append(row)

    duplicate_ids = sorted(file_id for file_id, rows in by_id.items() if file_id and len(rows) > 1)

    def pick_dss_row(master_row: dict[str, str]) -> dict[str, str] | None:
        file_id = master_row.get("File_ID", "")
        if not file_id or file_id not in by_id:
            return None

        candidates = by_id[file_id]
        if len(candidates) == 1:
            return candidates[0]

        file_kideval = master_row.get("File_kideval", "")
        transcript_rel = f"data/annotated/ENNI_batchalign/{file_kideval}" if file_kideval else ""
        for candidate in candidates:
            if candidate.get("DSS_tbl_transcript_relpath", "").endswith(transcript_rel):
                return candidate

        if file_kideval:
            for candidate in candidates:
                if candidate.get("DSS_tbl_transcript_relpath", "").endswith(file_kideval):
                    return candidate

        return candidates[0]

    merged_fields = master_fields + [field for field in dss_columns if field not in master_fields]
    merged_rows: list[dict[str, str]] = []
    matched = 0
    missing = 0
    for row in master_rows:
        merged = dict(row)
        match = pick_dss_row(row)
        if match is None:
            missing += 1
            for field in dss_columns:
                merged.setdefault(field, "")
        else:
            matched += 1
            for field in dss_columns:
                merged[field] = match.get(field, "")
        merged_rows.append(merged)

    _write_csv(out_path, merged_fields, merged_rows)
    print(f"Merged DSS features into {out_path}")
    print(f"Matched rows: {matched}")
    print(f"Missing rows: {missing}")
    if duplicate_ids:
        print(f"Duplicate File_ID values in DSS table resolved by transcript path matching: {', '.join(duplicate_ids)}")
    return 0
