from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path


_FILE_ID_RE = re.compile(r"(\d+)\.cha$")
_ERROR_RE = re.compile(r"\[\*\s+([^\]]+)\]")


def _extract_file_id(path: Path) -> str:
    match = _FILE_ID_RE.search(path.name)
    return match.group(1) if match else ""


def _safe_code_name(code: str) -> str:
    token_map = {
        ":": "_c_",
        "+": "_plus_",
        "=": "_eq_",
        "'": "_apos_",
        "-": "_dash_",
    }
    pieces: list[str] = []
    for char in code.lower():
        if char.isalnum():
            pieces.append(char)
        elif char in token_map:
            pieces.append(token_map[char])
        else:
            pieces.append("_")
    safe = "".join(pieces)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "unknown"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _group_name(code: str) -> str:
    if code.startswith("m:") or code.startswith("m-"):
        return "morphology"
    if code.startswith("s:"):
        return "syntax"
    if code.startswith("p:"):
        return "phonology"
    return "other"


def build_error_code_feature_table(
    transcript_root: str,
    output_csv: str,
    output_inventory_json: str,
) -> int:
    root = Path(transcript_root).expanduser().resolve()
    out_csv = Path(output_csv).expanduser().resolve()
    out_json = Path(output_inventory_json).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[2]

    rows: list[dict[str, str]] = []
    corpus_counts: Counter[str] = Counter()
    corpus_group_counts: Counter[str] = Counter()

    for path in sorted(root.rglob("*.cha")):
        text = path.read_text(encoding="utf-8", errors="replace")
        file_counts: Counter[str] = Counter()
        file_group_counts: Counter[str] = Counter()
        chi_utterances = 0
        chi_utterances_with_error = 0

        for line in text.splitlines():
            if not line.startswith("*CHI:\t"):
                continue
            chi_utterances += 1
            codes = [code.strip() for code in _ERROR_RE.findall(line)]
            if not codes:
                continue

            chi_utterances_with_error += 1
            for code in codes:
                file_counts[code] += 1
                file_group_counts[_group_name(code)] += 1
                corpus_counts[code] += 1
                corpus_group_counts[_group_name(code)] += 1

        row: dict[str, str] = {
            "File_ID": _extract_file_id(path),
            "error_source": str(path),
            "error_relpath": str(path.relative_to(repo_root)),
            "chi_utterances": str(chi_utterances),
            "chi_utterances_with_error_code": str(chi_utterances_with_error),
            "error_total_annotations": str(sum(file_counts.values())),
            "error_morphology_total": str(file_group_counts["morphology"]),
            "error_syntax_total": str(file_group_counts["syntax"]),
            "error_phonology_total": str(file_group_counts["phonology"]),
            "error_other_total": str(file_group_counts["other"]),
        }
        for code, count in sorted(file_counts.items()):
            row[f"err_{_safe_code_name(code)}"] = str(count)
        rows.append(row)

    dynamic_columns = sorted(
        {
            column
            for row in rows
            for column in row
            if column.startswith("err_")
        }
    )
    fieldnames = [
        "File_ID",
        "error_source",
        "error_relpath",
        "chi_utterances",
        "chi_utterances_with_error_code",
        "error_total_annotations",
        "error_morphology_total",
        "error_syntax_total",
        "error_phonology_total",
        "error_other_total",
    ] + dynamic_columns

    normalized_rows = [{field: row.get(field, "0" if field.startswith("err_") else "") for field in fieldnames} for row in rows]
    _write_csv(out_csv, fieldnames, normalized_rows)

    inventory = {
        "transcript_root": str(root),
        "files": len(rows),
        "unique_error_codes": len(corpus_counts),
        "error_code_counts": dict(sorted(corpus_counts.items(), key=lambda item: (-item[1], item[0]))),
        "error_group_counts": dict(sorted(corpus_group_counts.items())),
        "feature_to_source_code": {
            f"err_{_safe_code_name(code)}": code
            for code in sorted(corpus_counts)
        },
        "error_columns": dynamic_columns,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote error-code feature table to {out_csv}")
    print(f"Wrote error-code inventory to {out_json}")
    return 0
