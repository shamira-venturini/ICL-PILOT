from __future__ import annotations

import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


_NS = {"ss": "urn:schemas-microsoft-com:office:spreadsheet"}
_AGE_RE = re.compile(r"^(\d+);(\d+)\.(\d+)$")


def _read_spreadsheetml_rows(path: Path) -> list[list[str]]:
    root = ET.parse(path).getroot()
    worksheet = root.find("ss:Worksheet", _NS)
    if worksheet is None:
        raise ValueError(f"No worksheet found in {path}")

    table = worksheet.find("ss:Table", _NS)
    if table is None:
        raise ValueError(f"No worksheet table found in {path}")

    rows: list[list[str]] = []
    max_len = 0
    for row in table.findall("ss:Row", _NS):
        current: list[str] = []
        column_index = 1
        for cell in row.findall("ss:Cell", _NS):
            explicit_index = cell.attrib.get("{urn:schemas-microsoft-com:office:spreadsheet}Index")
            if explicit_index is not None:
                target_index = int(explicit_index)
                while column_index < target_index:
                    current.append("")
                    column_index += 1
            data = cell.find("ss:Data", _NS)
            current.append("" if data is None else "".join(data.itertext()))
            column_index += 1
        max_len = max(max_len, len(current))
        rows.append(current)

    return [row + [""] * (max_len - len(row)) for row in rows]


def _dedupe_headers(headers: list[str]) -> tuple[list[str], list[str], list[str]]:
    seen: dict[str, int] = {}
    renamed: list[str] = []
    duplicates: list[str] = []
    out: list[str] = []

    for header in headers:
        name = header or "Unnamed"
        count = seen.get(name, 0) + 1
        seen[name] = count
        if count == 1:
            out.append(name)
            continue
        if name not in duplicates:
            duplicates.append(name)
        new_name = f"{name}_{count}"
        out.append(new_name)
        renamed.append(new_name)

    return out, duplicates, renamed


def _load_raw_frame(path: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    rows = _read_spreadsheetml_rows(path)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    headers, duplicate_headers, renamed_columns = _dedupe_headers(rows[0])
    frame = pd.DataFrame(rows[1:], columns=headers, dtype=str).fillna("")
    return frame, duplicate_headers, renamed_columns


def _normalize_numeric_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    numeric_columns: list[str] = []
    for column in out.columns:
        series = out[column].astype(str).str.strip()
        nonblank = series[series != ""]
        if nonblank.empty:
            continue
        converted = pd.to_numeric(nonblank, errors="coerce")
        if converted.notna().all():
            numeric_columns.append(column)
            out[column] = pd.to_numeric(series.replace("", pd.NA), errors="coerce")
    return out, numeric_columns


def _age_to_months(age_text: str) -> str:
    match = _AGE_RE.match(age_text.strip())
    if not match:
        return ""
    years = int(match.group(1))
    months = int(match.group(2))
    days = int(match.group(3))
    approx = years * 12 + months + (days / 30.0)
    return str(approx)


def _extract_file_id(file_name: str) -> str:
    stem = Path(file_name).name
    for suffix in (".cha", ".tbl.cex", ".dss.cex"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def process_generated_kideval(kideval_xls: str | Path, output_root: str | Path) -> int:
    source_path = Path(kideval_xls).expanduser().resolve()
    out_root = Path(output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    copied_xls = out_root / "kideval.xls"
    shutil.copy2(source_path, copied_xls)

    raw_frame, duplicate_headers, renamed_columns = _load_raw_frame(source_path)
    raw_csv = out_root / "kideval_raw.csv"
    raw_frame.to_csv(raw_csv, index=False)

    warning_mask = raw_frame.iloc[:, 0].astype(str).str.contains(
        "IF YOU DID NOT MARK ERRORS|Please be sure to mark utterances",
        regex=True,
        na=False,
    )
    blank_mask = raw_frame.apply(lambda row: all(str(value).strip() == "" for value in row), axis=1)
    clean_frame = raw_frame.loc[~warning_mask & ~blank_mask].copy()
    clean_frame, numeric_columns = _normalize_numeric_columns(clean_frame)

    clean_csv = out_root / "kideval_clean.csv"
    clean_xlsx = out_root / "kideval_clean.xlsx"
    clean_frame.to_csv(clean_csv, index=False)
    clean_frame.to_excel(clean_xlsx, index=False)

    qc_summary = {
        "source_file": str(source_path),
        "raw_shape": [int(raw_frame.shape[0]), int(raw_frame.shape[1])],
        "clean_shape": [int(clean_frame.shape[0]), int(clean_frame.shape[1])],
        "removed_warning_rows": int(warning_mask.sum()),
        "removed_blank_rows": int(blank_mask.sum()),
        "duplicate_headers_resolved": duplicate_headers,
        "renamed_columns": renamed_columns,
        "numeric_columns_count": len(numeric_columns),
        "numeric_columns_sample": numeric_columns[:15],
        "parquet_written": False,
    }
    (out_root / "kideval_qc_summary.json").write_text(
        json.dumps(qc_summary, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return 0


def process_generated_mlt(
    mlt_xls: str | Path,
    kideval_clean_csv: str | Path,
    output_root: str | Path,
) -> int:
    source_path = Path(mlt_xls).expanduser().resolve()
    kideval_path = Path(kideval_clean_csv).expanduser().resolve()
    out_root = Path(output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    copied_xls = out_root / "mlt.xls"
    shutil.copy2(source_path, copied_xls)

    raw_frame, _, _ = _load_raw_frame(source_path)
    raw_frame["File_ID"] = raw_frame["File"].map(_extract_file_id)
    raw_frame["Age_Month_approx"] = raw_frame["Age"].map(_age_to_months)

    raw_csv = out_root / "mlt_raw.csv"
    raw_frame.to_csv(raw_csv, index=False)

    clean_frame, _numeric_columns = _normalize_numeric_columns(raw_frame)
    clean_csv = out_root / "mlt_clean.csv"
    clean_xlsx = out_root / "mlt_clean.xlsx"
    clean_frame.to_csv(clean_csv, index=False)
    clean_frame.to_excel(clean_xlsx, index=False)

    kideval_frame = pd.read_csv(kideval_path, dtype=str).fillna("")
    mlt_ids = clean_frame["File_ID"].astype(str)
    kideval_ids = kideval_frame["File"].astype(str).map(_extract_file_id)

    qc_summary = {
        "mlt_rows": int(len(clean_frame)),
        "kideval_rows": int(len(kideval_frame)),
        "mlt_duplicate_file_id": int(mlt_ids.duplicated().sum()),
        "kideval_duplicate_file_id": int(kideval_ids.duplicated().sum()),
        "merged_rows": int(len(kideval_frame)),
        "merged_missing_mlt": int((~kideval_ids.isin(set(mlt_ids))).sum()),
    }
    (out_root / "mlt_qc_summary.json").write_text(
        json.dumps(qc_summary, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return 0
