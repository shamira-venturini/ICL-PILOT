from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

from .paths import FROZEN_ROSTER_CSV


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_MULTISPACE_RE = re.compile(r"\s+")
_TRAILING_CLAUSE_PUNCT_RE = re.compile(r"[,:;.!?]+$")
_HAS_LEXICAL_CONTENT_RE = re.compile(r"[A-Za-z0-9]")
_LEADING_BOUNDARY_PUNCT_RE = re.compile(r"^[,.;:!?]+\s*")
_CHAT_PUNCT_RE = re.compile(r"\s*([,.;:!?])")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_AGE_RE = re.compile(r"(\d+);(\d{1,2})(?:\.(\d{1,2}))?\Z")


def _normalize_surface(text: str) -> str:
    text = str(text or "").replace("\n", " ").replace("\r", " ")
    text = text.replace("“", '"').replace("”", '"')
    text = _MULTISPACE_RE.sub(" ", text).strip()
    return text


def _has_lexical_content(text: str) -> bool:
    return bool(_HAS_LEXICAL_CONTENT_RE.search(_normalize_surface(text)))


def _strip_leading_boundary_punct(text: str) -> str:
    normalized = _normalize_surface(text)
    return _LEADING_BOUNDARY_PUNCT_RE.sub("", normalized).strip()


def _chat_space_punctuation(text: str) -> str:
    normalized = _normalize_surface(text)
    if not normalized:
        return ""
    spaced = _CHAT_PUNCT_RE.sub(r" \1", normalized)
    return _MULTISPACE_RE.sub(" ", spaced).strip()


def _sentence_lines(text: str) -> list[str]:
    normalized = _normalize_surface(text)
    if not normalized or not _has_lexical_content(normalized):
        return []
    parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(normalized) if part.strip()]
    lines: list[str] = []
    for part in parts:
        if part[-1] not in ".!?":
            part = f"{part}."
        spaced = _chat_space_punctuation(part)
        if spaced:
            lines.append(spaced)
    fallback = normalized if normalized[-1] in ".!?" else f"{normalized}."
    fallback = _chat_space_punctuation(fallback)
    return lines or ([fallback] if fallback else [])


def _sentence_lines_with_tail(text: str) -> tuple[list[str], str]:
    normalized = _normalize_surface(text)
    if not normalized:
        return [], ""
    parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(normalized) if part.strip()]
    complete: list[str] = []
    tail = ""
    for part in parts:
        if part[-1] in ".!?":
            complete.append(part)
        else:
            tail = part
    return complete, tail


def _split_first_segment(text: str) -> tuple[str, str]:
    normalized = _strip_leading_boundary_punct(text)
    if not normalized:
        return "", ""
    parts = _SENTENCE_SPLIT_RE.split(normalized, maxsplit=1)
    first = parts[0].strip()
    rest = parts[1].strip() if len(parts) > 1 else ""
    return first, rest


def _strip_clause_edges(text: str) -> str:
    text = _normalize_surface(text)
    text = _TRAILING_CLAUSE_PUNCT_RE.sub("", text).strip()
    return text


def _format_quote_prefix_line(text: str) -> str:
    clause = _strip_clause_edges(text)
    if not _has_lexical_content(clause):
        return ""
    return f'{clause} +"/.' if clause else '+"/.'


def _format_quote_line(text: str) -> str:
    clause = _normalize_surface(text).replace('"', "").strip()
    if not _has_lexical_content(clause):
        return ""
    if clause and clause[-1] not in ".!?":
        clause = f"{clause}."
    clause = _chat_space_punctuation(clause)
    return f'+\" {clause}'.strip()


def _format_quote_suffix_line(text: str) -> str:
    clause = _strip_clause_edges(text).replace('"', "")
    if not _has_lexical_content(clause):
        return ""
    return f'{clause} +".' if clause else '+".'


def _chat_utterance_lines(text: str) -> list[str]:
    normalized = _normalize_surface(text)
    if not normalized:
        return []
    if '"' not in normalized:
        return _sentence_lines(normalized)

    lines: list[str] = []
    remaining = normalized
    while '"' in remaining:
        before, _, quoted_and_after = remaining.partition('"')
        quoted, closing_quote, after = quoted_and_after.partition('"')
        if not closing_quote:
            stripped = (before + " " + quoted).replace('"', "").strip()
            lines.extend(_sentence_lines(stripped))
            remaining = after
            break

        completed_before, before_tail = _sentence_lines_with_tail(before)
        lines.extend(completed_before)
        if before_tail:
            lines.append(_format_quote_prefix_line(before_tail))

        quote_lines, quote_tail = _sentence_lines_with_tail(quoted)
        for quote_line in quote_lines:
            lines.append(_format_quote_line(quote_line))
        if quote_tail:
            lines.append(_format_quote_line(quote_tail))

        if before_tail:
            remaining = _strip_leading_boundary_punct(after)
            continue

        suffix, rest = _split_first_segment(after)
        if suffix:
            lines.append(_format_quote_suffix_line(suffix))
        remaining = _strip_leading_boundary_punct(rest)

    if remaining:
        lines.extend(_sentence_lines(remaining.replace('"', "")))

    return [line for line in lines if line]


def _build_chat_text(
    *,
    corpus_name: str,
    age: str,
    group: str,
    story_id: str,
    story_text: str,
    comments: list[str],
) -> str:
    utterances = _chat_utterance_lines(story_text)
    if not utterances:
        raise ValueError("Cannot write CHAT file for empty story text")

    lines = [
        "@UTF8",
        "@Begin",
        "@Languages:\teng",
        "@Participants:\tEXA Investigator, CHI Target_Child",
        f"@ID:\teng|{corpus_name}|EXA|||||Investigator|||",
        f"@ID:\teng|{corpus_name}|CHI|{age}||{group}||Target_Child|||",
        f"@Types:\tcross, narrative, generated, {group}",
        f"@G:\t{story_id}",
    ]
    lines.extend(f"@Comment:\t{comment}" for comment in comments)
    lines.extend(f"*CHI:\t{utterance}" for utterance in utterances)
    lines.append("@End")
    return "\n".join(lines) + "\n"


def _read_csv_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    # Keep literal strings (including "NA") and empty cells distinct. In
    # particular, a missing narrative must never become the story "nan".
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or any(not name for name in fields):
            raise ValueError(f"Duplicate or empty CSV column names: {path}")
        missing = required.difference(fields)
        if missing:
            raise ValueError(f"Input CSV is missing required columns: {sorted(missing)} ({path})")
        rows = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CSV record ending at line {reader.line_num}: {path}")
            rows.append(row)
    if not rows:
        raise ValueError(f"Input CSV contains no records: {path}")
    return rows


def _safe_id(value: str, field: str) -> str:
    value = value.strip()
    if not _SAFE_ID_RE.fullmatch(value):
        raise ValueError(f"Missing or unsafe {field}: {value!r}")
    return value


def _validate_age(value: str, field: str) -> None:
    match = _AGE_RE.fullmatch(value)
    if match is None or int(match[2]) > 11 or (match[3] and int(match[3]) > 31):
        raise ValueError(f"Missing or invalid CHAT {field}: {value!r}")


def _load_roster_metadata(roster_csv: str | Path | None) -> dict[str, dict[str, str]]:
    if roster_csv is None:
        return {}
    roster_path = Path(roster_csv).expanduser().resolve()
    roster = _read_csv_rows(roster_path, {"pair_id", "td_age", "sli_age"})
    metadata: dict[str, dict[str, str]] = {}
    for row in roster:
        pair_id = _safe_id(row["pair_id"], "roster pair_id")
        if pair_id in metadata:
            raise ValueError(f"Duplicate roster pair_id: {pair_id}")
        metadata[pair_id] = {key: value.strip() for key, value in row.items()}
    return metadata


def _resolve_metadata(
    row: dict[str, str], roster: dict[str, str], field: str,
) -> tuple[str, str]:
    recorded = row.get(field, "").strip()
    frozen = roster.get(field, "").strip()
    if recorded and frozen and recorded != frozen:
        raise ValueError(f"CSV {field} disagrees with frozen roster: {recorded!r} != {frozen!r}")
    return (recorded, "generation_csv") if recorded else (frozen, "roster" if frozen else "")


def convert_generated_story_csv_to_cha(
    *,
    input_csv: str | Path,
    output_dir: str | Path,
    roster_csv: str | Path | None = FROZEN_ROSTER_CSV,
    corpus_name: str = "ICL-PILOT",
) -> dict[str, Path]:
    """Validate a complete two-stage CSV, then export to a fresh directory.

    Existing transcripts are never overwritten. All rows are checked before
    writing; an I/O failure can leave a partial new directory for inspection.
    Pass ``roster_csv=None`` only for a CSV containing its own ages and child IDs.
    """
    input_path = Path(input_csv).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    if out_dir.exists():
        raise FileExistsError(f"CHAT output directory already exists; choose a fresh directory: {out_dir}")

    required = {"bundle_id", "pair_id", "target_story", "stage1_td_story", "stage2_sli_story"}
    rows = _read_csv_rows(input_path, required)
    roster_metadata = _load_roster_metadata(roster_csv)
    source_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    if not corpus_name.strip() or any(char in corpus_name for char in "|\r\n\t"):
        raise ValueError("corpus_name must be nonempty and contain no CHAT field separators")

    stage_dirs = {
        "stage1_td": out_dir / "stage1_td",
        "stage2_sli": out_dir / "stage2_sli",
    }
    manifest_rows: list[dict[str, str]] = []
    planned_files: dict[Path, str] = {}
    seen_prompt_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        pair_id = _safe_id(row["pair_id"], "pair_id")
        story_id = row["target_story"].strip()
        if story_id not in {"A1", "A2", "A3", "B1", "B2", "B3"}:
            raise ValueError(f"Invalid ENNI target_story at CSV record {row_number}: {story_id!r}")
        bundle_id = _safe_id(row["bundle_id"], "bundle_id")
        round_id = row.get("round_id", "").strip()
        if round_id:
            _safe_id(round_id, "round_id")
        replicate_id = row.get("replicate_id", "").strip()
        if replicate_id and (not replicate_id.isdecimal() or int(replicate_id) < 1):
            raise ValueError(f"Invalid replicate_id: {replicate_id!r}")
        if roster_csv is not None and pair_id not in roster_metadata:
            raise ValueError(f"Unknown pair_id in frozen roster: {pair_id}")
        roster_row = roster_metadata.get(pair_id, {})
        cohort, _ = _resolve_metadata(row, roster_row, "cohort")
        if cohort:
            _safe_id(cohort, "cohort")
        prompt_ids = [row.get(key, "").strip() for key in
                      ("stage1_prompt_record_id", "stage2_prompt_record_id")]
        if any(key in row for key in ("stage1_prompt_record_id", "stage2_prompt_record_id")):
            for prompt_id in prompt_ids:
                _safe_id(prompt_id, "prompt_record_id")
            if prompt_ids[0] == prompt_ids[1]:
                raise ValueError("Stage 1 and Stage 2 must have distinct prompt record IDs")
            if seen_prompt_ids.intersection(prompt_ids):
                raise ValueError("A prompt record ID is reused across CSV records")
            seen_prompt_ids.update(prompt_ids)

        base_stem = f"{bundle_id}_{story_id}"
        stage_specs = [
            {
                "stage": "stage1_td",
                "group": "TD",
                "prefix": "td",
                "text": row["stage1_td_story"],
                "prompt_record_id": prompt_ids[0],
                "parent_prompt_record_id": "",
            },
            {
                "stage": "stage2_sli",
                "group": "SLI",
                "prefix": "sli",
                "text": row["stage2_sli_story"],
                "prompt_record_id": prompt_ids[1],
                "parent_prompt_record_id": prompt_ids[0],
            },
        ]

        for spec in stage_specs:
            stage = spec["stage"]
            if row.get(f"{stage.split('_')[0]}_group", spec["group"]).strip() != spec["group"]:
                raise ValueError(f"Unexpected clinical group for {stage} in CSV record {row_number}")
            age, age_source = _resolve_metadata(row, roster_row, f"{spec['prefix']}_age")
            _validate_age(age, f"{spec['prefix']}_age")
            child_id, child_source = _resolve_metadata(row, roster_row, f"{spec['prefix']}_child_id")
            if not child_id.isdecimal() or int(child_id) < 1:
                raise ValueError(f"Missing or invalid {spec['prefix']}_child_id: {child_id!r}")
            cohort_match = re.fullmatch(r"(\d+)-year-old", cohort)
            if cohort_match and int(age.split(";")[0]) != int(cohort_match[1]):
                raise ValueError(f"Age {age!r} disagrees with cohort {cohort!r}")
            story_text = _normalize_surface(spec["text"])
            if not story_text or not _has_lexical_content(story_text):
                raise ValueError(f"Empty or nonlexical {stage} story in CSV record {row_number}")

            output_path = stage_dirs[stage] / f"{base_stem}_{spec['group']}.cha"
            if output_path in planned_files:
                raise ValueError(f"Duplicate CHAT destination in CSV record {row_number}: {output_path.name}")
            comments = [
                f"source_csv={input_path}",
                f"bundle_id={bundle_id}",
                f"pair_id={pair_id}",
                f"target_story={story_id}",
                f"stage={spec['stage']}",
                f"group={spec['group']}",
            ]
            if round_id:
                comments.append(f"round_id={round_id}")
            if replicate_id:
                comments.append(f"replicate_id={replicate_id}")
            comments.append(f"source_child_id={child_id}")
            if cohort:
                comments.append(f"cohort={cohort}")
            for field in ("prompt_record_id", "parent_prompt_record_id"):
                if spec[field]:
                    comments.append(f"{field}={spec[field]}")

            chat_text = _build_chat_text(
                corpus_name=corpus_name,
                age=age,
                group=spec["group"],
                story_id=story_id,
                story_text=story_text,
                comments=comments,
            )
            planned_files[output_path] = chat_text
            manifest_rows.append(
                {
                    "source_csv": str(input_path),
                    "round_id": round_id,
                    "bundle_id": bundle_id,
                    "pair_id": pair_id,
                    "replicate_id": replicate_id,
                    "target_story": story_id,
                    "stage": spec["stage"],
                    "group": spec["group"],
                    "source_child_id": child_id,
                    "age": age,
                    "output_path": str(output_path),
                    "cohort": cohort,
                    "age_source": age_source,
                    "source_child_id_source": child_source,
                    "prompt_record_id": spec["prompt_record_id"],
                    "parent_prompt_record_id": spec["parent_prompt_record_id"],
                    "source_csv_sha256": source_hash,
                    "chat_sha256": hashlib.sha256(chat_text.encode("utf-8")).hexdigest(),
                }
            )

    # No destination is created until the entire input has passed preflight.
    out_dir.mkdir(parents=True, exist_ok=False)
    for stage_dir in stage_dirs.values():
        stage_dir.mkdir()
    for output_path, chat_text in planned_files.items():
        with output_path.open("x", encoding="utf-8", newline="") as handle:
            handle.write(chat_text)
    manifest_path = out_dir / "chat_manifest.csv"
    with manifest_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    return {
        "output_dir": out_dir,
        "manifest_csv": manifest_path,
        "stage1_td_dir": stage_dirs["stage1_td"],
        "stage2_sli_dir": stage_dirs["stage2_sli"],
    }
