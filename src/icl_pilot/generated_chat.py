from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_MULTISPACE_RE = re.compile(r"\s+")
_TRAILING_CLAUSE_PUNCT_RE = re.compile(r"[,:;.!?]+$")
_HAS_LEXICAL_CONTENT_RE = re.compile(r"[A-Za-z0-9]")
_LEADING_BOUNDARY_PUNCT_RE = re.compile(r"^[,.;:!?]+\s*")
_CHAT_PUNCT_RE = re.compile(r"\s*([,.;:!?])")


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


def _load_roster_metadata(roster_csv: str | Path | None) -> dict[str, dict[str, str]]:
    if roster_csv is None:
        return {}
    roster_path = Path(roster_csv).expanduser().resolve()
    if not roster_path.exists():
        return {}

    roster = pd.read_csv(roster_path)
    metadata: dict[str, dict[str, str]] = {}
    for _, row in roster.iterrows():
        metadata[str(row["pair_id"])] = {
            "sli_age": str(row.get("sli_age", "")),
            "td_age": str(row.get("td_age", "")),
            "sli_file_kideval": str(row.get("sli_file_kideval", "")),
            "td_file_kideval": str(row.get("td_file_kideval", "")),
        }
    return metadata


def convert_generated_story_csv_to_cha(
    *,
    input_csv: str | Path,
    output_dir: str | Path,
    roster_csv: str | Path | None = "configs/generation/four_year_old_frozen_roster.csv",
    corpus_name: str = "ICL-PILOT",
) -> dict[str, Path]:
    input_path = Path(input_csv).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(input_path)
    roster_metadata = _load_roster_metadata(roster_csv)

    required = {"bundle_id", "pair_id", "target_story", "stage1_td_story", "stage2_sli_story"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    stage_dirs = {
        "stage1_td": out_dir / "stage1_td",
        "stage2_sli": out_dir / "stage2_sli",
    }
    for stage_dir in stage_dirs.values():
        stage_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    for _, row in frame.iterrows():
        pair_id = str(row["pair_id"])
        story_id = str(row["target_story"])
        bundle_id = str(row["bundle_id"])
        round_id = str(row.get("round_id", ""))
        replicate_id = str(row.get("replicate_id", ""))
        roster_row = roster_metadata.get(pair_id, {})

        base_stem = f"{bundle_id}_{story_id}"
        stage_specs = [
            {
                "stage": "stage1_td",
                "group": "TD",
                "age": str(row.get("td_age", "")) or roster_row.get("td_age", ""),
                "text": str(row.get("stage1_td_story", "")),
                "child_id": str(row.get("td_child_id", "")),
            },
            {
                "stage": "stage2_sli",
                "group": "SLI",
                "age": str(row.get("sli_age", "")) or roster_row.get("sli_age", ""),
                "text": str(row.get("stage2_sli_story", "")),
                "child_id": str(row.get("sli_child_id", "")),
            },
        ]

        for spec in stage_specs:
            story_text = _normalize_surface(spec["text"])
            if not story_text:
                continue

            output_path = stage_dirs[spec["stage"]] / f"{base_stem}_{spec['group']}.cha"
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
            if spec["child_id"]:
                comments.append(f"source_child_id={spec['child_id']}")

            chat_text = _build_chat_text(
                corpus_name=corpus_name,
                age=spec["age"],
                group=spec["group"],
                story_id=story_id,
                story_text=story_text,
                comments=comments,
            )
            output_path.write_text(chat_text, encoding="utf-8")
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
                    "source_child_id": spec["child_id"],
                    "age": spec["age"],
                    "output_path": str(output_path),
                }
            )

    manifest_path = out_dir / "chat_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()) if manifest_rows else [
            "source_csv",
            "round_id",
            "bundle_id",
            "pair_id",
            "replicate_id",
            "target_story",
            "stage",
            "group",
            "source_child_id",
            "age",
            "output_path",
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)

    return {
        "output_dir": out_dir,
        "manifest_csv": manifest_path,
        "stage1_td_dir": stage_dirs["stage1_td"],
        "stage2_sli_dir": stage_dirs["stage2_sli"],
    }
