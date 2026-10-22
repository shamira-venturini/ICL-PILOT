from __future__ import annotations

import csv
from collections import Counter
import hashlib
import importlib.util
import json
import re
import warnings
from pathlib import Path


_FILE_ID_RE = re.compile(r"(\d+)\.cha$")
_AGE_RE = re.compile(r"^@ID:[ \t]*[^|\n]*\|[^|\n]*\|CHI\|([^|\n]*)\|([^|\n]*)\|([^|\n]*)\|", re.MULTILINE)
_CHAT_AGE_RE = re.compile(r"(\d+);(\d{1,2})(?:\.(\d{0,2}))?")
_STORY_RE = re.compile(r"^@G:[ \t]*(\S+)[ \t]*$")
_MAIN_TIER_RE = re.compile(r"^\*([^:\s]+):[ \t]*(.*)$")
_ANNOTATION_TOKEN_RE = re.compile(r"(?:^|\s)\S*\|\S+")
_TIMING_RE = re.compile(r"\x15[^\x15]*\x15")
_ANGLE_RE = re.compile(r"<([^<>]*)>")
_BRACKET_RE = re.compile(r"\[[^\]]*\]")
CHAT_STORY_PARSER_VERSION = "chat-story-v2"


def _extract_file_id(path: Path) -> str:
    match = _FILE_ID_RE.search(path.name)
    return match.group(1) if match else ""


def _load_semrp_callable(semrp_repo: Path):
    module_path = semrp_repo / "semantic_r_precision.py"
    if not module_path.exists():
        raise FileNotFoundError(f"SemR-p module not found: {module_path}")

    spec = importlib.util.spec_from_file_location("semantic_r_precision", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "calculate_sem_r_p"):
        raise AttributeError(f"{module_path} does not expose calculate_sem_r_p")
    return module.calculate_sem_r_p


def _parse_age_group(text: str) -> tuple[str, str, str]:
    """Read literal CHI metadata; months are completed years * 12 + months.

    CHAT ages use years;months.days. Days stay in the raw age field; unknown
    months and malformed ages remain undefined rather than becoming zero.
    """
    matches = list(_AGE_RE.finditer(text.lstrip("\ufeff")))
    if not matches:
        return "", "", ""
    values = {tuple(value.strip() for value in match.groups()) for match in matches}
    if len(values) != 1:
        raise ValueError("Conflicting CHI @ID headers; age/group metadata is ambiguous.")
    age_raw, _sex, group = values.pop()
    age_months = ""
    age = _CHAT_AGE_RE.fullmatch(age_raw)
    if age:
        years, months, days = age.groups()
        if int(months) < 12 and (not days or int(days) <= 31):
            age_months = str(int(years) * 12 + int(months))
    return age_raw, age_months, group


def _clean_chat_surface(text: str) -> str:
    text = _TIMING_RE.sub(" ", text)
    text = _ANGLE_RE.sub(r"\1", text)
    text = _BRACKET_RE.sub(" ", text)
    text = text.replace("\u0015", " ")
    text = text.replace("\t", " ")
    text = re.sub(r"\+\.\.\.", " ", text)
    text = re.sub(r"\+/{1,2}\.?", " ", text)
    text = re.sub(r"\+\s*,", " ", text)
    text = re.sub(r"&[+-][A-Za-z:']+", " ", text)
    text = re.sub(r"\b0[A-Za-z']+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _iter_story_utterances(
    text: str, diagnostics: list[dict[str, object]] | None = None,
) -> dict[str, list[str]]:
    """Read complete CHI main tiers, keeping dependent-tier text separate.

    Some existing processed ENNI files lost dependent-tier labels but retained
    wrapped morphology/dependency lines. Recognizable orphan annotation blocks
    are excluded and reported. This does not repair or rewrite source CHAT.
    The existing surface normalization is applied after joining a complete tier.
    """
    stories: dict[str, list[str]] = {}
    current_story = ""
    speaker = ""
    parts: list[str] = []
    orphan_block = False
    issues: list[dict[str, object]] = []

    def flush() -> None:
        nonlocal parts
        if current_story and speaker == "CHI" and parts:
            cleaned = _clean_chat_surface(" ".join(parts))
            if cleaned:
                stories[current_story].append(cleaned)
        parts = []

    def report(line_number: int, kind: str) -> None:
        issues.append({"line_number": line_number, "story_id": current_story,
                       "speaker": speaker, "kind": kind})

    for line_number, raw_line in enumerate(text.lstrip("\ufeff").splitlines(), start=1):
        story_match = _STORY_RE.match(raw_line)
        if story_match:
            flush()
            current_story = story_match.group(1)
            stories.setdefault(current_story, [])
            speaker, orphan_block = "", False
            continue
        main = _MAIN_TIER_RE.match(raw_line)
        if main:
            flush()
            speaker, content = main.groups()
            parts, orphan_block = [content], False
            continue
        if raw_line.startswith(("%", "@", "*")):
            flush()
            speaker, orphan_block = "", False
            continue
        if speaker and raw_line.startswith(("\t", " ")) and raw_line.strip():
            content = raw_line.strip()
            if orphan_block:
                report(line_number, "orphan_annotation_continuation")
                continue
            # Codes inside a wrapped bracket comment belong to the main tier.
            current = " ".join(parts)
            in_bracket = current.count("[") > current.count("]")
            if not in_bracket and _ANNOTATION_TOKEN_RE.search(_BRACKET_RE.sub(" ", content)):
                report(line_number, "orphan_annotation_continuation")
                flush()
                orphan_block = True
                continue
            visible = _TIMING_RE.sub("", content).strip()
            previous = _BRACKET_RE.sub("", _TIMING_RE.sub("", current)).strip()
            delimiter_only = re.fullmatch(r'(?:[.!?]|\+["/.,!?]*[.!?])', visible)
            if delimiter_only and re.search(r"[.!?]$", previous):
                report(line_number, "redundant_punctuation_continuation")
                flush()
                orphan_block = True
                continue
            parts.append(content)
        elif raw_line.strip():
            if speaker:
                report(line_number, "unmarked_continuation")
            flush()
            speaker, orphan_block = "", False
    flush()
    if diagnostics is not None:
        diagnostics.extend(issues)
    elif issues:
        warnings.warn(f"Excluded {len(issues)} malformed continuation lines; pass diagnostics to inspect them.",
                      UserWarning, stacklevel=2)
    return stories


def extract_story_unit_narratives(
    transcript_root: str,
    output_csv: str,
    summary_json: str | None = None,
) -> int:
    root = Path(transcript_root).expanduser().resolve()
    out_csv = Path(output_csv).expanduser().resolve()
    qc_path = Path(summary_json).expanduser().resolve() if summary_json else out_csv.with_suffix(".parsing_qc.json")
    if not root.is_dir():
        raise FileNotFoundError(f"CHAT transcript root does not exist: {root}")
    if out_csv == qc_path:
        raise ValueError("Narrative CSV and parsing QC must use different output paths.")
    if out_csv.exists() or qc_path.exists():
        raise FileExistsError("Narrative CSV or parsing QC already exists; choose fresh output paths.")
    paths = sorted(root.rglob("*.cha"))
    if not paths:
        raise ValueError(f"No CHAT files found under {root}")

    fieldnames = [
        "File_ID",
        "Group",
        "Age_Raw",
        "Age_Month",
        "transcript_path",
        "story_id",
        "story_unit_count",
        "story_units_json",
        "story_text",
    ]

    rows: list[dict[str, str]] = []
    sources: list[dict[str, object]] = []
    for path in paths:
        raw = path.read_bytes()
        text = raw.decode("utf-8-sig")
        age_raw, age_months, group = _parse_age_group(text)
        issues: list[dict[str, object]] = []
        story_map = _iter_story_utterances(text, diagnostics=issues)
        child_id = _extract_file_id(path)
        for missing, kind in ((not child_id, "missing_child_id"),
                              (not age_raw, "missing_age"),
                              (bool(age_raw) and not age_months, "unknown_or_invalid_age_months"),
                              (not group, "missing_group")):
            if missing:
                issues.append({"kind": kind})
        sources.append({"transcript_path": str(path.relative_to(root)),
                        "sha256": hashlib.sha256(raw).hexdigest(), "File_ID": child_id,
                        "Group": group, "Age_Raw": age_raw, "Age_Month": age_months,
                        "story_rows": len(story_map), "issues": issues})
        for story_id, units in story_map.items():
            rows.append(
                {
                    "File_ID": child_id,
                    "Group": group,
                    "Age_Raw": age_raw,
                    "Age_Month": age_months,
                    "transcript_path": str(path),
                    "story_id": story_id,
                    "story_unit_count": str(len(units)),
                    "story_units_json": json.dumps(units, ensure_ascii=True),
                    "story_text": " ".join(units),
                }
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    issue_counts = Counter(issue["kind"] for source in sources for issue in source["issues"])
    qc = {"parser_version": CHAT_STORY_PARSER_VERSION,
          "surface_normalization": "legacy_surface_v1; applied after joining main-tier continuations",
          "age_month_definition": "12 * years + months; days retained only in Age_Raw; unknown months remain blank",
          "transcript_root": str(root), "transcript_files": len(sources), "story_rows": len(rows),
          "issue_counts": dict(issue_counts), "sources": sources,
          "output_csv_sha256": hashlib.sha256(out_csv.read_bytes()).hexdigest()}
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    with qc_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(qc, indent=2) + "\n")

    print(f"Wrote story-unit narrative table to {out_csv}")
    print(f"Parsing QC: {qc_path}; issue counts: {dict(issue_counts)}")
    return 0


def build_story_unit_reference_template(
    transcript_root: str,
    output_json: str,
) -> int:
    root = Path(transcript_root).expanduser().resolve()
    out_json = Path(output_json).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    stories: set[str] = set()
    for path in sorted(root.rglob("*.cha")):
        text = path.read_text(encoding="utf-8", errors="replace")
        stories.update(_iter_story_utterances(text).keys())

    template = {
        story_id: {
            "story_id": story_id,
            "title": "",
            "source": "official ENNI picture sequence",
            "notes": "Fill this with concise ordered canonical story units derived from the official ENNI materials.",
            "units": [],
        }
        for story_id in sorted(stories)
    }

    out_json.write_text(json.dumps(template, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote story-unit reference template to {out_json}")
    return 0


def score_story_unit_semrp(
    narrative_csv: str,
    reference_json: str,
    semrp_repo: str,
    output_csv: str,
    k: int = 3,
    model_name_or_path: str = "uclanlp/keyphrase-mpnet-v1",
) -> int:
    input_csv = Path(narrative_csv).expanduser().resolve()
    ref_json = Path(reference_json).expanduser().resolve()
    repo = Path(semrp_repo).expanduser().resolve()
    out_csv = Path(output_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    calculate_sem_r_p = _load_semrp_callable(repo)
    references = json.loads(ref_json.read_text(encoding="utf-8"))

    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    fieldnames = list(rows[0].keys()) + ["story_unit_semrp"] if rows else ["story_unit_semrp"]
    scored_rows: list[dict[str, str]] = []
    for row in rows:
        story_id = row.get("story_id", "")
        ref_entry = references.get(story_id, {})
        ref_units = ref_entry.get("units", []) if isinstance(ref_entry, dict) else ref_entry
        pred_units = json.loads(row.get("story_units_json", "[]") or "[]")
        score = ""
        if ref_units and pred_units:
            score = str(
                calculate_sem_r_p(
                    predictions=pred_units,
                    references=ref_units,
                    k=k,
                    model_name_or_path=model_name_or_path,
                )
            )
        scored = dict(row)
        scored["story_unit_semrp"] = score
        scored_rows.append(scored)

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scored_rows)

    print(f"Wrote story-unit SemR-p scores to {out_csv}")
    return 0
