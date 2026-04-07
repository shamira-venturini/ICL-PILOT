from __future__ import annotations

import csv
import importlib.util
import json
import re
from pathlib import Path


_FILE_ID_RE = re.compile(r"(\d+)\.cha$")
_AGE_RE = re.compile(r"@ID:\s+eng\|ENNI\|CHI\|([^|]+)\|([^|]*)\|([^|]+)\|")
_STORY_RE = re.compile(r"^@G:\s+(\S+)\s*$")
_TIMING_RE = re.compile(r"\x15[^\x15]*\x15")
_ANGLE_RE = re.compile(r"<([^<>]*)>")
_BRACKET_RE = re.compile(r"\[[^\]]*\]")


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
    match = _AGE_RE.search(text)
    if not match:
        return "", "", ""
    age_raw, sex, group = match.groups()
    try:
        years, months, _days = age_raw.split(";")
        age_months = str(int(years) * 12 + int(months))
    except Exception:
        age_months = ""
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


def _iter_story_utterances(text: str) -> dict[str, list[str]]:
    stories: dict[str, list[str]] = {}
    current_story = ""
    for raw_line in text.splitlines():
        story_match = _STORY_RE.match(raw_line)
        if story_match:
            current_story = story_match.group(1)
            stories.setdefault(current_story, [])
            continue
        if not current_story or not raw_line.startswith("*CHI:\t"):
            continue
        content = raw_line.split("\t", 1)[1]
        cleaned = _clean_chat_surface(content)
        if cleaned:
            stories[current_story].append(cleaned)
    return stories


def extract_story_unit_narratives(
    transcript_root: str,
    output_csv: str,
) -> int:
    root = Path(transcript_root).expanduser().resolve()
    out_csv = Path(output_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

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
    for path in sorted(root.rglob("*.cha")):
        text = path.read_text(encoding="utf-8", errors="replace")
        age_raw, age_months, group = _parse_age_group(text)
        story_map = _iter_story_utterances(text)
        for story_id, units in story_map.items():
            rows.append(
                {
                    "File_ID": _extract_file_id(path),
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

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote story-unit narrative table to {out_csv}")
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
