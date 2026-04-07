from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .story_generation_design import _cohort_label, _profile_label, _read_story_ids


_STORY_ORDER = ["A1", "A2", "A3", "B1", "B2", "B3"]
_AGE_RE = re.compile(r"^(?P<years>\d+);(?P<months>\d+)\.(?P<days>\d+)$")
_SPLIT_TRANSCRIPT_RE = re.compile(r"^(?P<file_id>\d+)_(?P<story>[AB][123])$")
_BASE_TRANSCRIPT_RE = re.compile(r"^(?P<file_id>\d+)$")


@dataclass(frozen=True)
class ChildRecord:
    file_id: str
    group: str
    age: str
    age_value: float
    age_months: int
    file_kideval: str
    severity_band: str
    profile_label: str
    base_transcript_path: Path
    available_stories: tuple[str, ...]


@dataclass(frozen=True)
class ReferenceRow:
    target_story: str
    e1_story: str
    e2_story: str
    target_td_child_id: str
    target_td_age: str
    target_td_age_value: float
    prompt_sli_child_id: str
    prompt_sli_age: str
    prompt_sli_age_value: float
    prompt_sli_severity_band: str
    prompt_sli_profile_label: str


def _parse_age_value(age: str) -> float:
    match = _AGE_RE.match(age.strip())
    if not match:
        raise ValueError(f"Unsupported age format: {age!r}")
    years = int(match.group("years"))
    months = int(match.group("months"))
    days = int(match.group("days"))
    return years * 12 + months + (days / 31.0)


def _load_transcript_index(transcript_root: Path) -> tuple[dict[tuple[str, str], Path], dict[tuple[str, str, str], Path]]:
    base_index: dict[tuple[str, str], Path] = {}
    split_index: dict[tuple[str, str, str], Path] = {}
    for path in transcript_root.rglob("*.cha"):
        rel_parts = path.relative_to(transcript_root).parts
        if len(rel_parts) < 2:
            continue
        group = rel_parts[0]
        stem = path.stem
        split_match = _SPLIT_TRANSCRIPT_RE.match(stem)
        if split_match:
            split_index[(group, split_match.group("file_id"), split_match.group("story"))] = path
            continue
        base_match = _BASE_TRANSCRIPT_RE.match(stem)
        if base_match:
            base_index[(group, base_match.group("file_id"))] = path
    return base_index, split_index


def _load_child_records(
    dev_measures_csv: Path,
    severity_csv: Path,
    transcript_root: Path,
    cohort: str,
) -> list[ChildRecord]:
    base_index, split_index = _load_transcript_index(transcript_root)

    with dev_measures_csv.open(newline="", encoding="utf-8") as handle:
        dev_rows = list(csv.DictReader(handle))
    with severity_csv.open(newline="", encoding="utf-8") as handle:
        severity_rows = list(csv.DictReader(handle))

    severity_by_id = {
        row["File_ID"]: row
        for row in severity_rows
        if row.get("Group") in {"TD", "SLI"}
    }

    seen_ids: set[str] = set()
    children: list[ChildRecord] = []
    for row in dev_rows:
        file_id = row.get("File_ID", "").strip()
        group = row.get("Group", "").strip()
        if not file_id or file_id in seen_ids or group not in {"TD", "SLI"}:
            continue

        age_months_raw = row.get("Age(Month)", "").strip()
        if not age_months_raw:
            continue
        age_months = int(float(age_months_raw))
        if _cohort_label(age_months) != cohort:
            continue

        severity_row = severity_by_id.get(file_id)
        if severity_row is None:
            continue

        transcript_path = base_index.get((group, file_id))
        if transcript_path is None:
            continue

        split_stories = {
            story
            for (indexed_group, indexed_file_id, story), _path in split_index.items()
            if indexed_group == group and indexed_file_id == file_id and story in _STORY_ORDER
        }
        available_stories = tuple(
            story
            for story in _STORY_ORDER
            if story in set(_read_story_ids(transcript_path)) or story in split_stories
        )
        if not available_stories:
            continue

        age = row.get("Age", "").strip()
        if not age:
            continue

        severity_band = "typical"
        profile_label = "typical"
        if group == "SLI":
            severity_band = severity_row.get("severity_band_sli_tertile_sli_only", "").strip() or "unspecified"
            profile_label = _profile_label(severity_row)

        children.append(
            ChildRecord(
                file_id=file_id,
                group=group,
                age=age,
                age_value=_parse_age_value(age),
                age_months=age_months,
                file_kideval=row.get("filename", "").strip() or severity_row.get("File_kideval", "").strip(),
                severity_band=severity_band,
                profile_label=profile_label,
                base_transcript_path=transcript_path,
                available_stories=available_stories,
            )
        )
        seen_ids.add(file_id)

    return sorted(children, key=lambda child: (child.group, child.age_value, int(child.file_id)))


def _load_reference_manifest(reference_manifest_csv: Path) -> list[ReferenceRow]:
    with reference_manifest_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    reference_rows: list[ReferenceRow] = []
    for row in rows:
        reference_rows.append(
            ReferenceRow(
                target_story=row["target_story"],
                e1_story=row["e1_story"],
                e2_story=row["e2_story"],
                target_td_child_id=row["target_td_child_id"],
                target_td_age=row["target_td_age"],
                target_td_age_value=_parse_age_value(row["target_td_age"]),
                prompt_sli_child_id=row["prompt_sli_child_id"],
                prompt_sli_age=row["prompt_sli_age"],
                prompt_sli_age_value=_parse_age_value(row["prompt_sli_age"]),
                prompt_sli_severity_band=row["prompt_sli_severity_band"],
                prompt_sli_profile_label=row["prompt_sli_profile_label"],
            )
        )
    return sorted(reference_rows, key=lambda row: _STORY_ORDER.index(row.target_story))


def _assign_eval_td_children(
    sli_children: list[ChildRecord],
    td_children: list[ChildRecord],
) -> dict[str, ChildRecord]:
    remaining = td_children[:]
    assignments: dict[str, ChildRecord] = {}
    for sli_child in sorted(sli_children, key=lambda child: (child.age_value, int(child.file_id))):
        chosen = min(
            remaining,
            key=lambda td_child: (abs(td_child.age_value - sli_child.age_value), td_child.age_value, int(td_child.file_id)),
        )
        assignments[sli_child.file_id] = chosen
        remaining.remove(chosen)
    return assignments


def _locate_story_transcript(
    child: ChildRecord,
    story_id: str,
    split_index: dict[tuple[str, str, str], Path],
) -> tuple[Path, str]:
    split_path = split_index.get((child.group, child.file_id, story_id))
    if split_path is not None:
        return split_path, "single_story"
    if story_id in child.available_stories:
        return child.base_transcript_path, "multi_story"
    raise ValueError(f"Story {story_id} not available for child {child.file_id}")


def _choose_td_prompt(
    candidates: list[ChildRecord],
    reference_row: ReferenceRow,
    usage_counts: dict[str, int],
) -> ChildRecord:
    eligible = [
        child
        for child in candidates
        if reference_row.target_story in child.available_stories and reference_row.e1_story in child.available_stories
    ]
    if not eligible:
        raise ValueError(f"No TD prompt candidate found for target {reference_row.target_story}")
    return min(
        eligible,
        key=lambda child: (
            usage_counts.get(child.file_id, 0),
            abs(child.age_value - reference_row.target_td_age_value),
            int(child.file_id),
        ),
    )


def _choose_sli_prompt(
    candidates: list[ChildRecord],
    reference_row: ReferenceRow,
    usage_counts: dict[str, int],
) -> ChildRecord:
    eligible = [child for child in candidates if reference_row.e2_story in child.available_stories]
    if not eligible:
        raise ValueError(f"No SLI prompt candidate found for e2 story {reference_row.e2_story}")
    return min(
        eligible,
        key=lambda child: (
            0 if child.profile_label == reference_row.prompt_sli_profile_label else 1,
            0 if child.severity_band == reference_row.prompt_sli_severity_band else 1,
            usage_counts.get(child.file_id, 0),
            abs(child.age_value - reference_row.prompt_sli_age_value),
            int(child.file_id),
        ),
    )


def build_architecture_loocv_plan(
    dev_measures_csv: str,
    severity_csv: str,
    transcript_root: str,
    reference_manifest_csv: str,
    output_dir: str,
    cohort: str = "4-year-old",
) -> int:
    dev_path = Path(dev_measures_csv).expanduser().resolve()
    severity_path = Path(severity_csv).expanduser().resolve()
    transcript_root_path = Path(transcript_root).expanduser().resolve()
    reference_path = Path(reference_manifest_csv).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    children = _load_child_records(
        dev_measures_csv=dev_path,
        severity_csv=severity_path,
        transcript_root=transcript_root_path,
        cohort=cohort,
    )
    reference_rows = _load_reference_manifest(reference_path)
    _, split_index = _load_transcript_index(transcript_root_path)

    td_children = [child for child in children if child.group == "TD"]
    sli_children = [child for child in children if child.group == "SLI"]
    td_eval_assignments = _assign_eval_td_children(sli_children=sli_children, td_children=td_children)

    fold_rows: list[dict[str, object]] = []
    prompt_manifest_rows: list[dict[str, object]] = []
    eval_manifest_rows: list[dict[str, object]] = []

    for fold_number, eval_sli in enumerate(sorted(sli_children, key=lambda child: (child.age_value, int(child.file_id))), start=1):
        eval_td = td_eval_assignments[eval_sli.file_id]
        fold_id = f"fold_{fold_number:02d}_sli_{eval_sli.file_id}"

        td_prompt_pool = [child for child in td_children if child.file_id != eval_td.file_id]
        sli_prompt_pool = [child for child in sli_children if child.file_id != eval_sli.file_id]

        td_usage: dict[str, int] = {}
        sli_usage: dict[str, int] = {}
        fold_td_prompt_ids: list[str] = []
        fold_sli_prompt_ids: list[str] = []

        for reference_row in reference_rows:
            td_prompt = _choose_td_prompt(td_prompt_pool, reference_row, td_usage)
            sli_prompt = _choose_sli_prompt(sli_prompt_pool, reference_row, sli_usage)

            td_target_path, td_target_scope = _locate_story_transcript(td_prompt, reference_row.target_story, split_index)
            td_e1_path, td_e1_scope = _locate_story_transcript(td_prompt, reference_row.e1_story, split_index)
            sli_e2_path, sli_e2_scope = _locate_story_transcript(sli_prompt, reference_row.e2_story, split_index)

            prompt_manifest_rows.append(
                {
                    "fold_id": fold_id,
                    "heldout_sli_child_id": eval_sli.file_id,
                    "heldout_sli_age": eval_sli.age,
                    "heldout_td_child_id": eval_td.file_id,
                    "heldout_td_age": eval_td.age,
                    "target_story": reference_row.target_story,
                    "e1_story": reference_row.e1_story,
                    "e2_story": reference_row.e2_story,
                    "reference_td_child_id": reference_row.target_td_child_id,
                    "reference_td_age": reference_row.target_td_age,
                    "target_td_child_id": td_prompt.file_id,
                    "target_td_age": td_prompt.age,
                    "target_td_target_story_path": str(td_target_path),
                    "target_td_target_story_scope": td_target_scope,
                    "target_td_e1_story_path": str(td_e1_path),
                    "target_td_e1_story_scope": td_e1_scope,
                    "reference_sli_child_id": reference_row.prompt_sli_child_id,
                    "reference_sli_age": reference_row.prompt_sli_age,
                    "reference_sli_severity_band": reference_row.prompt_sli_severity_band,
                    "reference_sli_profile_label": reference_row.prompt_sli_profile_label,
                    "prompt_sli_child_id": sli_prompt.file_id,
                    "prompt_sli_age": sli_prompt.age,
                    "prompt_sli_severity_band": sli_prompt.severity_band,
                    "prompt_sli_profile_label": sli_prompt.profile_label,
                    "prompt_sli_e2_story_path": str(sli_e2_path),
                    "prompt_sli_e2_story_scope": sli_e2_scope,
                }
            )

            td_usage[td_prompt.file_id] = td_usage.get(td_prompt.file_id, 0) + 1
            sli_usage[sli_prompt.file_id] = sli_usage.get(sli_prompt.file_id, 0) + 1
            fold_td_prompt_ids.append(td_prompt.file_id)
            fold_sli_prompt_ids.append(sli_prompt.file_id)

        for eval_child in [eval_td, eval_sli]:
            for story_id in eval_child.available_stories:
                eval_path, eval_scope = _locate_story_transcript(eval_child, story_id, split_index)
                eval_manifest_rows.append(
                    {
                        "fold_id": fold_id,
                        "eval_group": eval_child.group,
                        "eval_child_id": eval_child.file_id,
                        "eval_child_age": eval_child.age,
                        "eval_story_id": story_id,
                        "eval_transcript_path": str(eval_path),
                        "eval_transcript_scope": eval_scope,
                    }
                )

        fold_rows.append(
            {
                "fold_id": fold_id,
                "cohort": cohort,
                "heldout_sli_child_id": eval_sli.file_id,
                "heldout_sli_age": eval_sli.age,
                "heldout_sli_severity_band": eval_sli.severity_band,
                "heldout_sli_profile_label": eval_sli.profile_label,
                "heldout_td_child_id": eval_td.file_id,
                "heldout_td_age": eval_td.age,
                "heldout_td_age_gap_months": round(abs(eval_td.age_value - eval_sli.age_value), 3),
                "n_target_rows": len(reference_rows),
                "n_unique_td_prompt_children": len(set(fold_td_prompt_ids)),
                "n_unique_sli_prompt_children": len(set(fold_sli_prompt_ids)),
                "td_prompt_child_ids": "/".join(sorted(set(fold_td_prompt_ids), key=int)),
                "sli_prompt_child_ids": "/".join(sorted(set(fold_sli_prompt_ids), key=int)),
            }
        )

    fold_manifest_path = out_dir / "loocv_folds.csv"
    prompt_manifest_path = out_dir / "loocv_prompt_manifest.csv"
    eval_manifest_path = out_dir / "loocv_eval_manifest.csv"
    summary_path = out_dir / "loocv_summary.json"

    with fold_manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fold_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fold_rows)

    with prompt_manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prompt_manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(prompt_manifest_rows)

    with eval_manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(eval_manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(eval_manifest_rows)

    summary = {
        "cohort": cohort,
        "reference_manifest_csv": str(reference_path),
        "transcript_root": str(transcript_root_path),
        "folds": len(fold_rows),
        "td_children_in_pool": len(td_children),
        "sli_children_in_pool": len(sli_children),
        "stories_per_fold": len(reference_rows),
        "eval_rows_per_fold_summary": {
            "min": min(sum(1 for row in eval_manifest_rows if row["fold_id"] == fold["fold_id"]) for fold in fold_rows),
            "max": max(sum(1 for row in eval_manifest_rows if row["fold_id"] == fold["fold_id"]) for fold in fold_rows),
        },
        "unique_td_eval_children": len({row["heldout_td_child_id"] for row in fold_rows}),
        "unique_sli_eval_children": len({row["heldout_sli_child_id"] for row in fold_rows}),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote fold summary to {fold_manifest_path}")
    print(f"Wrote prompt manifest to {prompt_manifest_path}")
    print(f"Wrote evaluation manifest to {eval_manifest_path}")
    print(f"Wrote summary to {summary_path}")
    return 0
