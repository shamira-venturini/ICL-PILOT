"""Build traceable all-age story-text exports for corpus-level evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


DEFAULT_GILLAM_TD_SELECTION_SEED = 20260827

RECORD_COLUMNS: Tuple[str, ...] = (
    "source",
    "source_dataset",
    "source_csv",
    "source_row_number",
    "cohort_year",
    "cohort_label",
    "round_id",
    "bundle_id",
    "pair_id",
    "replicate_id",
    "story_slot_order",
    "story_id",
    "story_title",
    "stage",
    "group",
    "group_label_source",
    "comparison_target_group",
    "source_child_id",
    "td_child_id",
    "sli_child_id",
    "age_months",
    "age_raw",
    "prompt_td_story_e1",
    "prompt_sli_story_e2",
    "transcript_path",
    "text_normalization",
    "story_text",
)

GILLAM_RECORD_COLUMNS: Tuple[str, ...] = (
    "source",
    "source_dataset",
    "source_csv",
    "source_row_number",
    "source_child_id",
    "group",
    "group_label_source",
    "story_id",
    "story_title",
    "story_slot_order",
    "age_months",
    "cohort_year",
    "cohort_label",
    "sex",
    "transcript_path",
    "transcript_corpus_relative_path",
    "transcript_sha256",
    "n_child_utterances",
    "n_child_tokens",
    "text_normalization",
    "story_text",
)


def _relative_path(path: Path, project_root: Path) -> str:
    return str(path.resolve().relative_to(project_root.resolve()))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError("{} is missing required columns: {}".format(source, sorted(missing)))


def _text_or_none(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _group_from_transcript_path(value: object) -> Optional[str]:
    """Return an explicit ENNI corpus label embedded in a transcript path.

    This is deliberately limited to literal directory names, rather than any
    linguistic or derived-measure inference.  It is used only where the
    story-unit table has a missing group label.
    """

    if pd.isna(value):
        return None
    candidates = {
        part.upper()
        for part in Path(str(value)).parts
        if part.upper() in {"TD", "SLI"}
    }
    if len(candidates) == 1:
        return candidates.pop()
    return None


def _clean_chat_child_text(text: str) -> str:
    """Normalize CHAT child utterances for the text-classification input.

    The transformation is intentionally conservative and matches the existing
    raw-text classification experiment: CHAT annotations are removed, lexical
    material within retracing parentheses is retained, and the result is
    lower-cased word text.  Raw CHAT files remain unchanged and are referenced
    by path and checksum in the derived export.
    """

    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"&=[A-Za-z_:-]+", " ", text)
    text = re.sub(r"\bxxx\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\(([^)]*)\)", r"\1", text)
    text = text.replace(":", "")
    text = re.sub(r"[^A-Za-z' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _clean_enni_story_text(text: str) -> str:
    """Convert extracted ENNI CHAT text to readable plain narrative text.

    This removes transcription notation while retaining the child's lexical
    material, grammatical forms, repetitions, retracings, and punctuation.
    Parenthesized partial-word restorations are joined to the surrounding word
    (for example, ``(be)cause`` becomes ``because``), while pause markers and
    non-speech events are removed.
    """

    text = re.sub(r"\x15[^\x15]*\x15", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"&\*[^\s]+", " ", text)
    text = re.sub(r"&=[A-Za-z_:-]+", " ", text)
    text = re.sub(r"&[+-][A-Za-z:']+", " ", text)
    text = text.replace("&~", "")
    text = re.sub(r"<([^>]*)>", r"\1", text)
    text = re.sub(r"\((?:\.+)\)", " ", text)
    text = re.sub(r"\(([^)]*)\)", r"\1", text)
    text = re.sub(r"\b([A-Za-z']+)@[A-Za-z]+\b", r"\1", text)
    text = re.sub(r"\b0[A-Za-z']+", " ", text)
    text = re.sub(r"\b(?:xxx|yyy|www)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\S)\++(?:[\"/.!?]+)?", " ", text)
    text = re.sub(r"(?<!\S):(?!\S)", " ", text)
    text = re.sub(r"\s+([,.;?!])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_chat_child_utterances(raw_text: str) -> List[str]:
    """Read complete *CHI turns, including their continuation lines."""

    utterances: List[str] = []
    current: List[str] = []
    in_child = False
    for raw_line in raw_text.splitlines():
        if raw_line.startswith("*CHI:"):
            if current:
                utterances.append(" ".join(current))
            current = [raw_line.split(":", 1)[1].strip()]
            in_child = True
        elif raw_line.startswith("*") or raw_line.startswith("%") or raw_line.startswith("@"):
            if current:
                utterances.append(" ".join(current))
            current = []
            in_child = False
        elif in_child and (raw_line.startswith("\t") or raw_line.startswith(" ")):
            current.append(raw_line.strip())
    if current:
        utterances.append(" ".join(current))
    return utterances


def _extract_chat_child_stories(raw_text: str) -> List[Dict[str, object]]:
    """Extract story-bounded *CHI turns using literal Gillam @G markers."""

    stories: List[Dict[str, object]] = []
    current_story: Optional[Dict[str, object]] = None
    current_utterance: List[str] = []
    in_child = False

    def flush_utterance() -> None:
        nonlocal current_utterance
        if current_story is not None and current_utterance:
            current_story["utterances"].append(" ".join(current_utterance))
        current_utterance = []

    for line_number, raw_line in enumerate(raw_text.splitlines(), start=1):
        story_match = re.match(r"^@G:\s*(.*?)\s*$", raw_line)
        if story_match:
            flush_utterance()
            if current_story is not None:
                stories.append(current_story)
            current_story = {
                "story_id": story_match.group(1),
                "story_marker_line_number": line_number,
                "utterances": [],
            }
            in_child = False
            continue
        if raw_line.startswith("*CHI:"):
            flush_utterance()
            if current_story is not None:
                current_utterance = [raw_line.split(":", 1)[1].strip()]
                in_child = True
            continue
        if raw_line.startswith("*") or raw_line.startswith("%") or raw_line.startswith("@"):
            flush_utterance()
            in_child = False
            continue
        if in_child and (raw_line.startswith("\t") or raw_line.startswith(" ")):
            current_utterance.append(raw_line.strip())

    flush_utterance()
    if current_story is not None:
        stories.append(current_story)
    return stories


def _gillam_child_metadata(raw_text: str, transcript: Path, corpus_root: Path) -> Dict[str, object]:
    id_match = re.search(
        r"^@ID:\s*eng\|Gillam\|CHI\|([^|]+)\|([^|]*)\|",
        raw_text,
        flags=re.MULTILINE,
    )
    if id_match is None:
        raise ValueError("Missing Gillam child @ID metadata in {}".format(transcript))
    age_raw, sex = id_match.groups()
    age_match = re.match(r"^(\d+);(\d*)", age_raw)
    if age_match is None:
        raise ValueError("Unparseable Gillam age {!r} in {}".format(age_raw, transcript))
    years = int(age_match.group(1))
    months = int(age_match.group(2) or 0)
    relative_parts = transcript.resolve().relative_to(corpus_root.resolve()).parts
    group_candidates = [part.upper() for part in relative_parts if part.upper() in {"TD", "SLI"}]
    if len(set(group_candidates)) != 1:
        raise ValueError("Cannot resolve a literal Gillam group from {}".format(transcript))
    return {
        "source_child_id": transcript.stem,
        "group": group_candidates[0],
        "age_raw": age_raw,
        "age_months": years * 12 + months,
        "cohort_year": years,
        "sex": sex.strip().lower(),
    }


def _generated_records(
    *,
    source_path: Path,
    cohort_year: int,
    project_root: Path,
) -> List[Dict[str, object]]:
    frame = pd.read_csv(source_path)
    _require_columns(
        frame,
        (
            "round_id",
            "bundle_id",
            "pair_id",
            "replicate_id",
            "target_story",
            "stage1_td_story",
            "stage2_sli_story",
            "td_child_id",
            "sli_child_id",
        ),
        source_path,
    )

    source_csv = _relative_path(source_path, project_root)
    records: List[Dict[str, object]] = []
    for row_index, row in frame.iterrows():
        common = {
            "source": "synthetic",
            "source_dataset": "ICL-PILOT generated ENNI narratives",
            "source_csv": source_csv,
            "source_row_number": int(row_index) + 2,
            "cohort_year": cohort_year,
            "cohort_label": row.get("cohort_label", "{}-year-old".format(cohort_year)),
            "round_id": row.get("round_id"),
            "bundle_id": row.get("bundle_id"),
            "pair_id": row.get("pair_id"),
            "replicate_id": row.get("replicate_id"),
            "story_slot_order": row.get("story_slot_order"),
            "story_id": row.get("target_story"),
            "story_title": row.get("story_title"),
            "td_child_id": row.get("td_child_id"),
            "sli_child_id": row.get("sli_child_id"),
            "age_months": pd.NA,
            "age_raw": row.get("sli_age"),
            "prompt_td_story_e1": row.get("prompt_td_story_e1"),
            "prompt_sli_story_e2": row.get("prompt_sli_story_e2"),
            "transcript_path": pd.NA,
            "text_normalization": "synthetic_source_text_as_exported",
        }
        stage_specs = (
            ("stage1_td", row.get("stage1_group", "TD"), "td_child_id", "stage1_td_story"),
            ("stage2_sli", row.get("stage2_group", "SLI"), "sli_child_id", "stage2_sli_story"),
        )
        for stage, group, source_child_key, story_key in stage_specs:
            story_text = _text_or_none(row.get(story_key))
            if story_text is None:
                continue
            record = dict(common)
            record.update(
                {
                    "stage": stage,
                    "group": group,
                    "group_label_source": "generation_csv",
                    "comparison_target_group": group,
                    "source_child_id": row.get(source_child_key),
                    "story_text": story_text,
                }
            )
            records.append(record)
    return records


def _real_records(
    *,
    measures_path: Path,
    stories_path: Path,
    project_root: Path,
    minimum_age_months: float,
    maximum_age_months: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    measures = pd.read_csv(measures_path)
    stories = pd.read_csv(stories_path)
    _require_columns(measures, ("File_ID", "Group", "Age(Month)"), measures_path)
    _require_columns(
        stories,
        ("File_ID", "Group", "Age_Raw", "transcript_path", "story_id", "story_text"),
        stories_path,
    )

    # Do not alter the source table.  A small number of its rows lack a Group
    # value even though their transcript path contains the literal ENNI corpus
    # directory label (for example, /TD/B/523.cha).  Resolve only those blank
    # cells from that direct source metadata and retain an audit field.
    stories = stories.copy()
    stories["_source_row_number"] = stories.index + 2
    stories["Group"] = stories["Group"].replace(r"^\s*$", pd.NA, regex=True)
    stories["group_label_source"] = "story_unit_narratives"
    missing_group = stories["Group"].isna()
    if missing_group.any():
        resolved_groups = stories.loc[missing_group, "transcript_path"].map(
            _group_from_transcript_path
        )
        unresolved_index = resolved_groups.index[resolved_groups.isna()]
        if len(unresolved_index):
            unresolved_rows = stories.loc[
                unresolved_index, ["File_ID", "transcript_path", "story_id"]
            ].to_dict(orient="records")
            raise ValueError(
                "Cannot resolve a missing ENNI Group from transcript-path metadata: {}"
                .format(unresolved_rows)
            )
        stories.loc[resolved_groups.index, "Group"] = resolved_groups
        stories.loc[resolved_groups.index, "group_label_source"] = "transcript_path"

    children = measures.loc[
        measures["Age(Month)"].between(minimum_age_months, maximum_age_months),
        ["File_ID", "Group", "Age(Month)"],
    ].copy()
    children = children.drop_duplicates(subset=["File_ID", "Group"])
    available_story_children = stories[["File_ID", "Group"]].drop_duplicates()
    missing_story_children = children.merge(
        available_story_children,
        on=["File_ID", "Group"],
        how="left",
        indicator=True,
    )
    missing_story_children = missing_story_children.loc[
        missing_story_children["_merge"] == "left_only"
    ].drop(columns="_merge")
    joined = stories.merge(
        children,
        on=["File_ID", "Group"],
        how="inner",
        validate="many_to_one",
    )
    if joined.empty:
        raise ValueError("No real ENNI stories matched the requested age range.")

    source_csv = _relative_path(stories_path, project_root)
    records: List[Dict[str, object]] = []
    for row_index, row in joined.iterrows():
        source_story_text = _text_or_none(row.get("story_text"))
        if source_story_text is None:
            continue
        story_text = _clean_enni_story_text(source_story_text)
        if not story_text:
            raise ValueError(
                "ENNI text normalization removed all content for File_ID={} story_id={}.".format(
                    row.get("File_ID"), row.get("story_id")
                )
            )
        age_months = float(row["Age(Month)"])
        records.append(
            {
                "source": "real",
                "source_dataset": "ENNI real narratives",
                "source_csv": source_csv,
                "source_row_number": int(row["_source_row_number"]),
                "cohort_year": int(age_months // 12),
                "cohort_label": "{}-year-old".format(int(age_months // 12)),
                "round_id": pd.NA,
                "bundle_id": pd.NA,
                "pair_id": pd.NA,
                "replicate_id": pd.NA,
                "story_slot_order": pd.NA,
                "story_id": row.get("story_id"),
                "story_title": pd.NA,
                "stage": pd.NA,
                "group": row.get("Group"),
                "group_label_source": row.get("group_label_source"),
                "comparison_target_group": row.get("Group"),
                "source_child_id": row.get("File_ID"),
                "td_child_id": pd.NA,
                "sli_child_id": pd.NA,
                "age_months": age_months,
                "age_raw": row.get("Age_Raw"),
                "prompt_td_story_e1": pd.NA,
                "prompt_sli_story_e2": pd.NA,
                "transcript_path": row.get("transcript_path"),
                "text_normalization": "enni_chat_plain_text_v1",
                "story_text": story_text,
            }
        )
    return records, missing_story_children.to_dict(orient="records")


def _require_explicit_paths(**paths) -> None:
    missing = [name for name, value in paths.items() if value is None or not str(value).strip()]
    if missing:
        raise ValueError("Provide explicit paths for " + ", ".join(missing) + "; historical Phase 2 defaults are retired.")


def build_gillam_external_text_records(
    *,
    project_root: str | Path = ".",
    output_csv: str | Path | None = None,
    transcripts_dir: str | Path | None = None,
    minimum_year: int = 5,
    maximum_year: int = 7,
    td_selection_seed: int = DEFAULT_GILLAM_TD_SELECTION_SEED,
) -> Mapping[str, Path]:
    """Export one labelled Gillam story per row for external testing.

    This export is deliberately separate from the all-story ENNI source used
    for generated-versus-real corpus comparisons. All eligible SLI children are
    retained. TD children with the three canonical stories are selected using
    a deterministic hash rank to match SLI counts in every age-by-sex stratum.
    Gillam remains test-only in the notebook's external-transfer evaluation.
    """

    _require_explicit_paths(output_csv=output_csv, transcripts_dir=transcripts_dir)
    root = Path(project_root).expanduser().resolve()
    output_path = (root / output_csv).resolve()
    transcripts_arg = Path(transcripts_dir).expanduser()
    transcripts_path = (
        transcripts_arg.resolve()
        if transcripts_arg.is_absolute()
        else (root / transcripts_arg).resolve()
    )
    canonical_stories = ("McDonalds", "LateForSchool", "Aliens")
    story_order = {story: index for index, story in enumerate(canonical_stories, start=1)}

    children: List[Dict[str, object]] = []
    seen_ids: Dict[str, str] = {}
    for transcript in sorted(transcripts_path.rglob("*.cha")):
        raw_text = transcript.read_text(encoding="utf-8", errors="ignore")
        metadata = _gillam_child_metadata(raw_text, transcript, transcripts_path)
        child_id = str(metadata["source_child_id"])
        if child_id in seen_ids:
            raise ValueError(
                "Duplicate Gillam transcript ID {}: {}, {}".format(
                    child_id, seen_ids[child_id], transcript
                )
            )
        seen_ids[child_id] = str(transcript)
        if not minimum_year <= int(metadata["cohort_year"]) <= maximum_year:
            continue
        stories = _extract_chat_child_stories(raw_text)
        children.append({**metadata, "transcript": transcript, "stories": stories})

    sli_children = [child for child in children if child["group"] == "SLI"]
    td_children = [child for child in children if child["group"] == "TD"]
    sli_strata = Counter((child["cohort_year"], child["sex"]) for child in sli_children)
    complete_td: Dict[Tuple[int, str], List[Dict[str, object]]] = defaultdict(list)
    excluded_td: List[Dict[str, object]] = []
    for child in td_children:
        sequence = [story["story_id"] for story in child["stories"]]
        if Counter(sequence) == Counter(canonical_stories):
            complete_td[(int(child["cohort_year"]), str(child["sex"]))].append(child)
        else:
            excluded_td.append(
                {"source_child_id": child["source_child_id"], "story_sequence": sequence}
            )

    selected_td: List[Dict[str, object]] = []
    selected_td_by_stratum: Dict[str, List[str]] = {}
    for stratum, required in sorted(sli_strata.items()):
        candidates = complete_td.get(stratum, [])
        if len(candidates) < required:
            raise ValueError(
                "Insufficient complete TD transcripts for stratum {}: need {}, found {}".format(
                    stratum, required, len(candidates)
                )
            )
        ranked = sorted(
            candidates,
            key=lambda child: hashlib.sha256(
                "{}|{}".format(td_selection_seed, child["source_child_id"]).encode("utf-8")
            ).hexdigest(),
        )
        chosen = ranked[:required]
        selected_td.extend(chosen)
        selected_td_by_stratum["{}|{}".format(*stratum)] = [
            str(child["source_child_id"]) for child in chosen
        ]

    selected_children = sli_children + selected_td
    if len(sli_children) != len(selected_td):
        raise ValueError("Gillam child-level group matching failed.")

    records: List[Dict[str, object]] = []
    metadata_path = transcripts_path / "0metadata.cdc"
    source_metadata = str(metadata_path.resolve()) if metadata_path.exists() else pd.NA
    for child in sorted(selected_children, key=lambda item: str(item["source_child_id"])):
        transcript = Path(child["transcript"])
        for story in child["stories"]:
            story_id = str(story["story_id"])
            if story_id not in story_order:
                continue
            utterances = list(story["utterances"])
            story_text = _clean_enni_story_text(" ".join(utterances))
            if not story_text:
                raise ValueError(
                    "No usable *CHI text for {} / {}".format(child["source_child_id"], story_id)
                )
            records.append(
                {
                    "source": "real_external",
                    "source_dataset": "Gillam real narratives",
                    "source_csv": source_metadata,
                    "source_row_number": int(story["story_marker_line_number"]),
                    "source_child_id": child["source_child_id"],
                    "group": child["group"],
                    "group_label_source": "literal Gillam corpus directory",
                    "story_id": story_id,
                    "story_title": story_id,
                    "story_slot_order": story_order[story_id],
                    "age_months": child["age_months"],
                    "cohort_year": child["cohort_year"],
                    "cohort_label": "{}-year-old".format(child["cohort_year"]),
                    "sex": child["sex"],
                    "transcript_path": str(transcript.resolve()),
                    "transcript_corpus_relative_path": str(transcript.relative_to(transcripts_path)),
                    "transcript_sha256": _sha256(transcript),
                    "n_child_utterances": len(utterances),
                    "n_child_tokens": len(story_text.split()),
                    "text_normalization": "gillam_chat_plain_text_v2",
                    "story_text": story_text,
                }
            )

    frame = pd.DataFrame(records, columns=list(GILLAM_RECORD_COLUMNS))
    if frame.duplicated(["source_child_id", "story_id"]).any():
        raise ValueError("Gillam export contains duplicate child/story rows.")
    if frame["story_text"].isna().any() or (frame["story_text"].str.strip() == "").any():
        raise ValueError("Gillam export contains empty story text.")
    unique_by_group = frame.groupby("group")["source_child_id"].nunique().to_dict()
    if unique_by_group.get("TD") != unique_by_group.get("SLI"):
        raise ValueError("Gillam export is not balanced by unique child.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    summary_path = output_path.with_name("{}_summary.json".format(output_path.stem))
    summary = {
        "output_csv": _relative_path(output_path, root),
        "document_unit": "one Gillam @G-bounded story",
        "intended_use": "external test only; never append to ENNI training or generic real-vs-synthetic corpora",
        "source": {
            "transcript_root": str(transcripts_path),
            "metadata_path": source_metadata,
            "metadata_sha256": _sha256(metadata_path) if metadata_path.exists() else None,
        },
        "selection": {
            "minimum_year": minimum_year,
            "maximum_year": maximum_year,
            "sli_rule": "all eligible SLI children",
            "td_rule": "complete canonical-story transcripts, hash-ranked to match each SLI age-by-sex stratum",
            "td_selection_seed": td_selection_seed,
            "selected_td_child_ids_by_age_sex": selected_td_by_stratum,
            "excluded_noncanonical_td_transcripts": excluded_td,
        },
        "column_semantics": {
            "source_row_number": "one-based line number of the story's @G marker in the source CHAT transcript"
        },
        "text_normalization": "gillam_chat_plain_text_v2",
        "records_by_group": frame.groupby("group").size().to_dict(),
        "unique_children_by_group": unique_by_group,
        "unique_children_by_age_sex_group": (
            frame.drop_duplicates(["source_child_id"])
            .groupby(["cohort_year", "sex", "group"])
            .size().reset_index(name="n_children").to_dict(orient="records")
        ),
        "records_by_story_group": (
            frame.groupby(["story_id", "group"]).size()
            .reset_index(name="n_records").to_dict(orient="records")
        ),
        "total_records": int(len(frame)),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"output_csv": output_path, "summary_json": summary_path}


def build_all_story_text_records(
    *,
    project_root: str | Path = ".",
    output_csv: str | Path | None = None,
    generation_sources: Optional[Sequence[Tuple[int, str]]] = None,
    real_measures_csv: str | Path | None = None,
    real_stories_csv: str | Path | None = None,
    minimum_age_months: float = 49.9,
    maximum_age_months: float = 95,
) -> Mapping[str, Path]:
    """Export all 4--7-year-old generated stories plus matched real ENNI texts.

    The output stays in long form: every generated stage and every real
    story is one row. Source CSV and row-number fields preserve provenance.
    Literal SLI labels are retained from the corpus.
    """

    if not generation_sources:
        raise ValueError("Provide generation_sources as (cohort_year, CSV path) pairs from the intended generation run.")

    _require_explicit_paths(output_csv=output_csv, real_measures_csv=real_measures_csv, real_stories_csv=real_stories_csv)
    root = Path(project_root).expanduser().resolve()
    output_path = (root / output_csv).resolve()
    measures_path = (root / real_measures_csv).resolve()
    stories_path = (root / real_stories_csv).resolve()

    generated: List[Dict[str, object]] = []
    source_inventory: List[Dict[str, object]] = []
    for cohort_year, source in generation_sources:
        source_path = (root / source).resolve()
        rows = _generated_records(
            source_path=source_path,
            cohort_year=cohort_year,
            project_root=root,
        )
        generated.extend(rows)
        source_inventory.append(
            {
                "cohort_year": cohort_year,
                "path": _relative_path(source_path, root),
                "sha256": _sha256(source_path),
                "exported_story_rows": len(rows),
            }
        )

    real, real_children_without_story_text = _real_records(
        measures_path=measures_path,
        stories_path=stories_path,
        project_root=root,
        minimum_age_months=minimum_age_months,
        maximum_age_months=maximum_age_months,
    )
    records = pd.DataFrame(generated + real, columns=list(RECORD_COLUMNS))
    if records.empty:
        raise ValueError("The story-text export is empty.")
    if records["story_text"].isna().any() or (records["story_text"].str.strip() == "").any():
        raise ValueError("The story-text export contains empty narratives.")

    generated_count = int((records["source"] == "synthetic").sum())
    expected_generated_count = sum(item["exported_story_rows"] for item in source_inventory)
    if generated_count != expected_generated_count:
        raise ValueError(
            "Generated-row count mismatch: expected {}, exported {}.".format(
                expected_generated_count, generated_count
            )
        )
    if records.duplicated(
        subset=["source", "source_csv", "source_row_number", "stage"]
    ).any():
        raise ValueError("Duplicate source-row/stage provenance in story-text export.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(output_path, index=False)

    summary_path = output_path.with_name("{}_summary.json".format(output_path.stem))
    summary = {
        "output_csv": _relative_path(output_path, root),
        "minimum_age_months": minimum_age_months,
        "maximum_age_months": maximum_age_months,
        "generated_sources": source_inventory,
        "real_sources": [
            {"path": _relative_path(measures_path, root), "sha256": _sha256(measures_path)},
            {"path": _relative_path(stories_path, root), "sha256": _sha256(stories_path)},
        ],
        "real_children_without_story_text": real_children_without_story_text,
        "real_group_label_resolution": (
            records.loc[records["source"] == "real"]
            .groupby("group_label_source", dropna=False)
            .size()
            .reset_index(name="n_records")
            .to_dict(orient="records")
        ),
        "text_normalization": {
            "synthetic": "synthetic_source_text_as_exported",
            "real": "enni_chat_plain_text_v1",
            "real_source_csv_preserved": True,
        },
        "records_by_source_group_stage": (
            records.groupby(["source", "group", "stage"], dropna=False)
            .size()
            .reset_index(name="n_records")
            .to_dict(orient="records")
        ),
        "records_by_cohort_source": (
            records.groupby(["cohort_year", "source"], dropna=False)
            .size()
            .reset_index(name="n_records")
            .to_dict(orient="records")
        ),
        "total_records": int(len(records)),
        "synthetic_records": generated_count,
        "real_records": int((records["source"] == "real").sum()),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"output_csv": output_path, "summary_json": summary_path}


def _generation_source(value: str) -> Tuple[int, str]:
    age, separator, path = value.partition("=")
    if not separator or not path.strip():
        raise argparse.ArgumentTypeError("Use AGE=CSV, for example 4=runs/test/4_year_old_rotating_story_generations.csv")
    try:
        cohort_year = int(age)
    except ValueError:
        raise argparse.ArgumentTypeError("AGE must be an integer from 4 to 7")
    if cohort_year not in {4, 5, 6, 7}:
        raise argparse.ArgumentTypeError("AGE must be an integer from 4 to 7")
    return cohort_year, path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export traceable ENNI and Gillam text-classification inputs."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-csv")
    parser.add_argument("--real-measures-csv")
    parser.add_argument("--real-stories-csv")
    parser.add_argument(
        "--generation-source", action="append", type=_generation_source,
        metavar="AGE=CSV", help="Generation CSV for the intended run; repeat for each cohort. Required for the ENNI export.",
    )
    parser.add_argument("--minimum-age-months", type=float, default=49.9)
    parser.add_argument("--maximum-age-months", type=float, default=95)
    parser.add_argument(
        "--gillam-external-only",
        action="store_true",
        help="Write the separate story-level Gillam external-test export only.",
    )
    parser.add_argument("--gillam-output-csv")
    parser.add_argument("--gillam-transcripts-dir")
    parser.add_argument("--gillam-minimum-year", type=int, default=5)
    parser.add_argument("--gillam-maximum-year", type=int, default=7)
    parser.add_argument(
        "--gillam-td-selection-seed", type=int, default=DEFAULT_GILLAM_TD_SELECTION_SEED
    )
    args = parser.parse_args()

    if args.gillam_external_only:
        if not args.gillam_output_csv or not args.gillam_transcripts_dir:
            parser.error("Specify --gillam-output-csv and --gillam-transcripts-dir for the intended reference sample.")
        artifacts = build_gillam_external_text_records(
            project_root=args.project_root,
            output_csv=args.gillam_output_csv,
            transcripts_dir=args.gillam_transcripts_dir,
            minimum_year=args.gillam_minimum_year,
            maximum_year=args.gillam_maximum_year,
            td_selection_seed=args.gillam_td_selection_seed,
        )
    else:
        if not args.generation_source:
            parser.error("Specify --generation-source AGE=CSV for each cohort in the intended generation run.")
        if not all((args.output_csv, args.real_measures_csv, args.real_stories_csv)):
            parser.error("Specify --output-csv, --real-measures-csv and --real-stories-csv for the intended experiment.")
        artifacts = build_all_story_text_records(
            project_root=args.project_root,
            output_csv=args.output_csv,
            generation_sources=args.generation_source,
            real_measures_csv=args.real_measures_csv,
            real_stories_csv=args.real_stories_csv,
            minimum_age_months=args.minimum_age_months,
            maximum_age_months=args.maximum_age_months,
        )
    print("Wrote {}".format(artifacts["output_csv"]))
    print("Wrote {}".format(artifacts["summary_json"]))


if __name__ == "__main__":
    main()
