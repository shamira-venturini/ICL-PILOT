"""Build versioned, verified ENNI inputs from an unchanged official ZIP.

Run with ``python -m icl_pilot.maintenance.enni_corpus ARCHIVE --project-root ROOT``.
Raw files and existing definitive inputs are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
import zipfile

from ..chat.enni_metadata import approximate_age_months, read_chi_metadata
from ..chat.story_unit_semrp import _iter_story_utterances, CHAT_STORY_PARSER_VERSION
from ..chat.text_cleaning import TEXT_CLEANING_POLICY


SOURCE_REL = Path("data/raw/enni/2026-10-28")
INPUT_REL = Path("configs/generation/inputs")
STORIES_NAME = "story_unit_narratives_enni_2026-10-28.csv"
PARTICIPANTS_NAME = "enni_participants_2026-10-28.csv"
AUDIT_REL = Path("docs/reproducibility")
STUDY_CHAT_REL = Path("data/inputs/enni_verified_2026-10-28/chat_4_to_7")
STORY_IDS = ("A1", "A2", "A3", "B1", "B2", "B3")


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_csv(path: Path, rows: list, fields: list | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def read_demographics(raw: bytes) -> dict:
    """Decode the original XLS read-only with the project's R/readxl runtime."""
    executable = shutil.which("Rscript")
    if executable is None:
        raise RuntimeError("The original 0demo.xls cross-check requires Rscript and the R readxl package.")
    code = ("args <- commandArgs(TRUE); "
            "if(!requireNamespace('readxl',quietly=TRUE)) stop('Install the R readxl package'); "
            "x <- as.data.frame(readxl::read_excel(args[1])); "
            "write.csv(x,args[2],row.names=FALSE,na='')")
    with tempfile.TemporaryDirectory(prefix="enni_demographics_") as temporary:
        source, output = Path(temporary)/"0demo.xls", Path(temporary)/"demographics.csv"
        source.write_bytes(raw)
        result = subprocess.run([executable,"--vanilla","-e",code,str(source),str(output)],
                                capture_output=True,text=True)
        if result.returncode:
            raise RuntimeError("Read-only ENNI demographic decoding failed: " + result.stderr)
        with output.open(newline="",encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    if not rows or not {"ID","sex","age","Group","NoAudio"}.issubset(rows[0]):
        raise ValueError("Unexpected official demographic columns.")
    result = {str(int(float(r["ID"]))): r for r in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate participant IDs in the original demographics.")
    return result


def intake(archive: Path, project_root: Path) -> dict:
    archive, root = archive.resolve(), project_root.resolve()
    source_dir, audit_dir = root / SOURCE_REL, root / AUDIT_REL
    stories_path = root / INPUT_REL / STORIES_NAME
    participant_path = root / INPUT_REL / PARTICIPANTS_NAME
    if any(p.exists() for p in (source_dir, audit_dir, stories_path, participant_path,
                                stories_path.with_suffix(".parsing_qc.json"))):
        raise FileExistsError("ENNI intake outputs already exist; never overwrite a verified source or input.")
    raw_zip = archive.read_bytes()
    sources, participants, stories, excluded = [], [], [], []
    demographics, sex_corrections, demographic_sha = {}, [], None
    names_by_id = {}
    with zipfile.ZipFile(archive) as bundle:
        members = [x for x in bundle.infolist() if not x.is_dir()]
        names = [x.filename for x in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP member paths are ambiguous.")
        for member in members:
            name = PurePosixPath(member.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in member.filename:
                raise ValueError("Unsafe ZIP member path: " + member.filename)
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("ZIP symlinks are not supported.")
        if bundle.testzip() is not None:
            raise ValueError("ZIP integrity check failed.")
        if "0demo.xls" in names:
            demographic_bytes = bundle.read("0demo.xls")
            demographics = read_demographics(demographic_bytes)
            demographic_sha = sha256(demographic_bytes)
        # Parse everything before creating any outputs.
        for name in sorted(n for n in names if n.lower().endswith(".cha")):
            raw = bundle.read(name)
            text = raw.decode("utf-8-sig")
            metadata = read_chi_metadata(text)
            child_id = Path(name).stem
            names_by_id.setdefault(child_id, []).append(name)
            pids = re.findall(r"^@PID:\s*(\S+)", text, re.MULTILINE)
            if len(pids) != 1:
                raise ValueError("Expected one source PID: " + name)
            issues, preparation = [], []
            units = _iter_story_utterances(text, diagnostics=issues, preparation=preparation)
            if set(units) != set(STORY_IDS):
                raise ValueError("Incomplete/unexpected story set: " + name)
            reasons = metadata.pop("issues")
            if not child_id.isdigit():
                reasons.append("invalid_filename_id")
            if metadata["Group"] and name.split("/")[0] != metadata["Group"]:
                reasons.append("directory_header_disagreement")
            source = {"transcript_path": name, "source_pid": pids[0], "sha256": sha256(raw),
                      "File_ID": child_id, **metadata, "status": "excluded" if reasons else "included",
                      "exclusion_reasons": reasons, "issues": issues, "text_preparation": preparation}
            sources.append(source)
            if reasons:
                excluded.append({k: source[k] for k in ("transcript_path", "File_ID", "Group", "Age_Raw", "exclusion_reasons")})
                continue
            if any(r["File_ID"] == child_id for r in participants):
                raise ValueError("Multiple fully specified participants share an ID: " + child_id)
            demo = demographics.get(child_id)
            sex = metadata["Sex"]
            if demographics:
                if not demo:
                    raise ValueError("Verified CHI ID absent from original demographics: " + child_id)
                # The spreadsheet codes the SLI set as SLI-A/B; a blank flag is
                # only cross-checked against an already explicit TD CHI header.
                demo_group = "SLI" if demo["Group"] in {"SLI-A","SLI-B"} else "TD" if demo["Group"] == "" else None
                if demo_group != metadata["Group"] or int(float(demo["age"])) != metadata["Age_Month"] // 12:
                    raise ValueError("CHI/demographic clinical group or age-year conflict: " + child_id)
                demo_sex = {"M":"male","F":"female"}.get(demo["sex"])
                if demo_sex is None:
                    raise ValueError("Unknown original demographic sex: " + child_id)
                if sex != demo_sex:
                    sex_corrections.append({"File_ID":child_id,"Sex_CHI":sex,"Sex_demographics":demo_sex,
                                            "source_path":name,"basis":"original 0demo.xls participant table"})
                sex = demo_sex
            path = (SOURCE_REL / "transcripts" / name).as_posix()
            participants.append({"File_ID": child_id, "Group": metadata["Group"], "Age_Raw": metadata["Age_Raw"],
                                 "Age": metadata["Age_Raw"], "Age(Month)": metadata["Age_Month"],
                                 "Age_Month_approx": approximate_age_months(metadata["Age_Raw"]),
                                 "Sex": sex, "Sex_CHI":metadata["Sex"],
                                 "demographic_group_raw": demo["Group"] if demo else "",
                                 "demographic_age_years": demo["age"] if demo else "",
                                 "demographic_no_audio": demo["NoAudio"] if demo else "",
                                 "File_kideval": name,
                                 "transcript_path": path, "source_pid": pids[0], "source_sha256": source["sha256"]})
            for story in STORY_IDS:
                narrative = " ".join(units[story])
                if not narrative:
                    raise ValueError("Empty eligible narrative: " + name + ":" + story)
                stories.append({"File_ID": child_id, "Group": metadata["Group"], "Age_Raw": metadata["Age_Raw"],
                                "Age_Month": metadata["Age_Month"], "transcript_path": path,
                                "story_id": story, "story_unit_count": len(units[story]),
                                "story_units_json": json.dumps(units[story], ensure_ascii=True), "story_text": narrative,
                                "source_pid": pids[0], "source_sha256": source["sha256"]})
        source_dir.mkdir(parents=True)
        (source_dir / "ENNI.zip").write_bytes(raw_zip)
        for member in members:
            target = source_dir / "transcripts" / member.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.read(member))
    write_csv(participant_path, participants)
    write_csv(stories_path, stories)
    study = [r for r in participants if 48 <= r["Age(Month)"] < 96]
    counts = [{"cohort_year": year, "TD": sum(r["Group"] == "TD" and r["Age(Month)"] // 12 == year for r in study),
               "SLI": sum(r["Group"] == "SLI" and r["Age(Month)"] // 12 == year for r in study)} for year in range(4, 8)]
    qc = {"schema_version": 1, "parser_version": CHAT_STORY_PARSER_VERSION, "source_chat_reparsed": True,
          "archive_sha256": sha256(raw_zip), "archive_path": (SOURCE_REL / "ENNI.zip").as_posix(),
          "text_preparation": {"policy": TEXT_CLEANING_POLICY, "performed_at": "chat_to_csv_extraction"},
          "age_month_definition": "12 * completed years + completed months; days do not round cohorts",
          "transcript_files": len(sources), "included_participants": len(participants), "story_rows": len(stories),
          "excluded_sources": excluded, "duplicate_filename_ids": {k: v for k, v in names_by_id.items() if len(v) > 1},
          "group_counts": dict(Counter(r["Group"] for r in participants)), "study_counts": counts,
          "study_children": len(study), "study_story_rows": len(study) * 6,
          "demographics": {"source":"0demo.xls","sha256":demographic_sha,"participants":len(demographics),
                           "included_group_age_year_agreements":len(participants) if demographics else 0,
                           "sex_authority":"original demographic table; original CHI values retained as Sex_CHI",
                           "sex_corrections":sex_corrections,"participant_523":demographics.get("523")},
          "sources": sources, "output_csv_sha256": sha256(stories_path.read_bytes()),
          "participant_csv_sha256": sha256(participant_path.read_bytes())}
    write_json(stories_path.with_suffix(".parsing_qc.json"), qc)
    write_json(audit_dir / "intake.json", {k: v for k, v in qc.items() if k != "sources"})
    write_csv(audit_dir / "sample_counts.csv", counts)
    return qc


def build_roster(project_root: Path) -> dict:
    """A new assignment for all eligible SLI children; preserve the older roster."""
    import pandas as pd
    from ..generation.frozen_roster import _solve_pairing

    root = project_root.resolve()
    destination = root / "configs/generation/master_frozen_roster_enni_2026-10-28.csv"
    if destination.exists():
        raise FileExistsError("Verified roster already exists; choose a new study version.")
    frame = pd.read_csv(root / INPUT_REL / PARTICIPANTS_NAME)
    results = []
    for year in range(4, 8):
        subset = frame.loc[(frame["Age(Month)"] >= year * 12) & (frame["Age(Month)"] < (year + 1) * 12)].copy()
        subset["age_months_approx"] = subset["Age_Month_approx"]
        pairs = _solve_pairing(subset, "{}-year-old".format(year))
        pairs = pairs.drop(columns=["sli_severity_band", "sli_profile_label"])
        pairs["notes"] = "Fresh ENNI CHI metadata; completed-month cohort; days/30 for within-cohort age matching only"
        results.append(pairs)
    roster = pd.concat(results, ignore_index=True)
    roster.to_csv(destination, index=False)
    report = {"roster_path": str(destination.relative_to(root)), "sha256": sha256(destination.read_bytes()),
              "pairs": len(roster), "mean_age_gap_months": float(roster["age_gap_months"].mean()),
              "max_age_gap_months": float(roster["age_gap_months"].max()),
              "cohort_pairs": roster.groupby("cohort").size().to_dict(),
              "generation_batches": sum((len(rows) + 2) // 3 for _, rows in roster.groupby("cohort")),
              "jobs_per_model": len(roster) * 6 * 10, "narratives_all_three_models": len(roster) * 6 * 10 * 2 * 3}
    write_json(root / AUDIT_REL / "roster.json", report)
    return report


def build_study_chat_view(project_root: Path) -> dict:
    """Prepare verified study copies; log demographic header fixes, never raw edits."""
    root = project_root.resolve()
    destination = root / STUDY_CHAT_REL
    if destination.exists():
        raise FileExistsError("Study CHAT input view already exists.")
    with (root / INPUT_REL / PARTICIPANTS_NAME).open(newline="", encoding="utf-8") as handle:
        study = [r for r in csv.DictReader(handle) if 48 <= int(r["Age(Month)"]) < 96]
    payloads, inventory, corrections = [], [], []
    for row in study:
        original = root / row["transcript_path"]
        from ..chat.corpus_archives import original_source_bytes
        raw = original_source_bytes(original, root)
        if sha256(raw) != row["source_sha256"]:
            raise ValueError("Raw source does not match canonical manifest: " + row["File_ID"])
        prepared = raw
        if row["Sex"] != row["Sex_CHI"]:
            text = raw.decode("utf-8-sig")
            pattern = re.compile(r"(^@ID:[ \t]*[^|\n]*\|[^|\n]*\|CHI\|[^|\n]*\|)[^|\n]*(\|)", re.MULTILINE)
            updated, count = pattern.subn(lambda match: match[1] + row["Sex"] + match[2], text)
            if count != 1:
                raise ValueError("Cannot uniquely reconcile the working-copy CHI sex: " + row["File_ID"])
            prepared = updated.encode("utf-8")
            corrections.append({"File_ID":row["File_ID"],"field":"CHI sex","previous":row["Sex_CHI"],
                                "corrected":row["Sex"],"basis":"original 0demo.xls"})
        target = STUDY_CHAT_REL / row["File_kideval"]
        payloads.append((root / target, prepared))
        inventory.append({"File_ID": row["File_ID"], "Group": row["Group"], "Age_Raw": row["Age_Raw"],
                          "source_path": row["transcript_path"], "input_path": target.as_posix(),
                          "source_sha256": row["source_sha256"], "input_sha256":sha256(prepared)})
    for target, raw in payloads:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    report = {"study_chat_dir": STUDY_CHAT_REL.as_posix(), "files": len(inventory), "sources": inventory,
              "header_corrections":corrections,
              "note": "Study working copies; only documented demographic sex-header fixes. Words, boundaries, clinical groups, ages and existing tiers are unchanged. Fresh downstream annotation/features remain pending."}
    write_json(root / AUDIT_REL / "study_chat_inventory.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    qc = intake(args.archive, args.project_root)
    roster = build_roster(args.project_root)
    build_study_chat_view(args.project_root)
    print(json.dumps({"included_participants": qc["included_participants"], "study_counts": qc["study_counts"],
                      "excluded_sources": qc["excluded_sources"], "roster": roster}, indent=2))


if __name__ == "__main__":
    main()
