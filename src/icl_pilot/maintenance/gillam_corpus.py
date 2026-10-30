"""Validate the original Gillam ZIP and prepare its external-only story export."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

from .enni_corpus import sha256, write_csv, write_json
from ..evaluation.story_text_exports import (
    _gillam_child_metadata, _select_gillam_sessions, build_gillam_external_text_records,
)

RAW = Path("data/raw/gillam/2026-10-28")
OUTPUT = Path("data/inputs/gillam/external_story_records.csv")
AUDIT = Path("docs/reproducibility/gillam.json")


def intake(archive: Path, project_root: Path) -> dict:
    root = project_root.resolve()
    archive = archive.resolve()
    raw_root = root / RAW
    output = root / OUTPUT
    if raw_root.exists() or output.exists() or (root/AUDIT).exists():
        raise FileExistsError("Gillam intake outputs already exist; preserve the verified source and export.")
    sources, anomalies, seen_ids, seen_pids, all_metadata = [], [], set(), set(), []
    with zipfile.ZipFile(archive) as bundle:
        members = [m for m in bundle.infolist() if not m.is_dir()]
        if len({m.filename for m in members}) != len(members):
            raise ValueError("Duplicate Gillam ZIP paths.")
        for member in members:
            name = PurePosixPath(member.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in member.filename:
                raise ValueError("Unsafe ZIP member: " + member.filename)
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("ZIP symlinks are not supported.")
        if bundle.testzip() is not None:
            raise ValueError("Gillam ZIP integrity check failed.")
        for member in sorted(members,key=lambda m:m.filename):
            if not member.filename.endswith(".cha"):
                continue
            name = member.filename
            raw = bundle.read(member)
            text = raw.decode("utf-8-sig")
            metadata = _gillam_child_metadata(text,raw_root/"transcripts"/name,raw_root/"transcripts")
            pids = re.findall(r"^@PID:\s*(\S+)",text,re.MULTILINE)
            child = metadata["source_child_id"]
            session = metadata["source_session_id"]
            if session in seen_ids or len(pids) != 1 or pids[0] in seen_pids:
                raise ValueError("Duplicate or ambiguous Gillam source identity: " + name)
            seen_ids.add(session);seen_pids.add(pids[0]);all_metadata.append(metadata)
            folder = re.fullmatch(r"(\d+)([fm])",Path(name).parent.name)
            issues = []
            if metadata["age_months"] is None:
                issues.append("unknown_completed_months")
            if folder and int(folder[1]) != metadata["cohort_year"]:
                issues.append("folder_age_disagrees_with_CHI_header")
            if folder and folder[2] != {"female":"f","male":"m"}[metadata["sex"]]:
                issues.append("folder_sex_disagrees_with_CHI_header")
            if issues:
                anomalies.append({"source_path":name,"age_raw":metadata["age_raw"],
                                  "sex":metadata["sex"],"issues":issues})
            sources.append({"source_path":name,"source_child_id":child,"source_session_id":session,"source_pid":pids[0],
                            "group":metadata["group"],"age_raw":metadata["age_raw"],
                            "age_months":metadata["age_months"],"sex":metadata["sex"],
                            "source_sha256":sha256(raw)})
        unique_children, repeated_sessions = _select_gillam_sessions(all_metadata)
        # Preserve every member unchanged, including out-of-study-age records.
        raw_root.mkdir(parents=True)
        (raw_root/"Gillam.zip").write_bytes(archive.read_bytes())
        for member in members:
            path = raw_root/"transcripts"/member.filename
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(bundle.read(member))
    artifacts = build_gillam_external_text_records(project_root=root,transcripts_dir=RAW/"transcripts",output_csv=OUTPUT)
    summary = json.loads(artifacts["summary_json"].read_text())
    with output.open(newline="",encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    counts = {}
    for row in records:
        key = (row["group"],row["source_child_id"])
        counts.setdefault(key,[]).append(row["story_id"])
    incomplete = [{"group":group,"source_child_id":child,"available_stories":stories}
                  for (group,child),stories in counts.items() if len(stories)<3]
    report = {"archive_path":(RAW/"Gillam.zip").as_posix(),"archive_sha256":sha256(archive.read_bytes()),
              "raw_members":len(members),"transcripts":len(sources),
              "raw_group_counts":{group:sum(r["group"]==group for r in sources) for group in ("TD","SLI")},
              "duplicate_session_ids":0,"duplicate_pids":0,
              "raw_unique_children":len(unique_children),"repeated_child_sessions":repeated_sessions,
              "group_authority":"explicit @Types, verified against corpus directory",
              "age_sex_authority":"literal CHI headers, independent of age/sex folders",
              "source_metadata_anomalies":anomalies,
              "output_csv":OUTPUT.as_posix(),"output_csv_sha256":sha256(output.read_bytes()),
              "selected_children":len(counts),"selected_children_by_group":summary["unique_children_by_group"],
              "selected_child_age_range_months":[min(int(r["age_months"]) for r in records),max(int(r["age_months"]) for r in records)],
              "selection":summary["selection"],"selected_age_sex_group_counts":summary["unique_children_by_age_sex_group"],
              "story_rows":len(records),"stories_by_group":summary["records_by_group"],
              "incomplete_selected_children":incomplete,
              "intended_use":"external classifier evaluation only; never model selection, training, validation or ENNI descriptive reference"}
    write_csv(root/"docs/reproducibility/gillam_source_inventory.csv",sources)
    write_json(root/AUDIT,report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("--project-root",type=Path,required=True)
    args = parser.parse_args()
    result = intake(args.archive,args.project_root)
    print(json.dumps({k:result[k] for k in ("transcripts","selected_children","selected_children_by_group",
                                          "selected_child_age_range_months","story_rows","stories_by_group",
                                          "incomplete_selected_children")},indent=2))


if __name__ == "__main__":
    main()
