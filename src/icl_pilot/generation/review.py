"""Read-only generation QC reports; flags never change analysis eligibility."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

from .recovery import atomic_json, timestamp
from .review_html import review_html

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs/generation/review.json"
STAGES = {"stage1_td": "stage1_td_story", "stage2_sli": "stage2_sli_story"}
WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)
COMMON_FIELDS = ("model_key", "model_id", "model_revision", "run_id", "cohort", "cohort_label",
                 "round_id", "bundle_id", "pair_id", "replicate_id", "target_story", "td_child_id", "sli_child_id")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def collapse(text):
    return " ".join(text.split())


def load_review_config(path=DEFAULT_CONFIG):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    if (config.get("schema_version") != 1 or config.get("policy") != "generation-review-only-v1"
            or config.get("review_only") is not True or config.get("exclude_stories") is not False):
        raise ValueError("Generation review must remain review-only, without story exclusions")
    for name, minimum in (("word_run_flag_at", 2), ("phrase_min_repeats", 2), ("phrase_min_words", 4),
                          ("phrase_max_unit_words", 2), ("instruction_min_words", 4)):
        if type(config.get(name)) is not int or config[name] < minimum:
            raise ValueError("Invalid review screening threshold: " + name)
    if config["phrase_max_unit_words"] > 20 or set(config.get("template_files", {})) != set(STAGES):
        raise ValueError("Use bounded phrase screening and both stage templates")
    if not isinstance(config.get("role_markers"), list) or any(not isinstance(marker, str) or not marker for marker in config["role_markers"]):
        raise ValueError("Review role markers must be nonempty literal strings")
    patterns, templates = {}, {}
    for stage, relative in config["template_files"].items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Review templates must stay in their configuration directory")
        source = path.parent / relative
        template = json.loads(source.read_text())
        if template.get("stage") != stage:
            raise ValueError("Review template stage mismatch")
        selected = []
        for message in template["messages"]:
            for line in message["lines"]:
                line = line.strip().lstrip("- ")
                if not line or "{" in line or "}" in line:
                    continue  # Never screen story scaffolds, exemplars or Stage 1 text as instructions.
                words = WORD.findall(line)
                if line.endswith(":") and len(words) >= 2:
                    regex = r"^[ \t]*(?:\#{1,6}[ \t]+)?(?:\*\*)?" + re.escape(line)
                elif len(words) >= config["instruction_min_words"]:
                    regex = r"\s+".join(re.escape(part) for part in line.split())
                else:
                    continue
                selected.append((line, re.compile(regex, re.I | re.M)))
        selected.extend((marker, re.compile(re.escape(marker), re.I)) for marker in config["role_markers"])
        patterns[stage] = selected
        templates[stage] = {"path": str(source), "sha256": digest(source)}
    return config, patterns, templates


def repetition_evidence(text, config):
    matches = list(WORD.finditer(text))
    words = [m.group().casefold() for m in matches]
    longest = {"word": "", "occurrences": 0}
    index = 0
    while index < len(words):
        end = index + 1
        while end < len(words) and words[end] == words[index]:
            end += 1
        if end - index > longest["occurrences"]:
            longest = {"word": words[index], "occurrences": end - index,
                       "start": matches[index].start(), "end": matches[end - 1].end()}
        index = end
    loop = None
    for width in range(2, config["phrase_max_unit_words"] + 1):
        index = 0
        while index + 2 * width <= len(words):
            phrase = words[index:index + width]
            if len(set(phrase)) < 2 or phrase != words[index + width:index + 2 * width]:
                index += 1
                continue
            end = index + 2 * width
            while words[end:end + width] == phrase:
                end += width
            repeats = (end - index) // width
            if repeats >= config["phrase_min_repeats"] and end - index >= config["phrase_min_words"]:
                candidate = {"phrase": " ".join(phrase), "occurrences": repeats, "words_in_loop": end - index,
                             "start": matches[index].start(), "end": matches[end - 1].end()}
                if loop is None or candidate["words_in_loop"] > loop["words_in_loop"]:
                    loop = candidate
            index = end - width
    return len(words), longest, loop


def screen_story(text, stage, parent, stopping, config, patterns):
    flags, evidence = [], {}
    if not text.strip():
        flags.append("empty_output")
    if stopping.get("finish_reason") == "token_ceiling":
        flags.append("generation_token_limit")
        evidence["generation_token_limit"] = stopping
    elif stopping.get("reached_token_ceiling") is True and not stopping.get("finish_reason"):
        flags.append("token_limit_reached_stop_unknown")
    if stage == "stage2_sli" and text.strip() and collapse(text) == collapse(parent):
        flags.append("unchanged_stage2")
    leaks = []
    for marker, pattern in patterns[stage]:
        found = pattern.search(text)
        if found:
            leaks.append({"marker": marker, "start": found.start(), "end": found.end()})
    if leaks:
        flags.append("possible_prompt_leakage")
        evidence["possible_prompt_leakage"] = leaks
    count, longest, loop = repetition_evidence(text, config)
    if longest["occurrences"] >= config["word_run_flag_at"]:
        flags.append("long_word_run")
        evidence["long_word_run"] = longest
    if loop:
        flags.append("repeated_phrase_loop")
        evidence["repeated_phrase_loop"] = loop
    return {"flags": flags, "evidence": evidence, "word_count": count,
            "max_word_run": longest["occurrences"], "max_word_run_word": longest["word"]}


def _journal(path, needed):
    """Read once; retain only selected record metadata, never invent missing stops."""
    prompts, outputs = {}, {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            key = event.get("record_id")
            if key not in needed:
                continue
            kind = event.get("kind")
            if kind == "prompt":
                if key in prompts:
                    raise ValueError("Duplicate selected prompt in generation journal")
                prompts[key] = {name: event.get(name) for name in
                    ("stage", "run_id", "pair_id", "bundle_id", "target_story", "parent_record_id", "parent_text_sha256")}
            elif kind == "output":
                if key in outputs or key not in prompts or text_hash(event["output_text"]) != event["output_text_sha256"]:
                    raise ValueError("Duplicate, orphan or changed selected generation output")
                outputs[key] = {name: event.get(name) for name in
                    ("output_text_sha256", "finish_reason", "generated_token_count", "terminal_token_id", "reached_token_ceiling")}
    return prompts, outputs


def _csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if not fields or len(fields) != len(set(fields)) or any(not field for field in fields) or set(STAGES.values()) - set(fields):
            raise ValueError("Review inputs require distinct named columns and both stage text fields")
        rows = list(reader)
    if not rows or any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Empty or malformed generation CSV")
    return rows


def write_generation_review(csvs, output_dir, *, journals=None, config_path=DEFAULT_CONFIG):
    """Export all stories and a flagged subset for inspection, without filtering."""
    config, patterns, templates = load_review_config(config_path)
    sources = {label: Path(value).expanduser().resolve() for label, value in csvs.items()}
    journals = journals or {}
    if not sources or set(journals) - set(sources) or len(set(sources.values())) != len(sources):
        raise ValueError("Select unique CSV sources and matching optional journal labels")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", label) for label in sources):
        raise ValueError("Use simple unique review collection labels")
    inputs, rows_by_label, needed_by_journal = {}, {}, defaultdict(set)
    for label, path in sources.items():
        source_hash = digest(path)
        rows = _csv(path)
        if digest(path) != source_hash:
            raise ValueError("Generation CSV changed while it was being read")
        selected = Path(journals[label]).expanduser().resolve() if label in journals else path.parent / "prompt_records.jsonl"
        if label in journals and not selected.is_file():
            raise FileNotFoundError(selected)
        journal = selected if selected.is_file() else None
        inputs[label] = {"path": str(path), "sha256": source_hash, "rows": len(rows),
                         "journal": str(journal) if journal else None, "journal_sha256": digest(journal) if journal else None}
        rows_by_label[label] = rows
        if journal:
            for row in rows:
                needed_by_journal[journal].update(row.get(stage + "_prompt_record_id", "") for stage in ("stage1", "stage2"))
    indexed = {path: _journal(path, keys - {""}) for path, keys in needed_by_journal.items()}
    records, seen, used_records = [], set(), set()
    for label, rows in rows_by_label.items():
        source = inputs[label]
        for index, row in enumerate(rows, 1):
            if row.get("run_id") and row.get("model_key") and row.get("bundle_id") and row.get("target_story"):
                key = tuple(row[name] for name in ("run_id", "model_key", "bundle_id", "target_story"))
                if key in seen:
                    raise ValueError("Overlapping generation jobs selected; use aggregate or constituent CSVs")
                seen.add(key)
            ids = {stage: text_hash(json.dumps([source["sha256"], label, index, stage]))[:24] for stage in STAGES}
            for stage, field in STAGES.items():
                text = row[field]
                record_id = row.get("stage1_prompt_record_id" if stage == "stage1_td" else "stage2_prompt_record_id", "")
                stop, metadata_status = {}, "journal unavailable"
                if source["journal"] and record_id:
                    record_key = (source["journal"], record_id)
                    if record_key in used_records:
                        raise ValueError("Overlapping selected generation prompt records")
                    used_records.add(record_key)
                    prompts, outputs = indexed[Path(source["journal"])]
                    if record_id not in outputs or record_id not in prompts:
                        raise ValueError("CSV references an unfinished or missing generation journal record")
                    prompt, stop = prompts[record_id], outputs[record_id]
                    if prompt["stage"] != stage or stop["output_text_sha256"] != text_hash(text):
                        raise ValueError("Generation CSV text/stage disagrees with its recorded output")
                    for name in ("run_id", "pair_id", "bundle_id", "target_story"):
                        if row.get(name) and row[name] != prompt[name]:
                            raise ValueError("Generation CSV identity disagrees with journal: " + name)
                    if stage == "stage2_sli" and (prompt["parent_record_id"] != row.get("stage1_prompt_record_id")
                            or prompt["parent_text_sha256"] != text_hash(row["stage1_td_story"])):
                        raise ValueError("Stage 2 review requires its exact recorded Stage 1 parent")
                    metadata_status = "recorded" if stop.get("finish_reason") else "stopping reason not recorded"
                elif source["journal"]:
                    metadata_status = "CSV has no prompt record ID"
                metadata = {key: value for key, value in row.items() if key not in STAGES.values()}
                record = {"review_id": ids[stage], "paired_review_id": ids["stage2_sli" if stage == "stage1_td" else "stage1_td"],
                          "input_label": label, "source_csv": source["path"], "source_row": index,
                          **{name: row.get(name, "") for name in COMMON_FIELDS}, "stage": stage,
                          "group": "TD" if stage == "stage1_td" else "SLI", "story_text": text,
                          "story_text_sha256": text_hash(text), "prompt_record_id": record_id,
                          "stopping_metadata": metadata_status, **{name: stop.get(name) for name in
                          ("finish_reason", "generated_token_count", "terminal_token_id", "reached_token_ceiling")},
                          "metadata": metadata, "review_status": "pending", "review_notes": ""}
                record.update(screen_story(text, stage, row["stage1_td_story"], stop, config, patterns))
                records.append(record)
    for label, source in inputs.items():
        if digest(source["path"]) != source["sha256"] or (source["journal"] and digest(source["journal"]) != source["journal_sha256"]):
            raise ValueError("Generation inputs changed while the review was being prepared")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "review_manifest.json"
    manifest = {"schema_version": 1, "policy": config["policy"], "status": "writing_reports",
                "created_at_utc": timestamp(), "review_only": True, "stories_excluded": 0,
                "analysis_inputs_modified": False, "config": config, "inputs": inputs, "templates": templates,
                "config_sha256": digest(config_path), "wrapper_sha256": digest(__file__),
                "html_renderer_sha256": digest(Path(__file__).with_name("review_html.py")), "outputs": {}}
    atomic_json(manifest_path, manifest)
    try:
        summaries = {}
        for record in records:
            key = tuple(record[name] for name in ("input_label", "model_key", "run_id", "cohort", "cohort_label", "stage"))
            summary = summaries.setdefault(key, {**dict(zip(("input_label", "model_key", "run_id", "cohort", "cohort_label", "stage"), key)),
                "stories": 0, "flagged_stories": 0, "recorded_stopping_reasons": 0, "flag_counts": Counter()})
            summary["stories"] += 1
            summary["flagged_stories"] += bool(record["flags"])
            summary["recorded_stopping_reasons"] += record["stopping_metadata"] == "recorded"
            summary["flag_counts"].update(record["flags"])
        summary = list(summaries.values())
        manifest["summary"] = {"jobs": sum(source["rows"] for source in inputs.values()), "stories": len(records),
            "flagged_stories": sum(bool(r["flags"]) for r in records),
            "stories_with_recorded_stopping_reasons": sum(r["stopping_metadata"] == "recorded" for r in records),
            "flag_counts": dict(Counter(flag for record in records for flag in record["flags"]))}
        csv_records = [{**{name: value for name, value in record.items() if name not in ("flags", "evidence", "metadata")},
                        "flags": "|".join(record["flags"]), "evidence_json": json.dumps(record["evidence"], ensure_ascii=False),
                        "metadata_json": json.dumps(record["metadata"], ensure_ascii=False)} for record in records]
        for name, selected in (("all_stories.csv", csv_records), ("flagged_stories.csv", [r for r in csv_records if r["flags"]])):
            with (output / name).open("x", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(csv_records[0]))
                writer.writeheader(); writer.writerows(selected)
        atomic_json(output / "review_data.json", {"stories": records, "summary": summary})
        atomic_json(output / "flag_summary.json", {"overall": manifest["summary"], "by_source_stage_cohort": summary})
        (output / "review.html").write_text(review_html(records, manifest["summary"]), encoding="utf-8")
        manifest["outputs"] = {name: digest(output / name) for name in
            ("all_stories.csv", "flagged_stories.csv", "review_data.json", "flag_summary.json", "review.html")}
        # Check again after report writing so a concurrent source change never produces a pass.
        for source in inputs.values():
            if digest(source["path"]) != source["sha256"] or (source["journal"] and digest(source["journal"]) != source["journal_sha256"]):
                raise ValueError("Generation inputs changed during report writing")
        manifest["status"] = "review_report_complete"
    except BaseException as error:
        manifest.update(status="failed_or_interrupted", error=str(error))
        raise
    finally:
        atomic_json(manifest_path, manifest)
    return manifest_path
