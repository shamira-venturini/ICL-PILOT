from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

from .paths import FROZEN_ROSTER_CSV


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_MULTISPACE_RE = re.compile(r"\s+")
_TRAILING_CLAUSE_PUNCT_RE = re.compile(r"[,:;.!?]+$")
_HAS_LEXICAL_CONTENT_RE = re.compile(r"[A-Za-z0-9]")
_LEADING_BOUNDARY_PUNCT_RE = re.compile(r"^[,.;:!?]+\s*")
_CHAT_PUNCT_RE = re.compile(r"\s*([,.;!?])")
# Protect notation before handling prose punctuation. In particular, a pause
# dot is not an utterance boundary and a colon inside a word marks lengthening.
_CHAT_NOTATION_RE = re.compile(
    r'\+(?:"/\.|"\.|"|//[.?]|/[.?]|\.{3}|\.)'
    r'|\([^()]*\)|\[[^\[\]]*\]|&\*[A-Z][A-Z0-9]*:[^\s.!?,;]+'
)
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_AGE_RE = re.compile(r"(\d+);(\d{1,2})(?:\.(\d{1,2}))?\Z")
_NUMBER_WORDS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")


def _number_words(value: str) -> str:
    if "." in value:
        whole, decimal = value.split(".", 1)
        return _number_words(whole) + " point " + " ".join(_NUMBER_WORDS[int(digit)] for digit in decimal)
    if len(value) > 1 and value.startswith("0"):
        return " ".join(_NUMBER_WORDS[int(digit)] for digit in value)
    number = int(value)
    if number < 20:
        return _NUMBER_WORDS[number]
    if number < 100:
        return _TENS[number // 10] + (" " + _NUMBER_WORDS[number % 10] if number % 10 else "")
    for scale, name in ((10**9, "billion"), (10**6, "million"), (1000, "thousand"), (100, "hundred")):
        if number >= scale:
            head, tail = divmod(number, scale)
            return _number_words(str(head)) + " " + name + (" " + _number_words(str(tail)) if tail else "")
    raise ValueError(f"Unsupported number: {value}")


def _normalize_generated_notation(text: str) -> tuple[str, list[dict[str, str]]]:
    """Normalize explicit written notation, retaining an exact replacement log.

    This does not correct grammar, choose missing words or infer error tags.
    Number reading is a documented orthographic convention for generated text.
    """
    changes: list[dict[str, str]] = []

    def substitute(pattern, replacement, kind):
        nonlocal text

        def apply(match):
            after = replacement(match)
            if after != match.group():
                changes.append({"kind": kind, "before": match.group(), "after": after})
            return after

        text = re.sub(pattern, apply, text)

    substitute(r"\(\s*(\.{1,3}|\d+(?:\.\d+)?)\s*\)", lambda m: "(" + m[1] + ")", "pause_spacing")
    # An isolated omitted suffix belongs to its preceding word, not a new word.
    substitute(r"\b([A-Za-z][A-Za-z'’]*)\s+\((s|ed|ing)\)",
               lambda m: m[1] + "(" + m[2] + ")", "omitted_suffix_spacing")
    # Parentheses enclosing entire words are prose punctuation, not CHAT word
    # shortenings. Keep every word; do not reinterpret them as omitted speech.
    def prose_parentheses(match):
        content = match[1]
        return content if re.search(r"[A-Za-z]", content) else match.group()

    substitute(r"(?<![\w])\(([^()]*)\)(?![\w])", prose_parentheses, "prose_parentheses")
    substitute(r"&\*\s*([A-Z][A-Z0-9]*)\s*:\s*([^\s.!?,;]+)",
               lambda m: "&*" + m[1] + ":" + m[2], "interposed_speaker_spacing")
    substitute(r"(?<![\w])'((?:[^']|'(?=\w))*[.!?])'(?=\s|$)",
               lambda m: '"' + m[1] + '"', "single_quoted_utterance")
    text, protected = _protect_notation(text)
    substitute(r"\$([0-9]+(?:\.[0-9]+)?)",
               lambda m: _number_words(m[1]) + (" dollar" if float(m[1]) == 1 else " dollars"), "currency_spelling")
    substitute(r"\b([AB])([123])\b", lambda m: m[1] + "@l " + _number_words(m[2]), "story_identifier_spelling")
    substitute(r"(?<![\w])\d+(?:\.\d+)?(?![\w])", lambda m: _number_words(m.group()), "number_spelling")
    substitute(r"&(?![=*~+\-])", lambda m: " and ", "ampersand_spelling")
    events = r"(?:coughs|sneezes|laughs|giggles|whimpers|sighs|yawns|cries|growls)"
    substitute(r"\*(" + events + r")\*", lambda m: "&=" + m[1], "nonverbal_event_notation")
    substitute(r"(?<![&\w])=(" + events + r")\b", lambda m: "&=" + m[1], "nonverbal_event_notation")
    # Uncoded stand-alone prose symbols have no spoken word content. Their
    # exact removal is logged; do not guess a pause, retracing or error code.
    substitute(r"(?<!\S)[+/@](?=\s|[.!?,;:]|$)", lambda m: "", "uncoded_prose_symbol")
    substitute(r"([!?])[.!?]+", lambda m: m[1], "utterance_terminator_run")
    substitute(r"(?<![+.])\.{2,}", lambda m: " +...", "trailing_off_notation")
    return _restore_notation(text, protected), changes


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


def _protect_notation(text: str, *, keep_terminators: bool = False) -> tuple[str, dict[str, str]]:
    protected: dict[str, str] = {}

    def replace(match):
        value = match.group()
        terminal = ""
        if keep_terminators and value.startswith("+") and value[-1:] in ".!?":
            value, terminal = value[:-1], value[-1]
        key = "\ue000" + chr(0xE100 + len(protected)) + "\ue001"
        protected[key] = value
        return key + terminal

    return _CHAT_NOTATION_RE.sub(replace, text), protected


def _restore_notation(text: str, protected: dict[str, str]) -> str:
    for key, value in protected.items():
        text = text.replace(key, value)
    return text


def _sentence_parts(text: str, *, maxsplit: int = 0) -> list[str]:
    guarded, protected = _protect_notation(text, keep_terminators=True)
    return [_restore_notation(part, protected) for part in _SENTENCE_SPLIT_RE.split(guarded, maxsplit=maxsplit)]


def _chat_space_punctuation(text: str) -> str:
    normalized = _normalize_surface(text)
    if not normalized:
        return ""
    guarded, protected = _protect_notation(normalized)
    spaced = _CHAT_PUNCT_RE.sub(r" \1", guarded)
    return _restore_notation(_MULTISPACE_RE.sub(" ", spaced).strip(), protected)


def _sentence_lines(text: str) -> list[str]:
    normalized = _normalize_surface(text)
    if not normalized or not _has_lexical_content(normalized):
        return []
    parts = [part.strip() for part in _sentence_parts(normalized) if part.strip()]
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
    parts = [part.strip() for part in _sentence_parts(normalized) if part.strip()]
    complete: list[str] = []
    tail = ""
    for part in parts:
        if part[-1] in ".!?":
            complete.append(part)
        else:
            tail = part
    return complete, tail


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
    clause = _normalize_surface(_strip_clause_edges(text).replace('"', ""))
    if not _has_lexical_content(clause):
        return ""
    return f'{clause} +".' if clause else '+".'


def _repair_quote_layout(lines: list[str]) -> list[str]:
    """Pair explicit quotation linkers; use inline quotes when no frame exists.

    Do not invent a reporting clause or speaker for standalone quoted text.
    CHAT supports Unicode quotation marks without cross-utterance linkers.
    """
    lines = list(lines)
    for index, line in enumerate(lines):
        if line.endswith('+"/.'):
            if index + 1 < len(lines) and not lines[index + 1].startswith('+" '):
                if not lines[index + 1].endswith(('+"/.', '+".')):
                    lines[index + 1] = '+" ' + lines[index + 1]
                else:
                    lines[index] = line[:-4].rstrip() + " ."
            elif index + 1 == len(lines):
                lines[index] = line[:-4].rstrip() + " ."

    index = 0
    while index < len(lines):
        if not lines[index].startswith('+" '):
            index += 1
            continue
        end = index + 1
        while end < len(lines) and lines[end].startswith('+" '):
            end += 1
        follows = index > 0 and lines[index - 1].endswith('+"/.')
        precedes = end < len(lines) and lines[end].endswith('+".')
        if not follows and not precedes:
            for position in range(index, end):
                body = lines[position][3:].strip()
                match = re.search(r"(\+(?:\.{3}|//[.?]|/[.?]|\.)|[.!?])$", body)
                terminal = match.group() if match else "."
                content = body[:match.start()].rstrip() if match else body
                lines[position] = "“" + content + "” " + terminal
        elif follows and precedes:
            lines[end] = lines[end][:-3].rstrip() + " ."
        index = end

    for index, line in enumerate(lines):
        if line.endswith('+".') and (index == 0 or not lines[index - 1].startswith('+" ')):
            lines[index] = line[:-3].rstrip() + " ."
    return lines


def _chat_utterance_lines(text: str, *, formatting_changes: list[dict[str, str]] | None = None) -> list[str]:
    normalized = _normalize_surface(text)
    if not normalized:
        return []
    if '"' not in normalized:
        return _sentence_lines(normalized)

    # Recognize complete CHAT controls before ordinary quotes, including mixed
    # model outputs that open with +" but close with a plain double quote.
    # Each explicit initiator starts a new main tier; a quoted word or pause
    # must not be mistaken for an additional participant.
    lines: list[str] = []
    quoted = False
    buffer = ""

    def append_plain(value):
        value = _strip_leading_boundary_punct(value)
        if value in {"+.", "+...", "+/.", "+//."} and lines:
            lines[-1] = re.sub(r"[.!?]$", "", lines[-1]).rstrip() + " " + value
        else:
            lines.extend(_sentence_lines(value))

    def append_quoted(value):
        for part in _sentence_parts(value):
            formatted = _format_quote_line(_strip_leading_boundary_punct(part))
            if formatted:
                lines.append(formatted)

    def append_prefix(value):
        completed, tail = _sentence_lines_with_tail(value)
        for part in completed:
            append_plain(part)
        if tail:
            formatted = _format_quote_prefix_line(tail)
            if formatted:
                lines.append(formatted)

    for part in re.split(r'(\+"/\.|\+"\.|\+"|")', normalized):
        if part == '+"/.':
            append_prefix(buffer)
            buffer, quoted = "", False
        elif part == '+"':
            (append_quoted if quoted else append_plain)(buffer)
            buffer, quoted = "", True
        elif part == '+".':
            if quoted:
                append_quoted(buffer)
            else:
                parts = [p for p in _sentence_parts(buffer) if _has_lexical_content(p)]
                for value in parts[:-1]:
                    append_plain(value)
                if parts:
                    formatted = _format_quote_suffix_line(parts[-1])
                    if formatted:
                        lines.append(formatted)
            buffer, quoted = "", False
        elif part == '"':
            (append_quoted if quoted else append_prefix)(buffer)
            buffer, quoted = "", not quoted
        else:
            buffer += part
    (append_quoted if quoted else append_plain)(buffer)
    lines = [line for line in lines if line]
    repaired = _repair_quote_layout(lines)
    if repaired != lines and formatting_changes is not None:
        formatting_changes.append({"kind": "quotation_link_layout", "before": "\n".join(lines), "after": "\n".join(repaired)})
    return repaired


def _build_chat_text(
    *,
    corpus_name: str,
    age: str,
    group: str,
    story_id: str,
    story_text: str,
    comments: list[str],
    formatting_changes: list[dict[str, str]] | None = None,
) -> str:
    utterances = _chat_utterance_lines(story_text, formatting_changes=formatting_changes)
    if not utterances:
        raise ValueError("Cannot write CHAT file for empty story text")

    lines = [
        "@UTF8",
        "@Begin",
        "@Languages:\teng",
        "@Participants:\tEXA Investigator, CHI Target_Child",
        f"@ID:\teng|{corpus_name}|EXA|||||Investigator|||",
        f"@ID:\teng|{corpus_name}|CHI|{age}||{group}||Target_Child|||",
        f"@Types:\tcross, narrative, {group}",
        "@Comment:\tdata_source=generated",
        f"@G:\t{story_id}",
    ]
    lines.extend(f"@Comment:\t{comment}" for comment in comments)
    if formatting_changes:
        lines.append(f"@Comment:\tformatting_normalization_count={len(formatting_changes)}")
    lines.extend(f"*CHI:\t{utterance}" for utterance in utterances)
    lines.append("@End")
    return "\n".join(lines) + "\n"


def _correct_generated_types(text: str) -> str:
    """Correct the known generated-only header defect; leave transcript tiers intact."""
    lines = text.splitlines(keepends=True)
    indices = [i for i, line in enumerate(lines) if line.startswith("@Types:")]
    if len(indices) != 1:
        raise ValueError("Expected exactly one @Types header")
    index = indices[0]
    fields = [field.strip() for field in lines[index].split(":", 1)[1].strip().split(",")]
    generated_marker = any(re.fullmatch(r"@Comment:\s+data_source=generated", line.rstrip("\r\n"))
                           for line in lines)
    legacy = len(fields) == 4 and fields[:3] == ["cross", "narrative", "generated"]
    current = len(fields) == 3 and fields[:2] == ["cross", "narrative"] and generated_marker
    if not (legacy or current) or fields[-1] not in {"TD", "SLI"}:
        raise ValueError("Expected a supported generated TD/SLI header; real corpus files are outside this repair")
    group = fields[-1]
    child_ids = [line.split(":", 1)[1].strip().split("|") for line in lines if line.startswith("@ID:")]
    child_ids = [fields for fields in child_ids if len(fields) > 5 and fields[2] == "CHI"]
    if len(child_ids) != 1 or child_ids[0][5] != group:
        raise ValueError("@Types group must match the existing CHI @ID group")
    if current:
        return text
    ending = "\r\n" if lines[index].endswith("\r\n") else "\n"
    lines[index] = f"@Types:\tcross, narrative, {group}{ending}"
    if not generated_marker:
        lines.insert(index + 1, "@Comment:\tdata_source=generated" + ending)
    return "".join(lines)


def repair_generated_chat_types(input_dir: str | Path, output_dir: str | Path) -> Path:
    """Copy generated CHAT into a fresh directory, fixing only the known @Types defect.

    Original files and their existing manifests are never rewritten. The returned
    manifest inventories source/output hashes; this repair is not full CHAT validation.
    """
    source, destination = Path(input_dir).resolve(), Path(output_dir).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Repair inputs and outputs must be separate directory trees")
    if destination.exists():
        raise FileExistsError(f"Choose a fresh formatting output directory: {destination}")
    files = sorted(source.rglob("*.cha"))
    if not files:
        raise ValueError(f"No CHAT files found: {source}")
    planned, inventory = [], []
    for path in files:
        before = path.read_bytes()
        try:
            after = _correct_generated_types(before.decode("utf-8")).encode("utf-8")
        except ValueError as error:
            raise ValueError(f"Cannot repair {path}: {error}") from error
        relative = path.relative_to(source)
        planned.append((relative, after))
        inventory.append({"relative_path": relative.as_posix(),
                          "source_sha256": hashlib.sha256(before).hexdigest(),
                          "output_sha256": hashlib.sha256(after).hexdigest(),
                          "changed": before != after})
    # Validate the entire collection before creating any output.
    destination.mkdir(parents=True, exist_ok=False)
    for relative, content in planned:
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as handle:
            handle.write(content)
    manifest = {"schema_version": 1, "operation": "generated_chat_types_header_v1",
                "status": "formatting_complete", "source_directory": str(source),
                "output_directory": str(destination),
                "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "files_checked": len(inventory), "files_changed": sum(item["changed"] for item in inventory),
                "chat_syntax_validation": "not_run", "files": inventory}
    manifest_path = destination / "formatting_manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


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
            story_text, formatting_changes = _normalize_generated_notation(story_text)

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
                formatting_changes=formatting_changes,
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
                    "source_text_sha256": hashlib.sha256(spec["text"].encode("utf-8")).hexdigest(),
                    "formatting_changes": json.dumps(formatting_changes, ensure_ascii=False, separators=(",", ":")),
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
