"""Historical repair procedures for the pre-cleanup generated-data layout.

Default paths and confirmed cases document that earlier layout. Reusable repair
functions remain available for explicit inputs and regression checks; the current
notebook uses icl_pilot.chat.annotation for new TalkTag runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path

from ..paths import ROOT


ERROR_RE = re.compile(r"\[\*\s+([^\]]+)\]")
BRACKET_RE = re.compile(r"\[[^\]]*\]")
TOKEN_RE = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", re.UNICODE)
CORRECTION_AFTER_TOKEN_RE = re.compile(r"(?:\s+\[:[^\]]*\])+")

SMALL_NUMBER_WORDS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
}
TENS_WORDS = {
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}


@dataclass(frozen=True)
class PairSpec:
    label: str
    source_root: Path
    target_root: Path


@dataclass(frozen=True)
class SurfaceToken:
    value: str
    start: int
    end: int


@dataclass(frozen=True)
class InlineTag:
    code: str
    anchor: int | None


@dataclass(frozen=True)
class RepairRecord:
    file: str
    line: int
    status: str
    current: str
    source: str
    reason: str


@dataclass(frozen=True)
class SpeakerBlock:
    line_index: int
    lines: list[str]


@dataclass(frozen=True)
class ConfirmedMergeRepair:
    label: str
    source_file: str
    source_utterance_index: int
    target_files: tuple[str, ...]


CONFIRMED_MERGE_REPAIRS = (
    ConfirmedMergeRepair(
        label="batchalign_split_13_way",
        source_file=(
            "data/generated/4_year_old/cha/stage2_sli/"
            "pair_sli427_td406_rep08_B1_SLI.cha"
        ),
        source_utterance_index=0,
        target_files=(
            "data/generated/4_year_old/annotated_batchalign/stage2_sli_annotated/"
            "pair_sli427_td406_rep08_B1_SLI.cha",
            "data/generated/4_year_old/reannotated_talktag/stage2_sli_reannotated/"
            "pair_sli427_td406_rep08_B1_SLI.cha",
            "data/generated/4_year_old/reannotated_talktag/"
            "bundled_synthetic_children/stage2_sli/"
            "pair_sli427_td406_rep08_SLI_bundle.cha",
        ),
    ),
    ConfirmedMergeRepair(
        label="batchalign_split_parenthetical",
        source_file=(
            "data/generated/4_year_old/cha/stage2_sli/"
            "pair_sli477_td415_rep05_A2_SLI.cha"
        ),
        source_utterance_index=3,
        target_files=(
            "data/generated/4_year_old/annotated_batchalign/stage2_sli_annotated/"
            "pair_sli477_td415_rep05_A2_SLI.cha",
            "data/generated/4_year_old/reannotated_talktag/stage2_sli_reannotated/"
            "pair_sli477_td415_rep05_A2_SLI.cha",
            "data/generated/4_year_old/reannotated_talktag/"
            "bundled_synthetic_children/stage2_sli/"
            "pair_sli477_td415_rep05_SLI_bundle.cha",
        ),
    ),
    ConfirmedMergeRepair(
        label="batchalign_split_coordinated_clause",
        source_file=(
            "data/generated/4_year_old/cha/stage2_sli/"
            "pair_sli480_td450_rep09_A2_SLI.cha"
        ),
        source_utterance_index=1,
        target_files=(
            "data/generated/4_year_old/annotated_batchalign/stage2_sli_annotated/"
            "pair_sli480_td450_rep09_A2_SLI.cha",
            "data/generated/4_year_old/reannotated_talktag/stage2_sli_reannotated/"
            "pair_sli480_td450_rep09_A2_SLI.cha",
            "data/generated/4_year_old/reannotated_talktag/"
            "bundled_synthetic_children/stage2_sli/"
            "pair_sli480_td450_rep09_SLI_bundle.cha",
        ),
    ),
)

CONFIRMED_INSERTED_LINE = (
    "*CHI:\tthere is a cat and dog they playing with a [* m:allo] "
    "airplane near a pool ."
)
CONFIRMED_INSERTED_LINE_TARGETS = (
    "data/generated/4_year_old/annotated_batchalign/stage2_sli_annotated/"
    "pair_sli480_td450_rep07_A1_SLI.cha",
    "data/generated/4_year_old/reannotated_talktag/stage2_sli_reannotated/"
    "pair_sli480_td450_rep07_A1_SLI.cha",
    "data/generated/4_year_old/reannotated_talktag/"
    "bundled_synthetic_children/stage2_sli/"
    "pair_sli480_td450_rep07_SLI_bundle.cha",
)

CONFIRMED_TRUNCATION_SOURCE = (
    "data/generated/5_year_old/5_year_old_cha/stage2_sli/"
    "pair_sli529_td545_rep06_A2_SLI.cha"
)
CONFIRMED_TRUNCATION_SOURCE_UTTERANCE = 4
CONFIRMED_TRUNCATION_TARGETS = (
    "data/generated/5_year_old/talktag-annotated/"
    "5_year_old_cha_stage2_sli_annotated/"
    "pair_sli529_td545_rep06_A2_SLI.cha",
    "data/generated/5_year_old/fully-annotated/"
    "5_year_old_cha_stage2_sli_reannotated/"
    "pair_sli529_td545_rep06_A2_SLI.cha",
    "data/generated/5_year_old/fully-annotated/"
    "bundled_synthetic_children/stage2_sli/"
    "pair_sli529_td545_rep06_SLI_bundle.cha",
)

CONFIRMED_EXPOSED_RESTORATION_SOURCE = (
    "data/generated/4_year_old/cha/stage2_sli/"
    "pair_sli480_td450_rep07_A1_SLI.cha"
)
CONFIRMED_EXPOSED_RESTORATION_SOURCE_UTTERANCE = 9
CONFIRMED_EXPOSED_RESTORATION_CURRENT = (
    "*CHI:\tnow both monkey [* m:0s:a] happy and play some more ."
)
CONFIRMED_EXPOSED_RESTORATION_TARGETS = (
    "data/generated/4_year_old/annotated_batchalign/stage2_sli_annotated/"
    "pair_sli480_td450_rep07_A1_SLI.cha",
    "data/generated/4_year_old/reannotated_talktag/"
    "bundled_synthetic_children/stage2_sli/"
    "pair_sli480_td450_rep07_SLI_bundle.cha",
)


def default_pair_specs(repo_root: Path) -> list[PairSpec]:
    generated = repo_root / "data" / "generated"
    return [
        PairSpec(
            "age4_td",
            generated / "4_year_old" / "cha" / "stage1_td",
            generated / "4_year_old" / "reannotated_talktag" / "stage1_td_reannotated",
        ),
        PairSpec(
            "age4_sli",
            generated / "4_year_old" / "cha" / "stage2_sli",
            generated / "4_year_old" / "reannotated_talktag" / "stage2_sli_reannotated",
        ),
        PairSpec(
            "age5_td",
            generated / "5_year_old" / "5_year_old_cha" / "stage1_td",
            generated
            / "5_year_old"
            / "talktag-annotated"
            / "5_year_old_cha_stage1_td_annotated",
        ),
        PairSpec(
            "age5_sli",
            generated / "5_year_old" / "5_year_old_cha" / "stage2_sli",
            generated
            / "5_year_old"
            / "talktag-annotated"
            / "5_year_old_cha_stage2_sli_annotated",
        ),
        PairSpec(
            "age6_td",
            generated / "6_year_old" / "0_6_year_old_cha" / "stage1_td",
            generated
            / "6_year_old"
            / "1_talktag-annotated"
            / "6_year_old_cha_stage1_td_annotated",
        ),
        PairSpec(
            "age6_sli",
            generated / "6_year_old" / "0_6_year_old_cha" / "stage2_sli",
            generated
            / "6_year_old"
            / "1_talktag-annotated"
            / "6_year_old_cha_stage2_sli_annotated",
        ),
        PairSpec(
            "age7_td",
            generated / "7_year_old" / "0_7_year_old_cha" / "stage1_td",
            generated
            / "7_year_old"
            / "1_talktag-annotated"
            / "7_year_old_cha_stage1_td_annotated",
        ),
        PairSpec(
            "age7_sli",
            generated / "7_year_old" / "0_7_year_old_cha" / "stage2_sli",
            generated
            / "7_year_old"
            / "1_talktag-annotated"
            / "7_year_old_cha_stage2_sli_annotated",
        ),
    ]


def _mask_brackets(line: str) -> str:
    chars = list(line)
    for match in BRACKET_RE.finditer(line):
        chars[match.start() : match.end()] = " " * (match.end() - match.start())
    return "".join(chars)


def _surface_tokens(line: str) -> list[SurfaceToken]:
    body_start = line.find("\t") + 1 if "\t" in line else line.find(":") + 1
    masked = _mask_brackets(line)
    return [
        SurfaceToken(match.group(0).casefold(), match.start(), match.end())
        for match in TOKEN_RE.finditer(masked, body_start)
    ]


def _normalize_code(code: str) -> tuple[str, str | None]:
    compact = code.strip()
    if compact == "m :zero's":
        return "m:0's", "zero_possessive"
    normalized = re.sub(r"\s*:\s*", ":", compact)
    if normalized != compact and " " not in normalized:
        return normalized, "spacing"
    return compact, None


def _normalize_and_filter_tags(line: str) -> tuple[str, list[str]]:
    reasons: list[str] = []
    tokens = _surface_tokens(line)
    replacements: list[tuple[int, int, str]] = []

    for match in ERROR_RE.finditer(line):
        code, reason = _normalize_code(match.group(1))
        anchor = sum(token.end <= match.start() for token in tokens) - 1
        anchor_value = tokens[anchor].value if 0 <= anchor < len(tokens) else None
        if code == "m:+ing to" and anchor_value == "gonna":
            start = match.start()
            while start > 0 and line[start - 1] == " ":
                start -= 1
            replacements.append((start, match.end(), ""))
            reasons.append("remove_gonna_tag")
            continue
        replacement = f"[* {code}]"
        if replacement != match.group(0):
            replacements.append((match.start(), match.end(), replacement))
        if reason is not None:
            reasons.append(reason)

    for start, end, replacement in reversed(replacements):
        line = line[:start] + replacement + line[end:]
    return line, reasons


def _inline_tags(line: str) -> list[InlineTag]:
    tokens = _surface_tokens(line)
    tags: list[InlineTag] = []
    for match in ERROR_RE.finditer(line):
        anchor = sum(token.end <= match.start() for token in tokens) - 1
        tags.append(InlineTag(match.group(1).strip(), anchor if anchor >= 0 else None))
    return tags


def _token_mapping(source: list[SurfaceToken], target: list[SurfaceToken]) -> dict[int, int]:
    matcher = SequenceMatcher(
        None,
        [token.value for token in target],
        [token.value for token in source],
        autojunk=False,
    )
    mapping: dict[int, int] = {}
    for target_start, source_start, size in matcher.get_matching_blocks():
        for offset in range(size):
            mapping[target_start + offset] = source_start + offset
    return mapping


def _safe_anchor(
    anchor: int | None,
    mapping: dict[int, int],
    source_count: int,
    target_count: int,
) -> int | None:
    if anchor is None or anchor not in mapping:
        return None
    source_anchor = mapping[anchor]
    if anchor > 0 and mapping.get(anchor - 1) != source_anchor - 1:
        return None
    if anchor + 1 < target_count and mapping.get(anchor + 1) != source_anchor + 1:
        return None
    if not 0 <= source_anchor < source_count:
        return None
    return source_anchor


def _insertion_position(source_line: str, token: SurfaceToken) -> int:
    position = token.end
    correction = CORRECTION_AFTER_TOKEN_RE.match(source_line[position:])
    if correction is not None:
        position += correction.end()
    return position


def reconstruct_from_source(
    source_line: str,
    target_line: str,
    *,
    preserve_source_surface: bool = False,
) -> tuple[str, list[str]]:
    normalized_target, reasons = _normalize_and_filter_tags(target_line)
    if _is_spoken_comment_conversion(source_line, normalized_target):
        reasons.append("restore_comment")
        return source_line, list(dict.fromkeys(reasons))
    if (
        not preserve_source_surface
        and _lexical_signature(source_line) == _lexical_signature(normalized_target)
    ):
        return normalized_target, list(dict.fromkeys(reasons))

    source_tokens = _surface_tokens(source_line)
    target_tokens = _surface_tokens(normalized_target)
    mapping = _token_mapping(source_tokens, target_tokens)
    insertions: dict[int, list[str]] = {}
    rejected: list[str] = []

    for tag in _inline_tags(normalized_target):
        source_anchor = _safe_anchor(
            tag.anchor,
            mapping,
            len(source_tokens),
            len(target_tokens),
        )
        if source_anchor is None:
            rejected.append(tag.code)
            continue
        position = _insertion_position(source_line, source_tokens[source_anchor])
        insertions.setdefault(position, []).append(tag.code)

    repaired = source_line
    for position in sorted(insertions, reverse=True):
        annotation = "".join(f" [* {code}]" for code in insertions[position])
        repaired = repaired[:position] + annotation + repaired[position:]

    if ERROR_RE.sub("", normalized_target) != source_line:
        reasons.append("restore_source_text")
    if rejected:
        reasons.append("reject_unaligned_tag:" + ",".join(rejected))
    return repaired, list(dict.fromkeys(reasons))


def _speaker_blocks(lines: list[str]) -> tuple[list[str], list[SpeakerBlock], list[str]]:
    indices = [index for index, line in enumerate(lines) if line.startswith("*CHI:")]
    if not indices:
        raise ValueError("Transcript has no *CHI tier.")
    prefix = lines[: indices[0]]
    blocks: list[SpeakerBlock] = []
    for position, start in enumerate(indices):
        next_speaker = indices[position + 1] if position + 1 < len(indices) else len(lines)
        end = next_speaker
        for index in range(start + 1, next_speaker):
            if lines[index].startswith("@"):
                end = index
                break
        blocks.append(SpeakerBlock(start, lines[start:end]))
    suffix_start = blocks[-1].line_index + len(blocks[-1].lines)
    return prefix, blocks, lines[suffix_start:]


def _source_utterance(repo_root: Path, relative_path: str, index: int) -> str:
    path = repo_root / relative_path
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    _, blocks, _ = _speaker_blocks(lines)
    try:
        return blocks[index].lines[0]
    except IndexError as exc:
        raise ValueError(
            f"Missing source utterance {index} in {relative_path}."
        ) from exc


def _combine_speaker_lines(blocks: list[SpeakerBlock]) -> str:
    bodies = []
    for block in blocks:
        line = block.lines[0]
        body = line.split("\t", 1)[1] if "\t" in line else line.split(":", 1)[1]
        bodies.append(body.strip())
    return "*CHI:\t" + " ".join(bodies)


def _replace_block_span(
    lines: list[str],
    blocks: list[SpeakerBlock],
    start: int,
    end: int,
    replacement: str | None,
) -> list[str]:
    line_start = blocks[start].line_index
    line_end = blocks[end - 1].line_index + len(blocks[end - 1].lines)
    replacement_lines = [] if replacement is None else [replacement]
    return lines[:line_start] + replacement_lines + lines[line_end:]


def merge_exact_split_utterance(
    lines: list[str], source_line: str
) -> tuple[list[str], bool]:
    """Restore one source utterance that a later stage split into several tiers."""
    _, blocks, _ = _speaker_blocks(lines)
    source_signature = _lexical_signature(source_line)

    if any(_lexical_signature(block.lines[0]) == source_signature for block in blocks):
        return lines, False

    matches: list[tuple[int, int]] = []
    for start in range(len(blocks)):
        combined: list[str] = []
        for end in range(start + 1, len(blocks) + 1):
            combined.extend(_lexical_signature(blocks[end - 1].lines[0]))
            if len(combined) > len(source_signature):
                break
            if end - start > 1 and combined == source_signature:
                matches.append((start, end))

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one multi-line lexical match for confirmed split; "
            f"found {len(matches)}."
        )

    start, end = matches[0]
    combined_target = _combine_speaker_lines(blocks[start:end])
    repaired, _ = reconstruct_from_source(
        source_line,
        combined_target,
        preserve_source_surface=True,
    )
    return _replace_block_span(lines, blocks, start, end, repaired), True


def remove_exact_speaker_block(
    lines: list[str], target_line: str
) -> tuple[list[str], bool]:
    """Remove one confirmed inserted utterance and all of its dependent tiers."""
    _, blocks, _ = _speaker_blocks(lines)
    signature = _lexical_signature(target_line)
    matches = [
        index
        for index, block in enumerate(blocks)
        if _lexical_signature(block.lines[0]) == signature
    ]
    if not matches:
        return lines, False
    if len(matches) != 1:
        raise ValueError(
            f"Expected one confirmed inserted line, found {len(matches)} matches."
        )
    index = matches[0]
    return _replace_block_span(lines, blocks, index, index + 1, None), True


def restore_confirmed_truncation(
    lines: list[str], source_line: str
) -> tuple[list[str], bool]:
    """Restore the one verified TalkTag output-cap truncation."""
    _, blocks, _ = _speaker_blocks(lines)
    source_signature = _lexical_signature(source_line)
    if any(_lexical_signature(block.lines[0]) == source_signature for block in blocks):
        return lines, False

    truncated_signature = ["her"] * 18
    matches = [
        index
        for index, block in enumerate(blocks)
        if _lexical_signature(block.lines[0]) == truncated_signature
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected one confirmed repeated-'her' truncation; "
            f"found {len(matches)}."
        )
    index = matches[0]
    repaired, _ = reconstruct_from_source(source_line, blocks[index].lines[0])
    return _replace_block_span(lines, blocks, index, index + 1, repaired), True


def restore_exact_source_block(
    lines: list[str], source_line: str, current_line: str
) -> tuple[list[str], bool]:
    """Restore one reviewed lexical corruption and discard its stale tiers."""
    _, blocks, _ = _speaker_blocks(lines)
    source_signature = _lexical_signature(source_line)
    if any(_lexical_signature(block.lines[0]) == source_signature for block in blocks):
        return lines, False

    current_signature = _lexical_signature(current_line)
    matches = [
        index
        for index, block in enumerate(blocks)
        if _lexical_signature(block.lines[0]) == current_signature
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected one reviewed lexical-corruption block; "
            f"found {len(matches)}."
        )
    index = matches[0]
    return _replace_block_span(lines, blocks, index, index + 1, source_line), True


def _number_words(token: str) -> list[str]:
    if not token.isdigit():
        return [token]
    value = int(token)
    if value in SMALL_NUMBER_WORDS:
        return [SMALL_NUMBER_WORDS[value]]
    if value in TENS_WORDS:
        return [TENS_WORDS[value]]
    if 20 < value < 100:
        return [TENS_WORDS[(value // 10) * 10], SMALL_NUMBER_WORDS[value % 10]]
    return [token]


def _lexical_signature(line: str) -> list[str]:
    signature: list[str] = []
    tokens = _surface_tokens(line)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if (
            token.value == "$"
            and index + 1 < len(tokens)
            and tokens[index + 1].value.isdigit()
        ):
            signature.extend(_number_words(tokens[index + 1].value))
            signature.append("dollars")
            index += 2
            continue
        if not any(char.isalnum() for char in token.value):
            index += 1
            continue
        for part in token.value.split("_"):
            signature.extend(_number_words(part))
        index += 1
    return signature


def _is_spoken_comment_conversion(source_line: str, target_line: str) -> bool:
    return '+"' in source_line and "&=" in target_line and "_" in target_line


def _utterance_similarity(source_line: str, target_line: str) -> float:
    source = _lexical_signature(source_line)
    target = _lexical_signature(ERROR_RE.sub("", target_line))
    if not source and not target:
        return 1.0
    if not source or not target:
        return 0.0
    return SequenceMatcher(None, target, source, autojunk=False).ratio()


def _safe_lexical_restore(source_line: str, target_line: str) -> bool:
    source = _lexical_signature(source_line)
    target = _lexical_signature(ERROR_RE.sub("", target_line))
    if source == target:
        return True
    return SequenceMatcher(None, target, source, autojunk=False).ratio() >= 0.75


def _alignment_score(source_line: str, target_line: str) -> float:
    similarity = _utterance_similarity(source_line, target_line)
    if similarity < 0.3:
        return -1.2
    return 2.0 * similarity - 1.0


def _align_utterances(
    source_lines: list[str],
    target_lines: list[str],
) -> list[tuple[int | None, int | None]]:
    source_count = len(source_lines)
    target_count = len(target_lines)
    gap = -0.45
    scores = [[0.0] * (target_count + 1) for _ in range(source_count + 1)]
    moves = [[""] * (target_count + 1) for _ in range(source_count + 1)]
    for source_index in range(1, source_count + 1):
        scores[source_index][0] = source_index * gap
        moves[source_index][0] = "source"
    for target_index in range(1, target_count + 1):
        scores[0][target_index] = target_index * gap
        moves[0][target_index] = "target"

    for source_index in range(1, source_count + 1):
        for target_index in range(1, target_count + 1):
            choices = {
                "pair": scores[source_index - 1][target_index - 1]
                + _alignment_score(
                    source_lines[source_index - 1],
                    target_lines[target_index - 1],
                ),
                "source": scores[source_index - 1][target_index] + gap,
                "target": scores[source_index][target_index - 1] + gap,
            }
            move = max(choices, key=choices.get)
            scores[source_index][target_index] = choices[move]
            moves[source_index][target_index] = move

    alignment: list[tuple[int | None, int | None]] = []
    source_index = source_count
    target_index = target_count
    while source_index or target_index:
        move = moves[source_index][target_index]
        if move == "pair":
            alignment.append((source_index - 1, target_index - 1))
            source_index -= 1
            target_index -= 1
        elif move == "source":
            alignment.append((source_index - 1, None))
            source_index -= 1
        elif move == "target":
            alignment.append((None, target_index - 1))
            target_index -= 1
        else:
            raise RuntimeError("Utterance alignment backtrace failed.")
    alignment.reverse()
    return alignment


def build_repair_plan(
    repo_root: Path,
    specs: list[PairSpec],
) -> tuple[dict[Path, list[str]], list[RepairRecord], dict[str, int]]:
    planned_files: dict[Path, list[str]] = {}
    records: list[RepairRecord] = []
    summary = {
        "pair_specs": len(specs),
        "files_checked": 0,
        "files_changed": 0,
        "lines_checked": 0,
        "lines_changed": 0,
        "spacing_annotations_repaired": 0,
        "zero_possessive_annotations_repaired": 0,
        "gonna_tags_removed": 0,
        "source_text_restorations": 0,
        "comment_lines_restored": 0,
        "lines_with_unaligned_tags_rejected": 0,
        "manual_check_lines": 0,
        "expected_nonlexical_lines_omitted": 0,
        "missing_source_lines_for_check": 0,
        "non_source_lines_for_check": 0,
    }

    for spec in specs:
        source_paths = sorted(spec.source_root.glob("*.cha"))
        target_paths = sorted(spec.target_root.glob("*.cha"))
        source_names = {path.name for path in source_paths}
        target_names = {path.name for path in target_paths}
        if source_names != target_names:
            missing_targets = sorted(source_names - target_names)
            missing_sources = sorted(target_names - source_names)
            raise ValueError(
                f"{spec.label}: source/target filenames differ; "
                f"missing targets={missing_targets[:5]}, missing sources={missing_sources[:5]}"
            )

        for source_path in source_paths:
            target_path = spec.target_root / source_path.name
            source_lines = source_path.read_text(encoding="utf-8", errors="strict").splitlines()
            target_text = target_path.read_text(encoding="utf-8", errors="strict")
            target_lines = target_text.splitlines()
            _, source_blocks, _ = _speaker_blocks(source_lines)
            _, target_blocks, _ = _speaker_blocks(target_lines)
            source_chi = [block.lines[0] for block in source_blocks]
            target_chi = [block.lines[0] for block in target_blocks]
            alignment = _align_utterances(source_chi, target_chi)
            rebuilt = list(target_lines)
            file_records: list[RepairRecord] = []

            aligned_source: dict[int, int] = {
                target_index: source_index
                for source_index, target_index in alignment
                if source_index is not None and target_index is not None
            }
            equal_counts = len(source_chi) == len(target_chi)

            for target_index, block in enumerate(target_blocks):
                original = block.lines[0]
                normalized, tag_reasons = _normalize_and_filter_tags(original)
                summary["spacing_annotations_repaired"] += tag_reasons.count("spacing")
                summary["zero_possessive_annotations_repaired"] += tag_reasons.count(
                    "zero_possessive"
                )
                summary["gonna_tags_removed"] += tag_reasons.count("remove_gonna_tag")
                current = normalized
                source_index = aligned_source.get(target_index)
                source_line = source_chi[source_index] if source_index is not None else ""
                reasons = list(dict.fromkeys(tag_reasons))
                status = "AUTO"

                if source_line and _is_spoken_comment_conversion(source_line, normalized):
                    current, repair_reasons = reconstruct_from_source(source_line, normalized)
                    reasons.extend(reason for reason in repair_reasons if reason not in reasons)
                elif source_line and _lexical_signature(source_line) != _lexical_signature(normalized):
                    if equal_counts and source_index == target_index and _safe_lexical_restore(
                        source_line, normalized
                    ):
                        current, repair_reasons = reconstruct_from_source(source_line, normalized)
                        reasons.extend(reason for reason in repair_reasons if reason not in reasons)
                    else:
                        status = "CHECK"
                        reasons.append("lexical_mismatch")
                elif source_index is None:
                    status = "CHECK"
                    reasons.append("non_source_line")
                    summary["non_source_lines_for_check"] += 1

                rebuilt[block.line_index] = current
                summary["lines_checked"] += 1
                if current != original:
                    summary["lines_changed"] += 1
                    summary["source_text_restorations"] += reasons.count("restore_source_text")
                    summary["comment_lines_restored"] += reasons.count("restore_comment")
                    summary["lines_with_unaligned_tags_rejected"] += sum(
                        reason.startswith("reject_unaligned_tag:") for reason in reasons
                    )
                if current != original or status == "CHECK":
                    file_records.append(
                        RepairRecord(
                            file=str(target_path.relative_to(repo_root)),
                            line=block.line_index + 1,
                            status=status,
                            current=current,
                            source=source_line,
                            reason=";".join(dict.fromkeys(reasons)),
                        )
                    )

            for alignment_position, (source_index, target_index) in enumerate(alignment):
                if source_index is None or target_index is not None:
                    continue
                if not _lexical_signature(source_chi[source_index]):
                    summary["expected_nonlexical_lines_omitted"] += 1
                    continue
                next_target = next(
                    (
                        later_target
                        for _, later_target in alignment[alignment_position + 1 :]
                        if later_target is not None
                    ),
                    None,
                )
                line_hint = (
                    target_blocks[next_target].line_index + 1
                    if next_target is not None
                    else target_blocks[-1].line_index + 1
                )
                file_records.append(
                    RepairRecord(
                        file=str(target_path.relative_to(repo_root)),
                        line=line_hint,
                        status="CHECK",
                        current="",
                        source=source_chi[source_index],
                        reason="missing_source_line",
                    )
                )
                summary["missing_source_lines_for_check"] += 1

            summary["manual_check_lines"] += sum(
                record.status == "CHECK" for record in file_records
            )

            summary["files_checked"] += 1
            if rebuilt != target_lines:
                summary["files_changed"] += 1
                planned_files[target_path] = rebuilt
            records.extend(file_records)

    return planned_files, records, summary


def _write_manual_check(path: Path, records: list[RepairRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["file", "line", "status", "current", "source", "reason"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(asdict(record) for record in records if record.status == "CHECK")


def apply_repair_plan(planned_files: dict[Path, list[str]]) -> None:
    for path, lines in planned_files.items():
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_confirmed_structural_repair_plan(
    repo_root: Path,
) -> tuple[dict[Path, list[str]], dict[str, object]]:
    planned_files: dict[Path, list[str]] = {}
    records: list[dict[str, object]] = []

    def current_lines(path: Path) -> list[str]:
        if path in planned_files:
            return planned_files[path]
        return path.read_text(encoding="utf-8", errors="strict").splitlines()

    for repair in CONFIRMED_MERGE_REPAIRS:
        source_line = _source_utterance(
            repo_root,
            repair.source_file,
            repair.source_utterance_index,
        )
        for relative_target in repair.target_files:
            target_path = repo_root / relative_target
            before = current_lines(target_path)
            after, changed = merge_exact_split_utterance(before, source_line)
            if changed:
                planned_files[target_path] = after
            records.append(
                {
                    "action": "restore_source_segmentation",
                    "changed": changed,
                    "label": repair.label,
                    "source_file": repair.source_file,
                    "source_utterance_index": repair.source_utterance_index,
                    "target_file": relative_target,
                }
            )

    for relative_target in CONFIRMED_INSERTED_LINE_TARGETS:
        target_path = repo_root / relative_target
        before = current_lines(target_path)
        after, changed = remove_exact_speaker_block(
            before,
            CONFIRMED_INSERTED_LINE,
        )
        if changed:
            planned_files[target_path] = after
        records.append(
            {
                "action": "remove_cross_record_inserted_line",
                "changed": changed,
                "target_file": relative_target,
            }
        )

    truncation_source = _source_utterance(
        repo_root,
        CONFIRMED_TRUNCATION_SOURCE,
        CONFIRMED_TRUNCATION_SOURCE_UTTERANCE,
    )
    for relative_target in CONFIRMED_TRUNCATION_TARGETS:
        target_path = repo_root / relative_target
        before = current_lines(target_path)
        after, changed = restore_confirmed_truncation(before, truncation_source)
        if changed:
            planned_files[target_path] = after
        records.append(
            {
                "action": "restore_talktag_output_cap_truncation",
                "changed": changed,
                "source_file": CONFIRMED_TRUNCATION_SOURCE,
                "source_utterance_index": CONFIRMED_TRUNCATION_SOURCE_UTTERANCE,
                "target_file": relative_target,
            }
        )

    exposed_source = _source_utterance(
        repo_root,
        CONFIRMED_EXPOSED_RESTORATION_SOURCE,
        CONFIRMED_EXPOSED_RESTORATION_SOURCE_UTTERANCE,
    )
    for relative_target in CONFIRMED_EXPOSED_RESTORATION_TARGETS:
        target_path = repo_root / relative_target
        before = current_lines(target_path)
        after, changed = restore_exact_source_block(
            before,
            exposed_source,
            CONFIRMED_EXPOSED_RESTORATION_CURRENT,
        )
        if changed:
            planned_files[target_path] = after
        records.append(
            {
                "action": "propagate_exposed_source_text_restoration",
                "changed": changed,
                "source_file": CONFIRMED_EXPOSED_RESTORATION_SOURCE,
                "source_utterance_index": (
                    CONFIRMED_EXPOSED_RESTORATION_SOURCE_UTTERANCE
                ),
                "target_file": relative_target,
            }
        )

    changed_this_run = 0
    for record in records:
        changed = bool(record.pop("changed"))
        record["changed_this_run"] = changed
        record["resolved"] = True
        changed_this_run += changed
    summary: dict[str, object] = {
        "files_changed_this_run": len(planned_files),
        "files_in_scope": len({str(record["target_file"]) for record in records}),
        "repairs_changed_this_run": changed_this_run,
        "repairs_resolved": len(records),
        "records": records,
    }
    return planned_files, summary


def run_confirmed_structural_repairs(
    repo_root: Path,
    apply: bool,
) -> dict[str, object]:
    planned_files, summary = build_confirmed_structural_repair_plan(repo_root)
    if apply:
        apply_repair_plan(planned_files)
        output_path = (
            repo_root
            / "outputs"
            / "annotation_qc"
            / "talktag_confirmed_structural_repairs_summary.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def build_derived_copy_normalization_plan(
    repo_root: Path,
) -> tuple[dict[Path, list[str]], dict[str, int]]:
    canonical_roots = [spec.target_root.resolve() for spec in default_pair_specs(repo_root)]
    planned_files: dict[Path, list[str]] = {}
    summary = {
        "files_checked": 0,
        "files_changed": 0,
        "lines_changed": 0,
        "spacing_annotations_repaired": 0,
        "zero_possessive_annotations_repaired": 0,
        "gonna_tags_removed": 0,
    }

    for path in sorted((repo_root / "data" / "generated").rglob("*.cha")):
        resolved = path.resolve()
        if any(_is_under(resolved, canonical_root) for canonical_root in canonical_roots):
            continue
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
        changed = False
        for index, line in enumerate(lines):
            if not line.startswith("*CHI:"):
                continue
            normalized, reasons = _normalize_and_filter_tags(line)
            if normalized == line:
                continue
            lines[index] = normalized
            changed = True
            summary["lines_changed"] += 1
            summary["spacing_annotations_repaired"] += reasons.count("spacing")
            summary["zero_possessive_annotations_repaired"] += reasons.count(
                "zero_possessive"
            )
            summary["gonna_tags_removed"] += reasons.count("remove_gonna_tag")
        summary["files_checked"] += 1
        if changed:
            summary["files_changed"] += 1
            planned_files[path] = lines
    return planned_files, summary


def run_derived_copy_normalization(repo_root: Path, apply: bool) -> dict[str, int]:
    planned_files, summary = build_derived_copy_normalization_plan(repo_root)
    if apply:
        apply_repair_plan(planned_files)
        output_path = (
            repo_root
            / "outputs"
            / "annotation_qc"
            / "talktag_derived_copy_normalization_summary.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary


def run(repo_root: Path, apply: bool) -> dict[str, int]:
    specs = default_pair_specs(repo_root)
    planned_files, records, summary = build_repair_plan(repo_root, specs)
    output_root = repo_root / "outputs" / "annotation_qc"
    if apply:
        apply_repair_plan(planned_files)
        _write_manual_check(output_root / "talktag_manual_checks.tsv", records)
        (output_root / "talktag_repair_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restore source wording and repair malformed TalkTag annotations."
    )
    parser.add_argument(
        "--repo-root",
        default=str(ROOT),
        help="ICL-PILOT repository root.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply repairs and write the QC artifacts. Without this flag, run a dry audit.",
    )
    parser.add_argument(
        "--derived-copies-only",
        action="store_true",
        help="Normalize malformed tags in non-canonical generated transcript copies only.",
    )
    parser.add_argument(
        "--confirmed-structural-repairs-only",
        action="store_true",
        help=(
            "Apply the reviewed source-segmentation, inserted-line, and output-cap "
            "repairs only."
        ),
    )
    args = parser.parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    if args.derived_copies_only and args.confirmed_structural_repairs_only:
        parser.error(
            "--derived-copies-only and --confirmed-structural-repairs-only "
            "are mutually exclusive"
        )
    if args.confirmed_structural_repairs_only:
        summary = run_confirmed_structural_repairs(repo_root, args.apply)
    elif args.derived_copies_only:
        summary = run_derived_copy_normalization(repo_root, args.apply)
    else:
        summary = run(repo_root, args.apply)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
