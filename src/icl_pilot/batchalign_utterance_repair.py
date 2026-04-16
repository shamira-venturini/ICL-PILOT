from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


_PUNCT_TOKENS = {".", "!", "?", ",", ";", ":"}
_NO_MOR_TOKENS = {"+", "++", "[!]", "/.", "/"}
_QUOTE_MARKERS = {"+", "++", "+.", "/.", "/", "+/.", "+//.", '+"/.', '+".', '+"'}
_MOR_DELIMITER_TOKENS = {".", "!", "?", "+.", "/.", "+/.", "+//.", "+//?", "+/?", "+..?", "+!?", "+..."}
_MONEY_RE = re.compile(r"\$(\d+)")
_DIGIT_RE = re.compile(r"\b(\d+)\b")
_SPACED_DOT_PAUSE_RE = re.compile(r"\((?:\s*\.\s*){2,}\)")
_NONFINAL_BANG_RE = re.compile(r"!\s+(?=\S)")


@dataclass(frozen=True)
class ChatBlock:
    main_text: str
    dep_lines: tuple[str, ...]


@dataclass(frozen=True)
class PlannedBlock:
    main_text: str
    dep_lines: tuple[str, ...]


@dataclass(frozen=True)
class BatchalignUtteranceRepairResult:
    files_examined: int
    files_touched: int
    split_blocks_repaired: int
    extra_blocks_removed: int
    mor_tiers_preserved: int
    mor_tiers_dropped: int
    gra_tiers_dropped: int
    appended_source_only_blocks: int
    failed_files: tuple[str, ...]


@dataclass(frozen=True)
class BatchalignDelimiterRepairResult:
    files_examined: int
    files_touched: int
    chi_lines_touched: int
    failed_files: tuple[str, ...]


@dataclass(frozen=True)
class BatchalignMarkerRepairResult:
    files_examined: int
    files_touched: int
    chi_lines_touched: int
    failed_files: tuple[str, ...]


@dataclass(frozen=True)
class BatchalignPunctuationOnlyRemovalResult:
    files_examined: int
    files_touched: int
    chi_blocks_removed: int
    failed_files: tuple[str, ...]


@dataclass(frozen=True)
class BatchalignDepfileCleanupResult:
    files_examined: int
    files_touched: int
    chi_lines_touched: int
    mor_lines_touched: int
    gra_lines_dropped: int
    failed_files: tuple[str, ...]


def repair_batchalign_utterances(
    source_root: str | Path,
    target_root: str | Path,
) -> BatchalignUtteranceRepairResult:
    source_dir = Path(source_root).expanduser().resolve()
    target_dir = Path(target_root).expanduser().resolve()

    files_examined = 0
    files_touched = 0
    split_blocks_repaired = 0
    extra_blocks_removed = 0
    mor_tiers_preserved = 0
    mor_tiers_dropped = 0
    gra_tiers_dropped = 0
    appended_source_only_blocks = 0
    failed_files: list[str] = []

    for source_path in sorted(source_dir.rglob("*.cha")):
        relative = source_path.relative_to(source_dir)
        target_path = target_dir / relative
        if not target_path.exists():
            continue

        files_examined += 1
        source_lines = _extract_chi_lines(source_path)
        header_lines, target_blocks, footer_lines = _parse_target_chat(target_path)

        try:
            plan = _build_repair_plan(source_lines=source_lines, target_blocks=target_blocks)
        except ValueError:
            failed_files.append(str(relative))
            continue

        new_lines = [*header_lines]
        for block in plan.blocks:
            new_lines.append(f"*CHI:\t{block.main_text}")
            new_lines.extend(block.dep_lines)
        new_lines.extend(footer_lines)
        new_text = "\n".join(new_lines) + "\n"
        old_text = target_path.read_text(encoding="utf-8")

        if new_text != old_text:
            target_path.write_text(new_text, encoding="utf-8")
            files_touched += 1

        split_blocks_repaired += plan.split_blocks_repaired
        extra_blocks_removed += plan.extra_blocks_removed
        mor_tiers_preserved += plan.mor_tiers_preserved
        mor_tiers_dropped += plan.mor_tiers_dropped
        gra_tiers_dropped += plan.gra_tiers_dropped
        appended_source_only_blocks += plan.appended_source_only_blocks

    return BatchalignUtteranceRepairResult(
        files_examined=files_examined,
        files_touched=files_touched,
        split_blocks_repaired=split_blocks_repaired,
        extra_blocks_removed=extra_blocks_removed,
        mor_tiers_preserved=mor_tiers_preserved,
        mor_tiers_dropped=mor_tiers_dropped,
        gra_tiers_dropped=gra_tiers_dropped,
        appended_source_only_blocks=appended_source_only_blocks,
        failed_files=tuple(failed_files),
    )


def repair_batchalign_quote_markers(
    source_root: str | Path,
    target_root: str | Path,
) -> BatchalignMarkerRepairResult:
    source_dir = Path(source_root).expanduser().resolve()
    target_dir = Path(target_root).expanduser().resolve()

    files_examined = 0
    files_touched = 0
    chi_lines_touched = 0
    failed_files: list[str] = []

    for source_path in sorted(source_dir.rglob("*.cha")):
        relative = source_path.relative_to(source_dir)
        target_path = target_dir / relative
        if not target_path.exists():
            continue

        source_lines = _extract_chi_lines(source_path)
        target_lines = target_path.read_text(encoding="utf-8").splitlines()
        source_index = 0
        changed = False
        new_lines: list[str] = []

        for line in target_lines:
            if not line.startswith("*CHI:\t"):
                new_lines.append(line)
                continue

            if source_index >= len(source_lines):
                failed_files.append(str(relative))
                new_lines = []
                break

            source_text = source_lines[source_index]
            previous_source = source_lines[source_index - 1] if source_index > 0 else None
            next_source = source_lines[source_index + 1] if source_index + 1 < len(source_lines) else None
            target_text = line.split("\t", 1)[1]
            repaired = _restore_target_quote_markers(
                previous_source_text=previous_source,
                source_text=source_text,
                next_source_text=next_source,
                target_text=target_text,
            )
            if repaired != target_text:
                changed = True
                chi_lines_touched += 1
            new_lines.append(f"*CHI:\t{repaired}")
            source_index += 1

        if not new_lines:
            continue
        if source_index != len(source_lines):
            failed_files.append(str(relative))
            continue

        files_examined += 1
        if changed:
            target_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            files_touched += 1

    return BatchalignMarkerRepairResult(
        files_examined=files_examined,
        files_touched=files_touched,
        chi_lines_touched=chi_lines_touched,
        failed_files=tuple(sorted(set(failed_files))),
    )


def repair_batchalign_delimiters(
    source_root: str | Path,
    target_root: str | Path,
) -> BatchalignDelimiterRepairResult:
    source_dir = Path(source_root).expanduser().resolve()
    target_dir = Path(target_root).expanduser().resolve()

    files_examined = 0
    files_touched = 0
    chi_lines_touched = 0
    failed_files: list[str] = []

    for source_path in sorted(source_dir.rglob("*.cha")):
        relative = source_path.relative_to(source_dir)
        target_path = target_dir / relative
        if not target_path.exists():
            continue

        source_lines = _extract_chi_lines(source_path)
        target_lines = target_path.read_text(encoding="utf-8").splitlines()
        source_index = 0
        changed = False
        new_lines: list[str] = []

        for line in target_lines:
            if not line.startswith("*CHI:\t"):
                new_lines.append(line)
                continue

            if source_index >= len(source_lines):
                failed_files.append(str(relative))
                new_lines = []
                break

            original = line.split("\t", 1)[1]
            repaired = _restore_target_delimiters(
                source_text=source_lines[source_index],
                target_text=original,
            )
            if repaired != original:
                changed = True
                chi_lines_touched += 1
            new_lines.append(f"*CHI:\t{repaired}")
            source_index += 1

        if not new_lines:
            continue
        if source_index != len(source_lines):
            failed_files.append(str(relative))
            continue

        files_examined += 1
        if changed:
            target_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            files_touched += 1

    return BatchalignDelimiterRepairResult(
        files_examined=files_examined,
        files_touched=files_touched,
        chi_lines_touched=chi_lines_touched,
        failed_files=tuple(sorted(set(failed_files))),
    )


def remove_punctuation_only_utterances(
    target_root: str | Path,
) -> BatchalignPunctuationOnlyRemovalResult:
    target_dir = Path(target_root).expanduser().resolve()

    files_examined = 0
    files_touched = 0
    chi_blocks_removed = 0
    failed_files: list[str] = []

    for target_path in sorted(target_dir.rglob("*.cha")):
        relative = target_path.relative_to(target_dir)

        try:
            header_lines, target_blocks, footer_lines = _parse_target_chat(target_path)
        except ValueError:
            failed_files.append(str(relative))
            continue

        files_examined += 1
        kept_blocks: list[ChatBlock] = []
        removed_here = 0

        for block in target_blocks:
            if _target_lexical_tokens(block.main_text):
                kept_blocks.append(block)
                continue
            removed_here += 1

        if not removed_here:
            continue

        new_lines = [*header_lines]
        for block in kept_blocks:
            new_lines.append(f"*CHI:\t{block.main_text}")
            new_lines.extend(block.dep_lines)
        new_lines.extend(footer_lines)

        old_text = target_path.read_text(encoding="utf-8")
        new_text = "\n".join(new_lines) + "\n"
        if new_text != old_text:
            target_path.write_text(new_text, encoding="utf-8")
            files_touched += 1

        chi_blocks_removed += removed_here

    return BatchalignPunctuationOnlyRemovalResult(
        files_examined=files_examined,
        files_touched=files_touched,
        chi_blocks_removed=chi_blocks_removed,
        failed_files=tuple(sorted(set(failed_files))),
    )


def cleanup_depfile_surface_errors(
    target_root: str | Path,
) -> BatchalignDepfileCleanupResult:
    target_dir = Path(target_root).expanduser().resolve()

    files_examined = 0
    files_touched = 0
    chi_lines_touched = 0
    mor_lines_touched = 0
    gra_lines_dropped = 0
    failed_files: list[str] = []

    for target_path in sorted(target_dir.rglob("*.cha")):
        relative = target_path.relative_to(target_dir)

        try:
            lines = target_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            failed_files.append(str(relative))
            continue

        files_examined += 1
        changed = False
        new_lines: list[str] = []
        current_block_has_bare_equals_event = False
        current_block_is_event_only = False
        current_chi_text: str | None = None

        for line in lines:
            if line.startswith("*CHI:\t"):
                original = line.split("\t", 1)[1]
                current_block_has_bare_equals_event = _starts_with_bare_equals_event(original)
                repaired = _cleanup_chi_depfile_surface(original)
                current_block_is_event_only = _is_event_only_chi_line(repaired)
                current_chi_text = repaired
                if repaired != original:
                    changed = True
                    chi_lines_touched += 1
                new_lines.append(f"*CHI:\t{repaired}")
                continue

            if line.startswith("%mor:\t"):
                if current_block_is_event_only:
                    changed = True
                    mor_lines_touched += 1
                    continue
                original = line.split("\t", 1)[1]
                repaired = _cleanup_mor_depfile_surface(original)
                if current_block_has_bare_equals_event:
                    repaired = _drop_leading_equals_event_mor_token(repaired)
                if current_chi_text is not None:
                    repaired = _restore_mor_delimiter_from_chi(repaired, current_chi_text)
                if repaired != original:
                    changed = True
                    mor_lines_touched += 1
                new_lines.append(f"%mor:\t{repaired}")
                continue

            if line.startswith("%gra:\t") and (
                current_block_has_bare_equals_event
                or current_block_is_event_only
            ):
                changed = True
                gra_lines_dropped += 1
                continue

            new_lines.append(line)

        if changed:
            target_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            files_touched += 1

    return BatchalignDepfileCleanupResult(
        files_examined=files_examined,
        files_touched=files_touched,
        chi_lines_touched=chi_lines_touched,
        mor_lines_touched=mor_lines_touched,
        gra_lines_dropped=gra_lines_dropped,
        failed_files=tuple(sorted(set(failed_files))),
    )


@dataclass(frozen=True)
class _RepairPlan:
    blocks: tuple[PlannedBlock, ...]
    split_blocks_repaired: int
    extra_blocks_removed: int
    mor_tiers_preserved: int
    mor_tiers_dropped: int
    gra_tiers_dropped: int
    appended_source_only_blocks: int


@dataclass(frozen=True)
class _SplitPiece:
    main_text: str
    mor_token_count: int


def _build_repair_plan(
    *,
    source_lines: list[str],
    target_blocks: list[ChatBlock],
) -> _RepairPlan:
    planned_blocks: list[PlannedBlock] = []
    source_index = 0
    split_blocks_repaired = 0
    extra_blocks_removed = 0
    mor_tiers_preserved = 0
    mor_tiers_dropped = 0
    gra_tiers_dropped = 0
    appended_source_only_blocks = 0

    for block in target_blocks:
        target_lex = _target_lexical_tokens(block.main_text)

        if not target_lex:
            if source_index < len(source_lines) and not _source_lexical_tokens(source_lines[source_index]):
                inserted = _normalize_source_for_insert(source_lines[source_index])
                planned_blocks.append(PlannedBlock(main_text=inserted, dep_lines=()))
                source_index += 1
            else:
                extra_blocks_removed += 1
            continue

        matched_source_lines: list[str] = []
        accumulated_lex: list[str] = []

        while source_index < len(source_lines):
            source_line = source_lines[source_index]
            matched_source_lines.append(source_line)
            accumulated_lex.extend(_source_lexical_tokens(source_line))
            source_index += 1

            if accumulated_lex == target_lex:
                break
            if source_index == len(source_lines) and _is_lexical_prefix(accumulated_lex, target_lex):
                break
            if len(accumulated_lex) > len(target_lex):
                raise ValueError("source overshot target lexical sequence")
        else:
            raise ValueError("source exhausted before target blocks")

        pieces = _split_target_surface(block.main_text, matched_source_lines)
        if pieces is None:
            raise ValueError("could not split target surface by source boundaries")

        if len(matched_source_lines) == 1 and pieces[0].main_text == block.main_text:
            planned_blocks.append(block_as_planned(block))
            continue

        split_blocks_repaired += 1
        new_blocks, preserved_mor, dropped_mor, dropped_gra = _rewrite_split_block(block, pieces)
        planned_blocks.extend(new_blocks)
        mor_tiers_preserved += preserved_mor
        mor_tiers_dropped += dropped_mor
        gra_tiers_dropped += dropped_gra

    while source_index < len(source_lines):
        source_line = source_lines[source_index]
        if _source_lexical_tokens(source_line):
            raise ValueError("source still has lexical content after target blocks")
        planned_blocks.append(
            PlannedBlock(main_text=_normalize_source_for_insert(source_line), dep_lines=())
        )
        appended_source_only_blocks += 1
        source_index += 1

    return _RepairPlan(
        blocks=tuple(planned_blocks),
        split_blocks_repaired=split_blocks_repaired,
        extra_blocks_removed=extra_blocks_removed,
        mor_tiers_preserved=mor_tiers_preserved,
        mor_tiers_dropped=mor_tiers_dropped,
        gra_tiers_dropped=gra_tiers_dropped,
        appended_source_only_blocks=appended_source_only_blocks,
    )


def block_as_planned(block: ChatBlock) -> PlannedBlock:
    return PlannedBlock(main_text=block.main_text, dep_lines=block.dep_lines)


def _rewrite_split_block(
    block: ChatBlock,
    pieces: list[_SplitPiece],
) -> tuple[list[PlannedBlock], int, int, int]:
    mor_line = next((line for line in block.dep_lines if line.startswith("%mor:\t")), None)
    gra_line = next((line for line in block.dep_lines if line.startswith("%gra:\t")), None)
    mor_tokens = mor_line.split("\t", 1)[1].split() if mor_line else []

    keep_mor = bool(mor_tokens) and sum(piece.mor_token_count for piece in pieces) == len(mor_tokens)
    new_blocks: list[PlannedBlock] = []
    preserved_mor = 0
    dropped_mor = 0
    dropped_gra = 0
    mor_index = 0

    for piece in pieces:
        dep_lines: list[str] = []
        if keep_mor and piece.mor_token_count:
            next_index = mor_index + piece.mor_token_count
            dep_lines.append("%mor:\t" + " ".join(mor_tokens[mor_index:next_index]))
            mor_index = next_index
            preserved_mor += 1
        elif mor_line:
            dropped_mor += 1

        if gra_line:
            dropped_gra += 1

        new_blocks.append(PlannedBlock(main_text=piece.main_text, dep_lines=tuple(dep_lines)))

    return new_blocks, preserved_mor, dropped_mor, dropped_gra


def _split_target_surface(target_text: str, source_lines: list[str]) -> list[_SplitPiece] | None:
    target_tokens = target_text.split()
    target_lex = [_target_token_lex(token) for token in target_tokens]
    token_index = 0
    pieces: list[_SplitPiece] = []

    for source_line in source_lines:
        needed = _source_lexical_tokens(source_line)
        if not needed:
            pieces.append(
                _SplitPiece(
                    main_text=_normalize_source_for_insert(source_line),
                    mor_token_count=0,
                )
            )
            continue

        start_index = token_index
        matched: list[str] = []

        while token_index < len(target_tokens) and len(matched) < len(needed):
            current_lex = target_lex[token_index]
            if current_lex:
                matched.append(current_lex)
            token_index += 1

        if matched != needed:
            return None

        while token_index < len(target_tokens) and not target_lex[token_index]:
            token_index += 1

        span_tokens = target_tokens[start_index:token_index]
        pieces.append(
            _SplitPiece(
                main_text=" ".join(span_tokens),
                mor_token_count=sum(_target_token_has_mor(token) for token in span_tokens),
            )
        )

    if token_index < len(target_tokens):
        trailing = target_tokens[token_index:]
        trailing_has_lexical_content = any(target_lex[token_index:])
        if not trailing_has_lexical_content and pieces:
            last_piece = pieces[-1]
            pieces[-1] = _SplitPiece(
                main_text=" ".join([*last_piece.main_text.split(), *trailing]),
                mor_token_count=last_piece.mor_token_count
                + sum(_target_token_has_mor(token) for token in trailing),
            )

    return pieces


def _parse_target_chat(path: Path) -> tuple[list[str], list[ChatBlock], list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header_lines: list[str] = []
    blocks: list[ChatBlock] = []
    index = 0

    while index < len(lines) and not lines[index].startswith("*CHI:\t"):
        header_lines.append(lines[index])
        index += 1

    while index < len(lines):
        if lines[index] == "@End":
            return header_lines, blocks, ["@End"]

        if not lines[index].startswith("*CHI:\t"):
            raise ValueError(f"Unexpected line in {path}: {lines[index]!r}")

        main_text = lines[index].split("\t", 1)[1]
        index += 1
        dep_lines: list[str] = []

        while index < len(lines) and not lines[index].startswith("*CHI:\t") and lines[index] != "@End":
            dep_lines.append(lines[index])
            index += 1

        blocks.append(ChatBlock(main_text=main_text, dep_lines=tuple(dep_lines)))

    return header_lines, blocks, []


def _extract_chi_lines(path: Path) -> list[str]:
    return [
        line.split("\t", 1)[1]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("*CHI:\t")
    ]


def _normalize_source_for_insert(text: str) -> str:
    text = text.replace("( .)", "(.)")
    text = _SPACED_DOT_PAUSE_RE.sub(_collapse_spaced_dot_pause, text)
    text = text.replace("+/ .", "+/.")
    text = text.replace("! .", "!")
    text = re.sub(r"(?<!\S)&(?![=*A-Za-z_])", "and", text)
    text = _MONEY_RE.sub(lambda match: f"{_number_to_words(int(match.group(1)))} dollars", text)
    text = _DIGIT_RE.sub(lambda match: _number_to_words(int(match.group(1))), text)
    text = _NONFINAL_BANG_RE.sub("[!] ", text)
    return text


def _collapse_spaced_dot_pause(match: re.Match[str]) -> str:
    dot_count = match.group(0).count(".")
    return "(" + "." * dot_count + ")"


def _source_lexical_tokens(text: str) -> list[str]:
    tokens = _normalize_source_for_insert(text).replace(":", " ").split()
    lexical_tokens: list[str] = []

    for token in tokens:
        if token in _QUOTE_MARKERS or token in _PUNCT_TOKENS or token == "[!]":
            continue
        if token.startswith("&=") or token.startswith("&*"):
            lexical_tokens.append(token.lower())
            continue

        normalized = token.lower().replace("(", "").replace(")", "").replace(":", "")
        normalized = normalized.strip(".,!?;:\"")
        if normalized:
            lexical_tokens.append(normalized)

    return lexical_tokens


def _target_lexical_tokens(text: str) -> list[str]:
    return [token for token in (_target_token_lex(part) for part in text.split()) if token]


def _target_token_lex(token: str) -> str:
    if token in _QUOTE_MARKERS or token in _PUNCT_TOKENS or token == "[!]":
        return ""
    if token.startswith("&=") or token.startswith("&*"):
        return token.lower()
    return token.lower().replace("(", "").replace(")", "").replace(":", "").strip(".,!?;:\"")


def _target_token_has_mor(token: str) -> bool:
    if token in _NO_MOR_TOKENS:
        return False
    if token.startswith("&=") or token.startswith("&*"):
        return False
    return True


def _is_lexical_prefix(prefix: list[str], full: list[str]) -> bool:
    return len(prefix) <= len(full) and full[: len(prefix)] == prefix


def _number_to_words(value: int) -> str:
    ones = {
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
    tens = {
        20: "twenty",
        30: "thirty",
        40: "forty",
        50: "fifty",
        60: "sixty",
        70: "seventy",
        80: "eighty",
        90: "ninety",
    }

    if value < 20:
        return ones[value]
    if value < 100:
        tens_value = (value // 10) * 10
        ones_value = value % 10
        if not ones_value:
            return tens[tens_value]
        return f"{tens[tens_value]} {ones[ones_value]}"
    if value < 1000:
        hundreds_value = value // 100
        remainder = value % 100
        if not remainder:
            return f"{ones[hundreds_value]} hundred"
        return f"{ones[hundreds_value]} hundred {_number_to_words(remainder)}"
    return str(value)


def _restore_target_quote_markers(
    *,
    previous_source_text: str | None,
    source_text: str,
    next_source_text: str | None,
    target_text: str,
) -> str:
    desired_prefix = _desired_source_prefix(source_text, previous_source_text)
    desired_suffix = _desired_source_suffix(source_text, next_source_text)

    if source_text.strip() in {"+.", "+ ."}:
        return "+."

    body = target_text
    body = _strip_current_prefix_marker(body)
    body = _strip_current_special_suffix(body)

    if desired_prefix:
        body = f"{desired_prefix}{body.lstrip()}".rstrip()

    if desired_suffix:
        if desired_suffix.startswith("+"):
            body = re.sub(r"(?: \.| !| \?)$", "", body)
        body = _join_body_and_suffix(body, desired_suffix)

    return body


def _desired_source_prefix(source_text: str, previous_source_text: str | None) -> str:
    stripped = source_text.strip()

    if stripped.startswith('+" '):
        return '+" '
    if stripped.startswith("+ "):
        # Bare `+` is not a valid quote initiator in CHAT; normalize to `+"`.
        return '+" '
    if previous_source_text and _source_opens_quote(previous_source_text):
        return '+" '
    for marker in ("+, ", "++ ", "+^ "):
        if stripped.startswith(marker):
            return marker
    return ""


def _desired_source_suffix(source_text: str, next_source_text: str | None) -> str:
    compact = source_text.replace(" ", "").strip()
    stripped = source_text.strip()

    if compact in {"+.", "+."}:
        return "+."
    if re.search(r'\+\+?"/\.$', compact):
        return '+"/.'
    if compact.endswith("+/.") and next_source_text and _desired_source_prefix(next_source_text, source_text):
        return '+"/.'
    if compact.endswith('+".'):
        return '+".'
    if stripped.endswith("+ .") and stripped not in {"+ .", "+."}:
        return '+".'
    for marker in ("+//?", "+//.", "+/?", "+/.", "+..?", "+!?", "+..."):
        if compact.endswith(marker):
            return marker
    return ""


def _source_opens_quote(source_text: str) -> bool:
    compact = source_text.replace(" ", "").strip()
    if re.search(r'\+\+?"/\.$', compact):
        return True
    if compact.endswith("+/."):
        return True
    return False


def _strip_current_prefix_marker(text: str) -> str:
    stripped = text.lstrip()
    for marker in ('+" ', "+, ", "++ ", "+^ ", "+ "):
        if stripped.startswith(marker):
            return stripped[len(marker) :]
    return stripped


def _strip_current_special_suffix(text: str) -> str:
    stripped = text.rstrip()
    for marker in (' +"/.', ' +".', " +/.", " +//.", " +//?", " +/?", " +..?", " +!?", " +...", " + .", "+."):
        if stripped.endswith(marker):
            return stripped[: -len(marker)].rstrip()
    return stripped


def _restore_target_delimiters(*, source_text: str, target_text: str) -> str:
    source_normalized = _normalize_source_for_insert(source_text).strip()
    source_compact = source_normalized.replace(" ", "")

    if source_compact == "/.":
        return "/."
    if source_compact == "+.":
        return "+ ."

    target_tokens = target_text.split()
    while target_tokens and _is_closing_token(target_tokens[-1]):
        target_tokens.pop()
    target_body = " ".join(target_tokens).strip()

    quote_prefix = source_compact.endswith('+"/.') or source_compact.endswith('++"/.')
    quote_suffix = source_compact.endswith('+".') or source_compact.endswith("+.")
    should_add_leading_plus = (
        (source_normalized.startswith('+" ') or source_normalized.startswith("+ "))
        and not quote_prefix
        and not quote_suffix
    )

    if should_add_leading_plus and not target_body.startswith("+ "):
        target_body = f"+ {target_body}".strip()

    if quote_prefix:
        return _join_body_and_suffix(target_body, "+/.")
    if quote_suffix:
        return _join_body_and_suffix(target_body, "+ .")

    punctuation_suffix = _source_punctuation_suffix(source_normalized)
    if punctuation_suffix:
        return _join_body_and_suffix(target_body, punctuation_suffix)
    return target_body or source_normalized


def _is_closing_token(token: str) -> bool:
    return token in {".", "!", "?", ",", ";", ":", "[!]", "+", "/.", "+/.", "+//."}


def _join_body_and_suffix(body: str, suffix: str) -> str:
    if body:
        return f"{body} {suffix}"
    return suffix


def _source_punctuation_suffix(source_normalized: str) -> str:
    match = re.search(r"([.?!]+)$", source_normalized.replace(" ", ""))
    if not match:
        return ""
    punctuation = match.group(1)
    return " ".join(punctuation)


def _cleanup_chi_depfile_surface(text: str) -> str:
    cleaned = _normalize_source_for_insert(text)
    cleaned = _collapse_word_internal_colon_spacing(cleaned)
    cleaned = _normalize_parenthesized_i(cleaned)
    cleaned = _strip_quote_markers_for_depfile(cleaned)
    cleaned = _normalize_bare_ampersand_events(cleaned)
    cleaned = _normalize_simple_events(cleaned)
    cleaned = _strip_terminal_plus_dot(cleaned)
    cleaned = _collapse_redundant_final_delimiters(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _cleanup_mor_depfile_surface(text: str) -> str:
    cleaned = text.rstrip()
    for marker in (' +"/.', ' +".', " +.", " + .", "+."):
        if cleaned.endswith(marker):
            cleaned = cleaned[: -len(marker)].rstrip()
            if cleaned:
                cleaned = f"{cleaned} ."
            else:
                cleaned = "."
            break
    cleaned = re.sub(
        r"num\|(\d+)",
        lambda match: f"num|{_number_to_words(int(match.group(1)))}",
        cleaned,
    )
    return re.sub(r"\s+", " ", cleaned).strip()


def _restore_mor_delimiter_from_chi(mor_text: str, chi_text: str) -> str:
    desired = _desired_mor_delimiter_from_chi(chi_text)
    if not desired:
        return mor_text

    tokens = mor_text.split()
    while tokens and tokens[-1] in _MOR_DELIMITER_TOKENS:
        tokens.pop()

    if tokens:
        tokens.append(desired)
        return " ".join(tokens)
    return desired


def _desired_mor_delimiter_from_chi(chi_text: str) -> str:
    tokens = chi_text.split()
    if not tokens:
        return ""

    last = tokens[-1]
    if last in _MOR_DELIMITER_TOKENS:
        if last == "+.":
            return "."
        return last
    return ""


def _collapse_redundant_final_delimiters(text: str) -> str:
    tokens = text.split()
    if len(tokens) < 2:
        return text.strip()

    tail_start = len(tokens)
    while tail_start > 0 and tokens[tail_start - 1] in _MOR_DELIMITER_TOKENS:
        tail_start -= 1

    tail = tokens[tail_start:]
    if len(tail) <= 1:
        return text.strip()

    return " ".join([*tokens[:tail_start], tail[0]]).strip()


def _collapse_word_internal_colon_spacing(text: str) -> str:
    collapsed = text
    while True:
        updated = re.sub(r"(?<=\w)\s*:(?:\s*:)+(?=\w)", ":", collapsed)
        updated = re.sub(r"(?<=\w)\s+:(?=\w)", ":", updated)
        updated = re.sub(r"(?<=\w):\s+(?=\w)", ":", updated)
        if updated == collapsed:
            return updated
        collapsed = updated


def _is_event_only_chi_line(text: str) -> bool:
    return re.match(r"^&=[A-Za-z][A-Za-z_:-]* [.!?]$", text.strip()) is not None


def _starts_with_bare_equals_event(text: str) -> bool:
    return re.match(r"^=[A-Za-z][A-Za-z_-]*\b", text.strip()) is not None


def _normalize_bare_ampersand_events(text: str) -> str:
    normalized = re.sub(
        r"(?<!\S)&([A-Za-z][A-Za-z_-]*)\s+voice(?=\s+[.!?]$)",
        r"&=\1_voice",
        text,
    )
    return re.sub(
        r"(?<!\S)&([A-Za-z][A-Za-z_-]*)(?=\s+[.!?]$)",
        r"&=\1",
        normalized,
    )


def _normalize_simple_events(text: str) -> str:
    normalized = re.sub(r"\[=([A-Za-z][A-Za-z_-]*)\]", r"&=\1", text)
    return re.sub(r"(?<![&\w\[])=([A-Za-z][A-Za-z_-]*)\b", r"&=\1", normalized)


def _strip_terminal_plus_dot(text: str) -> str:
    stripped = text.rstrip()
    updated = re.sub(r"\s*\+\s*\.$", "", stripped)
    if updated != stripped:
        return (updated.rstrip() + " .").strip()
    return stripped


def _drop_leading_equals_event_mor_token(text: str) -> str:
    tokens = text.split()
    if tokens and re.search(r"\|=[^\s]+", tokens[0]):
        return " ".join(tokens[1:])
    return text


def _normalize_parenthesized_i(text: str) -> str:
    normalized = re.sub(r"\(i\)(?=[A-Za-z])", "i", text)
    normalized = re.sub(r"(^|\s)\(i\)(?=\s|$)", r"\1", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _strip_quote_markers_for_depfile(text: str) -> str:
    stripped = text.strip()

    for marker in ('+" ', "+ "):
        if stripped.startswith(marker):
            stripped = stripped[len(marker) :].lstrip()
            break

    for marker in (' +"/.', ' +".'):
        if stripped.endswith(marker):
            stripped = stripped[: -len(marker)].rstrip()
            if stripped:
                stripped = f"{stripped} ."
            else:
                stripped = "."
            break

    return stripped
