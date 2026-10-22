"""Restore tagged generated CHAT against an authoritative pre-annotation copy.

The reference supplies all speech, punctuation, headers and utterance boundaries.
Existing annotation can be transferred only to unique, context-stable lexical
anchors. Explicit review decisions may remove labels; new labels are never made.
"""
from __future__ import annotations

from array import array
from dataclasses import dataclass
import re

from .generated_chat import _number_words

_BRACKET = re.compile(r"\[[^\[\]]*\]")
_WORD = re.compile(r"(?:\([A-Za-z]+\)|[^\W\d_])(?:[^\W\d_]|['’:](?=[^\W\d_])|\([A-Za-z]+\))*|\d+", re.UNICODE)
_NON_SPEECH = re.compile(r"&\*[A-Z][A-Z0-9]*:[^\s.!?,;]+|&=[^\s.!?,;]+")
_MORPHOSYNTAX = re.compile(r"^%(?:u?mor|u?gra):", re.MULTILINE)


def _priority(code: str) -> int:
    if re.match(r"\[(?:::?|=)\s", code):
        return 0
    if code.startswith('[*'):
        return 1
    if re.fullmatch(r"\[/{1,3}\]", code):
        return 3
    return 2


@dataclass(frozen=True)
class Word:
    value: str
    start: int
    end: int
    utterance: int


def _value(word: str) -> str:
    if word.isdigit():
        return _number_words(word).replace(" ", "")
    return re.sub(r"[()'’:]+", "", word).casefold()


def _scan(text: str) -> tuple[list[Word], list[dict]]:
    words: list[Word] = []
    codes: list[dict] = []
    offset, utterance = 0, -1
    for line in text.splitlines(keepends=True):
        if line.startswith("*CHI:"):
            utterance += 1
            beginning = line.index(":") + 1
            masked = list(line)
            brackets = list(_BRACKET.finditer(line, beginning))
            non_speech = list(_NON_SPEECH.finditer(line, beginning))
            for match in brackets + non_speech:
                masked[match.start():match.end()] = " " * (match.end() - match.start())
            for match in _WORD.finditer("".join(masked), beginning):
                words.append(Word(_value(match.group()), offset + match.start(),
                                  offset + match.end(), utterance))
            for match in brackets:
                preceding = [i for i, word in enumerate(words)
                             if word.utterance == utterance and word.end <= offset + match.start()]
                anchor = preceding[-1] if preceding else None
                # A code following an interposed speaker/nonverbal event cannot
                # be attributed to the earlier CHI word across that event.
                intervening_event = any(event.end() <= match.start() and
                    (anchor is None or offset + event.start() >= words[anchor].end)
                    for event in non_speech)
                codes.append({"index": len(codes), "code": match.group(), "anchor": anchor,
                              "event_intervenes": intervening_event,
                              "utterance": utterance, "start": offset + match.start(),
                              "line": line.rstrip("\r\n")})
        offset += len(line)
    return words, codes


def _unique_alignment(source: list[Word], reference: list[Word]) -> dict[int, int]:
    """Keep only word pairs forced in every optimal order-preserving alignment."""
    n, m = len(source), len(reference)
    if n * m > 4_000_000 or max(n, m) > 65534:
        raise ValueError("Transcript too large for conservative annotation recovery")
    prefix = [array("H", [0]) * (m + 1) for _ in range(n + 1)]
    suffix = [array("H", [0]) * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            prefix[i + 1][j + 1] = (prefix[i][j] + 1 if source[i].value == reference[j].value
                                    else max(prefix[i][j + 1], prefix[i + 1][j]))
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            suffix[i][j] = (suffix[i + 1][j + 1] + 1 if source[i].value == reference[j].value
                            else max(suffix[i + 1][j], suffix[i][j + 1]))
    optimum = prefix[n][m]
    mapping: dict[int, int] = {}
    for i, word in enumerate(source):
        candidates = [j for j, target in enumerate(reference)
                      if word.value == target.value and prefix[i][j] + 1 + suffix[i + 1][j + 1] == optimum]
        optional = any(prefix[i][j] + suffix[i + 1][j] == optimum for j in range(m + 1))
        if len(candidates) == 1 and not optional:
            mapping[i] = candidates[0]
    return mapping


def restore_annotations_to_reference(
    reference: str, annotated: str, *, decisions: dict[int, dict] | None = None,
) -> tuple[str, dict]:
    """Rebuild a derived copy with reference speech and conservatively kept tags.

    Decisions are indexed by CHI bracket-code occurrence, beginning at zero.
    A reviewed decision requires a reason. Review can confirm a label when a
    neighboring word changes, but cannot force an ambiguous word attachment or
    introduce a new label. Dependent tiers must agree exactly;
    morphological tiers require a separate re-annotation workflow.
    """
    if _MORPHOSYNTAX.search(reference) or _MORPHOSYNTAX.search(annotated):
        raise ValueError("Morphosyntactic tiers require separate re-annotation")
    dependent = lambda text: [line for line in text.splitlines() if line.startswith("%")]
    if dependent(reference) != dependent(annotated):
        raise ValueError("Dependent tiers differ; refusing to discard or realign them")
    for text in (reference, annotated):
        if not any(line.startswith("*CHI:") for line in text.splitlines()):
            raise ValueError("Reference and annotated text must both contain CHI speech")
        lines = text.splitlines()
        if any(line.startswith("*CHI:") and i + 1 < len(lines) and
               lines[i + 1][:1].isspace() and lines[i + 1].strip()
               for i, line in enumerate(lines)):
            raise ValueError("Wrapped speech needs explicit unfolding before recovery")
    source_words, codes = _scan(annotated)
    target_words, target_codes = _scan(reference)
    mapping = _unique_alignment(source_words, target_words)
    existing = {}
    for code in target_codes:
        existing.setdefault(code['anchor'], []).append(code['code'])
    consumed = {}
    insertions: dict[int, list[str]] = {}
    records = []
    decisions = decisions or {}
    if set(decisions) - {code['index'] for code in codes}:
        raise ValueError("Review decision refers to an absent annotation")
    for code in codes:
        record = dict(code)
        anchor = code['anchor']
        record['source_word'] = source_words[anchor].value if anchor is not None else None
        decision = decisions.get(code['index'], {})
        if decision and (decision.get('action') not in {'remove', 'retain'} or not decision.get('reason')):
            raise ValueError("Reviewed decisions require retain/remove and a reason")
        if decision:
            record['review'] = decision
        if decision.get('action') == 'remove':
            if not decision.get('reason'):
                raise ValueError("Annotation removal requires a recorded reason")
            if anchor in mapping and code['code'] in existing.get(mapping[anchor], []):
                raise ValueError("Cannot remove an annotation present in the authoritative reference")
            record.update(status='removed', reason=decision['reason'], review=decision)
        elif code['event_intervenes'] or anchor is None:
            record.update(status='removed', reason='no_CHI_word_anchor')
        elif anchor not in mapping:
            record.update(status='removed', reason='changed_or_ambiguous_word_anchor')
        else:
            target = mapping[anchor]
            record.update(reference_word=target_words[target].value,
                          reference_utterance=target_words[target].utterance)
            before_ok = anchor == 0 or (anchor - 1 in mapping and mapping[anchor - 1] == target - 1)
            after_ok = anchor + 1 == len(source_words) or (anchor + 1 in mapping and mapping[anchor + 1] == target + 1)
            if not (before_ok and after_ok) and decision.get('action') != 'retain':
                record.update(status='removed', reason='changed_or_ambiguous_adjacent_context')
            elif re.search(r"[<>]", annotated[source_words[anchor].end:code['start']]):
                record.update(status='removed', reason='scope_requires_explicit_review')
            else:
                available = existing.get(target, [])
                used = consumed.setdefault(target, [])
                if available.count(code['code']) > used.count(code['code']):
                    used.append(code['code'])
                    record.update(status='already_in_reference', reason='reference_annotation_preserved')
                else:
                    # Reconstructions precede errors; errors on retraced words
                    # precede [/]. Never move the reference's own codes.
                    position = target_words[target].end
                    for match in _BRACKET.finditer(reference, position):
                        gap = reference[position:match.start()]
                        if gap.strip() or '\n' in gap:
                            break
                        if _priority(match.group()) > _priority(code['code']):
                            break
                        position = match.end()
                    insertions.setdefault(position, []).append(code['code'])
                    reason = ('reviewed_context_with_unique_word_anchor' if decision.get('action') == 'retain'
                              else 'unique_anchor_and_unchanged_adjacent_words')
                    record.update(status='retained', reason=reason,
                                  reference_position=position)
        records.append(record)
    result, cursor = [], 0
    added = []
    for position, values in sorted(insertions.items()):
        values = sorted(values, key=_priority)
        addition = "".join(" " + value for value in values)
        result.extend([reference[cursor:position], addition])
        added.append({'reference_position': position, 'text': addition})
        cursor = position
    result.append(reference[cursor:])
    return "".join(result), {
        'source_annotations': len(codes), 'annotations': records,
        'insertions': added, 'source_words': len(source_words), 'reference_words': len(target_words),
        'reference_supplies_all_speech_and_boundaries': True,
        'new_error_labels_inferred': False,
    }
