"""Deletion-only text preparation for CHAT-to-CSV extraction.

The frozen extraction and linguistic-analysis inputs are never rewritten.
Unknown speech is deleted, never guessed, corrected or replaced by a word.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re


TEXT_CLEANING_POLICY = "chat-story-text-v3"
_MARKERS = (
    ("annotator_annotations", re.compile(r"\[[^\]]*\]")),
    ("pause_markers", re.compile(r"\((?:\.{1,3}|\d+(?:\.\d+)?)\)")),
    ("unintelligible_markers", re.compile(r"(?<!\w)(?:xxx|yyy|www)(?!\w)", re.IGNORECASE)),
    ("non_speech_codes", re.compile(r"&=[\w:-]+")),
)


@dataclass(frozen=True)
class PreparedStoryText:
    text: str
    utterances: tuple[str, ...]
    removed: dict[str, int]
    original_utterances: int
    dropped_utterances: int


def prepare_extracted_story(text: str, utterances_json: str | None = None, *,
                            allow_empty: bool = False) -> PreparedStoryText:
    """Remove markers while preserving every remaining spoken form in order.

    Parenthesized word fragments, repetitions and nonstandard grammar remain
    literal. Wholly non-speech utterances are dropped with their punctuation.
    CSVs without an utterance array are treated as one continuous narrative.
    """
    if not isinstance(text, str) or (not text.strip() and not allow_empty):
        raise ValueError("Story text must contain nonempty text before preparation.")
    units = [text] if utterances_json is None else json.loads(utterances_json)
    if (not isinstance(units, list) or not all(isinstance(unit, str) for unit in units)
            or " ".join(units) != text):
        raise ValueError("Story utterance array disagrees with its story text.")
    removed = {name: 0 for name, _pattern in _MARKERS}
    prepared = []
    for unit in units:
        for name, pattern in _MARKERS:
            unit, count = pattern.subn(" ", unit)
            removed[name] += count
        unit = re.sub(r"\s+", " ", unit).strip()
        if any(character.isalnum() for character in unit):
            prepared.append(unit)
    if not prepared and not allow_empty:
        raise ValueError("Story has no spoken text after CHAT marker removal; no words are imputed.")
    return PreparedStoryText(" ".join(prepared), tuple(prepared), removed,
                            len(units), len(units) - len(prepared))


def unwanted_notation(text: str) -> list[str]:
    """Validation only: identify agreed notation without transforming text."""
    return [name for name, pattern in _MARKERS if pattern.search(text)]
