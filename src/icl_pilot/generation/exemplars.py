"""Validate and select text already cleaned by CHAT-to-CSV extraction."""
from __future__ import annotations

from dataclasses import dataclass
import json

from ..chat.text_cleaning import TEXT_CLEANING_POLICY, unwanted_notation

EXEMPLAR_POLICY = TEXT_CLEANING_POLICY


@dataclass(frozen=True)
class ValidatedExemplar:
    text: str
    utterances: tuple[str, ...]


def validate_exemplar(text: str, utterances_json: str | None = None) -> ValidatedExemplar:
    """Return literal input; reject unfinished extraction instead of cleaning it."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Exemplar must contain nonempty text before generation.")
    units = [text] if utterances_json is None else json.loads(utterances_json)
    if (not isinstance(units, list) or not all(isinstance(unit, str) for unit in units)
            or " ".join(units) != text):
        raise ValueError("Exemplar utterance array disagrees with its story text.")
    unwanted = unwanted_notation(text)
    if unwanted:
        raise ValueError("Exemplar CSV still contains " + ", ".join(unwanted)
                         + "; clean it during CHAT-to-CSV extraction before generation.")
    if any(not any(character.isalnum() for character in unit) for unit in units):
        raise ValueError("Exemplar CSV contains an empty/non-speech utterance; complete extraction cleanup first.")
    if any(unit != " ".join(unit.split()) for unit in units):
        raise ValueError("Exemplar CSV spacing is not prepared; complete extraction cleanup first.")
    return ValidatedExemplar(text, tuple(units))
