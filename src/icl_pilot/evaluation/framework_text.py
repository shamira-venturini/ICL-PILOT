"""Common analysis-only text preparation, without correcting spoken forms."""
from __future__ import annotations

import re

from ..chat.text_cleaning import prepare_extracted_story

TEXT_POLICY = "framework-chat-text-v3"


def normalize_framework_text(text):
    """Delete CHAT notation; preserve grammar, repetitions, case and fragments.

    The same function is applied to ENNI, Gillam and synthetic exports. Source
    files and Stage 1 parent strings are never edited. Parenthesized partial
    words remain literal, so (be)cause is not reconstructed as because.
    """
    text = str(text)
    text = re.sub(r"\x15[^\x15]*\x15", " ", text)
    text = prepare_extracted_story(text, allow_empty=True).text
    text = re.sub(r"&\*[^\s]+", " ", text)
    text = re.sub(r"&[+-]([A-Za-z:']+)", r"\1", text)
    text = text.replace("&~", "")
    text = re.sub(r"<([^>]*)>", r"\1", text)
    text = re.sub(r"\b([A-Za-z']+)@[A-Za-z]+\b", r"\1", text)
    text = re.sub(r"\b0[A-Za-z']+", " ", text)
    text = re.sub(r"(?<!\S)\++(?:[\"/.!?]+)?", " ", text)
    text = re.sub(r"(?<!\S):(?!\S)", " ", text)
    text = re.sub(r"\s+([,.;?!])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()
