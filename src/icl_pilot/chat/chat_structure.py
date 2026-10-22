"""Conservative structural formatting of generated, error-tagged CHAT copies.

This keeps main-tier boundaries and every bracketed annotation. It does not
reconstruct missing speech, repair grammar, or reuse stale morphological tiers.
"""
from __future__ import annotations

import re

from .generated_chat import (
    _CHAT_PUNCT_RE,
    _correct_generated_types,
    _normalize_generated_notation,
    _protect_notation,
    _repair_quote_layout,
    _restore_notation,
)

_BRACKET = re.compile(r"\[[^\[\]]*\]")
_LEXICAL = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*|\d+", re.UNICODE)
_MORPHOSYNTAX = re.compile(r"^%(?:u?mor|u?gra):", re.MULTILINE)


def _shield_annotations(body: str) -> tuple[str, dict[str, str]]:
    annotations: dict[str, str] = {}

    def replace(match):
        key = "\uf000" + chr(0xF100 + len(annotations)) + "\uf001"
        if key in body:
            raise ValueError("Reserved annotation placeholder occurs in the input")
        annotations[key] = match.group()
        return key

    return _BRACKET.sub(replace, body), annotations


def _normalized(body: str) -> tuple[str, list[dict[str, str]]]:
    shielded, annotations = _shield_annotations(body)
    normalized, changes = _normalize_generated_notation(shielded)
    return _restore_notation(normalized, annotations), changes


def _signature(body: str) -> list[str]:
    normalized, _ = _normalized(body)
    return _LEXICAL.findall(_BRACKET.sub(" ", normalized))


def annotation_anchors(body: str) -> list[tuple[str, list[str]]]:
    """Identify each exact code by the normalized lexical prefix preceding it."""
    return [(match.group(), _signature(body[:match.start()]))
            for match in _BRACKET.finditer(body)]


def generated_surface_tokens(text: str) -> list[str]:
    """Fingerprint generated CHI speech independently of added inline codes.

    Compare case-insensitively, expand CHAT word shortenings, and ignore written
    apostrophe variants. Retain repeated/retraced words and grammatical forms.
    This checks textual preservation, not annotation correctness.
    """
    tokens: list[str] = []
    for line in text.splitlines():
        if not line.startswith("*CHI:"):
            continue
        body, _ = _normalized(line.split(":", 1)[1])
        body = _BRACKET.sub(" ", body)
        body = re.sub(r"(?<=\w)\(([A-Za-z]+)\)", r"\1", body)
        body = re.sub(r"\(([A-Za-z]+)\)(?=[A-Za-z])", r"\1", body)
        tokens.extend(token.casefold().replace("’", "").replace("'", "")
                      for token in _LEXICAL.findall(body))
    return tokens


def _format_body(body: str) -> tuple[str, list[dict[str, str]]]:
    shielded, annotations = _shield_annotations(body)
    normalized, changes = _normalize_generated_notation(shielded)
    # Pad opaque codes without changing their contents. CHAT requires separation
    # between codes and words. Protect pause/lengthening/linker notation too.
    for key in annotations:
        normalized = normalized.replace(key, " " + key + " ")
    guarded, notation = _protect_notation(normalized)
    guarded = _CHAT_PUNCT_RE.sub(r" \1 ", guarded)
    guarded = re.sub(r",(?:\s*,)+", ",", guarded)
    guarded = re.sub(r"\s+", " ", guarded).strip()
    formatted = _restore_notation(_restore_notation(guarded, notation), annotations)
    formatted = re.sub(r"“\s+", "“", re.sub(r"\s+”", "”", formatted))
    # A retracing/repetition code attaches to the preceding word or scope;
    # a comma between that word and [/] creates an invalid attachment.
    formatted = re.sub(r",\s*(\[/{1,3}\])", r"\1", formatted)
    # Existing linkers must remain intact; add only a missing terminal period.
    if formatted and formatted[-1] not in ".!?":
        formatted += " ."
    return formatted, changes


def format_generated_annotated_chat(text: str) -> tuple[str, list[dict]]:
    """Return a derived structural repair, refusing annotation or lexical drift.

    Only generated CHAT with the supported provenance headers is accepted.
    Normalization of explicit numbers/prose notation follows the declared CSV
    export convention. No main tier is added, removed, split, or merged. Every
    bracketed code and its normalized lexical anchor must remain identical.
    If morphosyntactic tiers exist, body changes require fresh annotation and
    are refused here. The caller must still run Chatter on the result.
    """
    corrected = _correct_generated_types(text)
    lines = corrected.splitlines()
    changes: list[dict] = []
    if corrected != text:
        changes.append({"kind": "generated_types_header"})
    indices = [i for i, line in enumerate(lines) if line.startswith("*CHI:")]
    if not indices:
        raise ValueError("No CHI main tiers")
    # This function formats existing single-line generated main tiers. Wrapped
    # speech requires a separate, explicit unfolding operation.
    for i in indices:
        if i + 1 < len(lines) and lines[i + 1][:1].isspace() and lines[i + 1].strip():
            raise ValueError("Wrapped main tier requires explicit review")
    before = [lines[i].split(":", 1)[1].strip() for i in indices]
    formatted = []
    for i, body in zip(indices, before):
        after, operations = _format_body(body)
        if body != after:
            changes.append({"kind": "main_tier_formatting", "line": i + 1,
                            "before": body, "after": after, "operations": operations})
        formatted.append(after)
    # Quotation links may refer only to adjacent CHI tiers, not across another
    # speaker, dependent tier, or intervening story/header boundary.
    for start in range(len(indices)):
        if start and indices[start] == indices[start - 1] + 1:
            continue
        end = start + 1
        while end < len(indices) and indices[end] == indices[end - 1] + 1:
            end += 1
        repaired = _repair_quote_layout(formatted[start:end])
        if repaired != formatted[start:end]:
            changes.append({"kind": "quotation_link_layout", "line": indices[start] + 1,
                            "before": formatted[start:end], "after": repaired})
        formatted[start:end] = repaired
    for i, original, after in zip(indices, before, formatted):
        if _signature(original) != _signature(after):
            raise ValueError("Structural repair would change lexical content at line " + str(i + 1))
        if annotation_anchors(original) != annotation_anchors(after):
            raise ValueError("Structural repair would change an annotation or its anchor at line " + str(i + 1))
        lines[i] = "*CHI:\t" + after
    if _MORPHOSYNTAX.search(text) and before != formatted:
        raise ValueError("Main-tier changes would require fresh morphosyntactic annotation")
    return "\n".join(lines) + "\n", changes
