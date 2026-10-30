"""Per-utterance preservation receipts, independent of retained CHAT snapshots."""
from __future__ import annotations

import hashlib
import json


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def text_fingerprint(text, mode="annotation"):
    """Ignore only documented additions; retain speaker/order/boundary identity.

    speech: before/after shared cleaning, with inline transcription annotations
    removed from the comparison. annotation: TalkTag may add its inline codes.
    exact: UD must retain every main-tier code as well as the spoken text.
    """
    from .annotation_preparation import _records, _speech_tokens
    from .annotation import _surface
    if mode not in {"speech", "annotation", "exact"}:
        raise ValueError("Unknown CHAT preservation policy")
    utterances, headers, dependent = [], [], []
    for record in _records(text):
        value = record["text"].replace("\r\n", "\n")
        if value.startswith("*"):
            speaker, body = value.split(":", 1)
            content = (_speech_tokens(body) if mode == "speech" else _surface(body)
                       if mode == "annotation" else " ".join(body.split()))
            utterances.append([speaker, _hash(content)])
        elif value.startswith("@"):
            headers.append(value.rstrip("\n"))
        elif value.startswith("%") and value.split(":", 1)[0] not in {"%mor", "%gra", "%umor", "%ugra"}:
            dependent.append(" ".join(value.split()))
    return {"schema_version": 1, "mode": mode, "utterances": utterances,
            "main_tiers": len(utterances), "chi_main_tiers": sum(speaker == "*CHI" for speaker, _ in utterances),
            "utterances_sha256": _hash(utterances), "headers_sha256": _hash(headers),
            "other_dependent_sha256": _hash(dependent)}


def require_text_preserved(expected, text, *, compare_dependent=True):
    actual = text_fingerprint(text, expected["mode"])
    fields = ["utterances", "headers_sha256"] + (["other_dependent_sha256"] if compare_dependent else [])
    if any(actual[key] != expected[key] for key in fields):
        changed = [i + 1 for i in range(max(len(expected["utterances"]), len(actual["utterances"])))
                   if expected["utterances"][i:i + 1] != actual["utterances"][i:i + 1]]
        raise ValueError("Protected CHAT text/metadata changed; utterance indices: " + str(changed[:20]))
    return actual
