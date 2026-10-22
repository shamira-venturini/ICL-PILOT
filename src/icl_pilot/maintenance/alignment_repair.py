from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import re


@dataclass
class RepairCandidate:
    path: Path
    speaker_line: int
    umor_index: int
    ugra_index: int
    mor_index: int
    gra_index: int
    umor_len: int
    mor_len: int
    lemma_jaccard: float
    reason: str


_TIMESTAMP_RE = re.compile(r"\x15[^\x15]*\x15$")
_TRAILING_ANN_RE = re.compile(r"(\s*(\[[^\]]+\]|&=\S+))*\s*$")


def _tier_tokens(line: str) -> list[str]:
    body = line.split("\t", 1)[1] if "\t" in line else line.split(":", 1)[1]
    return [token for token in body.split() if token not in {".", "?", "!"}]


def _tier_lemmas(tokens: Iterable[str]) -> list[str]:
    lemmas: list[str] = []
    for token in tokens:
        if "|" not in token:
            lemmas.append(token)
            continue
        lemma = token.split("|", 1)[1]
        lemma = lemma.split("-", 1)[0]
        lemma = lemma.split("~", 1)[0]
        lemmas.append(lemma)
    return lemmas


def _speaker_body(line: str) -> str:
    body = line.split("\t", 1)[1] if "\t" in line else line.split(":", 1)[1]
    body = _TIMESTAMP_RE.sub("", body).strip()
    body = _TRAILING_ANN_RE.sub("", body).strip()
    return body


def _ending_markers(text: str) -> tuple[bool, str]:
    has_quote = '+"/' in text
    without_quote = text.replace('+"/', "").rstrip()
    punct = without_quote[-1] if without_quote and without_quote[-1] in ".?!" else ""
    return has_quote, punct


def _lemma_jaccard(umor_line: str, mor_line: str) -> tuple[int, int, float]:
    umor_tokens = _tier_tokens(umor_line)
    mor_tokens = _tier_tokens(mor_line)
    umor_lemmas = set(_tier_lemmas(umor_tokens))
    mor_lemmas = set(_tier_lemmas(mor_tokens))
    intersection = len(umor_lemmas & mor_lemmas)
    union = len(umor_lemmas | mor_lemmas) or 1
    return len(umor_tokens), len(mor_tokens), intersection / union


def _is_suspicious_alignment(umor_line: str, mor_line: str) -> tuple[bool, int, int, float]:
    umor_len, mor_len, lemma_jaccard = _lemma_jaccard(umor_line, mor_line)
    min_len = min(umor_len, mor_len)
    flag = min_len >= 4 and (
        lemma_jaccard <= 0.25
        or (umor_len != mor_len and lemma_jaccard < 0.15)
    )
    return flag, umor_len, mor_len, lemma_jaccard


def _has_delimiter_mismatch(speaker_line: str, umor_line: str, mor_line: str) -> bool:
    speaker_markers = _ending_markers(_speaker_body(speaker_line))
    umor_markers = _ending_markers(_speaker_body(umor_line))
    mor_markers = _ending_markers(_speaker_body(mor_line))
    return speaker_markers == umor_markers and speaker_markers != mor_markers


def _rewrite_gra_from_ugra(ugra_line: str) -> str:
    prefix = "%gra:\t" if "\t" in ugra_line else "%gra:"
    body = ugra_line.split("\t", 1)[1] if "\t" in ugra_line else ugra_line.split(":", 1)[1]
    converted: list[str] = []
    for token in body.split():
        parts = token.split("|")
        if len(parts) == 3 and parts[2] == "ROOT":
            parts[1] = "0"
            token = "|".join(parts)
        converted.append(token)
    return prefix + " ".join(converted)


def find_repair_candidates(root_dir: str) -> list[RepairCandidate]:
    root = Path(root_dir).expanduser().resolve()
    candidates: list[RepairCandidate] = []

    for path in sorted(root.rglob("*.cha")):
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        i = 0
        while i < len(lines):
            if not lines[i].startswith("*"):
                i += 1
                continue

            j = i + 1
            umor = ugra = mor = gra = None
            while j < len(lines) and not lines[j].startswith("*") and not lines[j].startswith("@"):
                if lines[j].startswith("%umor:"):
                    umor = j
                elif lines[j].startswith("%ugra:"):
                    ugra = j
                elif lines[j].startswith("%mor:"):
                    mor = j
                elif lines[j].startswith("%gra:"):
                    gra = j
                j += 1

            if None not in (umor, ugra, mor, gra):
                suspicious, umor_len, mor_len, lemma_jaccard = _is_suspicious_alignment(
                    lines[umor], lines[mor]
                )
                if suspicious:
                    candidates.append(
                        RepairCandidate(
                            path=path,
                            speaker_line=i + 1,
                            umor_index=umor,
                            ugra_index=ugra,
                            mor_index=mor,
                            gra_index=gra,
                            umor_len=umor_len,
                            mor_len=mor_len,
                            lemma_jaccard=lemma_jaccard,
                            reason="content-mismatch",
                        )
                    )
                elif _has_delimiter_mismatch(lines[i], lines[umor], lines[mor]):
                    candidates.append(
                        RepairCandidate(
                            path=path,
                            speaker_line=i + 1,
                            umor_index=umor,
                            ugra_index=ugra,
                            mor_index=mor,
                            gra_index=gra,
                            umor_len=umor_len,
                            mor_len=mor_len,
                            lemma_jaccard=lemma_jaccard,
                            reason="delimiter-mismatch",
                        )
                    )

            i = j

    return candidates


def repair_alignments(root_dir: str, dry_run: bool = False) -> int:
    candidates = find_repair_candidates(root_dir)
    if not candidates:
        print("No suspicious %mor/%gra disalignments found.")
        return 0

    by_file: dict[Path, list[RepairCandidate]] = {}
    for candidate in candidates:
        by_file.setdefault(candidate.path, []).append(candidate)

    for path, file_candidates in by_file.items():
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        file_candidates.sort(key=lambda item: item.mor_index)
        for candidate in file_candidates:
            lines[candidate.mor_index] = lines[candidate.umor_index].replace("%umor:", "%mor:", 1)
            lines[candidate.gra_index] = _rewrite_gra_from_ugra(lines[candidate.ugra_index])
        if not dry_run:
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        f"{'Would repair' if dry_run else 'Repaired'} {len(candidates)} speaker blocks "
        f"across {len(by_file)} files under {Path(root_dir).resolve()}"
    )
    for candidate in candidates[:40]:
        print(
            f"- {candidate.path}:{candidate.speaker_line} "
            f"(umor={candidate.umor_len}, mor={candidate.mor_len}, "
            f"jaccard={candidate.lemma_jaccard:.3f}, reason={candidate.reason})"
        )
    if len(candidates) > 40:
        print(f"... and {len(candidates) - 40} more")
    return 0
