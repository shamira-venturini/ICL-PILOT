from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .generated_chat import _chat_space_punctuation, _normalize_surface


@dataclass(frozen=True)
class BatchalignMainlineRepairResult:
    files_touched: int
    chi_lines_touched: int
    quote_lines_fixed: int


def repair_batchalign_mainlines(root_dir: str | Path) -> BatchalignMainlineRepairResult:
    root = Path(root_dir).expanduser().resolve()
    files_touched = 0
    chi_lines_touched = 0
    quote_lines_fixed = 0

    for path in root.rglob("*.cha"):
        lines = path.read_text(encoding="utf-8").splitlines()
        changed = False
        new_lines: list[str] = []

        for line in lines:
            if line.startswith("*CHI:\t"):
                prefix, utterance = line.split("\t", 1)
                original = utterance
                utterance = utterance.replace("“", '"').replace("”", '"')
                if '"' in utterance:
                    quote_lines_fixed += 1
                    utterance = utterance.replace('"', "")
                utterance = _chat_space_punctuation(_normalize_surface(utterance))
                line = f"{prefix}\t{utterance}"
                if utterance != original:
                    changed = True
                    chi_lines_touched += 1
            new_lines.append(line)

        if changed:
            path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            files_touched += 1

    return BatchalignMainlineRepairResult(
        files_touched=files_touched,
        chi_lines_touched=chi_lines_touched,
        quote_lines_fixed=quote_lines_fixed,
    )
