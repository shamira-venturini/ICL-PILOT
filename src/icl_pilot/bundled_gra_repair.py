from __future__ import annotations

import argparse
from pathlib import Path

PUNCT_TOKENS = {".", "!", "?"}
VERB_PREFIXES = ("verb|", "aux|")


def _tier_body(line: str) -> str:
    if "\t" in line:
        return line.split("\t", 1)[1]
    return line.split(":", 1)[1]


def _gra_prefix(line: str) -> str:
    return "%gra:\t" if "\t" in line else "%gra:"


def _gra_from_ugra(ugra_line: str) -> str:
    converted: list[str] = []
    for token in _tier_body(ugra_line).split():
        parts = token.split("|")
        if len(parts) == 3 and parts[2] == "ROOT":
            parts[1] = "0"
            token = "|".join(parts)
        converted.append(token)
    return _gra_prefix(ugra_line) + " ".join(converted)


def _mor_from_umor(umor_line: str) -> str:
    if "\t" in umor_line:
        return umor_line.replace("%umor:\t", "%mor:\t", 1)
    return umor_line.replace("%umor:", "%mor:", 1)


def _fallback_gra_from_mor(mor_line: str) -> str:
    tokens = _tier_body(mor_line).split()
    non_punct = [index for index, token in enumerate(tokens, start=1) if token not in PUNCT_TOKENS]
    if not non_punct:
        return _gra_prefix(mor_line)

    root_index = next(
        (index for index, token in enumerate(tokens, start=1) if token.startswith(VERB_PREFIXES)),
        non_punct[0],
    )

    converted: list[str] = []
    for index, token in enumerate(tokens, start=1):
        if token in PUNCT_TOKENS:
            converted.append(f"{index}|{root_index}|PUNCT")
        elif index == root_index:
            converted.append(f"{index}|0|ROOT")
        elif token.startswith("intj|"):
            converted.append(f"{index}|{root_index}|DISCOURSE")
        else:
            converted.append(f"{index}|{root_index}|DEP")
    return _gra_prefix(mor_line) + " ".join(converted)


def repair_missing_gra(root_dir: str | Path) -> tuple[int, int, int, int]:
    root = Path(root_dir).expanduser().resolve()
    changed_files = 0
    inserted_from_umor = 0
    inserted_from_ugra = 0
    inserted_from_mor = 0

    for path in sorted(root.rglob("*.cha")):
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        new_lines: list[str] = []
        changed = False
        i = 0

        while i < len(lines):
            line = lines[i]
            if not line.startswith("*"):
                new_lines.append(line)
                i += 1
                continue

            block = [line]
            i += 1
            while i < len(lines) and not lines[i].startswith("*") and not lines[i].startswith("@"):
                block.append(lines[i])
                i += 1

            umor_index = next((idx for idx, item in enumerate(block) if item.startswith("%umor:")), None)
            mor_index = next((idx for idx, item in enumerate(block) if item.startswith("%mor:")), None)
            gra_index = next((idx for idx, item in enumerate(block) if item.startswith("%gra:")), None)
            ugra_index = next((idx for idx, item in enumerate(block) if item.startswith("%ugra:")), None)

            if mor_index is None and umor_index is not None:
                insert_at = ugra_index + 1 if ugra_index is not None and ugra_index > umor_index else umor_index + 1
                block.insert(insert_at, _mor_from_umor(block[umor_index]))
                inserted_from_umor += 1
                changed = True
                mor_index = insert_at
                if ugra_index is not None and insert_at <= ugra_index:
                    ugra_index += 1

            if mor_index is not None and gra_index is None:
                if ugra_index is not None:
                    gra_line = _gra_from_ugra(block[ugra_index])
                    inserted_from_ugra += 1
                else:
                    gra_line = _fallback_gra_from_mor(block[mor_index])
                    inserted_from_mor += 1
                block.insert(mor_index + 1, gra_line)
                changed = True

            new_lines.extend(block)

        if changed:
            path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            changed_files += 1

    return changed_files, inserted_from_umor, inserted_from_ugra, inserted_from_mor


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair missing %gra tiers in bundled synthetic CHAT files.")
    parser.add_argument("root_dir", nargs="+", help="One or more directories containing bundled .cha files.")
    args = parser.parse_args()

    total_changed_files = 0
    total_from_umor = 0
    total_from_ugra = 0
    total_from_mor = 0
    for root_dir in args.root_dir:
        changed_files, inserted_from_umor, inserted_from_ugra, inserted_from_mor = repair_missing_gra(root_dir)
        total_changed_files += changed_files
        total_from_umor += inserted_from_umor
        total_from_ugra += inserted_from_ugra
        total_from_mor += inserted_from_mor

    print(f"changed_files={total_changed_files}")
    print(f"inserted_from_umor={total_from_umor}")
    print(f"inserted_from_ugra={total_from_ugra}")
    print(f"inserted_from_mor={total_from_mor}")


if __name__ == "__main__":
    main()
