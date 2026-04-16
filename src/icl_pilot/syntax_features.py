from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


_UMOR_RE = re.compile(r"^%umor:\t(.*)$")
_UGRA_RE = re.compile(r"^%ugra:\t(.*)$")

_SUBORDINATION_PREFIXES = ("CCOMP", "XCOMP", "ADVCL", "ACL", "CSUBJ")
_COORDINATION_PREFIXES = ("CONJ", "CC")
_SUBJECT_PREFIXES = ("NSUBJ", "CSUBJ")
_OBJECT_RELATIONS = {"OBJ", "IOBJ"}
_AUXCOP_PREFIXES = ("AUX", "COP")
_OBLIQUE_PREFIXES = ("OBL",)
_COMPLEX_NP_PREFIXES = ("AMOD", "NUMMOD", "COMPOUND", "NMOD", "APPOS", "ACL")

_FEATURE_COLUMNS = [
    "Syntax_Utts_Analyzed",
    "Syntax_Tokens_Analyzed",
    "Syntax_Mean_DepLen",
    "Syntax_Max_DepLen",
    "Syntax_Mean_TreeDepth",
    "Syntax_Max_TreeDepth",
    "Syntax_Subordination_Utt_Prop",
    "Syntax_Coordination_Utt_Prop",
    "Syntax_Subject_Utt_Prop",
    "Syntax_Object_Utt_Prop",
    "Syntax_AuxCop_Utt_Prop",
    "Syntax_Oblique_Utt_Prop",
    "Syntax_ComplexNP_Utt_Prop",
    "Syntax_FiniteVerb_Tokens_per_Utt",
]


def _after_tab(line: str) -> str:
    if "\t" in line:
        return line.split("\t", 1)[1].strip()
    return line.split(":", 1)[1].strip()


def _iter_chi_blocks(path: Path) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    current: dict[str, str] | None = None

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.rstrip("\n")
        if line.startswith("*CHI:"):
            if current is not None:
                blocks.append(current)
            current = {"speaker": line}
            continue
        if current is None:
            continue
        if line.startswith("*"):
            blocks.append(current)
            current = None
            continue
        if line.startswith("%umor:"):
            current["umor"] = _after_tab(line)
        elif line.startswith("%ugra:"):
            current["ugra"] = _after_tab(line)
        elif line.startswith("@"):
            blocks.append(current)
            current = None

    if current is not None:
        blocks.append(current)
    return blocks


def _parse_umor_tokens(umor: str) -> list[str]:
    return [token for token in umor.split() if token]


def _parse_ugra_tokens(ugra: str) -> list[tuple[int, int, str]] | None:
    parsed: list[tuple[int, int, str]] = []
    for token in ugra.split():
        parts = token.split("|")
        if len(parts) != 3:
            return None
        try:
            index = int(parts[0])
            head = int(parts[1])
        except ValueError:
            return None
        relation = parts[2].upper()
        parsed.append((index, head, relation))
    return parsed


def _is_punct(umor_token: str, relation: str) -> bool:
    if relation == "PUNCT":
        return True
    if "|" not in umor_token:
        return True
    pos = umor_token.split("|", 1)[0].lower()
    return pos in {"cm"}


def _count_finite_verbal_segments(umor_tokens: list[str]) -> int:
    count = 0
    for token in umor_tokens:
        for segment in token.split("~"):
            if "|" not in segment:
                continue
            pos, rest = segment.split("|", 1)
            if pos.lower() in {"verb", "aux"} and "-Fin-" in rest:
                count += 1
    return count


def _relation_matches(relation: str, prefixes: tuple[str, ...]) -> bool:
    return any(relation == prefix or relation.startswith(f"{prefix}-") for prefix in prefixes)


def _safe_depth(index: int, head_map: dict[int, int], relation_map: dict[int, str], cache: dict[int, int], stack: set[int]) -> int:
    if index in cache:
        return cache[index]
    if index in stack:
        cache[index] = 0
        return 0
    relation = relation_map.get(index, "")
    head = head_map.get(index, index)
    if relation == "ROOT" or head == index or head == 0:
        cache[index] = 0
        return 0
    stack.add(index)
    depth = 1 + _safe_depth(head, head_map, relation_map, cache, stack)
    stack.remove(index)
    cache[index] = depth
    return depth


def _summarize_transcript(path: Path) -> tuple[dict[str, float], dict[str, int]]:
    blocks = _iter_chi_blocks(path)
    utterances = 0
    tokens_analyzed = 0
    dep_lengths: list[int] = []
    tree_depths: list[int] = []
    finite_verbs = 0
    subordination_utts = 0
    coordination_utts = 0
    subject_utts = 0
    object_utts = 0
    auxcop_utts = 0
    oblique_utts = 0
    complex_np_utts = 0
    skipped_missing = 0
    skipped_parse = 0
    skipped_mismatch = 0

    for block in blocks:
        umor = block.get("umor", "")
        ugra = block.get("ugra", "")
        if not umor or not ugra:
            skipped_missing += 1
            continue

        umor_tokens = _parse_umor_tokens(umor)
        ugra_tokens = _parse_ugra_tokens(ugra)
        if ugra_tokens is None:
            skipped_parse += 1
            continue
        if len(umor_tokens) != len(ugra_tokens):
            skipped_mismatch += 1
            continue

        head_map = {index: head for index, head, _ in ugra_tokens}
        relation_map = {index: relation for index, _, relation in ugra_tokens}
        depth_cache: dict[int, int] = {}
        nonpunct_indices: list[int] = []
        relations_present: list[str] = []

        for (index, head, relation), umor_token in zip(ugra_tokens, umor_tokens):
            if _is_punct(umor_token, relation):
                continue
            nonpunct_indices.append(index)
            relations_present.append(relation)
            tokens_analyzed += 1
            depth = _safe_depth(index, head_map, relation_map, depth_cache, set())
            tree_depths.append(depth)
            if relation != "ROOT" and head != index and head != 0:
                dep_lengths.append(abs(index - head))

        if not nonpunct_indices:
            continue

        utterances += 1
        finite_verbs += _count_finite_verbal_segments(umor_tokens)

        relation_set = set(relations_present)
        if any(_relation_matches(relation, _SUBORDINATION_PREFIXES) for relation in relation_set):
            subordination_utts += 1
        if any(_relation_matches(relation, _COORDINATION_PREFIXES) for relation in relation_set):
            coordination_utts += 1
        if any(_relation_matches(relation, _SUBJECT_PREFIXES) for relation in relation_set):
            subject_utts += 1
        if any(relation in _OBJECT_RELATIONS for relation in relation_set):
            object_utts += 1
        if any(_relation_matches(relation, _AUXCOP_PREFIXES) for relation in relation_set):
            auxcop_utts += 1
        if any(_relation_matches(relation, _OBLIQUE_PREFIXES) for relation in relation_set):
            oblique_utts += 1
        if any(_relation_matches(relation, _COMPLEX_NP_PREFIXES) for relation in relation_set):
            complex_np_utts += 1

    def _mean(values: list[int]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    features = {
        "Syntax_Utts_Analyzed": float(utterances),
        "Syntax_Tokens_Analyzed": float(tokens_analyzed),
        "Syntax_Mean_DepLen": _mean(dep_lengths),
        "Syntax_Max_DepLen": float(max(dep_lengths)) if dep_lengths else 0.0,
        "Syntax_Mean_TreeDepth": _mean(tree_depths),
        "Syntax_Max_TreeDepth": float(max(tree_depths)) if tree_depths else 0.0,
        "Syntax_Subordination_Utt_Prop": (subordination_utts / utterances) if utterances else 0.0,
        "Syntax_Coordination_Utt_Prop": (coordination_utts / utterances) if utterances else 0.0,
        "Syntax_Subject_Utt_Prop": (subject_utts / utterances) if utterances else 0.0,
        "Syntax_Object_Utt_Prop": (object_utts / utterances) if utterances else 0.0,
        "Syntax_AuxCop_Utt_Prop": (auxcop_utts / utterances) if utterances else 0.0,
        "Syntax_Oblique_Utt_Prop": (oblique_utts / utterances) if utterances else 0.0,
        "Syntax_ComplexNP_Utt_Prop": (complex_np_utts / utterances) if utterances else 0.0,
        "Syntax_FiniteVerb_Tokens_per_Utt": (finite_verbs / utterances) if utterances else 0.0,
    }
    qc = {
        "chi_blocks_total": len(blocks),
        "chi_blocks_analyzed": utterances,
        "chi_blocks_skipped_missing_tiers": skipped_missing,
        "chi_blocks_skipped_parse_errors": skipped_parse,
        "chi_blocks_skipped_token_mismatch": skipped_mismatch,
    }
    return features, qc


def build_syntax_feature_table_from_master(
    master_csv: str,
    output_csv: str,
    output_summary_json: str,
) -> int:
    master_path = Path(master_csv).expanduser().resolve()
    output_path = Path(output_csv).expanduser().resolve()
    summary_path = Path(output_summary_json).expanduser().resolve()

    master = pd.read_csv(master_path)
    required = ["File_ID", "DSS_tbl_transcript"]
    missing = [column for column in required if column not in master.columns]
    if missing:
        raise ValueError(f"Missing required columns in master: {', '.join(missing)}")

    rows: list[dict[str, object]] = []
    total_qc = {
        "chi_blocks_total": 0,
        "chi_blocks_analyzed": 0,
        "chi_blocks_skipped_missing_tiers": 0,
        "chi_blocks_skipped_parse_errors": 0,
        "chi_blocks_skipped_token_mismatch": 0,
    }

    for row in master.to_dict("records"):
        file_id = str(row.get("File_ID", ""))
        transcript = str(row.get("DSS_tbl_transcript", "")).strip()
        transcript_path = Path(transcript).expanduser()
        if not transcript_path.is_absolute():
            transcript_path = (master_path.parent / transcript_path).resolve()
        features, qc = _summarize_transcript(transcript_path)
        rows.append(
            {
                "File_ID": file_id,
                "syntax_source": str(transcript_path),
                **features,
            }
        )
        for key, value in qc.items():
            total_qc[key] += value

    feature_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_path, index=False)

    summary = {
        "input_master_csv": str(master_path),
        "output_csv": str(output_path),
        "rows": int(len(feature_df)),
        "feature_columns": _FEATURE_COLUMNS,
        "totals": total_qc,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote syntax feature table to {output_path}")
    print(f"Wrote syntax feature summary to {summary_path}")
    return 0


def merge_syntax_features_into_master(
    master_csv: str,
    syntax_feature_csv: str,
    output_csv: str | None = None,
) -> int:
    master_path = Path(master_csv).expanduser().resolve()
    syntax_path = Path(syntax_feature_csv).expanduser().resolve()
    output_path = Path(output_csv).expanduser().resolve() if output_csv else master_path

    master = pd.read_csv(master_path)
    syntax_df = pd.read_csv(syntax_path)

    required = ["File_ID", "syntax_source", *_FEATURE_COLUMNS]
    missing = [column for column in required if column not in syntax_df.columns]
    if missing:
        raise ValueError(f"Missing required syntax-feature columns: {', '.join(missing)}")

    drop_columns = [column for column in ["syntax_source", *_FEATURE_COLUMNS] if column in master.columns]
    if drop_columns:
        master = master.drop(columns=drop_columns)

    merged = master.merge(
        syntax_df[required],
        on="File_ID",
        how="left",
        validate="one_to_one",
    )
    merged.to_csv(output_path, index=False)
    print(f"Wrote master with syntax features to {output_path}")
    return 0
