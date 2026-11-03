from __future__ import annotations

from pathlib import Path

import pytest

from icl_pilot.features.syntax_features import _parse_umor_tokens, _summarize_transcript


@pytest.mark.parametrize("morphology,expected", [
    (
        "pron|he-Prs-Nom-S3~aux|be-Fin-Ind-Pres-S3 verb|run-Part-Pres .",
        ["pron|he-Prs-Nom-S3", "aux|be-Fin-Ind-Pres-S3", "verb|run-Part-Pres", "."],
    ),
    (
        "aux|do-Fin-Ind-Pres-S3~part|not verb|run-Inf .",
        ["aux|do-Fin-Ind-Pres-S3", "part|not", "verb|run-Inf", "."],
    ),
    (
        "noun|dog-Sing~part|s noun|ball-Sing .",
        ["noun|dog-Sing", "part|s", "noun|ball-Sing", "."],
    ),
    (
        "pron|he-Prs-Nom-S3~aux|would-Fin-Ind-Past~aux|have-Inf verb|go-Part-Past .",
        ["pron|he-Prs-Nom-S3", "aux|would-Fin-Ind-Past", "aux|have-Inf", "verb|go-Part-Past", "."],
    ),
    ("  noun|ball-Sing   cm|cm   .  ", ["noun|ball-Sing", "cm|cm", "."]),
])
def test_morphology_expands_clitics_in_dependency_order(morphology: str, expected: list[str]) -> None:
    assert _parse_umor_tokens(morphology) == expected


@pytest.mark.parametrize("tiers", [("mor", "gra"), ("umor", "ugra")])
@pytest.mark.parametrize("contracted", [True, False])
def test_contraction_and_full_form_have_same_syntax_metrics(
    tmp_path: Path, tiers: tuple[str, str], contracted: bool,
) -> None:
    # Both forms have pronoun/auxiliary/verb/punctuation nodes. The auxiliary
    # is finite; the progressive participle is not. Punctuation contributes
    # neither a dependency-length observation nor a token/depth observation.
    morphology = "pron|he-Prs-Nom-S3{joiner}aux|be-Fin-Ind-Pres-S3 verb|run-Part-Pres ."
    morphology = morphology.format(joiner="~" if contracted else " ")
    surface = "he's running ." if contracted else "he is running ."
    path = tmp_path / "child.cha"
    path.write_text(
        f"@Begin\n*CHI:\t{surface}\n%{tiers[0]}:\t{morphology}\n"
        f"%{tiers[1]}:\t1|3|NSUBJ 2|3|AUX 3|0|ROOT 4|3|PUNCT\n@End\n",
        encoding="utf-8",
    )
    before = path.read_bytes()
    features, qc = _summarize_transcript(path)
    assert qc == {
        "chi_blocks_total": 1, "chi_blocks_analyzed": 1,
        "chi_blocks_skipped_missing_tiers": 0,
        "chi_blocks_skipped_parse_errors": 0,
        "chi_blocks_skipped_token_mismatch": 0,
    }
    assert features["Syntax_Tokens_Analyzed"] == 3
    assert features["Syntax_Mean_DepLen"] == 1.5
    assert features["Syntax_Max_DepLen"] == 2
    assert features["Syntax_Mean_TreeDepth"] == pytest.approx(2 / 3)
    assert features["Syntax_Max_TreeDepth"] == 1
    assert features["Syntax_FiniteVerb_Tokens_per_Utt"] == 1
    assert features["Syntax_Subject_Utt_Prop"] == 1
    assert features["Syntax_AuxCop_Utt_Prop"] == 1
    assert path.read_bytes() == before


@pytest.mark.parametrize("morphology,graph,finite_verbs,auxcop,mean_length", [
    (
        "pron|he-Prs-Nom-S3 aux|do-Fin-Ind-Pres-S3~part|not verb|run-Inf .",
        "1|4|NSUBJ 2|4|AUX 3|4|ADVMOD 4|0|ROOT 5|4|PUNCT", 1, 1, 2,
    ),
    (
        "det|the noun|dog-Sing~part|s noun|ball-Sing .",
        "1|2|DET 2|4|NMOD-POSS 3|2|CASE 4|0|ROOT 5|4|PUNCT", 0, 0, 4 / 3,
    ),
])
def test_negation_and_possessive_clitics_keep_their_own_nodes(
    tmp_path: Path, morphology: str, graph: str, finite_verbs: int, auxcop: int, mean_length: float,
) -> None:
    path = tmp_path / "child.cha"
    path.write_text(
        f"*CHI:\tfixture .\n%mor:\t{morphology}\n%gra:\t{graph}\n", encoding="utf-8",
    )
    features, qc = _summarize_transcript(path)
    assert qc["chi_blocks_analyzed"] == 1
    assert qc["chi_blocks_skipped_token_mismatch"] == 0
    assert features["Syntax_Tokens_Analyzed"] == 4
    assert features["Syntax_Mean_DepLen"] == pytest.approx(mean_length)
    assert features["Syntax_FiniteVerb_Tokens_per_Utt"] == finite_verbs
    assert features["Syntax_AuxCop_Utt_Prop"] == auxcop


def test_clitic_expansion_does_not_hide_invalid_or_missing_dependencies(tmp_path: Path) -> None:
    path = tmp_path / "child.cha"
    morphology = "pron|he-Prs-Nom-S3~aux|be-Fin-Ind-Pres-S3 verb|run-Part-Pres ."
    path.write_text(
        f"*CHI:\the's running .\n%mor:\t{morphology}\n"
        "%gra:\t1|3|NSUBJ 2|3|AUX 3|0|ROOT\n"  # Punctuation node missing.
        f"*CHI:\the's running .\n%mor:\t{morphology}\n"
        "%gra:\t1|3|NSUBJ invalid 3|0|ROOT 4|3|PUNCT\n"
        "*CHI:\t0 .\n",
        encoding="utf-8",
    )
    _, qc = _summarize_transcript(path)
    assert qc == {
        "chi_blocks_total": 3, "chi_blocks_analyzed": 0,
        "chi_blocks_skipped_missing_tiers": 1,
        "chi_blocks_skipped_parse_errors": 1,
        "chi_blocks_skipped_token_mismatch": 1,
    }
