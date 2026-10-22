import pytest

from icl_pilot.chat.chat_structure import (
    annotation_anchors, format_generated_annotated_chat, generated_surface_tokens,
)


def chat(*bodies, tier=""):
    return (
        "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n"
        "@ID:\teng|ICL-PILOT|CHI|4;01.27||SLI||Target_Child|||\n"
        "@Types:\tcross, narrative, generated, SLI\n@G:\tA1\n"
        + "".join("*CHI:\t" + body + "\n" for body in bodies)
        + tier + "@End\n"
    )


def bodies(text):
    return [line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("*CHI:")]


@pytest.mark.parametrize("before,expected", [
    ("he go [* m:0ed] to pool.", "he go [* m:0ed] to pool ."),
    ("I , , I goed [: went] [* m:=ed] .", "I , I goed [: went] [* m:=ed] ."),
    ("he got 5 [* s:r] balloons .", "he got five [* s:r] balloons ."),
    ("he go[* m:0ed]to pool.", "he go [* m:0ed] to pool ."),
    ("it was pooo:l (.) good .", "it was pooo:l (.) good ."),
    ("he go [* m:+ing to] .", "he go [* m:+ing to] ."),
    ("he go [* m :zero's] .", "he go [* m :zero's] ."),
    ("he go [* m:0ed]", "he go [* m:0ed] ."),
    ("he go [* m:0ed] , [/] go [* m:0ed] .", "he go [* m:0ed] [/] go [* m:0ed] ."),
])
def test_format_preserves_codes_grammar_and_anchors(before, expected):
    result, changes = format_generated_annotated_chat(chat(before))
    assert bodies(result) == [expected]
    assert annotation_anchors(before) == annotation_anchors(expected)
    assert "@Types:\tcross, narrative, SLI\n" in result
    assert format_generated_annotated_chat(result)[0] == result


def test_quote_links_preserve_tagged_tiers():
    source = chat('+" he goed [: went] [* m:=ed] .', "then he go [* m:0ed] .")
    result, _ = format_generated_annotated_chat(source)
    assert bodies(result) == ['“he goed [: went] [* m:=ed]” .', 'then he go [* m:0ed] .']
    assert format_generated_annotated_chat(result)[0] == result


def test_no_main_tier_changes_with_existing_morphology():
    with pytest.raises(ValueError, match="fresh morphosyntactic"):
        format_generated_annotated_chat(chat("he go.", tier="%mor:\tpro|he v|go .\n"))


def test_header_only_repair_preserves_dependent_tiers():
    source = chat("he go [* m:0ed] .", tier="%mor:\tpro|he v|go .\n%gra:\t1|2|SUBJ 2|0|ROOT 3|2|PUNCT\n")
    result, _ = format_generated_annotated_chat(source)
    assert source[source.index("*CHI:"):] == result[result.index("*CHI:"):]


def test_wrapped_tier_requires_review():
    with pytest.raises(ValueError, match="Wrapped main tier"):
        format_generated_annotated_chat(chat("he go\n\tto pool ."))


def test_real_data_refused():
    with pytest.raises(ValueError, match="Expected a supported"):
        format_generated_annotated_chat(chat("he go .").replace(
            "@Types:\tcross, narrative, generated, SLI", "@Types:\tcross, narrative, SLI"))


def test_surface_comparison_ignores_annotations_and_written_variants():
    assert generated_surface_tokens(chat("He is running, dont worry .")) == generated_surface_tokens(
        chat("he (i)s runnin(g) [: running] [* s:r] , don't worry ."))


@pytest.mark.parametrize("changed", ["he goes .", "he was go .", "he .", "he go go ."])
def test_surface_comparison_detects_changed_speech(changed):
    assert generated_surface_tokens(chat("he go .")) != generated_surface_tokens(chat(changed))
