from __future__ import annotations

import unittest

from icl_pilot.maintenance.talktag_repair import (
    _align_utterances,
    _lexical_signature,
    _safe_lexical_restore,
    merge_exact_split_utterance,
    remove_exact_speaker_block,
    reconstruct_from_source,
    restore_confirmed_truncation,
    restore_exact_source_block,
)


class TalkTagRepairTests(unittest.TestCase):
    def test_normalizes_spacing(self) -> None:
        source = "*CHI:\the go home ."
        target = "*CHI:\the go [* m :03s :a] home ."
        repaired, reasons = reconstruct_from_source(source, target)
        self.assertEqual(repaired, "*CHI:\the go [* m:03s:a] home .")
        self.assertIn("spacing", reasons)

    def test_normalizes_zero_possessive(self) -> None:
        source = "*CHI:\tmy friend dog ran ."
        target = "*CHI:\tmy friend [* m :zero's] dog ran ."
        repaired, reasons = reconstruct_from_source(source, target)
        self.assertEqual(repaired, "*CHI:\tmy friend [* m:0's] dog ran .")
        self.assertIn("zero_possessive", reasons)

    def test_removes_gonna_tag(self) -> None:
        source = "*CHI:\tthey gonna fix it ."
        target = "*CHI:\tthey gonna [* m:+ing to] fix it ."
        repaired, reasons = reconstruct_from_source(source, target)
        self.assertEqual(repaired, source)
        self.assertIn("remove_gonna_tag", reasons)

    def test_restores_deleted_token_and_rejects_adjacent_tag(self) -> None:
        source = "*CHI:\the had lots of balloons ."
        target = "*CHI:\the [* m:base:ed] lots of balloons ."
        repaired, reasons = reconstruct_from_source(source, target)
        self.assertEqual(repaired, source)
        self.assertIn("restore_source_text", reasons)
        self.assertIn("reject_unaligned_tag:m:base:ed", reasons)

    def test_preserves_safe_tag_when_change_is_distant(self) -> None:
        source = "*CHI:\the go home with my dog ."
        target = "*CHI:\the go [* m:03s:a] home with dog ."
        repaired, reasons = reconstruct_from_source(source, target)
        self.assertEqual(repaired, "*CHI:\the go [* m:03s:a] home with my dog .")
        self.assertIn("restore_source_text", reasons)

    def test_places_tag_after_existing_correction(self) -> None:
        source = "*CHI:\the swimmed [: swam] home ."
        target = "*CHI:\the swimmed [: swam] [* m:=ed] home ."
        repaired, _ = reconstruct_from_source(source, target)
        self.assertEqual(repaired, target)

    def test_aligns_a_missing_source_utterance(self) -> None:
        source = [
            "*CHI:\tone line .",
            "*CHI:\t+. ",
            "*CHI:\tlast line .",
        ]
        target = [
            "*CHI:\tone line .",
            "*CHI:\tlast line .",
        ]
        self.assertEqual(
            _align_utterances(source, target),
            [(0, 0), (1, None), (2, 1)],
        )

    def test_high_similarity_replacement_is_safe_to_restore(self) -> None:
        self.assertTrue(
            _safe_lexical_restore(
                "*CHI:\twe have 5 dollars .",
                "*CHI:\twe have five dollars .",
            )
        )

    def test_silence_marker_has_no_lexical_content(self) -> None:
        self.assertEqual(_lexical_signature("*CHI:\t+."), [])

    def test_digit_and_word_numbers_compare_equally(self) -> None:
        self.assertEqual(
            _lexical_signature("*CHI:\tthat is $50 ."),
            _lexical_signature("*CHI:\tthat is fifty dollars ."),
        )
        self.assertEqual(
            _lexical_signature("*CHI:\tthat is $5 ."),
            _lexical_signature("*CHI:\tthat is five dollars ."),
        )

    def test_restores_spoken_comment(self) -> None:
        source = '*CHI:\t+" &gentle voice .'
        target = "*CHI:\t&=gentle_voice ."
        repaired, reasons = reconstruct_from_source(source, target)
        self.assertEqual(repaired, source)
        self.assertIn("restore_comment", reasons)
        self.assertFalse(
            _safe_lexical_restore(
                "*CHI:\tHere is a short narrative with a monkey .",
                "*CHI:\tThe monkey went home .",
            )
        )

    def test_merges_exact_split_and_drops_stale_dependent_tiers(self) -> None:
        source = "*CHI:\the go home and she walk back ."
        target = [
            "@Begin",
            "*CHI:\the go [* m:03s:a] home .",
            "%mor:\tstale one",
            "*CHI:\tand she walk [* m:03s:a] back .",
            "%mor:\tstale two",
            "@End",
        ]
        repaired, changed = merge_exact_split_utterance(target, source)
        self.assertTrue(changed)
        self.assertEqual(
            repaired,
            [
                "@Begin",
                "*CHI:\the go [* m:03s:a] home and she walk [* m:03s:a] back .",
                "@End",
            ],
        )

    def test_removes_inserted_line_and_its_dependent_tier(self) -> None:
        lines = [
            "@Begin",
            "*CHI:\toriginal line .",
            "%mor:\toriginal",
            "*CHI:\tinserted line .",
            "%mor:\tinserted",
            "@End",
        ]
        repaired, changed = remove_exact_speaker_block(
            lines,
            "*CHI:\tinserted line .",
        )
        self.assertTrue(changed)
        self.assertEqual(
            repaired,
            ["@Begin", "*CHI:\toriginal line .", "%mor:\toriginal", "@End"],
        )

    def test_restores_confirmed_repeated_her_truncation(self) -> None:
        source = "*CHI:\ther her her her her knee got hurt ."
        target = [
            "@Begin",
            "*CHI:\t"
            + " ".join(
                [
                    "her [* s:r:gc:pro]",
                    "her [* s:r:gc:pro]",
                    "her [* s:r:gc:pro]",
                ]
                + ["her [!]"] * 14
                + ["her"]
            ),
            "@End",
        ]
        repaired, changed = restore_confirmed_truncation(target, source)
        self.assertTrue(changed)
        self.assertEqual(
            repaired[1],
            "*CHI:\ther [* s:r:gc:pro] her [* s:r:gc:pro] "
            "her [* s:r:gc:pro] her her knee got hurt .",
        )

    def test_restores_exact_source_block_and_drops_stale_tier(self) -> None:
        source = "*CHI:\tnow both monkey [/] monkey happy ."
        current = "*CHI:\tnow both monkey [* m:0s:a] happy ."
        lines = ["@Begin", current, "%mor:\tstale", "@End"]
        repaired, changed = restore_exact_source_block(lines, source, current)
        self.assertTrue(changed)
        self.assertEqual(repaired, ["@Begin", source, "@End"])


if __name__ == "__main__":
    unittest.main()
