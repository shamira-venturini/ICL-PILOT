from __future__ import annotations

import json
from pathlib import Path

from icl_pilot.tier_stripper import strip_tiers


def _write_chat(path: Path) -> str:
    text = "\n".join(
        [
            "@Begin",
            "*CHI:\the runned home .",
            "%err:\trunned = ran [* m:ed]",
            "%mor:\tpro:sub|he v|run-PAST adv|home .",
            "%gra:\t1|2|SUBJ 2|0|ROOT 3|2|JCT 4|2|PUNCT",
            "%umor:\the|PRON run|VERB home|ADV .|PUNCT",
            "%ugra:\t1|2|nsubj 2|0|root 3|2|advmod 4|2|punct",
            "@End",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")
    return text


def test_strip_tiers_dry_run_does_not_change_file(tmp_path: Path) -> None:
    transcript = tmp_path / "sample.cha"
    original = _write_chat(transcript)

    assert strip_tiers(tmp_path, dry_run=True) == 0

    assert transcript.read_text(encoding="utf-8") == original


def test_strip_tiers_removes_old_and_ud_tiers_only(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first_transcript = first_root / "sample.cha"
    second_transcript = second_root / "sample.cha"
    _write_chat(first_transcript)
    _write_chat(second_transcript)
    summary_path = tmp_path / "summary.json"

    assert strip_tiers([first_root, second_root], summary_json=summary_path) == 0

    for transcript in (first_transcript, second_transcript):
        result = transcript.read_text(encoding="utf-8")
        assert "%mor:" not in result
        assert "%gra:" not in result
        assert "%umor:" not in result
        assert "%ugra:" not in result
        assert "*CHI:\the runned home ." in result
        assert "%err:\trunned = ran [* m:ed]" in result

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["totals"]["transcript_files"] == 2
    assert summary["totals"]["files_with_tiers"] == 2
    assert summary["totals"]["tier_lines_total"] == 8
    assert summary["totals"]["tier_lines_remaining"] == 0
    assert summary["totals"]["speaker_lines"] == summary["totals"]["speaker_lines_after"]


def test_strip_tiers_removes_whole_annotation_blocks_and_keeps_wrapped_speech(tmp_path: Path) -> None:
    transcript = tmp_path / "sample.cha"
    kept = ["@Begin", "@G:\tA1", "*CHI:\the runned",
            "\thome .", "%err:\trunned = ran [* m:ed]",
            "\tmanual annotation continuation", "@Comment:\tkeep this comment",
            "*EXA:\tand then ?", "@End"]
    removed = ["%mor:\tpro:sub|he", "\tv|run-PAST adv|home .",
               "%gra:\t1|2|SUBJ", "\t2|0|ROOT 3|2|JCT 4|2|PUNCT",
               "%umor:\the|PRON", "\trun|VERB home|ADV .|PUNCT",
               "%ugra:\t1|2|nsubj", "\t2|0|root 3|2|advmod 4|2|punct"]
    lines = kept[:6] + removed + kept[6:]
    original = "\n".join(lines) + "\n"
    transcript.write_text(original)
    assert strip_tiers(tmp_path, dry_run=True) == 0
    assert transcript.read_text() == original
    summary = tmp_path / "summary.json"
    assert strip_tiers(tmp_path, summary_json=summary) == 0
    assert transcript.read_text() == "\n".join(kept) + "\n"
    qc = json.loads(summary.read_text())
    assert qc["totals"]["tier_lines_total"] == 4
    assert qc["totals"]["tier_continuation_lines_total"] == 4
    assert qc["totals"]["tier_continuation_lines"] == {"mor": 1, "gra": 1, "umor": 1, "ugra": 1}
