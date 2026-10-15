# CHAT extraction for generation inputs

`src/icl_pilot/story_unit_semrp.py` implements `chat-story-v2`. It reads literal
CHI metadata and extracts story-bounded CHI main tiers. It follows the
[TalkBank CHAT manual](https://talkbank.org/0info/manuals/CHAT.html) for ages and
the distinction between main and dependent tiers.

## Rebuild a new input

With the source corpus available, run from the repository root:

```bash
PYTHONPATH=src python -m icl_pilot extract-story-unit-narratives \
  data/annotated/ENNI_batchalign \
  data/processed/generation_inputs/story_unit_narratives_chat_v2.csv
```

The command writes a new CSV and `story_unit_narratives_chat_v2.parsing_qc.json`.
Existing CSV/QC outputs are protected: choose a fresh filename for a comparison
or a new corpus version. `--summary-json` can specify a separate QC path.
In Colab, the same builder is available as
`icl_pilot.story_unit_semrp.extract_story_unit_narratives(transcript_root, output_csv)`.

`configs/generation/experiment.json` points new generation runs to this input.
The historical `phase2/ENNI/measures/story_units/story_unit_narratives.csv`
remains unchanged for analysis and original prompt provenance.

## Parsing rules

- Ages use `years;months.days`. `Age_Month` is `12 * years + months`, with
  completed months only. The full raw age, including days, is retained.
  Unknown months, missing ages and malformed ages remain blank and are flagged.
- Group values come from CHI `@ID` fields. No diagnosis, missing age or group is
  inferred from a folder, filename, `@Types`, or linguistic measures.
  Conflicting CHI headers fail rather than selecting one silently.
- Wrapped CHI main-tier lines are joined before the existing surface cleaner
  runs. Other speakers, dependent tiers and their continuations are excluded.
  A story boundary flushes the previous utterance before changing its story.
- Some processed ENNI files contain orphan dependent-tier continuations.
  Unbracketed pipe-bearing annotation tokens and redundant delimiter-only
  continuations after a completed main tier identify such blocks. They are
  excluded and reported by source file, line, story and speaker. These checks
  are a safeguard for this malformed processed input; they do not restore tiers.
- Surface cleaning retains its historical policy. Repetitions and the child's
  grammatical forms are preserved; correction annotations do not replace the
  spoken forms. Pause notation, unintelligibility markers and non-speech codes
  retain their existing treatment. A future normalization change needs a new
  input version.
- UTF-8 decoding is strict. Missing/empty transcript roots and existing output
  files fail explicitly. Raw CHAT bytes are never changed by extraction.

The QC sidecar records the parser version, age definition, relative source paths,
source SHA-256 hashes, unknown metadata and excluded-line details. The CSV retains
its existing column schema and absolute transcript-path convention for compatibility.

## Corpus verification on 14 October 2026

All 361 source CHAT files were rehashed unchanged. The new export contains 2,166
story rows with identical narrative text and utterance arrays to the historical
CSV; age months were populated in 2,160 rows. All 564 selected exemplars have
ages matching the master roster.

`TD/B/523.cha` has empty CHI age/group metadata and an ID collision with an
SLI-labelled file. Its six story rows remain unknown and are excluded from the
TD prompt lookup. The parser reports 33,437 orphan annotation continuation lines
and 948 redundant delimiter continuations across the full source directory.
These are parsing/QC counts, not linguistic error counts.

The tier-removal utility was separately repaired to remove the continuations of
`%mor`, `%gra`, `%umor` and `%ugra` along with their labels. Wrapped main speech
and other tiers, including `%err`, are retained. This code was tested on fixtures;
it was not run on corpus files.

The source/CSV comparison and checksums are under
`outputs/maintenance/chat_parser_fix_2026-10-14/`.
