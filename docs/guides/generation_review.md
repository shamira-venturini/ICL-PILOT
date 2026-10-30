# Review generated stories

The generation notebook has a clickable generation-review block. It creates a
reviewable record for every selected Stage 1 and Stage 2 story. Nothing is
excluded, edited, regenerated, resampled or replaced. Review flags and human notes
do not feed into annotation, corpus metrics, classifier sampling or R analyses.

## Run the block

After generation, run the generation-review cell. To inspect saved outputs alone:

1. Select `MODE="review_only"` in the notebook settings.
2. Set `GENERATION_CSVS` to labelled generator/cohort CSV paths. Choose an aggregate
   CSV or its constituent cohorts, avoiding overlapping jobs.
3. Run common setup, then jump to generation review. No model, conversion or
   annotation setup is needed.

```python
GENERATION_CSVS = {
    "llama": Path("/content/drive/MyDrive/LLAMA_RUN/all_cohorts_rotating_story_generations.csv"),
    "qwen": Path("/content/drive/MyDrive/QWEN_RUN/all_cohorts_rotating_story_generations.csv"),
    "gemma": Path("/content/drive/MyDrive/GEMMA_RUN/all_cohorts_rotating_story_generations.csv"),
}
REVIEW_JOURNALS = {}  # Defaults to prompt_records.jsonl beside each CSV.
DOWNLOAD_REVIEW_HTML = True  # Colab: download the standalone page to open locally.
```

If journals are elsewhere, supply `REVIEW_JOURNALS={CSV_LABEL: Path(...)}`.
Explicit missing journals stop review. When no adjacent journal is available,
the report still displays all stories but marks stopping information unknown.
It never infers truncation from text length, punctuation or the BERT input limit.
Selected journal outputs must match CSV text hashes, stage and available job
identities. Stage 2 must identify its exact Stage 1 parent. Legacy CSVs without
prompt record IDs display unavailable stopping metadata rather than invented values.

## Review flags

The common settings are in `configs/generation/review.json`. They apply to every
generator and stage, with Stage 2 identity comparison applying only to Stage 2.

- `empty_output`: no non-whitespace text.
- `generation_token_limit`: the actual recorded finish reason is token_ceiling.
  A natural end token exactly at the budget is not flagged as a cutoff.
- `token_limit_reached_stop_unknown`: the journal records a reached budget but
  no stopping reason, so the cause is uncertain.
- `unchanged_stage2`: nonempty Stage 2 matches its Stage 1 parent after collapsing
  whitespace. Case and punctuation are not silently normalized. A changed story
  does not establish clinical fidelity.
- `possible_prompt_leakage`: literal instruction/header or role markers appear
  in output. Patterns come from static parts of the frozen templates. Expected
  story content, exemplars and Stage 1 parent text are not searched as instructions.
- `long_word_run`: at least 20 consecutive occurrences of the same word.
- `repeated_phrase_loop`: a phrase of 2–8 words repeats at least five times and
  spans at least 20 words; single-word runs use their own flag.

Repetition screening ignores punctuation between words and letter case. It can
therefore join material across sentence boundaries. Thresholds are screening
heuristics, not clinical norms or exclusion criteria. Original text is retained
in full for contextual judgment. Ordinary repetitions, grammatical errors and
other potentially meaningful language characteristics are not filtered.

## Reports and notes

Each execution writes a new timestamped folder under `RUN_OUTPUT_DIR/generation_review/`:

- `review.html`: offline page with collection/stage/age/flag/status filters,
  text/metadata search and full paired Stage 1/Stage 2 displays.
- `all_stories.csv`: every selected story, original text, identities, stopping
  metadata, flags, evidence and empty review-note fields.
- `flagged_stories.csv`: an inspection subset, never a filtered analysis input.
- `review_data.json`: structured full story records and source/stage/cohort summaries.
- `flag_summary.json`: denominators, flag counts and stopping-metadata coverage.
- `review_manifest.json`: review policy, thresholds, source/template/report hashes
  and an explicit zero-exclusions record.

Use the page's pending/checked/needs-discussion status and notes for the selected
story. The paired output is visible for context. Notes stay in browser storage
when available; download the review-notes CSV before closing the page to retain
them with the study records. The original report files remain immutable. No
analysis reads these statuses or notes. Local report IDs are stable for identical
labelled source CSVs, allowing browser notes to carry across report reruns.

Reports travel with ordinary Drive/ZIP exports. Input paths and hashes remain
recorded; retain the original generation journals when moving CSVs. Source rows,
both stage texts and every complete bundle remain unchanged. Structural CHAT
validation remains a separate required processing step.
