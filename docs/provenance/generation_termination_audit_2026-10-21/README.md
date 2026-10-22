# Historical generation endings audit — 21 October 2026

The author requested checking historical outputs before choosing the revised
generation ceiling. This audit covers all 2,820 paired story jobs / 5,640 stage
outputs from ages four through seven. The unit counted below is a generated
story output; these are correlated replicates, not independent children.

## Findings

| Age | Outputs per stage | TD-like texts at 180 proxy tokens | SLI-conditioned texts at 180 proxy tokens |
| --- | ---: | ---: | ---: |
| 4 | 660 | 1 | 10 |
| 5 | 720 | 1 | 9 |
| 6 | 660 | 4 | 12 |
| 7 | 780 | 4 | 16 |
| Total | 2,820 | 10 (0.35%) | 47 (1.67%) |

The counts describe saved texts re-encoded with a pinned tokenizer, not recovered
generation token counts. There are 57 texts exactly at 180, seven at 178–179,
and none above 180. Median proxy length is 81 tokens in Stage 1 and 75 in Stage 2;
the nearest-rank 95th percentiles are 137 and 138. Nine of the ten Stage 1 texts
at 180 are B3; the other is A3. Two jobs have both stages at 180.

All 64 boundary tails were manually inspected as text. Examples at exactly 180:

- Age 6, `pair_sli607_td634_rep05`, A3, TD-like: ends with “The mommy”.
- Age 7, `pair_sli733_td711_rep06`, B3, TD-like: ends with “They each got a”.
- Age 7, `pair_sli723_td771_rep04`, B3, SLI-conditioned: ends at “The”.
- Age 5, `pair_sli529_td545_rep10`, A2, SLI-conditioned: includes a run of
  101 consecutive instances of “see” under the case-insensitive word-count rule.

Some boundary texts have complete-looking final clauses, and some contain
Original/Rewritten or Original/Transformed commentary. These are different
response problems. Raising the budget could extend a repetition sequence and
does not itself enforce narrative-only output. Nine of the 57 exact-boundary
texts have terminal punctuation. Punctuation does not prove a natural stop;
its absence does not prove truncation or a language error.

## Evidence and limits

The four raw CSVs were read from the verified earlier data snapshot:

`/Volumes/TranscendBU/ICL-PILOT_af80b106b6dd_20261016T085254Z/Data/data_snapshot.tar.gz`

Archive SHA-256:
`f5ad10779df14859da62ccfaf3f67e9f85e3cba15bf7eaae72485f46f938b0e0`.
Each CSV matches its archived manifest. All 5,640 text hashes and their job
metadata also match the pre-cleanup generation baseline from 14 October.
The CSVs were retired before the later complete study freeze, which indexes
this earlier archive. They were read through temporary copies, without restoring
the historical data folder or changing any transcript.

The raw CSVs do not record output token IDs, generated-token counts, stopping
reasons or per-run token ceilings. No generation journals/run manifests were
present in the data snapshot. The 180-token reference comes from preserved
notebook source, not a recovered runtime log.

Re-encoding used `avans06/Meta-Llama-3.2-8B-Instruct` tokenizer JSON at revision
`61f0ad72eb603310b03ec8a3260fd912a82a1cb5`, verified against its public Git blob.
The historical installed revision is unknown. Removed special tokens, stripped
whitespace and re-tokenization can change counts. Consequently these results
support concern about a ceiling effect but do not identify a confirmed truncation
rate or classify individual stops as EOS versus budget exhaustion.

## Records and reproduction

- [`audit.json`](audit.json): provenance, hashes, methods, findings and external verification.
- [`by_age_stage.csv`](by_age_stage.csv): all four cohorts and both stages.
- [`by_story_stage.csv`](by_story_stage.csv): all six targets and both stages.
- [`manual_boundary_review.csv`](manual_boundary_review.csv): 64 tails, source identities,
  text hashes and review notes. These notes are not clinical error annotations.

The external report directory recorded in `audit.json` contains the complete
per-output inventory, finer age/story summaries, tokenizer snapshot, audit code,
test source and file-by-file archive verification. The reusable tool is
`src/icl_pilot/maintenance/generation_endings_audit.py`. After restoring only the
four CSVs to a separate working directory, run:

```bash
PYTHONPATH=src python -m icl_pilot.maintenance.generation_endings_audit \
  --source 4=/path/to/four_year_old_rotating_story_generations.csv \
  --source 5=/path/to/5_year_old_rotating_story_generations.csv \
  --source 6=/path/to/6_year_old_rotating_story_generations.csv \
  --source 7=/path/to/7_year_old_rotating_story_generations.csv \
  --tokenizer-json /path/to/pinned/tokenizer.json \
  --reference-limit 180 --output-dir /path/to/fresh/audit
```

The recorded re-tokenization environment was Python 3.12.8 with tokenizers 0.22.2.
The audit core and 11 targeted tests passed under Python 3.8.5. No model weights
were downloaded. No generation, annotation or existing linguistic analysis was
rerun, and no generation settings or prompts were changed during this audit.

The revised ceiling remains undecided. The evidence supports testing a more
generous safeguard in a balanced pilot with the new official model, recording
EOS/budget termination and handling capped Stage 1 parents explicitly. Prompt
instructions requesting short narratives are a separate length constraint.
