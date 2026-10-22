# Framework evaluation inputs — pending revised study

Set both paths in [`inputs.json`](inputs.json) when the revised exports exist:

- `text_records_csv`: the selected experiment's real/generated story-text export.
- `gillam_external_text_csv`: the separately selected external-test text export.

Paths may be absolute or relative to the checkout. The notebook stops before
creating results while either path is pending or does not resolve to a file.
No historical Phase 2 table is selected automatically. The external-corpus choice
and evaluation settings remain subject to the revised protocol review.

The reusable CSV exporter also requires explicit input and output paths:

```bash
PYTHONPATH=src python -m icl_pilot.evaluation.story_text_exports \
  --generation-source 4=data/generated/MY_RUN/4_year_old_rotating_story_generations.csv \
  --real-measures-csv data/inputs/MY_REFERENCE/dev_measures.csv \
  --real-stories-csv data/inputs/MY_REFERENCE/story_unit_narratives.csv \
  --output-csv data/generated/MY_RUN/evaluation/story_text_records.csv
```

Repeat `--generation-source` for additional cohorts. For the separate external
export, use `--gillam-external-only`, `--gillam-transcripts-dir` and
`--gillam-output-csv`. Selection, normalization and age-window defaults are
retained historical methods; review them before a revised study. These exporters
read their explicit sources and do not incorporate later CHAT repairs into CSV
text automatically.
