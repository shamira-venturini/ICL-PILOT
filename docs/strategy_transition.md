# Strategy Transition

The earlier repository state mixed three different concerns in one place:

- immutable source corpora
- exploratory phase-1 prompting experiments
- ad hoc extraction and evaluation scripts

The revised strategy in the February 2026 project report changes the project shape enough that incremental cleanup was not sufficient. The repository is now organized around the next pipeline:

1. Extract and normalize raw ENNI material.
2. Build age-matched generation configurations.
3. Generate stage-1 TD baselines.
4. Generate stage-2 DLD variants from those baselines.
5. Convert, annotate, and evaluate outputs reproducibly.

## Checked-in configuration

Two report-derived configuration files are now versioned under `configs/generation/`:

- `cross_story_counterbalancing.csv`: Table 3 from the report, encoded as target-story to exemplar-story rules.
- `sample_wise_multistep_manifest.csv`: Table 4 from the report, encoded as the 24 generation configurations.

## Important note

The PDF extraction surfaced a few ambiguous subject IDs in the 5-year-old cohort:

- `528 2`
- `568 2`
- `576 2`

These values are preserved verbatim in the manifest for traceability. Before using the manifest for production generation, they should be normalized against the source transcripts in `data/raw/`.
