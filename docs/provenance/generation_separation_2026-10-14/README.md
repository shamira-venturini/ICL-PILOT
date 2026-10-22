# Prompt/evaluation separation check

The original generation CSVs define the actual prompt children and rounds.
Their metadata agree with all 940 rows in the current synthetic feature table.
`check_r_memberships.R` executes only the unchanged R Markdown's main and
developmental data-assembly expressions. It stops before any statistical model,
plot, or report render and writes identity-only membership tables here.
`audit_memberships.py` compares those memberships independently against the
original generation records.

- Ages four–seven contain 17 distinct age-specific rounds.
- Both round-stacked R data assemblies have **zero within-round prompt/evaluation
  child overlaps**, including a check that ignores group labels.
- The historical four-year-old Python pool under `bundle_round_holdout` contains
  TD 459 in round 2 and TD 417 in round 3, although those children were prompts.
  The saved pairwise-distance table includes 20 comparisons for each child,
  **40 contaminated comparisons** in total. Saved pools/results were not repaired.
- The R PCA/correlation (`pca_base`) and distributional (`profile_df`) sections
  independently use full-reference rows without round-specific exclusions.
  They are descriptive reference comparisons, not certified held-out evaluation.
  The round-stacked pass must not be generalized to the whole R report.

No scientific model or metric was recomputed. Existing R source, tables,
generated narratives, and result files remained unchanged. Exact checks and
input hashes are in `separation_audit.json`.
