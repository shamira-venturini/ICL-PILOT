# Linguistic analysis inputs

The definitive roster is selected in inputs.R. Four derived feature files remain pending:

| Entry | Environment override |
|---|---|
| real_features_csv | ICL_PILOT_REAL_CSV |
| synthetic_features_csv | ICL_PILOT_SYN_CSV |
| frozen_roster_csv | ICL_PILOT_ROSTER_CSV |
| real_dev_csv | ICL_PILOT_REAL_DEV_CSV |
| synthetic_dev_csv | ICL_PILOT_SYN_DEV_CSV |

Paths may be absolute or relative to the repository root. A nonempty environment override takes precedence. All five entries must resolve to files before analysis starts. ICL_PILOT_PROJECT_ROOT selects an explicit checkout.

enni_metadata.R checks both real feature tables against the canonical participant manifest. Conflicting identities, groups or available source paths stop execution. Ages use completed CHAT months; sex follows the original demographics table. Metadata checks do not certify measurement lineage.

Create feature tables from the definitive study transcripts and newly annotated synthetic outputs, with documented source hashes and annotation/CLAN versions. The R notebook contains analysis code without empirical interpretation text. Reviewer-requested statistical revisions remain pending; running the existing formulas is not a completed validation of the domain analysis.

Validation:

    Rscript --vanilla tests/test_r_analysis_inputs.R
