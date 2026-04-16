# Feature Redundancy Audit

- Input: `/Users/shamiraventurini/PycharmProjects/ICL-PILOT/phase2/measures/severity_profile_table.csv`
- Rows: `360`
- Columns: `35`
- Numeric columns considered for pairwise audit: `30`
- High-correlation threshold: `0.95`
- Cluster threshold: `0.98`

## Constant Columns
- none

## ID-Like Columns
- `File_ID`
- `File_kideval`
- `profile_dss_subscales`

## Exact Duplicate Groups
- `substantive`: `DSS_tbl_warning_50, severity_use_with_caution`
- `substantive`: `VOCD_D_optimum_average_severity_z, severity_lexical`

## High-Correlation Clusters
- `DSS_tbl_warning_50, severity_use_with_caution`
- `VOCD_D_optimum_average_severity_z, severity_lexical`
- `Word_Errors_per_utt_severity_z, severity_morphosyntax_burden`

## Top High-Correlation Pairs
- `DSS_tbl_warning_50` vs `severity_use_with_caution`: pearson=1.0000, spearman=1.0000, exact_equal=1, max_abs_diff=0.000000
- `VOCD_D_optimum_average_severity_z` vs `severity_lexical`: pearson=1.0000, spearman=1.0000, exact_equal=1, max_abs_diff=0.000000
- `Word_Errors_per_utt_severity_z` vs `severity_morphosyntax_burden`: pearson=0.9861, spearman=0.9260, exact_equal=0, max_abs_diff=8.185054
- `DSS_tbl_score_severity_z` vs `severity_structural`: pearson=0.9708, spearman=0.8832, exact_equal=0, max_abs_diff=3.252675
- `DSS_tbl_MV_severity_z` vs `severity_structural`: pearson=0.9677, spearman=0.8540, exact_equal=0, max_abs_diff=18.335483
- `severity_core_composite` vs `severity_domain_composite`: pearson=0.9585, spearman=0.8853, exact_equal=0, max_abs_diff=4.886489
- `severity_morphosyntax_burden` vs `severity_core_composite`: pearson=0.9568, spearman=0.6343, exact_equal=0, max_abs_diff=10.484712
- `severity_morphosyntax_burden` vs `severity_domain_composite`: pearson=0.9528, spearman=0.6127, exact_equal=0, max_abs_diff=12.713490

## Recommended Exclusions Before Severity Modeling
- `File_ID`
- `File_kideval`
- `profile_dss_subscales`
- `severity_lexical`
- `severity_use_with_caution`
