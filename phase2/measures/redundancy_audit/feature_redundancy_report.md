# Feature Redundancy Audit

- Input: `/Users/shamiraventurini/PycharmProjects/ICL-PILOT/phase2/measures/dev_measures.csv`
- Rows: `360`
- Columns: `85`
- Numeric columns considered for pairwise audit: `60`
- High-correlation threshold: `0.95`
- Cluster threshold: `0.98`

## Constant Columns
- `Language`
- `Corpus`
- `Code`
- `Race`
- `SES`
- `Role`
- `Education`
- `Custom_field`
- `Design`
- `Activity`
- `MLU50_Utts`
- `Utt_Errors`
- `DSS_tbl_command`
- `DSS_tbl_score_is_na`

## ID-Like Columns
- `File_kideval`
- `File_ID`
- `File_mlt`
- `DSS_tbl_source`
- `DSS_tbl_relpath`
- `DSS_tbl_transcript`
- `DSS_tbl_transcript_relpath`

## Exact Duplicate Groups
- `constant-only`: `DSS_tbl_score_is_na, Utt_Errors`
- `substantive`: `FREQ_tokens, mor_Words`
- `constant-only`: `Custom_field, Education, Race, SES`

## High-Correlation Clusters
- `#utterances, MLU_Utts, TD_Utts, Total_Utts`
- `#words, FREQ_tokens, TD_Words, mor_Words`
- `Age(Month), Age_Month_approx`
- `DSS, DSS_tbl_TOT, DSS_tbl_score`
- `MLU50_Morphemes, MLU50_Words`
- `MLU_Morphemes, MLU_Words, words/utterances`

## Top High-Correlation Pairs
- `FREQ_tokens` vs `mor_Words`: pearson=1.0000, spearman=1.0000, exact_equal=1, max_abs_diff=0.000000
- `DSS` vs `DSS_tbl_TOT`: pearson=1.0000, spearman=0.9999, exact_equal=0, max_abs_diff=834.960000
- `DSS` vs `DSS_tbl_score`: pearson=1.0000, spearman=0.9999, exact_equal=0, max_abs_diff=0.180000
- `TD_Utts` vs `#utterances`: pearson=1.0000, spearman=0.9999, exact_equal=0, max_abs_diff=2.000000
- `Age(Month)` vs `Age_Month_approx`: pearson=0.9999, spearman=0.9998, exact_equal=0, max_abs_diff=1.000000
- `Total_Utts` vs `#utterances`: pearson=0.9998, spearman=0.9997, exact_equal=0, max_abs_diff=6.000000
- `Total_Utts` vs `TD_Utts`: pearson=0.9998, spearman=0.9997, exact_equal=0, max_abs_diff=6.000000
- `MLU_Words` vs `words/utterances`: pearson=0.9993, spearman=0.9992, exact_equal=0, max_abs_diff=0.343000
- `Total_Utts` vs `MLU_Utts`: pearson=0.9990, spearman=0.9988, exact_equal=0, max_abs_diff=15.000000
- `FREQ_tokens` vs `#words`: pearson=0.9985, spearman=0.9981, exact_equal=0, max_abs_diff=105.000000
- `mor_Words` vs `#words`: pearson=0.9985, spearman=0.9981, exact_equal=0, max_abs_diff=105.000000
- `MLU_Utts` vs `TD_Utts`: pearson=0.9984, spearman=0.9981, exact_equal=0, max_abs_diff=16.000000

## Recommended Exclusions Before Severity Modeling
- `Activity`
- `Code`
- `Corpus`
- `Custom_field`
- `DSS_tbl_command`
- `DSS_tbl_relpath`
- `DSS_tbl_score_is_na`
- `DSS_tbl_source`
- `DSS_tbl_transcript`
- `DSS_tbl_transcript_relpath`
- `Design`
- `Education`
- `File_ID`
- `File_kideval`
- `File_mlt`
- `Language`
- `MLU50_Utts`
- `Race`
- `Role`
- `SES`
- `Utt_Errors`
- `mor_Words`
