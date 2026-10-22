# R analysis inputs — awaiting the revised study

**Signpost:** update [`inputs.R`](inputs.R) when the revised datasets and their
paths are available. All five paths are deliberately blank. The R entry points
stop before reading data while an entry is pending or its file is missing.
There is no automatic fallback to Phase 2 data or the historical prompt roster.

| Entry | File to supply | Optional environment override |
| --- | --- | --- |
| `real_features_csv` | Raw linguistic feature table for the revised real-reference sample. | `ICL_PILOT_REAL_CSV` |
| `synthetic_features_csv` | Corresponding synthetic feature table, including prompt ancestry and round metadata. | `ICL_PILOT_SYN_CSV` |
| `frozen_roster_csv` | Frozen prompt-pair roster for that same experiment. | `ICL_PILOT_ROSTER_CSV` |
| `real_dev_csv` | Real developmental/CLAN measures used by the morphology analyses. | `ICL_PILOT_REAL_DEV_CSV` |
| `synthetic_dev_csv` | Corresponding synthetic developmental/CLAN measures. | `ICL_PILOT_SYN_DEV_CSV` |

Paths can be absolute or relative to the repository root. A nonempty environment
override takes precedence for its entry. All five entries must resolve to files;
partial overrides cannot silently select older data for the remaining inputs.
The existing `ICL_PILOT_ANALYSIS_TAG` plot suffix in the main R Markdown is retained.

The maintained R Markdown finds the checkout by searching upward from its working
directory, including when knitting from `notebooks/`. When running from elsewhere,
set `ICL_PILOT_PROJECT_ROOT` to the checkout path. There are no machine-specific paths.

`notebooks/clinical_analyses_reordered.Rmd` is the maintained R analysis. The
older R Markdown variant, extracted script and rendered reports are frozen
externally. Previously, those variants differed in their real-reference defaults;
future corpus choice is explicit in this configuration. Statistical code and
interpretations remain available for protocol review. R figures now go to dated
folders under `results/linguistic_evaluation/`. Render reports under `results/`
when the revised inputs are ready.

This configuration checks file availability. Compatibility of schemas, annotation
versions, experiment ancestry and revised statistical assumptions still needs to
be assessed when the new inputs exist. Setting paths does not perform that review.

Check input handling without running any analyses:

```bash
Rscript --vanilla tests/test_r_analysis_inputs.R
```
