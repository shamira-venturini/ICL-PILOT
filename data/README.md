# Active data workspace

Historical generated CHAT, annotations and QC records were frozen externally
and removed locally after archive verification. This includes all 5,640 corrected
pre-annotation CHAT files and all 5,640 retained TalkTag files. Their exact paths,
hashes, repair decisions and validation records remain recoverable from the
[study-freeze index](../docs/provenance/study_freeze_2026-10-16.md).

This workspace is ready for inputs and outputs needed by the revised experiment.
New generation runs create separate folders under `generated/`; the combined
notebook currently uses `generated/direct_generation/<run_id>/`. Supply real
corpus inputs explicitly when rebuilding transcripts or measurements.

The preserved exemplar CSV, parsing-QC sidecar, roster and story scaffolds remain
in [`configs/generation/`](../configs/generation/). They support the working
baseline and tests; the revised generation protocol is pending. Analysis and
framework input paths are explicit pending settings under `configs/analysis/`
and `configs/evaluation/`. Historical data are never selected as a fallback.
