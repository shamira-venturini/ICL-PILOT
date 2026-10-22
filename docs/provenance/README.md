# Repository cleanup provenance

**The previous study is now frozen externally.**
[`study_freeze_2026-10-16.md`](study_freeze_2026-10-16.md) identifies the complete
working-tree snapshot, local retirement, earlier archives and restoration steps.
The machine-readable companion records verification and retained-code checks.
Older paths in the records below describe historical states, not current inputs.

[`generation_termination_audit_2026-10-21/`](generation_termination_audit_2026-10-21/README.md)
reports decoded-text/token-proxy screening of all 5,640 original stage outputs,
with source verification, cohort/story summaries and 64 boundary-tail reviews.
It distinguishes possible ceiling effects from confirmed stopping reasons,
which the historical CSVs did not record.

[`real_story_tokenizers_2026-10-21/`](real_story_tokenizers_2026-10-21/README.md)
compares real ENNI story lengths using the pinned official Llama, historical
reference and proposed Qwen tokenizers. It distinguishes saved CHAT notation
from a light descriptive cleanup. The official file was verified using a public
copy whose Git blob hash matches the official revision exactly; Colab execution
is no longer needed. The subsequent author-approved change to 640 new tokens
per stage is recorded in
[`generation_ceiling_update_2026-10-21.json`](generation_ceiling_update_2026-10-21.json),
including before/after config hashes and validation. Earlier reports retain
their historical settings.

[`generation_sampling_update_2026-10-21.json`](generation_sampling_update_2026-10-21.json)
records the author's subsequent selection of temperature 0.8 and top-p 0.9 for
both stages, the prior values and validation. The cited motivation is the
author's review of earlier research; no specific reference was supplied in the
instruction. The ceiling remains 640 new tokens and top-k remains inherited.

The first approved change after the freeze is recorded in
[`counterbalancing_update_2026-10-20.json`](counterbalancing_update_2026-10-20.json).
It identifies the revised story assignments, their previous archived state,
configuration hashes and validation. The freeze itself retains the old schedule.

[`model_source_correction_2026-10-20.json`](model_source_correction_2026-10-20.json)
records the subsequent switch to a pinned official Llama checkpoint, the
publisher's description of the previous community conversion and the source and
software checks. It preserves the distinction between today's repository
metadata and the unknown historical installed revision.

This folder preserves 119 selected audit records from 14–16 October 2026,
occupying approximately 5 MiB. It includes the generation baseline, recorded
pair discrepancies, prompt/evaluation separation checks, input-relocation
checks, backup verification and transcript-repair summaries.

These are historical records. Their original contents, paths and descriptions
are preserved; files or scripts mentioned within them may now be archived
externally. Use [`current_strategy_handoff.md`](../current_strategy_handoff.md)
for current decisions and [`configs/generation/`](../../configs/generation/) for
retained generation inputs. Historical transcript QC is inside the verified
study freeze at its original `data/qc/` paths; use the archive index above.

The complete former `outputs/maintenance/` and
`outputs/manual-20260601-icl-update/` directories were archived on TranscendBU,
with every regular-file hash and symlink verified before local removal.
[`outputs_cleanup_2026-10-16.json`](outputs_cleanup_2026-10-16.json) identifies
the archive, its SHA-256 and the exact source/hash of each retained local report.
The external archive also preserves omitted inventories, one-off audit scripts,
test fixtures, intermediate exports, presentation files and validator caches.

The working Chatter executable and its source record now live under
[`tools/chatter/`](../../tools/chatter/). The framework-evaluation bundle was
subsequently moved unchanged to the historical
`results/framework_based_evaluation/run_20260827T153330Z/` location;
`framework_results_relocation_2026-10-16.json` records its 180 file hashes.
That bundle and the former `outputs/` annotation-QC records were then included
in the complete study freeze and removed locally. Use the archive index above
to restore them into a separate workspace.

The full `phase2/` backup and first cleanup are recorded in
[`phase2_cleanup_2026-10-16.json`](phase2_cleanup_2026-10-16.json).
[`replications_removal_2026-10-16.json`](replications_removal_2026-10-16.json)
records the subsequent removal of the externally verified `phase2/replications/`
workspace, including its scripts and historical results.

[`source_reorganization_2026-10-16.json`](source_reorganization_2026-10-16.json)
records the Python package moves, verified pre-edit source snapshot and checks
that prompts, schedules, retained data and scientific function bodies were preserved.

[`r_input_signpost_2026-10-16.json`](r_input_signpost_2026-10-16.json) records the
replacement of hardcoded R inputs with the shared pending-input configuration,
including the pre-edit backup and checks that statistical expressions were preserved.
