# Profiling backup and later cleanup

Current location: `/Volumes/TranscendBU/ICL-PILOT_78a5cadad659_20261014T092052Z/Profiling`.

Folder naming follows the previous project backup: `ICL-PILOT_78a5cadad659_<UTC timestamp>`, with `Profiling` as the research-area subfolder. The verified tar archive is unchanged. `CURRENT_LOCATION.json` records this relocation; the immutable archive and original manifest retain the creation location for provenance.

This archive was created on 14 October 2026. No local research files were deleted.

The archive contains current files under `repository_snapshot/`, including
ignored/untracked files, plus metadata and three older severity-model files
recovered from Git HEAD under `git_HEAD_deleted/`. The latter were already
absent locally and have not been restored into the repository.

`MANIFEST.json` records every current source path, content SHA256, role, size,
and deletion-review status. `VERIFICATION.json` records the completed checks.
`profiling_candidates.tsv` is a conservative review inventory, not permission
to delete everything included in this backup.

Keep active raw feature tables (`severity_features_raw_*.csv`), feature builders
(`severity_features.py`, `generated_severity_features.py`), developmental/CLAN
inputs, frozen rosters, counterbalancing records, current R analysis, and
generation notebooks locally. Their names or profiling metadata do not make
them unused profiling analyses.

Before removing dedicated profiling modules, remove/update their imports and
CLI commands in `src/icl_pilot/cli.py`. `frozen_roster.py` and
`generation/workflow.py` currently load the profile table, require its metadata,
and import helpers from `story_generation_design.py`; `architecture_pilot_cv.py`
also imports helpers there. Preserve frozen child/pair/round assignments while
decoupling these dependencies. Reproduce the existing 47-pair roster and
generation schedule without using severity/profile criteria, and verify
unchanged IDs, ordering, age matching, and exclusions before deleting inputs.

Dedicated profiling outputs and documentation can be archived after their
references are reviewed. Keep these historical findings separate from the
current paper: derived severity/profile scores do not condition the active
prompts or enter the current RStudio models. Older exploratory severity models
are retained as provenance, not current paper-facing analyses.

The backup also includes shared context and upstream feature artifacts so the
profiling work can be reconstructed. Raw human transcripts, official ENNI
pictures, complete generated-text collections, and the whole Git object database
are outside this scoped backup; do not remove them on the strength of this
backup. The checked Git commit and dirty-worktree status are recorded.

To inspect archive paths: `tar -tzf profiling_snapshot.tar.gz`.
To verify the external package: run `shasum -a 256 -c SHA256SUMS` here.
To restore for review, extract to a new empty directory with
`tar -xzf profiling_snapshot.tar.gz -C /path/to/empty/restore-directory`.
Then select files from `repository_snapshot/` without overwriting newer work.
