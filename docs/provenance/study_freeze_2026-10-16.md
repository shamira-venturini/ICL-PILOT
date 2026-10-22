# Study freeze — 16 October 2026

The author authorized a complete as-is snapshot followed by retirement of the
previous study's local artifacts. The revised scientific protocol is pending.

## Verified snapshot

- Archive: `/Volumes/TranscendBU/ICL-PILOT_af80b106b6dd_20261016T135824Z/StudyFreeze/repository_as_is.tar.gz`
- SHA-256: `69ceb14329c0404ce0cf017493385d1a2257d7fdab52f6e91a040529fcf3c9b0`
- 66,841 regular-file hashes and three symlink targets verified by read-back.
- The complete source inventory was rechecked before any restructuring.
- Includes hidden files, Git history/uncommitted state, `.venv`, R workspaces,
  source, configs, notebooks, data, results and audit records.
- `MANIFEST.json`, `VERIFICATION.json` and `SHA256SUMS` are beside the archive.
- `EARLIER_ARCHIVES.json` indexes 54 earlier archives, including research artifacts
  already retired before this snapshot. Those archives were indexed, not freshly
  reverified in this task. Their manifests preserve earlier source states.

## Local retirement

20,222 historical/cache files were matched to the full snapshot and removed
locally: Phase 2, generated CHAT and its QC, old result payloads, outputs,
older R variants, rendered HTML, R workspaces/history and disposable metadata.
`RETIREMENT.json` beside the archive contains the exact paths, hashes and sizes.
Approximately 253.4 MiB of allocated local space was released.

Reusable source, frozen generation inputs/prompts/roster/scaffolds, documentation,
tests and the verified Chatter tool remain. The configured `.venv` remains because
PyCharm points to it. Its packages, the local Python packages and R environment
were recorded beside the archive. Its actual Python version is 3.12.8, while the
IDE label says 3.13; neither is asserted as a historical generation runtime.

## Retained repository — completed 20 October 2026

- Maintained R Markdown moved to `notebooks/clinical_analyses_reordered.Rmd`.
- Framework inputs now use the pending `configs/evaluation/inputs.json`.
- CSV exporters require explicit input/output paths; Phase 2 defaults removed.
- The unused default-roster wrapper that targeted Phase 2 was removed.
- 270 historical notebook output blocks were archived and cleared.
- Working-tree docs/status notices identify the freeze, retained interfaces and
  pending revised protocol.
- On resuming on 19 October, all 312 Python tests passed again, including native
  Chatter regressions, with three pre-existing dependency warnings. Base-R input
  tests passed, including root discovery when knitting from `notebooks/`.
- All Python source/test files parse with Python 3.8 syntax; the retained R
  Markdown's 533 expressions parse without executing the statistical analysis.
  Layout validation and all 60 maintained local documentation links pass.
- The 2,820 generation jobs, evaluation manifest and 5,640 rendered stage-message
  lists match the recorded pre-freeze fingerprints. Stage 2 rendering uses a
  fixed verification parent, without inference. The fingerprints also match the
  external saved baseline, checked directly on 20 October.
- All ten existing non-README configuration payloads are byte-identical to the
  full snapshot. The generation notebook's 16 code cells are unchanged; only
  the framework notebook's input/settings cell changed.
- 346 existing Python function/class definitions are unchanged. Only explicit
  CSV input/output handling and removal of an unused default-roster wrapper
  alter existing definitions; a new framework input resolver was added.
  Of 50 original Python module files, 48 are byte-identical; the two changed
  modules contain those documented path-handling changes.
- The R comparison confirms that all 526 expressions outside the setup chunk
  are unchanged except for three output-path expressions. No statistical model
  specification changed during this restructuring.
- The full archive SHA-256 was checked again on 20 October and is unchanged.
- No models, GPU generation, TalkTag inference or scientific analyses were rerun.
- No files were staged, committed or pushed.

Work was paused at the author's request and completed after reconnecting
TranscendBU on 20 October. No further historical data were removed on resuming.
The original `RESTRUCTURE_CHECKPOINT/` remains beside the full archive and
preserves the comparison baseline.

## Compact retained-source checkpoint

The cleaned repository is separately preserved in:

`/Volumes/TranscendBU/ICL-PILOT_af80b106b6dd_20261016T135824Z/StudyFreeze/POST_RESTRUCTURE_2026-10-20/retained_repository.tar.gz`

This checkpoint contains retained source, configs and frozen exemplar inputs,
notebooks, tests, documentation, project metadata, the data/results signposts,
and the Chatter executable/source record. It excludes `.git`, `.venv`, IDE
settings, caches and historical data/results already in the full freeze.
The completion index and JSON receipt are stored alongside this compact archive
as `STUDY_FREEZE_INDEX.md` and `RESTRUCTURE_STATUS.json`, rather than inside it,
so they can record its final verification without a circular archive hash.
`MANIFEST.json`, `VERIFICATION.json` and `SHA256SUMS` document its file-by-file
read-back verification. Comparison reports and the verification scripts are
retained in its `checks/` directory.

The local machine-readable companion `study_freeze_2026-10-16.json` records the
completed checks and both archive locations. The full snapshot remains the
source for Git state, the configured environment and historical study artifacts.

## Restoration

Restore into a separate empty destination. Archive members have an `ICL-PILOT/`
prefix and include `.git` and ignored files. For example:

```bash
tar -xzf "/Volumes/TranscendBU/ICL-PILOT_af80b106b6dd_20261016T135824Z/StudyFreeze/repository_as_is.tar.gz" -C "/path/to/empty-destination"
```

This is an exact working-tree freeze, not a claim that every historical result
was generated by the current cleaned source. Use each artifact's provenance and
the earlier archive index when reconstructing older runs.
