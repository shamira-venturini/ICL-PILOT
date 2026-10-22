# Generation configuration cleanup — 14 October 2026

The author confirmed a shared generation protocol for ages four through seven
and chose Colab as the primary generation interface. This cleanup changes
generation setup/configuration and its inspection utilities. Existing generated
text, CHAT annotations, feature tables, R source, and scientific results are
unchanged. No model was loaded and no CUDA inference was run locally.

## Backup and retirement

Verified external backup:

`/Volumes/TranscendBU/ICL-PILOT_78a5cadad659_20261014T115428Z/Generation/`

- Archive: `generation_snapshot.tar.gz` (193,485 bytes).
- SHA-256: `f911ebe4cea11c148b6e11d80d71ed6e6179ea02fa92efaf3a3a379d7b15c280`.
- All **106 archived source/configuration/documentation files** were reopened,
  hashed, and compared with the local pre-change files after external transfer.
- **54 unchanged archived files were removed locally**: 50 old configuration
  files (including Finder metadata) and four obsolete generation-design notes.
  Removed source size: 859,718 bytes. Detailed paths/hashes are in
  `deletion_plan.json`; completion is recorded in `deletions.json`.
- The temporary local archive was removed after external verification. Local
  inventory and audit sidecars remain; this does not duplicate corpus data.

The retained counterbalancing table and legacy combined roster were not rewritten.
The combined roster remains at its old path because R still reads it; it is
clearly labelled as compatibility material rather than a generator input.

## Active generation sources

- `configs/generation/experiment.json`: common settings, relative input paths,
  author-confirmed protocol provenance, ten replicates, three pairs per round.
- `configs/generation/rosters/`: four frozen rosters reconstructed from actual
  recorded child identities and order, without severity/profile fields.
- `notebooks/colab_direct_generation_pipeline.ipynb`: settings, checked input
  preparation, CUDA loading, unchanged prompt functions, generation and exports.
- `src/icl_pilot/generation/config.py`: portable deterministic setup, required
  exemplar validation, round/child separation checks, and run input fingerprints.
- `README.md`, `configs/generation/README.md`, and `docs/colab_generation_package.md`:
  current entry points and instructions.

The new four-year-old roster uses SLI 474 / TD 459 and SLI 475 / TD 417, preserving
what was actually generated. Age/KIDEVAL metadata are joined by child ID. No
new pairing optimization or retrospective narrative regeneration was performed.
The original mistaken roster is preserved in the archive. Original CHAT exports
with missing ages remain unchanged; the new setup exports the known TD ages
correctly for new runs, as verified with test fixtures.

The Colab notebook derives all settings from one configuration, includes age
four by default, uses unique run folders, protects existing generation CSVs,
and records input/configuration/code hashes, the resolved model revision and
runtime package versions. Installation requests generation dependencies rather
than the full analysis dependency set. Per-job inference resume and a pinned
historical model revision are not established by this initial cleanup.

`show-manifest` now compiles the current frozen design instead of displaying
the invalid report-era CSV. The layout check inspects required code/config
files; the notebook separately checks the two corpus-derived prompt inputs.
The generic pairing builder refuses to overwrite existing frozen rosters.

## Validation and boundaries

- Notebook settings/input cells executed locally with actual prompt inputs.
- **All 2,820 jobs match original saved identities exactly**: pair, child IDs,
  round, replicate, target story, age field, and exemplar slots.
- **17 focused tests passed**, including missing/duplicate exemplars, overlap
  detection, frozen-roster preservation, run collision protection, counterbalance
  validation, preserved prompt functions and the two reconciled TD export ages.
- CLI layout/manifest checks, notebook syntax checks and `git diff --check` passed.
- Original prompt-function SHA-256 remains
  `6f85397963bd0b66024e323f9d358b8380cff96fdb20aeef073e85331ab569dd`.
- The separate membership audit found no overlap in the 17 round-stacked main
  and developmental R pools. It also documents 40 contaminated legacy Python
  comparisons and full-reference descriptive R sections. Analyses/results were
  neither corrected nor rerun.
- The existing local environment cannot perform the notebook's model-loading
  import (`BitsAndBytesConfig` is absent from its installed transformers).
  CUDA execution still requires a Colab run. GPU inference is not certified by
  the CPU setup tests. Existing pandas/numexpr warnings are recorded.

Machine-readable records: `external_verification.json`,
`portable_roster_provenance.json`, `portable_setup_validation.json`,
`validation_commands.json`, and `final_preservation.json`.

The substantial pre-existing dirty Git worktree was preserved. Nothing was
staged, committed, pushed, or published.
