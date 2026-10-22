# Story scaffolds as generation configuration — 14 October 2026

The canonical, project-authored ENNI content scaffolds now live in
`configs/generation/story_unit_reference.json`. The previous file under
`phase2/ENNI/measures/story_units/` was moved byte-for-byte; no duplicate
canonical copy remains. Its SHA-256 is
`e1cd483eb249e5ce80169cd73b43dee6456a104b0e0c1789af75c874647b353e`.

`experiment.json` schema 3 resolves the top-level `story_reference_json`
relative to the configuration directory. Only the human exemplar CSV is
resolved through `DATA_ROOT`. Colab instructions, layout checks, source-of-truth
documentation and the existing focused test fixtures were updated accordingly.
The fixtures use separate corpus and configuration roots to exercise this
path boundary. Existing prompt text, child identities and cohort settings
were preserved.

Verified external backup before the move:

`/Volumes/TranscendBU/ICL-PILOT_78a5cadad659_20261014T125103Z/GenerationScaffolds/generation_scaffolds_snapshot.tar.gz`

Archive SHA-256:
`a7aa511727a72c32f3103c6a613e5792384a106f41d04aff61dc00bc58392562`.

All archive members were decompressed and hashed against their sources.
`backup_verification.json` records the exact inventory and volume UUID.

Validation in `validation.json` and `validation_commands.json` confirms:

- 21 focused tests passed, with three pre-existing dependency warnings.
- All 2,820 full generation rows, including scaffold and exemplar text, and
  the complete evaluation manifest retain their pre-move fingerprints.
- The notebook prompt functions retain their original SHA-256.
- Notebook cells 5, 8 and 9 ran with an external data root containing only
  the human exemplar CSV; the scaffolds were loaded from configuration.
- 6,036 other protected research files and all three R source files are unchanged.
- Python/notebook syntax, CLI layout validation and `git diff --check` passed.

No raw corpus, generated narrative, annotation or scientific result was edited.
No model inference or scientific analysis was run. Temporary setup artifacts
were removed automatically. Existing unrelated worktree changes were preserved;
no files were staged, committed or published.
