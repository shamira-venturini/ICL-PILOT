# Single master roster consolidation — 14 October 2026

The user authorized replacing the four maintained cohort rosters with one
canonical master. The current source is
[`configs/generation/master_frozen_roster_4_to_7_year_old.csv`](../../../configs/generation/master_frozen_roster_4_to_7_year_old.csv).
It preserves all 47 pairs, cohort-local pair orders, and age/source metadata
from the reconciled cohort inputs. No matching optimization was run.

`experiment.json` schema 2 names the master once. The portable Colab setup
reads it once, filters by cohort, and derives cohort snapshots in run outputs.
Mixed-cohort scheduling is rejected. The four original cohort CSVs, the
obsolete older master, and its compatibility README were removed after
external backup verification. Both empty configuration subfolders were removed.
Only four files remain in `configs/generation/`.

## Backup

`/Volumes/TranscendBU/ICL-PILOT_78a5cadad659_20261014T123617Z/GenerationRoster/`

- `generation_roster_snapshot.tar.gz`: pre-change configurations, relevant code,
  notebook, documentation, tests, and current R Markdown.
  SHA-256: `1b1f568726b128bc53667603797e339df821f37c0605f1d9fbd10dfc36d54e05`.
- `r_path_references.tar.gz`: the older R Markdown and existing extracted R file,
  backed up before their roster path references changed.
  SHA-256: `616a3aa33a992ed89f8039ceb6d9070b6fe13d9cd898371ebf5c616576874143`.

Every archived member was decompressed and hashed against its source.
`backup_verification.json`, `supplemental_backup.json` and `deletions.json`
record the source hashes and exact retired paths. No local archive duplicate
was created. The backup follows the existing project-prefix/timestamp style.

## Checks

- 21 focused tests passed, with three pre-existing dependency warnings.
- All 2,820 job identities match the saved original generations.
- Full generation schedules and rotating holdout manifests match their
  pre-consolidation CSVs byte for byte.
- The master rows match every backed-up reconciled cohort roster.
- Notebook input/setup cells 5, 8 and 9 ran on CPU using the real prompt inputs.
  Per-cohort run snapshots match the master subsets; prompt-function text
  retains its original SHA-256.
- R main/developmental membership assembly ran again. Its exported memberships
  match the previous audit exactly, preserving zero within-round overlap
  across 17 age-specific rounds. Only roster path references changed in the
  three R files; no models, scientific outputs, or rendered reports were run.
- 6,037 protected research files were rehashed; none changed.
- CLI layout/manifest checks, Python/notebook syntax, and `git diff --check` passed.

Machine-readable evidence is in `validation.json` and `validation_commands.json`.
Historical Python four-year-old split problems and descriptive full-reference
R comparisons retain the limitations documented in the earlier separation audit.
CUDA model loading/inference was not run locally. This consolidation validates
setup and identity preservation, not new model outputs or scientific results.

No changes were staged, committed or published. Existing unrelated worktree
changes were preserved.
