# Generation baseline — 14 October 2026

Step 1 only: record the generation implementation, inputs, executed sample
structure, and existing failures before cleanup. No configuration, production
code, source transcript, generated narrative, annotation, analysis, or result
was changed. No model was downloaded or loaded. No generation, annotation,
feature extraction, or scientific analysis was run.

This directory is a checksum and provenance baseline, **not an external backup**.
It contains no copies of corpus transcripts or generated narratives. The
inventoried source files occupy 20,388,339 bytes; the audit itself is about 2 MB.
The external drive was not mounted, and backup/deletion steps have not begun.

## Evidence and sources of truth

- `inventory.csv`: SHA-256, size, and role for **6,116 existing files**,
  including all 52 configuration-folder files, the active notebook and relevant
  source, direct generation inputs, referenced source transcripts, original
  generation CSVs, and unannotated generated CHAT exports.
- `baseline.json`: environment, Git state, cohort checks, semantic schedule
  fingerprints, and a second hash pass confirming that all 6,116 files remained
  unchanged.
- `observed_prompt_pairs.csv`: the **47 pairs actually present in saved
  generation CSVs**, with their recorded rounds. This is audit evidence, not a
  replacement frozen roster.
- `observed_job_fingerprints.csv`: metadata and separate Stage 1/Stage 2 text
  hashes for each of the **2,820 saved paired-stage jobs**. Text is not copied.
- `current_prompt_functions.py.txt`: exact prompt-function source from current
  notebook cell 12. This freezes the current templates; it does not establish
  which source version was executed for past generations.
- `dependency_map.json`: generation inputs, readers, producers, historical
  consumers, and existing notebook write behaviour.
- `commands.json`: exact baseline-check commands, exit codes, stdout, and stderr.
- `capture_baseline.py`: the audit implementation. It refuses to overwrite an
  existing baseline. To repeat later, select a **new** output directory:

  ```bash
  PYTHONDONTWRITEBYTECODE=1 python \
    outputs/maintenance/generation_baseline_2026-10-14/capture_baseline.py \
    --output-dir outputs/maintenance/generation_baseline_comparison_NEW
  ```

For what was generated, prioritize the original paired-stage CSVs and their
recorded identifiers. For the current implementation, use the captured source
hashes and notebook settings. Do not silently equate these evidence sources.

## Generation flow

1. Developmental child metadata can create an age-only frozen roster. Existing
   cohort rosters are reused by the current package builder.
2. The roster and `cross_story_counterbalancing.csv` determine pair rounds,
   target-story jobs, opposite-set exemplar slots, and split bookkeeping.
3. `phase2/ENNI/measures/story_units/story_unit_reference.json` supplies the
   project-authored, picture-derived content scaffold.
4. `phase2/ENNI/measures/story_units/story_unit_narratives.csv` supplies natural
   exemplar text by `(File_ID, Group, story_id)`.
5. Notebook cells 12–15 construct TD-like prompts, generate Stage 1 text, and
   insert that text into the Stage 2 transformation prompt. Both outputs are
   saved in one CSV row.
6. `generated_chat.py` converts paired-stage rows into separate stage-specific
   CHAT files. Annotation and subsequent processing are outside this baseline.

The primary generation unit remains one target story. A pair/replicate groups
six stories into a bundle. Replicates do not represent independent human
participants. Current prompt functions contain no explicit child-age or
severity/profile fields; age matching acts through exemplar selection. The
notebook filename's `direct` refers to native model execution, not a one-stage
generation architecture.

## Observed generations

| Age | Recorded pairs | Rounds | Replicates per pair | Paired-stage jobs | Narratives across both stages |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 11 | 4 | 10 | 660 | 1,320 |
| 5 | 12 | 4 | 10 | 720 | 1,440 |
| 6 | 11 | 4 | 10 | 660 | 1,320 |
| 7 | 13 | 5 | 10 | 780 | 1,560 |
| Total | 47 | 17 within-age rounds | — | 2,820 | 5,640 |

Original CSV locations:

- Age 4: `data/annotated/ENNI_batchalign/four_year_old_rotating_story_generations.csv`.
- Age 5: `data/generated/5_year_old/5_year_old_cha/5_year_old_rotating_story_generations.csv`.
- Age 6: `data/generated/6_year_old/0_6_year_old_cha/6_year_old_rotating_story_generations.csv`.
- Age 7: `data/generated/7_year_old/0_7_year_old_cha/7_year_old_rotating_story_generations.csv`.

Across these CSVs, every job key is unique, both stage texts are nonempty, each
bundle has exactly A1–A3 and B1–B3, exemplar slots obey the counterbalance table,
and saved numbered story spines match the current reference JSON. Each cohort
also has the expected number of stage-specific unannotated CHAT files. These
are structural checks, not claims of linguistic or clinical validity.

## Discrepancies that must survive cleanup

### Four-year-old recorded pairing differs from the frozen roster

| SLI child | TD child in saved generations | TD child in current frozen roster | Recorded round |
| --- | --- | --- | --- |
| 474 | 459 | 417 | round_02 |
| 475 | 417 | 459 | round_03 |

The combined ages-4–7 roster agrees with the individual frozen roster, not
with these two recorded generation pairs. This affects **120 target-story
jobs** (two pairs × ten replicates × six stories), or 240 stage outputs.
The actual TD prompt-child membership therefore also differs in rounds 2
and 3. The cause is not established by this baseline. Neither version was
rewritten, and no new authoritative pairing was selected.

All other shared four-year-old job metadata agree with the current schedule.
Ages five, six, and seven match the current ten-replicate schedule on every
checked job key, pair, round, age field, and exemplar slot.

The saved CHAT manifest and a read-only scan of unannotated CHAT headers both
show **120 missing Stage 1 age fields** for the two discrepant four-year-old
pairs, 60 files per pair. No Stage 2 age fields are missing, and no missing
ages were found in the corresponding raw exports for ages five through seven.
The converter obtains TD age from a matching roster entry when the CSV lacks
it; the mismatched pair IDs are consistent with this failure mode. Historical
execution details have not been recovered, so this is an explanation supported
by current code and saved metadata, not proof of the original execution path.
No CHAT files or manifests were repaired. Downstream effects were not examined.

### Saved packages versus executed sample size

The five saved age-specific packages (ages 5–9) contain one replicate per pair.
They match the current builder when it is called with `n_replicates=1`.
The observed CSVs for ages 4–7 instead contain replicates 1–10. These package
summaries must not be cited as the completed experiment's output counts.
There is no four-year-old rotating package directory alongside the 5–9
packages; only its standalone and combined frozen rosters are present there.
The 8- and 9-year-old configurations are recorded as configurations only;
this audit did not locate corresponding original generation CSVs in the
inspected generation locations.

### Historical model execution is incompletely recorded

The original generation CSVs have no model, model revision, seed, temperature,
top-p, or token-limit columns. They preserve selected child/story identifiers
and numbered story spines, but not full rendered prompts or exemplar-text
snapshots. The current notebook has no executed cells or saved cell outputs.
Consequently its source documents the current implementation without proving
all historical run settings.

Current executable notebook settings include:

- Model ID `avans06/Meta-Llama-3.2-8B-Instruct`; no pinned model/tokenizer revision.
- Seed 42; sampling enabled; temperature 0.7; top-p 0.95; 180 new tokens per stage.
- GPU loading requests NF4 4-bit quantization; CPU loading requests float32.
- Ten replicates; three prompt pairs per round; configured ages 5–9.
- `START_COHORT_LABEL = "6-year-old"`, which currently skips age 5 during the loop.
- Periodic CSV writes every 25 jobs; existing aggregate CSV removed at loop start.
- The schedule's `generation_seed` is not passed to individual inference calls.
  Notebook seeding initializes a shared random stream. Starting at another
  cohort or rerunning a subset therefore does not establish identical outputs.

The notebook's explanatory model-ID text also differs from its executable
model ID. No online model documentation was fetched during this local audit.

### Remaining historical dependencies

The original report manifest is still the default for `show-manifest` and is
included in the layout check. Its 24 configurations are not the active notebook
schedule; 14 exemplar assignments conflict with the current counterbalance
table. Historical pilot transcript paths are missing locally. The five
profiling-grid files and four LOOCV records remain configuration artifacts.
They require verified backup and consumer updates before removal in a later
step. Official pictures and authored scaffold materials are supporting
provenance, even when not read directly by the inference loop.

## Baseline checks and limitations

- **Seven existing focused tests passed** in `tests/test_profiling_retirement.py`.
  Pytest cache and bytecode writes were disabled; temporary test outputs stayed
  inside this audit directory.
- In-memory schedule reconstruction succeeded for all six configured cohorts
  (ages 4–9), without writing schedules or rosters.
- Configured round prompt/evaluation memberships are disjoint. This does not
  resolve the separate four-year-old observed-versus-configured discrepancy.
- All configured exemplar keys are present and nonempty. The narrative input
  has 2,166 rows, no duplicate lookup keys, and six blank-group rows; none of
  those blank-group rows is selected by the checked TD/SLI prompt lookups.
  All 2,166 referenced transcript paths currently exist.
- `show-counterbalance` and the legacy `show-manifest --limit 1` command run.
- `validate-layout` exits 1 because `data/raw`, `data/interim`, and
  `data/evaluation` do not exist. The direct generation inputs were checked
  separately and exist. No placeholder directories were created.
- The model-loading import probe fails: installed `transformers==4.12.5`
  does not export `BitsAndBytesConfig`. Python is 3.8.5. No packages were
  installed or upgraded. A pandas/numexpr compatibility warning is also recorded.
- No LLM inference, GPU check, annotation tool, analysis, result comparison,
  or external framework evaluation was run. Full experiment reproducibility
  is not certified by this step.

The Git worktree already contained extensive unrelated changes. Its existing
generation-scope modified files and HEAD are recorded in `baseline.json`;
this audit added only files under this maintenance directory. All 6,116
inventoried pre-existing files were rehashed unchanged. The repository's
blanket `docs/` ignore rule was recorded but not changed.

## Boundary for the next step

This baseline is complete as an inventory, dependency trace, and record of
known disagreements. It does not declare the generation workflow ready for
reviewers. Later work must preserve the saved narrative texts and distinguish
the actually executed four-year-old pairs from the current roster before
normalizing configurations. No backup, removal, refactoring, or scientific
analysis change is authorized by this baseline itself.
