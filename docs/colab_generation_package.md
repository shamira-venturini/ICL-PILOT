# Generation and annotation notebook

Use `notebooks/colab_direct_generation_pipeline.ipynb` as the single processing
interface. Its blocks cover common setup, generation, CSV-to-CHAT conversion,
TalkTag error annotation and QC/export. The existing filename is retained.
Evaluation stays in `notebooks/synthtexteval_demo.ipynb` and
`clinical_analyses_reordered.Rmd`; their scientific code is unchanged.

## Routes

| `MODE` | Input | Enabled blocks |
| --- | --- | --- |
| `full` | Configured ENNI exemplars | Generation, conversion, annotation, export |
| `generation_only` | Configured ENNI exemplars | Generation and export |
| `from_csv` | `GENERATION_CSVS` mapping of labels to saved CSVs | Conversion, annotation, export |
| `convert_only` | `GENERATION_CSVS` | Conversion and export |
| `from_chat` | `CHAT_INPUT_DIRS` mapping of labels to CHAT directories | Annotation and export |

Run common setup, then enabled blocks. A fresh session can start from saved
CSVs or CHAT files without generation variables, the generation model or the
human exemplar corpus. Specify either cohort CSVs or their aggregate;
overlapping bundle/story rows are rejected. Each run uses a new output directory.

Generation requires CUDA. TalkTag uses a separate environment installed only
by its block, keeping generation dependencies unchanged. Colab and local Jupyter
checkouts are supported; annotation requires Python 3.10 or later. Local project
helpers retain Python 3.8 support. CPU checks do not establish a complete runtime
lock or GPU compatibility; that requires the planned Colab trial.

## Generation and volume

Generation reads `configs/generation/experiment.json`, the frozen master roster,
cross-story rules, project-authored scaffolds and frozen prompt templates. The
author confirmed the shared protocol for ages four through seven. Prompt content,
exemplar choices and sampling calls are preserved.

`REPLICATES=None` uses ten replicates. A positive integer overrides only the new
run's volume. Counts appear before model loading: 47 pairs and six stories yield
282 narratives per stage per replicate. Effective settings are hashed in
`run_manifest.json`. More replicates do not add independent human participants.

Set `DATA_ROOT` to the root containing
`data/processed/generation_inputs/story_unit_narratives_chat_v2.csv`; keep its
parsing QC sidecar. Configuration assets resolve against their config directory.
Setup checks exemplar ages, availability, frozen identities and within-round
prompt/evaluation separation; it does not reoptimize pairs.

Generation writes exact prompt/output records and Stage 1 parent links to
`prompt_records.jsonl`, cohort/aggregate CSVs and design snapshots. It preserves
schedule order and one sequential sampling stream. New CSVs also carry `td_age`.
The schedule's legacy `generation_seed` field is not used to reseed each job;
the actual global seed is the configured seed recorded with the run.
See `generation_prompt_records.md` for the journal schema and limits.

## Conversion and annotation

The notebook releases the generator before annotation model loading. Conversion
reads saved CSVs and creates stage directories with metadata/ancestry checks;
see `post_generation_processing.md`. It does not load a model.

`ANNOTATION_STAGES` defaults to both `stage1_td` and `stage2_sli`. Every selected
subdirectory must exist. A single CHAT collection is processed directly; its
supplied label does not infer a clinical group.

`configs/annotation/talktag.json` pins source commit
`a9df736acf4efaa0c6ec79240a2d36348de983ab` and CLI options for new annotation runs.
The wrapper verifies the installed commit, records package versions, copies CHAT
inputs into a working snapshot and runs the CLI in a separate process. All
selected cohorts/stages share one invocation, avoiding repeated model loads.
Wrapped CHI main tiers are joined only in that copy. Source/snapshot hashes, command,
settings, log and tool report remain with outputs. This is not a recovered
historical version or a tested complete software/model lock. Source/API inspection
used the [pinned upstream code](https://github.com/OliverHennhoefer/talk-tag/tree/a9df736acf4efaa0c6ec79240a2d36348de983ab).

QC checks file/utterance coverage, produced word order, punctuation, boundaries,
headers and non-target tiers. Added error tags and target reconstructions are
ignored in the surface comparison. Missing files, textual changes or lost tiers
stop the block and retain outputs for review. A nonzero debug limit is labelled
partial. The tool's `annotated_lines` count denotes changed lines; unchanged
utterances are not treated as unprocessed merely because no tag was added.
QC does not establish linguistic tag accuracy or comparable clinical annotation
across real/synthetic data. No historical repair utilities run automatically.
CLAN measurements and statistical evaluation remain separate.

## Persistence and export

`pipeline_manifest.json` records routes, source paths/hashes and block statuses.
Annotation directories contain input snapshots, annotated CHAT files,
`annotation_manifest.json`, `talktag.log` and the upstream report. The final
archive contains all local run artifacts with relative paths. Archive creation,
Colab download and runtime release have separate controls; download and runtime
release are off by default.

Drive copies verify hashes and preserve the local hierarchy. A destination
belonging to another run is rejected. Copies occur after generation setup, at
generation cohort boundaries and after later blocks. CSV progress is saved every
`SAVE_EVERY` complete jobs; journal writes are flushed/synced per event. Frequent
durable generation checkpoints, sampling-state recovery and per-job resume remain
pending. Interrupted generation still requires a fresh generation run; completed
CSVs can feed later blocks.

Starting from an old CSV/CHAT directory records its paths/hashes but cannot
reconstruct missing model/prompt provenance. Retain the source run's manifests
and journal separately when available.
