# Generation and annotation notebook

Use `notebooks/colab_direct_generation_pipeline.ipynb` as the single processing
interface. Its blocks cover common setup, generation, CSV-to-CHAT conversion,
TalkTag error annotation and QC/export. The existing filename is retained.
Evaluation stays in `notebooks/synthtexteval_demo.ipynb` and
`notebooks/clinical_analyses_reordered.Rmd`; their scientific code is unchanged.
The framework notebook writes each new evaluation to a fresh directory under
`results/framework_based_evaluation/`. The preserved August 2026 results are in
`run_20260827T153330Z/`; result directories cannot be reused by its setup helper.

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
by its block, keeping generation dependencies unchanged. Environment creation
uses `venv --without-pip`; the kernel's pip (>=22.3) installs into that environment
using `--python`, avoiding a dependency on `ensurepip` in hosted Python builds.
An interrupted creation is repaired without clearing the directory. Colab and local Jupyter
checkouts are supported; annotation requires Python 3.10 or later. Local project
helpers retain Python 3.8 support. CPU checks do not establish a complete runtime
lock or GPU compatibility; that requires the planned Colab trial.

## Generation and volume

For a small end-to-end check, use `MODE="full"`, `SMOKE_TEST=True`, and
`SMOKE_TEST_JOBS=12`, the current notebook defaults. This runs the first twelve
jobs in the unchanged schedule: two replicates of all six target stories for
SLI 444 / TD 410 in the four-year-old cohort, yielding **12 TD-like + 12
SLI-conditioned narratives**, followed by conversion, annotation and QC for both.
Set `SMOKE_TEST_JOBS=1` for the original one-story check, or another positive
integer no larger than the configured schedule. The Python setup API retains
one job as its default when no count is passed. Setup prints the selected count
before model loading. The run ID begins `enni_smoke_`; manifests and generation
CSVs mark it `test_only`. These outputs are not study data or independent children.
The full validated study design remains as provenance; a separate
`generation_design/execution_schedule.csv` records the actual selected jobs.

This batch retains seed 42 and one sequential random stream. Repeating the same
prompts, seed and execution environment may repeat the same outputs; later jobs
provide additional draws for the six story conditions. This is a local diagnostic
of repeated unchanged transformations for one prompt pair, not an estimate
across all children or age cohorts. Changing the test size does not alter the
configured prompts, source pairs, decoding settings or error-insertion rules.
The current prompt wording is described below; unchanged outputs are retained.

Smoke mode still requires the normal exemplar CSV under `DATA_ROOT` and loads
both models, so installation/download/model-loading time remains. It limits
inference and transcript output, not model size. To execute the full study,
set `SMOKE_TEST=False` and start a fresh run. `REPLICATES=1` alone still schedules
282 jobs, so it is not a small-job limit. The switch applies only to new generation;
saved-input routes process their explicitly selected inputs.

Generation reads `configs/generation/experiment.json`, the frozen master roster,
cross-story rules, project-authored scaffolds and frozen prompt templates. The
author confirmed the shared protocol for ages four through seven. The active
prompts implement the reviewed generation/rewriting pair, without age/status
labels or directional length instructions; exemplar choices and canonical story
content are preserved. On 21 October 2026, the author adopted an output
ceiling of **640 new tokens per stage**, shared across all cohorts, in
`experiment.json`. Setup prints both effective stage limits. The shared runner
passes them explicitly to the notebook's model call and prompt journal.
This exceeds the longest stored real story at ages four through seven (568
tokens), permits earlier end-of-sequence stopping and sets no minimum length.
The journal also records actual generated-token counts for reviewing ceiling
hits. Start a fresh run after updating; an existing run cannot be reused with
changed limits. The author subsequently selected **temperature 0.8 and top-p
0.9** for both stages and all cohorts. Setup prints these shared configuration
values; each model call and its prompt record receives them. The later removal
of age/status and length cues changes prompt wording only. Top-k remains inherited pending
an explicit protocol decision.

`REPLICATES=None` uses ten replicates. A positive integer overrides only the new
run's volume. Counts appear before model loading: 47 pairs and six stories yield
282 narratives per stage per replicate. Effective settings are hashed in
`run_manifest.json`. More replicates do not add independent human participants.

Both Colab (including its VS Code extension) and local Jupyter default to the
checkout as `DATA_ROOT`. The frozen exemplar CSV and its parsing-QC sidecar now
travel with the configuration in `configs/generation/inputs/`. An accompanying
checkout containing these files needs no separate input upload. The
`generation-input` cell records the CSV path/hash in `pipeline_manifest.json`.
The sidecar records extraction provenance and is not needed for inference.

`ICL_PILOT_DATA_ROOT` can explicitly select a different input tree. That tree must
contain `configs/generation/inputs/story_unit_narratives_chat_v2.csv`. Existing
Drive trees using the former `data/processed/generation_inputs/` path are not
modified automatically: use the checkout default or deliberately place the CSV
at the new configured path in the selected input tree.

Optional upload staging remains available. Upload the CSV using VS Code's
**Upload to Colab** or the Colab file panel and set `GENERATION_INPUT_UPLOAD` to
the uploaded runtime path, such as `/content/story_unit_narratives_chat_v2.csv`.
The input cell copies it only if the configured destination is missing, verifies
its SHA-256, and preserves conflicting existing inputs. The default is `None`
because the checkout already supplies the input. Saved-CSV/CHAT routes skip
exemplar staging and do not require this CSV.

Drive output persistence is unchanged. Drive is also mounted when an explicit
input override requires it, even if `SAVE_TO_DRIVE=False`. Paths are interpreted
by the selected kernel: Colab cannot read the Mac's `/Users/...` paths directly.
The roster, scaffolds and prompt-template paths resolve against the configuration
directory; the exemplar CSV path remains relative to `DATA_ROOT`.
Setup checks exemplar ages, availability, frozen identities and within-round
prompt/evaluation separation; it does not reoptimize pairs.

The reviewed pair adopted on 21 October uses the same system message, all four
story-information fields (events, required content, optional content, additional
constraints) and full instruction list in both stages. The example guides
wording, sentence construction, style and level of detail identically. Stage 2
additionally receives the actual Stage 1 narrative to rewrite with its own
selected exemplar. Additional constraints contain only event-order/no-new-event
rules. No template includes age/status labels, ENNI/task labels, story/exemplar
IDs, titles or replicate IDs; selection and logs preserve that metadata.
Canonical story content, literal exemplars/parents and inference settings are
preserved. No stage is instructed to shorten its output; the configured ceiling
remains 640. This pair is distinct from the full neutral variant at Git commit
`9a59fa21`. See `generation_prompt_records.md` for scope and history.
The complete Stage 1 preview prints both template hashes; actual Stage 2 prompts
are recorded with their generated Stage 1 parents in the journal. To use the
revised prompts in an existing Colab session, pull the checkout, restart the kernel
and run from the top with `MODE="full"`, `SMOKE_TEST=True` and
`SMOKE_TEST_JOBS=12`. The settings create a fresh run ID; old outputs remain intact.

Generation writes exact prompt/output records and Stage 1 parent links to
`prompt_records.jsonl`, cohort/aggregate CSVs and design snapshots. It preserves
schedule order and one sequential sampling stream. New CSVs also carry `td_age`.
The schedule's legacy `generation_seed` field is not used to reseed each job;
the actual global seed is the configured seed recorded with the run.
See `generation_prompt_records.md` for the journal schema and limits.

The read-only `generation-copy-check` cell compares both stage texts in each
selected generation CSV, ignoring whitespace differences. It reports unchanged
counts and the first ten affected pair/story IDs. It can also inspect saved-CSV
routes without inference; CHAT-only routes skip it. Copy flags do not alter the
source CSV, discard cases, introduce errors or trigger selective resampling.
An unchanged Stage 2 output demonstrates no textual transformation for that
job. A changed output, or an annotation with no error tags, does not establish
DLD-like fidelity; those require the separate linguistic evaluation.

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

If an already-running Colab session has the earlier installer and fails while
creating the annotation environment, recover it without rerunning generation:

```python
subprocess.run([sys.executable, "-m", "venv", "--system-site-packages",
                "--without-pip", str(ANNOTATION_ENV)], check=True)
subprocess.run([sys.executable, "-m", "pip", "--python", str(ANNOTATION_ENV),
                "install", "--ignore-installed", "pip"], check=True)
```

Then rerun only annotation setup, annotation, and export. This leaves generated
CSVs and CHAT inputs in place. The updated helper does not need this recovery
cell; it uses the kernel's pip to target the annotation environment directly.

Wrapped CHI main tiers are joined only in that copy. Source/snapshot hashes, command,
settings, log and tool report remain with outputs. This is not a recovered
historical version or a tested complete software/model lock. Source/API inspection
used the [pinned upstream code](https://github.com/OliverHennhoefer/talk-tag/tree/a9df736acf4efaa0c6ec79240a2d36348de983ab).

TalkTag's exact CHAT output is retained in `annotation/raw_talktag/`. A deterministic
postprocessor produces `annotation/annotated/`: it aligns surface tokens within
each CHI utterance, restores the input snapshot's capitalization and spacing,
and retains TalkTag's inline tags at the same token boundaries. Alignment permits
capitalization differences only; changed words, punctuation, order or utterance
boundaries stop processing for that file. Existing source annotations must also
be retained at their original anchors. No grammatical correction is performed.

QC then compares source and postprocessed surface tokens **case-sensitively**,
checks file/utterance coverage, and verifies headers and non-target/dependent
tiers. Added error tags and target reconstructions are ignored in the surface
comparison. Missing files, substantive textual changes or lost tiers stop the
block and retain outputs for review. Raw/derived file hashes and per-file
formatting-restoration flags are recorded in the manifest. A nonzero debug limit
is labelled partial. The tool's `annotated_lines` count denotes changed lines; unchanged
utterances are not treated as unprocessed merely because no tag was added.
QC does not establish linguistic tag accuracy or comparable clinical annotation
across real/synthetic data. No historical repair utilities run automatically.
CLAN measurements and statistical evaluation remain separate.

If annotation raises `Annotation QC needs review`, inspect the run's
`annotation/annotation_manifest.json`: `qc_issues` identifies failed checks,
while `expected_target_utterances` and `reported_target_utterances` describe
coverage. Compare any affected file under `annotation/input_snapshot/` with its
counterpart under `annotation/raw_talktag/` and, when produced,
`annotation/annotated/`. Earlier runs stored the raw output directly in
`annotation/annotated/`. Originals and failed outputs are retained. Exceptions
also include counts by QC issue; no check is bypassed.

An existing batch can be postprocessed without rerunning generation or TalkTag.
After updating the checkout, reload the helper in a running kernel and point it
at the **exact failed run**, since rerunning settings changes `RUN_OUTPUT_DIR`:

```python
import importlib
import json
from pathlib import Path
import icl_pilot.chat.annotation as annotation
importlib.reload(annotation)

failed_run = Path("/content/ICL-PILOT/data/generated/direct_generation/"
                  "enni_smoke_20261015T094156357444Z")
checked = annotation.postprocess_annotation_run(
    failed_run / "annotation/annotation_manifest.json")
result = json.loads(checked.read_text())
print(result["status"], result["qc_issues"])
```

This verifies recorded source/snapshot/raw hashes and repeats all coverage and
preservation checks. It creates `annotation/postprocessed/` and
`annotation/postprocessed_manifest.json`; the old manifest and raw files are
unchanged. A repeated call refuses to replace these artifacts; inspect the saved
report or supply a fresh `output_name`. A successfully reviewed batch can be
recorded in its pipeline manifest and copied to the existing Drive export root:

```python
from icl_pilot.pipeline import record_stage, sync_run
status = "complete" if result["status"] == "annotation_complete" else "partial_debug"
record_stage(failed_run, "annotation", status, manifests=[str(checked)])
if IS_COLAB:
    sync_run(failed_run, DRIVE_EXPORT_ROOT / failed_run.name)
```

Earlier exports also contain a four-field `@Types` header. New conversions use
the correct three fields and a separate generated-provenance comment. For a
saved annotated batch, follow the [header-only repair procedure](post_generation_processing.md#correcting-saved-generated-chat-headers)
to produce a fresh `annotation/formatted/` copy without model inference. Keep
its formatting manifest alongside the annotation record; full CHAT syntax
validation remains a separate step.

## Persistence and export

`pipeline_manifest.json` records routes, source paths/hashes and block statuses.
Annotation directories contain input snapshots, raw TalkTag and postprocessed
CHAT files, `annotation_manifest.json`, `talktag.log` and the upstream report.
Use the manifest's `annotated_output_directory` for the reviewed files; it is
`annotated` for new runs and `postprocessed` after the recovery above. The final
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
