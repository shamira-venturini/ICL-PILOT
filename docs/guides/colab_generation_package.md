# Generation, annotation and evaluation notebook

Use notebooks/colab_direct_generation_pipeline.ipynb. Select one MODEL_KEY from configs/generation/models.json per fresh run. Run setup from the top after changing code or configuration.

Framework evaluation has its own settings and clickable section 6. Run the
repository cell, then jump directly there to evaluate saved text without
generation or annotation. The section is disabled by default; corpus metrics,
quality-model inference, main BERT fitting and the separate ratio experiment
have independent switches. Colab results persist incrementally to Drive.
See the [framework block guide](framework_notebook.md) for inputs, the isolated
runtime, original-smoke-upload support, resume and compact verified exports.

The linked **domain features** section runs KIDEVAL English UD on completed
annotations, either from the current processing run or independently from saved
manifests/ZIPs. Set `RUN_DOMAIN_FEATURES=True` to enable it; the default is off.
It exports comparable six-story units and does not run IPSyn or R analyses.
See [domain feature settings and exports](domain_features.md).

## Real ENNI annotation pilot

`RUN_PROFILE="real_enni_pilot"` selects a bounded `from_chat` run with six
four-year-old children: TD 410/450/419 and SLI 444/480/478. The first three pairs
in the definitive roster provide 36 narratives and 550 CHI utterances. This is
a processing pilot, not a representative accuracy estimate. Original real-data
boundaries remain unchanged. Generation and synthetic segmentation are skipped.

In a fresh Colab GPU runtime, upload both `ICL-PILOT_smoke_bundle.zip` and
`ICL-PILOT_ENNI_annotation_pilot_inputs.zip`. The active smoke notebook already
selects this profile; Run All performs shared cleaning, TalkTag, UD, preservation,
Chatter, retention and export. It verifies the input-package checksum, selection
manifest and every file, preserves conflicting existing files, and marks the run
test-only. The original ENNI ZIP is included once as its recovery reference.
Gillam is not required for this pilot because its external classifier evaluation
uses narrative text rather than UD features. Corpus ZIPs on the Mac are not
automatically available in a new Colab runtime.

The archive is copied to `/content/<archive-name>.zip` for downloading through
VS Code's Colab Contents file menu. Keep the earlier synthetic smoke archive for
the later annotation-quality comparison. The main notebook keeps the normal
`RUN_PROFILE="generation"` default, which also permits its existing saved-input
modes. The smoke notebook can return to the generation test using that profile.

See [pilot selection and checks](../reproducibility/real_annotation_pilot.json).

## Modes

- generation_only produces raw staged narratives and generation provenance.
- review_only starts from GENERATION_CSVS and writes review-only story QC reports,
  without generation, CHAT conversion or annotation.
- convert_only starts from saved generation CSVs and stops after segmentation/CHAT validation.
- full generates, converts and annotates.
- from_csv starts from explicitly selected generation CSVs.
- from_chat starts from explicitly selected CHAT directories without generation or segmentation.
- prepare_only performs the shared cleaning and Chatter check on saved real and
  synthetic CHAT, without loading TalkTag or UD models.
- morphosyntax_only adds fresh Batchalign 3 English UD tiers to saved CHAT,
  without generation, segmentation or TalkTag. Run common setup, then jump to
  the UD setup/run cells. The notebook has a clickable outline.

The notebook's configuration cell defines the supported mode names, paths, cohorts, stage selection, smoke controls and output/synchronization roots. Inspect it before execution. Colab paths refer to the Colab runtime, not arbitrary local Mac paths.

## Preflight and generation

The default smoke test limits jobs and labels outputs as software fixtures. Confirm all cohorts, model/template options and finish records before a scientific run. Smoke output cannot enter evaluation.

The full design is 48 pairs * 6 targets * 10 replicates = 2,880 jobs per model, each with Stage 1 and Stage 2. Both stages use the pinned sampling settings, complete prepared exemplars and native stopping tokens. Stage 2 receives the same model's exact Stage 1 parent.

Setup verifies input/QC hashes, complete exemplars, roster ages, assignment balance and within-round prompt/evaluation separation. It writes an effective configuration, source hashes, selected exemplar snapshot, schedule and split bookkeeping into a fresh run directory.

Keep the shared run-level seed policy. Use [verified recovery](generation_recovery.md) for interruptions; do not reseed or regenerate completed calls.

## Processing and exports

Use the notebook's clickable generation-review section to inspect saved outputs
before further processing. The [review guide](generation_review.md) describes the
paired-story HTML page, flags, journal checks and downloadable notes. Every story
remains eligible; no story or bundle is discarded or resampled by review.

Release generator resources before the pinned whole-narrative segmentation worker runs. Inspect boundary/runtime records and required Chatter results. The [shared preparation block](shared_annotation_preparation.md) removes old linguistic annotations using one policy for real and synthetic CHAT. TalkTag operates on these checked working snapshots and retains preservation/QC reports.

Generated CHAT conversion records curly-to-ASCII apostrophe replacements before
segmentation; CHAT does not accept curly apostrophes inside words. Raw generated
text remains unchanged. The notebook verifies that this rule is available, then
checks actual CHAT files, replacement logs and source CSV hashes before Chatter.
The exact older converter from the original smoke upload can receive the
reviewed four-line correction; both source hashes and the update are recorded
in `chat_formatting_policy.json`. Unknown older implementations are not patched.
`chat_formatting_manifest.json` records checked files and replacement counts.
Hyphens are unchanged. If a later processing step fails, retain that run and
start a fresh `from_csv` or `from_chat` processing run from the saved inputs.
Do not rerun generation or the settings cell to recover a later step: the
settings cell creates a new timestamped run path. Keep smoke-derived processing
explicitly marked `test_only` and separate from study outputs.

For the joint study run, finish each generator's CHAT conversion, choose
from_chat, include real ENNI once plus the three generator CHAT roots, and enter
their provenance in CHAT_INPUT_METADATA. Start at shared preparation after
common setup. One TalkTag invocation and one UD worker process all selected
collections while preserving unique labels and nested paths. Real ENNI is never
resegmented. Standard structural and alignment validation applies equally to all
collections. Optional strict cross-utterance quotation/completion checks are not
enabled, and original markers are preserved.

Shared preparation references explicitly labelled real originals by identity,
checksum and available CHAT PID instead of creating duplicate original backups.
Working copies are used during processing. The final retention block checks the
complete chain, keeps final annotated CHAT and fingerprints, and removes verified
successful-run intermediates. Audits, logs and validation reports remain. Keep one
original ZIP per real corpus; configure ORIGINAL_CORPUS_ARCHIVES for saved Colab
paths. Required archives are checked before annotation starts and are referenced
instead of copied into each run. Failed/incomplete runs retain working files.

The full/from_csv/from_chat routes then run [Batchalign 3 UD annotation](batchalign_annotation.md)
by default. Set RUN_MORPHOSYNTAX=False to skip it. The isolated CPU worker
preserves words, inline codes, headers and utterance boundaries, replaces stale
morphology on derived copies, and requires complete coverage plus final Chatter
validation. Canonical outputs are morphosyntax/annotated/ with UD %mor/%gra.
R analysis stays separate; traditional MOR/MEGRASP and IPSyn are not run.

The complete run folder retains generation outputs, actual prompt/parent records,
conversion/segmentation manifests, final annotations and validator results. The
retention receipt records final full-file and utterance hashes plus original
ZIP/member recovery references. Drive synchronization mirrors only explicitly
verified intermediate removals; changed copies stop cleanup. See the
[verification/recovery instructions](shared_annotation_preparation.md#final-verification-and-storage).
Confirm the downloaded archive before releasing the Colab runtime.

Full-checkpoint execution, manual boundary accuracy review and complete scientific annotation are separate checks. A successful local fixture or structural validation does not establish them.
