# Shared preparation and annotation

Use the generation notebook's clickable **shared preparation** section after
common setup. Select `MODE="from_chat"` to prepare and annotate real ENNI once
alongside all completed generator CHAT collections. Input labels are provenance
keys, not clinical labels inferred from narrative content.

```python
CHAT_INPUT_DIRS = {
    "real_enni": REPO_DIR / "data/inputs/enni_verified_2026-10-28/chat_4_to_7",
    "llama": Path("/content/drive/MyDrive/LLAMA_RUN/chat"),
    "qwen": Path("/content/drive/MyDrive/QWEN_RUN/chat"),
    "gemma": Path("/content/drive/MyDrive/GEMMA_RUN/chat"),
}
CHAT_INPUT_METADATA = {
    "real_enni": {"source_type": "real", "corpus": "ENNI"},
    "llama": {"source_type": "synthetic", "generator": "llama3_1_8b"},
    "qwen": {"source_type": "synthetic", "generator": "qwen3_8b"},
    "gemma": {"source_type": "synthetic", "generator": "gemma2_9b"},
}
```

Replace generator paths with the actual saved data directories, each containing
CHAT files and no duplicate snapshots. Every nested relative path is retained;
equal basenames in different labelled collections cannot overwrite each other.
Direct stage subdirectories obey `ANNOTATION_STAGES`; generator roots containing
cohort subdirectories retain all selected nested CHAT files. Select individual
cohort/stage roots when finer filtering is needed. Overlapping selections fail.
Add actual generation run IDs to metadata when known; unknown provenance remains
unspecified rather than being invented.

`MODE="prepare_only"` runs only common setup and shared preparation/validation.
It is useful for reviewing audits without loading annotation models. A subsequent
from_chat run can use the original collection selection in a fresh output tree;
do not overwrite a failed or completed processing directory. The full/from_csv
routes also use this policy before TalkTag. Real supplied CHAT never enters the
segmentation worker.

## Preparation policy

`configs/annotation/preparation.json` defines a common deletion-only pass:

- Remove all dependent tiers and continuations, including previous morphology,
  dependencies, comments and coded judgments.
- Remove inline error codes, replacement/correction hints, glosses, alternative
  transcriptions and annotator comments. Keep the child's produced form rather
  than substituting a proposed target word.
- Retain headers/metadata, speakers and main-tier boundaries; fragments, unknown
  speech, repetitions/retracing, pauses, uncertainty, emphasis, actions, overlaps,
  language precodes, structural clause markers and bch/trn utterance bookkeeping.
- Unwrap a scope only when all its adjacent annotation codes are removed. Retain
  scopes required by repetition/retracing or other surviving codes. Delete newly
  exposed forbidden whitespace next to a closing scope or initial tier tab.

No grammar repair, guessed speech, fragment expansion or resegmentation occurs.
For collections explicitly marked `source_type="real"`, original files are read
without creating duplicate original-file backups. Their file identities, source
checksums and available CHAT PIDs are recorded. Real originals need not remain
expanded locally after preparation; their verified original ZIP is the recovery
reference. An available working source is checked against its recorded checksum.
Synthetic and unspecified collections use temporary original bytes in
`annotation_preparation/source_snapshot/`. Source type is never
guessed from directory names; use the explicit `CHAT_INPUT_METADATA` example above.

Prepared copies go into `prepared/`, and every deletion has a Unicode-offset audit
under `audits/`. Speech fingerprints, retained-code anchors, headers and main-tier
counts are checked. The manifest records the retention policy and collection,
source, prepared and audit identities/hashes. Full Chatter validation must pass
before TalkTag starts. Existing completed manifests keep their recorded policy.

The prepared roots enter one TalkTag invocation. Fresh TalkTag outputs retain the
same collection labels and enter one Batchalign UD worker. Hash checks link the
final UD outputs to TalkTag, shared preparation and the original sources. The
working copies support each stage's preservation checks. With the default lean
retention setting, successful-run intermediate CHAT is removed by the final
retention block. Final CHAT, audits, metadata, logs and validation reports travel
with Drive/ZIP exports. Real data are processed once, independently of classifier
folds. Preparation never deletes the selected source inputs or original ZIPs.

TalkTag postprocessing version 3 always rebuilds derived CHAT from the exact
pre-annotation source. Annotator-side word substitutions, deletions, inserted
words, punctuation, layout and boundary changes are automatically undone. Source
grammar, repetitions, transcription codes, existing annotations, headers and
dependent tiers are retained. New tags are transferred only from unchanged
tokens with one source position across all optimal token alignments. Repeated
words with ambiguous positions do not receive arbitrarily relocated tags.
Raw TalkTag outputs and source snapshots remain unchanged.

The annotation manifest records restoration edits, unmapped tags and unassessed
source words by utterance. Files with unavailable tag anchors have partial
annotation coverage, without stopping processing or excluding stories/bundles.
Error-feature extraction must mark affected utterances' error annotations as
unavailable rather than interpreting omitted tags as absence of errors. Text,
lexical and fresh UD measures still use the complete restored source. Final
preservation checks remain exact on source speech, punctuation and boundaries,
and Chatter still checks structural validity.

For a completed TalkTag process requiring source restoration, keep the same notebook
kernel and run settings. Execute the **Source-preservation policy** cell and then
the annotation-run cell. It verifies retained raw hashes and calls
`postprocess_annotation_run` to write a fresh reviewed manifest and derived
output directory, followed by preservation and Chatter checks. It does not
repeat TalkTag inference or overwrite the failed manifest. Continue through the
remaining validation/UD/retention cells after this recheck passes. Changed
settings or prepared-input references prevent this recovery route.

## Final verification and storage

The notebook defaults to `LEAN_ANNOTATION=True`. Keep one original ENNI ZIP and
one original Gillam ZIP at the paths in `configs/annotation/retention.json`, or set
`ORIGINAL_CORPUS_ARCHIVES` to their existing locations. These ZIPs are referenced,
never duplicated into individual runs. Colab needs an explicit saved ZIP path
because raw corpora are not in Git. Preparation checks required real-corpus ZIPs
before loading annotation models. Real collections require explicit corpus
metadata and a PID identifying one original member.

Per-utterance fingerprints record speaker, order, boundaries and case-sensitive
content, plus header/dependent-tier fingerprints. Shared cleaning compares spoken
content while allowing documented annotation deletions. TalkTag permits its
inline additions; UD must preserve existing main-tier codes as well as words.
The finalizer checks the complete source/preparation/TalkTag/UD chain, file
coverage, Chatter references and byte hashes before deleting anything. Thus the
final word/boundary comparison reaches back to pre-cleaning inputs.

The clickable **final text verification and retention** block runs after the last
annotation stage and before export. It retains the final annotated CHAT and an
`annotation_retention.json` receipt with full-file hashes, per-utterance
fingerprints, audit hashes, and original ZIP/member recovery references. It removes
only verified run-owned source snapshots, prepared CHAT, raw TalkTag/Batchalign
CHAT and the intermediate TalkTag CHAT when UD is complete. It preserves original
generation CSVs/journals, external input folders, logs and all audit/validation
records. A failed or incomplete run keeps its working files. Interrupted pruning
can resume from its recorded exact file list.

Drive synchronization removes only the corresponding checksum-matching
intermediates. Changed or unexpected Drive files are preserved and stop cleanup.
Earlier stage reports remain evidence of checks performed before compaction;
they are not live checks on deleted files. Verify a retained or relocated run with:

```python
from icl_pilot.chat.retention import verify_retained_annotations
pipeline = json.loads((run_root / "pipeline_manifest.json").read_text())
reference = pipeline["stages"]["annotation_retention"]
verify_retained_annotations(reference, run_root=run_root)
```

To recover an individual original without expanding a corpus again:

```python
from icl_pilot.chat.corpus_archives import restore_original_member
receipt = verify_retained_annotations(reference, run_root=run_root)
member = receipt["original_members"][0]  # Select the intended relative_path/PID.
archive = receipt["original_archives"][member["corpus"]]
restore_original_member(archive, member, Path("/fresh/recovery/child.cha"))
```

Recovery verifies ZIP and member checksums and refuses an existing destination.
This restores the downloaded original, including its original annotations; it
does not guess repairs to model output. Rebuild the ENNI study view to reapply the
two documented sex-header corrections. Start a fresh processing run to rerun
annotation after compaction. `prepare_only` and standalone `morphosyntax_only`
retain their working files because they lack a complete shared annotation chain.

## Real-source validation

Use standard Chatter structural and tier-alignment checks on original and
prepared CHAT. Optional strict cross-utterance quotation/completion linker checks
are not enabled: those relationships are outside the study outcomes. This policy
applies equally to real and synthetic files. The existing quotation markers,
continuation markers, spoken words and main-tier boundaries are preserved.
No automatic notation repairs or error-code suppressions are applied. See the
[validation record](../reproducibility/annotation_preparation.json) and
[file-level validation inventory](../reproducibility/annotation_source_diagnostics.csv).
All 238 original ENNI files and all 238 prepared copies pass this policy.

`morphosyntax_only` remains available to add UD tiers to existing annotated CHAT.
It deliberately retains existing inline codes and skips fresh TalkTag cleaning;
use from_chat for the common fresh real-plus-synthetic annotation protocol.
