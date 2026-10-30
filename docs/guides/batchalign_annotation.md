# Batchalign 3 English UD annotation

The generation notebook includes a separately runnable UD block after TalkTag.
R analyses remain in their own notebook. This annotation block runs no traditional
MOR/POST/MEGRASP chain, IPSyn, DSS or feature-table calculation.

## Run routes

- `full`, `from_csv` and `from_chat`: shared cleaning/Chatter, TalkTag, its preservation/Chatter gates,
  then Batchalign UD. `RUN_MORPHOSYNTAX=False` stops after TalkTag.
- `morphosyntax_only`: set labelled `CHAT_INPUT_DIRS`, run repository/settings/
  dependencies/common setup, then `morphosyntax-setup` and `morphosyntax-run`.
  This loads neither a generator nor TalkTag and never resegments supplied CHAT.
  The wrapper validates fresh prepared copies, so obsolete morphology can be
  replaced without treating it as valid input to the new analysis.

For fresh annotation of real ENNI and all synthetic outputs, use from_chat and
the [shared preparation route](shared_annotation_preparation.md). Include real
ENNI once and all generator CHAT collections; the combined TalkTag output
automatically supplies labelled UD inputs and checked preparation provenance.
For the UD-only route use already annotated CHAT when fresh TalkTag codes should
be retained; it does not strip existing error codes or rerun TalkTag. Select data
directories, not an entire run containing several copies. Both real and synthetic
data use the same annotation configuration.

## Pinned runtime and models

`configs/annotation/batchalign.json` fixes Batchalign 3 package 0.10.7,
Stanza 1.15.0, Torch 2.11.0, CPU execution, one pipeline worker, preserved
tokenization, two Torch threads and a separate seed 42 in the isolated process.
The source modules, English model catalog and eight model files have checksum
pins. All actual package, binary and model hashes are written into each manifest.

Installation uses a separate `.venv-morphosyntax` locally or
`/content/icl_pilot_ud_env` on Colab. Models use `.cache/stanza_ud` locally or
`/content/icl_pilot_ud_models`. These paths can be changed to retained caches.
`INSTALL_DEPENDENCIES=False` requires an existing runtime and verified models.
Model downloads preserve conflicting cache files and stop on checksum mismatch.

The adapter calls the reviewed `StanzaBackend` and Batchalign's `morphotag`
recipe. It selects only tokenize, English GUM MWT, POS, lemma and dependency
processors, with `package=None` to prevent Stanza from adding unrelated default
processors. Stanza model downloads are disabled during inference. Retokenization
is false, utterance splitting is disabled, and result caching is bypassed.
The backend's catalog refresh is checked against the pinned catalog before any
input is tagged; a changed catalog stops processing.

## Preservation and validation

Each source file is copied byte-for-byte into `input_snapshot/`. Old `%mor`,
`%gra`, `%umor` and `%ugra` records and their continuations are removed from
`prepared/` only, with counts recorded, to avoid already-tagged skips.
Chatter validates those exact prepared files with the existing full checks.

On the shared preparation/TalkTag route, these are copies of derived, newly
annotated working inputs. The earlier shared preparation step references real
corpus originals by identity/hash/PID without backing up their original files.

One isolated process annotates all selected files. Each output identity is
checked, including missing, duplicate and extra files. Raw output is available in
`raw_batchalign/` until successful final retention. Its only permitted added header is the pinned worker's
provenance comment; that comment is recorded separately. Header, main-tier,
inline-code, punctuation or non-morphological dependent-tier changes fail QC.
Whitespace-only record reflow is permitted in native output.

Final `annotated/` files insert fresh UD `%mor`/`%gra` under the original main
tiers for all speakers, preserving the original record layout and other content.
Old morphology aliases are replaced, not silently left as stale alternatives.
CHI coverage is recorded separately. An utterance without morphology is accepted
only when it contains no alignable lexical speech; its index/speaker is recorded.
Every lexical utterance must have both new tiers. Chatter validates the final
files, including tier alignment and UD relations. No errors are suppressed.

The manifest retains source/prepared/output hashes, replaced-tier counts,
command, worker/source/runtime/model identities and both validation references.
On the shared route it also retains the original preparation reference, collection
metadata and upstream TalkTag manifest hash. Those outputs are checked again
before and after UD; a changed source, audit or TalkTag output blocks completion.
Failures and partial outputs remain available for review. A rerun uses a fresh
processing directory; outputs are not overwritten or silently reused.

## Outputs and limits

Use `RUN_OUTPUT_DIR/morphosyntax/annotated/` and
`morphosyntax_manifest.json` for downstream work. Final CHAT, fingerprints,
manifests and diagnostics travel with Drive/ZIP exports. The default notebook
retention block removes verified intermediate CHAT only after the shared chain
passes; failed/standalone UD runs keep their working copies. See
[final verification and recovery](shared_annotation_preparation.md#final-verification-and-storage).
Models and software environments stay in their separate caches.

These `%mor`/`%gra` are UD annotations. Downstream feature readers must use
this recorded lineage; older readers that require `%umor`/`%ugra` need an
explicit input adapter when fresh feature tables are constructed. Traditional
MOR scoring rules must not be applied as though these tiers were traditional
MOR. Annotation/segmentation and linguistic-tag accuracy still need manual
review; software checks do not establish clinical fidelity.

The implementation has a native CPU smoke check for error codes, wrapped speech,
contractions, fresh retagging and Chatter validation. Full corpus execution and
Colab integration are separate from that check.

Upstream references: [Batchalign 0.10.7](https://pypi.org/project/batchalign/0.10.7/),
[Stanza models](https://github.com/stanfordnlp/stanza-resources),
and [UD relation definitions](https://universaldependencies.org/u/dep/).
