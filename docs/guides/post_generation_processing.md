# CHAT, annotation and linguistic measurement

## Inputs

Real-data annotation begins from data/inputs/enni_verified_2026-10-28/chat_4_to_7/. Synthetic annotation begins from newly generated CSVs with complete model and Stage 1/Stage 2 provenance. The supplied original corpus remains untouched.

Use notebooks/colab_direct_generation_pipeline.ipynb for conversion, TalkTag and
Batchalign 3 English UD annotation. The definitive pair roster is
configs/generation/master_frozen_roster_enni_2026-10-28.csv. Saved CHAT can enter
the independent morphosyntax_only route; see [UD annotation](batchalign_annotation.md).

## Required sequence

1. Segment complete synthetic narratives with the pinned worker and write CHAT.
2. Validate all input CHAT with checksum-pinned Chatter.
3. Run TalkTag on exact working snapshots using configs/annotation/talktag.json.
4. Compare output against the pre-annotation reference and validate final CHAT again.
5. Add fresh Batchalign 3 UD morphology/dependencies on checked copies; require
   complete coverage, preservation QC and final Chatter validation.
6. Run final retention: verify text and metadata fingerprints against the source
   and stage chain, keep final annotated CHAT/audits and one original ZIP per
   real corpus, then remove only verified successful-run intermediate CHAT.
   Failed/incomplete runs keep working copies. See
   [verification and recovery](shared_annotation_preparation.md#final-verification-and-storage).
7. Run compatible measurement tools and build fresh interpretable feature tables
   with source hashes and tier/tool lineage. Traditional MOR/MEGRASP and IPSyn
   are not part of the annotation notebook.

Real supplied boundaries remain unchanged. Never segment after annotation or automatically replace words to repair alignment.

The annotation wrapper verifies input coverage and source/output hashes. It preserves words, punctuation, utterance boundaries, headers and existing dependent tiers except explicitly documented preparation. Failed or changed outputs remain available for review; they are not silently accepted or discarded.

Chatter certifies structure. Preservation checks certify the declared unchanged content. Neither establishes clinical error-tag, parsing or boundary accuracy.

## Features and analysis

Specify every feature's source tool/tier, units, numerator/denominator, missingness and bundle aggregation. The core families include morpheme-based MLU, lexical diversity, developmental syntax, dependency structure, disruption and morphosyntactic error rates.

Past-overgeneralization opportunities are overt past finite lexical verbs from
UD morphology plus explicit bare-past tokens m:0ed and m:base:ed. Fresh annotation
uses %mor; older %umor inputs need explicit lineage mapping in feature-table
construction. Zero opportunities remain missing. Do not normalize all
developmental scores into rates.

Fresh measurement inputs and the domain statistical revisions remain pending. The R metadata guard checks identities, canonical age and sex; it cannot certify that a feature table was measured from the correct annotation lineage. Use exact manifests and document versions rather than inferring lineage from filenames.
