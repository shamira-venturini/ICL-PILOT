# Qwen smoke-run inspection, 2 November 2026

The complete generation-to-export software smoke test passed. This establishes
execution and preservation for this bounded run; linguistic annotation accuracy,
clinical fidelity and the other two generators are not established by this test.

## Evidence and integrity

- Run: `enni_smoke_qwen3_8b_20261102T094720088747Z`.
- Downloaded archive: `enni_smoke_qwen3_8b_20261102T094720088747Z_20261102T100143090130Z.zip`.
- Archive SHA256: `a54acbef2c7f5ed39905446eaedd014bd11a677a37aed54d71b5254b474b6dd0`.
- Archive size: 496,728 bytes; 167 members; ZIP CRC checks passed.
- Fresh local verification of the relocated retention receipt passed, including
  all 24 final CHAT hashes, exact text fingerprints and 57 retained audit hashes.
- The 192 declared intermediate CHAT copies are absent as intended. Generation
  CSVs, prompt/output journal and final annotations remain available.

## Execution and coverage

Qwen3-8B used the configured checkpoint, thinking disabled, seed 42,
temperature 0.8, top-p 0.9, top-k 0 and 640 new tokens. The effective generation
configuration in the run manifest agrees with these settings.

The test contains one four-year-old prompt pair, six target stories and two
reversed exemplar assignments: 12 paired-stage jobs, 24 narratives and two
complete six-story bundles per stage. All 12 Stage 2 parent hashes agree with
the corresponding exact Stage 1 output. The journal contains 24 prompt records
and 24 output records. The cohort and aggregate CSVs contain the same 12 jobs;
they are alternative exports and must not be concatenated as separate samples.

TalkTag completed 298/298 CHI utterances without QC issues. Source restoration
recorded zero word-edit utterances, zero unmapped tags and zero unassessed words.
Complete coverage means complete processing, not verified error-tag accuracy.
Batchalign UD produced morphology and dependencies for all 298 utterances,
without nonlexical exclusions. Chatter checks, retention, smoke verification
and ZIP export checks all completed. The immutable Stanza resource fix worked
in the actual Colab run.

## Generated-text findings

All 24 generated narratives were read. The review report flags three stories:

- Stage 2 B2, replicate 1, and Stage 2 B1, replicate 2, are unchanged from their
  Stage 1 parents after whitespace normalization.
- Stage 1 A2, replicate 2, reached 640 tokens. Its 555 whitespace-counted words
  contain a long recurring sequence about the sign, bench and animals, ending
  mid-phrase. Its Stage 2 descendant also retains some repeated content.

Other outputs show varying amounts of rewriting. Some Stage 2 narratives become
lists of short fragments; others remain close to their Stage 1 parents. These
are descriptive observations from one prompt pair, not clinical conclusions.
All stories and bundles remain retained under the agreed review-only policy.

## Annotation findings requiring interpretation

Detailed inspection of final CHAT examples and all four new inline error tags
identified issues worth addressing before interpreting utterance-based or
error-based linguistic measures:

- Segmentation can join several generated fragments into a single main tier.
  For example, Stage 2 A1, replicate 1, starts with
  `pool fox and bear red ball ball rolls into water .`.
- The same file splits the phrase "no running" across two main tiers:
  `edge has sign that says “no” .` and `“running” ball floats .`.
  This changes the linguistic unit used by subsequent annotation and measures;
  the concern is the boundary, not the quotation glyph.
- Stage 2 A2, replicate 1, includes `fox runs toward [* s:r:prep] water .`.
  This looks like a questionable error judgment and illustrates why successful
  preservation and Chatter checks do not validate the linguistic labels.
- UD parses of fragmented text can also be questionable: in Stage 2 A1,
  replicate 1, `bear` is analyzed as a verb in `fox and bear happy again .`.

There is one inline error tag in Stage 1 and three in Stage 2. These counts must
not be treated as validated clinical error rates. No text, boundaries, prompts,
labels or eligibility rules were changed during this inspection.

The segmentation log includes a 562 > 512 tokenizer warning for the long
narrative. Inspection of the exact checksum-verified Batchalign inference source
confirms that its batch predictor uses overlapping token windows. Full output
word coverage and final preservation checks passed; this warning did not imply
that the narrative was truncated during segmentation.

## Limits and next decisions

The smoke test exercises generation, segmentation, CHAT conversion, shared
preparation, TalkTag, UD, preservation, lean retention and export. It does not
exercise fresh feature extraction, domain statistics or framework classifiers.
The software path now completes, while the observed boundary and annotation
examples need consideration before relying on linguistic outcomes. This test
provides no estimate of their frequency across models, ages or prompt pairs.
