# Definitive ENNI inputs

This directory contains one participant table and one prepared story table.

- enni_participants_2026-10-28.csv: 360 usable participant records with literal CHI age/group, original demographic sex, raw Sex_CHI, source PID/path and SHA-256.
- story_unit_narratives_enni_2026-10-28.csv: 2,160 individual stories and their original child-utterance arrays.
- story_unit_narratives_enni_2026-10-28.parsing_qc.json: complete source and preparation audit.

The study window is 48 <= completed age months < 96: 238 children, 190 TD and 48 SLI-labelled, yielding 1,428 real stories.

The supplied ZIP and original members are preserved under data/raw/enni/2026-10-28/. TD/B/523.cha is excluded because its CHI age/group are missing and its numeric ID conflicts with the fully identified SLI 523 record. Group is never inferred from a folder. The original demographics table independently identifies 523 as SLI-A.

Prepared text removes annotator comments, pauses, unintelligibility markers and non-speech codes. It preserves grammatical forms, repetitions and parenthesized partial words. There are no inferred words or grammar corrections. Generation validates and selects this literal text.

For annotation, use data/inputs/enni_verified_2026-10-28/chat_4_to_7/. Its 238 working copies retain words, boundaries and existing tiers; the two documented study-age sex-header adjustments follow the original demographics table. Raw files remain unchanged.

See [ENNI extraction](../../../docs/guides/chat_narrative_parsing.md) and [reproducibility](../../../docs/reproducibility/README.md).
