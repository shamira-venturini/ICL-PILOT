# KIDEVAL notebook implementation and execution check

The main pipeline and existing smoke notebook now contain independently runnable
domain-feature blocks. Their settings default to off. The original smoke upload
remains unchanged; a checksum-pinned support ZIP supplies feature code,
canonical metadata and the pinned official CLAN source archive.

The native tools were rebuilt from the unmodified V 21-Sep-2026 source on macOS
arm64. The final saved-input check then started from a separate unpacking of the
original smoke project, applied the support ZIP through the notebook cell, and
ran runtime verification, extraction, preview and archive export. It used the
six recovered real ENNI pilot files and the saved 24-story Qwen pilot ZIP.

The check completed with:

- 30 final annotated inputs: six real children and four synthetic six-story
  bundles, representing 60 canonical stories and ten output rows.
- 814 child utterances within canonical stories. The manifest separately
  records 34 real child utterances outside those sections.
- Ten valid derived CHAT bundles; all final annotated input hashes preserved.
- Six media headers and 475 audio bullets removed only from derived bundles.
- Native KIDEVAL English UD reports and DSS category tables; no IPSyn columns
  or normative-database comparisons.
- Five units with native DSS sample eligibility. Two had partial TalkTag
  coverage, leaving three analysis-available DSS sentence scores. Three units
  overall retained partial-coverage flags. Missing scores were not filled.
- Combined and separate real/synthetic CSVs, raw reports, runtime receipt,
  validation, provenance and a verified feature ZIP. All ten rows are test-only.
- 40 passing focused checks covering features, framework notebook regression
  and the syntax reader. This includes duplicate/age/ancestry rejection,
  checksums, incomplete units, partial/unknown error coverage, native options,
  bad report identities, raw-error exclusions and disabled notebook behavior.

The companion JSON records exact file hashes and local artifact paths. The
[feature guide](../guides/domain_features.md) describes execution and outputs.
This was a software execution check on saved annotations, not a linguistic
accuracy study or full-study extraction. Linux/Colab compilation was not executed
locally. Custom error measures and the separate R statistical revisions remain
pending; no manuscript text or scientific result tables were changed.
