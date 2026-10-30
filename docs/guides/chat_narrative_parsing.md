# ENNI source preparation

The original archive is data/raw/enni/2026-10-28/ENNI.zip. Its SHA-256 is 25e360e43ec5984ccd54e9b4679b5070a5544ff09d98829e9496622c30394dcf. Preserve it and every original member unchanged.

## Participant identity

Read literal CHI age/group headers. The original 0demo.xls table independently checks group and age year and supplies demographic sex. Retain the literal Sex_CHI value and document any discrepancy. Never infer a clinical label from a folder.

The archive contains 361 CHAT files. TD/B/523.cha lacks CHI age/group and shares its numeric ID with the fully identified SLI 523 record. Exclude the incomplete file. The original demographics table identifies 523 as male, age 5.498630137, SLI-A. This exclusion yields 360 usable participant records.

The study window is 48 <= completed months < 96: 238 children (190 TD, 48 SLI), 1,428 stories. Ages use 12 * years + months from years;months.days. Preserve days without rounding into the next age year. Days/30 is confined to fine within-cohort matching.

## Narrative extraction

Use story markers and complete CHI main tiers; retain their original utterance boundaries. Keep dependent tiers separate. Remove timing/annotator notation, pauses, unintelligibility markers and non-speech codes using the common extraction preparation. Preserve word forms, repetitions and parenthesized partial words; never infer speech or repair grammar.

The prepared CSV retains File_ID, Group, Age_Raw, Age_Month, source path, story ID, utterance count/array, text, source PID and SHA-256. Every usable participant supplies A1–A3 and B1–B3. Its QC sidecar records raw hashes, exclusion reasons and per-story preparation.

## Reproduction

In a fresh checkout before the versioned outputs exist:

    PYTHONPATH=src python -m icl_pilot.maintenance.enni_corpus /path/to/ENNI.zip --project-root /path/to/ICL-PILOT

The one-time demographic XLS cross-check requires Rscript with readxl. The builder rejects existing outputs, unsafe archive paths, ambiguous usable IDs, missing metadata and incomplete story sets. It builds the canonical participant/story tables, pair roster and study-only CHAT input directory.

Working copies in data/inputs/enni_verified_2026-10-28/chat_4_to_7/ contain only eligible sources. Their two study-age sex-header adjustments are documented; speech, boundaries and existing tiers are unchanged. [Reproducibility records](../reproducibility/README.md) provide source/input hashes and counts.
