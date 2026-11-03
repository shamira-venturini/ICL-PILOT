# Study reproducibility

These records describe the definitive study and its source integrity.

| Record | Contents |
|---|---|
| intake.json | Original ZIP hash, complete/excluded identities, demographic cross-check and eligible counts. |
| sample_counts.csv | Study-age TD/SLI counts. |
| roster.json | Definitive Hungarian assignment summary and generation volume. |
| study_chat_inventory.json | Original and working-copy hashes, including documented sex-header preparation. |
| tokenizer_files.json | Exact tokenizer download identities and verified checksums. |
| tokenizer_audit/ | Prepared-text token counts and independent Gemma tokenizer agreement. |
| classifier_design.json | Current all-child grouped fold assignments/counts, protocol identity and planned job counts; no model fitting. |
| bert_tokenizer_audit/ | Pinned BERT lengths, special-token/truncation checks, per-story counts and age/group summaries for ENNI and selected Gillam. |
| validation.json | Current software checks and scientific execution status. |
| external_archive.json | Recovery snapshot location and checksum. |
| gillam.json | Original Gillam archive integrity, header checks, cohort selection and available story counts. |
| gillam_source_inventory.csv | Source session IDs/PIDs, normalized child IDs, literal ages/sex, groups and transcript hashes for the full Gillam download. |
| annotation_retention.json | Lean-retention software checks and actual ZIP-backed recovery/reproduction verification; no full corpus annotation. |
| original_archive_retention.json | Verified removal inventory for redundant expanded originals; canonical ZIPs remain unchanged. |
| qwen_smoke_apostrophe_fix.json | Genuine 24-story Colab smoke failure, audited CHAT apostrophe fix and fresh native validation of all 24 files using recorded boundary actions; full annotation remains pending. |
| talktag_quote_restoration.json | Actual saved TalkTag diff, exact restoration of all 16 affected utterances, native CHAT checks and regression tests; Colab reprocessing/UD completion remains pending. |
| automatic_text_restoration.json | Current automatic source-restoration policy, substitution/deletion/insertion/repetition cases, original-text fingerprints and native Chatter checks; unavailable tag anchors are audited. |
| ud_resource_pins.json | Colab UD setup checksum failure, exact historical catalog/model revisions, unchanged checksums, startup-refresh and original-upload compatibility checks. |
| qwen_smoke_review_20261102.md | Actual completed Colab generation-to-export smoke run; independent local integrity/fingerprint checks, generation flags and concrete linguistic annotation limitations. |
| real_annotation_pilot.json | Frozen six-child ENNI annotation-pilot selection, upload checksums, native preparation/recovery checks and execution status. |
| enni_ud_provenance_recovery_20261102.json | Actual six-child Colab UD version-comment replacement, unchanged source main tiers, local recovery without inference and fresh Chatter checks. |
| annotation_comparison_20261102.md | Preliminary AI-assisted review of 144 real/synthetic utterances, case-level findings, limits on reliability claims and a separate syntax-reader compatibility issue. |
| syntax_reader_clitics_20261102.json | Corrected clitic-to-dependency alignment, 12 regression checks and before/after verification on 30 saved pilot files; all 100 falsely skipped utterances recovered without changing annotations. |
| framework_notebook_20261102.json | Independent evaluation blocks in both notebooks, isolated runtime, explicit input assembly, persistent/resumable runs, original-upload support and 111 passing checks; full model-based scientific execution remains pending. |
| framework_actual_data_20261102.md / .json | Actual Qwen-pilot plus real ENNI/Gillam execution, dependency/schema fixes, five one-update BERT CPU fits, GPT-2/MiniLM, ratio reuse, summaries and verified export; 121 regression checks passed. |
| kideval_notebook_20261102.md / .json | Independent KIDEVAL English UD blocks in both notebooks; native extraction/export from six real children and four synthetic bundles, preserved sources and explicit missingness; 40 focused checks passed. |

Original ZIPs remain under data/raw/enni/2026-10-28/ and data/raw/gillam/2026-10-28/.
Expanded originals are redundant once member checksums and ZIP-backed readers
are verified; their removal inventory is recorded separately. Selected ENNI study
CHAT inputs remain available. Canonical participant and prepared story tables,
including the full parsing/source QC, are in configs/generation/inputs/.
There is one active pair roster.

Reproduce intake into fresh destinations:

    PYTHONPATH=src python -m icl_pilot.maintenance.enni_corpus /path/to/ENNI.zip --project-root /path/to/ICL-PILOT

The one-time XLS cross-check requires Rscript/readxl. The builder refuses existing destinations, incomplete or conflicting usable identities, unsafe archive members and missing story sets. It writes only derived inputs and working copies; original members remain unchanged.

The study includes 238 children (190 TD, 48 SLI), 1,428 real stories, 48 prompt pairs and 18 generation batches. Generated volume is 17,280 stories across three models. Tokenizer maxima among study stories are 598 for Gemma and 568 for Llama/Qwen, below the 640 ceiling.

The separate BERT audit uses the pinned classifier revision and counts [CLS]/[SEP]
within the 512-token limit. After the exact framework text preparation, one of
1,428 ENNI study stories exceeds the limit (0.070%); it is also the only over-limit
story in the 576-story prompt-pair subset (0.174%). The current classifier pool
contains all 1,428 study stories. The story is child 725,
SLI, age 7, B3: 604 tokens including specials, with 92 content tokens removed by
right truncation (15.28% of that story's content tokens). None of the 543 selected
Gillam narratives exceeds the limit; its maximum is 460. These are real-data
length checks, not classifier results. Generated narratives remain pending.

Generation-prepared and classifier-prepared text are audited separately because
their CHAT-notation handling differs. The primary classifier counts above use
classifier-prepared text. Token IDs agree for all 4,863 audited text representations
against a WordPiece pipeline reconstructed from vocab.txt using the same tokenizers
engine. The audit records exact input/tokenizer hashes and local package versions.
It loads no model weights and changes no source text or model settings.

The four small pinned tokenizer/configuration files and their download manifest
are stored in data/inputs/tokenizers/bert_base_uncased/86b5e0934494bd15c9632b12f734a8a67f723594/.
Reproduce the BERT audit into a fresh destination with:

    PYTHONPATH=src python -m icl_pilot.maintenance.bert_story_tokenizer_audit --project-root /path/to/ICL-PILOT --tokenizer-dir /path/to/ICL-PILOT/data/inputs/tokenizers/bert_base_uncased/86b5e0934494bd15c9632b12f734a8a67f723594 --output-dir /path/to/fresh-audit

The external Gillam sample contains 182 children (91 TD, 91 SLI) and 543 available
narratives. Its original ZIP preserves all members under data/raw/gillam/2026-10-28/.
Reproduce it in fresh destinations with:

    PYTHONPATH=src python -m icl_pilot.maintenance.gillam_corpus /path/to/Gillam.zip --project-root /path/to/ICL-PILOT

Gillam age/sex folder disagreements are logged; literal CHI ages and sex define
selection. @Types agrees with the clinical-group directory for all 668 source
transcripts. Four unknown-month records lie outside the study age window and
are not assigned invented month values. The sessions a-1-68pre and a-1-68 share
an explicit child ID stem, sex and birth date. Only the earliest eligible recording
is exported, so 92 eligible SLI transcript IDs represent 91 unique children. The
TD sample is matched after that correction. Birth-date agreement alone never
merges different child IDs. Raw files remain unchanged.

Participant selection, source-header exclusions and demographic consistency are source-quality decisions. They do not provide empirical evidence of generated linguistic fidelity or classifier utility. Scientific generation, fresh annotation/measurement, domain statistical revisions and classifier evaluation remain pending.

The [shared annotation preparation record](annotation_preparation.json) covers
the 238-file source-preservation and Chatter comparison, plus mixed-collection
software checks. Standard structural and alignment checks are required; optional
strict cross-utterance quotation/completion checks are not enabled for either
real or synthetic CHAT. The [file-level inventory](annotation_source_diagnostics.csv)
records current source/prepared status with original words and boundaries preserved.
