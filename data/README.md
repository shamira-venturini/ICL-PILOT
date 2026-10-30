# Study data

Keep the official ZIPs under raw/enni/2026-10-28/ and raw/gillam/2026-10-28/ unchanged.
One ZIP per corpus retains every original member. Redundant expanded originals
can be removed only after byte-for-byte ZIP verification; the removal inventory
is in docs/reproducibility/original_archive_retention.json. ZIP-backed readers and
exact-member recovery avoid expanding the corpora again.

The canonical participant and prepared story tables are in configs/generation/inputs/. Their source hashes, eligibility and preparation audit define the study reference: 238 children aged 4–7, 190 TD and 48 SLI-labelled.

Use inputs/enni_verified_2026-10-28/chat_4_to_7/ for real-data annotation. It contains only the 238 verified study records. Words, boundaries, groups, ages and existing tiers remain literal; documented sex-header adjustments follow the original demographics table.

Keep generated runs, annotation outputs, CLAN measurements and derived feature tables in purpose-specific fresh directories. Preserve source, model, child/story, pair, batch, replicate and stage identities. Never treat repeated synthetic bundles as independent children.

The notebook verifies final text fingerprints before removing successful-run
intermediate CHAT. Keep final annotated files, retention receipts, audits and
diagnostics. Failed/incomplete runs retain their working files. Selected ENNI
study inputs and original generation CSVs/journals remain available.

Corpus and run payloads are ignored by Git. Current verification records are in docs/reproducibility/.
