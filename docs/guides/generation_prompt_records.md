# Prompt and output provenance

The canonical Stage 1 and Stage 2 templates are configs/generation/prompts/stage1.json and stage2.json. The project-authored story scaffolds are in story_unit_reference.json. The experiment names the complete prepared exemplar CSV and definitive pair roster.

For every call, preserve:

- Model key, exact checkpoint/revision and run identity.
- Cohort, pair, generation batch, replicate, target and selected exemplar identities.
- Canonical messages and native model-facing messages/rendered prompt.
- Template options, stop IDs and effective generation settings.
- Raw output, output token count, finish reason and content hash.
- For Stage 2, the exact Stage 1 parent and its hash.

Do not insert age/group labels or bookkeeping IDs into model-facing templates. Keep them in the journal and manifests. Do not shorten exemplars, repair grammatical forms or reconstruct partial words.

Stage 2 ancestry must resolve to the same model, pair, replicate and target. Missing, ambiguous or hash-mismatched parents stop execution. A replicated story ID is not evidence of ancestry without the parent record.

The notebook exports prompt_records.jsonl with generation_design/ and raw generation outputs. Completed-call checkpoints and journal boundaries support [verified recovery](generation_recovery.md). Never invent missing record IDs, model versions, token counts or stopping reasons.
