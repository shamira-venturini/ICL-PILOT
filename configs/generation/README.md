# Generation protocol

## Authoritative files

| File | Role |
|---|---|
| experiment.json | Shared design, cohort sizes, sampling, seed policy and input paths. |
| models.json | Exact model IDs/revisions and native chat options. |
| master_frozen_roster_enni_2026-10-28.csv | All 48 one-to-one within-cohort age-matched pairs and their order. |
| cross_story_counterbalancing.csv | Two reversed opposite-set exemplar assignments for each target. |
| story_unit_reference.json | Project-authored, picture-derived content scaffolds for A1–B3. |
| prompts/stage1.json | Complete generation instructions. |
| prompts/stage2.json | Complete rewriting instructions. |
| inputs/enni_participants_2026-10-28.csv | Canonical participant identity, group, raw/completed age and source hashes. |
| inputs/story_unit_narratives_enni_2026-10-28.csv | Complete prepared child narratives and utterance arrays. |
| inputs/story_unit_narratives_enni_2026-10-28.parsing_qc.json | Source identity, exclusion and deletion-only preparation audit. |

## Design

The reference contains 238 children, 190 TD and 48 SLI-labelled, at completed ages 48 <= months < 96. The original demographics table independently checks group and age year. Age cohorts do not round days upward. Fine within-cohort Hungarian matching uses days/30.

| Age | TD reference | SLI reference/pairs | Generation batches | Stories per stage per model |
|---|---:|---:|---:|---:|
| 4 | 45 | 11 | 4 | 660 |
| 5 | 46 | 13 | 5 | 780 |
| 6 | 49 | 11 | 4 | 660 |
| 7 | 50 | 13 | 5 | 780 |
| Total | 190 | 48 | 18 | 2,880 |

Six stories form an analytical bundle in A1–A3, B1–B3 order, retaining story boundaries. Ten replicates yield 480 bundles per stage per model, 5,760 stories per model and 17,280 across three models. Repeated bundles are not independent human participants.

For each target, exemplars come from the opposite story set. Five replicates use each of two reversed assignments. Stage 1 receives the scaffold and TD exemplar; Stage 2 receives the scaffold, selected SLI exemplar and that model's exact Stage 1 parent.

## Model execution

The notebook selects one model key per fresh run. All models use their native templates with the shared system sentence folded into the user message. Qwen thinking is disabled. Both stages use temperature 0.8, top-p 0.9, top-k 0 and 640 new tokens, with native EOS/end-of-turn stopping. Loading uses the specified 4-bit NF4 configuration and records the actual compute dtype and any float16 fallback.

Master seed 42 initializes Python, NumPy and PyTorch CPU/CUDA once before model loading. Replicates are consecutive stochastic draws, without reseeding. Completed-call checkpoints retain RNG states and exact parents. Use verified recovery, never reconstruct missing parents or narrative seeds.

Prompts preserve complete prepared exemplars. They do not prescribe a narrator age, diagnosis, error pattern or verbal length target. Text preparation removes transcription notation without guessing words, repairing grammar or reconstructing partial words.

## Execution and outputs

Use [the generation notebook](../../notebooks/colab_direct_generation_pipeline.ipynb) and [its guide](../../docs/guides/colab_generation_package.md). Smoke outputs are software fixtures and cannot enter scientific evaluation.

Every run has a fresh directory, an effective configuration, source hashes, selected exemplar snapshot, schedule, pair/batch manifests, rendered prompt records, raw outputs, token/finish records and parent hashes. Ten replicates are the study default; an explicit smoke override must be recorded and cannot redefine the scientific design.

The [reproducibility records](../../docs/reproducibility/README.md) contain the source and roster checks. Mean absolute matching gap is approximately 0.104 months and maximum 0.433 months.
