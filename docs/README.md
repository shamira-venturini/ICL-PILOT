# Documentation

Start with [`current_strategy_handoff.md`](current_strategy_handoff.md) for the
current study status, pending protocol decisions and known limitations.
The previous study is frozen externally; use the archive index under `provenance/`
for historical data and results.

| Folder | Contents |
| --- | --- |
| [`guides/`](guides/) | Instructions and reference notes for the maintained workflow. |
| [`background/`](background/) | The original project report and earlier or exploratory design notes. |
| [`provenance/`](provenance/) | Historical audit records, backup verification and relocation receipts. |

The guides are:

| Document | Purpose |
| --- | --- |
| [Revised-study setup](guides/revised_study_setup.md) | Current preparation status and decisions still needed. |
| [Model selection](guides/model_selection.md) | Official Llama correction, selection rationale and proposed text/vision comparisons within one Colab GPU. |
| [Generation and annotation notebook](guides/colab_generation_package.md) | Run modes, generation volume, TalkTag, persistence and exports. |
| [CHAT narrative extraction](guides/chat_narrative_parsing.md) | Build and check the frozen exemplar input. |
| [Prompt records](guides/generation_prompt_records.md) | Frozen templates, prompt history and generation ancestry. |
| [Post-generation processing](guides/post_generation_processing.md) | CSV-to-CHAT conversion, formatting, annotation and validation. |
| [Four-year-old roster](guides/four_year_old_frozen_roster.md) | Supporting explanation of the recorded pairs and their reconciliation. |

The background documents are:

| Document | Status |
| --- | --- |
| [Original project report](background/data-augmentation_project-report.pdf) | Historical project framing and results. |
| [Four-year-old evaluation plan](background/four_year_old_evaluation_plan.md) | Earlier bundle-level evaluation proposal. |
| [Profiling and generation proposal](background/generation_and_evaluation_process.tex) | Historical proposal; profiling is retired. |
| [Image-based task design](background/image_based_story_task_design.md) | Exploratory picture-derived task-packet workflow. |
| [Story spine prompt pack](background/story_spine_prompt_pack.md) | Earlier scaffold usage and prompt-shell notes. |
| [Story-unit SemR-p plan](background/story_unit_semrp_plan.md) | Exploratory content-evaluation plan. |

Background notes may describe proposed procedures or historical data paths.
Use the current handoff and operational configuration to establish the active
protocol. Commands in the guides are run from the repository root unless stated
otherwise. Moving these documents did not change experimental methods or data.
