# Study documentation

Read [the definitive study status](current_strategy_handoff.md) first.

| Guide | Purpose |
|---|---|
| [Colab generation](guides/colab_generation_package.md) | Notebook modes, setup, smoke checks and exports. |
| [Model implementation](guides/model_selection.md) | The three pinned models and shared runtime policy. |
| [Prompt records](guides/generation_prompt_records.md) | Exact messages, output ancestry and hashes. |
| [Generation recovery](guides/generation_recovery.md) | Completed-call checkpoints and RNG restoration. |
| [ENNI extraction](guides/chat_narrative_parsing.md) | Participant metadata, eligibility and text preparation. |
| [Narrative segmentation](guides/narrative_segmentation.md) | Whole-narrative boundaries and preservation. |
| [Output processing](guides/post_generation_processing.md) | CHAT validation, TalkTag and linguistic measurement. |
| [Framework evaluation](guides/framework_evaluation.md) | Corpus metrics, classifier design and uncertainty. |
| [Framework notebook blocks](guides/framework_notebook.md) | Independent saved-text evaluation in the main notebook, switches, runtime, resume and export. |
| [Domain features](guides/domain_features.md) | KIDEVAL English UD extraction from final real/synthetic annotations, six-story units, missing values and exports. |
| [Reproducibility](reproducibility/README.md) | Current source, roster and tokenizer checks. |

All guides describe the definitive study. Scientific steps that have not been executed are identified as pending.
