# Source map

| Package | Responsibility |
|---|---|
| generation | Frozen roster, cross-story schedule, prompts, model adapters, execution, recorded parents and RNG recovery. |
| chat | Canonical metadata, text preparation, generated segmentation/CHAT, TalkTag and validation/preservation. |
| features | CLAN table readers and interpretable structural, lexical, disruption and morphosyntactic measures. |
| evaluation | Text exports, shared real splits, generator-aware classifiers, paired uncertainty and corpus diagnostics. |
| maintenance | Explicit source intake, tokenizer auditing and optional data-format utilities. |

The active entry points are the two Python notebooks and the R analysis source. paths.py names the definitive roster; configs/generation/experiment.json names the authoritative prepared exemplars.

maintenance.enni_corpus imports the official ZIP without changing raw files, checks the original demographic table, excludes incomplete identities, builds the participant/story tables and within-cohort roster, and prepares study-only CHAT copies. It requires Rscript/readxl for the one-time XLS cross-check. Generation itself reads the frozen CSVs.

chat.text_cleaning performs deletion-only preparation before narratives are saved. generation.exemplars validates literal text. chat.segmentation predicts whole-narrative boundaries without replacing words; chat.annotation enforces Chatter gates and source preservation. These checks do not establish clinical annotation accuracy.

generation.randomness and generation.recovery implement run-level seeding and exact completed-call recovery. generation.prompt_records preserves canonical messages, native model-facing templates, output hashes and Stage 2 ancestry.

The framework_* modules implement the classifier and descriptive protocol in configs/evaluation/protocol.json. Gaussian Fréchet distance uses means and covariances. Dataset identity, generator lineage and common reference samples are validated before fits.

Feature builders retain their public function names, including severity_feature names for raw linguistic tables. They do not calculate a severity/profile model. Statistical specifications remain in the R notebook and require the documented domain-analysis revisions before reporting results.

Maintenance utilities operate only when explicitly invoked. They are not an automatic preprocessing sequence. Any transformation must preserve or explicitly audit source words, headers, boundaries, identities and tiers.
