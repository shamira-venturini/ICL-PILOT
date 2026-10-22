# Revised study: preparation status

The author authorized freezing the previous study and retaining adjustable
implementation for a fresh experiment. The complete pre-restructuring working
tree is preserved on TranscendBU; see the study-freeze index in `docs/provenance/`.

The three retained interfaces are under `notebooks/`:

- `colab_direct_generation_pipeline.ipynb`: generation, CHAT conversion and TalkTag.
- `synthtexteval_demo.ipynb`: framework evaluation, awaiting two explicit inputs.
- `clinical_analyses_reordered.Rmd`: retained R models, awaiting five explicit inputs.

Saved notebook execution outputs were archived and cleared. The older R variant,
extracted R script, rendered HTML reports, old data, measurements, results and QC
records are archived externally. Their original paths are recorded in the freeze
manifest. No historical result was recomputed or relabelled as a revised result.

Before generating a revised dataset, record decisions on:

- the reviewer comments, their evidence and the resulting protocol changes;
- generation versus rewriting controls and the interpretation they support;
- model/runtime versions and generation settings;
- prompt/evaluation child separation, dependence and evaluation units;
- annotation, CLAN processing, feature definitions and the analysis plan.

On 20 October 2026, the author adopted the proposed symmetric counterbalancing
schedule as the first change after the freeze. The working configuration is
`enni_staged_4_to_7_counterbalance_v2`; its
[`generation README`](../../configs/generation/README.md) records all assignments.
The default ten replicates use each assignment five times per prompt pair and
target, preserving opposite-set exemplars and balancing complexity levels overall.
The existing alternating scheduler is retained. Smoke-test truncation and odd
replicate overrides do not guarantee exact balance.

The author also requested an official Llama checkpoint and confirmed that model
comparisons must fit within one Colab GPU. The active generator is now
`meta-llama/Llama-3.1-8B-Instruct`, with a pinned revision. The
[model-selection note](model_selection.md) records the provenance correction,
selection criteria, proposed Qwen3 text comparator and possible vision pilot.
Those comparator/vision conditions remain proposals. The official checkpoint
requires approved Hugging Face access and a fresh Colab loading/inference check.

On 21 October 2026, the author adopted 640 new tokens for both stages and all
cohorts, based on the checked maximum of 568 tokens among real stories at ages
four through seven. The generation configuration records the rationale and
previous limits. The notebook prints and passes the configured limits.

The author subsequently selected temperature 0.8 and top-p 0.9 after checking
previous research. Both stages and all cohorts use these shared settings from
the generation configuration. The inherited top-k policy remains pending.

The author then requested removal of narrator-age and clinical/non-clinical
status references from prompt instructions. Templates and inserted scaffold
constraints now omit those cues. Age matching and group labels remain in
exemplar selection/provenance; story content, exemplar text and the Stage 1
parent are preserved. The [prompt guide](generation_prompt_records.md) records
the current scope and earlier wording. Child pairs remain the earlier baseline.
The author subsequently removed story-length requests from both stages,
including short-narrative wording and clause-shortening/detail-omission guidance.
The author then supplied the reviewed generation/rewriting pair: both stages
share the system message, all four story-information fields and full instruction
list, including exemplar-based wording, sentence construction, style and level
of detail. Stage 2 receives the exact Stage 1 parent to rewrite with its selected
exemplar. Additional constraints are content-only. Story/exemplar IDs, titles,
replicate IDs and ENNI labels are absent from the templates; identities remain
in validation and logs. The configured 640-new-token ceiling is unchanged. Other
scientific decisions, including the generation-versus-rewriting control, still
require review before a revised dataset is generated. The archived configuration
and results retain their original schedule.

Supply revised data paths through `configs/analysis/inputs.R` and
`configs/evaluation/inputs.json` once the corresponding files exist. New
generation uses a fresh run under `data/generated/`; framework results use fresh
folders under `results/framework_based_evaluation/`. R figures use dated folders
under `results/linguistic_evaluation/`; choose an output directory under `results/`
when rendering the R Markdown report. Setting input paths does not itself
validate dataset schemas, annotation equivalence or the revised study design.
