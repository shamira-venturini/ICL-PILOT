# ICL-PILOT

The definitive study evaluates whether three instruction-tuned language models can generate ENNI narratives that preserve developmental and clinical language patterns and support TD-versus-DLD classification.

## Study design

- Real reference: 238 children aged 4–7 years, 190 TD and 48 SLI-labelled.
- Prompt selection: 48 age-matched TD–SLI pairs, organized into 18 age-specific generation batches.
- Models: Llama-3.1-8B-Instruct, Qwen3-8B with thinking disabled, and Gemma-2-9B-IT.
- Stage 1 generates from a story scaffold and a cross-story TD exemplar. Stage 2 rewrites that same model's exact output with a cross-story SLI exemplar.
- Six targets and ten replicates yield 2,880 stories per stage per model, 480 six-story bundles per stage per model, and 17,280 stories overall.
- Sampling: temperature 0.8, top-p 0.9, top-k 0, 640 new tokens per stage. Seed 42 initializes each model run once.

The procedure is evaluated as one practical design. Stage changes combine rewriting and exemplar conditioning.

## Working files

| Location | Purpose |
|---|---|
| [Generation configuration](configs/generation/README.md) | Model pins, prompts, story scaffolds, prepared exemplars and the definitive pair roster. |
| [Generation notebook](notebooks/colab_direct_generation_pipeline.ipynb) | Generation, CHAT conversion and TalkTag annotation. |
| [Framework notebook](notebooks/synthtexteval_demo.ipynb) | Shared splits, BERT classifiers and corpus diagnostics. |
| [Domain analysis](notebooks/clinical_analyses_reordered.Rmd) | Linguistic analysis code; feature inputs and statistical revisions are pending. |
| [Documentation](docs/README.md) | Current methods and execution guides. |
| [Reproducibility](docs/reproducibility/README.md) | Source integrity, participant selection, roster and tokenizer checks. |

## Execution

Install the project dependencies in the intended runtime. Colab uses its own environment. From the repository root:

    PYTHONPATH=src python -m icl_pilot validate-layout
    PYTHONPATH=src python -m icl_pilot show-manifest --limit 5
    PYTHONPATH=src python -m pytest -q
    Rscript --vanilla tests/test_r_analysis_inputs.R

The notebooks stop before scientific evaluation when their required inputs are absent. Configure the four pending feature paths in [R inputs](configs/analysis/inputs.R) and the two combined text inputs in [framework inputs](configs/evaluation/inputs.json). The definitive roster is already selected.

Scientific generation, fresh feature extraction, domain statistical revisions and classifier fitting remain to be completed. Local software checks are not scientific results.
