# ICL-PILOT

Reusable tools for synthetic child-narrative generation, CHAT processing,
linguistic feature preparation and evaluation.

**The previous study is frozen on TranscendBU. The revised protocol is pending.**
See [the archive index](docs/provenance/study_freeze_2026-10-16.md) for the complete
working-tree snapshot, earlier archives and restoration instructions. Historical
data, measurements and results have been retired locally after hash verification.

| Interface | Purpose and status |
| --- | --- |
| [Generation and annotation](notebooks/colab_direct_generation_pipeline.ipynb) | Revised story counterbalancing, a pinned official Llama generator and a 640-new-token ceiling per stage; other protocol decisions pending. |
| [Framework evaluation](notebooks/synthtexteval_demo.ipynb) | Retained evaluation implementation; two input paths are pending. |
| [R analysis](notebooks/clinical_analyses_reordered.Rmd) | Retained statistical models; five input paths and protocol review are pending. |

Saved notebook outputs, older R variants and rendered reports are preserved in
the freeze. Retained interfaces have clear status notices and clean execution
outputs. The cleanup did not approve a revised scientific design or rerun results.

| Folder | Role |
| --- | --- |
| [src/](src/README.md) | Reusable Python package grouped by responsibility. |
| [configs/](configs/README.md) | Preserved generation assets and explicit pending analysis/evaluation inputs. |
| [notebooks/](notebooks/) | The three maintained execution interfaces. |
| [data/](data/README.md) | Inputs and generated outputs needed by future active runs. |
| [results/](results/README.md) | Outputs from separately identified evaluation runs. |
| [docs/](docs/README.md) | Workflow guides, pending decisions, historical context and archive provenance. |
| [tests/](tests/) | Python and base-R regression checks. |
| [tools/chatter/](tools/chatter/README.md) | Verified local CHAT validator and source record. |

Start with [the revised-study setup note](docs/guides/revised_study_setup.md).
The generation interface can run a software smoke check (`SMOKE_TEST=True`;
twelve jobs by default). On 20 October 2026, the author adopted revised story
counterbalancing in `enni_staged_4_to_7_counterbalance_v2` and corrected the generator
to `meta-llama/Llama-3.1-8B-Instruct`, with a pinned revision. See the
[model-selection note](docs/guides/model_selection.md) for rationale and proposed
comparisons within one Colab GPU. On 21 October, the output ceiling was raised to
640 new tokens for both stages based on real-story lengths. The author then
selected temperature 0.8 and top-p 0.9 for both stages and all cohorts, then removed
age/status and directional length cues from prompt instructions. The current
reviewed generation/rewriting pair shares story-information fields and guidance;
story/exemplar and replicate IDs remain in logs. Canonical story content,
the exemplar CSV and child roster retain the earlier baseline. Further protocol
review remains pending. New generation requires a fresh run folder; Colab
supplies its own runtime and cannot use arbitrary local Mac paths.

Before evaluation, supply the intended experiment's files in:

- [configs/analysis/inputs.R](configs/analysis/inputs.R): five R input paths.
- [configs/evaluation/inputs.json](configs/evaluation/inputs.json): two framework input paths.

Both evaluation entry points stop before data processing while inputs are pending.
The current framework settings, raw-feature definitions and statistical models
remain available for review. New data/schema compatibility and the revised design
still require validation once the inputs exist. New framework results use fresh
folders under `results/framework_based_evaluation/`; R figures use dated folders
under `results/linguistic_evaluation/`.

From the repository root:

```bash
PYTHONPATH=src python -m icl_pilot validate-layout
PYTHONPATH=src python -m icl_pilot show-manifest --limit 5
PYTHONPATH=src python -m pytest -q
Rscript --vanilla tests/test_r_analysis_inputs.R
```

The configured local `.venv/` was backed up and its installed packages recorded.
It remains locally because the PyCharm project interpreter points to it. Git
history and the current uncommitted state are included in the full freeze.
