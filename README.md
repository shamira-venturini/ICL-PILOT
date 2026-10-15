# ICL-PILOT

ICL-PILOT generates synthetic ENNI-style child narratives using age-matched,
cross-story exemplars and a two-stage TD-like to SLI-conditioned workflow.

Start with [the generation and annotation notebook](notebooks/colab_direct_generation_pipeline.ipynb).
It provides separate blocks for setup, generation, CHAT conversion, TalkTag error
annotation, QC and exports. Run them sequentially or start from saved CSV/CHAT
inputs. Use the project
revision accompanying the notebook and provide the ENNI prompt inputs listed
in the configuration. The source corpus is not supplied by an installation.

The shared protocol covers ages four through seven. Its settings and recorded
child pairs are described in [the configuration guide](configs/generation/README.md).
The author confirmed a common generation configuration across those cohorts.
New runs store input fingerprints and resolved model/runtime metadata.
They also record the exact inference prompts and Stage 1–Stage 2 parent links.
The notebook's `REPLICATES` setting can change the volume for a new run without
editing the frozen study settings; counts are shown before model loading.

| Interface | Purpose |
| --- | --- |
| [Generation and annotation notebook](notebooks/colab_direct_generation_pipeline.ipynb) | Prepare synthetic narratives and error-annotated CHAT files. |
| [Python evaluation notebook](notebooks/synthtexteval_demo.ipynb) | Existing framework evaluation in Colab/Jupyter. |
| [R Markdown analysis](clinical_analyses_reordered.Rmd) | Existing ENNI linguistic and statistical analyses. |

The evaluation files retain their existing code and results. This consolidation
does not revalidate their analyses. Read [the workflow guide](docs/colab_generation_package.md)
for run modes, input requirements and pending GPU/checkpoint work.

- `configs/generation/experiment.json`: shared settings and input paths.
- `configs/generation/master_frozen_roster_4_to_7_year_old.csv`: the single frozen roster for ages four–seven.
- `configs/generation/story_unit_reference.json`: fixed picture-derived story scaffolds supplied with the configuration.
- `configs/generation/prompts/`: the frozen Stage 1 and Stage 2 message templates.
- `src/icl_pilot/generation/`: reusable setup and validation behind the notebook.
- `configs/annotation/talktag.json`: pinned TalkTag source and error-annotation options for new runs.
- `src/icl_pilot/annotation.py` and `pipeline.py`: annotation QC and file-based notebook steps.
- `data/generated/`: existing artifacts and separate new run folders.
- `docs/current_strategy_handoff.md`: decisions and known provenance issues.
- `clinical_analyses_reordered.Rmd`: existing integrated analysis; only its roster path changed during consolidation.

Lightweight inspection from the repository root:

```bash
PYTHONPATH=src python -m icl_pilot validate-layout
PYTHONPATH=src python -m icl_pilot show-counterbalance
PYTHONPATH=src python -m icl_pilot show-manifest --limit 5
```

The manifest command displays jobs from the shared configuration rather than
the retired report-era table. Historical configurations and generation source
are externally backed up with checksums; the local audit is under
`outputs/maintenance/generation_cleanup_2026-10-14/`.

After generation, use [the post-generation processing guide](docs/post_generation_processing.md)
for CSV-to-CHAT conversion and its metadata checks. Conversion preserves the
existing utterance rules, requires a fresh output directory and retains available
prompt-record links. TalkTag error annotation is included in the same notebook;
CLAN measurement and evaluation remain separate. The notebook has been checked
with mocked inference; a real Colab GPU integration run remains pending.
