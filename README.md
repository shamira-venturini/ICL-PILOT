# ICL-PILOT

This repository is now split into two clear areas:

- `data/`, `configs/`, `docs/`, and `src/icl_pilot/` hold the clean scaffold for the next generation pipeline based on the sample-wise multi-step strategy described in the project report.
- `legacy/phase1_prompting/` preserves the earlier exploratory prompting experiments, scripts, and outputs without letting them dominate the working tree.

## Repository layout

- `data/raw/enni/`: original ENNI corpus files as currently available in this repo.
- `data/raw/story_units/`: story-level extracted or normalized CHAT files from the previous phase.
- `data/interim/`: reproducible intermediate tables created from raw sources.
- `data/processed/`: prompt-ready manifests and normalized datasets for the new strategy.
- `data/generated/`: new TD stage-1 and DLD stage-2 generations.
- `data/evaluation/`: evaluation-ready exports and metrics.
- `configs/generation/`: checked-in configuration files derived from the revised strategy in the report.
- `docs/`: transition notes and strategy documentation.
- `legacy/phase1_prompting/`: preserved phase-1 code, outputs, and generated artifacts.
- `src/icl_pilot/`: small package scaffold for building the new pipeline as code instead of one-off scripts.

## Next-step workflow

1. Normalize and validate the raw source locations under `data/raw/`.
2. Build reproducible extraction and pairing steps against the manifests in `configs/generation/`.
3. Implement stage-1 TD baseline generation and stage-2 TD-to-DLD transformation in `src/icl_pilot/`.
4. Save all new outputs under `data/`, not at the repository root.

## Quick inspection

With `src` on `PYTHONPATH`, you can inspect the scaffold:

```bash
PYTHONPATH=src python -m icl_pilot validate-layout
PYTHONPATH=src python -m icl_pilot show-counterbalance
PYTHONPATH=src python -m icl_pilot show-manifest --limit 5
```
