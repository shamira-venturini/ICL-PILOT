from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"
LEGACY_DIR = ROOT / "legacy"

RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
GENERATED_DIR = DATA_DIR / "generated"
EVALUATION_DIR = DATA_DIR / "evaluation"

COUNTERBALANCE_CSV = CONFIGS_DIR / "generation" / "cross_story_counterbalancing.csv"
MANIFEST_CSV = CONFIGS_DIR / "generation" / "sample_wise_multistep_manifest.csv"


def expected_layout() -> dict[str, Path]:
    return {
        "configs": CONFIGS_DIR,
        "data": DATA_DIR,
        "data/raw": RAW_DIR,
        "data/interim": INTERIM_DIR,
        "data/processed": PROCESSED_DIR,
        "data/generated": GENERATED_DIR,
        "data/evaluation": EVALUATION_DIR,
        "docs": DOCS_DIR,
        "legacy": LEGACY_DIR,
        "counterbalance_csv": COUNTERBALANCE_CSV,
        "generation_manifest_csv": MANIFEST_CSV,
    }
