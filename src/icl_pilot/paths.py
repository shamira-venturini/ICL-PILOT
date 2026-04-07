from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"
LEGACY_DIR = ROOT / "archive"

RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
GENERATED_DIR = DATA_DIR / "generated"
EVALUATION_DIR = DATA_DIR / "evaluation"
ENNI_IMAGES_DIR = DATA_DIR / "ENNI_images"
STORY_PACKET_DIR = PROCESSED_DIR / "story_packets"

COUNTERBALANCE_CSV = CONFIGS_DIR / "generation" / "cross_story_counterbalancing.csv"
MANIFEST_CSV = CONFIGS_DIR / "generation" / "sample_wise_multistep_manifest.csv"
FROZEN_ROSTER_CSV = CONFIGS_DIR / "generation" / "four_year_old_frozen_roster.csv"


def expected_layout() -> dict[str, Path]:
    return {
        "configs": CONFIGS_DIR,
        "data": DATA_DIR,
        "data/raw": RAW_DIR,
        "data/interim": INTERIM_DIR,
        "data/processed": PROCESSED_DIR,
        "data/generated": GENERATED_DIR,
        "data/evaluation": EVALUATION_DIR,
        "data/ENNI_images": ENNI_IMAGES_DIR,
        "data/processed/story_packets": STORY_PACKET_DIR,
        "docs": DOCS_DIR,
        "archive": LEGACY_DIR,
        "counterbalance_csv": COUNTERBALANCE_CSV,
        "generation_manifest_csv": MANIFEST_CSV,
        "frozen_roster_csv": FROZEN_ROSTER_CSV,
    }
