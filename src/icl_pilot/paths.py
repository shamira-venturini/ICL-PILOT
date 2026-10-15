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
GENERATION_CONFIG_JSON = CONFIGS_DIR / "generation" / "experiment.json"
FROZEN_ROSTER_CSV = CONFIGS_DIR / "generation" / "master_frozen_roster_4_to_7_year_old.csv"
STORY_REFERENCE_JSON = CONFIGS_DIR / "generation" / "story_unit_reference.json"


def expected_layout() -> dict[str, Path]:
    """Required generation code/configuration; run/data directories are separate."""
    return {
        "configs": CONFIGS_DIR,
        "source_package": ROOT / "src" / "icl_pilot",
        "generation_notebook": ROOT / "notebooks" / "colab_direct_generation_pipeline.ipynb",
        "counterbalance_csv": COUNTERBALANCE_CSV,
        "generation_config_json": GENERATION_CONFIG_JSON,
        "frozen_roster_csv": FROZEN_ROSTER_CSV,
        "story_reference_json": STORY_REFERENCE_JSON,
        "stage1_prompt_template": CONFIGS_DIR / "generation" / "prompts" / "stage1.json",
        "stage2_prompt_template": CONFIGS_DIR / "generation" / "prompts" / "stage2.json",
    }
