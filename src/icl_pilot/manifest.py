from __future__ import annotations

from csv import DictReader
from dataclasses import dataclass
from pathlib import Path

from .paths import COUNTERBALANCE_CSV, GENERATION_CONFIG_JSON


@dataclass(frozen=True)
class CounterbalanceRule:
    target_set: str
    target_story: str
    e1_story: str
    e2_story: str


@dataclass(frozen=True)
class GenerationConfig:
    cohort: str
    target_story: str
    target_subject_ids: str
    target_age: str
    e1_story: str
    td_baselines: int
    e2_subject_ids: str
    e2_subject_age: str
    e2_story: str
    dld_outputs: int
    pair_id: str
    round_id: str
    replicate_id: int


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(DictReader(handle))


def load_counterbalance_rules(path: Path = COUNTERBALANCE_CSV) -> list[CounterbalanceRule]:
    return [CounterbalanceRule(**row) for row in _read_rows(path)]


def load_generation_manifest(path: Path = GENERATION_CONFIG_JSON) -> list[GenerationConfig]:
    from .generation.config import compile_experiment_design

    _config, schedule, _evaluation, _rosters = compile_experiment_design(path)
    rows = schedule.to_dict(orient="records")
    configs: list[GenerationConfig] = []
    for row in rows:
        configs.append(
            GenerationConfig(
                cohort=row["cohort"],
                target_story=row["target_story"],
                target_subject_ids=str(row["td_child_id"]),
                target_age=row["td_age"],
                e1_story=row["prompt_td_story_e1"],
                td_baselines=1,
                e2_subject_ids=str(row["sli_child_id"]),
                e2_subject_age=row["sli_age"],
                e2_story=row["prompt_sli_story_e2"],
                dld_outputs=1,
                pair_id=row["pair_id"],
                round_id=row["round_id"],
                replicate_id=int(row["replicate_id"]),
            )
        )
    return configs
