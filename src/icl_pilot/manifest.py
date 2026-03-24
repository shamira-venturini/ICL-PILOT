from __future__ import annotations

from csv import DictReader
from dataclasses import dataclass
from pathlib import Path

from .paths import COUNTERBALANCE_CSV, MANIFEST_CSV


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


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(DictReader(handle))


def load_counterbalance_rules(path: Path = COUNTERBALANCE_CSV) -> list[CounterbalanceRule]:
    return [CounterbalanceRule(**row) for row in _read_rows(path)]


def load_generation_manifest(path: Path = MANIFEST_CSV) -> list[GenerationConfig]:
    rows = _read_rows(path)
    configs: list[GenerationConfig] = []
    for row in rows:
        configs.append(
            GenerationConfig(
                cohort=row["cohort"],
                target_story=row["target_story"],
                target_subject_ids=row["target_subject_ids"],
                target_age=row["target_age"],
                e1_story=row["e1_story"],
                td_baselines=int(row["td_baselines"]),
                e2_subject_ids=row["e2_subject_ids"],
                e2_subject_age=row["e2_subject_age"],
                e2_story=row["e2_story"],
                dld_outputs=int(row["dld_outputs"]),
            )
        )
    return configs
