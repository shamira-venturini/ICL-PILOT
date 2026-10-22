from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from icl_pilot.cli import build_parser
from icl_pilot.generation.frozen_roster import build_frozen_roster_manifest
from icl_pilot.features.generated_dev_measures import _bundle_metadata
from icl_pilot.generation import (
    build_real_bundle_template,
    build_generation_package,
    build_rotating_generation_schedule,
    load_frozen_roster,
)


def _developmental_input(tmp_path: Path) -> Path:
    # Ages deliberately make a greedy nearest-neighbour match suboptimal.
    rows = [
        (401, "SLI", "4;02.00", 50, 50.0),
        (402, "SLI", "4;03.00", 51, 51.0),
        (410, "TD", "4;02.27", 50, 50.9),
        (420, "TD", "4;00.00", 48, 48.0),
    ]
    frame = pd.DataFrame(rows, columns=["File_ID", "Group", "Age", "Age(Month)", "Age_Month_approx"])
    frame["File_kideval"] = frame["File_ID"].astype(str) + ".cha"
    path = tmp_path / "developmental.csv"
    frame.to_csv(path, index=False)
    return path


def test_pairing_without_profiles_preserves_global_age_assignment(tmp_path: Path) -> None:
    output = tmp_path / "roster.csv"
    assert build_frozen_roster_manifest(str(_developmental_input(tmp_path)), str(output)) == 0
    roster = pd.read_csv(output)
    assert list(zip(roster["sli_child_id"], roster["td_child_id"])) == [(401, 420), (402, 410)]
    assert roster["td_child_id"].is_unique
    assert roster[["sli_severity_band", "sli_profile_label"]].isna().all().all()


def test_rotating_schedule_accepts_rosters_without_legacy_labels(tmp_path: Path) -> None:
    output = tmp_path / "roster.csv"
    build_frozen_roster_manifest(str(_developmental_input(tmp_path)), str(output))
    pd.read_csv(output).drop(columns=["sli_severity_band", "sli_profile_label"]).to_csv(output, index=False)
    roster = load_frozen_roster(output)
    assert roster["sli_profile_label"].eq("").all()
    counterbalance = Path(__file__).resolve().parents[1] / "configs/generation/cross_story_counterbalancing.csv"
    schedule = build_rotating_generation_schedule(output, counterbalance, n_replicates=2, pairs_per_round=1)
    assert len(schedule) == 24
    assert schedule["round_id"].nunique() == 2
    assert schedule.groupby(["pair_id", "replicate_id"])["target_story"].nunique().eq(6).all()
    assert all(target[0] != exemplar[0] for target, exemplar in
               zip(schedule["target_story"], schedule["prompt_td_story_e1"]))


def test_real_templates_include_developmental_children_without_profiles(tmp_path: Path) -> None:
    template = build_real_bundle_template(_developmental_input(tmp_path))
    assert set(template["bundle_id"]) == {"real_401", "real_402", "real_410", "real_420"}
    assert not any(column.startswith("severity_") for column in template.columns)


def test_generation_package_preserves_existing_frozen_roster(tmp_path: Path) -> None:
    developmental = _developmental_input(tmp_path)
    roster = tmp_path / "four_year_old_frozen_roster.csv"
    build_frozen_roster_manifest(str(developmental), str(roster))
    frame = pd.read_csv(roster)
    frame["sli_profile_label"] = "historical_profile"
    frame.to_csv(roster, index=False)
    original = roster.read_bytes()
    build_generation_package(
        developmental,
        Path(__file__).resolve().parents[1] / "configs/generation/cross_story_counterbalancing.csv",
        tmp_path,
        n_replicates=1,
    )
    assert roster.read_bytes() == original


def test_duplicate_children_are_rejected_without_the_old_profile_join(tmp_path: Path) -> None:
    path = _developmental_input(tmp_path)
    frame = pd.read_csv(path)
    pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate child/group"):
        build_frozen_roster_manifest(str(path), str(tmp_path / "roster.csv"))
    with pytest.raises(ValueError, match="duplicate child/group"):
        build_real_bundle_template(path)


def test_new_generation_metadata_keeps_unknown_profiles_missing(tmp_path: Path) -> None:
    frame = pd.DataFrame({
        "bundle_id": ["pair_sli401_td420_rep01"] * 6,
        "pair_id": ["pair_sli401_td420"] * 6,
        "round_id": ["round_01"] * 6,
        "replicate_id": [1] * 6,
        "target_story": ["A1", "A2", "A3", "B1", "B2", "B3"],
        "story_slot_order": range(1, 7),
        "sli_child_id": [401] * 6,
        "td_child_id": [420] * 6,
        "sli_age": ["4;02.00"] * 6,
    })
    path = tmp_path / "generations.csv"
    frame.to_csv(path, index=False)
    metadata = _bundle_metadata(path)
    assert len(metadata) == 2
    assert metadata["sli_severity_band_generated_from"].isna().all()
    assert metadata["sli_profile_label_generated_from"].isna().all()
    frame["sli_severity_band"] = "historical_band"
    frame["sli_profile_label"] = "historical_profile"
    frame.to_csv(path, index=False)
    historical = _bundle_metadata(path)
    assert historical["sli_profile_label_generated_from"].eq("historical_profile").all()


def test_cli_retires_profiling_commands_but_keeps_raw_feature_builder() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    for command in ["build-age-adjusted-severity-table", "build-severity-profile-table",
                    "build-severity-bands", "build-story-generation-design", "build-architecture-loocv-plan"]:
        assert command not in help_text
    assert "build-severity-feature-table" in help_text
    args = parser.parse_args(["build-frozen-roster", "developmental.csv", "output.csv"])
    assert args.output_csv == "output.csv"
    assert not hasattr(args, "severity_profile_csv")
