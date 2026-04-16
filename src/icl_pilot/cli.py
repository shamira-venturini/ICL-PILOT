from __future__ import annotations

import argparse

from .architecture_pilot_cv import build_architecture_loocv_plan
from .age_adjustment import build_age_adjusted_severity_table
from .alignment_repair import repair_alignments
from .batchalign_runner import run_morphotag
from .batchalign_batch_runner import run_morphotag_batched
from .bundled_text_perplexity import build_bundled_td_text_perplexity, build_bundled_text_perplexity
from .bundled_text_semantic_eval import build_bundled_text_semantic_evaluation
from .dss_features import build_dss_feature_table, merge_dss_into_master
from .error_codes import build_error_code_feature_table
from .error_rate_features import add_error_rate_features_to_master
from .feature_redundancy import audit_feature_redundancy
from .filelist import write_cha_filelist
from .frozen_roster import build_frozen_roster_manifest
from .generated_bundle_evaluation import build_generated_bundle_evaluation
from .grouped_error_features import build_grouped_error_feature_table
from .grouped_error_features import merge_grouped_error_features_into_master
from .generated_text_evaluation import build_generated_text_evaluation
from .manifest import load_counterbalance_rules, load_generation_manifest
from .pause_normalizer import normalize_initial_pauses
from .past_verb_features import (
    build_past_verb_feature_table,
    merge_past_verb_features_into_master,
)
from .paths import expected_layout
from .quote_normalizer import normalize_quote_terminators
from .severity_features import build_severity_feature_table
from .severity_bands import build_severity_bands
from .severity_profile import build_severity_profile_table
from .story_generation_design import build_story_generation_design
from .story_packet_design import (
    bootstrap_story_packets,
    build_story_packet_template,
    extract_story_panels,
)
from .story_unit_semrp import (
    build_story_unit_reference_template,
    extract_story_unit_narratives,
    score_story_unit_semrp,
)
from .tier_promotion import promote_batchalign_tiers
from .tier_stripper import strip_tiers


def _cmd_validate_layout() -> int:
    missing = [f"{name}: {path}" for name, path in expected_layout().items() if not path.exists()]
    if missing:
        print("Missing layout entries:")
        for item in missing:
            print(f"- {item}")
        return 1

    print("Layout OK")
    return 0


def _cmd_show_layout() -> int:
    for name, path in expected_layout().items():
        print(f"{name}: {path}")
    return 0


def _cmd_show_counterbalance() -> int:
    for rule in load_counterbalance_rules():
        print(
            f"{rule.target_set} {rule.target_story}: "
            f"E1={rule.e1_story} E2={rule.e2_story}"
        )
    return 0


def _cmd_show_manifest(limit: int | None) -> int:
    rows = load_generation_manifest()
    if limit is not None:
        rows = rows[:limit]

    for row in rows:
        print(
            f"{row.cohort} | target={row.target_story} ({row.target_subject_ids}, {row.target_age}) "
            f"| E1={row.e1_story} x{row.td_baselines} "
            f"| E2={row.e2_story} ({row.e2_subject_ids}, {row.e2_subject_age}) "
            f"| DLD outputs={row.dld_outputs}"
        )
    return 0


def _cmd_run_batchalign_morphotag(args: argparse.Namespace) -> int:
    return run_morphotag(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        keeptokens=not args.retokenize,
        lexicon=args.lexicon,
        override_cache=args.override_cache,
        force_cpu=args.force_cpu,
        workers=args.workers,
        dry_run=args.dry_run,
    )


def _cmd_run_batchalign_morphotag_batched(args: argparse.Namespace) -> int:
    return run_morphotag_batched(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        keeptokens=not args.retokenize,
        lexicon=args.lexicon,
        manifest_csv=args.manifest_csv,
        start_batch=args.start_batch,
        max_batches=args.max_batches,
        skip_existing=not args.no_skip_existing,
        dry_run=args.dry_run,
    )


def _cmd_write_cha_filelist(args: argparse.Namespace) -> int:
    return write_cha_filelist(root_dir=args.input_dir, output_file=args.output_file)


def _cmd_repair_batchalign_alignment(args: argparse.Namespace) -> int:
    return repair_alignments(root_dir=args.input_dir, dry_run=args.dry_run)


def _cmd_normalize_initial_pauses(args: argparse.Namespace) -> int:
    return normalize_initial_pauses(root_dir=args.input_dir, dry_run=args.dry_run)


def _cmd_normalize_quote_terminators(args: argparse.Namespace) -> int:
    return normalize_quote_terminators(root_dir=args.input_dir, dry_run=args.dry_run)


def _cmd_promote_batchalign_tiers(args: argparse.Namespace) -> int:
    return promote_batchalign_tiers(root_dir=args.input_dir, dry_run=args.dry_run)


def _cmd_strip_legacy_tiers(args: argparse.Namespace) -> int:
    return strip_tiers(root_dir=args.input_dir, dry_run=args.dry_run)


def _cmd_build_dss_feature_table(args: argparse.Namespace) -> int:
    return build_dss_feature_table(
        dss_root=args.dss_dir,
        output_csv=args.output_csv,
        output_qc_json=args.output_qc_json,
    )


def _cmd_merge_dss_into_master(args: argparse.Namespace) -> int:
    return merge_dss_into_master(
        master_csv=args.master_csv,
        dss_csv=args.dss_csv,
        output_csv=args.output_csv,
    )


def _cmd_audit_feature_redundancy(args: argparse.Namespace) -> int:
    return audit_feature_redundancy(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        corr_threshold=args.corr_threshold,
        cluster_threshold=args.cluster_threshold,
    )


def _cmd_build_severity_feature_table(args: argparse.Namespace) -> int:
    return build_severity_feature_table(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        output_spec_csv=args.output_spec_csv,
        output_summary_json=args.output_summary_json,
    )


def _cmd_build_age_adjusted_severity_table(args: argparse.Namespace) -> int:
    return build_age_adjusted_severity_table(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        output_model_csv=args.output_model_csv,
        output_summary_json=args.output_summary_json,
    )


def _cmd_build_error_code_feature_table(args: argparse.Namespace) -> int:
    return build_error_code_feature_table(
        transcript_root=args.transcript_root,
        output_csv=args.output_csv,
        output_inventory_json=args.output_inventory_json,
    )


def _cmd_build_grouped_error_feature_table(args: argparse.Namespace) -> int:
    return build_grouped_error_feature_table(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        output_summary_json=args.output_summary_json,
    )


def _cmd_merge_grouped_error_features_into_master(args: argparse.Namespace) -> int:
    return merge_grouped_error_features_into_master(
        master_csv=args.master_csv,
        grouped_csv=args.grouped_csv,
        output_csv=args.output_csv,
    )


def _cmd_add_error_rate_features_to_master(args: argparse.Namespace) -> int:
    return add_error_rate_features_to_master(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
    )


def _cmd_build_past_verb_feature_table(args: argparse.Namespace) -> int:
    return build_past_verb_feature_table(
        transcript_root=args.transcript_root,
        error_code_csv=args.error_code_csv,
        output_csv=args.output_csv,
        output_summary_json=args.output_summary_json,
    )


def _cmd_merge_past_verb_features_into_master(args: argparse.Namespace) -> int:
    return merge_past_verb_features_into_master(
        master_csv=args.master_csv,
        past_feature_csv=args.past_feature_csv,
        output_csv=args.output_csv,
    )


def _cmd_build_severity_profile_table(args: argparse.Namespace) -> int:
    return build_severity_profile_table(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        output_summary_json=args.output_summary_json,
    )


def _cmd_build_severity_bands(args: argparse.Namespace) -> int:
    return build_severity_bands(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        output_summary_json=args.output_summary_json,
    )


def _cmd_build_story_generation_design(args: argparse.Namespace) -> int:
    return build_story_generation_design(
        severity_banded_csv=args.severity_banded_csv,
        counterbalance_csv=args.counterbalance_csv,
        transcript_root=args.transcript_root,
        output_dir=args.output_dir,
        min_prompt_candidates=args.min_prompt_candidates,
        min_eval_candidates=args.min_eval_candidates,
    )


def _cmd_build_architecture_loocv_plan(args: argparse.Namespace) -> int:
    return build_architecture_loocv_plan(
        dev_measures_csv=args.dev_measures_csv,
        severity_csv=args.severity_csv,
        transcript_root=args.transcript_root,
        reference_manifest_csv=args.reference_manifest_csv,
        output_dir=args.output_dir,
        cohort=args.cohort,
    )


def _cmd_build_frozen_roster(args: argparse.Namespace) -> int:
    return build_frozen_roster_manifest(
        dev_measures_csv=args.dev_measures_csv,
        severity_profile_csv=args.severity_profile_csv,
        output_csv=args.output_csv,
        age_min_months=args.age_min_months,
        age_max_months=args.age_max_months,
    )


def _cmd_build_generated_bundle_evaluation(args: argparse.Namespace) -> int:
    return build_generated_bundle_evaluation(
        generated_severity_csv=args.generated_severity_csv,
        real_severity_csv=args.real_severity_csv,
        frozen_roster_csv=args.frozen_roster_csv,
        output_dir=args.output_dir,
        pairs_per_round=args.pairs_per_round,
    )


def _cmd_build_generated_text_evaluation(args: argparse.Namespace) -> int:
    return build_generated_text_evaluation(
        synthetic_root=args.synthetic_root,
        real_root=args.real_root,
        output_dir=args.output_dir,
        age_years=args.age_years,
    )


def _cmd_build_bundled_td_text_perplexity(args: argparse.Namespace) -> int:
    return build_bundled_td_text_perplexity(
        synthetic_root=args.synthetic_root,
        real_root=args.real_root,
        output_dir=args.output_dir,
        age_years=args.age_years,
        folds=args.folds,
        order=args.order,
        alpha=args.alpha,
        random_seed=args.random_seed,
        exclude_synthetic_source_children_from_real=args.exclude_synthetic_source_children_from_real,
    )


def _cmd_build_bundled_text_perplexity(args: argparse.Namespace) -> int:
    return build_bundled_text_perplexity(
        synthetic_root=args.synthetic_root,
        real_root=args.real_root,
        output_dir=args.output_dir,
        group=args.group,
        age_years=args.age_years,
        folds=args.folds,
        order=args.order,
        alpha=args.alpha,
        random_seed=args.random_seed,
        exclude_synthetic_source_children_from_real=args.exclude_synthetic_source_children_from_real,
    )


def _cmd_build_bundled_text_semantic_evaluation(args: argparse.Namespace) -> int:
    return build_bundled_text_semantic_evaluation(
        synthetic_root=args.synthetic_root,
        real_root=args.real_root,
        output_dir=args.output_dir,
        group=args.group,
        age_years=args.age_years,
        model_name=args.model_name,
        batch_size=args.batch_size,
        bootstrap_reps=args.bootstrap_reps,
        bootstrap_seed=args.bootstrap_seed,
        exclude_synthetic_source_children_from_real=args.exclude_synthetic_source_children_from_real,
    )


def _cmd_extract_story_panels(args: argparse.Namespace) -> int:
    return extract_story_panels(
        input_pdf=args.input_pdf,
        output_dir=args.output_dir,
        story_id=args.story_id,
        rotate_degrees=args.rotate_degrees,
    )


def _cmd_build_story_packet_template(args: argparse.Namespace) -> int:
    return build_story_packet_template(
        panel_dir=args.panel_dir,
        output_json=args.output_json,
        story_id=args.story_id,
        source_pdf=args.source_pdf,
    )


def _cmd_bootstrap_story_packets(args: argparse.Namespace) -> int:
    return bootstrap_story_packets(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        rotate_degrees=args.rotate_degrees,
        overwrite_templates=args.overwrite_templates,
    )


def _cmd_extract_story_unit_narratives(args: argparse.Namespace) -> int:
    return extract_story_unit_narratives(
        transcript_root=args.transcript_root,
        output_csv=args.output_csv,
    )


def _cmd_build_story_unit_reference_template(args: argparse.Namespace) -> int:
    return build_story_unit_reference_template(
        transcript_root=args.transcript_root,
        output_json=args.output_json,
    )


def _cmd_score_story_unit_semrp(args: argparse.Namespace) -> int:
    return score_story_unit_semrp(
        narrative_csv=args.narrative_csv,
        reference_json=args.reference_json,
        semrp_repo=args.semrp_repo,
        output_csv=args.output_csv,
        k=args.k,
        model_name_or_path=args.model_name_or_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Utilities for the cleaned ICL-PILOT scaffold.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("validate-layout")
    subparsers.add_parser("show-layout")
    subparsers.add_parser("show-counterbalance")

    manifest_parser = subparsers.add_parser("show-manifest")
    manifest_parser.add_argument("--limit", type=int, default=None)

    batchalign_parser = subparsers.add_parser("run-batchalign-morphotag")
    batchalign_parser.add_argument("input_dir")
    batchalign_parser.add_argument("output_dir")
    batchalign_parser.add_argument("--retokenize", action="store_true")
    batchalign_parser.add_argument("--lexicon", default=None)
    batchalign_parser.add_argument("--override-cache", action="store_true")
    batchalign_parser.add_argument("--force-cpu", action="store_true")
    batchalign_parser.add_argument("--workers", type=int, default=None)
    batchalign_parser.add_argument("--dry-run", action="store_true")

    batchalign_batched_parser = subparsers.add_parser("run-batchalign-morphotag-batched")
    batchalign_batched_parser.add_argument("input_dir")
    batchalign_batched_parser.add_argument("output_dir")
    batchalign_batched_parser.add_argument("--batch-size", type=int, default=200)
    batchalign_batched_parser.add_argument("--retokenize", action="store_true")
    batchalign_batched_parser.add_argument("--lexicon", default=None)
    batchalign_batched_parser.add_argument("--manifest-csv", default=None)
    batchalign_batched_parser.add_argument("--start-batch", type=int, default=1)
    batchalign_batched_parser.add_argument("--max-batches", type=int, default=None)
    batchalign_batched_parser.add_argument("--no-skip-existing", action="store_true")
    batchalign_batched_parser.add_argument("--dry-run", action="store_true")

    filelist_parser = subparsers.add_parser("write-cha-filelist")
    filelist_parser.add_argument("input_dir")
    filelist_parser.add_argument("output_file")

    repair_parser = subparsers.add_parser("repair-batchalign-alignment")
    repair_parser.add_argument("input_dir")
    repair_parser.add_argument("--dry-run", action="store_true")

    pause_parser = subparsers.add_parser("normalize-initial-pauses")
    pause_parser.add_argument("input_dir")
    pause_parser.add_argument("--dry-run", action="store_true")

    quote_parser = subparsers.add_parser("normalize-quote-terminators")
    quote_parser.add_argument("input_dir")
    quote_parser.add_argument("--dry-run", action="store_true")

    promote_parser = subparsers.add_parser("promote-batchalign-tiers")
    promote_parser.add_argument("input_dir")
    promote_parser.add_argument("--dry-run", action="store_true")

    strip_parser = subparsers.add_parser("strip-legacy-mor-gra")
    strip_parser.add_argument("input_dir")
    strip_parser.add_argument("--dry-run", action="store_true")

    dss_table_parser = subparsers.add_parser("build-dss-feature-table")
    dss_table_parser.add_argument("dss_dir")
    dss_table_parser.add_argument("output_csv")
    dss_table_parser.add_argument("output_qc_json")

    dss_merge_parser = subparsers.add_parser("merge-dss-into-master")
    dss_merge_parser.add_argument("master_csv")
    dss_merge_parser.add_argument("dss_csv")
    dss_merge_parser.add_argument("--output-csv", default=None)

    audit_parser = subparsers.add_parser("audit-feature-redundancy")
    audit_parser.add_argument("input_csv")
    audit_parser.add_argument("output_dir")
    audit_parser.add_argument("--corr-threshold", type=float, default=0.95)
    audit_parser.add_argument("--cluster-threshold", type=float, default=0.98)

    severity_parser = subparsers.add_parser("build-severity-feature-table")
    severity_parser.add_argument("input_csv")
    severity_parser.add_argument("output_csv")
    severity_parser.add_argument("output_spec_csv")
    severity_parser.add_argument("output_summary_json")

    age_adjust_parser = subparsers.add_parser("build-age-adjusted-severity-table")
    age_adjust_parser.add_argument("input_csv")
    age_adjust_parser.add_argument("output_csv")
    age_adjust_parser.add_argument("output_model_csv")
    age_adjust_parser.add_argument("output_summary_json")

    error_parser = subparsers.add_parser("build-error-code-feature-table")
    error_parser.add_argument("transcript_root")
    error_parser.add_argument("output_csv")
    error_parser.add_argument("output_inventory_json")

    grouped_error_parser = subparsers.add_parser("build-grouped-error-feature-table")
    grouped_error_parser.add_argument("input_csv")
    grouped_error_parser.add_argument("output_csv")
    grouped_error_parser.add_argument("output_summary_json")

    grouped_error_merge_parser = subparsers.add_parser("merge-grouped-error-features-into-master")
    grouped_error_merge_parser.add_argument("master_csv")
    grouped_error_merge_parser.add_argument("grouped_csv")
    grouped_error_merge_parser.add_argument("--output-csv", default=None)

    error_rate_parser = subparsers.add_parser("add-error-rate-features-to-master")
    error_rate_parser.add_argument("input_csv")
    error_rate_parser.add_argument("--output-csv", default=None)

    past_verb_parser = subparsers.add_parser("build-past-verb-feature-table")
    past_verb_parser.add_argument("transcript_root")
    past_verb_parser.add_argument("error_code_csv")
    past_verb_parser.add_argument("output_csv")
    past_verb_parser.add_argument("output_summary_json")

    past_verb_merge_parser = subparsers.add_parser("merge-past-verb-features-into-master")
    past_verb_merge_parser.add_argument("master_csv")
    past_verb_merge_parser.add_argument("past_feature_csv")
    past_verb_merge_parser.add_argument("--output-csv", default=None)

    severity_profile_parser = subparsers.add_parser("build-severity-profile-table")
    severity_profile_parser.add_argument("input_csv")
    severity_profile_parser.add_argument("output_csv")
    severity_profile_parser.add_argument("output_summary_json")

    severity_band_parser = subparsers.add_parser("build-severity-bands")
    severity_band_parser.add_argument("input_csv")
    severity_band_parser.add_argument("output_csv")
    severity_band_parser.add_argument("output_summary_json")

    story_design_parser = subparsers.add_parser("build-story-generation-design")
    story_design_parser.add_argument("severity_banded_csv")
    story_design_parser.add_argument("counterbalance_csv")
    story_design_parser.add_argument("transcript_root")
    story_design_parser.add_argument("output_dir")
    story_design_parser.add_argument("--min-prompt-candidates", type=int, default=2)
    story_design_parser.add_argument("--min-eval-candidates", type=int, default=3)

    architecture_loocv_parser = subparsers.add_parser("build-architecture-loocv-plan")
    architecture_loocv_parser.add_argument("dev_measures_csv")
    architecture_loocv_parser.add_argument("severity_csv")
    architecture_loocv_parser.add_argument("transcript_root")
    architecture_loocv_parser.add_argument("reference_manifest_csv")
    architecture_loocv_parser.add_argument("output_dir")
    architecture_loocv_parser.add_argument("--cohort", default="4-year-old")

    frozen_roster_parser = subparsers.add_parser("build-frozen-roster")
    frozen_roster_parser.add_argument("dev_measures_csv")
    frozen_roster_parser.add_argument("severity_profile_csv")
    frozen_roster_parser.add_argument("output_csv")
    frozen_roster_parser.add_argument("--age-min-months", type=int, default=48)
    frozen_roster_parser.add_argument("--age-max-months", type=int, default=59)

    generated_eval_parser = subparsers.add_parser("build-generated-bundle-evaluation")
    generated_eval_parser.add_argument("generated_severity_csv")
    generated_eval_parser.add_argument("real_severity_csv")
    generated_eval_parser.add_argument("frozen_roster_csv")
    generated_eval_parser.add_argument("output_dir")
    generated_eval_parser.add_argument("--pairs-per-round", type=int, default=3)

    generated_text_eval_parser = subparsers.add_parser("build-generated-text-evaluation")
    generated_text_eval_parser.add_argument("synthetic_root")
    generated_text_eval_parser.add_argument("real_root")
    generated_text_eval_parser.add_argument("output_dir")
    generated_text_eval_parser.add_argument("--age-years", type=int, default=4)

    bundled_td_ppl_parser = subparsers.add_parser("build-bundled-td-text-perplexity")
    bundled_td_ppl_parser.add_argument("synthetic_root")
    bundled_td_ppl_parser.add_argument("real_root")
    bundled_td_ppl_parser.add_argument("output_dir")
    bundled_td_ppl_parser.add_argument("--age-years", type=int, default=4)
    bundled_td_ppl_parser.add_argument("--folds", type=int, default=5)
    bundled_td_ppl_parser.add_argument("--order", type=int, default=3)
    bundled_td_ppl_parser.add_argument("--alpha", type=float, default=0.5)
    bundled_td_ppl_parser.add_argument("--random-seed", type=int, default=0)
    bundled_td_ppl_parser.add_argument("--exclude-synthetic-source-children-from-real", action="store_true")

    bundled_ppl_parser = subparsers.add_parser("build-bundled-text-perplexity")
    bundled_ppl_parser.add_argument("synthetic_root")
    bundled_ppl_parser.add_argument("real_root")
    bundled_ppl_parser.add_argument("output_dir")
    bundled_ppl_parser.add_argument("--group", choices=["TD", "SLI"], required=True)
    bundled_ppl_parser.add_argument("--age-years", type=int, default=4)
    bundled_ppl_parser.add_argument("--folds", type=int, default=5)
    bundled_ppl_parser.add_argument("--order", type=int, default=3)
    bundled_ppl_parser.add_argument("--alpha", type=float, default=0.5)
    bundled_ppl_parser.add_argument("--random-seed", type=int, default=0)
    bundled_ppl_parser.add_argument("--exclude-synthetic-source-children-from-real", action="store_true")

    bundled_semantic_parser = subparsers.add_parser("build-bundled-text-semantic-evaluation")
    bundled_semantic_parser.add_argument("synthetic_root")
    bundled_semantic_parser.add_argument("real_root")
    bundled_semantic_parser.add_argument("output_dir")
    bundled_semantic_parser.add_argument("--group", choices=["TD", "SLI"], required=True)
    bundled_semantic_parser.add_argument("--age-years", type=int, default=4)
    bundled_semantic_parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    bundled_semantic_parser.add_argument("--batch-size", type=int, default=16)
    bundled_semantic_parser.add_argument("--bootstrap-reps", type=int, default=10000)
    bundled_semantic_parser.add_argument("--bootstrap-seed", type=int, default=0)
    bundled_semantic_parser.add_argument("--exclude-synthetic-source-children-from-real", action="store_true")

    panel_extract_parser = subparsers.add_parser("extract-story-panels")
    panel_extract_parser.add_argument("input_pdf")
    panel_extract_parser.add_argument("output_dir")
    panel_extract_parser.add_argument("--story-id", default=None)
    panel_extract_parser.add_argument("--rotate-degrees", type=int, default=180)

    story_packet_parser = subparsers.add_parser("build-story-packet-template")
    story_packet_parser.add_argument("panel_dir")
    story_packet_parser.add_argument("output_json")
    story_packet_parser.add_argument("--story-id", default=None)
    story_packet_parser.add_argument("--source-pdf", default=None)

    bootstrap_story_packets_parser = subparsers.add_parser("bootstrap-story-packets")
    bootstrap_story_packets_parser.add_argument("input_dir")
    bootstrap_story_packets_parser.add_argument("output_dir")
    bootstrap_story_packets_parser.add_argument("--rotate-degrees", type=int, default=180)
    bootstrap_story_packets_parser.add_argument("--overwrite-templates", action="store_true")

    story_unit_extract_parser = subparsers.add_parser("extract-story-unit-narratives")
    story_unit_extract_parser.add_argument("transcript_root")
    story_unit_extract_parser.add_argument("output_csv")

    story_unit_ref_parser = subparsers.add_parser("build-story-unit-reference-template")
    story_unit_ref_parser.add_argument("transcript_root")
    story_unit_ref_parser.add_argument("output_json")

    story_unit_score_parser = subparsers.add_parser("score-story-unit-semrp")
    story_unit_score_parser.add_argument("narrative_csv")
    story_unit_score_parser.add_argument("reference_json")
    story_unit_score_parser.add_argument("semrp_repo")
    story_unit_score_parser.add_argument("output_csv")
    story_unit_score_parser.add_argument("--k", type=int, default=3)
    story_unit_score_parser.add_argument("--model-name-or-path", default="uclanlp/keyphrase-mpnet-v1")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate-layout":
        return _cmd_validate_layout()
    if args.command == "show-layout":
        return _cmd_show_layout()
    if args.command == "show-counterbalance":
        return _cmd_show_counterbalance()
    if args.command == "show-manifest":
        return _cmd_show_manifest(args.limit)
    if args.command == "run-batchalign-morphotag":
        return _cmd_run_batchalign_morphotag(args)
    if args.command == "run-batchalign-morphotag-batched":
        return _cmd_run_batchalign_morphotag_batched(args)
    if args.command == "write-cha-filelist":
        return _cmd_write_cha_filelist(args)
    if args.command == "repair-batchalign-alignment":
        return _cmd_repair_batchalign_alignment(args)
    if args.command == "normalize-initial-pauses":
        return _cmd_normalize_initial_pauses(args)
    if args.command == "normalize-quote-terminators":
        return _cmd_normalize_quote_terminators(args)
    if args.command == "promote-batchalign-tiers":
        return _cmd_promote_batchalign_tiers(args)
    if args.command == "strip-legacy-mor-gra":
        return _cmd_strip_legacy_tiers(args)
    if args.command == "build-dss-feature-table":
        return _cmd_build_dss_feature_table(args)
    if args.command == "merge-dss-into-master":
        return _cmd_merge_dss_into_master(args)
    if args.command == "audit-feature-redundancy":
        return _cmd_audit_feature_redundancy(args)
    if args.command == "build-severity-feature-table":
        return _cmd_build_severity_feature_table(args)
    if args.command == "build-age-adjusted-severity-table":
        return _cmd_build_age_adjusted_severity_table(args)
    if args.command == "build-error-code-feature-table":
        return _cmd_build_error_code_feature_table(args)
    if args.command == "build-grouped-error-feature-table":
        return _cmd_build_grouped_error_feature_table(args)
    if args.command == "merge-grouped-error-features-into-master":
        return _cmd_merge_grouped_error_features_into_master(args)
    if args.command == "add-error-rate-features-to-master":
        return _cmd_add_error_rate_features_to_master(args)
    if args.command == "build-past-verb-feature-table":
        return _cmd_build_past_verb_feature_table(args)
    if args.command == "merge-past-verb-features-into-master":
        return _cmd_merge_past_verb_features_into_master(args)
    if args.command == "build-severity-profile-table":
        return _cmd_build_severity_profile_table(args)
    if args.command == "build-severity-bands":
        return _cmd_build_severity_bands(args)
    if args.command == "build-story-generation-design":
        return _cmd_build_story_generation_design(args)
    if args.command == "build-architecture-loocv-plan":
        return _cmd_build_architecture_loocv_plan(args)
    if args.command == "build-frozen-roster":
        return _cmd_build_frozen_roster(args)
    if args.command == "build-generated-bundle-evaluation":
        return _cmd_build_generated_bundle_evaluation(args)
    if args.command == "build-generated-text-evaluation":
        return _cmd_build_generated_text_evaluation(args)
    if args.command == "build-bundled-td-text-perplexity":
        return _cmd_build_bundled_td_text_perplexity(args)
    if args.command == "build-bundled-text-perplexity":
        return _cmd_build_bundled_text_perplexity(args)
    if args.command == "build-bundled-text-semantic-evaluation":
        return _cmd_build_bundled_text_semantic_evaluation(args)
    if args.command == "extract-story-panels":
        return _cmd_extract_story_panels(args)
    if args.command == "build-story-packet-template":
        return _cmd_build_story_packet_template(args)
    if args.command == "bootstrap-story-packets":
        return _cmd_bootstrap_story_packets(args)
    if args.command == "extract-story-unit-narratives":
        return _cmd_extract_story_unit_narratives(args)
    if args.command == "build-story-unit-reference-template":
        return _cmd_build_story_unit_reference_template(args)
    if args.command == "score-story-unit-semrp":
        return _cmd_score_story_unit_semrp(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
