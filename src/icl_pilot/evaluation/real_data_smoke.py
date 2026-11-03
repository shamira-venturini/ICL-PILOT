"""Bounded framework execution on actual pilot/ENNI/Gillam text.

This diagnostic never admits pilot outputs to the scientific preparation route.
It uses one real fold, one classifier seed and one actual CPU update per fit.
Its measurements and predictions are software checks, not study estimates.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import zipfile

import pandas as pd

from .framework_data import (PreparedEvaluation, _identity, balanced_corpora,
                             child_id, digest, load_protocol)
from .framework_splits import assign_classifier_folds, classifier_partitions
from .framework_sampling import bootstrap_synthetic
from .notebook import _json, sha256, create_session, prepare_session, export_session, _ratio_plot
from .story_text_exports import _real_records, _generated_records

PILOT_SHA256 = "a54acbef2c7f5ed39905446eaedd014bd11a677a37aed54d71b5254b474b6dd0"


def prepare_diagnostic(root, archive, output):
    root, archive, output = Path(root).resolve(), Path(archive).resolve(), Path(output).resolve()
    protocol = load_protocol(root)
    experiment = json.loads((root / 'configs/generation/experiment.json').read_text())
    participants = root / experiment['provenance']['participant_manifest']
    stories = root / experiment['inputs']['narratives_csv']
    gillam = root / 'data/inputs/gillam/external_story_records.csv'
    paths = [archive, participants, stories, gillam, root / 'configs/evaluation/protocol.json',
             root / 'configs/generation/experiment.json', root / 'configs/generation/models.json',
             root / 'configs/generation' / experiment['roster_csv']]
    inputs = {str(path): sha256(path) for path in paths}
    implementation = {str(path): sha256(path) for path in sorted((root / 'src/icl_pilot/evaluation').glob('*.py'))}
    if inputs[str(archive)] != PILOT_SHA256:
        raise ValueError('Use the reviewed, unmodified Qwen pilot archive.')
    manifest_path = output / 'diagnostic_manifest.json'
    if output.exists():
        previous = json.loads(manifest_path.read_text())
        if previous['inputs'] != inputs or previous['implementation'] != implementation:
            raise ValueError('Diagnostic sources changed; preserve this run and select a fresh output directory.')
    else:
        output.mkdir(parents=True, exist_ok=False)
    _json(manifest_path, {'test_only': True, 'scientific_results': False, 'inputs': inputs,
        'implementation': implementation, 'archive': str(archive), 'scope': 'Actual-data bounded software execution',
        'limitations': 'One four-year-old pair, one generator, one seed, one CPU update per fit; no accuracy inference.'})
    with zipfile.ZipFile(archive) as zipped:
        member = 'all_cohorts_rotating_story_generations.csv'
        raw = zipped.read(member)
    csv = output / 'pilot_generations.csv'
    if csv.exists() and csv.read_bytes() != raw:
        raise ValueError('Saved pilot CSV changed; preserved.')
    csv.write_bytes(raw)
    real_rows, missing = _real_records(measures_path=participants, stories_path=stories,
        project_root=root, minimum_age_months=48, maximum_age_months=96)
    if missing:
        raise ValueError('Canonical ENNI reference is incomplete.')
    synthetic = pd.DataFrame(_generated_records(source_path=csv, cohort_year=None, project_root=root))
    if len(synthetic) != 24 or not synthetic['test_only'].astype(str).str.lower().eq('true').all():
        raise ValueError('Unexpected pilot coverage or missing test-only flags.')
    records = pd.concat([pd.DataFrame(real_rows), synthetic], ignore_index=True)
    combined = output / 'guard_input_records.csv'
    records.to_csv(combined, index=False)
    # Exercise the actual notebook entry point with the real records: rejection
    # must be specifically the smoke safeguard, after canonical real checks pass.
    guard = output / 'scientific_guard_check.json'
    if not guard.exists():
        session = create_session(root, text_records_csv=combined, gillam_csv=gillam,
                                 output_root=output / 'scientific_guard')
        try:
            prepare_session(session)
        except ValueError as exc:
            if 'Software-fixture generation cannot enter scientific evaluation' not in str(exc):
                raise
            _json(guard, {'passed': True, 'expected_rejection': str(exc), 'session': str(session)})
        else:
            raise AssertionError('Scientific preparation admitted pilot generations.')
    real = _identity(pd.DataFrame(real_rows), 'enni')
    external = _identity(pd.read_csv(gillam, dtype={'source_child_id': str}), 'gillam')
    assert len(real) == 1428 and real['child_key'].nunique() == 238
    assert len(external) == 543 and external['child_key'].nunique() == 182
    roster = pd.read_csv(root / 'configs/generation' / experiment['roster_csv'], dtype=str)
    children = []
    for cohort, rows in roster.groupby('cohort', sort=True):
        for index, row in enumerate(rows.sort_values('pair_order', key=lambda values: values.astype(int)).itertuples()):
            for group, field in [('TD', 'td_child_id'), ('SLI', 'sli_child_id')]:
                children.append({'cohort_year': int(cohort.split('-')[0]), 'group': group,
                    'source_child_id': child_id(getattr(row, field)), 'pair_id': row.pair_id,
                    'round_id': 'round_{:02d}'.format(index // experiment['pairs_per_round'] + 1)})
    real = assign_classifier_folds(real, pd.DataFrame(children), protocol)
    real.to_csv(output / 'full_real_fold_inventory.csv', index=False)
    external.to_csv(output / 'full_gillam_inventory.csv', index=False)
    for field in ['source_child_id', 'td_child_id', 'sli_child_id']:
        synthetic[field] = synthetic[field].map(child_id)
    ancestors = set(synthetic['td_child_id']) | set(synthetic['sli_child_id'])
    for fold, frames in classifier_partitions(real, protocol):
        if ancestors <= set(frames['train']['source_child_id']):
            break
    else:
        raise ValueError('Pilot ancestors are never in the same training fold.')
    frames = {name: rows.loc[rows['cohort_year'].eq(4)].copy().reset_index(drop=True)
              for name, rows in frames.items()}
    # Keep all real training children in this actual age/fold. For bounded
    # validation/test prediction retain one complete held-out pair per partition.
    for split in ('validation', 'test'):
        pair = sorted(set(frames[split].loc[frames[split]['is_prompt_child'], 'pair_id']))[0]
        frames[split] = frames[split].loc[frames[split]['pair_id'].eq(pair)].reset_index(drop=True)
    groups = {name: set(rows['child_key']) for name, rows in frames.items()}
    assert not (groups['train'] & groups['validation'] or groups['train'] & groups['test'] or groups['validation'] & groups['test'])
    for name, rows in frames.items():
        rows.to_csv(output / ('diagnostic_real_' + name + '.csv'), index=False)
    chosen = external.groupby('group')['child_key'].apply(lambda values: sorted(set(values))[0]).tolist()
    external = external.loc[external['child_key'].isin(chosen)].reset_index(drop=True)
    external.to_csv(output / 'diagnostic_gillam_test.csv', index=False)
    diagnostic = copy.deepcopy(protocol)
    diagnostic.pop('protocol_sha256')
    diagnostic.update(test_only=True, generator_keys=['qwen3_8b'], cohorts=[4], synthetic_to_real_ratio=1,
                      scientific_protocol_sha256=protocol['protocol_sha256'])
    diagnostic['generator_identities'] = {'qwen3_8b': protocol['generator_identities']['qwen3_8b']}
    diagnostic['classifier'].update(test_only=True, diagnostic_max_steps=1, seeds=[42], batch_size=2, eval_batch_size=4)
    diagnostic['uncertainty']['bootstrap_samples'] = 20
    diagnostic['protocol_sha256'] = digest(diagnostic)
    _json(output / 'diagnostic_protocol.json', diagnostic)
    synthetic['sample_id'] = ['synthetic:' + digest(row)[:24] for row in synthetic[['model_key','run_id','pair_id','replicate_id','story_id','stage']].to_dict('records')]
    synthetic['output_text'] = synthetic['story_text']
    synthetic['Label'] = synthetic['group'].map({'SLI': 0, 'TD': 1})
    pools = {'qwen3_8b': synthetic}
    sampled, membership = bootstrap_synthetic(pools, frames['train'], diagnostic, fold, 1)
    pd.DataFrame(membership).to_csv(output / 'diagnostic_synthetic_bootstrap.csv', index=False)
    reference = real.loc[real['cohort_year'].eq(4)].copy()
    samples = balanced_corpora(reference, synthetic, diagnostic, output)
    for sample in samples.values():
        sampling = sample['path'] / 'sampling_manifest.json'
        record = json.loads(sampling.read_text())
        record.update(test_only=True, scope='Bounded actual-pilot software check; no study inference')
        _json(sampling, record)
    prepared = PreparedEvaluation(diagnostic, reference, synthetic, external,
        {fold: {**frames, 'synthetic_pool': pools, 'synthetic': sampled}}, samples, output)
    _json(output / 'diagnostic_counts.json', {'full_enni_children': 238, 'full_enni_stories': 1428,
        'full_gillam_children': 182, 'full_gillam_stories': 543, 'pilot_stories': len(synthetic),
        'training_fold': fold, 'classifiers_real_partitions': {name: {'stories': len(rows), 'children': rows['child_key'].nunique(),
        'group_stories': rows['group'].value_counts().to_dict()} for name, rows in frames.items()},
        'external_test_stories': len(external), 'external_test_children': external['child_key'].nunique(),
        'corpus_rows_per_source': len(samples['qwen3_8b']['real']), 'prompt_ancestors': sorted(ancestors)})
    return prepared


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--action', required=True, choices=['inputs', 'descriptive', 'quality', 'classifiers', 'summary', 'ratio-plan', 'ratios', 'ratio-summary', 'export'])
    args = parser.parse_args()
    prepared = prepare_diagnostic(args.project_root, args.archive, args.output)
    output = prepared.output_dir
    from .framework_classifier import write_training_plan, run_classifiers, cpu_diagnostic_backend
    from .framework_ratios import prepare_ratio_experiment, run_ratio_experiment, summarize_ratio_experiment
    from .framework_metrics import summarize_predictions
    if args.action == 'inputs':
        write_training_plan(prepared, output / 'classification')
    elif args.action == 'descriptive':
        from .framework_descriptive import run_descriptive
        run_descriptive(prepared)
    elif args.action == 'quality':
        from .framework_quality import run_text_quality
        run_text_quality(prepared)
    elif args.action == 'classifiers':
        run_classifiers(prepared, output / 'classification', backend=cpu_diagnostic_backend)
    elif args.action == 'summary':
        predictions = pd.read_csv(output / 'classification/all_story_predictions.csv', dtype={'source_child_id': str})
        summarize_predictions(predictions, prepared.protocol, output / 'classification/summary')
    elif args.action in {'ratio-plan', 'ratios', 'ratio-summary'}:
        experiment = prepare_ratio_experiment(prepared, output / 'ratio_experiment', ratios=(1, 10))
        if args.action == 'ratios':
            run_ratio_experiment(experiment, backend=cpu_diagnostic_backend, reuse_from=(output / 'classification',))
        elif args.action == 'ratio-summary':
            predictions = pd.read_csv(output / 'ratio_experiment/all_ratio_story_predictions.csv', dtype={'source_child_id': str})
            result = summarize_ratio_experiment(predictions, prepared.protocol, output / 'ratio_experiment/summary')
            _ratio_plot(result['summary'], prepared.protocol, (1, 10), output / 'ratio_experiment')
    elif args.action == 'export':
        config = json.loads((output / 'diagnostic_manifest.json').read_text())
        session = output / 'notebook_session.json'
        _json(session, {**config, 'schema_version': 1, 'project_root': args.project_root,
                        'output_dir': str(output), 'test_only': True})
        print(export_session(session))
    receipts = output / 'diagnostic_blocks'
    receipts.mkdir(exist_ok=True)
    _json(receipts / (args.action + '.json'), {'test_only': True, 'status': 'passed', 'action': args.action})
    print('Completed actual-data diagnostic:', args.action, output)


if __name__ == '__main__':
    main()
