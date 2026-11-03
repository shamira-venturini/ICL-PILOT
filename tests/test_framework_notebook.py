"""Main-notebook evaluation integration, without model downloads or GPU fits."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pandas as pd
import pytest

from icl_pilot.evaluation import notebook as workflow
from icl_pilot.evaluation.notebook_runtime import patch_toolkit, SYNTHTEXTEVAL_REVISION
from test_framework_revision import fixture_study, fixture_backend

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def session_inputs(fixture_study, tmp_path):
    records, external, protocol = fixture_study
    root = tmp_path / "project"
    config = root / "configs/generation"
    config.mkdir(parents=True)
    evaluation = root / "configs/evaluation"
    evaluation.mkdir()
    shutil.copytree(ROOT / "src/icl_pilot/evaluation", root / "src/icl_pilot/evaluation",
                    ignore=shutil.ignore_patterns('__pycache__'))
    protocol = copy.deepcopy(protocol)
    protocol['generator_catalog'] = 'configs/generation/models.json'
    (evaluation / "protocol.json").write_text(json.dumps(protocol))
    (config / 'models.json').write_text(json.dumps({'models': protocol['generator_identities']}))
    real = records.loc[records.source.eq('real')].copy()
    children = real[['source_child_id', 'group']].drop_duplicates().rename(columns={'source_child_id': 'File_ID', 'group': 'Group'})
    children['Age(Month)'] = 48
    children.to_csv(config / 'participants.csv', index=False)
    stories = real.rename(columns={'source_child_id': 'File_ID', 'group': 'Group'})
    stories['Age_Raw'] = '4;00.00'
    stories['transcript_path'] = 'fixture.cha'
    stories.to_csv(config / 'stories.csv', index=False)
    syn = records.loc[records.source.eq('synthetic')]
    roster = syn[['pair_id', 'td_child_id', 'sli_child_id']].drop_duplicates()
    roster['cohort'] = '4-year-old'
    roster.to_csv(config / 'roster.csv', index=False)
    (config / 'experiment.json').write_text(json.dumps({
        'roster_csv': 'roster.csv', 'n_replicates': 2,
        'provenance': {'participant_manifest': 'configs/generation/participants.csv'},
        'inputs': {'narratives_csv': 'configs/generation/stories.csv'},
    }))
    sources = tmp_path / 'saved outputs outside checkout'
    sources.mkdir()
    records.to_csv(sources / 'combined.csv', index=False)
    external.to_csv(sources / 'gillam.csv', index=False)
    generated = {}
    for model, rows in syn.groupby('model_key'):
        wide = []
        for _, job in rows.groupby(['pair_id', 'replicate_id', 'story_id']):
            first = job.loc[job.stage.eq('stage1_td')].iloc[0].to_dict()
            first.update(cohort_label='4-year-old', target_story=first['story_id'],
                bundle_id=first['pair_id'] + str(first['replicate_id']),
                stage1_td_story=first['story_text'],
                stage2_sli_story=job.loc[job.stage.eq('stage2_sli'), 'story_text'].iloc[0])
            wide.append(first)
        path = sources / (model + '.csv')
        pd.DataFrame(wide).to_csv(path, index=False)
        generated[model] = [path]
    return root, sources, generated


def start(session_inputs, **kwargs):
    root, sources, generated = session_inputs
    selected = kwargs.pop('generation_csvs', None)
    return workflow.create_session(root, gillam_csv=sources / 'gillam.csv',
        text_records_csv=None if selected else sources / 'combined.csv',
        generation_csvs=selected, ratios=(1, 2), **kwargs)


def test_both_input_routes_produce_same_reference_splits_and_training_plan(session_inputs):
    first = start(session_inputs)
    second = start(session_inputs, generation_csvs=session_inputs[2])
    a, b = workflow.prepare_session(first), workflow.prepare_session(second)
    assert a.real['sample_id'].tolist() == b.real['sample_id'].tolist()
    assert a.real['heldout_fold'].tolist() == b.real['heldout_fold'].tolist()
    assert len(a.synthetic) == len(b.synthetic)
    assert set(b.synthetic['model_key']) == {'alpha', 'beta'}
    workflow.run_action(first, 'inputs')
    assert (first.parent / 'classification/training_plan.json').is_file()
    assert not list(first.parent.rglob('completed.json'))
    workflow.run_action(first, 'descriptive')
    assert (first.parent / 'lexical_overlap_by_generator.csv').is_file()


def test_worker_process_runs_only_requested_saved_text_blocks(session_inputs):
    session = start(session_inputs)
    python = os.environ.get('ICL_FRAMEWORK_TEST_PYTHON', sys.executable)
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONDONTWRITEBYTECODE='1')
    for action in ('inputs', 'descriptive'):
        subprocess.run([python, '-m', 'icl_pilot.evaluation.notebook', '--session', str(session), '--action', action],
                       env=env, check=True, capture_output=True, text=True)
    assert (session.parent / 'lexical_overlap_by_generator.csv').is_file()
    assert not list(session.parent.rglob('all_story_predictions.csv'))


def test_missing_and_duplicate_inputs_stop_before_creating_results(session_inputs):
    root, sources, generated = session_inputs
    with pytest.raises(FileNotFoundError):
        workflow.create_session(root, text_records_csv=sources / 'missing.csv', gillam_csv=sources / 'gillam.csv')
    with pytest.raises(ValueError, match='Duplicate generation'):
        start(session_inputs, generation_csvs={'alpha': [generated['alpha'][0]] * 2, 'beta': generated['beta']})
    assert not (root / 'results').exists()


def test_explicit_persistent_output_root_receives_incremental_results(session_inputs, tmp_path):
    persistent = tmp_path / 'mounted Drive'
    session = start(session_inputs, output_root=persistent)
    assert session.parent.parent == persistent
    workflow.run_action(session, 'inputs')
    assert (session.parent / 'classification/training_plan.csv').is_file()
    assert not (session_inputs[0] / 'results').exists()


@pytest.mark.parametrize('problem', ['smoke', 'overlap', 'incomplete', 'reference', 'meta'])
def test_input_guards_and_review_only_flags(session_inputs, problem):
    root, sources, generated = session_inputs
    records_path = sources / 'combined.csv'
    frame = pd.read_csv(records_path)
    if problem == 'smoke':
        frame.loc[frame.source.eq('synthetic'), 'test_only'] = True
    elif problem == 'overlap':
        frame = pd.concat([frame, frame.loc[frame.source.eq('synthetic')].iloc[:1]])
    elif problem == 'incomplete':
        frame = frame.loc[~(frame.source.eq('synthetic') & frame.replicate_id.eq(2))]
    elif problem == 'reference':
        frame.loc[frame.source.eq('real'), 'story_text'] = 'changed text'
    else:
        frame.loc[frame.source.eq('synthetic'), 'story_text'] = 'I replaced the main characters to match the target story'
    frame.to_csv(records_path, index=False)
    session = start(session_inputs)
    if problem == 'meta':
        study = workflow.prepare_session(session)
        assert len(study.synthetic) == len(frame.loc[frame.source.eq('synthetic')])
        assert len(pd.read_csv(session.parent / 'review_only_meta_generations.csv')) == len(study.synthetic)
    else:
        with pytest.raises(ValueError):
            workflow.prepare_session(session)


def test_resume_preserves_inputs_and_rejects_changed_sources(session_inputs):
    session = start(session_inputs)
    workflow.prepare_session(session)
    original = session.read_bytes()
    assert workflow.create_session(session_inputs[0], resume_dir=session.parent) == session
    assert session.read_bytes() == original
    source = session_inputs[1] / 'combined.csv'
    source.write_text(source.read_text() + '\n')
    with pytest.raises(ValueError, match='changed'):
        workflow.create_session(session_inputs[0], resume_dir=session.parent)


def test_relocated_export_resumes_with_verified_original_sources(session_inputs, tmp_path):
    session = start(session_inputs)
    workflow.run_action(session, 'inputs')
    relocated = tmp_path / 'restored evaluation'
    shutil.copytree(session.parent, relocated)
    resumed = workflow.create_session(session_inputs[0], resume_dir=relocated)
    assert json.loads(resumed.read_text())['output_dir'] == str(relocated)
    assert (relocated / 'session_relocations.json').is_file()
    workflow.run_action(resumed, 'descriptive')
    assert (relocated / 'lexical_overlap_by_generator.csv').is_file()
    assert not (session.parent / 'lexical_overlap_by_generator.csv').exists()


def test_fitting_summaries_resume_and_compact_export(session_inputs):
    session = start(session_inputs)
    calls = []
    def backend(*args):
        calls.append(args[0]['classifier_run_id'])
        return fixture_backend(*args)
    workflow.run_action(session, 'inputs')
    with pytest.raises(RuntimeError, match='block first'):
        workflow.run_action(session, 'summary')
    workflow.run_action(session, 'classifiers', backend=backend)
    count = len(calls)
    workflow.run_action(session, 'classifiers', backend=backend)
    assert len(calls) == count
    workflow.run_action(session, 'summary')
    workflow.run_action(session, 'ratio-plan')
    assert len(calls) == count
    workflow.run_action(session, 'ratios', backend=backend)
    assert len(calls) > count
    workflow.run_action(session, 'ratio-summary')
    assert (session.parent / 'ratio_experiment/child_balanced_accuracy_by_ratio.png').is_file()
    checkpoint = session.parent / 'classification/model/large.bin'
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b'preserved model')
    archive = workflow.export_session(session)
    with zipfile.ZipFile(archive) as zipped:
        inventory = json.loads(zipped.read('portable_file_manifest.json'))
        assert 'notebook_session.json' in inventory
        assert 'classification/all_story_predictions.csv' in inventory
        assert not any('/model/' in name for name in zipped.namelist())
        for name, expected in inventory.items():
            assert hashlib.sha256(zipped.read(name)).hexdigest() == expected
    assert checkpoint.read_bytes() == b'preserved model'
    predictions = session.parent / 'classification/all_story_predictions.csv'
    predictions.write_text(predictions.read_text() + '\n')
    with pytest.raises(ValueError, match='changed'):
        workflow.run_action(session, 'summary')


def test_toolkit_patch_accepts_only_exact_reviewed_source(tmp_path, monkeypatch):
    source = tmp_path / 'toolkit'
    utils = source / 'synthtexteval/eval/downstream/classify/utils.py'
    utils.parent.mkdir(parents=True)
    original = 'return tokenizer.encode_plus(text, pad_to_max_length=True)\n'
    utils.write_text(original)
    def output(command, **kwargs):
        return SYNTHTEXTEVAL_REVISION + '\n' if 'rev-parse' in command else original
    monkeypatch.setattr('icl_pilot.evaluation.notebook_runtime.subprocess.check_output', output)
    first = patch_toolkit(source)
    assert patch_toolkit(source) == first
    assert utils.read_text() == 'return tokenizer(text, padding="max_length")\n'
    utils.write_text('unreviewed edit')
    with pytest.raises(ValueError, match='Unreviewed'):
        patch_toolkit(source)
    assert utils.read_text() == 'unreviewed edit'


NOTEBOOKS = [ROOT / 'notebooks/colab_direct_generation_pipeline.ipynb',
             ROOT / 'ICL-PILOT-smoke-test-20261030/ICL-PILOT_smoke_test.ipynb']


@pytest.mark.parametrize('path', NOTEBOOKS)
def test_disabled_evaluation_cells_execute_without_inputs_installation_or_models(path, tmp_path):
    notebook = json.loads(path.read_text())
    namespace = dict(REPO_DIR=tmp_path, Path=Path, IS_COLAB=False)
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and cell.get('id', '').startswith('framework-'):
            source = ''.join(cell['source'])
            compile(source, cell['id'], 'exec')
            exec(source, namespace)
    assert namespace['RUN_FRAMEWORK_EVALUATION'] is False
    assert namespace['RUN_EVAL_TEXT_QUALITY'] is False
    assert namespace['RUN_EVAL_CLASSIFICATION'] is False
    assert namespace['RUN_EVAL_RATIOS'] is False
    assert namespace['framework_session'] is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('changed', [False, True])
def test_smoke_support_package_updates_only_recognised_sources(tmp_path, changed):
    notebook = json.loads(NOTEBOOKS[1].read_text())
    source = ''.join(next(c['source'] for c in notebook['cells'] if c.get('id') == 'framework-support'))
    root = tmp_path.resolve()
    with zipfile.ZipFile(ROOT / 'ICL-PILOT-smoke-test-20261030/ICL-PILOT_smoke_bundle.zip') as original:
        original.extractall(root)
    project = root / 'ICL-PILOT'
    path = project / 'src/icl_pilot/evaluation/story_text_exports.py'
    if changed:
        path.write_text('unreviewed local edit')
    before = path.read_bytes()
    namespace = dict(Path=Path, RUN_FRAMEWORK_EVALUATION=True, REPO_DIR=project,
        FRAMEWORK_SUPPORT_ZIP=ROOT / 'ICL-PILOT-smoke-test-20261030/ICL-PILOT_framework_support_20261102.zip')
    if changed:
        with pytest.raises(ValueError, match='preserved'):
            exec(source, namespace)
        assert path.read_bytes() == before
        assert not (project / 'src/icl_pilot/evaluation/notebook.py').exists()
    else:
        exec(source, namespace)
        assert path.read_bytes() == (ROOT / 'src/icl_pilot/evaluation/story_text_exports.py').read_bytes()
        assert (project / 'data/inputs/gillam/external_story_records.csv').is_file()
        exec(source, namespace)  # Repeated setup is idempotent.
