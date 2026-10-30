"""Lean storage must preserve auditability and never prune failed/changed runs."""
import json
from pathlib import Path
import shutil
import sys
import zipfile

import pytest

from icl_pilot.chat import annotation_preparation as prep, corpus_archives as archives
from icl_pilot.chat.annotation import annotate_chat_directory, load_annotation_config, sha256_file
from icl_pilot.chat import morphosyntax as ud, retention
from icl_pilot.chat.preservation import text_fingerprint, require_text_preserved
from icl_pilot.pipeline import (begin_pipeline, sync_run, record_stage, package_run,
                               annotation_inputs, annotation_collection_metadata)
from test_annotation_pipeline import fake_tool, fake_validator, notebook_cell
from test_annotation_preparation import CHAT
from test_morphosyntax import fake_worker

ROOT = Path(__file__).resolve().parents[1]


def archive_config(tmp_path, monkeypatch, text):
    root = tmp_path / 'originals'
    paths = {}
    for name in ['ENNI', 'Gillam']:
        path = root / 'data/raw' / name.lower() / '2026-10-28' / (name + '.zip')
        path.parent.mkdir(parents=True)
        with zipfile.ZipFile(path, 'w') as bundle:
            bundle.writestr('TD/sample.cha', text)
            bundle.writestr('0metadata.cdc', 'corpus metadata')
        paths[name] = path
    config = root / 'retention.json'
    config.write_text(json.dumps(dict(policy='validated-annotation-retention-v1', preserve_failed_runs=True,
        archives={name:dict(path=path.relative_to(root).as_posix(),sha256=sha256_file(path)) for name,path in paths.items()})))
    monkeypatch.setattr(archives, 'DEFAULT_CONFIG', config)
    return root, config, paths


def completed_run(tmp_path, monkeypatch, *, ud_enabled=True):
    fake_tool(monkeypatch)
    original = CHAT.replace('@Begin\n', '@Begin\n@PID:\t11312/fixture\n')
    root, config, paths = archive_config(tmp_path, monkeypatch, original)
    inputs = {}
    for label in ['real', 'synthetic']:
        folder = tmp_path / 'inputs' / label
        folder.mkdir(parents=True)
        (folder / 'sample.cha').write_text(original)
        inputs[label] = folder
    run = begin_pipeline(tmp_path / 'run', mode='from_chat', csvs={}, chat_dirs=inputs)
    metadata = {'real':dict(source_type='real',corpus='ENNI'), 'synthetic':dict(source_type='synthetic')}
    prepared = prep.prepare_annotation_collections(inputs, run/'annotation_preparation', collection_metadata=metadata, validator={'fixture':True})
    record = prep.require_prepared_annotations(prep.preparation_reference(prepared))
    selected = {label:Path(item['prepared']) for label,item in record['collections'].items()}
    settings = load_annotation_config(ROOT/'configs/annotation/talktag.json')
    talk = annotate_chat_directory(selected, run/'annotation', config=settings, python=sys.executable,
        runtime={'source_commit':settings['revision']},validator={'fixture':True},preparation_reference=prep.preparation_reference(prepared))
    morph = None
    if ud_enabled:
        fake_worker(monkeypatch)
        selected, _, _ = prep.checked_talktag_collections(prep.preparation_reference(talk))
        morph = ud.annotate_morphosyntax(selected,run/'morphosyntax',python=sys.executable,
            model_dir=tmp_path/'models',validator={'fixture':True},upstream_annotation=prep.preparation_reference(talk))
    kwargs = dict(annotation_manifest=talk,morphosyntax_manifest=morph,project_root=root,config_path=config)
    return run, kwargs, inputs, paths


@pytest.mark.parametrize('replacement', ['she go', 'he goes', 'He go', 'he go go', 'he go .\n*CHI:\t'])
def test_fingerprints_detect_changes_to_words_case_repetition_and_boundaries(replacement):
    before = CHAT
    with pytest.raises(ValueError,match='Protected CHAT'):
        require_text_preserved(text_fingerprint(before),before.replace('he go',replacement))


def test_fingerprint_allows_documented_inline_additions_and_fresh_ud_only():
    before = CHAT
    after = before.replace('home .','home [* test] .')
    require_text_preserved(text_fingerprint(before),after)
    with pytest.raises(ValueError):
        require_text_preserved(text_fingerprint(before,'exact'),after)
    for changed in [before.replace('eng|test','eng|other'),before.replace('*CHI:', '*INV:')]:
        with pytest.raises(ValueError):require_text_preserved(text_fingerprint(before),changed)


@pytest.mark.parametrize('ud_enabled', [True,False])
def test_complete_run_keeps_final_files_audits_and_one_zip_reference(tmp_path,monkeypatch,ud_enabled):
    run, kwargs, inputs, paths = completed_run(tmp_path,monkeypatch,ud_enabled=ud_enabled)
    before = {label:(folder/'sample.cha').read_bytes() for label,folder in inputs.items()}
    reference = retention.finalize_annotation_run(run,**kwargs)
    record = retention.verify_retained_annotations(reference)
    assert record['status']=='complete' and len(record['outputs'])==2
    assert len(record['original_members'])==1
    assert record['original_members'][0]['member']=='TD/sample.cha'
    assert record['retained_audits'] and all((run/item['path']).is_file() for item in record['retained_audits'])
    assert all(not (run/item['path']).exists() for item in record['removed'])
    assert all((folder/'sample.cha').read_bytes()==before[label] for label,folder in inputs.items())
    assert all(path.is_file() for path in paths.values())
    assert not list(run.rglob('*.zip'))  # References, not additional corpus ZIP copies.
    assert len(list(run.rglob('*.cha')))==2
    assert retention.finalize_annotation_run(run,**kwargs)==reference
    # Verification survives a moved/unpacked run and does not require originals.
    relocated = tmp_path/'relocated';shutil.copytree(run,relocated)
    shutil.rmtree(kwargs['project_root'])
    retention.verify_retained_annotations(reference,run_root=relocated)
    (relocated/record['outputs'][0]['path']).write_text('changed')
    with pytest.raises(ValueError):retention.verify_retained_annotations(reference,run_root=relocated)


@pytest.mark.parametrize('failure',['source','raw','output','zip','status'])
def test_failure_preserves_all_intermediates(tmp_path,monkeypatch,failure):
    run, kwargs, inputs, paths = completed_run(tmp_path,monkeypatch)
    if failure=='source':(inputs['synthetic']/'sample.cha').write_text('changed')
    elif failure=='raw':next((run/'morphosyntax/raw_batchalign').rglob('*.cha')).write_text('changed')
    elif failure=='output':next((run/'morphosyntax/annotated').rglob('*.cha')).write_text('changed')
    elif failure=='zip':paths['ENNI'].write_text('wrong archive')
    else:
        path=kwargs['morphosyntax_manifest'];record=json.loads(path.read_text());record['status']='failed_or_interrupted';path.write_text(json.dumps(record))
    before = {p.relative_to(run).as_posix():p.read_bytes() for p in run.rglob('*') if p.is_file()}
    with pytest.raises(ValueError):retention.finalize_annotation_run(run,**kwargs)
    assert before=={p.relative_to(run).as_posix():p.read_bytes() for p in run.rglob('*') if p.is_file()}


def test_interrupted_cleanup_resumes_and_drive_drops_only_verified_intermediates(tmp_path,monkeypatch):
    run, kwargs, _, _ = completed_run(tmp_path,monkeypatch)
    drive=tmp_path/'drive';sync_run(run,drive)
    actual_prune=retention._prune
    def interrupt(root,record):
        (root/record['removed'][0]['path']).unlink()
        raise InterruptedError('fixture interrupt')
    monkeypatch.setattr(retention,'_prune',interrupt)
    with pytest.raises(InterruptedError):retention.finalize_annotation_run(run,**kwargs)
    assert json.loads((run/retention.RECEIPT).read_text())['status']=='pruning'
    monkeypatch.setattr(retention,'_prune',actual_prune)
    reference=retention.finalize_annotation_run(run,**kwargs)
    record_stage(run,'annotation_retention','complete',**reference)
    record=retention.verify_retained_annotations(reference)
    changed=drive/record['removed'][0]['path'];changed.write_text('preserve unexpected edited copy')
    with pytest.raises(ValueError,match='preserved'):sync_run(run,drive)
    assert changed.read_text()=='preserve unexpected edited copy'
    changed.write_bytes(b'')
    changed.unlink()
    (drive/'user_notes.txt').write_text('keep')
    sync_run(run,drive)
    retention.verify_retained_annotations(reference,run_root=drive)
    assert all(not (drive/item['path']).exists() for item in record['removed'])
    assert (drive/'user_notes.txt').read_text()=='keep'
    archive=package_run(run)
    with zipfile.ZipFile(archive) as bundle:
        assert not set(item['path'] for item in record['removed']) & set(bundle.namelist())


def test_original_zip_recovery_and_compaction_use_exact_verified_bytes(tmp_path,monkeypatch):
    original=CHAT.replace('@Begin\n','@Begin\n@PID:\t11312/fixture\n')
    root,config,paths=archive_config(tmp_path,monkeypatch,original)
    record=archives.verified_archive('ENNI',project_root=root,config_path=config)
    member=archives.archive_member_for_pid(record,'11312/fixture')
    target=tmp_path/'recovered.cha'
    archives.restore_original_member(record,member,target)
    assert target.read_text()==original
    with pytest.raises(FileExistsError):archives.restore_original_member(record,member,target)
    for path in paths.values():
        with zipfile.ZipFile(path) as bundle:bundle.extractall(path.parent/'transcripts')
    unexpected=paths['ENNI'].parent/'transcripts/TD/sample.cha'
    unexpected.write_text('unexpected edits')
    with pytest.raises(ValueError):archives.compact_extracted_originals(root,config_path=config)
    assert all((path.parent/'transcripts').is_dir() for path in paths.values())
    unexpected.write_text(original)
    receipt=archives.compact_extracted_originals(root,config_path=config)
    assert json.loads(receipt.read_text())['status']=='complete'
    assert all(path.is_file() and not (path.parent/'transcripts').exists() for path in paths.values())
    assert archives.original_source_bytes(unexpected,root).decode()==original
    assert archives.original_member_paths(unexpected.parent.parent,root,'ENNI')==[unexpected]
    assert archives.compact_extracted_originals(root,config_path=config)==receipt


def test_notebook_finalization_uses_saved_stage_paths(tmp_path,monkeypatch):
    run, kwargs, _, _ = completed_run(tmp_path,monkeypatch)
    for stage in ['annotation_preparation','chat_validation_after']:record_stage(run,stage,'complete')
    record_stage(run,'annotation','complete',manifests=[str(kwargs['annotation_manifest'])])
    record_stage(run,'morphosyntax','complete',manifest=str(kwargs['morphosyntax_manifest']))
    ns=dict(LEAN_ANNOTATION=True,RUN_ANNOTATION=True,RUN_MORPHOSYNTAX=True,RUN_OUTPUT_DIR=run,
        REPO_DIR=kwargs['project_root'],RETENTION_CONFIG_PATH=kwargs['config_path'],record_stage=record_stage,
        persist_run=lambda:None,json=json)
    exec(notebook_cell('annotation-retention'),ns)
    exec(notebook_cell('annotation-retention'),ns)
    assert json.loads((run/retention.RECEIPT).read_text())['status']=='complete'


def test_partial_expansion_uses_complete_zip_inventory_and_rejects_changed_members(tmp_path,monkeypatch):
    root,config,paths=archive_config(tmp_path,monkeypatch,CHAT)
    path=paths['Gillam']
    with zipfile.ZipFile(path,'a') as bundle:bundle.writestr('TD/second.cha',CHAT)
    settings=json.loads(config.read_text());settings['archives']['Gillam']['sha256']=sha256_file(path)
    config.write_text(json.dumps(settings))
    folder=path.parent/'transcripts';(folder/'TD').mkdir(parents=True)
    (folder/'TD/sample.cha').write_text(CHAT)
    assert len(archives.original_member_paths(folder,root,'Gillam'))==2
    assert archives.original_source_bytes(folder/'TD/second.cha',root).decode()==CHAT
    (folder/'TD/sample.cha').write_text('changed')
    with pytest.raises(ValueError,match='differs'):archives.original_source_bytes(folder/'TD/sample.cha',root)
    (folder/'TD/unexpected.cha').write_text(CHAT)
    with pytest.raises(ValueError,match='Unexpected'):archives.original_member_paths(folder,root,'Gillam')


def test_notebook_requires_real_recovery_zip_before_preparing_or_loading_models(tmp_path,monkeypatch):
    fake_validator(monkeypatch)
    root,config,paths=archive_config(tmp_path,monkeypatch,CHAT)
    source=tmp_path/'input';source.mkdir();(source/'sample.cha').write_text(CHAT)
    run=begin_pipeline(tmp_path/'run',mode='from_chat',csvs={},chat_dirs={'real':source})
    ns=dict(RUN_ANNOTATION=True,MODE='from_chat',RUN_CHAT_CONVERSION=False,RUN_GENERATION=False,
        INSTALL_DEPENDENCIES=False,LEAN_ANNOTATION=True,REPO_DIR=root,RETENTION_CONFIG_PATH=config,
        PREPARATION_CONFIG_PATH=ROOT/'configs/annotation/preparation.json',RUN_OUTPUT_DIR=run,
        chat_roots={'real':source},ANNOTATION_STAGES=['stage1_td','stage2_sli'],
        CHAT_INPUT_METADATA={'real':dict(source_type='real',corpus='ENNI')},
        annotation_inputs=annotation_inputs,annotation_collection_metadata=annotation_collection_metadata,
        record_stage=record_stage,persist_run=lambda:None,json=json,Path=Path)
    original_zip=paths['ENNI'].read_bytes();paths['ENNI'].unlink()
    with pytest.raises(ValueError,match='ZIP is missing'):exec(notebook_cell('annotation-preparation'),ns)
    assert not (run/'annotation_preparation').exists()
    assert (source/'sample.cha').read_text()==CHAT
    paths['ENNI'].write_bytes(original_zip)
    exec(notebook_cell('annotation-preparation'),ns)
    assert ns['preparation_record']['status']=='prepared'
    assert set(json.loads((run/'original_corpus_archives.json').read_text())['archives'])=={'ENNI','Gillam'}
