"""Shared real/synthetic preparation must delete judgments without editing speech."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

from icl_pilot.chat import annotation_preparation as prep, validation
from icl_pilot.chat.annotation import annotate_chat_directory, load_annotation_config
from icl_pilot.pipeline import annotation_inputs, annotation_collection_metadata, stage_flags
from test_annotation_pipeline import fake_tool, fake_validator

ROOT = Path(__file__).resolve().parents[1]
HEADER = ("@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n"
          "@ID:\teng|test|CHI|4;01.00||TD||Target_Child|||\n@Comment:\toriginal metadata\n@G:\tA1\n")
CHAT = HEADER + "*CHI:\the go [: goes] [* m:03s:a]\n\thome .\n%mor:\told\n%gra:\told\n@End\n"


def test_deletion_policy_preserves_spoken_forms_boundaries_and_transcription_codes():
    text = HEADER + ("*CHI:\t<he gotted> [: he got] [* m:+ed] (.) he (be)cause &+w xxx [=? word] .\n"
        "*CHI:\t<he [^ EXA: tell me] go [* m:03s:a]> [//] he go [/] go [?] [!] [=! whispers] [<] [^c] [- eng] . [+ bch]\n"
        "*CHI:\t<home> [:: house] [= place] [% note] [^ comment] . [+ other]\n"
        "%mor:\told\n\twrapped old tier\n%gra:\told\n%umor:\told\n%ugra:\told\n%com:\told note\n@End\n")
    result, audit = prep.clean_chat_annotations(text)
    assert 'he gotted' in result and 'he got]' not in result and 'house' not in result
    assert '<he  go > [//]' not in result  # Exposed whitespace before a retained scope closes is removed.
    assert '<he  go> [//]' in result
    for token in ['(.)', '(be)cause', '&+w', 'xxx', '[//]', '[/]', '[?]', '[!]', '[=! whispers]', '[<]', '[^c]', '[- eng]', '[+ bch]']:
        assert token in result
    assert '@Comment:\toriginal metadata' in result
    assert 'word' not in result and 'place' not in result and '[+ other]' not in result
    assert audit['main_tiers'] == 3 and audit['chi_main_tiers'] == 3
    assert audit['removed_dependent_tiers'] == dict.fromkeys(['%mor','%gra','%umor','%ugra','%com'],1)
    # The complete output is obtainable solely by deleting the audited spans.
    reconstructed = text
    for edit in reversed(audit['edits']):
        reconstructed = reconstructed[:edit['start']] + reconstructed[edit['end']:]
    assert reconstructed == result


@pytest.mark.parametrize('body', ['a[* error]b .', '<he go .', 'he ] go .'])
def test_ambiguous_or_word_fusing_edits_fail(body):
    with pytest.raises(ValueError):
        prep.clean_chat_annotations(HEADER + '*CHI:\t' + body + '\n@End\n')


def collections(tmp_path):
    result = {}
    for label in ['real_enni','llama','qwen','gemma']:
        root = tmp_path / 'inputs' / label
        root.mkdir(parents=True)
        (root / 'same.cha').write_text(CHAT)
        result[label] = root
    return result


def test_mixed_collections_get_one_preparation_one_talktag_and_complete_lineage(tmp_path, monkeypatch):
    calls = fake_tool(monkeypatch)
    sources = collections(tmp_path)
    original = {label: (root/'same.cha').read_bytes() for label,root in sources.items()}
    metadata = {label:{'source_type':'real' if label=='real_enni' else 'synthetic'} for label in sources}
    path = prep.prepare_annotation_collections(sources,tmp_path/'preparation',collection_metadata=metadata,validator={'fixture':True})
    ref = prep.preparation_reference(path)
    record = prep.require_prepared_annotations(ref)
    assert record['files'] == 4 and len(record['outputs']) == 4
    assert not (path.parent/'source_snapshot/real_enni').exists()
    assert record['collections']['real_enni']['original_snapshot_retained'] is False
    for label in ['llama','qwen','gemma']:
        assert (path.parent/'source_snapshot'/label/'same.cha').read_bytes()==original[label]
    selected = {label:Path(item['prepared']) for label,item in record['collections'].items()}
    config = load_annotation_config(ROOT/'configs/annotation/talktag.json')
    parent = annotate_chat_directory(selected,tmp_path/'talktag',python=sys.executable,config=config,
        runtime={'source_commit':config['revision']},validator={'fixture':True},preparation_reference=ref)
    checked, prepared, annotated = prep.checked_talktag_collections(prep.preparation_reference(parent))
    assert len(calls) == 1 and set(checked) == set(sources)
    assert annotated['input_preparation'] == ref
    assert annotated['collections']['real_enni']['metadata']['source_type'] == 'real'
    for label,root in sources.items():
        assert (root/'same.cha').read_bytes() == original[label]
        cleaned = (selected[label]/'same.cha').read_text()
        assert '%mor' not in cleaned and '[*' not in cleaned and '[:' not in cleaned
        assert '[* m:03s:a]' in (checked[label]/'same.cha').read_text()
    (checked['qwen']/'same.cha').write_text('tampered')
    with pytest.raises(ValueError,match='changed'):
        prep.checked_talktag_collections(prep.preparation_reference(parent))


@pytest.mark.parametrize('target',['source','snapshot','prepared','audit','manifest','extra'])
def test_tampering_prevents_use_for_annotation(tmp_path,monkeypatch,target):
    fake_validator(monkeypatch)
    sources = collections(tmp_path)
    path = prep.prepare_annotation_collections(sources,tmp_path/'run',validator={'fixture':True})
    ref = prep.preparation_reference(path)
    locations = {'source':sources['real_enni']/'same.cha',
                 'snapshot':path.parent/'source_snapshot/real_enni/same.cha',
                 'prepared':path.parent/'prepared/real_enni/same.cha',
                 'audit':path.parent/'audits/real_enni/same.json',
                 'manifest':path, 'extra':path.parent/'prepared/real_enni/extra.cha'}
    locations[target].write_text('changed')
    with pytest.raises((ValueError,RuntimeError)):
        prep.require_prepared_annotations(ref)


def test_explicit_metadata_follows_stage_selection_and_overlap_is_rejected(tmp_path,monkeypatch):
    fake_validator(monkeypatch)
    roots = collections(tmp_path)
    for stage in ['stage1_td','stage2_sli']:
        (roots['llama']/stage).mkdir()
        (roots['llama']/stage/'same.cha').write_text(CHAT)
    chosen = annotation_inputs({'real_enni':roots['real_enni'],'llama':roots['llama']},['stage1_td','stage2_sli'])
    meta = annotation_collection_metadata({'real_enni':roots['real_enni'],'llama':roots['llama']}, chosen,
        {'real_enni':{'source_type':'real'},'llama':{'source_type':'synthetic','generator':'llama3_1_8b'}})
    assert meta['llama_stage2_sli']['generation_stage'] == 'stage2_sli'
    assert meta['real_enni']['source_type'] == 'real'
    with pytest.raises(ValueError,match='Overlapping'):
        prep.prepare_annotation_collections({'a':roots['llama'],'b':roots['llama']/'stage1_td'},tmp_path/'bad')
    assert not (tmp_path/'bad').exists()
    assert stage_flags('prepare_only') == (False,False,False)


def test_real_sources_are_referenced_without_original_backups_or_permanent_source_dependency(tmp_path,monkeypatch):
    fake_tool(monkeypatch)
    sources=collections(tmp_path)
    source=sources['real_enni'];file=source/'same.cha'
    file.write_text(CHAT.replace('@Begin','@PID:\t11312/c-00036000-1\n@Begin'))
    original=file.read_bytes()
    path=prep.prepare_annotation_collections({'real_enni':source},tmp_path/'run',
        collection_metadata={'real_enni':{'source_type':'real','corpus':'ENNI'}},validator={'fixture':True})
    ref=prep.preparation_reference(path);record=prep.require_prepared_annotations(ref)
    assert not (path.parent/'source_snapshot').exists()
    assert record['inputs'][0]['source_sha256']==prep.sha256_file(file)
    assert record['outputs'][0]['source_pid']=='11312/c-00036000-1'
    assert file.read_bytes()==original
    # Simulate no longer keeping the retrievable real input locally.
    file.unlink();source.rmdir()
    record=prep.require_prepared_annotations(ref)
    selected={'real_enni':Path(record['collections']['real_enni']['prepared'])}
    config=load_annotation_config(ROOT/'configs/annotation/talktag.json')
    parent=annotate_chat_directory(selected,tmp_path/'talktag',python=sys.executable,config=config,
        runtime={'source_commit':config['revision']},validator={'fixture':True},preparation_reference=ref)
    checked,_,_=prep.checked_talktag_collections(prep.preparation_reference(parent))
    assert set(checked)=={'real_enni'}
    # Derived input and audit integrity remain required even without originals.
    (selected['real_enni']/'same.cha').write_text('changed')
    with pytest.raises(ValueError,match='changed'):prep.require_prepared_annotations(ref)


def test_available_real_source_is_still_checked_and_previous_retained_manifests_still_validate(tmp_path,monkeypatch):
    fake_validator(monkeypatch)
    sources=collections(tmp_path)
    path=prep.prepare_annotation_collections({'real_enni':sources['real_enni']},tmp_path/'real',
        collection_metadata={'real_enni':{'source_type':'real'}},validator={'fixture':True})
    ref=prep.preparation_reference(path)
    (sources['real_enni']/'same.cha').write_text('changed')
    with pytest.raises(ValueError,match='source changed'):prep.require_prepared_annotations(ref)
    legacy=prep.prepare_annotation_collections({'synthetic':sources['llama']},tmp_path/'retained',validator={'fixture':True})
    data=json.loads(legacy.read_text())
    del data['collections']['synthetic']['original_snapshot_retained']
    legacy.write_text(json.dumps(data))
    assert prep.require_prepared_annotations(prep.preparation_reference(legacy))['status']=='prepared'


def test_prepare_only_notebook_works_without_any_model_setup(tmp_path,monkeypatch):
    fake_validator(monkeypatch)
    ns = dict(Path=Path,sys=sys,IS_COLAB=False,MODE='prepare_only',RUN_GENERATION=False,
        RUN_CHAT_CONVERSION=False,RUN_ANNOTATION=False,SMOKE_TEST=False,SAVE_TO_DRIVE=False,
        GENERATION_CSVS={},CHAT_INPUT_DIRS=collections(tmp_path),RUN_OUTPUT_DIR=tmp_path/'run',
        ANNOTATION_STAGES=['stage1_td','stage2_sli'],INSTALL_DEPENDENCIES=False)
    notebook=json.loads((ROOT/'notebooks/colab_direct_generation_pipeline.ipynb').read_text())
    for name in ['common-setup','annotation-preparation']:
        exec(''.join(next(c['source'] for c in notebook['cells'] if c['id']==name)),ns)
    assert len(ns['prepared_annotation_inputs']) == 4
    assert 'annotation_python' not in ns and 'morphosyntax_python' not in ns
    assert json.loads((ns['RUN_OUTPUT_DIR']/'pipeline_manifest.json').read_text())['stages']['annotation_preparation']['status']=='complete'


def test_failed_chatter_is_a_blocking_gate_with_retained_preparation_audits(tmp_path,monkeypatch):
    def failed(inputs, folder, **kwargs):
        folder=Path(folder);folder.mkdir(parents=True)
        path=folder/'validation_manifest.json';path.write_text('{"status":"failed"}')
        raise validation.ChatterValidationError(path)
    monkeypatch.setattr(validation,'validate_chat_directories',failed)
    with pytest.raises(validation.ChatterValidationError):
        prep.prepare_annotation_collections(collections(tmp_path),tmp_path/'run',validator={'fixture':True})
    record=json.loads((tmp_path/'run/preparation_manifest.json').read_text())
    assert record['status']=='failed_or_interrupted' and record['validation']
    assert len(record['outputs'])==4 and (tmp_path/'run/audits/real_enni/same.json').exists()


def test_native_chatter_preparation(tmp_path):
    chatter=os.environ.get('ICL_PILOT_CHATTER')
    if not chatter:pytest.skip('Opt-in native Chatter check; no downloads')
    sources=collections(tmp_path)
    path=prep.prepare_annotation_collections(sources,tmp_path/'prepared',validator=validation.verify_chatter(chatter),test_only=True)
    record=prep.require_prepared_annotations(prep.preparation_reference(path))
    assert record['files']==4 and record['test_only']


def test_joint_notebook_route_with_native_ud_and_fixture_talktag(tmp_path,monkeypatch):
    python=os.environ.get('ICL_PILOT_UD_PYTHON');models=os.environ.get('ICL_PILOT_UD_MODELS');chatter=os.environ.get('ICL_PILOT_CHATTER')
    if not all((python,models,chatter)):
        pytest.skip('Opt-in native UD integration check; no downloads or TalkTag inference')
    real_run=subprocess.run
    calls=[]
    def run(command,**kwargs):
        if 'talk_tag.cli' not in command:
            if any(str(value).endswith('morphosyntax_worker.py') for value in command):calls.append('ud')
            return real_run(command,**kwargs)
        calls.append('talktag_fixture')
        source=Path(command[command.index('--input-dir')+1]);out=Path(command[command.index('--output-dir')+1])
        files=list(source.rglob('*.cha'))
        for path in files:
            text=re.sub(r'go\s+home','go [* m:03s:a] home',path.read_text())
            target=out/path.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text)
        (out/'_talk_tag_report.json').write_text(json.dumps(dict(total_files=len(files),processed_files=len(files),failed_files=0,target_lines=len(files))))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess,'run',run)
    monkeypatch.setattr('icl_pilot.chat.annotation.annotation_runtime_metadata',lambda python,cfg:{'source_commit':cfg['revision'],'fixture':True})
    sources=collections(tmp_path)
    ns=dict(Path=Path,sys=sys,MODE='from_chat',IS_COLAB=False,SAVE_TO_DRIVE=False,
        RUN_GENERATION=False,RUN_CHAT_CONVERSION=False,RUN_ANNOTATION=True,RUN_MORPHOSYNTAX=True,
        GENERATION_CSVS={},CHAT_INPUT_DIRS=sources,CHAT_INPUT_METADATA={label:{'source_type':'real' if label=='real_enni' else 'synthetic'} for label in sources},
        RUN_OUTPUT_DIR=tmp_path/'run',RUN_ID='native_fixture',SMOKE_TEST=False,INSTALL_DEPENDENCIES=False,
        ANNOTATION_STAGES=['stage1_td','stage2_sli'],ANNOTATION_ENV=tmp_path/'fixture',
        ANNOTATION_CONFIG_PATH=ROOT/'configs/annotation/talktag.json',
        MORPHOSYNTAX_ENV=Path(python).parent.parent,MORPHOSYNTAX_MODELS_DIR=Path(models),
        MORPHOSYNTAX_CONFIG_PATH=ROOT/'configs/annotation/batchalign.json',
        CHATTER_EXECUTABLE=chatter,CHATTER_CONFIG_PATH=ROOT/'configs/validation/chatter.json')
    notebook=json.loads((ROOT/'notebooks/colab_direct_generation_pipeline.ipynb').read_text())
    for name in ['common-setup','annotation-preparation','chat-validation-before','annotation-setup','annotation-run',
                 'chat-validation-after','morphosyntax-setup','morphosyntax-run']:
        exec(''.join(next(c['source'] for c in notebook['cells'] if c['id']==name)),ns)
    record=json.loads(ns['morphosyntax_manifest'].read_text())
    assert calls==['talktag_fixture','ud'] and record['status']=='complete'
    assert len(record['outputs'])==4 and set(record['collections'])==set(sources)
    assert record['input_preparation']==ns['annotation_preparation_reference']
    for label,root in sources.items():
        assert (root/'same.cha').read_text()==CHAT
        text=(ns['morphosyntax_output_dir']/label/'same.cha').read_text()
        assert '%mor:' in text and '%gra:' in text and '[* m:03s:a]' in text
