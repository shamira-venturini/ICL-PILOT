"""UD annotation preserves source speech/codes and requires complete fresh output."""
import ast
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from icl_pilot.chat import morphosyntax as ud, validation
from icl_pilot.pipeline import begin_pipeline, stage_flags

ROOT = Path(__file__).resolve().parents[1]
HEADER = ("@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n"
          "@ID:\teng|test|CHI|4;01.00||TD||Target_Child|||\n@Types:\tcross, narrative, TD\n@G:\tA1\n")
CHAT = HEADER + "*CHI:\the go [* m:03s:a]\n\thome .\n%com:\tkeep this note .\n@End\n"
MOR = "%mor:\tpron|he-Prs-Nom-S3 verb|go-Fin-Ind-Pres-S adv|home .\n"
GRA = "%gra:\t1|2|NSUBJ 2|0|ROOT 3|2|ADVMOD 4|2|PUNCT\n"
PROVENANCE = "@Comment:\tbatchalign c8cd2e0 | morphosyntax: stanza:1.15.0:noretok:native-mwt10 | 2026-10-29T12:00:00Z\n"


def runtime(config):
    return dict(packages={"batchalign":config['version'],"stanza":config['stanza_version'],"torch":config['torch_version']},
                source_sha256=config['source_sha256'], resources_sha256=config['resources']['sha256'],
                model_files={k:{'sha256':v['sha256']} for k,v in config['models'].items()},
                retokenize=False,resegment=False,cache_policy='Bypass',device='cpu',workers=1,
                engine_sha256={'fixture_engine':'0'*64})


def fake_worker(monkeypatch, mode='normal'):
    monkeypatch.setattr(ud,'prepare_morphosyntax_models',lambda root,*args,**kwargs:Path(root))
    def validate(inputs, folder, **kwargs):
        folder=Path(folder);folder.mkdir(parents=True)
        path=folder/'validation_manifest.json'
        path.write_text(json.dumps({'status':'passed','inputs':{k:str(v) for k,v in inputs.items()}}))
        return path
    monkeypatch.setattr(validation,'validate_chat_directories',validate)
    monkeypatch.setattr(validation,'require_validated_report',lambda ref:json.loads(Path(ref['manifest']).read_text()))
    calls=[]
    def run(command, **kwargs):
        calls.append(command)
        if mode=='exit':return SimpleNamespace(returncode=1)
        request_path,response_path=map(Path,command[-2:]);request=json.loads(request_path.read_text())
        raw=Path(request['output_dir']);raw.mkdir(parents=True)
        completed=[]
        for item in request['inputs']:
            if mode=='missing':continue
            text=Path(item['path']).read_text()
            assert not any(tier in text for tier in ('%mor:','%gra:','%umor:','%ugra:'))
            text=text.replace('home .\n','home .\n'+MOR+GRA).replace('@Types:',PROVENANCE+'@Types:')
            text=text.replace('[* m:03s:a]\n\thome','[* m:03s:a] home')
            if mode=='word':text=text.replace('home .','away .')
            if mode=='code':text=text.replace(' [* m:03s:a]','')
            if mode=='header':text=text.replace('4;01.00','5;01.00')
            if mode=='tier':text=text.replace(GRA,'')
            if mode=='boundary':text=text.replace('home .','home .\n*CHI:\tmore .')
            if mode=='comment':text=text.replace('keep this note','changed note')
            target=raw/item['relative_path'];target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text)
            completed.append(item['relative_path'])
        if mode=='extra':(raw/'extra.cha').write_text(CHAT)
        if mode=='duplicate':completed*=2
        meta=runtime(request['config'])
        if mode=='version':meta['packages']['batchalign']='0.8.2'
        if mode=='cache':meta['cache_policy']='Use'
        response_path.write_text(json.dumps({'completed_files':completed,'runtime':meta,'request_sha256':ud.sha256_file(request_path)}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(ud.subprocess,'run',run)
    return calls


def source_dir(tmp_path):
    source=tmp_path/'source';(source/'nested').mkdir(parents=True)
    text=CHAT.replace('%com:',MOR+GRA+'%umor:\told\n%ugra:\told\n%com:')
    (source/'nested/one.cha').write_text(text)
    return source,text


def test_fresh_ud_uses_copies_preserves_wrapped_speech_and_drops_stale_aliases(tmp_path,monkeypatch):
    calls=fake_worker(monkeypatch)
    source,text=source_dir(tmp_path)
    path=ud.annotate_morphosyntax({'saved':source},tmp_path/'run',python=sys.executable,model_dir=tmp_path/'models',validator={'fixture':True})
    record=json.loads(path.read_text())
    assert record['status']=='complete' and len(calls)==1
    assert (source/'nested/one.cha').read_text()==text
    assert (path.parent/'input_snapshot/saved/nested/one.cha').read_bytes()==(source/'nested/one.cha').read_bytes()
    output=(path.parent/'annotated/saved/nested/one.cha').read_text()
    assert '[* m:03s:a]\n\thome .\n'+MOR+GRA in output
    assert '%umor:' not in output and '%ugra:' not in output
    assert 'keep this note' in output and PROVENANCE not in output
    assert record['outputs'][0]['native_added_provenance']
    assert record['inputs'][0]['removed_tiers']=={'%mor':1,'%gra':1,'%umor':1,'%ugra':1}
    with pytest.raises(FileExistsError):
        ud.annotate_morphosyntax({'saved':source},tmp_path/'run',python=sys.executable,model_dir=tmp_path/'models',validator={'fixture':True})


@pytest.mark.parametrize('mode',['exit','missing','word','code','header','tier','boundary','comment','extra','duplicate','version','cache'])
def test_failures_retain_diagnostics_and_cannot_report_completion(tmp_path,monkeypatch,mode):
    fake_worker(monkeypatch,mode)
    source,text=source_dir(tmp_path)
    with pytest.raises((ValueError,RuntimeError)):
        ud.annotate_morphosyntax({'saved':source},tmp_path/'run',python=sys.executable,model_dir=tmp_path/'models',validator={'fixture':True})
    record=json.loads((tmp_path/'run/morphosyntax_manifest.json').read_text())
    assert record['status']=='failed_or_interrupted'
    assert (source/'nested/one.cha').read_text()==text
    assert (tmp_path/'run/batchalign.log').exists()


def test_nonlexical_utterances_have_explicit_exclusion_instead_of_missing_lexical_tags():
    source=HEADER+'*CHI:\txxx .\n@End\n'
    result=ud.check_morphosyntax_output(source,source)
    assert result['target_utterances']==1 and result['tagged_utterances']==0
    assert result['nonlexical_utterance_indices']==[{'speaker_utterance_index':1,'speaker':'*CHI'}]
    with pytest.raises(ValueError,match='Missing UD'):
        ud.check_morphosyntax_output(CHAT,CHAT)


def test_configuration_refuses_retokenization_and_model_download_never_overwrites_bad_cache(tmp_path):
    config=json.loads((ROOT/'configs/annotation/batchalign.json').read_text())
    config['retokenize']=True
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='preserved tokens'):ud.load_morphosyntax_config(path)
    models=tmp_path/'models';models.mkdir();(models/'resources.json').write_text('different catalog')
    with pytest.raises(ValueError,match='checksum mismatch'):
        ud.prepare_morphosyntax_models(models,download=False)
    assert (models/'resources.json').read_text()=='different catalog'


def test_input_overlap_and_saved_input_mode_requirements(tmp_path,monkeypatch):
    source,text=source_dir(tmp_path)
    assert stage_flags('morphosyntax_only')==(False,False,False)
    with pytest.raises(ValueError,match='CHAT_INPUT_DIRS'):
        begin_pipeline(tmp_path/'run',mode='morphosyntax_only',csvs={},chat_dirs={})
    with pytest.raises(ValueError,match='overlap'):
        ud._inventory({'a':source,'b':source/'nested'})


def test_notebook_ids_and_python38_syntax():
    notebook=json.loads((ROOT/'notebooks/colab_direct_generation_pipeline.ipynb').read_text())
    ids=[cell['id'] for cell in notebook['cells']]
    assert len(ids)==len(set(ids))
    assert {'morphosyntax-guide','morphosyntax-setup','morphosyntax-run'}.issubset(ids)
    assert not any('grasp' in name for name in ids)
    for cell in notebook['cells']:
        if cell['cell_type']=='code':ast.parse(''.join(cell['source']),feature_version=(3,8))


def test_native_saved_chat_notebook_route_without_generating_or_rerunning_talktag(tmp_path):
    python=os.environ.get('ICL_PILOT_UD_PYTHON');models=os.environ.get('ICL_PILOT_UD_MODELS');chatter=os.environ.get('ICL_PILOT_CHATTER')
    if not all((python,models,chatter)):
        pytest.skip('Opt-in native check needs pinned Batchalign runtime/models/Chatter; tests do not download')
    source,text=source_dir(tmp_path)
    (source/'contraction.cha').write_text(CHAT.replace('he go [* m:03s:a]\n\thome',"he can't go home"))
    ns=dict(Path=Path,sys=sys,MODE='morphosyntax_only',IS_COLAB=False,SAVE_TO_DRIVE=False,
            RUN_GENERATION=False,RUN_CHAT_CONVERSION=False,RUN_ANNOTATION=False,RUN_MORPHOSYNTAX=True,
            GENERATION_CSVS={},CHAT_INPUT_DIRS={'saved':source},RUN_OUTPUT_DIR=tmp_path/'run',RUN_ID='native_fixture',
            SMOKE_TEST=False,INSTALL_DEPENDENCIES=False,ANNOTATION_STAGES=['stage1_td','stage2_sli'],
            MORPHOSYNTAX_ENV=Path(python).parent.parent,MORPHOSYNTAX_MODELS_DIR=Path(models),
            MORPHOSYNTAX_CONFIG_PATH=ROOT/'configs/annotation/batchalign.json',
            CHATTER_EXECUTABLE=chatter,CHATTER_CONFIG_PATH=ROOT/'configs/validation/chatter.json')
    notebook=json.loads((ROOT/'notebooks/colab_direct_generation_pipeline.ipynb').read_text())
    for name in ('common-setup','morphosyntax-setup','morphosyntax-run'):
        exec(''.join(next(cell['source'] for cell in notebook['cells'] if cell['id']==name)),ns)
    record=json.loads(ns['morphosyntax_manifest'].read_text())
    assert record['status']=='complete' and len(record['outputs'])==2
    validation.require_validated_report(record['chatter']['before'])
    validation.require_validated_report(record['chatter']['after'])
    assert (source/'nested/one.cha').read_text()==text
    assert 'model' not in ns and 'annotation_python' not in ns and 'prepared' not in ns
