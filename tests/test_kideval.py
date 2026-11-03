"""Analysis-unit, provenance and export checks; native smoke evidence is recorded separately."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pandas as pd
import pytest

from icl_pilot.features import kideval as workflow
from icl_pilot.features.kideval_runtime import sha256, verify_runtime

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project(tmp_path):
    root = tmp_path / 'project'
    config = root / 'configs/generation'
    config.mkdir(parents=True)
    (root / 'configs/features').mkdir()
    (root / 'configs/features/kideval.json').write_bytes((ROOT / 'configs/features/kideval.json').read_bytes())
    (config / 'experiment.json').write_text(json.dumps({'roster_csv':'pairs.csv',
        'provenance':{'participant_manifest':'configs/generation/people.csv'}}))
    pd.DataFrame([{'File_ID':'410','source_pid':'pid410','Age(Month)':48,'Group':'TD'},
                  {'File_ID':'444','source_pid':'pid444','Age(Month)':49,'Group':'SLI'}]).to_csv(config / 'people.csv', index=False)
    pd.DataFrame([{'pair_id':'p1','td_child_id':'410','sli_child_id':'444'}]).to_csv(config / 'pairs.csv', index=False)
    (config / 'models.json').write_text(json.dumps({'models':{'model':{'id':'org/model','revision':'abc'}}}))
    return root


def chat(stories=workflow.STORIES, generated=False, run='run1'):
    text = '@UTF8\n@Begin\n@PID:\tpid410\n@Participants:\tCHI Child\n@ID:\teng|ENNI|CHI|4;00.00|male|TD|||Child|||\n@Media:\t410, audio\n'
    if generated:
        fields = dict(data_source='generated',model_key='model',model_id='org/model',model_revision='abc',
                      run_id=run,bundle_id='b1',pair_id='p1',replicate_id='01',stage='stage1_td',
                      source_child_id='410',target_story=stories[0])
        text += ''.join('@Comment:\t' + k + '=' + v + '\n' for k,v in fields.items())
    for story in stories:
        text += '@G:\t' + story + '\n*CHI:\the runs . \x15100_200\x15\n%mor:\tpron|he verb|run\n%gra:\t1|2|SUBJ 2|0|ROOT\n'
    return text + '@End\n'


def source(text, name='real/410.cha', coverage='complete'):
    return dict(text=text,relative_path=name,uri=name,manifest='fixture',sha256=hashlib.sha256(text.encode()).hexdigest(),
                test_only=True,error_annotation_coverage=coverage)


def test_story_sections_keep_boundaries_and_dependent_tiers():
    prefix, sections = workflow.story_sections(chat())
    assert len(sections) == 6
    assert '%gra:' in ''.join(sections['A1'])
    assert '*CHI:\the runs .' in ''.join(sections['A1'])
    assert '@PID:\tpid410\n' in prefix
    with pytest.raises(ValueError, match='Duplicate canonical'):
        workflow.story_sections(chat(('A1','A1')))


def test_real_and_synthetic_units_keep_distinct_ancestry_and_runs(project):
    sources = [source(chat())]
    for run in ('run1','run2'):
        sources.extend(source(chat((story,),True,run),run + '/' + story) for story in workflow.STORIES)
    units = workflow.assemble_units(sources,project)
    assert len(units) == 3
    assert all(len(u['stories']) == 6 for u in units.values())
    assert len({u['metadata']['File_ID'] for u in units.values()}) == 3
    with pytest.raises(ValueError,match='Duplicate child/model/bundle/story'):
        workflow.assemble_units(sources + [sources[0]],project)
    bad = source(chat(('A1',),True).replace('source_child_id=410','source_child_id=444'))
    with pytest.raises(ValueError,match='identity conflict'):
        workflow.assemble_units([bad],project)
    with pytest.raises(ValueError,match='age/group'):
        workflow.assemble_units([source(chat().replace('4;00.00','5;00.00'))],project)


def test_zip_reader_checks_final_hash_and_preserves_coverage(tmp_path):
    path = tmp_path / 'annotations.zip'
    text = chat()
    manifest = {'status':'complete','test_only':True,'annotated_output_directory':'annotated',
                'outputs':[{'relative_path':'real/410.cha','sha256':hashlib.sha256(text.encode()).hexdigest()}]}
    def write(payload):
        with zipfile.ZipFile(path,'w') as z:
            z.writestr('morphosyntax/morphosyntax_manifest.json',json.dumps(manifest))
            z.writestr('morphosyntax/annotated/real/410.cha',payload)
            z.writestr('annotation/annotation_manifest.json',json.dumps({'outputs':[
                {'relative_path':'real/410.cha','annotation_coverage':'partial'}]}))
    write(text)
    loaded,refs = workflow.read_final_inputs([path])
    assert loaded[0]['error_annotation_coverage'] == 'partial'
    assert refs[str(path)] == sha256(path)
    assert loaded[0]['text'] == text
    write(text.replace('runs','walks'))
    with pytest.raises(ValueError,match='checksum mismatch'):
        workflow.read_final_inputs([path])


def test_runtime_tampering_is_rejected(tmp_path):
    binary = tmp_path / 'binary'
    binary.write_bytes(b'original')
    receipt = tmp_path / 'runtime.json'
    receipt.write_text(json.dumps({'files':{str(binary):sha256(binary)}}))
    verify_runtime(receipt)
    binary.write_bytes(b'changed')
    with pytest.raises(ValueError,match='runtime file changed'):
        verify_runtime(receipt)


@pytest.mark.parametrize('coverage',['complete','partial','unknown'])
@pytest.mark.parametrize('report_ok',[True,False])
def test_export_missingness_incomplete_units_and_native_options(project,tmp_path,monkeypatch,coverage,report_ok):
    sources = [source(chat(),coverage=coverage),source(chat(('A1',),True),'synthetic/A1.cha')]
    original = tmp_path / 'source.cha'
    original.write_text(chat())
    reference = sha256(original)
    monkeypatch.setattr(workflow,'read_final_inputs',lambda paths:(sources,{str(original):reference}))
    config = json.loads((project / 'configs/features/kideval.json').read_text())
    runtime = tmp_path / 'runtime.json'
    runtime.write_text('{}')
    monkeypatch.setattr(workflow,'verify_runtime',lambda p:{'archive_sha256':config['archive_sha256'],
        'binaries':{'kideval':'kideval','dss':'dss'},'lib':'library'})
    monkeypatch.setattr(workflow.validation,'validate_chat_directories',lambda *a,**kw:{'status':'passed'})
    monkeypatch.setattr(workflow.validation,'validation_reference',lambda report:report)
    monkeypatch.setattr(workflow.validation,'require_validated_report',lambda report:None)
    seen = []
    def native(command,**kwargs):
        seen.append(command)
        assert command[1:4] == ['+0','+lengu','+t*CHI']
        assert '+o4' not in command and not any(v.startswith('+q') for v in command)
        if command[0] == 'kideval':
            (kwargs['cwd'] / 'result.kideval.xls').write_text('fixture')
        else:
            for path in kwargs['cwd'].glob('*.cha'):
                path.with_suffix('.tbl.cex').write_text('fixture')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(workflow.subprocess,'run',native)
    def raw(path):
        name = next(path.parent.glob('*.cha')).name if report_ok else 'unrecognized.cha'
        return pd.DataFrame([{'File':name,**dict.fromkeys(workflow.CORE,2),'DSS':8,
                              'Word_Errors':5,'Utt_Errors':4,'TD_Time_(secs)':90}]),[],[]
    monkeypatch.setattr(workflow,'_load_raw_frame',raw)
    monkeypatch.setattr(workflow,'parse_dss_tbl_file',lambda *a,**kw:{'DSS_tbl_score':'8','DSS_tbl_IP':'3'})
    output = tmp_path / 'features'
    if not report_ok:
        with pytest.raises(ValueError,match='rows do not match'):
            workflow.extract_kideval([],output,project_root=project,runtime=runtime,validator='fixture')
        assert json.loads((output / 'feature_manifest.json').read_text())['status'] == 'failed_or_interrupted'
        assert sha256(original) == reference
        return
    result = workflow.extract_kideval([],output,project_root=project,runtime=runtime,validator='fixture')
    manifest = json.loads(result.read_text())
    data = pd.read_csv(output / 'kideval_features.csv')
    assert len(data) == 2 and manifest['complete_units'] == 1 and len(seen) == 2
    real = data.loc[data.source.eq('real')].iloc[0]
    assert pd.isna(real.DSS) == (coverage != 'complete')
    assert real.DSS_tbl_IP == 3  # Category totals retain their raw meaning.
    assert real.retracing_per_utt == 1
    assert data.loc[data.source.eq('synthetic'),'MLU_Words'].isna().all()
    assert not {'Word_Errors','Utt_Errors','TD_Time_(secs)'} & set(data.columns)
    assert not any('IPSyn' in col for col in data.columns)
    assert sha256(original) == reference
    bundle = next((output / 'bundles').glob('*.cha')).read_text()
    assert '@Media:' not in bundle and '\x15' not in bundle
    assert '*CHI:\the runs .' in bundle and '%gra:' in bundle
    assert (output / 'real_kideval_features.csv').is_file()
    assert (output / 'synthetic_kideval_features.csv').is_file()
    assert manifest['ipsyn_computed'] is False
    with pytest.raises(FileExistsError):
        workflow.extract_kideval([],output,project_root=project,runtime=runtime,validator='fixture')


def test_notebook_feature_blocks_default_off_and_compile():
    for path in [ROOT / 'notebooks/colab_direct_generation_pipeline.ipynb',
                 ROOT / 'ICL-PILOT-smoke-test-20261030/ICL-PILOT_smoke_test.ipynb']:
        notebook = json.loads(path.read_text())
        cells = {c['id']:c for c in notebook['cells']}
        assert 'domain-features' in ''.join(cells['overview']['source'])
        namespace = dict(REPO_DIR=ROOT,IS_COLAB=False,Path=Path)
        for name in ['features-settings','features-support','features-runtime','features-run','features-preview','features-export']:
            code = ''.join(cells[name]['source'])
            exec(compile(code,str(path) + ':' + name,'exec'),namespace)
        assert namespace['RUN_DOMAIN_FEATURES'] is False
        assert namespace['feature_manifest'] is None
