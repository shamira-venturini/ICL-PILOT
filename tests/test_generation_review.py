"""QC is review-only: raw stories, ancestry and analysis eligibility stay intact."""
import ast
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from icl_pilot.generation import review
from icl_pilot.pipeline import stage_flags, package_run

ROOT = Path(__file__).resolve().parents[1]


def saved_inputs(tmp_path, *, journal=True):
    source = tmp_path/'saved';source.mkdir()
    rows=[];events=[{'kind':'run','run_id':'fixture','test_only':True}]
    examples=[('A1','he go home .','he\n go   home .','end_token','end_token'),
              ('A2','he went home .','Return only the narrative text. '+'dog '*30,'end_token','token_ceiling'),
              ('A3','the dog went home .','the dog went home '*6,'end_token','end_token'),
              ('B1','no no no no !','the rabbit went swimming .','end_token','end_token')]
    for i,(story,first,second,stop1,stop2) in enumerate(examples):
        row=dict(run_id='fixture',model_key='fixture_model',model_id='fixture/checkpoint',model_revision='fixture',
                 cohort='4-year-old',cohort_label='4-year-old',round_id='round_01',pair_id='pair_01',bundle_id='bundle_01',
                 replicate_id='1',target_story=story,stage1_td_story=first,stage2_sli_story=second,
                 stage1_prompt_record_id=f'first{i}',stage2_prompt_record_id=f'second{i}',test_only='True')
        rows.append(row)
        for stage,text,reason,key in [('stage1_td',first,stop1,row['stage1_prompt_record_id']),('stage2_sli',second,stop2,row['stage2_prompt_record_id'])]:
            events.append({'kind':'prompt','record_id':key,'stage':stage,**{k:row[k] for k in ['run_id','pair_id','bundle_id','target_story']},
                           'parent_record_id':row['stage1_prompt_record_id'] if stage=='stage2_sli' else None,
                           'parent_text_sha256':review.text_hash(first) if stage=='stage2_sli' else None})
            events.append({'kind':'output','record_id':key,'output_text':text,'output_text_sha256':review.text_hash(text),
                           'finish_reason':reason,'generated_token_count':640 if i==1 else len(text.split()),
                           'reached_token_ceiling':i==1,'terminal_token_id':1 if reason=='end_token' else 2})
    path=source/'generation.csv'
    with path.open('w',newline='') as handle:
        w=csv.DictWriter(handle,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    if journal:(source/'prompt_records.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    return path,rows,events


def test_all_stories_are_retained_and_paired_with_source_provenance_and_readable_reports(tmp_path):
    path,rows,_=saved_inputs(tmp_path);raw={p:p.read_bytes() for p in path.parent.iterdir()}
    report=review.write_generation_review({'fixture':path},tmp_path/'review')
    manifest=json.loads(report.read_text());data=json.loads((report.parent/'review_data.json').read_text())
    assert manifest['status']=='review_report_complete' and manifest['review_only']
    assert manifest['stories_excluded']==0 and manifest['analysis_inputs_modified'] is False
    assert manifest['summary']['stories']==8 and manifest['summary']['jobs']==4
    records={r['review_id']:r for r in data['stories']}
    for r in records.values():
        row=rows[r['source_row']-1];field=review.STAGES[r['stage']]
        assert r['story_text']==row[field] and r['story_text_sha256']==review.text_hash(row[field])
        other=records[r['paired_review_id']]
        assert other['target_story']==r['target_story'] and other['stage']!=r['stage']
    assert manifest['summary']['flag_counts']=={'unchanged_stage2':1,'generation_token_limit':1,'possible_prompt_leakage':1,'long_word_run':1,'repeated_phrase_loop':1}
    # A natural end at exactly 640 tokens is retained without a cutoff flag.
    first=next(r for r in records.values() if r['stage']=='stage1_td' and r['target_story']=='A2')
    assert first['reached_token_ceiling'] and 'generation_token_limit' not in first['flags']
    assert all(p.read_bytes()==value for p,value in raw.items())
    all_rows=list(csv.DictReader((report.parent/'all_stories.csv').open()))
    flagged=list(csv.DictReader((report.parent/'flagged_stories.csv').open()))
    assert len(all_rows)==8 and len(flagged)==3
    assert all(r['review_status']=='pending' for r in all_rows)
    html=(report.parent/'review.html').read_text()
    assert 'Stage 1 — TD-conditioned' in html and 'Stage 2 — SLI-conditioned' in html
    assert 'Download review notes' in html and 'story_text' in html
    for name,sha in manifest['outputs'].items():assert review.digest(report.parent/name)==sha
    again=review.write_generation_review({'fixture':path},tmp_path/'review2')
    assert [r['review_id'] for r in json.loads((again.parent/'review_data.json').read_text())['stories']]==list(records)
    with pytest.raises(FileExistsError):review.write_generation_review({'fixture':path},tmp_path/'review')


def test_missing_journal_leaves_stopping_unknown_without_guessing_from_length(tmp_path):
    path,_,_=saved_inputs(tmp_path,journal=False)
    report=review.write_generation_review({'fixture':path},tmp_path/'review')
    data=json.loads((report.parent/'review_data.json').read_text())
    assert all(r['finish_reason'] is None and r['stopping_metadata']=='journal unavailable' for r in data['stories'])
    assert not any('generation_token_limit' in r['flags'] for r in data['stories'])
    assert json.loads(report.read_text())['summary']['stories_with_recorded_stopping_reasons']==0


@pytest.mark.parametrize('corruption',['text','stage','parent','missing','duplicate'])
def test_mismatched_journal_cannot_supply_invented_stopping_or_parent_information(tmp_path,corruption):
    path,rows,events=saved_inputs(tmp_path)
    if corruption=='text':events[2]['output_text']='changed';events[2]['output_text_sha256']=review.text_hash('changed')
    if corruption=='stage':events[1]['stage']='stage2_sli'
    if corruption=='parent':events[3]['parent_text_sha256']='0'*64
    if corruption=='missing':events.pop(4)
    if corruption=='duplicate':events.append(events[2])
    (path.parent/'prompt_records.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    with pytest.raises(ValueError):review.write_generation_review({'fixture':path},tmp_path/'review')
    assert not (tmp_path/'review').exists()


def test_prompt_flags_use_template_instructions_not_expected_narrative_content():
    config,patterns,_=review.load_review_config()
    text='The rabbit and dog went to the swimming pool. He said no no no no!'
    assert review.screen_story(text,'stage1_td','',{},config,patterns)['flags']==[]
    assert 'possible_prompt_leakage' in review.screen_story('Target story events:\nThe rabbit went home.','stage1_td','',{},config,patterns)['flags']
    assert 'possible_prompt_leakage' in review.screen_story('<|im_start|>assistant\nThe rabbit went home.','stage1_td','',{},config,patterns)['flags']
    assert 'empty_output' in review.screen_story('  ','stage1_td','',{},config,patterns)['flags']
    # Ordinary and even unusually long real-speech examples do not hit the default long-loop flag.
    assert 'long_word_run' not in review.screen_story('she '*10+'tried again.','stage1_td','',{},config,patterns)['flags']


def test_reports_escape_html_and_javascript_without_altering_stored_story_text(tmp_path):
    path,rows,_=saved_inputs(tmp_path,journal=False)
    text='</script><img src=x onerror=alert(1)> café & "rabbit"\nsecond line'
    rows[0]['stage1_td_story']=text
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report=review.write_generation_review({'fixture':path},tmp_path/'review')
    html=(report.parent/'review.html').read_text()
    assert '</script><img' not in html and '\\u003c/script>' in html
    import re
    payload=json.loads(re.search(r'<script id="review-data" type="application/json">(.*?)</script>',html,re.S).group(1))
    assert payload['stories'][0]['story_text']==text
    scripts=re.findall(r'<script>(.*?)</script>',html,re.S)
    js=tmp_path/'review.js';js.write_text(scripts[0])
    import shutil
    node=shutil.which('node')
    if node:
        result=subprocess.run([node,'--check',str(js)],capture_output=True,text=True)
        assert result.returncode==0,result.stderr


def test_review_only_notebook_runs_without_generation_conversion_or_annotation(tmp_path):
    path,_,_=saved_inputs(tmp_path)
    ns=dict(Path=Path,sys=sys,MODE='review_only',IS_COLAB=False,SAVE_TO_DRIVE=False,
        RUN_GENERATION=False,RUN_CHAT_CONVERSION=False,RUN_ANNOTATION=False,SMOKE_TEST=False,
        GENERATION_CSVS={'fixture':path},CHAT_INPUT_DIRS={},RUN_OUTPUT_DIR=tmp_path/'run',RUN_ID='review',
        MAKE_ARCHIVE=False)
    n=json.loads((ROOT/'notebooks/colab_direct_generation_pipeline.ipynb').read_text())
    for name in ['common-setup','generation-review']:
        exec(''.join(next(c['source'] for c in n['cells'] if c['id']==name)),ns)
    assert ns['generation_review_manifest'].exists()
    pipeline=json.loads((ns['RUN_OUTPUT_DIR']/'pipeline_manifest.json').read_text())
    assert pipeline['stages']['generation_review']['status']=='complete'
    assert pipeline['stages']['generation']['status']==pipeline['stages']['chat_conversion']['status']==pipeline['stages']['annotation']['status']=='skipped'
    assert not any(name in ns for name in ['model','tokenizer','segmentation_runtime','annotation_python'])
    archive=package_run(ns['RUN_OUTPUT_DIR'])
    import zipfile
    with zipfile.ZipFile(archive) as z:assert any(name.endswith('/review.html') for name in z.namelist())
    assert stage_flags('review_only')==(False,False,False)
    for c in n['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']),feature_version=(3,8))


def test_overlapping_inputs_and_attempts_to_enable_exclusions_fail(tmp_path):
    path,_,_=saved_inputs(tmp_path)
    with pytest.raises(ValueError):review.write_generation_review({'a':path,'b':path},tmp_path/'bad')
    duplicate=path.parent/'aggregate.csv';duplicate.write_bytes(path.read_bytes())
    with pytest.raises(ValueError,match='Overlapping'):review.write_generation_review({'a':path,'b':duplicate},tmp_path/'bad')
    cfg=json.loads(review.DEFAULT_CONFIG.read_text());cfg['exclude_stories']=True
    target=tmp_path/'bad.json';target.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='review-only'):review.load_review_config(target)
