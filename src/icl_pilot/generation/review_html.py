"""Portable, offline story-review page; notes are never applied to analysis."""
import hashlib
import json


def review_html(records, summary):
    review_set_id = hashlib.sha256("|".join(r["review_id"] for r in records).encode()).hexdigest()
    payload = json.dumps({"stories": records, "summary": summary, "review_set_id": review_set_id}, ensure_ascii=False).replace("<", "\\u003c")
    return TEMPLATE.replace("__REVIEW_DATA__", payload)


TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ICL-PILOT — generation review</title>
<style>
:root{color-scheme:light;--ink:#18333c;--muted:#52656d;--line:#d6dfdf;--accent:#125b5b}
*{box-sizing:border-box}body{margin:0;background:#f3f6f5;color:var(--ink);font:15px/1.5 system-ui,sans-serif}
main{max-width:1450px;margin:auto;padding:26px}h1{font-size:27px;margin:0 0 8px}h2{font-size:20px;margin:0 0 12px}
p{margin:8px 0}.muted{color:var(--muted)}.card{background:white;border:1px solid var(--line);border-radius:10px;padding:20px;margin:18px 0}
.filters{display:flex;flex-wrap:wrap;gap:12px;align-items:end}label{display:block;font-weight:600;font-size:13px}
input,select,textarea,button{font:inherit}input[type=search]{min-width:240px}input,select,textarea{padding:8px;border:1px solid #aabbbb;border-radius:5px;max-width:100%}
button{padding:8px 12px;border:1px solid var(--accent);border-radius:5px;color:var(--accent);background:white;cursor:pointer}button:hover{background:#e9f2f0}button:disabled{opacity:.4;cursor:default}
.toolbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:12px 0}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;text-align:left}
th,td{padding:9px;border-bottom:1px solid var(--line);vertical-align:top;font-size:13px}th{background:#edf3f1}.flags{max-width:240px;overflow-wrap:anywhere}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}.story{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f8f7;padding:16px;border-radius:6px;max-height:570px;overflow:auto;margin:8px 0}
.current{outline:2px solid var(--accent)}.stage-card{padding:12px;border-radius:6px}.notes{display:grid;grid-template-columns:180px 1fr;gap:14px}.notes textarea{width:100%;min-height:100px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}.tag{display:inline-block;background:#fff0d7;padding:2px 7px;border-radius:4px;margin:2px;font-size:12px}
@media(max-width:800px){main{padding:14px}.pair,.notes{grid-template-columns:1fr}.filters>div{flex:1 1 180px}input[type=search]{min-width:0;width:100%}}
</style></head><body><main>
<h1>Generation review</h1>
<p>Inspect the original stories and their paired stages. Flags are screening suggestions; all stories remain in the dataset.</p>
<p class="muted">Review notes do not remove or edit stories and do not change sampling, annotation or evaluation.</p>
<section class="card"><p id="totals"></p><p id="stops" class="muted"></p>
<div class="filters">
<div><label for="search">Search text or metadata</label><input id="search" type="search" placeholder="Story text, pair, run, child…"></div>
<div><label for="collection">Collection / generator</label><select id="collection"></select></div>
<div><label for="stage">Stage</label><select id="stage"></select></div>
<div><label for="cohort">Age cohort</label><select id="cohort"></select></div>
<div><label for="flag">Review flag</label><select id="flag"></select></div>
<div><label for="statusFilter">Review status</label><select id="statusFilter"><option value="">All statuses</option>pending</option><option>checked</option><option>needs discussion</option></select></div>
<div><label><input id="flaggedOnly" type="checkbox" checked> Flagged stories only</label></div>
</div>
<div class="toolbar"><button id="download">Download review notes (CSV)</button><span class="muted">Notes are saved in this browser when available. Download them to keep a copy with the run.</span></div>
</section>
<section class="card"><div class="toolbar"><span id="count"></span><button id="previous">Previous</button><button id="next">Next</button></div>
<div class="table-wrap"><table><thead><tr><th>Collection / model</th><th>Age</th><th>Stage</th><th>Pair / replicate / story</th><th>Flags</th><th>Status</th><th></th></tr></thead><tbody id="rows"></tbody></table></div>
</section>
<section id="detail" class="card" hidden><h2 id="detailTitle"></h2><p id="detailMeta" class="muted"></p><div id="detailFlags"></div>
<div class="pair"><article id="firstCard" class="stage-card"><strong>Stage 1 — TD-conditioned</strong><div id="firstText" class="story"></div></article>
<article id="secondCard" class="stage-card"><strong>Stage 2 — SLI-conditioned</strong><div id="secondText" class="story"></div></article></div>
<p id="detailStop" class="muted"></p><details><summary>Flag evidence and source metadata</summary><pre id="evidence"></pre></details>
<div class="notes"><div><label for="reviewStatus">Review status for selected story</label><select id="reviewStatus"><option>pending</option><option>checked</option><option>needs discussion</option></select></div>
<div><label for="notes">Review notes for selected story</label><textarea id="notes" placeholder="Record what needs checking. These notes do not change analysis eligibility."></textarea></div></div>
<p id="saveMessage" class="muted" aria-live="polite"></p>
</section>
<p class="muted">Word and phrase screening ignores punctuation and letter case. Thresholds are review heuristics, not clinical norms. A recorded token-limit stop is distinct from a natural stop exactly at the limit.</p>
</main><script id="review-data" type="application/json">__REVIEW_DATA__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('review-data').textContent), stories=data.stories;
const byId=new Map(stories.map(r=>[r.review_id,r])), $=id=>document.getElementById(id);
const storageKey='icl-pilot-review-'+data.review_set_id;
let notes={},page=0,selected=null,filtered=[],storageAvailable=true;
try{notes=JSON.parse(localStorage.getItem(storageKey)||'{}')}catch(e){storageAvailable=false}
if(!notes||typeof notes!=='object'||Array.isArray(notes))notes={};
const age=r=>r.cohort_label||r.cohort||String(r.metadata.cohort_year||'unknown');
const state=r=>notes[r.review_id]||{status:'pending',notes:''};
const stageName=r=>r.stage==='stage1_td'?'Stage 1':'Stage 2';
const cell=(row,value)=>{const td=document.createElement('td');td.textContent=value;row.append(td);return td};
function options(id,values,label){const first=document.createElement('option');first.value='';first.textContent=label;$(id).append(first);
 for(const v of [...new Set(values)].sort()){const o=document.createElement('option');o.value=v;o.textContent=v;$(id).append(o)}}
options('collection',stories.map(r=>r.input_label),'All collections');options('stage',stories.map(r=>r.stage),'Both stages');
options('cohort',stories.map(age),'All ages');options('flag',stories.flatMap(r=>r.flags),'All flags');
$('totals').textContent=`${data.summary.stories.toLocaleString()} stories · ${data.summary.jobs.toLocaleString()} paired generation jobs · ${data.summary.flagged_stories.toLocaleString()} flagged stories · 0 exclusions`;
$('stops').textContent=`Recorded stopping reasons: ${data.summary.stories_with_recorded_stopping_reasons.toLocaleString()} of ${data.summary.stories.toLocaleString()} stories. Missing stopping information is shown as unknown.`;
const searchText=new Map(stories.map(r=>[r.review_id,(JSON.stringify(r.metadata)+' '+r.input_label+' '+r.story_text).toLowerCase()]));
function render(){const query=$('search').value.toLowerCase();filtered=stories.filter(r=>
 (!$('flaggedOnly').checked||r.flags.length)&&(!$('collection').value||r.input_label===$('collection').value)&&
 (!$('stage').value||r.stage===$('stage').value)&&(!$('cohort').value||age(r)===$('cohort').value)&&
 (!$('flag').value||r.flags.includes($('flag').value))&&(!$('statusFilter').value||state(r).status===$('statusFilter').value)&&
 (!query||searchText.get(r.review_id).includes(query)));
 page=Math.min(page,Math.max(0,Math.ceil(filtered.length/50)-1));$('rows').replaceChildren();
 for(const r of filtered.slice(page*50,page*50+50)){const tr=document.createElement('tr');cell(tr,r.input_label+(r.model_key?' / '+r.model_key:''));cell(tr,age(r));cell(tr,stageName(r));
  cell(tr,[r.pair_id||'pair unknown',r.replicate_id?'rep '+r.replicate_id:'',r.target_story||'story unknown'].filter(Boolean).join(' / '));
  cell(tr,r.flags.join(', ')||'No screening flags').className='flags';cell(tr,state(r).status);
  const td=cell(tr,'');const button=document.createElement('button');button.textContent='View';button.onclick=()=>show(r.review_id);td.append(button);$('rows').append(tr)}
 $('count').textContent=filtered.length?`${filtered.length.toLocaleString()} matching stories · page ${page+1} of ${Math.ceil(filtered.length/50)}`:'No matching stories. Clear filters or turn off “Flagged stories only” to inspect all stories.';
 $('previous').disabled=page===0;$('next').disabled=(page+1)*50>=filtered.length;
}
function show(id){selected=byId.get(id);const other=byId.get(selected.paired_review_id),first=selected.stage==='stage1_td'?selected:other,second=selected.stage==='stage2_sli'?selected:other;
 $('detail').hidden=false;$('detailTitle').textContent=`${selected.input_label} · ${selected.target_story||'story'} · ${stageName(selected)} selected`;
 $('detailMeta').textContent=`${age(selected)} · ${selected.pair_id||'pair unknown'} · replicate ${selected.replicate_id||'unknown'} · run ${selected.run_id||'unknown'}`;
 $('firstText').textContent=first?first.story_text:'Paired output unavailable';$('secondText').textContent=second?second.story_text:'Paired output unavailable';
 $('firstCard').classList.toggle('current',selected.stage==='stage1_td');$('secondCard').classList.toggle('current',selected.stage==='stage2_sli');$('detailFlags').replaceChildren();
 for(const name of selected.flags){const span=document.createElement('span');span.className='tag';span.textContent=name;$('detailFlags').append(span)}
 $('detailStop').textContent=`Selected story: ${selected.stopping_metadata}; finish reason ${selected.finish_reason||'unknown'}; generated tokens ${selected.generated_token_count??'unknown'}; longest same-word run ${selected.max_word_run}.`;
 $('evidence').textContent=JSON.stringify({evidence:selected.evidence,source_csv:selected.source_csv,source_row:selected.source_row,metadata:selected.metadata},null,2);
 $('reviewStatus').value=state(selected).status;$('notes').value=state(selected).notes;$('saveMessage').textContent='';$('detail').scrollIntoView({behavior:'smooth',block:'start'});
}
function save(){if(!selected)return;notes[selected.review_id]={status:$('reviewStatus').value,notes:$('notes').value};
 try{localStorage.setItem(storageKey,JSON.stringify(notes))}catch(e){storageAvailable=false}
 $('saveMessage').textContent=storageAvailable?'Saved in this browser. Analysis inputs remain unchanged.':'Notes are held on this page. Download them before closing it.';render();}
for(const id of ['search','collection','stage','cohort','flag','statusFilter','flaggedOnly'])$(id).addEventListener(id==='search'?'input':'change',()=>{page=0;render()});
$('previous').onclick=()=>{page--;render()};$('next').onclick=()=>{page++;render()};$('reviewStatus').onchange=save;$('notes').oninput=save;
$('download').onclick=()=>{const quote=v=> '"'+String(v??'').replaceAll('"','""')+'"';
 const header=['review_id','input_label','source_csv','source_row','model_key','run_id','pair_id','replicate_id','target_story','stage','flags','review_status','review_notes'];
 const lines=[header,...stories.map(r=>[r.review_id,r.input_label,r.source_csv,r.source_row,r.model_key,r.run_id,r.pair_id,r.replicate_id,r.target_story,r.stage,r.flags.join('|'),state(r).status,state(r).notes])];
 const blob=new Blob(['\ufeff'+lines.map(row=>row.map(quote).join(',')).join('\r\n')],{type:'text/csv;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');
 a.href=url;a.download='generation_review_notes.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
render();
</script></body></html>'''
