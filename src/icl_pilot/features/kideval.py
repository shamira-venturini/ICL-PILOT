"""KIDEVAL English-UD features from final annotated six-story ENNI units."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import zipfile

import pandas as pd

from ..chat.morphosyntax import _records, _tier, _body
from ..chat.enni_metadata import completed_age_months
from ..chat import validation
from .dss_features import parse_dss_tbl_file
from .spreadsheetml_measures import _load_raw_frame
from .kideval_runtime import sha256, load_config, verify_runtime

STORIES = ('A1','A2','A3','B1','B2','B3')
CORE = ['Total_Utts','MLU_Utts','MLU_Words','MLU_Morphemes','MLU50_Utts',
        'MLU50_Words','MLU50_Morphemes','FREQ_types','FREQ_tokens','NDW_100',
        'VOCD_D_optimum_average','Verbs_Utt','retracing','repetition','DSS_Utts','DSS','mor_Words',
        '*-PRESP','in','on','*-PL','*-POSS','u-cop','det:art','*-PAST','*-S3','irr-S3','u-aux','c-cop','c-aux','Total_non_zero_mors']


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + '\n', encoding='utf-8')


def _safe_member(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name:
        raise ValueError('Unsafe annotation member: ' + name)
    return path


def read_final_inputs(paths):
    """Read checked final CHAT directly from a UD manifest or exported ZIP."""
    loaded, references = [], {}
    for supplied in paths:
        path = Path(supplied).expanduser().resolve()
        references[str(path)] = sha256(path)
        zipped = zipfile.ZipFile(path) if zipfile.is_zipfile(path) else None
        try:
            if zipped:
                names = zipped.namelist()
                if len(names) != len(set(names)):
                    raise ValueError('Duplicate annotation ZIP members.')
                choices = [n for n in names if n.endswith('morphosyntax/morphosyntax_manifest.json')]
                if len(choices) != 1:
                    raise ValueError('Select a ZIP with one final morphosyntax manifest.')
                manifest_name = choices[0]
                m = json.loads(zipped.read(manifest_name))
                prefix = PurePosixPath(manifest_name).parent
                read = lambda name: zipped.read(str(_safe_member(name)))
                output_name = lambda item: str(prefix / _safe_member(m['annotated_output_directory']) / _safe_member(item['relative_path']))
            else:
                m = json.loads(path.read_text())
                read = lambda name: Path(name).read_bytes()
                output_name = lambda item: str(path.parent / _safe_member(m['annotated_output_directory']) / _safe_member(item['relative_path']))
            recovered = m.get('status') == 'postprocessing_complete'
            if m.get('status') != 'complete' and not recovered:
                raise ValueError('Feature inputs must be completed final UD outputs.')
            if recovered:
                output_name = lambda item: str(path.parent / _safe_member(item['path']))
            # Resolve per-file TalkTag coverage; never assume missing error
            # labels are error-free. Retention preserves this manifest.
            annotation = None
            test_only = m.get('test_only')
            auxiliary_zip = None
            if zipped:
                anns = [n for n in zipped.namelist() if n.endswith('annotation/annotation_manifest.json')]
                if len(anns) == 1:
                    annotation = json.loads(zipped.read(anns[0]))
            elif recovered and Path(m.get('source_archive', '')).is_file():
                original = Path(m['source_archive'])
                if sha256(original) != m['source_archive_sha256']:
                    raise ValueError('Recovery archive checksum mismatch.')
                references[str(original)] = sha256(original)
                auxiliary_zip = zipfile.ZipFile(original)
                anns = [n for n in auxiliary_zip.namelist() if n.endswith('annotation/annotation_manifest.json')]
                if len(anns) == 1:
                    annotation = json.loads(auxiliary_zip.read(anns[0]))
                pipelines = [n for n in auxiliary_zip.namelist() if n.endswith('pipeline_manifest.json')]
                if len(pipelines) == 1:
                    test_only = json.loads(auxiliary_zip.read(pipelines[0])).get('test_only')
            elif not zipped:
                explicit = m.get('upstream_annotation', {}).get('manifest')
                candidates = [Path(explicit)] if explicit and Path(explicit).is_file() else list(path.parent.parent.glob('annotation/**/annotation_manifest.json'))
                if len(candidates) == 1:
                    annotation = json.loads(candidates[0].read_text())
                    references[str(candidates[0])] = sha256(candidates[0])
            if auxiliary_zip:
                auxiliary_zip.close()
            coverage = {item['relative_path']: item.get('annotation_coverage', 'unknown')
                        for item in (annotation or {}).get('outputs', [])}
            for item in m['outputs']:
                name = output_name(item)
                payload = read(name)
                if hashlib.sha256(payload).hexdigest() != item['sha256']:
                    raise ValueError('Final annotated file checksum mismatch: ' + name)
                relative = item['path'][len('annotated/'):] if recovered else item['relative_path']
                if not zipped:
                    references[name] = item['sha256']
                loaded.append({'uri': str(path) + '::' + name if zipped else name,
                    'manifest': str(path), 'sha256': item['sha256'], 'text': payload.decode('utf-8-sig'),
                    'relative_path': relative, 'test_only': test_only,
                    'error_annotation_coverage': coverage.get(relative, 'unknown')})
        finally:
            if zipped:
                zipped.close()
    if not loaded:
        raise ValueError('Select at least one completed real/synthetic UD manifest or export ZIP.')
    return loaded, references


def story_sections(text):
    prefix, sections, active = [], {}, None
    begun = False
    for record in _records(text):
        tier = _tier(record)
        if tier == '@End':
            break
        if tier == '@G':
            begun = True
            candidate = _body(record)
            active = candidate if candidate in STORIES else None
            if active:
                if active in sections:
                    raise ValueError('Duplicate canonical story section: ' + active)
                sections[active] = []
        elif tier.startswith('*'):
            begun = True
        if active:
            sections[active].append(record)
        elif not begun:
            prefix.append(record)
    return prefix, sections


def assemble_units(sources, project_root):
    root = Path(project_root)
    generation = json.loads((root / 'configs/generation/experiment.json').read_text())
    people = pd.read_csv(root / generation['provenance']['participant_manifest'], dtype=str).fillna('')
    people = people.loc[people['Age(Month)'].astype(float).between(48,95)]
    by_id = people.set_index('File_ID').to_dict('index')
    by_pid = {row.source_pid: row.File_ID for row in people.itertuples() if row.source_pid}
    pairs = pd.read_csv(root / 'configs/generation' / generation['roster_csv'], dtype=str).set_index('pair_id').to_dict('index')
    catalog = json.loads((root / 'configs/generation/models.json').read_text())['models']
    units = {}
    for source in sources:
        records = _records(source['text'])
        comments = {}
        for record in records:
            if _tier(record) == '@Comment' and '=' in _body(record):
                key, value = _body(record).split('=', 1)
                if key in comments and comments[key] != value:
                    raise ValueError('Conflicting generation comment: ' + key)
                comments[key] = value
        ids = [_body(r).split('|') for r in records if _tier(r) == '@ID' and len(_body(r).split('|')) > 5 and _body(r).split('|')[2] == 'CHI']
        if len(ids) != 1:
            raise ValueError('Require one explicit CHI identity per annotated file.')
        age, group = ids[0][3], ids[0][5]
        prefix, sections = story_sections(source['text'])
        if comments.get('data_source') == 'generated':
            required = ['model_key','model_id','model_revision','run_id','bundle_id','pair_id','replicate_id','stage','source_child_id','target_story']
            if any(not comments.get(field) for field in required):
                raise ValueError('Synthetic CHAT lacks generation provenance.')
            model = comments['model_key']
            if model not in catalog or (comments['model_id'], comments['model_revision']) != (catalog[model]['id'], catalog[model]['revision']):
                raise ValueError('Synthetic checkpoint differs from the study catalogue.')
            stage = comments['stage']
            expected_group = {'stage1_td':'TD', 'stage2_sli':'SLI'}.get(stage)
            pair = pairs[comments['pair_id']]
            child = pair['td_child_id' if stage == 'stage1_td' else 'sli_child_id']
            if group != expected_group or child != comments['source_child_id'] or set(sections) != {comments['target_story']}:
                raise ValueError('Synthetic stage, child, group or story identity conflict.')
            key = ('synthetic', model, comments['run_id'], comments['bundle_id'], comments['pair_id'], comments['replicate_id'], stage)
            metadata = {k: comments.get(k, '') for k in ['model_key','model_id','model_revision','run_id','bundle_id','pair_id','replicate_id','stage','round_id']}
            metadata.update(source='synthetic', source_child_id=child,
                            File_ID=model + '_' + comments['run_id'] + '_' + comments['bundle_id'] + '_' + stage)
        else:
            pid = next((_body(r) for r in records if _tier(r) == '@PID'), '')
            child = by_pid.get(pid, Path(source['relative_path']).stem)
            key = ('real', child)
            metadata = {'source':'real', 'File_ID':child, 'source_child_id':child,
                        'model_key':'', 'stage':'real', 'run_id':'', 'bundle_id':'', 'pair_id':'', 'replicate_id':''}
        if child not in by_id or group != by_id[child]['Group'] or completed_age_months(age) != int(by_id[child]['Age(Month)']):
            raise ValueError('Annotated child age/group differs from the definitive ENNI metadata.')
        metadata.update(Group=group, Age_Raw=age, Age_Month=int(by_id[child]['Age(Month)']),
                        cohort_year=int(by_id[child]['Age(Month)']) // 12,
                        Sex=by_id[child].get('Sex', ''), Sex_CHI=ids[0][4])
        identity = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:24]
        unit = units.setdefault(identity, {'metadata':metadata, 'prefix':prefix, 'stories':{}, 'sources':[]})
        if unit['metadata'] != metadata:
            raise ValueError('Conflicting metadata within a six-story analysis unit.')
        for slot, body in sections.items():
            if slot in unit['stories']:
                raise ValueError('Duplicate child/model/bundle/story input: ' + str(key) + '/' + slot)
            unit['stories'][slot] = ''.join(body)
        selected_chi = sum(_tier(r) == '*CHI' for body in sections.values() for r in body)
        unit['sources'].append({**{k:v for k,v in source.items() if k != 'text'},
            'selected_child_utterances':selected_chi,
            'outside_canonical_child_utterances':sum(_tier(r) == '*CHI' for r in records) - selected_chi})
    return units


def extract_kideval(inputs, output_dir, *, project_root, runtime, validator=None):
    root, output = Path(project_root).resolve(), Path(output_dir).resolve()
    config = load_config(root / 'configs/features/kideval.json')
    native = verify_runtime(runtime)
    if native['archive_sha256'] != config['archive_sha256']:
        raise ValueError('CLAN runtime does not match the pinned feature configuration.')
    sources, references = read_final_inputs(inputs)
    units = assemble_units(sources, root)
    generation = json.loads((root / 'configs/generation/experiment.json').read_text())
    implementation = [Path(__file__), Path(__file__).with_name('kideval_runtime.py'),
        Path(__file__).with_name('dss_features.py'), Path(__file__).with_name('spreadsheetml_measures.py'),
        root / 'configs/features/kideval.json', root / 'configs/generation/experiment.json',
        root / 'configs/generation/models.json', root / 'configs/generation' / generation['roster_csv'],
        root / generation['provenance']['participant_manifest']]
    implementation_hashes = {str(p.resolve()):sha256(p) for p in implementation}
    output.mkdir(parents=True, exist_ok=False)
    bundles = output / 'bundles'
    bundles.mkdir()
    shutil.copyfile(runtime, output / 'clan_runtime.json')
    manifest = {'schema_version':1, 'status':'running', 'policy':config, 'inputs':references,
        'implementation':implementation_hashes,
        'python_version':platform.python_version(), 'pandas_version':pd.__version__,
        'runtime_manifest':str(runtime), 'runtime_sha256':sha256(runtime), 'units':[], 'commands':[]}
    _json(output / 'feature_manifest.json', manifest)
    try:
        inventory, complete = [], {}
        for identity, unit in sorted(units.items()):
            coverage = [s['error_annotation_coverage'] for s in unit['sources']]
            row = {**unit['metadata'], 'unit_id':identity, 'n_stories':len(unit['stories']),
                'complete_six_stories':set(unit['stories']) == set(STORIES),
                'TalkTag_Coverage':'complete' if all(v == 'complete' for v in coverage) else ('partial' if 'partial' in coverage else 'unknown'),
                'test_only':True if any(s['test_only'] is True for s in unit['sources']) else
                    (False if all(s['test_only'] is False for s in unit['sources']) else 'unknown')}
            row['feature_status'] = 'measured' if row['complete_six_stories'] else 'incomplete_bundle_missing_features'
            inventory.append(row)
            record = {'identity':identity, 'metadata':row, 'sources':unit['sources'],
                      'missing_stories':sorted(set(STORIES) - set(unit['stories']))}
            if row['complete_six_stories']:
                path = bundles / ('unit_' + identity + '.cha')
                # Canonical story order need not match recording order. These
                # derived analysis units have no audio; keep source timestamps
                # unchanged in the final annotations and remove only their
                # media links/bullets here, with explicit counts.
                prefix = [r for r in unit['prefix'] if _tier(r) != '@Media']
                text = ''.join(prefix) + ''.join(unit['stories'][s] for s in STORIES) + '@End\n'
                text, bullets = re.subn(r'\x15[^\x15]*\x15', '', text)
                path.write_text(text, encoding='utf-8')
                record.update(bundle_file=path.name, sha256=sha256(path), removed_audio_bullets=bullets,
                              removed_media_headers=len(unit['prefix']) - len(prefix), story_order=list(STORIES))
                complete[path.name] = row
            manifest['units'].append(record)
        pd.DataFrame(inventory).to_csv(output / 'unit_inventory.csv', index=False)
        measured = []
        if complete:
            selected = validator or validation.ensure_chatter(config_path=root / 'configs/validation/chatter.json')
            report = validation.validate_chat_directories({'bundles':bundles}, output / 'validation', validator=selected)
            manifest['chatter'] = validation.validation_reference(report)
            validation.require_validated_report(manifest['chatter'])
            # +0 is the native Unix file-input override. Without it, a notebook
            # subprocess is mistaken for a stdin pipeline, despite file arguments.
            command = [native['binaries']['kideval'], '+0', '+lengu', '+t*CHI', '+L' + native['lib'], 'unit_*.cha']
            manifest['commands'].append(command)
            with (output / 'kideval.log').open('w') as log:
                result = subprocess.run(command, cwd=bundles, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            reports = list(bundles.glob('*.kideval.xls'))
            if result.returncode or len(reports) != 1:
                raise RuntimeError('KIDEVAL failed or produced no unique report; see kideval.log.')
            raw, _, _ = _load_raw_frame(reports[0])
            raw.to_csv(output / 'kideval_all_columns.csv', index=False)
            rows = raw.loc[raw['File'].map(lambda v: Path(str(v)).name in complete)].copy()
            row_names = rows['File'].map(lambda v: Path(str(v)).name)
            if set(row_names) != set(complete) or row_names.duplicated().any():
                raise ValueError('KIDEVAL rows do not match all complete analysis units.')
            if any(c.startswith('IPSyn') for c in raw.columns) or not set(CORE) <= set(raw.columns):
                raise ValueError('Unexpected KIDEVAL English-UD output schema.')
            dss_command = [native['binaries']['dss'], '+0', '+lengu', '+t*CHI', '+L' + native['lib'], 'unit_*.cha']
            manifest['commands'].append(dss_command)
            with (output / 'dss.log').open('w') as log:
                dss = subprocess.run(dss_command, cwd=bundles, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            if dss.returncode:
                raise RuntimeError('DSS table extraction failed; see dss.log.')
            for raw_row in rows.to_dict('records'):
                name = Path(raw_row['File']).name
                row = {**complete[name]}
                for feature in CORE:
                    value = str(raw_row[feature]).strip()
                    row[feature] = None if value in {'','NA','N/A','.'} else float(value)
                table = bundles / name.replace('.cha', '.tbl.cex')
                if not table.is_file() and row['DSS_Utts']:
                    raise ValueError('Missing DSS category table: ' + name)
                dss_row = parse_dss_tbl_file(table, repo_root=output) if table.is_file() else {}
                for category in ['score','IP','PP','MV','SV','NG','CNJ','IR','WHQ']:
                    row['DSS_tbl_' + category] = None
                for key, value in dss_row.items():
                    if key in {'DSS_tbl_score','DSS_tbl_IP','DSS_tbl_PP','DSS_tbl_MV','DSS_tbl_SV','DSS_tbl_NG','DSS_tbl_CNJ','DSS_tbl_IR','DSS_tbl_WHQ'}:
                        row[key] = None if value in {'','NA','N/A'} else float(value)
                row['DSS_sample_eligible'] = row['DSS'] is not None
                if row['TalkTag_Coverage'] != 'complete':
                    row['DSS'] = row['DSS_tbl_score'] = None
                row['DSS_analysis_available'] = row['DSS'] is not None
                row['retracing_per_utt'] = row['retracing'] / row['Total_Utts'] if row['Total_Utts'] else None
                row['repetition_per_utt'] = row['repetition'] / row['Total_Utts'] if row['Total_Utts'] else None
                measured.append(row)
        # Incomplete units remain explicit rows, with missing numeric features.
        measured.extend(row for row in inventory if not row['complete_six_stories'])
        features = pd.DataFrame(measured).sort_values('unit_id')
        for column in CORE + ['DSS_tbl_' + c for c in ['score','IP','PP','MV','SV','NG','CNJ','IR','WHQ']] + [
                'DSS_sample_eligible','DSS_analysis_available','retracing_per_utt','repetition_per_utt']:
            if column not in features:
                features[column] = pd.NA
        features.to_csv(output / 'kideval_features.csv', index=False)
        for source in ('real','synthetic'):
            features.loc[features['source'].eq(source)].to_csv(output / (source + '_kideval_features.csv'), index=False)
        for name, expected in {**references, **implementation_hashes}.items():
            if sha256(name) != expected:
                raise ValueError('Feature source changed during extraction: ' + name)
        for record in manifest['units']:
            if record.get('bundle_file') and sha256(bundles / record['bundle_file']) != record['sha256']:
                raise ValueError('CLAN modified a bundled input transcript.')
        if sha256(runtime) != manifest['runtime_sha256']:
            raise ValueError('CLAN runtime receipt changed during extraction.')
        manifest.update(status='complete', rows=len(features), complete_units=len(complete),
            partial_or_unknown_error_annotation_units=int(features['TalkTag_Coverage'].ne('complete').sum()),
            ipsyn_computed=False, normative_comparisons=False, vocd_stochastic=True,
            excluded_analysis_columns=['Word_Errors','Utt_Errors','TD_Time_(secs)','TD_Words_Time','TD_Utts_Time'],
            outputs={p.relative_to(output).as_posix():sha256(p) for p in output.rglob('*') if p.is_file() and p.name != 'feature_manifest.json'})
    except BaseException as error:
        manifest.update(status='failed_or_interrupted', error=str(error))
        raise
    finally:
        _json(output / 'feature_manifest.json', manifest)
    return output / 'feature_manifest.json'
