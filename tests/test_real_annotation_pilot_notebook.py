"""A real-data pilot must retain its selection and skip generation/segmentation."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


def notebook_cell(name):
    notebook = json.loads((ROOT / 'notebooks/colab_direct_generation_pipeline.ipynb').read_text())
    return ''.join(next(cell['source'] for cell in notebook['cells'] if cell['id'] == name))


def setup_fixture(tmp_path, monkeypatch):
    project = tmp_path / 'project'
    shutil.copytree(ROOT / 'configs', project / 'configs')
    files = {'ENNI.zip': b'fixture original archive: no real corpus redistributed'}
    children = []
    for group in ('TD', 'SLI'):
        for index in range(3):
            name = 'chat/' + group + '/' + str(index) + '.cha'
            data = ('@UTF8\n@Begin\n*CHI:\tfixture ' + str(index) + ' .\n@End\n').encode()
            files[name] = data
            children.append(dict(path=name, Group=group, File_ID=group + str(index)))
    manifest = dict(children=children, child_count=6, narrative_count=36,
                    clinical_counts={'TD': 3, 'SLI': 3},
                    files={name: hashlib.sha256(data).hexdigest() for name, data in files.items()})
    manifest_bytes = json.dumps(manifest).encode()
    bundle = tmp_path / 'pilot.zip'
    with zipfile.ZipFile(bundle, 'w') as z:
        for name, data in files.items():
            z.writestr(name, data)
        z.writestr('pilot_manifest.json', manifest_bytes)
    code = notebook_cell('settings')
    code = re.sub(r"RUN_PROFILE = ['\"]generation['\"]", "RUN_PROFILE = 'real_enni_pilot'", code, count=1)
    bundle_sha = re.search(r'PILOT_INPUT_ZIP.read_bytes\(\)\).hexdigest\(\) != "([a-f0-9]{64})"', code).group(1)
    manifest_sha = re.search(r'_manifest_bytes\).hexdigest\(\) != "([a-f0-9]{64})"', code).group(1)
    code = code.replace(bundle_sha, hashlib.sha256(bundle.read_bytes()).hexdigest())
    code = code.replace(manifest_sha, hashlib.sha256(manifest_bytes).hexdigest())
    monkeypatch.setenv('ICL_PILOT_ENNI_PILOT_ZIP', str(bundle))
    namespace = dict(Path=Path, os=os, sys=sys, REPO_DIR=project, IS_COLAB=False)
    return code, namespace, bundle


def test_real_pilot_marks_test_only_and_skips_generation_and_segmentation(tmp_path, monkeypatch):
    code, ns, bundle = setup_fixture(tmp_path, monkeypatch)
    exec(code, ns)
    assert ns['MODE'] == 'from_chat' and ns['REAL_DATA_PILOT'] is True
    assert ns['RUN_GENERATION'] is ns['RUN_CHAT_CONVERSION'] is False
    assert ns['RUN_ANNOTATION'] is ns['RUN_MORPHOSYNTAX'] is True
    assert ns['SAVE_TO_DRIVE'] is False
    assert len(list(ns['CHAT_INPUT_DIRS']['real_enni'].rglob('*.cha'))) == 6
    exec(notebook_cell('common-setup'), ns)
    manifest = json.loads((ns['RUN_OUTPUT_DIR'] / 'pipeline_manifest.json').read_text())
    assert manifest['test_only'] is True
    assert manifest['stages']['generation']['status'] == manifest['stages']['segmentation']['status'] == 'skipped'
    assert manifest['stages']['real_annotation_pilot']['input_children'] == 6


@pytest.mark.parametrize('mode', ['changed_input', 'extra_file', 'changed_bundle'])
def test_pilot_refuses_changed_or_extra_inputs_without_overwriting_them(tmp_path, monkeypatch, mode):
    code, ns, bundle = setup_fixture(tmp_path, monkeypatch)
    exec(code, ns)
    root = ns['pilot_input_root']
    if mode == 'changed_input':
        target = root / 'chat/TD/0.cha'
        target.write_text('changed input must remain available')
    elif mode == 'extra_file':
        target = root / 'chat/extra.cha'
        target.write_text('unexpected extra transcript')
    else:
        target = bundle
        target.write_bytes(target.read_bytes() + b'changed package')
    before = target.read_bytes()
    with pytest.raises(ValueError):
        exec(code, ns)
    assert target.read_bytes() == before


def test_repeated_settings_verify_existing_input_bytes(tmp_path, monkeypatch):
    code, ns, bundle = setup_fixture(tmp_path, monkeypatch)
    exec(code, ns)
    root = ns['pilot_input_root']
    before = {str(path): path.read_bytes() for path in root.rglob('*') if path.is_file()}
    exec(code, ns)
    assert before == {str(path): path.read_bytes() for path in root.rglob('*') if path.is_file()}
