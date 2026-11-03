"""Recover the reviewed catalog/weights without relaxing integrity checks."""
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from icl_pilot.chat import morphosyntax as ud
from icl_pilot.chat.morphosyntax_worker import pin_catalog_refresh

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / 'configs/annotation/batchalign.json').read_text())


@pytest.mark.parametrize('target', ['catalog', 'model', 'model_path', 'revision'])
def test_mutable_or_mismatched_resource_locations_are_rejected(tmp_path, target):
    settings = config()
    if target == 'catalog':
        settings['resources']['url'] = settings['resources']['url'].replace(settings['resources']['revision'], 'main')
    elif target == 'revision':
        settings['models_revision'] = 'v1.15.0'
    else:
        spec = next(iter(settings['models'].values()))
        spec['url'] = spec['url'].replace(settings['models_revision'], 'v1.15.0') if target == 'model' else spec['url'] + '.other'
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(settings))
    with pytest.raises(ValueError):
        ud.load_morphosyntax_config(path)


def fixture_downloads(tmp_path):
    settings = config()
    payloads = {settings['resources']['url']: b'reviewed catalog'}
    settings['resources']['sha256'] = hashlib.sha256(b'reviewed catalog').hexdigest()
    for name, spec in settings['models'].items():
        data = name.encode()
        payloads[spec['url']] = data
        spec['sha256'] = hashlib.sha256(data).hexdigest()
        spec['md5'] = hashlib.md5(data).hexdigest()
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(settings))
    return settings, path, payloads


def test_download_uses_exact_revisions_recovers_partial_file_and_reuses_verified_cache(tmp_path, monkeypatch):
    settings, path, payloads = fixture_downloads(tmp_path)
    calls = []
    def fetch(url, **kwargs):
        calls.append(url)
        return io.BytesIO(payloads[url])
    monkeypatch.setattr(ud, 'urlopen', fetch)
    models = tmp_path / 'models'
    models.mkdir()
    (models / 'resources.json.downloading').write_bytes(b'incomplete previous download')
    assert ud.prepare_morphosyntax_models(models, path) == models
    assert set(calls) == set(payloads) and len(calls) == 9
    assert (models / 'resources.json').read_bytes() == b'reviewed catalog'
    assert not (models / 'resources.json.downloading').exists()
    ud.prepare_morphosyntax_models(models, path, download=False)
    assert len(calls) == 9


def test_mismatch_keeps_diagnostics_and_reports_actual_and_expected_hash(tmp_path, monkeypatch):
    settings, path, payloads = fixture_downloads(tmp_path)
    monkeypatch.setattr(ud, 'urlopen', lambda *args, **kwargs: io.BytesIO(b'changed upstream catalog'))
    models = tmp_path / 'models'
    with pytest.raises(ValueError) as error:
        ud.prepare_morphosyntax_models(models, path)
    assert settings['resources']['sha256'] in str(error.value)
    assert hashlib.sha256(b'changed upstream catalog').hexdigest() in str(error.value)
    assert settings['resources']['url'] in str(error.value)
    assert not (models / 'resources.json').exists()
    assert (models / 'resources.json.downloading').read_bytes() == b'changed upstream catalog'


def test_backend_startup_refresh_uses_same_immutable_catalog():
    settings = config()
    common = SimpleNamespace(DEFAULT_RESOURCES_VERSION='1.15.0', DEFAULT_RESOURCES_URL='moving main')
    stanza = SimpleNamespace(resources=SimpleNamespace(common=common))
    pin_catalog_refresh(stanza, settings)
    url = common.DEFAULT_RESOURCES_URL + '/resources_' + common.DEFAULT_RESOURCES_VERSION + '.json'
    assert url == settings['resources']['url']
    common.DEFAULT_RESOURCES_VERSION = 'unexpected version'
    with pytest.raises(ValueError, match='refresh version'):
        pin_catalog_refresh(stanza, settings)


@pytest.mark.parametrize('baseline', ['original', 'current', 'previous', 'unknown'])
def test_notebook_upgrades_old_bundle_without_overwriting_unreviewed_files(tmp_path, baseline):
    notebook = json.loads((ROOT / 'notebooks/colab_direct_generation_pipeline.ipynb').read_text())
    cell = ''.join(next(c['source'] for c in notebook['cells'] if c['id'] == 'morphosyntax-setup'))
    code = cell.split('# END REVIEWED UD RESOURCE UPGRADE\n', 1)[0]
    assignments = [node for node in ast.walk(ast.parse(code)) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == '_ud_updates' for target in node.targets)]
    updates = ast.literal_eval(assignments[0].value)
    project = tmp_path / 'project'
    shutil.copytree(ROOT / 'src', project / 'src', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(ROOT / 'configs', project / 'configs')
    for name, spec in updates.items():
        path = project / name
        data = path.read_text()
        replacements = None
        if baseline == 'previous' and spec.get('previous'):
            expected, replacements = next(iter(spec['previous'].items()))
        elif baseline not in {'current', 'previous'}:
            expected, replacements = spec['before'], spec['replacements']
        if replacements is not None:
            for previous, fixed in reversed(replacements):
                assert data.count(fixed) == 1
                data = data.replace(fixed, previous, 1)
            assert hashlib.sha256(data.encode()).hexdigest() == expected
        if baseline == 'unknown' and name.endswith('morphosyntax_worker.py'):
            data += '\n# Unreviewed local code.\n'
        path.write_text(data)
    before = {name: (project / name).read_bytes() for name in updates}
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'pipeline_manifest.json').write_text(json.dumps({'stages': {}, 'test_only': True}))
    setup = ('from pathlib import Path\nfrom icl_pilot.pipeline import record_stage\n'
             f'REPO_DIR = Path({str(project)!r})\nRUN_OUTPUT_DIR = Path({str(run)!r})\n'
             'RUN_MORPHOSYNTAX = True\nMORPHOSYNTAX_CONFIG_PATH = REPO_DIR / "configs/annotation/batchalign.json"\n')
    script = tmp_path / 'execute.py'
    script.write_text(setup + code)
    env = dict(os.environ, PYTHONPATH=str(project / 'src'), PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True)
    if baseline == 'unknown':
        assert result.returncode != 0
        assert all((project / name).read_bytes() == data for name, data in before.items())
    else:
        assert result.returncode == 0, result.stderr
        assert all(hashlib.sha256((project / name).read_bytes()).hexdigest() == spec['after'] for name, spec in updates.items())
        report = json.loads((run / 'morphosyntax_resource_policy.json').read_text())
        assert report['resources_sha256'] == config()['resources']['sha256']
        second = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True)
        assert second.returncode == 0, second.stderr
