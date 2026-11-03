"""Checksum-pinned native CLAN analysis tools for local Python and Colab."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import shutil
import ssl
import subprocess
import sys
from urllib.request import urlopen
import zipfile


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def load_config(path):
    config = json.loads(Path(path).read_text())
    if (config.get('schema_version') != 1 or config.get('language') != 'engu'
            or config.get('speaker') != '*CHI' or config.get('ipsyn') is not False
            or config.get('normative_comparisons') is not False
            or config.get('story_ids') != ['A1','A2','A3','B1','B2','B3']):
        raise ValueError('Use the study English-UD, CHI-only, six-story KIDEVAL configuration.')
    return config


def verify_runtime(runtime):
    record = json.loads(Path(runtime).read_text())
    for name, expected in record['files'].items():
        if not Path(name).is_file() or sha256(name) != expected:
            raise ValueError('CLAN runtime file changed: ' + name)
    return record


def install_clan(project_root, environment, config_path=None):
    root = Path(project_root).resolve()
    config = load_config(config_path or root / 'configs/features/kideval.json')
    environment = Path(environment).expanduser().resolve()
    receipt = environment / 'clan_runtime.json'
    if receipt.exists():
        existing = verify_runtime(receipt)
        if existing['archive_sha256'] != config['archive_sha256']:
            raise ValueError('Existing CLAN runtime belongs to another source version.')
        return receipt
    if sys.platform not in {'darwin', 'linux'}:
        raise RuntimeError('Use this native feature block on macOS or Linux/Colab.')
    environment.mkdir(parents=True, exist_ok=True)
    supplied = root / config['archive_path']
    archive = supplied if supplied.is_file() else environment / 'unix-clan.zip'
    if not archive.exists():
        temporary = archive.with_suffix('.downloading')
        try:
            import certifi
            context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            context = ssl.create_default_context()
        with urlopen(config['archive_url'], context=context, timeout=60) as source, temporary.open('wb') as target:
            shutil.copyfileobj(source, target)
        if sha256(temporary) != config['archive_sha256']:
            raise ValueError('CLAN source download differs from the pinned archive; download retained.')
        temporary.replace(archive)
    if sha256(archive) != config['archive_sha256']:
        raise ValueError('CLAN source archive checksum mismatch; existing archive preserved.')
    source = environment / 'source'
    # Expand only known archive members; preserve any conflicting earlier files.
    with zipfile.ZipFile(archive) as zipped:
        names = set()
        for member in zipped.infolist():
            relative = Path(member.filename)
            if relative.is_absolute() or '..' in relative.parts or '\\' in member.filename or member.filename in names:
                raise ValueError('Unsafe or duplicate CLAN archive member.')
            names.add(member.filename)
            if member.is_dir():
                continue
            target = source / relative
            if target.is_symlink() or any(p.is_symlink() for p in target.parents):
                raise ValueError('CLAN source destination contains a symlink.')
            data = zipped.read(member)
            if target.exists() and target.read_bytes() != data:
                # A failed build changes only obj/bin. Rebuilding these is safe;
                # modified source and language resources are never overwritten.
                if '/unix/obj/' not in member.filename and '/unix/bin/' not in member.filename:
                    raise ValueError('Existing modified CLAN source preserved: ' + str(target))
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    clan = source / 'unix-clan'
    compiler = shutil.which('clang++' if sys.platform == 'darwin' else 'g++')
    if not compiler or not shutil.which('make'):
        raise RuntimeError('CLAN feature extraction needs make and a C++ compiler (available in standard Colab).')
    flags = '-O2 -DUNX -Wno-deprecated-declarations -Wno-narrowing -Wno-write-strings'
    if sys.platform == 'darwin':
        flags += ' -DAPPLEUNX -fshort-wchar'
    command = ['make', '-B', '-j2', 'kideval', 'dss', 'CC=' + compiler, 'CFLAGS=' + flags]
    with (environment / 'build.log').open('w') as log:
        completed = subprocess.run(command, cwd=clan / 'src', stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError('CLAN build failed; see ' + str(environment / 'build.log'))
    binaries = {name: str(clan / 'unix/bin' / name) for name in ['kideval','dss']}
    resources = {str(p): sha256(p) for p in sorted((clan / 'lib').rglob('*')) if p.is_file()}
    files = {**resources, **{name: sha256(name) for name in binaries.values()},
             **{str(p): sha256(p) for p in sorted((clan / 'src').rglob('*')) if p.is_file()}}
    record = {'schema_version': 1, 'version': config['version'], 'archive': str(archive),
        'archive_sha256': config['archive_sha256'], 'binaries': binaries, 'lib': str(clan / 'lib'),
        'platform': platform.platform(), 'machine': platform.machine(),
        'compiler': subprocess.check_output([compiler, '--version'], text=True).splitlines()[0],
        'command': command, 'files': files}
    receipt.write_text(json.dumps(record, indent=2) + '\n')
    return receipt
