"""Pinned native Chatter CLI for Colab/Linux and macOS; no model dependencies."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tarfile
import urllib.request

from ..generation.recovery import atomic_bytes

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs/validation/chatter.json"
VALIDATION_ARGS = ["--force", "--format", "json", "--tui-mode", "disable",
                   "--parser", "tree-sitter", "--jobs", "2"]


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_chatter_config(path=None) -> dict:
    path = Path(path or DEFAULT_CONFIG).resolve()
    config = json.loads(path.read_text())
    if (config.get("schema_version") != 1 or config.get("version") != "0.30.0"
            or config.get("validation_args") != VALIDATION_ARGS):
        raise ValueError("Use the pinned Chatter version with all required validation checks.")
    for asset in config["assets"].values():
        if not re.fullmatch(r"talkbank-chatter-[a-z0-9_-]+\.tar\.xz", asset["name"]):
            raise ValueError("Unexpected Chatter release asset name.")
        if any(not re.fullmatch(r"[0-9a-f]{64}", asset[key]) for key in ("archive_sha256", "binary_sha256")):
            raise ValueError("Pin Chatter archive and executable SHA-256 hashes.")
    return config


def verify_chatter(path, *, config_path=None) -> dict:
    path = Path(path).expanduser().resolve()
    config = load_chatter_config(config_path)
    digest = sha256_file(path)
    matches = [(key, asset) for key, asset in config["assets"].items() if asset["binary_sha256"] == digest]
    if not matches:
        raise ValueError(f"Chatter executable differs from the pinned release; preserved: {path}")
    result = subprocess.run([str(path), "--version"], capture_output=True, text=True, check=False, timeout=30)
    if result.returncode or not re.match(r"^chatter 0\.30\.0(?:\s|$)", result.stdout):
        raise RuntimeError("Pinned Chatter executable could not confirm its version.")
    key, asset = matches[0]
    return {"path": str(path), "version": config["version"], "version_stdout": result.stdout.strip(),
            "binary_sha256": digest, "platform_build": key, "asset": asset,
            "config_path": str(Path(config_path or DEFAULT_CONFIG).resolve()),
            "config_sha256": sha256_file(Path(config_path or DEFAULT_CONFIG)),
            "validation_args": config["validation_args"]}


def ensure_chatter(*, executable=None, config_path=None, install=True, cache_dir=None) -> dict:
    """Reuse a verified CLI or download only the checksum-pinned native asset."""
    config = load_chatter_config(config_path)
    explicit = executable or os.environ.get("ICL_PILOT_CHATTER")
    if explicit:
        return verify_chatter(explicit, config_path=config_path)
    system = platform.system().lower()
    machine = {"arm64": "aarch64", "amd64": "x86_64"}.get(platform.machine().lower(), platform.machine().lower())
    key = f"{system}-{machine}"
    if key not in config["assets"]:
        raise RuntimeError(f"No pinned Chatter CLI for {key}; supply a supported runtime. Validation cannot be skipped.")
    cache = Path(cache_dir or ROOT / "tools/chatter/runtime") / key / "chatter"
    local = ROOT / "tools/chatter/chatter"
    candidates = [cache]
    if system == "darwin":
        candidates.append(local)  # Existing audited Intel build can run under Rosetta too.
    located = shutil.which("chatter")
    if located:
        candidates.append(Path(located))
    for candidate in candidates:
        if candidate.is_file():
            return verify_chatter(candidate, config_path=config_path)
    if not install:
        raise FileNotFoundError("Pinned Chatter is required. Enable dependency installation or set CHATTER_EXECUTABLE to its path.")
    asset = config["assets"][key]
    url = f"https://github.com/TalkBank/chatter/releases/download/v{config['version']}/{asset['name']}"
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read(64 * 1024 * 1024 + 1)
    if len(data) > 64 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != asset["archive_sha256"]:
        raise ValueError("Chatter release archive failed its pinned checksum; nothing installed.")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:xz") as archive:
        members = [member for member in archive.getmembers() if member.isfile() and Path(member.name).name == "chatter"]
        if len(members) != 1:
            raise ValueError("Chatter archive must contain exactly one regular CLI executable.")
        with archive.extractfile(members[0]) as handle:
            binary = handle.read(128 * 1024 * 1024 + 1)
    if hashlib.sha256(binary).hexdigest() != asset["binary_sha256"]:
        raise ValueError("Chatter executable checksum failed; nothing installed.")
    if cache.exists():
        raise FileExistsError(f"Chatter destination appeared during installation; preserved: {cache}")
    atomic_bytes(cache, binary)
    cache.chmod(0o755)
    return verify_chatter(cache, config_path=config_path)
