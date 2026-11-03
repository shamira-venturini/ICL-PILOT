"""Isolate SynthTextEval's dependencies from generation and annotation kernels."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

from .notebook import sha256

SYNTHTEXTEVAL_REVISION = "3d63a172418a985fd0d997e5c993f6a0390c294e"
TRANSFORMERS_VERSION = "4.57.6"
MODEL_REQUIREMENTS = (
    "transformers==4.57.6", "datasets==4.0.0", "sentence-transformers==5.2.2",
    "huggingface-hub==0.36.2", "accelerate>=0.26.0",
)


def _run(command, **kwargs):
    subprocess.run([str(part) for part in command], check=True, **kwargs)


def patch_toolkit(source):
    """Apply the same two compatibility repairs as the standalone notebook.

    Compare with the exact commit before editing; preserve unknown modifications.
    """
    source = Path(source)
    observed = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if observed != SYNTHTEXTEVAL_REVISION:
        raise ValueError("Unexpected SynthTextEval checkout revision; existing source preserved.")
    relative = "synthtexteval/eval/downstream/classify/utils.py"
    original = subprocess.check_output(["git", "-C", str(source), "show", "HEAD:" + relative], text=True)
    fixed = re.sub(r"tokenizer\s*\.\s*encode_plus\s*\(", "tokenizer(", original)
    fixed = re.sub(r"pad_to_max_length\s*=\s*True", 'padding="max_length"', fixed)
    target = source / relative
    if target.read_text() not in {original, fixed}:
        raise ValueError("Unreviewed SynthTextEval tokenizer changes; existing source preserved.")
    if fixed == original or "pad_to_max_length" in fixed or "encode_plus" in fixed:
        raise ValueError("Pinned SynthTextEval tokenizer compatibility patch no longer applies.")
    target.write_text(fixed)
    init = source / "synthtexteval/eval/__init__.py"
    if init.exists() and init.read_bytes() != b"":
        raise ValueError("Unexpected SynthTextEval packaging file; existing source preserved.")
    init.touch(exist_ok=True)
    return {relative: sha256(target), "synthtexteval/eval/__init__.py": sha256(init)}


def install_runtime(environment, *, toolkit=False):
    """Use a separate Python process and environment; no kernel restart needed."""
    if toolkit and sys.version_info < (3, 9):
        raise RuntimeError("The pinned Transformers evaluation runtime requires Python >=3.9.")
    environment = Path(environment).expanduser().resolve()
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    marker = environment / "icl_framework_runtime.json"
    if not python.is_file():
        if environment.exists() and any(environment.iterdir()):
            raise FileExistsError("Evaluation environment is nonempty without Python; preserved: " + str(environment))
    if not marker.is_file():
        # Colab provides a compatible CUDA torch. Local system packages can have
        # a different architecture from the selected notebook interpreter.
        command = [sys.executable, "-m", "venv"]
        if "google.colab" in sys.modules:
            command.append("--system-site-packages")
        _run(command + [environment])
    previous = json.loads(marker.read_text()) if marker.is_file() else {}
    if not previous:
        _run([python, "-m", "pip", "install", "--quiet", "numpy", "pandas", "scipy", "matplotlib"])
    source = environment / "synthtexteval-src"
    hashes = previous.get("patched_files", {})
    if toolkit:
        if not (source / ".git").exists():
            _run(["git", "clone", "https://github.com/kr-ramesh/synthtexteval.git", source])
            _run(["git", "-C", source, "checkout", "--detach", SYNTHTEXTEVAL_REVISION])
        hashes = patch_toolkit(source)
        if not previous.get("toolkit") or previous.get("model_requirements") != list(MODEL_REQUIREMENTS):
            # Resolve these together. Installing unconstrained upstream
            # requirements first can leave newer datasets/encoders requiring
            # Hub 1.x alongside Transformers 4.x, which requires Hub <1.
            _run([python, "-m", "pip", "install", "--quiet", "-r", source / "requirements.txt", *MODEL_REQUIREMENTS])
            _run([python, "-m", "pip", "install", "--quiet", "--no-deps", "-e", source])
    effective_toolkit = toolkit or previous.get("toolkit", False)
    settings = {"schema_version": 1, "python": str(python), "toolkit": effective_toolkit,
                "synthtexteval_revision": SYNTHTEXTEVAL_REVISION if effective_toolkit else None,
                "synthtexteval_source": str(source) if effective_toolkit else None,
                "transformers_version": TRANSFORMERS_VERSION if effective_toolkit else None,
                "model_requirements": list(MODEL_REQUIREMENTS) if effective_toolkit else [],
                "patched_files": hashes}
    # A new subprocess performs imports so the notebook cannot use stale modules.
    code = (
        "import json, platform\nfrom importlib import metadata\n"
        "packages = {}\n"
        "for name in ['numpy','pandas','scipy','matplotlib','torch','transformers','SynthTextEval','datasets','sentence-transformers']:\n"
        "    try: packages[name] = metadata.version(name)\n"
        "    except metadata.PackageNotFoundError: packages[name] = None\n"
        "import numpy, pandas, scipy\n"
        "print(json.dumps({'python': platform.python_version(), 'packages': packages}))\n"
    )
    if effective_toolkit:
        code += (
            "from pathlib import Path\nimport inspect, transformers\n"
            "from synthtexteval.eval.downstream.classify import utils\n"
            "from synthtexteval.eval.text_quality import metrics\n"
            f"assert transformers.__version__ == {TRANSFORMERS_VERSION!r}\n"
            f"assert Path({str(source)!r}).resolve() in Path(utils.__file__).resolve().parents\n"
            "text = inspect.getsource(utils.tokenize_data)\n"
            "assert 'encode_plus' not in text and 'pad_to_max_length' not in text\n"
            "assert getattr(metrics, 'QualEval', None) or getattr(metrics, 'TextQualityEval', None)\n"
        )
    result = subprocess.check_output([str(python), "-c", code], text=True)
    settings["verified_environment"] = json.loads(next(line for line in result.splitlines() if line.startswith('{"python":')))
    marker.write_text(json.dumps(settings, indent=2) + "\n")
    return python


def verify_runtime(python, *, toolkit=False):
    python = Path(python)
    marker = python.parent.parent / "icl_framework_runtime.json"
    if not python.is_file() or not marker.is_file():
        raise RuntimeError("Run the framework runtime setup cell first.")
    settings = json.loads(marker.read_text())
    if toolkit and not settings.get("toolkit"):
        raise RuntimeError("Set INSTALL_FRAMEWORK_TOOLKIT=True and rerun framework runtime setup before model inference.")
    if settings.get("toolkit"):
        source = Path(settings["synthtexteval_source"])
        if patch_toolkit(source) != settings["patched_files"]:
            raise ValueError("SynthTextEval source differs from the verified runtime.")
    return settings
