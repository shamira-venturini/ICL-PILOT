"""Isolated, version-checked Batchalign 3 English UD worker (no resegmentation)."""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import random
import sys


def file_digest(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_worker(request_path, response_path):
    request = json.loads(request_path.read_text())
    config = request["config"]
    os.environ["STANZA_RESOURCES_DIR"] = request["model_dir"]
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import batchalign as ba
    import torch

    versions = {name: metadata.version(name) for name in
                ("batchalign", "stanza", "torch", "numpy", "transformers", "huggingface-hub")}
    for package, key in (("batchalign", "version"), ("stanza", "stanza_version"), ("torch", "torch_version")):
        if versions[package].split("+")[0] != config[key]:
            raise ValueError("Unexpected annotation runtime version: " + package)
    package_root = Path(ba.__file__).parent
    sources = {name: file_digest(package_root / name) for name in config["source_sha256"]}
    if sources != config["source_sha256"]:
        raise ValueError("Batchalign annotation source differs from the reviewed release")
    model_dir = Path(request["model_dir"])

    def model_inventory():
        if file_digest(model_dir / "resources.json") != config["resources"]["sha256"]:
            raise ValueError("Stanza catalog differs from the pinned resource manifest")
        result = {}
        for name, spec in config["models"].items():
            path = model_dir / name
            if file_digest(path, "md5") != spec["md5"] or file_digest(path) != spec["sha256"]:
                raise ValueError("Stanza model differs from the pinned English model: " + name)
            result[name] = {"sha256": file_digest(path), "md5": spec["md5"], "bytes": path.stat().st_size}
        return result

    models = model_inventory()
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["torch_threads"])

    class EnglishUDBackend(ba.StanzaBackend):
        def _lang_config(self, lang, langs, key):
            settings = super()._lang_config(lang, langs, key)
            # Stanza 1.15's default package can add non-UD processors. Pin only
            # the five processors required by the reviewed Batchalign backend.
            settings.update(package=None, download_method=None, model_dir=str(model_dir),
                            processors={"tokenize": "combined_nocharlm", "mwt": "gum",
                                        "pos": "combined_charlm", "lemma": "combined_nocharlm",
                                        "depparse": "combined_charlm"})
            return settings

    backend = EnglishUDBackend(lang="en", retokenize=False, device=config["device"])
    if model_inventory() != models:
        raise ValueError("Stanza resources changed while initializing annotation")
    cache = ba.CacheSpec(path=str(response_path.parent / "unused_batchalign_cache.sqlite"), policy=ba.CachePolicy.Bypass)
    pipeline = ba.recipes.morphotag(stanza_backend=backend, workers=config["workers"], cache=cache)
    paths = {item["relative_path"]: Path(item["path"]) for item in request["inputs"]}
    inputs = [ba.inputs.chat_from_path(path, source_id=name) for name, path in paths.items()]
    raw = Path(request["output_dir"])
    raw.mkdir(parents=True, exist_ok=False)
    completed = []

    def save(outcome):
        if outcome.is_failed:
            raise RuntimeError("Batchalign failed for " + outcome.source_id + ": " + str(outcome.error))
        name = outcome.source_id
        if name not in paths or name in completed:
            raise ValueError("Unexpected/duplicate Batchalign output identity")
        destination = raw / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        outcome.write(str(destination))
        completed.append(name)
        print("Morphotagged", len(completed), "of", len(paths), name, flush=True)

    pipeline.run(inputs, outcome_callback=save, retain_outcomes=False)
    if set(completed) != set(paths):
        raise ValueError("Incomplete Batchalign file coverage")
    if model_inventory() != models:
        raise ValueError("Stanza models changed during annotation")
    from batchalign import _core
    core_path = Path(_core.__file__)
    binary_files = sorted(core_path.parent.glob("*.so")) + sorted(core_path.parent.glob("*.pyd"))
    runtime = {"python": sys.version, "packages": versions, "source_sha256": sources,
               "engine_sha256": {p.name: file_digest(p) for p in binary_files},
               "model_files": models, "resources_sha256": config["resources"]["sha256"],
               "retokenize": False, "resegment": False, "device": config["device"],
               "workers": config["workers"], "cache_policy": "Bypass", "torch_threads": torch.get_num_threads(),
               "adapter": "EnglishUDBackend._lang_config: explicit UD-only processors; downloads disabled"}
    response_path.write_text(json.dumps({"schema_version": 1, "completed_files": completed, "runtime": runtime,
                                         "request_sha256": file_digest(request_path)}, indent=2) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: morphosyntax_worker.py REQUEST.json RESPONSE.json")
    run_worker(Path(sys.argv[1]), Path(sys.argv[2]))
