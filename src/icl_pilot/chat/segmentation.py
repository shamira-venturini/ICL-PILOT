"""Whole-narrative Batchalign boundaries, applied to original surface tokens.

The ML runtime runs separately. Importing this module never imports torch,
downloads models, changes random states, or segments existing CHAT transcripts.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs/segmentation/batchalign.json"
POLICY = "batchalign3-word-boundaries-v1"
_TOKEN = re.compile(
    r'\[[^\[\]]*\]|\(\s*(?:\.{1,3}|\d+(?:\.\d+)?)\s*\)'
    r'|\+(?:"/\.|"\.|"|//[.?]|/[.?]|\.{3}|\.)'
    r'|&[=*][^\s,;.!?]+|[.!?,;"“”<>]|[^\s.!?,;"“”\[\]<>]+'
)


def atomic_json(path: Path, value: dict) -> None:
    """Keep the isolated worker independent of the generation/pandas package."""
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=".segmentation-", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_segmentation_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {"schema_version": 1, "policy": POLICY, "package": "batchalign",
                "version": "0.10.7", "transformers_version": "4.57.1",
                "torch_version": "2.10.0", "numpy_version": "2.2.0",
                "source_repository": "https://github.com/TalkBank/batchalign",
                "source_revision": "c8cd2e0eb36e4cf5a51195f169d8c5bf2d30f029",
                "inference_source_sha256": "ef145ca88a73bca1e6c40c96f87756a244b742b184e0ccd533b66ea92c980cac",
                "model_id": "talkbank/CHATUtterance-en",
                "model_revision": "764ec3f762c2e24df2def8df98b5fe34940085c6",
                "terminators": {"2": ".", "3": "?", "4": "!"},
                "final_fragment_terminator": "."}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Segmentation configuration differs from the reviewed pinned implementation.")
    if config.get("device") not in {"cpu", "cuda"}:
        raise ValueError("Segmentation device must be cpu or cuda.")
    size = config.get("narratives_per_batch")
    if type(size) is not int or not 1 <= size <= 32:
        raise ValueError("Use 1–32 narratives per segmentation batch.")
    files = ["config.json", "pytorch_model.bin", "special_tokens_map.json", "tokenizer.json",
             "tokenizer_config.json", "vocab.txt"]
    if config.get("model_files") != files:
        raise ValueError("Unexpected segmentation model files.")
    return config


def prepare_narrative(text: str) -> dict:
    """Tokenize without linguistic rewriting; retain source character offsets."""
    if not isinstance(text, str) or not text.strip() or re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text):
        raise ValueError("Segmentation needs nonempty narrative text without control characters.")
    pieces, words, offsets = [], [], []
    cursor = 0
    for match in _TOKEN.finditer(text):
        if text[cursor:match.start()].strip():
            raise ValueError("Unsupported or malformed CHAT notation in generated narrative.")
        token = match.group()
        if token.startswith("[") or token.startswith(("&=", "&*")) or re.fullmatch(r"\([.\d\s]+\)", token):
            kind = "annotation"
        elif token.startswith("+"):
            kind = "control"
        elif token in '.!?,;"“”<>':
            kind = "punctuation"
        elif re.search(r"[\w]", token):
            kind = "word"
            words.append(token)
            offsets.append([match.start(), match.end()])
        else:
            raise ValueError("Unsupported non-word notation: " + token)
        pieces.append({"text": token, "kind": kind, "start": match.start(), "end": match.end()})
        cursor = match.end()
    if text[cursor:].strip() or not words:
        raise ValueError("Narrative has no segmentable words or has unmatched notation.")
    return {"text": text, "pieces": pieces, "words": words, "word_offsets": offsets}


def render_segmented_narrative(prepared: dict, actions: list[int]) -> dict:
    """Use Batchalign's final word actions; no punctuation-based boundaries.

    Sentence punctuation is replaced by predicted terminators. Quotes use inline
    Unicode characters; they do not create additional main tiers. Protected CHAT
    annotations remain anchored. A predicted split inside a scope is refused.
    """
    words = prepared["words"]
    if len(actions) != len(words) or any(type(a) is not int or a not in range(6) for a in actions):
        raise ValueError("Batchalign action/word mismatch; no reconstruction fallback is allowed.")
    groups, group, word_index = [], 0, -1
    for action in actions:
        groups.append(group)
        if action in (2, 3, 4):
            group += 1
    surfaces = [[] for _ in range(groups[-1] + 1)]
    scopes, opening_quote, changes = [], False, []
    for piece in prepared["pieces"]:
        token, kind = piece["text"], piece["kind"]
        if kind == "word":
            word_index += 1
            if (word_index and groups[word_index] != groups[word_index - 1]
                    and any(scope_start < word_index for scope_start in scopes)):
                raise ValueError("Predicted boundary falls inside a CHAT scope; review retained narrative.")
            target = groups[word_index]
            surfaces[target].append(token)
        else:
            # Opening scope/quote belongs with the following word; all other
            # notation stays with its preceding word, or the first word.
            next_index = min(word_index + 1, len(words) - 1)
            target = groups[max(word_index, 0)]
            replacement = token
            if token == "<":
                scopes.append(next_index)
                target = groups[next_index]
            elif token == ">":
                if not scopes:
                    raise ValueError("Unmatched CHAT scope in generated narrative.")
                scopes.pop()
            elif token in '.!?':
                replacement = ""
            elif token in ('"', '“', '”', '+"'):
                if token == '+"':
                    replacement, opening_quote = "“", True
                elif token in ('“', '”'):
                    replacement, opening_quote = token, token == '“'
                else:
                    opening_quote = not opening_quote
                    replacement = "“" if opening_quote else "”"
                if opening_quote:
                    target = groups[next_index]
            elif token == '+"/.':
                replacement = ""  # reporting-frame linker, not a spoken word
            elif token == '+".':
                replacement, opening_quote = "”", False
            elif kind == "control":
                # Explicit interruption/trailing-off notation has meaning beyond
                # punctuation. Do not silently erase it or force a model split.
                if word_index < 0 or (word_index < len(words) - 1 and groups[word_index] == groups[word_index + 1]):
                    raise ValueError("CHAT terminator conflicts with predicted boundary; review retained narrative.")
            if replacement != token:
                changes.append({"start": piece["start"], "end": piece["end"],
                                "before": token, "after": replacement})
            if replacement:
                surfaces[target].append(replacement)
    if scopes:
        raise ValueError("Unclosed CHAT scope in generated narrative.")

    utterances, boundaries, continued_quote = [], [], False
    for index, tokens in enumerate(surfaces):
        end = max(i for i, g in enumerate(groups) if g == index)
        terminator = {2: ".", 3: "?", 4: "!"}.get(actions[end], ".")
        last_content = next((i for i in range(len(tokens) - 1, -1, -1) if tokens[i] not in ("“", "”")), None)
        if last_content is not None and tokens[last_content].startswith(("+...", "+/", "+.")):
            terminator = tokens.pop(last_content)
        # CHAT inline quotation pairs must be complete in each main tier.
        # Close/reopen across a learned boundary without adding another tier
        # or any spoken words. Complete an unmatched opening only as logged
        # punctuation formatting; an unmatched closing is ambiguous and refused.
        quote_open = continued_quote
        if continued_quote:
            tokens.insert(0, "“")
            changes.append({"start": prepared["word_offsets"][groups.index(index)][0],
                            "end": prepared["word_offsets"][groups.index(index)][0],
                            "before": "", "after": "“", "kind": "quote_boundary_reopen"})
        for token in tokens[int(continued_quote):]:
            if token == "“":
                if quote_open:
                    raise ValueError("Nested or ambiguous generated quotation; review retained narrative.")
                quote_open = True
            elif token == "”":
                if not quote_open:
                    raise ValueError("Unmatched closing quotation; review retained narrative.")
                quote_open = False
        continued_quote = quote_open
        if quote_open:
            tokens.append("”")
            position = prepared["word_offsets"][end][1]
            changes.append({"start": position, "end": position, "before": "", "after": "”",
                            "kind": "quote_boundary_close" if index < len(surfaces) - 1 else "unclosed_source_quote_close"})
        # Retain an explicit CHAT trailing-off/interruption terminator when it
        # coincides with the selected boundary; it already includes punctuation.
        tokens.append(terminator)
        line = " ".join(tokens)
        line = re.sub(r"“\s+", "“", line)
        line = re.sub(r"\s+”", "”", line)
        line = re.sub(r"<\s+", "<", line)
        line = re.sub(r"\s+>", ">", line)
        utterances.append(line)
        boundaries.append({"after_word": end, "action": actions[end],
                           "terminator": tokens[-1],
                           "final_fragment_default": actions[end] not in (2, 3, 4) and terminator == "."})
    if prepare_narrative(" ".join(utterances))["words"] != words:
        raise ValueError("Segmented CHAT failed original-word preservation.")
    return {"utterances": utterances, "source_text_sha256": _hash(prepared["text"]),
            "model_words_sha256": _hash(json.dumps(words, ensure_ascii=False)),
            "word_count": len(words), "word_offsets": prepared["word_offsets"],
            "actions": actions, "boundaries": boundaries, "punctuation_changes": changes,
            "word_preservation": "original_tokens_in_original_order", "policy": POLICY}


class BatchalignSegmenter:
    """One isolated model invocation for all selected CSVs/cohorts/stages."""

    def __init__(self, python: str | Path | None = None, *, config_path: str | Path = DEFAULT_CONFIG):
        self.python = Path(python or ROOT / ".venv-segmentation" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python"))
        self.config_path = Path(config_path).resolve()
        self.config = load_segmentation_config(self.config_path)

    def segment_many(self, texts: list[str], report_dir: str | Path) -> tuple[list[dict], dict]:
        if not texts:
            raise ValueError("Select at least one narrative to segment.")
        prepared = [prepare_narrative(text) for text in texts]
        if not self.python.is_file():
            raise FileNotFoundError("Segmentation runtime missing. Run the notebook segmentation setup block: " + str(self.python))
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=False)
        request = report_dir / "request.json"
        response = report_dir / "response.json"
        atomic_json(request, {"schema_version": 1, "config": self.config,
                              "word_sequences": [p["words"] for p in prepared]})
        request_hash = hashlib.sha256(request.read_bytes()).hexdigest()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        command = [str(self.python), "-m", "icl_pilot.chat.segmentation", str(request), str(response)]
        manifest = {"schema_version": 1, "status": "running", "policy": POLICY,
                    "config": self.config, "config_sha256": hashlib.sha256(self.config_path.read_bytes()).hexdigest(),
                    "command": command, "narratives": len(texts), "request_sha256": request_hash,
                    "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        manifest_path = report_dir / "segmentation_manifest.json"
        atomic_json(manifest_path, manifest)
        try:
            with (report_dir / "runtime.log").open("w", encoding="utf-8") as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            payload = json.loads(response.read_text(encoding="utf-8"))
            if (hashlib.sha256(request.read_bytes()).hexdigest() != request_hash or
                    payload.get("request_sha256") != request_hash or payload.get("schema_version") != 1):
                raise ValueError("Segmentation request/response identity changed.")
            batches = payload["actions"]
            if len(batches) != len(prepared):
                raise ValueError("Batchalign returned incomplete narrative coverage.")
            results = []
            for index, (p, actions) in enumerate(zip(prepared, batches)):
                try:
                    results.append(render_segmented_narrative(p, actions))
                except (ValueError, TypeError) as error:
                    manifest["failed_narrative"] = {"runtime_request_index": index,
                        "prepared_text_sha256": _hash(p["text"]), "error": str(error)}
                    raise
            manifest.update(status="complete", runtime=payload["runtime"],
                            response_sha256=hashlib.sha256(response.read_bytes()).hexdigest())
            return results, manifest
        except BaseException as error:
            manifest.update(status="failed_or_interrupted", error=str(error))
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise RuntimeError("Segmentation failed; originals retained. Review " + str(report_dir)) from error
        finally:
            atomic_json(manifest_path, manifest)


def install_segmentation_runtime(environment: str | Path, config_path: str | Path = DEFAULT_CONFIG) -> Path:
    """Isolated runtime with CPU-only Linux torch by default; no ASR extras."""
    if sys.version_info < (3, 10):
        raise RuntimeError("Batchalign segmentation needs Python >=3.10 (helpers still support 3.8).")
    config = load_segmentation_config(config_path)
    environment = Path(environment).expanduser().resolve()
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(environment)], check=True)
    pip = [sys.executable, "-m", "pip", "--python", str(python), "install", "--no-cache-dir"]
    if sys.platform.startswith("linux") and config["device"] == "cpu":
        subprocess.run(pip + ["--index-url", "https://download.pytorch.org/whl/cpu",
                              "torch==" + config["torch_version"]], check=True)
    subprocess.run(pip + [
                    "batchalign==" + config["version"], "transformers==" + config["transformers_version"],
                    "huggingface_hub==0.36.0", "torch==" + config["torch_version"],
                    "numpy==" + config["numpy_version"]], check=True)
    return python


def _worker(request_path: Path, response_path: Path) -> None:
    """Pinned upstream inference only; no legacy sentence reconstruction."""
    import importlib.metadata as metadata
    import inspect
    from huggingface_hub import snapshot_download
    import torch
    from batchalign.backends.asr.chatwhisper import BertUtteranceModel

    request = json.loads(request_path.read_text(encoding="utf-8"))
    config = request["config"]
    # Validate the received config using the same checked policy as the parent.
    config_file = request_path.parent / "effective_config.json"
    atomic_json(config_file, config)
    load_segmentation_config(config_file)
    versions = {name: metadata.version(name) for name in
                ("batchalign", "transformers", "torch", "numpy", "huggingface_hub", "tokenizers")}
    if versions["batchalign"] != config["version"] or versions["transformers"] != config["transformers_version"]:
        raise ValueError("Installed segmentation versions do not match configuration.")
    if versions["torch"].split("+")[0] != config["torch_version"] or versions["numpy"] != config["numpy_version"]:
        raise ValueError("Installed segmentation torch/numpy versions do not match configuration.")
    if tuple(int(x) for x in versions["torch"].split(".")[:2]) < (2, 6):
        raise ValueError("Pinned BERT .bin weights require torch >=2.6 for safe loading.")
    source = Path(inspect.getfile(BertUtteranceModel))
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_hash != config["inference_source_sha256"]:
        raise ValueError("Installed Batchalign inference source does not match audited release.")
    snapshot = Path(snapshot_download(config["model_id"], revision=config["model_revision"],
                                     allow_patterns=config["model_files"]))
    file_hashes = {}
    for name in config["model_files"]:
        path = snapshot / name
        with path.open("rb") as handle:
            digest = hashlib.sha256()
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        file_hashes[name] = digest.hexdigest()
    model = BertUtteranceModel(str(snapshot), device=config["device"])
    actions, sequences = [], request["word_sequences"]
    size = config["narratives_per_batch"]
    for start in range(0, len(sequences), size):
        batch = sequences[start:start + size]
        predicted = model.predict_actions_batch(batch)
        if len(predicted) != len(batch):
            raise ValueError("Incomplete segmentation batch.")
        actions.extend(predicted)
        print("Segmented", len(actions), "of", len(sequences), "narratives", flush=True)
    runtime = {"python": sys.version, "packages": versions, "source_sha256": source_hash,
               "model_id": config["model_id"], "model_revision": config["model_revision"],
               "model_file_sha256": file_hashes, "device": str(model.device),
               "compute_dtype": str(next(model.model.parameters()).dtype),
               "torch_threads": torch.get_num_threads(),
               "gpu_model": torch.cuda.get_device_name(model.device) if model.device.type == "cuda" else None,
               "inference": "Batchalign BertUtteranceModel.predict_actions_batch",
               "legacy_cleanup": False, "stanza_fallback": False}
    atomic_json(response_path, {"schema_version": 1, "actions": actions, "runtime": runtime,
                               "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest()})


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python -m icl_pilot.chat.segmentation REQUEST.json RESPONSE.json")
    _worker(Path(sys.argv[1]), Path(sys.argv[2]))
