"""Explicit model-free boundary predictor for metadata/formatting tests.

This fixture follows punctuation for predictable assertions. Production code
never uses it; learned-model integration is exercised separately.
"""
import json
from pathlib import Path

from icl_pilot.chat.segmentation import prepare_narrative, render_segmented_narrative


class StubSegmenter:
    def __init__(self, *args, **kwargs):
        self.calls = []

    def segment_many(self, texts, report_dir):
        self.calls.append(list(texts))
        results = []
        for text in texts:
            prepared = prepare_narrative(text)
            actions = [0] * len(prepared["words"])
            word = -1
            for piece in prepared["pieces"]:
                if piece["kind"] == "word":
                    word += 1
                elif piece["text"] in (".", "?", "!", "+...", "+/.", "+//.") and word >= 0:
                    actions[word] = {"?": 3, "!": 4}.get(piece["text"], 2)
            results.append(render_segmented_narrative(prepared, actions))
        metadata = {"status": "complete", "test_double": True, "narratives": len(texts)}
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=False)
        (report_dir / "segmentation_manifest.json").write_text(json.dumps(metadata))
        return results, metadata
