from __future__ import annotations

import io
import json
import re
from pathlib import Path

from PIL import Image
from pypdf import PdfReader


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
_STORY_ID_RE = re.compile(r"([AB]\d)", re.IGNORECASE)


def _infer_story_id(path: Path) -> str:
    match = _STORY_ID_RE.search(path.stem)
    if not match:
        raise ValueError(
            f"Could not infer story id from {path.name}. Pass an explicit --story-id."
        )
    return match.group(1).upper()


def _load_page_images(pdf_path: Path) -> list[tuple[int, object]]:
    reader = PdfReader(str(pdf_path))
    page_images: list[tuple[int, object]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        for image in page.images:
            page_images.append((page_number, image))
    return page_images


def extract_story_panels(
    input_pdf: str,
    output_dir: str,
    story_id: str | None = None,
    rotate_degrees: int = 180,
) -> int:
    pdf_path = Path(input_pdf).expanduser().resolve()
    out_root = Path(output_dir).expanduser().resolve()
    resolved_story_id = (story_id or _infer_story_id(pdf_path)).upper()

    page_images = _load_page_images(pdf_path)
    if not page_images:
        raise ValueError(f"No embedded panel images found in {pdf_path}.")

    story_dir = out_root / resolved_story_id
    story_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    panel_index = 1
    for page_number, image in page_images:
        rendered = Image.open(io.BytesIO(image.data)).convert("RGB")
        if rotate_degrees % 360:
            rendered = rendered.rotate(rotate_degrees, expand=True)

        panel_name = f"panel_{panel_index:02d}.png"
        panel_path = story_dir / panel_name
        rendered.save(panel_path)

        manifest_rows.append(
            {
                "panel_index": panel_index,
                "page_number": page_number,
                "image_path": str(panel_path),
                "width": rendered.width,
                "height": rendered.height,
            }
        )
        panel_index += 1

    manifest = {
        "story_id": resolved_story_id,
        "source_pdf": str(pdf_path),
        "rotate_degrees": rotate_degrees,
        "panel_count": len(manifest_rows),
        "panels": manifest_rows,
    }
    manifest_path = story_dir / "panel_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Extracted {len(manifest_rows)} panels to {story_dir}")
    print(f"Wrote panel manifest to {manifest_path}")
    return 0


def _panel_entries(panel_dir: Path) -> list[dict[str, object]]:
    panels = sorted(
        path for path in panel_dir.iterdir() if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )
    if not panels:
        raise ValueError(f"No panel image files found in {panel_dir}.")

    entries: list[dict[str, object]] = []
    for panel_index, path in enumerate(panels, start=1):
        entries.append(
            {
                "panel_index": panel_index,
                "image_path": str(path.resolve()),
                "visible_entities": [],
                "core_event_candidates": [],
                "optional_detail_candidates": [],
                "ambiguity_candidates": [],
                "notes": "",
            }
        )
    return entries


def build_story_packet_template(
    panel_dir: str,
    output_json: str,
    story_id: str | None = None,
    source_pdf: str | None = None,
) -> int:
    panel_dir_path = Path(panel_dir).expanduser().resolve()
    output_path = Path(output_json).expanduser().resolve()
    resolved_story_id = (story_id or _infer_story_id(panel_dir_path)).upper()

    packet = {
        "story_id": resolved_story_id,
        "source_pdf": str(Path(source_pdf).expanduser().resolve()) if source_pdf else "",
        "panel_dir": str(panel_dir_path),
        "panels": _panel_entries(panel_dir_path),
        "event_spine": [
            {
                "event_id": "e01",
                "label": "",
                "panel_indices": [],
                "must_include": True,
                "notes": "",
            }
        ],
        "task_design": {
            "content_invariants": [],
            "optional_content": [],
            "degrees_of_freedom": {
                "coverage": [],
                "ordering": [],
                "reference_style": [],
                "lexicalization": [],
                "causal_explicitness": [],
                "affect_dialogue": [],
                "fluency": [],
            },
            "profile_conditioning_fields": [
                "age_band",
                "mlu_band",
                "lexical_diversity_band",
                "reference_stability",
                "morphology_vulnerability",
                "fluency_level",
                "coverage_level",
            ],
            "prompt_constraints": [
                "Preserve the picture-supported event structure.",
                "Allow omission of optional details.",
                "Generate a fresh telling rather than editing a fixed adult script.",
            ],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote story packet template to {output_path}")
    return 0


def bootstrap_story_packets(
    input_dir: str,
    output_dir: str,
    rotate_degrees: int = 180,
    overwrite_templates: bool = False,
) -> int:
    input_dir_path = Path(input_dir).expanduser().resolve()
    output_dir_path = Path(output_dir).expanduser().resolve()
    pdf_paths = sorted(input_dir_path.glob("*.pdf"))
    if not pdf_paths:
        raise ValueError(f"No PDF files found in {input_dir_path}.")

    summary_rows: list[dict[str, object]] = []
    for pdf_path in pdf_paths:
        story_id = _infer_story_id(pdf_path)
        extract_story_panels(
            input_pdf=str(pdf_path),
            output_dir=str(output_dir_path),
            story_id=story_id,
            rotate_degrees=rotate_degrees,
        )

        template_path = output_dir_path / story_id / "story_packet.template.json"
        if overwrite_templates or not template_path.exists():
            build_story_packet_template(
                panel_dir=str(output_dir_path / story_id),
                output_json=str(template_path),
                story_id=story_id,
                source_pdf=str(pdf_path),
            )

        panel_manifest_path = output_dir_path / story_id / "panel_manifest.json"
        panel_manifest = json.loads(panel_manifest_path.read_text(encoding="utf-8"))
        summary_rows.append(
            {
                "story_id": story_id,
                "source_pdf": str(pdf_path),
                "panel_dir": str(output_dir_path / story_id),
                "panel_count": panel_manifest["panel_count"],
                "template_json": str(template_path),
            }
        )

    summary = {
        "input_dir": str(input_dir_path),
        "output_dir": str(output_dir_path),
        "rotate_degrees": rotate_degrees,
        "stories": summary_rows,
    }
    summary_path = output_dir_path / "story_packet_bootstrap_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote bootstrap summary to {summary_path}")
    return 0
