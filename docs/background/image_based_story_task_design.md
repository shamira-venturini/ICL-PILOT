# Image-Based Story Task Design

This note records the repo-level workflow for moving ENNI prompting away from adult-style verbal retelling and toward picture-derived task packets.

## Why this exists

The pilot report showed that the direct prompting conditions produced outputs that were either:

- too homogeneous at low shot counts, or
- clinically exaggerated at higher shot counts.

One likely reason is task anchoring: a verbal retelling setup gives the model a fixed macrostructure and strongly biases surface realization. The image-based workflow here keeps the event spine grounded in the ENNI materials while leaving more room for child-like variation in discourse realization.

## Repo workflow

### 1. Extract panel images from an ENNI slide PDF

```bash
PYTHONPATH=src python -m icl_pilot extract-story-panels \
  data/ENNI_images/story-a1-slide.pdf \
  data/processed/story_packets
```

This writes:

- `data/processed/story_packets/A1/panel_01.png`
- `data/processed/story_packets/A1/panel_02.png`
- ...
- `data/processed/story_packets/A1/panel_manifest.json`

The default extraction rotates the embedded images by `180` degrees because the current ENNI slide PDFs store the panel image content upside down relative to the page view.

### 2. Create a story-packet template

```bash
PYTHONPATH=src python -m icl_pilot build-story-packet-template \
  data/processed/story_packets/A1 \
  data/processed/story_packets/A1/story_packet.template.json \
  --source-pdf data/ENNI_images/story-a1-slide.pdf
```

The resulting JSON is intended to be hand-completed during task design.

### 3. Bootstrap all story packets at once

```bash
PYTHONPATH=src python -m icl_pilot bootstrap-story-packets \
  data/ENNI_images \
  data/processed/story_packets
```

This scans all ENNI slide PDFs in `data/ENNI_images/`, extracts their panel images, writes one packet template per story, and records the result in:

- `data/processed/story_packets/story_packet_bootstrap_summary.json`

## Story-packet schema

Each packet is built around:

- `panels`: what is visibly present in each panel
- `event_spine`: the ordered event skeleton that should usually be recoverable
- `task_design.content_invariants`: content that should remain stable across retellings
- `task_design.optional_content`: content that may be mentioned or omitted
- `task_design.degrees_of_freedom`: where generation is allowed to vary naturally

Important fields to annotate:

- `visible_entities`
- `core_event_candidates`
- `optional_detail_candidates`
- `ambiguity_candidates`

## Intended prompting use

The packet supports a three-step prompting setup:

1. choose content from the picture-supported event space
2. build a child narrative plan from that content
3. realize the narrative from scratch under a child profile

This is intentionally different from:

- giving the model a polished adult story
- then asking it to make that story "DLD-like"

The goal is to preserve ENNI content while reducing verbal-template collapse.

The filled prompt-ready spine lives in `phase2/ENNI/measures/story_units/story_unit_reference.json`, with a short usage guide in `docs/background/story_spine_prompt_pack.md`.
