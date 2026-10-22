# Story Spine Prompt Pack

This file is the prompt-ready companion to the official ENNI picture sequences.

The goal is not to feed the model an adult-style story.
The goal is to feed it a **short, picture-grounded event spine** that preserves the canonical plot while still leaving room for child-like realization.

## Source Of Truth

- Canonical story spine JSON: `phase2/ENNI/measures/story_units/story_unit_reference.json`
- Official ENNI picture sequences: `data/ENNI_images/` (for example `story-a1-slide.pdf`)
- Design note: `docs/background/image_based_story_task_design.md`
- Supporting narrative corpus: `phase2/ENNI/measures/story_units/story_unit_narratives.csv`

## How To Use It

1. Load the story spine for the target story.
2. Use the `units` list as the content plan.
3. Keep the order of the units fixed.
4. Allow surface variation in wording, but do not invent new plot events.
5. If you are doing the two-stage generation workflow, use the same spine for both stages:
   - stage 1: ordinary child-like realization
   - stage 2: DLD-like transformation

## Prompt Shell

```text
You are generating a child-style narrative from an ENNI picture story.

Use the canonical story spine below as the content plan.
Keep the order of events.
Write in short, child-appropriate clauses.
You may omit optional details, but do not add new major events.
Do not turn the spine into an adult-style polished retelling.

Story spine:
1. ...
2. ...
3. ...

Now write the narrative.
```

## Story Index

- `A1` - Ball in the pool
- `A2` - No running
- `A3` - Airplane in the pool
- `B1` - Sandcastle trouble
- `B2` - Picnic tummyache
- `B3` - Balloon trouble

## Notes

- The `units` are intentionally concise so they can function as prompt scaffolding.
- `content_invariants` are the things that should stay stable across retellings.
- `optional_content` are details that can be omitted or phrased differently.
- The file is suitable for prompt construction and for light-weight semantic alignment checks.
