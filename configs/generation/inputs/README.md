# Frozen generation exemplars

This folder accompanies `experiment.json`, the frozen roster, counterbalancing
rules, story scaffolds and prompt templates.

- `story_unit_narratives_chat_v2.csv`: 2,166 story records extracted from 361
  source CHAT files. It contains child IDs, literal corpus groups, ages, story
  IDs, utterance arrays and narrative text. The roster and counterbalancing rules
  determine which examples are used in each generation round.
- `story_unit_narratives_chat_v2.parsing_qc.json`: parser settings, source hashes,
  excluded-line records, missing metadata and the CSV SHA-256. It documents
  extraction and is not read by the generation model.

Both files were moved unchanged from `data/processed/generation_inputs/` on
16 October 2026. Their historical source paths remain provenance; those paths
need not exist to run generation from the CSV. Missing corpus metadata remain
missing, and unsupported records are not selected as prompt examples.

The notebook defaults to the checkout as `DATA_ROOT` in both Colab and Jupyter.
`experiment.json` specifies the CSV's path relative to that root. An explicit
external input root must contain the same `configs/generation/inputs/` hierarchy.

Treat these files as frozen research inputs. To compare a new extraction, write
to a fresh output path using the procedure in
[`docs/guides/chat_narrative_parsing.md`](../../../docs/guides/chat_narrative_parsing.md).
Do not overwrite this snapshot as part of an ordinary generation run.
