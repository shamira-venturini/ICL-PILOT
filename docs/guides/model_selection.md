# Model selection for the revised study

Status on 20 October 2026: the author requires official model sources and a
comparison that can run on one Colab GPU. The active configuration now uses
Meta's official Llama 3.1 8B Instruct, with an immutable revision. The comparator
and vision conditions below are recommendations awaiting a protocol decision.

## Official Llama correction

The previous configured ID was `avans06/Meta-Llama-3.2-8B-Instruct`, with no
requested revision. Its publisher describes removing the vision components and
cross-attention layers from Llama 3.2 11B Vision to produce a text model.
This is a community conversion. Meta's official Llama 3.2 release has 1B/3B
text models and 11B/90B vision models.
Sources: [conversion card](https://huggingface.co/avans06/Meta-Llama-3.2-8B-Instruct),
[Meta's release documentation](https://dev.meta.ai/llama/docs/model-cards-and-prompt-formats/llama3_2).

New runs use `meta-llama/Llama-3.1-8B-Instruct`, revision
`0e9e39f249a16976918f6564b8830bc894c89659`, checked against the public Hugging Face
metadata on 20 October. It is an official English-capable instruction-tuned
text model in the existing 8B size range. Its weights can be loaded through the
existing text-generation interface. Approved Hugging Face access and runtime
authentication are required. The revision is supplied to both model and tokenizer.
Sources: [official card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct),
[Llama technical report](https://arxiv.org/abs/2407.21783).

This changes the generator. Historical outputs retain their original attribution;
the current community repository revision does not recover the historical
installed weights. The correction is implemented in `configs/generation/experiment.json`.
It preserves the approved counterbalancing, child pairs, prompt text and decoding
settings. Colab loading and inference still require a fresh technical check.

## Proposed minimum comparison

| Role | Official checkpoint | Research justification |
| --- | --- | --- |
| Primary text model, configured | `meta-llama/Llama-3.1-8B-Instruct` | Documented official reference in the existing size range; publicly identifiable weights and instruction-following interface. |
| Proposed text comparator | `Qwen/Qwen3-8B`, with thinking disabled through its chat template | A separately trained model family at approximately the same parameter scale (8.2B), testing whether results extend beyond Llama. |
| Proposed vision pilot | `Qwen/Qwen3-VL-8B-Instruct` | Supports text and multiple images; suitable for testing ordered ENNI picture input, with a text-input control using the same checkpoint. |

Qwen3 requires an explicit `enable_thinking=False` setting for the proposed text
condition. The current notebook does not implement that adapter or a vision
loader. The VL model includes vision components and is not exactly parameter
matched to the text models. Sources:
[Qwen3 card](https://huggingface.co/Qwen/Qwen3-8B),
[Qwen3 report](https://arxiv.org/abs/2505.09388),
[Qwen3-VL card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct).

The selection criteria are official provenance, accessible weights, documented
versions, suitable generation interfaces, approximately comparable text-model
size, and measured feasibility on the available GPU. These justify a reproducible
comparison. Developmental or DLD fidelity remains an empirical question for the
study. General model benchmarks do not establish that fidelity.

## Vision: pilot before adding a full condition

Vision addresses the **Task-anchoring gap**: human ENNI narration starts from
pictures, whereas the current model receives a project-authored verbal scaffold.
The hypothesis is that picture input reduces verbal anchoring and changes content
or linguistic fidelity. That benefit has not been demonstrated for this project.

Use the same VL checkpoint for the control and image condition. A scaffold-only
versus scaffold-plus-pictures comparison estimates the effect of adding pictures
under a fixed verbal scaffold. Replacing the scaffold with pictures addresses
elicitation format, with both modality and content representation changing.
Choose the contrast explicitly. The latter more directly
examines dependence on the authored scaffold. Fix image order, resolution and
pixel/token budget, and specify which stages receive pictures. Keep exemplar
selection and the evaluation pipeline comparable. Stage 2 rewriting remains a
separate design issue.

The active `data/` folder contains no picture files. The pilot needs verified
copies of the ENNI picture sequences and recorded panel order/orientation.
The retained task-packet helpers are exploratory preparation tools; they do not
implement multimodal generation in the maintained notebook.

Qwen3-VL is the suggested pilot candidate. Meta's Llama 3.2 Vision model page
currently states that downloads are unavailable from the EU, which complicates
local/reviewer access. Source:
[official Llama Vision access notice](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct).

## Compute and comparison controls

Run models sequentially. Validate memory use and throughput with the actual Colab
GPU, prompt lengths and image budget before committing to the full design.
Four-bit loading is a candidate deployment policy; its settings and runtime must
be recorded consistently. A single GPU does not guarantee a particular VRAM size.

A proposed technical pilot could use one recorded pair per age cohort, all six
targets and two replicates: 48 story jobs / 96 stage outputs per condition, with
one replicate per assignment. It would require an explicit recorded selection;
the notebook's current prefix smoke test is not this balanced pilot. Use it to
check execution, truncation, formatting and resource cost, without selecting
models because they produce more errors or resemble held-out evaluation children.

For a subsequent full comparison, pre-specify shared prompts/exemplars, rounds,
replicates, sampling policy, comparable output budgets and annotation/evaluation
rules. Use each model's native chat template and pin model, tokenizer, processor
and software versions. Equal token limits can truncate models differently;
measure truncation. Decide how generation versus rewriting will be controlled
before scaling up. Keep model choice and tuning separate from held-out real-data
evaluation, and preserve pair/round/stage ancestry for every model condition.
