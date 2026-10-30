# Study models and runtime policy

The study includes exactly three instruction-tuned, text-only model families:

| Key | Checkpoint |
|---|---|
| llama3_1_8b | meta-llama/Llama-3.1-8B-Instruct |
| qwen3_8b | Qwen/Qwen3-8B |
| gemma2_9b | google/gemma-2-9b-it |

configs/generation/models.json pins every checkpoint/tokenizer revision and native template option. Qwen thinking is disabled. Do not substitute a community model conversion for the selected official checkpoint.

Use the native template for each model. Fold the same canonical system sentence into the user message before native formatting. Log canonical messages, actual model-facing messages, template options and native stop IDs. Each model uses its own exact Stage 1 output as the Stage 2 parent.

Both stages share temperature 0.8, top-p 0.9, top-k 0 and 640 new tokens. Record effective library/model generation settings, including any inherited controls, token counts and finish reasons. Do not assume identical effective decoding from only a short list of arguments.

Loading uses the configured 4-bit NF4 policy. Record actual device, compute dtype and any float16 fallback. Run one model per Colab GPU. Software pins are defined by the notebook/runtime configurations.

The reference-tokenizer audit verifies the actual prepared narratives. Across the 1,428 study stories, maxima are 598 Gemma tokens and 568 Llama/Qwen tokens. No study story reaches the 640 ceiling; this does not guarantee that generated stories will end before truncation.

The scientific comparison is fixed at these three models. Full GPU execution and manual output/segmentation review remain pending. No vision or additional generation-control condition is included.
