import os
import re
import pandas as pd
import random
import torch
from transformers import pipeline

# --- Configuration ---
# We will use Llama-3-8B-Instruct from Hugging Face
MODEL_TO_TEST = "meta-llama/Llama-3-8B-Instruct"

# Define the k-shot conditions for the pilot
K_SHOTS = [0, 1, 3, 5, 10]

# --- Model Loading ---
# This sets up the model pipeline. It will download the model the first time you run it.
print("Loading model... (this may take a few minutes)")
generator = pipeline(
    "text-generation",
    model=MODEL_TO_TEST,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)
print("Model loaded successfully.")


# --- Data Loading ---
def load_data(file_path):
    """Loads sentences from a TSV file into a list of dictionaries."""
    df = pd.read_csv(file_path, sep='\t')

    def get_keywords(sentence):
        words = re.findall(r'\b\w+\b', sentence.lower())
        stop_words = {'a', 'the', 'is', 'in', 'and', 'to', 'it'}
        return [word for word in words if word not in stop_words]

    records = []
    for index, row in df.iterrows():
        records.append({
            "id": f"item_{index}",
            "sentence": row['Sentence'],
            "keywords": get_keywords(row['Sentence'])
        })
    return records


# --- Prompting Logic for Llama-3 ---
# Llama-3 uses a specific chat template format. We must adhere to it.
def create_llama_prompt(instruction, examples, test_item_keywords):
    """Constructs the full prompt for Llama-3."""
    messages = [{"role": "system", "content": ""}]

    # Construct the user message with examples
    user_content = instruction + "\n\n---\n\n"
    for ex in examples:
        user_content += f"Keywords: {ex['keywords']}\nSentence: {ex['sentence']}\n\n"
    user_content += f"Keywords: {test_item_keywords}\nSentence:"

    messages.append({"role": "user", "content": user_content})
    return messages


# --- Main Execution ---
def run_pilot():
    """Executes the full pilot study using a Hugging Face model."""
    # Load the four mini-datasets (ensure they are uploaded to Colab)
    dld_examples = load_data("dld_pilot_example_pool.tsv")
    dld_test_set = load_data("dld_pilot_test_pool.tsv")
    td_examples = load_data("td_pilot_example_pool.tsv")
    td_test_set = load_data("td_pilot_test_pool.tsv")

    results = []
    arms = {
        "DLD": {"examples": dld_examples, "test": dld_test_set,
                "instruction": "Generate a single sentence like a four-year-old child with a developmental language disorder would, using the provided keywords."},
        "TD": {"examples": td_examples, "test": td_test_set,
               "instruction": "Generate a single sentence like a four-year-old child using the provided keywords."}
    }

    print("\n--- Starting Pilot Study ---")
    for arm_name, data in arms.items():
        print(f"\n--- Running Arm: {arm_name} ---")
        for k in K_SHOTS:
            print(f"  Running condition: k={k}")
            for test_item in data["test"]:
                # Construct the prompt using the Llama-3 format
                messages = create_llama_prompt(data["instruction"], random.sample(data["examples"], k) if k > 0 else [],
                                               test_item["keywords"])

                # --- Hugging Face API Call ---
                try:
                    outputs = generator(
                        messages,
                        max_new_tokens=40,
                        eos_token_id=generator.tokenizer.eos_token_id,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                    )
                    # Extract only the newly generated text
                    full_text = outputs[0]['generated_text']
                    # The generated text is the last message in the list
                    generated_text = full_text[-1]['content'].strip()

                except Exception as e:
                    print(f"    ERROR during generation: {e}")
                    generated_text = "GENERATION_ERROR"

                results.append({
                    "model": MODEL_TO_TEST,
                    "arm": arm_name,
                    "k_shot": k,
                    "test_item_id": test_item["id"],
                    "input_keywords": test_item["keywords"],
                    "target_sentence": test_item["sentence"],
                    "generated_sentence": generated_text
                })

    results_df = pd.DataFrame(results)
    results_df.to_csv("pilot_study_results_llama3.csv", index=False)
    print("\n--- Pilot Study Complete ---")
    print("Results saved to pilot_study_results_llama3.csv")


# --- Run the pilot ---
run_pilot()