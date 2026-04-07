import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer, util


def clean_text(text):
    """
    Cleans text:
    1. Removes words starting with '0' (e.g., '0prep', '0subj')
    2. Removes characters '<' and '>'
    """
    # Remove words starting with 0 (looks for word boundary, then 0, then word characters)
    text = re.sub(r'\b0\w*\b', '', text)
    # Remove < and > characters
    text = re.sub(r'[<>]', '', text)
    # Clean up extra spaces left behind
    text = ' '.join(text.split())
    return text


def process_jsonl(input_file, output_file, threshold=0.9):
    # 1. Load the data_original
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    if not data:
        print("File is empty.")
        return

    # 2. Extract and clean sentences
    # We store cleaned sentences for embedding, but keep the original objects for output
    cleaned_sentences = [clean_text(item['input']) for item in data]

    # 3. Calculate Embeddings
    # 'all-MiniLM-L6-v2' is fast and efficient for similarity tasks
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(cleaned_sentences, convert_to_tensor=True)

    # 4. Compute Cosine Similarity and Deduplicate
    # We use a greedy approach: keep the first occurrence, discard others > 0.9 similar
    keep_indices = []
    discard_indices = set()

    for i in range(len(data)):
        if i in discard_indices:
            continue

        keep_indices.append(i)

        # Compare current sentence i with all subsequent sentences j
        if i < len(data) - 1:
            # Calculate cosine similarities for all remaining sentences at once
            cos_scores = util.cos_sim(embeddings[i], embeddings[i + 1:])[0]

            for rel_idx, score in enumerate(cos_scores):
                abs_idx = i + 1 + rel_idx
                if score > threshold:
                    discard_indices.add(abs_idx)

    # 5. Save the filtered results
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx in keep_indices:
            f.write(json.dumps(data[idx]) + '\n')

    print(f"Original lines: {len(data)}")
    print(f"Duplicates removed: {len(discard_indices)}")
    print(f"Final lines saved: {len(keep_indices)}")


if __name__ == "__main__":
    # Change these filenames to match your local files
    input_path = "/Users/shamiraventurini/PycharmProjects/CLAN-annotator/curated_examples/synthetic/[* s:r].jsonl"
    output_path = ("/Users/shamiraventurini/PycharmProjects/CLAN-annotator/curated_examples/synthetic/[* s:r].jsonl"
                   ".jsonl")

    process_jsonl(input_path, output_path)