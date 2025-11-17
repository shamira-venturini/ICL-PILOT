import os
import re

# --- Configuration ---
DATA_DIR = "/Users/shamiraventurini/PycharmProjects/PythonProject/ICL-PILOT/data/Conti3"
OUTPUT_FILE = "/Users/shamiraventurini/PycharmProjects/PythonProject/ICL-PILOT/results/Conti3-sents/SLI_candidates.txt"

# --- CRITICAL ADDITION: List of Target Files ---
# The script will ONLY process .cha files with these base names.
TARGET_FILENAMES = {
    "040009", "040010", "040015", "040018", "040023_1", "040024", "040106",
    "040107_1", "040107", "040113", "040128", "040129", "040205", "040214",
    "040218", "040302", "040304", "040307", "040315", "040322", "040401",
    "040404", "040426", "040502", "040503", "040508", "040610", "040703",
    "040718", "040727", "040804", "040816", "040818", "040830", "040908",
    "040914", "041019", "041102", "041113",
}

# --- Regular Expressions ---
UTTERANCE_BLOCK_REGEX = re.compile(r"^\*CHI:.*?(?=\n\*|\n@|\Z)", re.MULTILINE | re.DOTALL)
HAS_NOUN_OR_PRONOUN_REGEX = re.compile(r"%mor:.*?(n[|]|pro[|])")
HAS_VERB_REGEX = re.compile(r"%mor:.*?(v[|]|v:|aux[|]|cop[|])")


# --- Main Program ---
def main():
    """
    Processes a SPECIFIC LIST of .cha files to extract usable sentences
    and their corresponding %mor tiers.
    """
    print("Starting extraction from TARGET files...")

    if not os.path.exists("../results/Bliss-sents"):
        os.makedirs("../results/Bliss-sents")

    usable_pairs = []
    processed_files_count = 0

    # Iterate through all files in the data directory
    for filename in os.listdir(DATA_DIR):
        # --- MODIFIED LOGIC: Check if the file is in our target list ---
        base_name, extension = os.path.splitext(filename)
        if extension == ".cha" and base_name in TARGET_FILENAMES:
            processed_files_count += 1
            file_path = os.path.join(DATA_DIR, filename)
            print(f"--> Processing target file: {filename}")

            try:
                with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
                    content = f.read()
            except Exception as e:
                print(f"  -> Could not read file {filename}: {e}")
                continue

            chi_blocks = UTTERANCE_BLOCK_REGEX.findall(content)

            for block in chi_blocks:
                has_subject = HAS_NOUN_OR_PRONOUN_REGEX.search(block)
                has_verb = HAS_VERB_REGEX.search(block)

                if has_subject and has_verb:
                    chi_line = ""
                    mor_line = ""
                    for line in block.split('\n'):
                        if line.startswith("*CHI:"):
                            chi_line = line
                        elif line.startswith("%mor:"):
                            mor_line = line

                    if chi_line and mor_line:
                        clean_sentence = chi_line.replace("*CHI:", "").strip()
                        clean_sentence = re.sub(r'\s*•\d+_\d+•\s*$', '', clean_sentence)
                        clean_mor = mor_line.replace("%mor:", "").strip()
                        usable_pairs.append((clean_sentence, clean_mor))

    # Write all found pairs to the output .tsv file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        out_f.write("Sentence\tMOR_Tier\n")
        for sentence, mor_tier in usable_pairs:
            out_f.write(f"{sentence}\t{mor_tier}\n")

    print("-" * 20)
    print(f"Extraction complete.")
    print(f"Processed {processed_files_count} out of {len(TARGET_FILENAMES)} target files.")
    print(f"Found {len(usable_pairs)} usable sentence-MOR pairs.")
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()