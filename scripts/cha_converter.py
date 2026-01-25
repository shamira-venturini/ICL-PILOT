import pandas as pd
import os
import spacy
import re

# Load Spacy
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess

    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# --- CONFIGURATION ---
INPUT_DIR = "/Users/shamiraventurini/PycharmProjects/ICL-PILOT/synthetic_data/ENNI_B1/csv"
OUTPUT_DIR = "/Users/shamiraventurini/PycharmProjects/ICL-PILOT/synthetic_data/ENNI_B1/cha2"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_cha_header(filename):
    return f"""@UTF8
@Begin
@Languages:	eng
@Participants:	CHI Target_Child
@ID:	eng|SyntheticDLD|CHI|5;00.|male|DLD||Target_Child|||
@Transcriber:	Llama-3-Simulator
@Comment:	Synthetic data generated via Few-Shot Prompting
"""


def clean_text_for_chat(text):
    if not isinstance(text, str): return ""

    # 1. Remove newlines/tabs (Critical for CLAN)
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

    # 2. Remove Double Quotes
    text = text.replace('"', '')

    # 3. Remove Commas
    text = text.replace(',', ' ')

    # 4. Smart Remove Single Quotes
    text = re.sub(r"(^|\W)'|'(\W|$)", r"\1\2", text)

    # 5. Fix "Lil" -> "Little"
    text = re.sub(r'\bLil\b', 'Little', text, flags=re.IGNORECASE)
    text = re.sub(r'\blil\b', 'little', text, flags=re.IGNORECASE)

    # 6. Remove illegal characters
    text = re.sub(r'[^\w\s\.\?\!]', '', text)

    # 7. Collapse spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# --- MAIN LOOP ---
print(f"Processing CSVs from: {INPUT_DIR}")
print(f"Saving CHA files to:  {OUTPUT_DIR}")

for filename in os.listdir(INPUT_DIR):
    if filename.endswith(".csv"):
        file_path = os.path.join(INPUT_DIR, filename)
        print(f"Processing: {filename}...")

        try:
            df = pd.read_csv(file_path)
            col_name = 'output_story' if 'output_story' in df.columns else 'transcript'
            if col_name not in df.columns: continue

            for index, row in df.iterrows():
                story_text = row[col_name]
                clean_story = clean_text_for_chat(story_text)

                # Use short filename
                base_name = os.path.splitext(filename)[0]
                file_id = f"{base_name}_{index + 1:03d}"

                doc = nlp(clean_story)
                cha_content = create_cha_header(file_id)

                for sent in doc.sents:
                    utt = sent.text.strip()
                    if utt:
                        # --- PUNCTUATION LOGIC ---

                        # 1. Handle Ellipsis (...)
                        if "..." in utt:
                            # Check if it is at the VERY END of the sentence
                            if utt.endswith("..."):
                                # It's a trailing off -> +...
                                utt = utt[:-3].strip() + " +..."
                            else:
                                # It's in the middle -> (...)
                                # This fixes the "Redundant delimiter" error
                                utt = utt.replace("...", " (...) ")

                        # 2. Handle Standard Terminators
                        # If we didn't already add +..., check for ? or ! or .
                        if not utt.endswith("+..."):
                            if utt.endswith("?"):
                                utt = utt[:-1].strip() + " ?"
                            elif utt.endswith("!"):
                                utt = utt[:-1].strip() + " !"
                            else:
                                # Default to period
                                utt = utt.rstrip(".").strip() + " ."

                        # 3. Final cleanup of spaces
                        utt = re.sub(r'\s+', ' ', utt)

                        cha_content += f"*CHI:\t{utt}\n"

                cha_content += "@End"

                with open(f"{OUTPUT_DIR}/{file_id}.cha", "w", encoding="utf-8") as f:
                    f.write(cha_content)

        except Exception as e:
            print(f" Error on {filename}: {e}")

print(f"\nDone! Files are in '{OUTPUT_DIR}/'.")