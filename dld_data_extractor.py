import os
import pandas as pd
import re

# --- CONFIGURATION ---
# Upload your 52 DLD .cha files to this folder in Colab
DATA_PATH = "dld_corpus/"


def clean_utterance(line):
    """Cleans CHAT codes from the speech line."""
    try:
        # Split at the first colon to remove *CHI:
        if ':' in line:
            text = line.split(':', 1)[1].strip()
        else:
            return ""

        # Remove codes in brackets [] and angle brackets <>
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'<.*?>', '', text)

        # Remove phonological fragments (&um, &er) and special chars
        text = re.sub(r'&[a-z]+', '', text)
        text = re.sub(r'[+\d\.]+', '', text)  # Remove +... or numbers
        text = text.replace("xxx", "")  # Remove unintelligible
        text = text.replace("yyy", "")
        text = text.replace("_", " ")  # Replace underscores with spaces

        # Remove punctuation for cleaner embedding
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()
    except:
        return ""


data_rows = []

print(f"Scanning files in {DATA_PATH}...")

for filename in os.listdir(DATA_PATH):
    if not filename.endswith(".cha"):
        continue

    filepath = os.path.join(DATA_PATH, filename)

    with open(filepath, "r", encoding="utf-8", errors='ignore') as f:
        lines = f.readlines()

    # Metadata placeholders
    corpus_name = "Unknown"
    child_id = "Unknown"
    age_string = "Unknown"  # Keeping raw string as requested
    diagnosis = "Unknown"

    current_gem = "General"
    current_transcript = []

    # 1. Extract Metadata from Headers
    for line in lines:
        # PARSE @ID
        # Example: @ID: eng|EisenbergGuo|CHI|3;06.|male|||Target_Child|||
        if line.startswith("@ID"):
            parts = line.split('|')
            if len(parts) > 3 and parts[2] == 'CHI':
                corpus_name = parts[1]  # Index 1 = Corpus (EisenbergGuo)
                child_id = parts[2]  # Index 2 = Speaker Code
                age_string = parts[3]  # Index 3 = Age (3;06.)

                # Sometimes diagnosis is in index 5 (ENNI), sometimes empty (Eisenberg)
                if len(parts) > 5 and parts[5].strip():
                    diagnosis = parts[5]

        # PARSE @Types (Alternative location for diagnosis)
        # Example: @Types: cross, pictures, SLI
        if line.startswith("@Types"):
            if "SLI" in line or "DLD" in line or "LI" in line:
                if diagnosis == "Unknown":
                    diagnosis = "DLD/SLI"  # Flag it if found here

    # 2. Extract Stories (Gems)
    for line in lines:
        # Detect new Story/Picture marker
        if line.startswith("@G:"):
            # SAVE PREVIOUS STORY
            if len(current_transcript) > 2:
                full_text = " ".join(current_transcript)
                data_rows.append({
                    "corpus": corpus_name,
                    "filename": filename,
                    "child_id": child_id,
                    "age_raw": age_string,
                    "diagnosis": diagnosis,
                    "story_id": current_gem,
                    "transcript": full_text
                })

            # START NEW STORY
            current_gem = line.replace("@G:", "").strip()
            current_transcript = []

        # Detect Child Speech
        elif line.startswith("*CHI:"):
            cleaned = clean_utterance(line)
            if cleaned:
                current_transcript.append(cleaned)

    # SAVE LAST STORY (End of file)
    if len(current_transcript) > 2:
        full_text = " ".join(current_transcript)
        data_rows.append({
            "corpus": corpus_name,
            "filename": filename,
            "child_id": child_id,
            "age_raw": age_string,
            "diagnosis": diagnosis,
            "story_id": current_gem,
            "transcript": full_text
        })

# --- CREATE DATAFRAME ---
df = pd.DataFrame(data_rows)

print(f"Processed {len(df)} total stories.")
print(df[['corpus', 'age_raw', 'story_id', 'transcript']].head())

# Save for the K-Means script
df.to_csv("dld_stories_cleaned.csv", index=False)