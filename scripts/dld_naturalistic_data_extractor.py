import os
import pandas as pd
import re

# --- CONFIGURATION ---
DATA_PATH = "/Users/shamiraventurini/PycharmProjects/ICL-PILOT/dld_narrative/"
OUTPUT_FILE = "../real_data/dld_stories_naturalistic.csv"


def smart_clean_utterance(line):
    """
    Cleans CHAT lines but PRESERVES disfluencies and mazes
    to simulate natural, messy speech.
    """
    if not line.startswith("*CHI:"):
        return None

    # Remove the speaker label
    text = line.split(':', 1)[1].strip()

    # 1. Handle Retracing [//] -> Replace with "..." to show a change of mind
    # Example: <he is> [//] the dog -> he is... the dog
    text = text.replace("[//]", "...")

    # 2. Handle Repetition [/] -> Just remove the code, keep the words
    # Example: <he is> [/] he is -> he is he is
    text = text.replace("[/]", "")

    # 3. Handle Fillers (&um, &uh) -> Remove the '&' but keep the sound
    text = re.sub(r'&([a-z]+)', r'\1', text)  # &um -> um

    # 4. Remove other bracketed codes (e.g., [*], [+ bch], [^ comment])
    text = re.sub(r'\[\^.*?\]', '', text)  # Remove comments first
    text = re.sub(r'\[.*?\]', '', text)  # Remove other codes

    # 5. Remove Angle Brackets < > (used for grouping mazes)
    text = text.replace("<", "").replace(">", "")

    # 6. Remove Unintelligible (xxx, yyy)
    text = text.replace("xxx", "").replace("yyy", "")

    # 7. Handle Pauses ( (.) or (..) ) -> Replace with commas or ...
    text = text.replace("(.)", ",").replace("(..)", "...")

    # 8. Clean up extra spaces created by removals
    text = re.sub(r'\s+', ' ', text).strip()

    # 9. Remove leading/trailing punctuation artifacts
    text = re.sub(r' \.', '.', text)
    text = re.sub(r' \?', '?', text)

    return text


# --- MAIN PROCESSING LOOP ---
data_rows = []

print(f"Scanning files in {DATA_PATH}...")

for filename in os.listdir(DATA_PATH):
    if not filename.endswith(".cha"): continue

    with open(os.path.join(DATA_PATH, filename), "r", encoding="utf-8", errors='ignore') as f:
        lines = f.readlines()

    # Metadata extraction (Simplified for brevity)
    age_raw = "0;0"
    corpus = "Unknown"
    current_gem = "General"
    current_transcript = []

    for line in lines:
        if line.startswith("@ID"):
            parts = line.split('|')
            if len(parts) > 3 and parts[2] == 'CHI':
                corpus = parts[1]
                age_raw = parts[3]

        if line.startswith("@G:"):
            # Save previous story
            if len(current_transcript) > 2:
                full_text = " ".join(current_transcript)
                data_rows.append({
                    "corpus": corpus,
                    "age_raw": age_raw,
                    "story_id": current_gem,
                    "transcript": full_text
                })
            current_gem = line.replace("@G:", "").strip()
            current_transcript = []

        elif line.startswith("*CHI:"):
            cleaned = smart_clean_utterance(line)
            if cleaned and len(cleaned) > 0:
                current_transcript.append(cleaned)

    # Save last story
    if len(current_transcript) > 2:
        full_text = " ".join(current_transcript)
        data_rows.append({
            "corpus": corpus,
            "age_raw": age_raw,
            "story_id": current_gem,
            "transcript": full_text
        })

# --- SAVE ---
df = pd.DataFrame(data_rows)
print(f"Extracted {len(df)} naturalistic stories.")
print("Sample (Raw vs Smart Cleaned):")
print(df['transcript'].iloc[0][:200])  # Preview

df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved to {OUTPUT_FILE}")