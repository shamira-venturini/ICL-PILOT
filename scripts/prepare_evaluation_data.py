import pandas as pd
import re

# --- CONFIGURATION ---
INPUT_FILE = "/Users/shamiraventurini/PycharmProjects/ICL-PILOT/real_data/dld_stories_naturalistic.csv"
OUTPUT_FILE = "/Users/shamiraventurini/PycharmProjects/ICL-PILOT/real_data/dld_stories_evaluation_set.csv"


def parse_age_float(age_str):
    try:
        clean_str = str(age_str).strip()
        parts = re.split(r'[;.]', clean_str)
        years = int(parts[0])
        months = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        return years + (months / 12.0)
    except:
        return 0.0


# 1. LOAD DATA
try:
    df = pd.read_csv(INPUT_FILE)
    print(f"Original Dataset Size: {len(df)} stories")
except FileNotFoundError:
    print("Error: Upload 'dld_stories_cleaned.csv' first!")
    df = pd.DataFrame()

if not df.empty:
    # Ensure 'age_float' column exists
    df['age_float'] = df['age_raw'].apply(parse_age_float)

    # 2. DEFINE EXCLUSION PATTERNS (Based on your pasted prompt)
    # These are unique phrases from the 10 examples you provided.
    exclusion_phrases = [
        "hes eating... he (i)s going to eat",  # Ex 1 (B2)
        "um , jumped uh... ,",  # Ex 2 (A2)
        "Is have balloon.",  # Ex 3 (B3)
        "The the those leave",  # Ex 4 (Aliens)
        "a elephant and. . a cow. and , the cow went in the water",  # Ex 5 (A1)
        "He has a airplane",  # Ex 6 (A3)
        "that elephant sad",  # Ex 7 (A1)
        "And then the horse is l w...",  # Ex 8 (A3)
        "the boy said ooh cool",  # Ex 9 (A1)
        "ele elephant say"  # Ex 10 (A1)
    ]

    # 3. FILTER THE DATAFRAME
    # Escape special characters just in case
    safe_phrases = [re.escape(p) for p in exclusion_phrases]
    pattern = '|'.join(safe_phrases)

    # --- CRITICAL FIXES ---
    # 1. regex=True (To make the '|' work as OR)
    # 2. case=False (To match 'The' with 'the')
    matches = df[df['transcript'].str.contains(pattern, case=False, regex=True, na=False)]

    print(f"\nFound {len(matches)} matching stories to remove.")

    if len(matches) > 0:
        print("Sample of removed stories:")
        print(matches[['story_id', 'age_raw', 'transcript']])
    else:
        print("WARNING: Found 0 matches. Check if the raw text in csv matches the prompt text.")

    # Create the clean set (Inverse filter)
    df_clean = df[~df['transcript'].str.contains(pattern, case=False, regex=True, na=False)].copy()

    # Filter B: Remove Age < 4.0 (and optionally > 6.0)
    df_clean = df_clean[
        (df_clean['age_float'] >= 4.0) &
        (df_clean['age_float'] < 6.0)
        ].copy()

    # 4. VERIFY AND SAVE
    print(f"\nRemaining stories for Evaluation: {len(df_clean)}")

    # Save to new csv
    df_clean.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved clean evaluation set to: {OUTPUT_FILE}")