#!/usr/bin/env python3

import json
import re
from collections import Counter
import os

# --- CONFIGURATION ---
INPUT_FILE = 'selected_syn_sentences.jsonl'
CLEAN_OUTPUT_FILE = 'clean_sentences_master.jsonl'
STATS_OUTPUT_FILE = 'error_codes_summary.txt'


def get_regex_patterns():
    """
    Returns a dictionary of patterns to look for.
    We use capturing groups to identify WHICH pattern matched.
    """
    patterns = {
        'error_code': r'\[\*\s*([^\]]+)\]',  # Captures inside [* ...]
        'omission': r'\b0\w+',  # 0aux, 0det, etc.
        'retracing': r'\[//\]',  # [//]
        'repetition': r'\[/\]',  # [/]
        'overlap': r'[<>]',  # < or >
        'unintel': r'xxx',  # xxx
        'pause': r'\(\.\+\)',  # (.+)
        'post_gram': r'\[\+\s*gram\]',  # [+ gram]
        'bracketed': r'\[::?\s*[^\]]+\]'  # [:: not] or [: replacement]
    }
    # Combine into one master regex for the filter
    combined_filter = "|".join([p.replace(r'([^\]]+)', r'[^\]]+') for p in patterns.values()])
    return patterns, combined_filter


def process_data():
    patterns, filter_regex = get_regex_patterns()
    error_counts = Counter()
    clean_data = []

    print(f"Reading {INPUT_FILE}...")

    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line: continue

            try:
                data = json.loads(line)
                # We check the 'output' field specifically
                text = data.get('output', '')

                # 1. COUNTING LOGIC
                # Check for [* ...] specifically to extract the code
                for code in re.findall(patterns['error_code'], text):
                    error_counts[code.strip()] += 1

                # Check for all other literal symbols
                if "[//]" in text: error_counts["[//]"] += text.count("[//]")
                if "[/]" in text: error_counts["[/]"] += (text.count("[/]") - text.count("[//]"))
                if "xxx" in text: error_counts["xxx"] += text.count("xxx")
                if "(..)" in text or "(.+)" in text: error_counts["(.+)"] += 1
                if "<" in text: error_counts["<"] += text.count("<")
                if ">" in text: error_counts[">"] += text.count(">")

                # Check for 0-omissions and [+ gram]
                for om in re.findall(patterns['omission'], text):
                    error_counts[om] += 1
                for gram in re.findall(patterns['post_gram'], text):
                    error_counts["[+ gram]"] += 1

                # 2. FILTERING LOGIC (The "Clean" check)
                # If the text has ANY of the forbidden patterns, we do NOT add it to clean_data
                if not re.search(filter_regex, text):
                    clean_data.append(data)

            except json.JSONDecodeError:
                print(f"Skipping line {line_num}: Invalid JSON")

    # --- SAVE CLEAN SENTENCES (Append Mode) ---
    print(f"Saving {len(clean_data)} clean sentences to {CLEAN_OUTPUT_FILE}...")
    with open(CLEAN_OUTPUT_FILE, 'a', encoding='utf-8') as f:
        for entry in clean_data:
            f.write(json.dumps(entry) + '\n')

    # --- SAVE STATS ---
    print(f"Saving statistics to {STATS_OUTPUT_FILE}...")
    with open(STATS_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(f"{'Pattern/Code':<20} | {'Count'}\n")
        f.write("-" * 30 + "\n")
        for code, count in sorted(error_counts.items()):
            f.write(f"{code:<20} | {count}\n")

    print("\n--- Summary ---")
    print(f"Total Unique Markers Found: {len(error_counts)}")
    print(f"Total Clean Lines Found:    {len(clean_data)}")


if __name__ == "__main__":
    process_data()