#!/usr/bin/env python3

import json
import re
import os


def append_clean_lines(input_jsonl, output_jsonl):
    """
    Reads a JSONL file and APPENDS lines where the 'output' field
    contains NO error codes or special CHAT symbols to the output file.
    """

    # The "Blacklist" Regex
    # Matches: [*], [/], [//], words starting with 0, <, >, xxx, or (.+)
    error_pattern = r'\[\*|\[\/+\]|\b0\w+|[<>]|xxx|\(\.\+\)'

    # 1. Load existing lines from the output file to prevent duplicates (Optional but recommended)
    existing_inputs = set()
    if os.path.exists(output_jsonl):
        with open(output_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # We use the 'input' or 'output' text as a unique key
                    existing_inputs.add(data.get('input', ''))
                except json.JSONDecodeError:
                    continue

    clean_count = 0
    duplicate_count = 0
    total_processed = 0

    print(f"Reading from: {input_jsonl}")
    print(f"Appending to: {output_jsonl}")

    # 2. Open input in 'r' (read) and output in 'a' (append) mode
    with open(input_jsonl, 'r', encoding='utf-8') as infile, \
            open(output_jsonl, 'a', encoding='utf-8') as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue

            total_processed += 1
            try:
                data = json.loads(line)
                output_text = data.get('output', '')
                input_text = data.get('input', '')

                # Check if it contains errors
                has_error = re.search(error_pattern, output_text)

                if not has_error:
                    # Check for duplicates so we don't add the same sentence twice
                    if input_text not in existing_inputs:
                        outfile.write(json.dumps(data) + '\n')
                        existing_inputs.add(input_text)
                        clean_count += 1
                    else:
                        duplicate_count += 1

            except json.JSONDecodeError:
                continue

    print(f"--- Results ---")
    print(f"Lines processed:      {total_processed}")
    print(f"New lines appended:   {clean_count}")
    print(f"Duplicates skipped:   {duplicate_count}")
    print(f"Total lines now in {output_jsonl}: {len(existing_inputs)}")


if __name__ == "__main__":
    # You can change these every time you run the script
    current_input = 'chat_fine_tuning_data_3_clean.jsonl'
    master_output = 'clean_sentences_AB.jsonl'

    append_clean_lines(current_input, master_output)