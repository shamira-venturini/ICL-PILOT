#!/usr/bin/env python3
"""
Script to identify past tense formation errors in .cha files.
Marks irregular verbs incorrectly conjugated as regular verbs with [* m:=ed].
"""

import os
import re

# Define irregular verbs and their correct past tense forms
IRREGULAR_VERBS = {
    "break": "broke",
    "choose": "chose",
    "feel": "felt",
    "think": "thought",
    "buy": "bought",
    "stick": "stuck",
    "give": "gave",
    "fall": "fell",
    "run": "ran",
    "tell": "told",
    "fly": "flew",
    "come": "came",
    "build": "built",
    "have": "had",
    "go": "went",
    "hurt": "hurt",
    "do": "did",
    "say": "said",
    "eat": "ate",
    "take": "took",
    "see": "saw",
    "pop": "popped",
    "blow": "blew",
    "bring": "brought",
}

def find_and_mark_errors(file_path):
    """
    Find and mark past tense errors in a .cha file.
    Returns a list of tuples: (line_number, original_line, corrected_line).
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    errors_found = []
    for i, line in enumerate(lines):
        # Check for irregular verbs conjugated as regular verbs (e.g., "thinked", "feeled")
        for verb, past_form in IRREGULAR_VERBS.items():
            # Match patterns like "thinked", "feeled", etc.
            pattern = re.compile(rf'\b{verb}ed\b')
            if pattern.search(line):
                # Mark the error
                corrected_line = re.sub(
                    rf'\b({verb}ed)\b',
                    rf'\g<1> [: {past_form}] [* m:=ed]',
                    line
                )
                errors_found.append((i + 1, line.strip(), corrected_line.strip()))

    return errors_found

def main():
    """
    Process all .cha files in the specified folder and report errors.
    """
    folder_path = "data/A_original"
    total_errors = 0
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".cha"):
            file_path = os.path.join(folder_path, filename)
            errors = find_and_mark_errors(file_path)
            if errors:
                print(f"File: {filename}")
                for line_num, original, corrected in errors:
                    print(f"  Line {line_num}: {original} -> {corrected}")
                total_errors += len(errors)

    print(f"\nTotal errors found: {total_errors}")

if __name__ == "__main__":
    main()
