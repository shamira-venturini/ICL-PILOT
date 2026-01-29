#!/usr/bin/env python3
"""
Script to identify and annotate agreement errors where 3rd person singular subjects are followed by bare regular verbs (e.g., "he play", "she eat").
Marks these errors with [* m:03s:a].
"""

import os
import re

# Singular subjects
SINGULAR_SUBJECTS = [
    "he", "she", "it", "this", "that", "one", "everyone", "someone", "anyone",
    "nobody", "somebody", "anybody", "either", "neither"
]

# Exclude these verbs (already covered by [* m:unv:a])
EXCLUDED_VERBS = [
    "be", "were", "have", "has", "do", "does", "go", "goes", "say", "says",
    "is", "was", "are", "am"
]

def find_and_annotate_03s_errors(file_path):
    """
    Find and annotate singular subject + bare regular verb errors in a .cha file.
    Writes the changes directly to the file.
    Returns the number of errors annotated.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    errors_found = 0
    for i, line in enumerate(lines):
        # Skip lines already marked with agreement errors
        if "*[" in line and ("m:03s:a" in line or "m:unv:a" in line or "m:vsg:a" in line):
            continue

        # Check for singular subjects followed by bare regular verbs
        for subject in SINGULAR_SUBJECTS:
            # Match any verb that is not in EXCLUDED_VERBS
            pattern = re.compile(
                rf'\b{subject}\s+([a-z]+)\b',
                re.IGNORECASE
            )
            match = pattern.search(line)
            if match:
                verb = match.group(1).lower()
                if verb not in EXCLUDED_VERBS:
                    # Annotate the error
                    lines[i] = re.sub(
                        rf'\b({subject}\s+{verb})\b',
                        rf'\g<1> [* m:03s:a]',
                        line,
                        flags=re.IGNORECASE
                    )
                    errors_found += 1

    # Write the changes back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)

    return errors_found

def main():
    """
    Process all .cha files in the specified folders and annotate errors.
    """
    folders = ["data/A_original", "data/B_original", "ENNI_B1_SYN"]
    total_errors = 0

    for folder_path in folders:
        print(f"\nAnnotating folder: {folder_path}")
        for filename in os.listdir(folder_path):
            if filename.endswith(".cha"):
                file_path = os.path.join(folder_path, filename)
                errors = find_and_annotate_03s_errors(file_path)
                if errors > 0:
                    print(f"File: {filename} - Annotated {errors} errors.")
                    total_errors += errors

    print(f"\nTotal [* m:03s:a] errors annotated: {total_errors}")

if __name__ == "__main__":
    main()
