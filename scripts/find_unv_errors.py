#!/usr/bin/env python3
"""
Script to identify subject-verb agreement errors where singular subjects are followed by bare verbs (e.g., "it have", "he do").
Marks these errors with [* m:unv:a].
"""

import os
import re

# Singular subjects
SINGULAR_SUBJECTS = [
    "it", "she", "he", "this", "that", "one", "everyone", "someone", "anyone",
    "nobody", "somebody", "anybody", "either", "neither"
]

# Bare verbs (e.g., "have", "do", "go", "say")
BARE_VERBS = [
    "be", "were", "have", "do", "go", "say"
]

def find_and_annotate_unv_errors(file_path):
    """
    Find and annotate singular subject + bare verb errors in a .cha file.
    Writes the changes directly to the file.
    Returns the number of errors annotated.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    errors_found = 0
    for i, line in enumerate(lines):
        # Skip lines that are already marked with [* m:unv:a]
        if "*[" in line and "m:unv:a" in line:
            continue

        # Check for singular subjects followed by bare verbs (e.g., "it have", "he do")
        for subject in SINGULAR_SUBJECTS:
            for verb in BARE_VERBS:
                pattern = re.compile(rf'\b{subject}\s+{verb}\b', re.IGNORECASE)
                if pattern.search(line):
                    # Annotate the error
                    lines[i] = re.sub(
                        rf'\b({subject}\s+{verb})\b',
                        rf'\g<1> [* m:unv:a]',
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
                errors = find_and_annotate_unv_errors(file_path)
                if errors > 0:
                    print(f"File: {filename} - Annotated {errors} errors.")
                    total_errors += errors

    print(f"\nTotal [* m:unv:a] errors annotated: {total_errors}")

if __name__ == "__main__":
    main()
