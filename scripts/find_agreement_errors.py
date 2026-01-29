#!/usr/bin/env python3
"""
Script to identify subject-verb agreement errors where plural subjects are followed by 3rd person singular verbs (e.g., "they was").
Marks these errors with [* m:vsg:a] .
"""

import os
import re

# Plural subjects
PLURAL_SUBJECTS = [
    "they", "we", "these", "those", "both", "many", "few", "several"
]

# 3rd person singular verbs
SINGULAR_VERBS = [
    "was", "has", "does", "is", "wants", "needs", "likes", "goes", "says"
]

def find_and_annotate_vsg_errors(file_path):
    """
    Find and annotate plural subject + singular verb errors in a .cha file.
    Writes the changes directly to the file.
    Returns the number of errors annotated.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    errors_found = 0
    for i, line in enumerate(lines):
        # Skip lines that are already marked with [* m:vsg:a] 
        if "*[" in line and "m:vsg:a" in line:
            continue

        # Check for plural subjects followed by singular verbs (e.g., "they was")
        for subject in PLURAL_SUBJECTS:
            for verb in SINGULAR_VERBS:
                pattern = re.compile(rf'\b{subject}\s+{verb}\b', re.IGNORECASE)
                if pattern.search(line):
                    # Annotate the error
                    lines[i] = re.sub(
                        rf'\b({subject}\s+{verb})\b',
                        rf'\g<1> [* m:vsg:a] ',
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
                errors = find_and_annotate_vsg_errors(file_path)
                if errors > 0:
                    print(f"File: {filename} - Annotated {errors} errors.")
                    total_errors += errors

    print(f"\nTotal [* m:vsg:a]  errors annotated: {total_errors}")

if __name__ == "__main__":
    main()
