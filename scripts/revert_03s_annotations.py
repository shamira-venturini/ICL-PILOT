#!/usr/bin/env python3
"""
Script to revert all [* m:03s:a] annotations from .cha files.
"""

import os
import re

def revert_03s_annotations(file_path):
    """
    Remove all [* m:03s:a] annotations from a .cha file.
    Returns the number of annotations removed.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    annotations_removed = 0
    for i, line in enumerate(lines):
        # Remove [* m:03s:a] annotations
        if "[* m:03s:a]" in line:
            lines[i] = re.sub(r'\s\[\* m:03s:a\]', '', line)
            annotations_removed += 1

    # Write the changes back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)

    return annotations_removed

def main():
    """
    Process all .cha files in the specified folders and revert annotations.
    """
    folders = ["data/A_original", "data/B_original", "ENNI_B1_SYN"]
    total_reverted = 0

    for folder_path in folders:
        print(f"\nReverting folder: {folder_path}")
        for filename in os.listdir(folder_path):
            if filename.endswith(".cha"):
                file_path = os.path.join(folder_path, filename)
                reverted = revert_03s_annotations(file_path)
                if reverted > 0:
                    print(f"File: {filename} - Reverted {reverted} annotations.")
                    total_reverted += reverted

    print(f"\nTotal [* m:03s:a] annotations reverted: {total_reverted}")

if __name__ == "__main__":
    main()
