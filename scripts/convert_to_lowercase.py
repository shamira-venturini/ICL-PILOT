#!/usr/bin/env python3
"""
Script to convert all text in CHA files to lowercase while preserving structure.
"""

import os
import glob

def convert_file_to_lowercase(file_path: str) -> None:
    """
    Convert all text in a CHA file to lowercase.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Convert to lowercase
    lowercase_content = content.lower()
    
    # Write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(lowercase_content)

def process_all_cha_files(directory: str) -> None:
    """
    Process all CHA files in the specified directory.
    """
    cha_files = glob.glob(os.path.join(directory, '*.cha'))
    
    print(f"Found {len(cha_files)} CHA files to process")
    
    for i, file_path in enumerate(cha_files, 1):
        print(f"Processing file {i}/{len(cha_files)}: {os.path.basename(file_path)}")
        convert_file_to_lowercase(file_path)
    
    print("Lowercase conversion completed for all files!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python convert_to_lowercase.py <directory_containing_cha_files>")
        sys.exit(1)
    
    directory = sys.argv[1]
    process_all_cha_files(directory)