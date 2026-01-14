#!/usr/bin/env python3
"""
Script to move entire text up one line in CHA files by removing the first empty line.
"""

import os
import glob

def move_text_up_one_line(file_path: str) -> None:
    """
    Move text up one line by removing the first empty line.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Remove first line if it's empty
    if lines and lines[0].strip() == '':
        new_lines = lines[1:]
    else:
        # If first line is not empty, just remove it anyway
        new_lines = lines[1:]
    
    # Write the modified content back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

def process_all_cha_files(directory: str) -> None:
    """
    Process all CHA files in the specified directory.
    """
    cha_files = glob.glob(os.path.join(directory, '*.cha'))
    
    print(f"Found {len(cha_files)} CHA files to process")
    
    for i, file_path in enumerate(cha_files, 1):
        print(f"Processing file {i}/{len(cha_files)}: {os.path.basename(file_path)}")
        move_text_up_one_line(file_path)
    
    print("Text movement completed for all files!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python move_text_up_one_line.py <directory_containing_cha_files>")
        sys.exit(1)
    
    directory = sys.argv[1]
    process_all_cha_files(directory)