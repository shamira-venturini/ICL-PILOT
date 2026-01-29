#!/usr/bin/env python3
"""
Add [+ gram] tag to lines containing [* m: in CHA files.
The [+ gram] should be added after punctuation but before the %mor line.
"""
import os
import re

def process_file(file_path):
    """
    Process a single CHA file to add [+ gram] tags where needed.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    modified = False
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Check if line contains [* m:, [* s:r:, or 0 and doesn't already have [+ gram]
        if ('[* m:' in line or '[* s:r:' in line or ' 0' in line or 'CHI:	0subj' in line or 'CHI:	0det' in line) and '[+ gram]' not in line:
            # Find the punctuation at the end of the line
            punctuation_match = re.search(r'[.!?]\s*$', line)
            if punctuation_match:
                punctuation_end = punctuation_match.end()
                # Insert [+ gram] after the punctuation
                new_line = line[:punctuation_end] + ' [+ gram]' + line[punctuation_end:]
                lines[i] = new_line + '\n'  # Keep the original line ending
                modified = True
        
        i += 1
    
    if modified:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(lines)
        print(f"Modified {file_path}")
    
    return modified

def main():
    data_dir = "data"
    
    # Find all CHA files
    cha_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.cha'):
                cha_files.append(os.path.join(root, file))
    
    print(f"Found {len(cha_files)} CHA files to process")
    
    modified_count = 0
    for file_path in cha_files:
        try:
            if process_file(file_path):
                modified_count += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print(f"\nProcessing complete. Modified {modified_count} files.")

if __name__ == "__main__":
    main()