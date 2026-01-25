#!/usr/bin/env python3
"""
Script to add CLAN error coding to CHA files following CHAT guidelines.
Based on CLAN manual and existing ENNI_B1_SYN files.
"""

import os
import re
import glob
from typing import List, Dict, Tuple, Optional

def analyze_and_add_error_coding(line: str) -> str:
    """
    Analyze a CHA utterance line and add CLAN error coding.
    Returns the modified line with error codes.
    """
    if not line.startswith('*CHI:'):
        return line
    
    # Extract the utterance text (after *CHI:\t)
    utterance = line[6:].strip()
    
    # Common error patterns for DLD based on the ENNI_B1_SYN files
    error_patterns = [
        # Past tense errors - base form instead of past tense
        (r'\bgo\b', 'went', '[* m:base:ed]'),
        (r'\bmake\b', 'made', '[* m:base:ed]'),
        (r'\bdig\b', 'dug', '[* m:base:ed]'),
        (r'\blift\b', 'lifted', '[* m:0ed]'),
        (r'\bpat\b', 'patted', '[* m:0ed]'),
        (r'\bwork\b', 'worked', '[* m:0ed]'),
        (r'\bplay\b', 'played', '[* m:0ed]'),
        (r'\bfeel\b', 'felt', '[* m:base:ed]'),
        (r'\bcry\b', 'cried', '[* m:0ed]'),
        (r'\bsmash\b', 'smashed', '[* m:0ed]'),
        (r'\bfall\b', 'fell', '[* m:base:ed]'),
        
        # Missing articles - add 0 prefix
        (r'\bRabbit\b', '0the Rabbit', None),
        (r'\bDog\b', '0the Dog', None),
        (r'\bCastle\b', '0the Castle', None),
        (r'\bShe\b', '0the She', None),
        (r'\bHe\b', '0the He', None),
        
        # Missing copula/auxiliary - "no" instead of "not"
        (r'\bno\b', 'not', '[* s:r]'),
        
        # Word order and other errors
        (r'\bwanna\b', 'wanted to', '[* m:0ed]'),
        (r'\bgogo\b', 'went', '[* m:base:ed]'),
    ]
    
    modified_utterance = utterance
    
    # Apply error patterns
    for pattern, correction, error_code in error_patterns:
        # Find matches and add error coding
        matches = re.finditer(pattern, modified_utterance)
        for match in matches:
            start, end = match.span()
            original_word = match.group()
            
            # Add correction and error code if applicable
            if error_code:
                replacement = f"{original_word} [: {correction}] {error_code}"
            else:
                replacement = correction
            
            modified_utterance = modified_utterance[:start] + replacement + modified_utterance[end:]
    
    # Handle missing articles more systematically
    # Add 0 prefix to words that should have articles
    words = modified_utterance.split()
    processed_words = []
    
    for i, word in enumerate(words):
        # Skip words that are already error-coded or corrected
        if word.startswith('[:') or word.startswith('[*]') or word.startswith('0'):
            processed_words.append(word)
            continue
            
        # Add 0 prefix to nouns that likely need articles
        if (word.lower() in ['rabbit', 'dog', 'castle', 'sand', 'bucket', 'mistake', 
                           'sandcastle', 'sandbox', 'shovels', 'work', 'fun'] and
            (i == 0 or (i > 0 and words[i-1].lower() not in ['the', 'a', 'an', 'his', 'her', 'their']))):
            processed_words.append(f"0{word}")
        else:
            processed_words.append(word)
    
    modified_utterance = ' '.join(processed_words)
    
    return f"*CHI:\t{modified_utterance}"

def add_error_coding_to_file(file_path: str) -> None:
    """
    Add error coding to a single CHA file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    
    for line in lines:
        if line.startswith('*CHI:'):
            # Process child utterances
            modified_line = analyze_and_add_error_coding(line.strip())
            new_lines.append(modified_line + '\n')
        else:
            new_lines.append(line)
    
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
        add_error_coding_to_file(file_path)
    
    print("Error coding completed for all files!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python add_clan_error_coding.py <file_or_directory>")
        sys.exit(1)
    
    path = sys.argv[1]
    
    # Check if it's a file or directory
    if os.path.isfile(path) and path.endswith('.cha'):
        print(f"Processing single file: {os.path.basename(path)}")
        add_error_coding_to_file(path)
        print("Error coding completed!")
    elif os.path.isdir(path):
        process_all_cha_files(path)
    else:
        print("Error: Path must be a CHA file or directory containing CHA files")
        sys.exit(1)