#!/usr/bin/env python3
"""
Extract text from *CHI: lines in ENNI/SLI files that have underscores in their filenames.
"""
import os
import re
import json

def extract_chi_text(file_path):
    """
    Extract text from *CHI: lines in a file, concatenate them, and strip annotations.
    Returns a single string containing all CHI text from the file.
    """
    chi_texts = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            
            # Check if line starts with *CHI:
            if line.startswith('*CHI:'):
                # Extract text after *CHI:
                text_part = line[5:].strip()  # Remove '*CHI:'
                
                # Remove annotations in square brackets, angle brackets, etc.
                # Pattern to remove various annotation formats:
                # [^c] - character-level annotations
                # [=! whispers] - meta annotations
                # [< text >] - overlapping speech
                # s:ome::body - phonetic transcription annotations (extract intended word)
                # etc.
                clean_text = re.sub(r'\[[^\]]*\]', '', text_part)  # Remove [annotations]
                clean_text = re.sub(r'<[^>]*>', '', clean_text)     # Remove <annotations>
                clean_text = re.sub(r'\{[^}]*\}', '', clean_text) # Remove {} annotations
                clean_text = re.sub(r'\([^)]*\)', '', clean_text) # Remove () annotations
                clean_text = re.sub(r'\^.*$', '', clean_text)      # Remove ^ annotations
                clean_text = re.sub(r'\+[^\s]*', '', clean_text)   # Remove + annotations
                clean_text = re.sub(r'&[^\s]*', '', clean_text)    # Remove & annotations
                clean_text = re.sub(r'\|[^\s]*', '', clean_text)  # Remove | annotations
                clean_text = re.sub(r'\\[^\s]*', '', clean_text) # Remove \ annotations
                
                # Handle phonetic transcription annotations like s:ome::body
                # Extract the intended/corrected word by removing colons and keeping the full word
                clean_text = re.sub(r'([a-zA-Z]+):+([a-zA-Z]+)', r'\1\2', clean_text)
                clean_text = re.sub(r'([a-zA-Z]):+', r'\1', clean_text)  # Remove remaining colons
                
                # Clean up multiple spaces and strip
                clean_text = ' '.join(clean_text.split()).strip()
                
                if clean_text:  # Only add if there's text left
                    chi_texts.append(clean_text)
    
    # Join all CHI lines from this file into a single text, separated by spaces
    return ' '.join(chi_texts) if chi_texts else None

def main():
    sli_dir = "ENNI/SLI"
    output_file = "p_text_SLI.jsonl"
    
    # Find all files with underscores in their names
    files_with_underscores = []
    for root, dirs, files in os.walk(sli_dir):
        for file in files:
            if '_' in file:
                files_with_underscores.append(os.path.join(root, file))
    
    print(f"Found {len(files_with_underscores)} files with underscores")
    
    # Process each file and collect all CHI texts
    all_chi_texts = []
    
    for file_path in files_with_underscores:
        try:
            chi_text = extract_chi_text(file_path)
            if chi_text:  # Only add if there's text
                all_chi_texts.append(chi_text)
                # Count the number of CHI lines in this file for reporting
                with open(file_path, 'r', encoding='utf-8') as f:
                    chi_lines = [line for line in f if line.strip().startswith('*CHI:')]
                print(f"Processed {file_path}: {len(chi_lines)} CHI lines -> 1 file entry")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Write to JSONL file
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in all_chi_texts:
            json_line = json.dumps({"text": text}, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"\nCreated {output_file} with {len(all_chi_texts)} CHI text entries")

if __name__ == "__main__":
    main()