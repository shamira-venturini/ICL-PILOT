#!/usr/bin/env python3

import json
import re

def add_slashes_between_repeats(text):
    """
    Add [/] between repeated words where not already present.
    Looks for patterns like "word word" and converts to "word [/] word"
    """
    # Pattern to find repeated words (including punctuation)
    # We'll look for word boundaries and repeated sequences
    pattern = r'(\b\w+)\s+\1\b'
    
    def replace_match(match):
        word = match.group(1)
        # Check if [/] is already present
        if f"{word} [/] {word}" not in match.group(0):
            return f"{word} [/] {word}"
        return match.group(0)
    
    # Apply the replacement
    result = re.sub(pattern, replace_match, text)
    return result

def process_jsonl_file(input_file, output_file):
    """
    Process a JSONL file and add [/] between repeated words in the output field
    """
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            try:
                data = json.loads(line)
                
                # Process the output field
                if 'output' in data:
                    data['output'] = add_slashes_between_repeats(data['output'])
                
                # Write the modified line
                json.dump(data, outfile, ensure_ascii=False)
                outfile.write('\n')
                
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                print(f"Problematic line: {line}")
                # Write the original line if there's an error
                outfile.write(line)

def process_text_file(input_file, output_file):
    """
    Process a text file and add [/] between repeated words
    """
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            processed_line = add_slashes_between_repeats(line)
            outfile.write(processed_line)

if __name__ == "__main__":
    # Process training_data.jsonl
    print("Processing training_data.jsonl...")
    process_jsonl_file('training_data.jsonl', 'training_data_processed.jsonl')
    
    # Process output_sentences_with_patterns.txt
    print("Processing output_sentences_with_patterns.txt...")
    process_text_file('output_sentences_with_patterns.txt', 'output_sentences_with_patterns_processed.txt')
    
    print("Processing complete!")