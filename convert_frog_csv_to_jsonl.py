#!/usr/bin/env python3
"""
Convert frog story CSV files to JSONL format, extracting only the text from output_story column.
"""
import csv
import json
import os

# Define the mapping from CSV filenames to JSONL filenames
csv_to_jsonl_mapping = {
    "condition_a_0shot.csv": "q_text_b1_A.jsonl",
    "condition_b_0shot.csv": "q_text_b1_B.jsonl",
    "condition_c_010shot.csv": "q_text_b1_C10.jsonl",
    "condition_c_2shot.csv": "q_text_b1_C02.jsonl",
    "condition_c_4shot.csv": "q_text_b1_C04.jsonl",
    "condition_c_6shot.csv": "q_text_b1_C06.jsonl"
}

def convert_csv_to_jsonl(csv_filepath, jsonl_filepath):
    """
    Convert a CSV file to JSONL format, extracting only the output_story column.
    """
    with open(csv_filepath, 'r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        
        with open(jsonl_filepath, 'w', encoding='utf-8') as jsonl_file:
            for row in reader:
                # Extract only the output_story text
                text = row['output_story'].strip()
                # Write as JSON object with just the text
                json_line = json.dumps({"text": text}, ensure_ascii=False)
                jsonl_file.write(json_line + '\n')
    
    print(f"Converted {csv_filepath} to {jsonl_filepath}")

def main():
    csv_dir = "archive_/frog_story/csv"
    output_dir = "archive_/frog_story/jsonl"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each CSV file
    for csv_filename, jsonl_filename in csv_to_jsonl_mapping.items():
        csv_path = os.path.join(csv_dir, csv_filename)
        jsonl_path = os.path.join(output_dir, jsonl_filename)
        
        if os.path.exists(csv_path):
            convert_csv_to_jsonl(csv_path, jsonl_path)
        else:
            print(f"Warning: {csv_path} not found")
    
    print("All conversions completed!")

if __name__ == "__main__":
    main()