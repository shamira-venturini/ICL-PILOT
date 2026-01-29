#!/usr/bin/env python3
"""
Extract error code frequencies from .cex files in ENNI_B1_SYN_freq folder
and save to a CSV file called synthetic_B1_freq.csv
"""

import os
import re
import csv
from collections import defaultdict

def extract_error_frequencies(file_path):
    """
    Extract error codes and their frequencies from a .cex file
    Returns a dictionary of {error_code: frequency}
    """
    error_frequencies = {}
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Find the "From file" line to get the original filename
        file_match = re.search(r'From file <(.*?)>', content)
        if file_match:
            original_file = os.path.basename(file_match.group(1))
        else:
            original_file = os.path.basename(file_path)
        
        # Extract error codes and frequencies from the Speaker: *CHI: section
        # Pattern matches lines like: "  9 [* m:03s:a]"
        # We need to capture both the frequency (digits) and the error code
        error_pattern = r'(\d+)\s+\[\*\s+[a-zA-Z]:[^\]]+\]'
        frequencies = re.findall(error_pattern, content)
        
        # Also extract the error codes themselves
        error_code_pattern = r'\[\*\s+[a-zA-Z]:[^\]]+\]'
        error_codes = re.findall(error_code_pattern, content)
        
        # Pair frequencies with error codes
        for i, error_code in enumerate(error_codes):
            if i < len(frequencies):
                error_frequencies[error_code] = int(frequencies[i])
            
        return original_file, error_frequencies
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None

def main():
    # Directory containing the .cex files
    cex_dir = "ENNI_B1_SYN_freq"
    
    # Collect all error codes across all files
    all_error_codes = set()
    file_data = []
    
    # First pass: collect all unique error codes
    for filename in os.listdir(cex_dir):
        if filename.endswith('.cex'):
            file_path = os.path.join(cex_dir, filename)
            original_file, error_freqs = extract_error_frequencies(file_path)
            
            if error_freqs is not None:
                all_error_codes.update(error_freqs.keys())
                file_data.append((original_file, error_freqs))
            else:
                # Include files with no error codes as empty dict
                file_data.append((original_file, {}))
    
    # Sort error codes for consistent column order
    sorted_error_codes = sorted(all_error_codes)
    
    # Write to CSV
    csv_filename = "synthetic_B1_freq.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        header = ["File"] + sorted_error_codes
        writer.writerow(header)
        
        # Write data rows
        for original_file, error_freqs in file_data:
            row = [original_file]
            for error_code in sorted_error_codes:
                row.append(error_freqs.get(error_code, 0))
            writer.writerow(row)
    
    print(f"Successfully created {csv_filename}")
    print(f"Found {len(file_data)} files with error data")
    print(f"Found {len(all_error_codes)} unique error codes")
    print(f"Error codes: {sorted_error_codes}")

if __name__ == "__main__":
    main()