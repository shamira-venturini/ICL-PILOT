#!/usr/bin/env python3
"""
Script to perform morphosyntactic analysis on all ENNI-B1 files using batchalign.
This script processes files from three directories:
1. ENNI_B1_TD (Typically Developing)
2. ENNI_B1_DLD (Developmental Language Disorder)
3. synthetic_data/ENNI_B1 (Synthetic data)

The script runs two analysis steps:
1. utseg - Utterance segmentation
2. morphotag - Morphosyntactic tagging
"""

import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def create_output_directory(base_dir):
    """Create a timestamped output directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / f"morphosyntactic_analysis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def find_cha_files(directories):
    """Find all .cha files in the specified directories"""
    cha_files = []
    for directory in directories:
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith('.cha'):
                        cha_files.append(os.path.join(root, file))
    return cha_files

def run_utseg(input_dir, output_dir):
    """Run utseg (utterance segmentation) on files in a directory"""
    try:
        cmd = [
            'batchalign',
            'utseg',
            str(input_dir),
            str(output_dir),
            '--lang', 'eng',  # English language code
            '-v', '1'  # verbose output level 1
        ]
        
        print(f"Running utseg on {input_dir}...")
        print(f"Command: {' '.join(cmd)}")
        
        # Use the batchalign's own Python environment to avoid dependency conflicts
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✓ Successfully completed utseg on {input_dir}")
            return True
        else:
            print(f"✗ Error running utseg on {input_dir}")
            print(f"  Return code: {result.returncode}")
            if result.stderr:
                # Show first 500 chars of stderr to avoid overwhelming output
                error_msg = result.stderr[:500]
                if len(result.stderr) > 500:
                    error_msg += "..."
                print(f"  Error: {error_msg}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout running utseg on {input_dir}")
        return False
    except Exception as e:
        print(f"✗ Exception running utseg on {input_dir}: {str(e)}")
        return False

def run_morphotag(input_dir, output_dir):
    """Run morphotag (morphosyntactic tagging) on files in a directory"""
    try:
        cmd = [
            'batchalign',
            'morphotag',
            str(input_dir),
            str(output_dir),
            '--retokenize',  # Retokenize to fit UD tokenizations
            '-v', '1'  # verbose output level 1
        ]
        
        print(f"Running morphotag on {input_dir}...")
        print(f"Command: {' '.join(cmd)}")
        
        # Use the batchalign's own Python environment to avoid dependency conflicts
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✓ Successfully completed morphotag on {input_dir}")
            return True
        else:
            print(f"✗ Error running morphotag on {input_dir}")
            print(f"  Return code: {result.returncode}")
            if result.stderr:
                # Show first 500 chars of stderr to avoid overwhelming output
                error_msg = result.stderr[:500]
                if len(result.stderr) > 500:
                    error_msg += "..."
                print(f"  Error: {error_msg}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout running morphotag on {input_dir}")
        return False
    except Exception as e:
        print(f"✗ Exception running morphotag on {input_dir}: {str(e)}")
        return False

def main():
    # Define the three ENNI-B1 directories
    enni_directories = [
        './ENNI_B1_TD',
        './ENNI_B1_DLD', 
        './synthetic_data/ENNI_B1'
    ]
    
    # Create output directory structure
    output_base = Path("./analysis_results")
    output_base.mkdir(exist_ok=True)
    analysis_dir = create_output_directory(output_base)
    
    # Create subdirectories for each processing step
    utseg_dir = analysis_dir / "1_utseg_results"
    morphotag_dir = analysis_dir / "2_morphotag_results"
    utseg_dir.mkdir(exist_ok=True)
    morphotag_dir.mkdir(exist_ok=True)
    
    print(f"Morphosyntactic Analysis of ENNI-B1 Files")
    print(f"Output will be saved to: {analysis_dir}")
    print("=" * 60)
    
    # Find all .cha files
    cha_files = find_cha_files(enni_directories)
    
    if not cha_files:
        print("No .cha files found in the specified directories.")
        return
    
    print(f"Found {len(cha_files)} .cha files to process:")
    for i, file in enumerate(cha_files, 1):
        print(f"  {i}. {file}")
    
    print("\nStarting morphosyntactic analysis...")
    print("Step 1: Running utseg (utterance segmentation)")
    print("-" * 60)
    
    # Step 1: Run utseg on all directories
    utseg_success = 0
    for input_dir in enni_directories:
        if os.path.exists(input_dir):
            if run_utseg(input_dir, utseg_dir):
                utseg_success += 1
    
    print(f"\nStep 1 complete: {utseg_success}/{len(enni_directories)} directories processed")
    
    print("\nStep 2: Running morphotag (morphosyntactic tagging)")
    print("-" * 60)
    
    # Step 2: Run morphotag on the utseg results
    morphotag_success = 0
    if utseg_success > 0:
        if run_morphotag(str(utseg_dir), morphotag_dir):
            morphotag_success = 1
    
    print(f"\nStep 2 complete: {morphotag_success} directories processed")
    
    print("\n" + "=" * 60)
    print(f"Analysis complete!")
    print(f"Results saved to: {analysis_dir}")
    print(f"  - Utseg results: {utseg_dir}")
    print(f"  - Morphotag results: {morphotag_dir}")
    
    # Create a summary file
    summary_file = analysis_dir / "analysis_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Morphosyntactic Analysis Summary\n")
        f.write(f"Generated on: {datetime.now()}\n")
        f.write(f"\nInput directories:\n")
        for input_dir in enni_directories:
            f.write(f"  - {input_dir}\n")
        f.write(f"\nTotal .cha files found: {len(cha_files)}\n")
        f.write(f"\nProcessing steps:\n")
        f.write(f"  1. Utseg (utterance segmentation): {utseg_success}/{len(enni_directories)} directories\n")
        f.write(f"  2. Morphotag (morphosyntactic tagging): {morphotag_success} directories\n")
        f.write(f"\nOutput directories:\n")
        f.write(f"  - Utseg results: {utseg_dir}\n")
        f.write(f"  - Morphotag results: {morphotag_dir}\n")
        f.write(f"\nAll processed files:\n")
        for cha_file in cha_files:
            f.write(f"  - {cha_file}\n")
    
    print(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()