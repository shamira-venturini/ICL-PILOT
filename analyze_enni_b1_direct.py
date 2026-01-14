#!/usr/bin/env python3
"""
Alternative script to perform morphosyntactic analysis on ENNI-B1 files.
This script tries to work around the NumPy compatibility issue by using
a more direct approach to analyze the CHAT files.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import subprocess

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

def analyze_chat_file(input_file, output_dir):
    """Analyze a single CHAT file using available tools"""
    try:
        input_path = Path(input_file)
        output_file = output_dir / input_path.name
        
        print(f"Analyzing {input_file}...")
        
        # First, try to copy the file to preserve the original
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Write to output directory
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✓ Copied {input_file} to {output_file}")
        return True
        
    except Exception as e:
        print(f"✗ Error processing {input_file}: {str(e)}")
        return False

def run_batchalign_with_workaround():
    """Try to run batchalign with various workarounds for the NumPy issue"""
    
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
    
    print("\nAttempting to run batchalign analysis...")
    
    # Try different approaches to run batchalign
    approaches = [
        {
            'name': 'Direct command with minimal options',
            'cmd': lambda input_dir, output_dir: [
                'batchalign', 'utseg', input_dir, output_dir, '--lang', 'eng'
            ]
        },
        {
            'name': 'Using batchalign Python module directly',
            'cmd': lambda input_dir, output_dir: [
                sys.executable, '-m', 'batchalign', 'utseg', input_dir, output_dir, '--lang', 'eng'
            ]
        }
    ]
    
    for approach in approaches:
        print(f"\nTrying approach: {approach['name']}")
        print("-" * 40)
        
        # Test on first directory
        test_dir = enni_directories[0]
        test_output = analysis_dir / "test_output"
        test_output.mkdir(exist_ok=True)
        
        cmd = approach['cmd'](test_dir, test_output)
        print(f"Command: {' '.join(str(x) for x in cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print("✓ Success! Using this approach...")
                
                # Process all directories with this working approach
                for input_dir in enni_directories:
                    if os.path.exists(input_dir):
                        dir_output = analysis_dir / Path(input_dir).name
                        dir_output.mkdir(exist_ok=True)
                        
                        final_cmd = approach['cmd'](input_dir, dir_output)
                        print(f"  Processing {input_dir} with: {' '.join(str(x) for x in final_cmd)}")
                        dir_result = subprocess.run(final_cmd, capture_output=True, text=True, timeout=300)
                        
                        if dir_result.returncode == 0:
                            print(f"✓ Successfully processed {input_dir}")
                        else:
                            print(f"✗ Failed to process {input_dir}")
                
                return True
            else:
                print(f"✗ Failed with return code {result.returncode}")
                if result.stderr:
                    print("Error output:")
                    print(result.stderr[:500])
                    
        except Exception as e:
            print(f"✗ Exception: {str(e)}")
    
    print("\n" + "=" * 60)
    print("All batchalign approaches failed due to NumPy compatibility issues.")
    print("Falling back to basic file analysis...")
    
    # Fallback: Copy all files to output directory for manual analysis
    fallback_dir = analysis_dir / "original_files"
    fallback_dir.mkdir(exist_ok=True)
    
    success_count = 0
    for cha_file in cha_files:
        if analyze_chat_file(cha_file, fallback_dir):
            success_count += 1
    
    print(f"\nFallback complete: {success_count}/{len(cha_files)} files copied")
    print(f"Original files are available in: {fallback_dir}")
    
    # Create summary
    summary_file = analysis_dir / "analysis_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Morphosyntactic Analysis Summary\n")
        f.write(f"Generated on: {datetime.now()}\n")
        f.write(f"\nStatus: Batchalign failed due to NumPy compatibility issues\n")
        f.write(f"Fallback: Copied {success_count} original files to {fallback_dir}\n")
        f.write(f"\nInput directories:\n")
        for input_dir in enni_directories:
            f.write(f"  - {input_dir}\n")
        f.write(f"\nTotal .cha files found: {len(cha_files)}\n")
        f.write(f"\nTo resolve the NumPy issue, try:\n")
        f.write(f"  1. Downgrade NumPy in the batchalign environment: numpy<2.0\n")
        f.write(f"  2. Or upgrade batchalign to a version compatible with NumPy 2.x\n")
    
    print(f"Summary saved to: {summary_file}")
    return False

def main():
    """Main function to run the analysis"""
    run_batchalign_with_workaround()

if __name__ == "__main__":
    main()