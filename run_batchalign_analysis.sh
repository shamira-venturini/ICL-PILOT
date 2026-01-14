#!/bin/bash

# Batchalign Morphosyntactic Analysis Script for ENNI-B1 Files
# This script runs utseg and morphotag on all ENNI-B1 directories

echo "=========================================="
echo "Batchalign Morphosyntactic Analysis"
echo "=========================================="
echo ""

# Create output directories
OUTPUT_BASE="analysis_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
UTSEG_OUTPUT="${OUTPUT_BASE}/utseg_results_${TIMESTAMP}"
MORPHOTAG_OUTPUT="${OUTPUT_BASE}/morphotag_results_${TIMESTAMP}"

mkdir -p "${UTSEG_OUTPUT}"
mkdir -p "${MORPHOTAG_OUTPUT}"

echo "Output directories created:"
echo "  - Utseg results: ${UTSEG_OUTPUT}"
echo "  - Morphotag results: ${MORPHOTAG_OUTPUT}"
echo ""

# Define the directories to process
DIRECTORIES=(
    "./ENNI_B1_TD"
    "./ENNI_B1_DLD" 
    "./synthetic_data/ENNI_B1"
)

echo "Directories to process:"
for dir in "${DIRECTORIES[@]}"; do
    if [ -d "$dir" ]; then
        count=$(find "$dir" -name "*.cha" | wc -l)
        echo "  - $dir ($count .cha files)"
    else
        echo "  - $dir (NOT FOUND)"
    fi
done
echo ""

# Function to run a batchalign command with timeout
run_batchalign() {
    local cmd_name="$1"
    local input_dir="$2"
    local output_dir="$3"
    shift 3
    local extra_args="$@"
    
    echo "Running ${cmd_name} on ${input_dir}..."
    
    # Create a log file
    local log_file="${output_dir}/$(basename "${input_dir}")_${cmd_name}.log"
    
    # Run the command (without timeout since it's not available on macOS)
    echo "  This may take a while..."
    if batchalign "${cmd_name}" "${input_dir}" "${output_dir}" "${extra_args}" > "${log_file}" 2>&1; then
        echo "✓ Successfully completed ${cmd_name} on ${input_dir}"
        return 0
    else
        echo "✗ Failed running ${cmd_name} on ${input_dir}"
        echo "  See log: ${log_file}"
        return 1
    fi
}

echo "Step 1: Running utseg (utterance segmentation)"
echo "----------------------------------------------"

utseg_success=0
utseg_total=0

for dir in "${DIRECTORIES[@]}"; do
    if [ -d "$dir" ]; then
        utseg_total=$((utseg_total + 1))
        
        # Create subdirectory for this dataset
        dir_output="${UTSEG_OUTPUT}/$(basename "${dir}")"
        mkdir -p "${dir_output}"
        
        if run_batchalign "utseg" "${dir}" "${dir_output}" "--lang" "eng"; then
            utseg_success=$((utseg_success + 1))
        fi
    fi
done

echo ""
echo "Step 1 Results: ${utseg_success}/${utseg_total} directories processed"
echo ""

if [ "${utseg_success}" -gt 0 ]; then
    echo "Step 2: Running morphotag (morphosyntactic tagging)"
    echo "--------------------------------------------------"
    
    if run_batchalign "morphotag" "${UTSEG_OUTPUT}" "${MORPHOTAG_OUTPUT}" "--retokenize"; then
        echo ""
        echo "=========================================="
        echo "✓ Analysis Complete!"
        echo "=========================================="
        echo "Results saved to: ${OUTPUT_BASE}"
        echo "  - Utseg results: ${UTSEG_OUTPUT}"
        echo "  - Morphotag results: ${MORPHOTAG_OUTPUT}"
        echo ""
        echo "Summary:"
        echo "  - Utseg: ${utseg_success}/${utseg_total} directories"
        echo "  - Morphotag: 1 directory"
        
        # Create a summary file
        summary_file="${OUTPUT_BASE}/analysis_summary_${TIMESTAMP}.txt"
        {
            echo "Batchalign Morphosyntactic Analysis Summary"
            echo "Generated on: $(date)"
            echo ""
            echo "Input directories:"
            for dir in "${DIRECTORIES[@]}"; do
                if [ -d "$dir" ]; then
                    count=$(find "$dir" -name "*.cha" | wc -l)
                    echo "  - $dir ($count files)"
                fi
            done
            echo ""
            echo "Processing results:"
            echo "  - Utseg: ${utseg_success}/${utseg_total} directories processed"
            echo "  - Morphotag: Successfully completed"
            echo ""
            echo "Output directories:"
            echo "  - Utseg results: ${UTSEG_OUTPUT}"
            echo "  - Morphotag results: ${MORPHOTAG_OUTPUT}"
            echo ""
            echo "Log files available in each output directory"
        } > "${summary_file}"
        
        echo "Summary saved to: ${summary_file}"
    else
        echo ""
        echo "✗ Morphotag failed"
    fi
else
    echo ""
    echo "✗ No directories were successfully processed in utseg step"
    echo "Check the log files in ${UTSEG_OUTPUT} for details"
fi