#!/bin/bash

# Script to set up a clean Python environment for batchalign
# This creates a conda environment with compatible versions

echo "=========================================="
echo "Setting up Batchalign Environment"
echo "=========================================="
echo ""

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "✓ Conda is available"
else
    echo "✗ Conda is not available"
    echo "Please install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create a new environment with compatible versions
ENV_NAME="batchalign_compat"

echo "Creating conda environment: ${ENV_NAME}"
echo ""

# Create environment with Python 3.11 and NumPy 1.24
conda create -n "${ENV_NAME}" python=3.11 numpy=1.24 -y

if [ $? -eq 0 ]; then
    echo "✓ Environment created successfully"
    echo ""
    
    # Activate the environment and install batchalign
    echo "Activating environment and installing batchalign..."
    source activate "${ENV_NAME}" || conda activate "${ENV_NAME}"
    
    pip install batchalign
    
    if [ $? -eq 0 ]; then
        echo "✓ Batchalign installed successfully"
        echo ""
        
        # Test the installation
        echo "Testing batchalign..."
        batchalign version
        
        if [ $? -eq 0 ]; then
            echo "✓ Batchalign is working!"
            echo ""
            echo "You can now run your analysis with:"
            echo "  conda activate ${ENV_NAME}"
            echo "  ./run_batchalign_analysis.sh"
            echo ""
            echo "Or run individual commands like:"
            echo "  batchalign utseg ./ENNI_B1_TD output_dir --lang eng"
        else
            echo "✗ Batchalign test failed"
        fi
    else
        echo "✗ Failed to install batchalign"
    fi
else
    echo "✗ Failed to create conda environment"
fi

echo ""
echo "=========================================="
echo "Environment setup complete"
echo "=========================================="