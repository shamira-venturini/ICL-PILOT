# ENNI-B1 Morphosyntactic Analysis in Google Colab

This guide explains how to run morphosyntactic analysis on your ENNI-B1 CHAT files using Google Colab with GPU acceleration.

## Step 1: Upload the Notebook to Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click "Upload" and select the `enni_b1_colab_analysis.ipynb` file
3. Or click "GitHub" and connect to your repository

## Step 2: Set Up Your GitHub Repository

**IMPORTANT**: You need to modify one line in the notebook:

In **Cell 2**, change this line:
```python
github_repo = "https://github.com/your-username/ICL-PILOT.git"
```

To your actual GitHub repository URL:
```python
github_repo = "https://github.com/your-actual-username/ICL-PILOT.git"
```

## Step 3: Enable GPU Acceleration

1. Click **Runtime** > **Change runtime type**
2. Select **GPU** as the hardware accelerator
3. Click **Save**

## Step 4: Run the Analysis

Execute the cells in order:

1. **Cell 1**: Sets up the environment and installs dependencies
2. **Cell 2**: Downloads your repository from GitHub
3. **Cell 3**: Sets up output directories
4. **Cell 4**: Runs utseg (utterance segmentation)
5. **Cell 5**: Runs morphotag (morphosyntactic tagging)
6. **Cell 6**: Generates summary and provides download link

## Step 5: Download Results

After completion, you'll get:
- A zip file with all analysis results
- Individual log files for each processing step
- Summary statistics

## Alternative Approach: spaCy Analysis

If batchalign has issues, **Cells 7-8** provide an alternative using spaCy for basic morphosyntactic analysis.

## Troubleshooting

### Common Issues:

1. **GPU not available**: Make sure you selected GPU in runtime settings
2. **Memory errors**: Process fewer files at a time
3. **Timeout errors**: The notebook has 15-minute timeouts per command
4. **GitHub download issues**: Make sure your repository is public or use a personal access token

### Manual File Upload:

If GitHub download fails, you can manually upload files:

```python
from google.colab import files
uploaded = files.upload()
```

## Expected Output Structure

```
analysis_results/
├── utseg_results_YYYYMMDD_HHMMSS/
│   ├── ENNI_B1_TD/
│   │   ├── *.cha (processed files)
│   │   └── ENNI_B1_TD_utseg.log
│   ├── ENNI_B1_DLD/
│   │   ├── *.cha (processed files)
│   │   └── ENNI_B1_DLD_utseg.log
│   └── synthetic_data_ENNI_B1/
│       ├── *.cha (processed files)
│       └── ENNI_B1_utseg.log
│
├── morphotag_results_YYYYMMDD_HHMMSS/
│   ├── *.cha (morphotag processed files)
│   └── morphotag.log
│
├── analysis_summary_YYYYMMDD_HHMMSS.txt
└── enni_b1_analysis_results_YYYYMMDD_HHMMSS.zip
```

## Requirements

- Google account (for Colab access)
- GitHub repository with your ENNI-B1 files
- Basic familiarity with Jupyter notebooks

## Support

If you encounter issues:
1. Check the log files in the output directories
2. Try the spaCy alternative approach
3. Consider processing smaller batches of files
4. Check Colab's resource usage (Runtime > Manage sessions)

The notebook is designed to handle the NumPy compatibility issues we encountered locally by using a clean Colab environment with proper GPU support.