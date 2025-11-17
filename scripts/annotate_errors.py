import os
import pandas as pd

# --- Configuration ---
# This is the master file of 5444 sentences you created.
INPUT_FILE = "/Users/shamiraventurini/PycharmProjects/PythonProject/ICL-PILOT/results/sli_all_candidates.tsv"
OUTPUT_FILE = "/Users/shamiraventurini/PycharmProjects/PythonProject/ICL-PILOT/results/final_annotated_sentences.tsv"


# --- Main Program ---
def extract_flags():
    """
    Reads the usable sentences and identifies which ones have pre-existing
    error flags ([* ...]) on the main tier.
    """
    print(f"Loading sentences from {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found at {INPUT_FILE}")
        return

    try:
        df = pd.read_csv(INPUT_FILE, sep='\t', on_bad_lines='skip')
    except Exception as e:
        print(f"Error reading the TSV file: {e}")
        return

    # Create the new column for our simple flag
    df['Error_Flag'] = 'No_Error_Present'

    print("Scanning for existing error flags ([*])...")

    # Iterate over each sentence
    for index, row in df.iterrows():
        sentence = str(row.get('Sentence', ''))

        # The core, simplified logic:
        if '[*' in sentence:
            df.at[index, 'Error_Flag'] = 'Error_Present'

    # Save the newly flagged data
    df.to_csv(OUTPUT_FILE, sep='\t', index=False)

    print("-" * 20)
    print("Error flag extraction complete.")

    # Provide a summary of the findings
    flag_counts = df['Error_Flag'].value_counts()
    print("Sentence Distribution:")
    print(flag_counts)

    print(f"\nFinal annotated data saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    extract_flags()