import os
import shutil

# --- CONFIGURATION ---
# The folder where your .cha files are located
TARGET_DIR = "/Users/shamiraventurini/PycharmProjects/ICL-PILOT/synthetic_data/synthetic_cha_annotated"

# Create a folder to move the bad files into (Safer than deleting immediately)
TRASH_DIR = os.path.join(TARGET_DIR, "failed_annotation")
os.makedirs(TRASH_DIR, exist_ok=True)


def is_file_annotated(filepath):
    """Checks if the file contains a %mor: tier."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith("%mor:"):
                    return True
        return False
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return False


# --- MAIN LOOP ---
print(f"Scanning {TARGET_DIR} for unannotated files...")
files_moved = 0
total_files = 0

for filename in os.listdir(TARGET_DIR):
    if filename.endswith(".cha"):
        total_files += 1
        filepath = os.path.join(TARGET_DIR, filename)

        # Check if it has the morphology tier
        if not is_file_annotated(filepath):
            print(f"Moving unannotated file: {filename}")
            shutil.move(filepath, os.path.join(TRASH_DIR, filename))
            files_moved += 1

print("-" * 30)
print(f"Scan Complete.")
print(f"Total .cha files scanned: {total_files}")
print(f"Files moved to 'failed_annotation': {files_moved}")
print(f"Files ready for analysis: {total_files - files_moved}")