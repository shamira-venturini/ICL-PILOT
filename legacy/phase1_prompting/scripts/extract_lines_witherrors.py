import os
import re
import json
import glob


def is_junk_fragment(text):
  """
  Identifies if a line is too short or grammatically useless
  to be a training example.
  """
  words = text.split()
  # 1. Skip anything with 2 words or fewer (e.g., "arm .", "the .")
  if len(words) <= 2:
    return True

  # 2. Skip lines that end in a "hanging" connector/preposition
  # (These are usually unfinished thoughts)
  hanging_words = {'and', 'the', 'a', 'an', 'but', 'or', 'so', 'in', 'on', 'at'}
  last_word = words[-1].lower().strip('.,?!')
  if last_word in hanging_words:
    return True

  return False

def clean_to_raw(text):
  """
  Converts a CHAT line into 'Raw' text by removing error codes
  and markers while properly handling omissions.
  """
  # 1. Remove the speaker tier (e.g., *CHI: ) and any initial tabs
  raw = re.sub(r'^\*[A-Z]{3}\d*:\s+', '', text)
  # 1. Remove speaker prefix like *1: or *1187:
  raw = re.sub(r'^\*\d+:\s+', '', raw)

  # 2. Remove CHAT error codes like [* m:vun:a] or [*]
  # and replacement brackets like [: saw]
  raw = re.sub(r'\[\*.*?\]', '', raw)
  raw = re.sub(r'\[:.*?\]', '', raw)

  # --- THE FIX ---
  # 3. Remove omitted word markers entirely (0det, 0subj, 0v, etc.)
  # Since these words were NOT spoken, they shouldn't be in the raw input.
  # This removes the '0' and all following alphanumeric characters.
  raw = re.sub(r'0\w+', '', raw)
  # ----------------

  # 4. Remove post-codes like [+ gram] or [+ sem]
  raw = re.sub(r'\[\+.*?\]', '', raw)

  # 5. Remove other CHAT markers like [//], [/], (.), etc.
  raw = re.sub(r'\[\/+\]', '', raw) # Retracings
  raw = re.sub(r'\(.*?\)', '', raw) # Pauses/Time
  raw = re.sub(r'\[.*?\]', '', raw) # Any other bracketed info

  # 5. Remove retracing markers [//] and [/] and pauses (.)
  raw = re.sub(r'\[\/+\]', '', raw)
  raw = re.sub(r'\(.*?\)', '', raw)

  # 6. Clean up whitespace
  raw = re.sub(r'\s+', ' ', raw).strip()

  # 6. Remove fragments and special symbols like &+sh or &-umm
  # (Optional: Keep these if you want the model to learn disfluency)
  #raw = re.sub(r'&\S+', '', raw)

  # 7. Clean up punctuation and extra whitespace
  raw = re.sub(r'\s+', ' ', raw) # Collapse multiple spaces
  # Remove TIMESTAMPS / BULLETS
  aw = re.sub(r'[\x15\d\s_\-]+\x15', '', raw) # Removes internal bullets
  raw = re.sub(r'\u0015.*?\u0015', '', raw) # Removes unicode bullets
  raw = re.sub(r'\x15', '', raw) # Removes stray markers
  raw = raw.replace('<', '').replace('>', '')


  raw = re.sub(r'\s+', ' ', raw).strip()

  return raw


def clean_chat_output(text):
  """
  Removes *CHI: and [+ gram]/[+ sem] from the output string.
  """
  # 1. Remove the speaker tier (e.g., *CHI: and the tab)
  clean = re.sub(r'^\*[A-Z]{3}\d*:\s+', '', text)
  # 1. Remove speaker prefix like *1: or *1187:
  clean = re.sub(r'^\*\d+:\s+', '', clean)
  # 2. Remove post-codes like [+ gram] or [+ sem] at the end
  clean = re.sub(r'\[\+.*?\]', '', clean)
  # Remove TIMESTAMPS / BULLETS
  clean = re.sub(r'[\x15]?\d+[_\-]\d+[\x15]?', '', clean)
  # 3. Clean up extra whitespace
  clean = re.sub(r'\s+', ' ', clean).strip()
  return clean


def scrape_chat_files(input_folder, output_jsonl):
  dataset = []

  # Look for all .cha files in the folder and subfolders
  files = glob.glob(os.path.join(input_folder, '**/*.cha'), recursive=True)
  print(f"Found {len(files)} files. Starting extraction...")

  for filepath in files:
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
      for line in f:
        # Only look at speaker tiers (lines starting with *)
        if line.startswith('*'):
          # Search for lines containing error codes or omissions
          if '[*'  or '0' or '[/]' or '[//]' or '<' or '>' not in line:
            chat_line = line.strip()
            raw_line = clean_to_raw(line)
            final_output = clean_chat_output(line)

            if is_junk_fragment(raw_line):
              continue

            # Ensure we don't have empty lines or perfect matches
            # (we want to learn errors!)
            if raw_line and raw_line != chat_line:
              dataset.append({
                "instruction": "Annotate the following sentence with CLAN error codes.",
                "input": raw_line,
                "output": final_output
              })

  # Save as JSONL (Standard format for fine-tuning)
  with open(output_jsonl, 'w', encoding='utf-8') as out:
    for entry in dataset:
      out.write(json.dumps(entry) + '\n')

  print(f"Success! Extracted {len(dataset)} training pairs to {output_jsonl}")


# --- RUN THE SCRAPER ---
# 1. Download data_original from TalkBank (e.g., CHILDES or AphasiaBank)
# 2. Point 'path_to_data' to that folder
path_to_data = 'ENNI_B1_SYN_tagged/'
output_file = 'training_data_B1_syn.jsonl'

scrape_chat_files(path_to_data, output_file)