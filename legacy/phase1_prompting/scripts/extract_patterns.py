#!/usr/bin/env python3

import json
import re


def extract_full_lines_with_patterns(input_file, patterns):
  """Extract the entire JSON object if the output matches specified patterns"""

  compiled_patterns = [re.compile(pattern) for pattern in patterns]
  matching_records = []

  with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      try:
        data = json.loads(line)
        output_text = data.get('output', '')

        # Check if any pattern matches the output field
        for pattern in compiled_patterns:
          if pattern.search(output_text):
            # Append the whole dictionary, not just the text
            matching_records.append(data)
            break

      except json.JSONDecodeError:
        continue

  return matching_records


def append_unique_records(new_records, existing_file):
  """Append new JSON objects that are not already in the existing file"""

  existing_hashes = set()

  # Read existing records to avoid duplicates
  # We use a string representation to check for uniqueness in the set
  try:
    with open(existing_file, 'r', encoding='utf-8') as f:
      for line in f:
        if line.strip():
          existing_hashes.add(line.strip())
  except FileNotFoundError:
    pass

  new_unique_count = 0
  with open(existing_file, 'a', encoding='utf-8') as f:
    for record in new_records:
      # Convert dict to string (ensuring keys are sorted for consistent hashing)
      record_string = json.dumps(record, ensure_ascii=False, sort_keys=True)

      if record_string not in existing_hashes:
        f.write(record_string + '\n')
        existing_hashes.add(record_string)
        new_unique_count += 1

  return new_unique_count

if __name__ == "__main__":
  input_file = "training_data_B1_syn.jsonl"
  output_file = "errors_checker.jsonl"
  
  # First extract [* p:w] and [* p:n] patterns (original task)
  patterns1 = [r'\[\* p:w\]', r'\[\* p:n\]']
  #sentences1 = extract_outputs_with_patterns(input_file, output_file, patterns1)
  
  # Then extract [* s:r:der] patterns and append unique ones
  #patterns2 = [r'\[\* s:r:der\]']
  #sentences2 = extract_outputs_with_patterns(input_file, output_file, patterns2)
  
  # Append only unique [* s:r:der] sentences
  #unique_added_der = append_unique_sentences(sentences2, output_file)
  
  # Now extract [* s:r:gc:pro] patterns and append unique ones
  #patterns3 = [r'\[\* s:r:gc:pro\]']
  #sentences3 = extract_outputs_with_patterns(input_file, output_file, patterns3)
  
  # Append only unique [* s:r:gc:pro] sentences
  #unique_added_gc_pro = append_unique_sentences(sentences3, output_file)

  # Now extract [* s:r] patterns and append unique ones
  #patterns4 = [r'\[\* s:r\]', r'\[\* s:ur\]']
  #sentences4 = extract_outputs_with_patterns(input_file, output_file, patterns4)

  # Append only unique [* s:r:gc:pro] sentences
  #unique_added_sr = append_unique_sentences(sentences4, output_file)

  # Now extract [* s:r] patterns and append unique ones
  #patterns5 = [r'\[\*s+p:]'
              # r"\[\* s:r:gc:der\]"
            #   ]
  #, r"\[\* m:\+\+ed:i\]", r"\[\* m:\+\+ed\]", r"\[\* m:\+\+er\]", r"\[\* m:\+\+ing\]"
  sentences5 = extract_full_lines_with_patterns(input_file, patterns1)

  # Append only unique [* s:r:gc:pro] sentences
  unique_added_plusplus = append_unique_records(sentences5, output_file)
