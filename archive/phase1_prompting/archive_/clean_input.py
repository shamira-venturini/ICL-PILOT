import json
import re


def strip_clan_annotations(text):
    """
    Removes all CLAN/CHAT markers from a string.
    """
    # 1. Remove words starting with 0 (omissions like 0subj, 0prep)
    text = re.sub(r'\b0\w+\b', '', text)

    # 2. Remove fillers/fragments starting with &+ or &- (e.g., &+ah, &-um)
    text = re.sub(r'&[+-]\w+', '', text)

    # 3. Remove content inside square brackets (e.g., [/], [//], [* m:0ed])
    text = re.sub(r'\[.*?\]', '', text)

    # 4. Remove angle brackets (keep the text inside)
    text = re.sub(r'[<>]', '', text)

    # 5. Remove parentheses but KEEP the text inside (e.g., (a)nother -> another)
    text = re.sub(r'[\(\)]', '', text)

    # 6. Remove special CHAT symbols like punctuation markers [!] [?] if they aren't inside []
    # (Step 3 usually covers this, but just in case)

    # 7. Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def clean_jsonl_file(input_file, output_file):
    processed_count = 0

    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                # Clean the 'input' sentence
                if 'input' in data:
                    data['input'] = strip_clan_annotations(data['input'])

                # Write the updated JSON object back
                outfile.write(json.dumps(data) + '\n')
                processed_count += 1

            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {e}")

    print(f"Successfully cleaned {processed_count} sentences.")
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    # Update these filenames as needed
    INPUT_JSONL = "training_data_clean_stripped.jsonl"
    OUTPUT_JSONL = "training_data_clean_stripped2.jsonl"

    clean_jsonl_file(INPUT_JSONL, OUTPUT_JSONL)