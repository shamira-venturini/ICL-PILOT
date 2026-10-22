# Real ENNI narrative lengths across tokenizers

Status on 21 October 2026: all four checks are complete, including the pinned
official Llama tokenizer. A public tokenizer-only copy matches the official
file's Git blob hash and size exactly. This enabled local verification without
authentication or Colab execution. The earlier Colab request is superseded.
No generation limit, prompt, model selection, source narrative or annotation
was changed by this comparison. The author subsequently adopted 640 new tokens
for both stages; that separate implementation is recorded in
[`generation_ceiling_update_2026-10-21.json`](../generation_ceiling_update_2026-10-21.json).
The tables and `ceilings_changed: false` in the comparison JSON describe the
audit itself, before that authorized configuration change.

## Source and scope

The input is the frozen
[`story_unit_narratives_chat_v2.csv`](../../../configs/generation/inputs/story_unit_narratives_chat_v2.csv).
Its SHA-256 is
`89643c6ca494e525b47eb3863a36bea3d702402c9dd1e3e5d42b401fa9828cbc`,
verified against its parsing-QC sidecar. It contains 2,166 individual story
records from 361 CHAT files: 2,160 records have ages four through nine; six
retain unknown age/group metadata. The age-four-to-seven subset has 1,428
stories. Each count concerns one child's one story, not a six-story bundle.

[`comparison.json`](comparison.json) records tokenizer
revisions, file hashes, source URLs, software versions, summary statistics and
the exact descriptive-cleanup function. [`per_story.csv`](per_story.csv)
contains all checked counts and original text hashes, without full narratives.
The earlier partial tables are preserved in the external archive identified in
the completed report; they have been replaced locally by the completed tables.

## Completed comparisons

| Tokenizer | Ages 4–7: stored CHAT surface | Ages 4–7: light cleanup | All saved: stored CHAT surface | All saved: light cleanup |
| --- | ---: | ---: | ---: | ---: |
| Historical community-model reference | 568 | 556 | 643 | 604 |
| Proposed Qwen3-8B | 568 | 556 | 643 | 604 |
| Proposed Qwen3-VL-8B-Instruct, text only | 568 | 556 | 643 | 604 |
| Pinned official Llama 3.1 8B Instruct | 568 | 556 | 643 | 604 |

These are maxima. Equal maxima do not mean identical tokenization: Qwen3-8B
differs from the historical reference on 110 of 2,166 stored-text counts (83 of
1,428 at ages four through seven), with differences ranging from −5 to +6.
The two Qwen tokenizers give identical counts on every checked story.
The official Llama tokenizer and historical reference give identical counts on
every checked story, in both representations; their ordered count fingerprints
match. This does not establish equivalence of the models or their chat templates.

All completed checks place the study-age maximum at SLI-labelled child 725,
age `7;10.30`, story B3. The full-snapshot maximum is TD child 905, age
`9;05.05`, B3. A 640-token budget clears all study-age texts, including their
saved notation. One nine-year-old stored text exceeds it (643 tokens); the
earlier approximately 6% buffer above the full-snapshot maximum referred to
lightly cleaned text (604 tokens), not its saved CHAT surface.

The counts use `tokenizers.Tokenizer.encode(text, add_special_tokens=False)`.
They exclude input prompts, chat-template wrappers, pictures and generated
end-of-turn tokens. An emitted stopping token also consumes output budget.
They do not establish how generated narratives will terminate. No weights were
downloaded, no GPU inference was run, and proposed Qwen conditions were not
adopted by this check.

The cleanup is diagnostic only. It preserves repetitions and grammatical forms,
joins letters in parenthesized word segments, removes bracketed annotations,
non-speech codes, pause notation, unintelligibility placeholders and CHAT
quotation markers, removes internal lengthening colons, and collapses whitespace.
It is the same light cleanup used for the preceding length estimate, not a
replacement CHAT parser, a CLAN word-count method or a modification to generation
inputs. Some surface punctuation remains. The unmodified stored-text count is
also reported to make that distinction inspectable.

## Official file identity and reproduction

The pinned official repository's public metadata identifies `tokenizer.json` as
9,085,657 bytes with Git blob SHA-1
`5cc5f00a5b203e90a27a3bd60d1ec393b07971e8`.
The public tokenizer-only repository
[`Xenova/Meta-Llama-3.1-Tokenizer`](https://huggingface.co/Xenova/Meta-Llama-3.1-Tokenizer/tree/d49ea33e96b3fcf8738bb79ff9d85da81742fa82)
lists the same blob. Downloaded bytes were independently hashed and matched
the official record. Their SHA-256 is
`79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4`.
Both immutable metadata URLs and the download URL are recorded in
`comparison.json`. The active model remains the official Meta checkpoint;
only these verified identical tokenizer bytes came from the public copy.

The following code loads that exact tokenizer without a login. Run from the
repository root with `huggingface_hub` and a compatible `tokenizers` installed
(the audit used 0.22.2). It reproduces the primary stored-text maxima. The exact
cleanup function for the second representation is stored in `comparison.json`.

```python
import csv, hashlib, json
from pathlib import Path
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

p = Path('configs/generation/inputs/story_unit_narratives_chat_v2.csv')
assert hashlib.sha256(p.read_bytes()).hexdigest() == '89643c6ca494e525b47eb3863a36bea3d702402c9dd1e3e5d42b401fa9828cbc'
f = Path(hf_hub_download('Xenova/Meta-Llama-3.1-Tokenizer', 'tokenizer.json',
    revision='d49ea33e96b3fcf8738bb79ff9d85da81742fa82', token=False))
b = f.read_bytes()
assert hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest() == '5cc5f00a5b203e90a27a3bd60d1ec393b07971e8'
t = Tokenizer.from_file(str(f))
rows = list(csv.DictReader(p.open(encoding='utf-8', newline='')))
n = [len(t.encode(r['story_text'], add_special_tokens=False).ids) for r in rows]
result = {'tokenizer_sha256': hashlib.sha256(b).hexdigest(),
          'all_counts_sha256': hashlib.sha256(json.dumps(n, separators=(',', ':')).encode()).hexdigest()}
for label, indices in [('all_saved', list(range(len(rows)))), ('ages_4_to_7',
    [i for i,r in enumerate(rows) if r['Age_Month'] and 48 <= int(r['Age_Month']) <= 95])]:
    i = max(indices, key=lambda i: n[i])
    result[label] = {'stories': len(indices), 'max_text_tokens': n[i],
        'at_or_over_640': sum(n[j] >= 640 for j in indices),
        'longest': {k: rows[i][k] for k in ['File_ID','Group','Age_Raw','story_id']}}
print(json.dumps(result, indent=2))
```
