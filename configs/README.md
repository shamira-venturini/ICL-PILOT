# Configuration status

The previous study is frozen externally. The revised protocol is pending.

| Folder | Current status |
| --- | --- |
| [`generation/`](generation/) | Revised story counterbalancing and a pinned official Llama 3.1 8B Instruct checkpoint; both stages use temperature 0.8, top-p 0.9 and a 640-new-token ceiling. Prompts, roster, scaffolds and exemplar inputs retain the earlier baseline. |
| [`annotation/`](annotation/) | Pinned portable TalkTag source/options for new runs. Review annotation choices for the revised protocol. |
| [`analysis/`](analysis/) | Five explicit R input paths, all pending. |
| [`evaluation/`](evaluation/) | Two explicit framework input paths, both pending. |

The working generation identifier is `enni_staged_4_to_7_counterbalance_v2`.
This includes the revised assignments, official model-source correction and
the 640-token output ceiling adopted on 21 October 2026, followed by the author's
selection of temperature 0.8 and top-p 0.9 for both stages. The rest of the revised
protocol remains pending. Retained child pairing, prompts, other decoding
settings and evaluation code still require review for the next study. Original assets remain
recoverable from the complete study freeze; see the archive index in
`docs/provenance/`.
