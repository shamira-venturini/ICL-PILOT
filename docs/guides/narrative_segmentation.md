# Generated narrative segmentation

Use the pinned Batchalign 3 English BERT word-boundary predictor defined in configs/segmentation/batchalign.json. Run it on each complete generated narrative before creating CHAT utterances. The worker is isolated from the generator and records runtime/model/tokenizer hashes.

The workflow is:

    whole narrative -> boundary prediction -> CHAT -> shared preparation and Chatter -> TalkTag -> preservation QC and Chatter -> Batchalign 3 UD

The CLI can split existing utterances but cannot merge boundaries that were imposed earlier. Do not pre-split generated prose by punctuation and do not resegment annotated CHAT. Real ENNI retains the supplied corpus boundaries.

Apply predicted word actions to original words. Preserve case, repetitions, grammatical forms and protected notation. Keep source offsets, window coverage, boundary actions and punctuation changes. Balance inline quotations without adding reporting words or quotation-induced utterances.

Before segmentation, convert curly single apostrophe glyphs (‘/’) to the ASCII
apostrophe required by CHAT, including inside contractions. Record every
replacement as `ascii_apostrophe` in the conversion manifest's
`formatting_changes`. Raw generation CSVs, prompt journals and source-text hashes
retain the exact model output. This formatting convention does not add, remove
or correct words; downstream preservation checks use the derived CHAT reference.

Malformed notation, inconsistent actions, incomplete word coverage or unresolvable scope/terminator conflicts stop conversion. Do not fall back to ASR, word reconstruction, grammar cleanup, punctuation-only splitting or another segmenter.

The pre-annotation CHAT is the fixed preservation reference. Chatter checks structural validity; source comparisons check words, boundaries, headers and tiers. These do not establish linguistic segmentation accuracy.

Manual review remains pending across ages, models, stages, fragments, repetitions, quotations and run-on structures. Compare with the intended ENNI unit convention and inspect utterance counts and MLU before scientific analyses.
