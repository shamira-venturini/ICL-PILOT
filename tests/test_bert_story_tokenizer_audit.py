import os
from pathlib import Path
import tempfile
import unittest

from icl_pilot.maintenance.bert_story_tokenizer_audit import audit, measure, summarize, VocabularyReference


class FakeTokenizer:
    sep_token_id = 102

    def __call__(self, text, add_special_tokens, truncation, verbose=False, max_length=None):
        ids = [101] + list(range(1000, 1000+len(text.split()))) + [102]
        if truncation and len(ids) > max_length:
            ids = ids[:max_length-1] + [102]
        return {"input_ids": ids}

    def num_special_tokens_to_add(self, pair=False):
        return 2


class TestBertAudit(unittest.TestCase):
    def test_special_tokens_determine_boundary(self):
        tokenizer = FakeTokenizer()
        exact = measure("dog "*510, tokenizer, tokenizer, 512)
        above = measure("dog "*511, tokenizer, tokenizer, 512)
        self.assertEqual(exact["original_tokens_including_special"], 512)
        self.assertFalse(exact["would_truncate"])
        self.assertEqual(above["original_tokens_including_special"], 513)
        self.assertTrue(above["would_truncate"])
        self.assertEqual(above["removed_content_tokens"], 1)
        self.assertAlmostEqual(above["removed_content_fraction"], 1/511)

    def test_disagreement_is_not_silently_accepted(self):
        class OtherTokenizer(FakeTokenizer):
            def __call__(self, *args, **kwargs):
                return {"input_ids": [101, 999, 102]}
        with self.assertRaisesRegex(ValueError, "disagree"):
            measure("dog", FakeTokenizer(), OtherTokenizer(), 512)

    def test_empty_text_and_empty_strata_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            measure(" ", FakeTokenizer(), FakeTokenizer(), 512)
        with self.assertRaisesRegex(ValueError, "empty"):
            summarize([])

    def test_children_and_story_counts_are_distinct(self):
        rows = [{"source_child_id": "725", **measure("dog "*n, FakeTokenizer(), FakeTokenizer(), 512)} for n in (510, 602)]
        result = summarize(rows)
        self.assertEqual((result["stories"], result["children"]), (2, 1))
        self.assertEqual(result["stories_above_limit"], 1)
        self.assertEqual(result["removed_content_tokens"], 92)
        self.assertAlmostEqual(result["percent_content_tokens_removed"], 100*92/1112)

    def test_existing_evidence_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            with self.assertRaises(FileExistsError):
                audit(path/"missing", path/"missing", path)

    @unittest.skipUnless(os.environ.get("BERT_AUDIT_TOKENIZER_DIR"), "Pinned tokenizer assets supplied explicitly")
    def test_pinned_assets_agree_and_preserve_special_tokens(self):
        from transformers import BertTokenizerFast
        path = Path(os.environ["BERT_AUDIT_TOKENIZER_DIR"])
        tokenizer = BertTokenizerFast.from_pretrained(str(path), local_files_only=True)
        reference = VocabularyReference(path/"vocab.txt")
        result = measure("dog "*511, tokenizer, reference, 512)
        self.assertEqual(result["original_tokens_including_special"], 513)
        self.assertEqual(result["removed_content_tokens"], 1)
        measure("her (be)cause runned . café", tokenizer, reference, 512)


if __name__ == "__main__":
    unittest.main()
