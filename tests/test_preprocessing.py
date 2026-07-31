import unittest

from project_utils import preprocess_comment


class PreprocessingTests(unittest.TestCase):
    def test_preserves_negation_words(self):
        processed = preprocess_comment("This is not good but not terrible either")
        self.assertIn("not", processed.split())
        self.assertIn("but", processed.split())

    def test_handles_none(self):
        self.assertEqual(preprocess_comment(None), "")

    def test_removes_newlines_and_special_characters(self):
        processed = preprocess_comment("Hello!\nWorld@@")
        self.assertEqual(processed, "hello! world")


if __name__ == "__main__":
    unittest.main()
