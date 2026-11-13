import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.spelling_compatibility import SpellingCompatibility
from coptic_examples.citation_recovery import token_spans
from coptic_examples.lexicon import Entry


class SpellingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        table = Path(self.directory.name) / "table.tab"
        table.write_text(
            "ⲓⲥ\tⲓⲏⲥⲟⲩⲥ\nⲡⲛⲁ\tⲡⲛⲉⲩⲙⲁ\nⲡⲛⲁ\tⲡⲛⲁ\nⲉⲣ\tⲣ\nⲁ\tⲁⲁ\nⲁ ⲃ\tⲁⲃ\nⲙⲉϣ\tⲛⲉϣ\nⲣⲣⲟⲙⲡⲉ\tⲛⲣⲟⲙⲡⲉ\nⲣⲟⲩⲟⲓⲛ\tⲣⲟⲩⲟⲉⲓⲛ\n"
        )
        self.normalizer = SpellingCompatibility(table)

    def test_case_and_combining_mark(self):
        self.assertEqual(self.normalizer.key("Ϩⲟⲟⲩ̅"), self.normalizer.key("ϩⲟⲟⲩ"))

    def test_unique_abbreviation(self):
        self.assertEqual(self.normalizer.key("ⲓⲥ"), "ⲓⲏⲥⲟⲩⲥ")

    def test_ambiguous_abbreviation_is_retained(self):
        self.assertEqual(self.normalizer.key("ⲡⲛⲁ"), "ⲡⲛⲁ")
        self.assertEqual(self.normalizer.ambiguous, 1)

    def test_short_forms_and_boundaries_are_protected(self):
        self.assertEqual(self.normalizer.key("ⲉⲣ"), "ⲉⲣ")
        self.assertEqual(self.normalizer.key("ⲁ"), "ⲁ")
        self.assertNotIn("ⲁⲃ", self.normalizer.mapping)

    def test_consonant_changes_are_retained(self):
        self.assertEqual(self.normalizer.key("ⲙⲉϣ"), "ⲙⲉϣ")
        self.assertEqual(self.normalizer.key("ⲣⲣⲟⲙⲡⲉ"), "ⲣⲣⲟⲙⲡⲉ")
        self.assertEqual(self.normalizer.key("ⲣⲟⲩⲟⲓⲛ"), "ⲣⲟⲩⲟⲉⲓⲛ")

    def test_full_token_alias_matches_without_editing_tokens(self):
        candidate = {"tokens": [{"id": "u1", "form": "ⲓⲥ", "lemma": "ⲓⲥ"}]}
        entry = Entry("C1", "ⲓⲏⲥⲟⲩⲥ", "PROPN", "", ("ⲓⲏⲥⲟⲩⲥ",))
        self.assertEqual(token_spans(candidate, entry), [])
        self.assertEqual(
            token_spans(candidate, entry, normalizer=self.normalizer.key), [["u1"]]
        )
        self.assertEqual(candidate["tokens"][0]["form"], "ⲓⲥ")

    def test_punctuation_and_substrings_still_excluded(self):
        entry = Entry("C1", "ⲓⲏⲥⲟⲩⲥ", "PROPN", "", ("ⲓⲏⲥⲟⲩⲥ",))
        for forms in [("ⲓ", ".", "ⲥ"), ("ⲁⲓⲥ",)]:
            candidate = {
                "tokens": [
                    {"id": f"u{i}", "form": form} for i, form in enumerate(forms)
                ]
            }
            self.assertEqual(
                token_spans(candidate, entry, normalizer=self.normalizer.key), []
            )


if __name__ == "__main__":
    unittest.main()
