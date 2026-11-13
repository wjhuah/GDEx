import json, unittest
from pathlib import Path


class Top20VocabularyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = json.loads(
            (Path(__file__).parent / "fixtures/homographs_top20.json").read_text()
        )
        cls.byword = {r["word"]: r for r in cls.rows}

    def test_twenty_distinct_words_and_required_seed_words(self):
        self.assertEqual(len(self.rows), 20)
        self.assertEqual(len(self.byword), 20)
        self.assertTrue(
            {"ⲉⲓⲱⲧ", "ϩⲟⲟⲩ", "ⲧⲱⲙⲧ", "ⲉϩⲉ", "ⲡⲉ", "ⲉ"} <= self.byword.keys()
        )

    def test_every_row_has_explicit_dialect_overlap_and_distinct_ids(self):
        for r in self.rows:
            with self.subTest(word=r["word"]):
                self.assertGreaterEqual(len(r["senses"]), 2)
                self.assertEqual(len(r["senses"]), len({s["tla"] for s in r["senses"]}))
                self.assertTrue(r["shared_dialects"])
                self.assertTrue(all(s["gloss"] for s in r["senses"]))
                self.assertFalse(r["automatic_negative_labels"])

    def test_same_pos_lexical_contrasts_are_retained(self):
        for word, first, second in [
            ("ⲉⲓⲱⲧ", "C992", "C998"),
            ("ⲉϩⲉ", "C771", "C772"),
            ("ϯ", "C4020", "C4050"),
            ("ⲉϩⲣⲁⲓ", "C6769", "C6775"),
        ]:
            senses = {s["tla"]: s for s in self.byword[word]["senses"]}
            self.assertEqual(senses[first]["pos"], senses[second]["pos"])
            self.assertNotEqual(senses[first]["gloss"], senses[second]["gloss"])

    def test_bohairic_noun_verb_case_is_kept(self):
        r = self.byword["ⲥⲟⲙⲥ"]
        ss = {s["tla"]: s for s in r["senses"]}
        self.assertEqual(set(r["shared_dialects"]), {"bohairic"})
        self.assertEqual(ss["C3553"]["gloss"], "looking")
        self.assertNotEqual(ss["C3551"]["pos"], ss["C3553"]["pos"])


if __name__ == "__main__":
    unittest.main()
