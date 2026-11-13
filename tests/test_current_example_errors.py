import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.recommendation_policy import (
    exclusion_reason,
    coptic_letters,
    parse_bias,
)
from coptic_examples.lexicon import Entry
from coptic_examples.retrieval import Search, AmbiguousLemma


class CurrentExampleErrors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = {
            r["case"]: r
            for r in json.loads(
                (
                    Path(__file__).parent / "fixtures/current_example_errors.json"
                ).read_text()
            )
        }
        cls.flags = parse_bias(
            (Path(__file__).resolve().parents[1] / "dialect_bias.md").read_text()
        )

    def test_actual_proverbs_9_12_is_rejected_for_length(self):
        c = self.rows["length_error"]["candidate"]
        self.assertTrue(c["id"].endswith("Proverbs_09.tt#12"))
        self.assertEqual(len(c["sent"].split()), 43)
        self.assertEqual(
            exclusion_reason(None, "C6971", "sahidic", c, {}), "over_30_bound_groups"
        )

    def test_actual_acts_20_14_is_excluded_from_c4327(self):
        c = self.rows["meet_not_amazed"]["candidate"]
        self.assertIn("met us", c["translation"])
        self.assertEqual(
            exclusion_reason(None, "C4327", "sahidic", c, self.flags),
            "reviewed_dialect_restriction",
        )
        self.assertFalse(self.flags.get("C4324", {}).get("sahidic", False))

    def test_actual_bohairic_amazed_keeps_separate_dialect(self):
        c = self.rows["amazed_bohairic"]["candidate"]
        self.assertEqual(c["dialect_id"], "bohairic")
        self.assertIn("amazed", c["translation"])
        self.assertNotEqual(
            exclusion_reason(None, "C4327", "bohairic", c, self.flags),
            "reviewed_dialect_restriction",
        )

    def test_father_title_and_barley_preserve_three_choices(self):
        search = Search.__new__(Search)
        search.lex = {
            t: Entry(t, "ⲉⲓⲱⲧ", "NOUN", gloss, ("ⲉⲓⲱⲧ",))
            for t, gloss in [
                ("C992", "father"),
                ("C993", "father (title)"),
                ("C998", "barley"),
            ]
        }
        for q in ["ⲉⲓⲱⲧ", "ⲈⲒⲰⲦ"]:
            with self.assertRaises(AmbiguousLemma) as ctx:
                search.resolve(q)
            self.assertEqual(
                {r["tla"] for r in ctx.exception.entries}, {"C992", "C993", "C998"}
            )


if __name__ == "__main__":
    unittest.main()
