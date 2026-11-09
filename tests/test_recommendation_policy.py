import sys, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.recommendation_policy import (
    coptic_letters,
    parse_bias,
    exclusion_reason,
)
from coptic_examples.retrieval import Search, AmbiguousLemma
from coptic_examples.lexicon import Entry

TABLE = """| TLA ID | Lemma | Sahidic rare | Bohairic rare |
| --- | --- | --- | --- |
| C4327 | ⲧⲱⲙⲧ | T | F |
"""


class RecommendationPolicyTests(unittest.TestCase):
    def test_letter_count_and_boundary(self):
        self.assertEqual(coptic_letters("Ϩ ⲟ̅ⲟⲩ ."), 4)
        for n, want in [(30, None), (31, "over_30_coptic_letters")]:
            self.assertEqual(
                exclusion_reason(None, "C992", "sahidic", {"sent": "ⲁ" * n}, {}), want
            )

    def test_dialect_entry_review(self):
        flags = parse_bias(TABLE)
        self.assertEqual(
            exclusion_reason(None, "C4327", "sahidic", {"sent": "ⲁ"}, flags),
            "reviewed_dialect_restriction",
        )
        self.assertIsNone(
            exclusion_reason(None, "C4327", "bohairic", {"sent": "ⲁ"}, flags)
        )
        self.assertIsNone(
            exclusion_reason(None, "C4324", "sahidic", {"sent": "ⲁ"}, flags)
        )

    def test_dynamic_dialect_and_invalid_values(self):
        self.assertTrue(
            parse_bias(
                TABLE.replace("Bohairic rare", "Akhmimic rare").replace(
                    "| T | F |", "| T | T |"
                )
            )["C4327"]["akhmimic"]
        )
        with self.assertRaises(ValueError):
            parse_bias(TABLE.replace("| T | F |", "| maybe | F |"))

    def test_capitalized_search_preserves_ambiguity(self):
        search = Search.__new__(Search)
        search.lex = {
            t: Entry(t, "ϩⲟⲟⲩ", p, g, ("ϩⲟⲟⲩ",))
            for t, p, g in [("C6965", "NOUN", "day"), ("C6971", "VERB", "be bad")]
        }
        for q in ["Ϩⲟⲟⲩ", "ϩⲟⲟⲩ", "ϨⲞⲞⲨ"]:
            with self.assertRaises(AmbiguousLemma):
                search.resolve(q)
            self.assertEqual(search.resolve(q, upos="NOUN").tla, "C6965")
        self.assertEqual(search.resolve("c6971").tla, "C6971")

    def test_straightforward_contrasts_stay_distinct(self):
        from coptic_examples.lexicon import apply_snapshot

        lex = {
            t: Entry(t, "ⲉϩⲉ", "NOUN", "ox, cow", ("ⲉϩⲉ",)) for t in ["C771", "C772"]
        }
        apply_snapshot(
            lex,
            [
                dict(id=t, senses=[dict(translations=dict(en=[g]))], forms=[])
                for t, g in [("C771", "cow"), ("C772", "ox")]
            ],
        )
        self.assertEqual([lex[t].gloss for t in ["C771", "C772"]], ["cow", "ox"])
        document = (
            Path(__file__).resolve().parents[1] / "regression_vocabulary.md"
        ).read_text()
        for tla in [
            "C992",
            "C993",
            "C998",
            "C6965",
            "C6971",
            "C4324",
            "C4327",
            "C771",
            "C772",
            "C2788",
            "C2790",
            "C614",
            "C618",
        ]:
            self.assertIn(tla, document)

    def test_forest_imputes_missing_inputs(self):
        import numpy as np
        from coptic_examples.forest import fit

        rows = [
            dict(y=1 if i % 2 else -1, tla="C992", x=[float(i % 2), float("nan")])
            for i in range(40)
        ]
        model = fit(rows)
        prediction = model.predict_proba([[0, float("nan")], [1, float("nan")]])
        self.assertEqual(prediction.shape, (2, 2))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertGreater(prediction[1, 1], prediction[0, 1])

    def test_long_current_error(self):
        self.assertEqual(
            exclusion_reason(None, "C6971", "sahidic", {"sent": "ⲡⲁϣⲏⲣⲉ " * 20}, {}),
            "over_30_coptic_letters",
        )


if __name__ == "__main__":
    unittest.main()
