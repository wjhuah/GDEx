import sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.editorial import order
from coptic_examples.sense import align_candidate
from coptic_examples.lexicon import Entry
from coptic_examples.crum_audit import extract_file
from coptic_examples.ui import search_page
from coptic_examples.config import Config


class EditorialTests(unittest.TestCase):
    def test_recorded_crum_order_precedes_model_score(self):
        rows = [
            dict(id="new", shortlist_score=0.99),
            dict(
                id="second",
                shortlist_score=0.8,
                crum_citations=[dict(article="a", rank=2)],
            ),
            dict(
                id="first",
                shortlist_score=0.1,
                crum_citations=[dict(article="a", rank=1)],
            ),
        ]
        self.assertEqual([r["id"] for r in order(rows)], ["first", "second", "new"])

    def test_pos_conflict_preserves_tt_tag_and_token(self):
        e = Entry("noun", "ⲧⲱⲙⲧ", "NOUN", "event", ("ⲧⲱⲙⲧ",))
        c = dict(
            dialect_id="sahidic",
            tokens=[dict(id="u1", form="ⲧⲱⲙⲛⲧ", lemma="ⲧⲱⲙⲧ", pos="VERB", xpos="V")],
        )
        self.assertEqual(align_candidate(c, e, True), [0])
        self.assertTrue(c["target_pos_mismatch"])
        self.assertEqual(c["tokens"][0]["xpos"], "V")

    def test_home_has_empty_query_and_no_sample_entries(self):
        page = search_page(Config())
        self.assertIn('value=""', page)
        self.assertNotIn("Sample Entries", page)
        self.assertNotIn("C998", page)

    def test_ambiguous_choices_keep_dialect_and_link_entries(self):
        e = dict(
            tla="C6965", lemma="ϩⲟⲟⲩ", pos="NOUN", gloss="day", forms=["ϩⲟⲟⲩ", "ⲉϩⲟⲟⲩ"]
        )
        page = search_page(Config(), choices=[e], query="ϩⲟⲟⲩ", dialect="bohairic")
        self.assertIn("query=C6965&amp;dialect=bohairic", page)
        self.assertIn("day", page)
        self.assertIn("ⲉϩⲟⲟⲩ", page)

    def test_dash_noun_heading_has_own_binding_and_variant_scope(self):
        verb = Entry("verb", "ⲧⲱⲙⲧ", "VERB", "meet befall", ("ⲧⲱⲙⲧ",))
        noun = Entry("noun", "ⲧⲱⲙⲧ", "NOUN", "meeting event", ("ⲧⲱⲙⲧ",))
        text = "416b\n¶ ⟨ⲧⲱⲙⲧ⟩ ⟪S⟫ ⟨gloss: meet⟩ ⟦Ge⟧{bible: Genesis 1:1}\n¶ ― ⟦nn⟧{annotation: noun} ⟪S⟫ ⟨gloss: event⟩ ⟦Eccl⟧{bible: Ecclesiastes 2:14}, ⟦9 2⟧{bible: Ecclesiastes 9:2} (⟪F⟫ ⟨ϫⲓⲛⲕⲱⲗϩ⟩)"
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sample.txt"
            p.write_text(text)
            rows = extract_file(
                p,
                {"verb": verb, "noun": noun},
                {"ⲧⲱⲙⲧ": ["verb", "noun"]},
                {"verb": {"416b"}, "noun": {"416b"}},
            )
        self.assertEqual(rows[0]["tla_candidates"], ["verb"])
        self.assertEqual(rows[1]["tla_candidates"], ["noun"])
        self.assertIn("S", rows[2]["dialect_codes"])
        self.assertIn("F", rows[2]["dialect_codes"])


if __name__ == "__main__":
    unittest.main()


class TypicalityTests(unittest.TestCase):
    def test_counts_exclude_current_span(self):
        import math
        from coptic_examples.typicality import Typicality

        e = Entry("verb", "ⲧⲱⲙⲧ", "VERB", "meet", ("ⲧⲱⲙⲧ",))
        c = dict(
            dialect_id="sahidic",
            tokens=[
                dict(
                    id="u1",
                    form="ⲧⲱⲙⲧ",
                    lemma="ⲧⲱⲙⲧ",
                    pos="VERB",
                    xpos="V",
                    head="",
                    dep="root",
                ),
                dict(
                    id="u2",
                    form="ⲡⲱϣⲥ",
                    lemma="ⲡⲱϣⲥ",
                    pos="NOUN",
                    xpos="N",
                    head="u1",
                    dep="obj",
                ),
            ],
        )
        stats = Typicality.__new__(Typicality)
        stats.frames = {("ⲧⲱⲙⲧ", "VERB", "root", "obj"): 3}
        stats.pairs = {("ⲧⲱⲙⲧ", "child:obj", "ⲡⲱϣⲥ"): 2}
        stats.totals = {"ⲧⲱⲙⲧ": 10}
        vector = stats.vector(c, e)
        self.assertAlmostEqual(vector[0], math.log(3))
        self.assertAlmostEqual(vector[1], 2 / 9)
        self.assertAlmostEqual(vector[2], math.log(2))
