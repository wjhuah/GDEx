import sys, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.token_highlight import render
from coptic_examples.shortlist import fit_linear, decision


class DefinitionTests(unittest.TestCase):
    def test_same_surface_aux_and_two_verbs_have_exact_offsets(self):
        ts = [
            dict(id="u97", form="ϣⲁ"),
            dict(id="u98", form="ϥ"),
            dict(id="u99", form="ϣⲁ"),
            dict(id="u115", form="ϣⲁ"),
        ]
        out = render("ϣⲁϥϣⲁ ϣⲁ", ts, ["u99", "u115"])
        self.assertTrue(out.startswith("ϣⲁϥ<mark"))
        self.assertEqual(out.count("<mark"), 2)
        self.assertIn('data-token-id="u99"', out)
        self.assertIn('data-token-id="u115"', out)

    def test_linear_pair_model_supports_ud_feature_subset(self):
        m = fit_linear([[1, 0], [-1, 0], [2, 1], [-2, -1]], [1, 0, 1, 0], False)
        self.assertGreater(decision(m, [[1, 0]])[0], 0)


class GateTests(unittest.TestCase):
    def fixture(self):
        from types import SimpleNamespace
        from collections import defaultdict
        from coptic_examples.relevance import DefinitionGate

        g = DefinitionGate.__new__(DefinitionGate)
        g.entry = SimpleNamespace(
            tla="own", forms=["ϣⲁ"], gloss="begin", pos="VERB", form_dialects={}
        )
        g.rivals = [
            SimpleNamespace(
                tla="rival", forms=["ϣⲁ"], gloss="rise", pos="VERB", form_dialects={}
            )
        ]
        g.vectors = SimpleNamespace(cosine=lambda a, b: 0.9 if a == "begin" else 0.92)
        g.citations = defaultdict(list)
        c = dict(
            id="s",
            dialect_id="sahidic",
            translation="the sun arises",
            tokens=[dict(id="u1", form="ϣⲁ", lemma="ϣⲁ", pos="VERB", xpos="V")],
        )
        return g, c

    def test_competing_crum_source_precedes_high_cosine(self):
        g, c = self.fixture()
        g.citations["s"] = [dict(tla="rival")]
        self.assertEqual(g.check(c)["reason"], "crum_supports_competing_definition")
        self.assertEqual(g.check(c)["status"], "review")

    def test_own_crum_source_precedes_similarity(self):
        g, c = self.fixture()
        g.citations["s"] = [dict(tla="own")]
        self.assertEqual(g.check(c)["reason"], "crum_definition_citation")

    def test_higher_competing_cosine_stays_in_review(self):
        g, c = self.fixture()
        self.assertEqual(g.check(c)["status"], "review")

    def test_manual_definition_evidence_is_preserved(self):
        g, c = self.fixture()
        self.assertEqual(
            g.check(c, manual_confirmed=True)["reason"], "manual_definition_example"
        )

    def test_explicit_crum_pos_alignment_preserves_original_tag(self):
        from coptic_examples.sense import targets

        g, c = self.fixture()
        c["tokens"][0]["pos"] = "NOUN"
        c["crum_pos_alignment"] = {"own": ["u1"]}
        self.assertEqual(targets(c, g.entry), [0])
        self.assertEqual(c["tokens"][0]["pos"], "NOUN")


class InclusionFeatureTests(unittest.TestCase):
    def fixture(self, missing=False):
        from types import SimpleNamespace
        from collections import defaultdict
        from coptic_examples.inclusion import EntryFeatures

        ef = EntryFeatures.__new__(EntryFeatures)
        a = SimpleNamespace(tla="father", forms=["ⲉⲓⲱⲧ"], gloss="father")
        b = SimpleNamespace(tla="barley", forms=["ⲉⲓⲱⲧ"], gloss="barley")
        ef.lex = {"father": a, "barley": b}
        ef.byform = defaultdict(set, {"ⲉⲓⲱⲧ": {"father", "barley"}})
        ef.memo = {}
        ef.vectors = SimpleNamespace(
            cosine=lambda gloss, text: (
                None if missing else 0.8 if gloss == "father" else 0.3
            )
        )
        c = dict(
            id="fixture",
            translation="his father came",
            tokens=[
                dict(xpos="PPERS"),
                dict(xpos="PPERI"),
                dict(xpos="V"),
                dict(xpos="N"),
            ],
        )
        return ef, a, b, c

    def test_dictionary_entry_changes_features_for_identical_sentence(self):
        from coptic_examples.inclusion import COLUMNS_INCLUSION

        ef, a, b, c = self.fixture()
        x = ef.vector(c, a, [0.0] * 44)
        y = ef.vector(c, b, [0.0] * 44)
        self.assertEqual(len(x), len(COLUMNS_INCLUSION))
        self.assertEqual(x[:44], y[:44])
        self.assertGreater(x[COLUMNS_INCLUSION.index("sense_cosine_margin")], 0)
        self.assertLess(y[COLUMNS_INCLUSION.index("sense_cosine_margin")], 0)

    def test_missing_semantic_cache_is_explicit(self):
        import math
        from coptic_examples.inclusion import COLUMNS_INCLUSION

        ef, a, b, c = self.fixture(True)
        x = ef.vector(c, a, [0.0] * 44)
        self.assertTrue(math.isnan(x[COLUMNS_INCLUSION.index("sense_cosine")]))
        self.assertEqual(x[COLUMNS_INCLUSION.index("sense_cosine_missing")], 1.0)

    def test_bound_and_independent_pronouns_have_separate_columns(self):
        from coptic_examples.inclusion import COLUMNS_INCLUSION

        ef, a, b, c = self.fixture()
        x = ef.vector(c, a, [0.0] * 44)
        self.assertEqual(
            x[COLUMNS_INCLUSION.index("bound_personal_pronoun_ratio")], 0.25
        )
        self.assertEqual(
            x[COLUMNS_INCLUSION.index("independent_personal_pronoun_ratio")], 0.25
        )


if __name__ == "__main__":
    unittest.main()
