import sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.learning import crum_grade, tab_grade, rank_observations, augment
from coptic_examples.lexicon import Entry, apply_snapshot
from coptic_examples.crum_audit import extract_file
from coptic_examples.annotations import forbidden


class LearningTests(unittest.TestCase):
    def test_continuous_source_scales(self):
        values = [crum_grade(i, 16) for i in range(1, 17)]
        self.assertEqual(values[0], 3)
        self.assertEqual(values[-1], 1.5)
        self.assertTrue(all(a > b for a, b in zip(values, values[1:])))
        self.assertEqual([tab_grade(i) for i in [1, 2, 3]], [3, 2, 1])
        self.assertIsNone(tab_grade(None))

    def test_unranked_positive_retains_only_lower_bound(self):
        rows = rank_observations(
            [
                dict(y=1, ranks=[]),
                dict(y=1, ranks=[dict(source="tab", grade=2)]),
                dict(y=-1, ranks=[]),
            ]
        )
        self.assertTrue(rows[0]["bounded"])
        self.assertEqual(rows[1]["target"], 2)
        self.assertEqual(rows[2]["target"], -1)
        self.assertFalse(rows[2]["bounded"])

    def test_current_definitions_follow_tla_identity(self):
        lex = {
            t: Entry(t, "ⲉϩⲉ", "NOUN", "ox, cow", ("ⲉϩⲉ",)) for t in ["C771", "C772"]
        }
        apply_snapshot(
            lex,
            [
                dict(id=t, senses=[dict(translations=dict(en=[gloss]))], forms=[])
                for t, gloss in [("C771", "cow"), ("C772", "ox")]
            ],
        )
        self.assertEqual(lex["C771"].gloss, "cow")
        self.assertEqual(lex["C772"].gloss, "ox")
        self.assertEqual(lex["C771"].original_gloss, "ox, cow")
        self.assertTrue(lex["C771"].gloss_source.endswith("C771"))

    def test_all_overrides_existing_selection(self):
        self.assertTrue(forbidden({}, {"ALL"}, True))

    def test_citation_gender_narrows_only_its_own_reference(self):
        lex = {
            "cow": Entry("cow", "ⲉϩⲉ", "NOUN", "cow", ("ⲉϩⲉ",), genders=("f",)),
            "ox": Entry("ox", "ⲉϩⲉ", "NOUN", "ox", ("ⲉϩⲉ",), genders=("m",)),
        }
        text = "64a\n¶ ⟨ⲉϩⲉ⟩ ⟪S⟫ ⟨gloss: ox⟩ & ⟨gloss: cow⟩: ⟦Ge⟧{bible: Genesis 41:20} ⟦f⟧{annotation: feminine} ⟪S⟫, ⟦Cor⟧{bible: First Epistle to the Corinthians 9:9} ⟦m⟧{annotation: masculine} ⟪S⟫, ⟦Ex⟧{bible: Exodus 29:1} ⟪S⟫"
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "test.txt"
            p.write_text(text)
            rows = extract_file(
                p, lex, {"ⲉϩⲉ": ["cow", "ox"]}, {"cow": {"64a"}, "ox": {"64a"}}
            )
        self.assertEqual(
            [r["tla_candidates"] for r in rows], [["cow"], ["ox"], ["cow", "ox"]]
        )
        self.assertEqual(rows[0]["tla_binding"], "citation_gender")


class AugmentationTests(unittest.TestCase):
    def make(self):
        lex = {
            t: Entry(t, "ⲉϩⲉ", "NOUN", g, ("ⲉϩⲉ",))
            for t, g in [("a", "cow"), ("b", "ox")]
        }

        class Features:
            def rivals(self, e):
                return [lex[t] for t in lex if t != e.tla]

            def vector(self, c, e):
                return [e.tla]

        rows = []
        for t, g in [("a", "cow"), ("b", "ox")]:
            c = dict(
                id=t,
                sent="ⲉϩⲉ " + t,
                translation=g,
                dialect_id="sahidic",
                tokens=[dict(id="u1", form="ⲉϩⲉ", lemma="ⲉϩⲉ", pos="NOUN", xpos="N")],
            )
            rows.append(
                dict(
                    tla=t,
                    family="ⲉϩⲉ",
                    text_key=t,
                    candidate=c,
                    y=1,
                    ranks=[],
                    sources=["crum"],
                )
            )
        return lex, Features(), rows

    def test_cross_pairs_recompute_target_features_and_protect_positives(self):
        lex, ef, rows = self.make()
        added = augment(rows, 1, "hard", ef, lex, {})
        self.assertEqual(len(added), 2)
        for r in added:
            self.assertEqual(r["y"], -1)
            self.assertEqual(r["x"], [r["tla"]])
            self.assertNotEqual(r["tla"], r["source_tla"])
            self.assertNotIn(
                (r["tla"], r["text_key"]), {(x["tla"], x["text_key"]) for x in rows}
            )

    def test_dual_meaning_sentence_is_skipped(self):
        lex, ef, rows = self.make()
        for r in rows:
            r["candidate"]["translation"] = "cow and ox"
        self.assertEqual(augment(rows, 1, "hard", ef, lex, {}), [])

    def test_shared_dictionary_form_needs_shared_actual_target_token(self):
        lex, ef, rows = self.make()
        rows[1]["candidate"]["tokens"][0].update(form="ⲣⲱⲙⲉ", lemma="ⲣⲱⲙⲉ")
        added = augment(rows, 1, "hard", ef, lex, {})
        self.assertFalse(any(r["source_tla"] == "b" for r in added))


class DuplicateEvidenceTests(unittest.TestCase):
    def test_duplicate_text_keeps_later_source_citation_and_priority(self):
        from coptic_examples.learning import deduplicate_candidates

        a = dict(
            id="gold",
            sent="ⲉϩⲉ",
            crum_citations=[],
            manual_selected=False,
            manual_priority=None,
        )
        b = dict(
            id="other",
            sent="ⲉϩⲉ",
            crum_citations=[dict(article="Crum", rank=8)],
            manual_selected=True,
            manual_priority=1,
        )
        out, count = deduplicate_candidates([a, b])
        self.assertEqual(count, 1)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["crum_citations"], b["crum_citations"])
        self.assertTrue(out[0]["manual_selected"])
        self.assertEqual(out[0]["manual_priority"], 1)


class CompatibilityTests(unittest.TestCase):
    def test_source_syntax_supports_python_311(self):
        import ast

        root = Path(__file__).resolve().parents[1]
        for folder in ["src", "scripts"]:
            for path in (root / folder).rglob("*.py"):
                with self.subTest(path=path.relative_to(root)):
                    ast.parse(
                        path.read_text(), filename=str(path), feature_version=(3, 11)
                    )


class IntegrityTests(unittest.TestCase):
    def test_damaged_three_token_span_with_full_verse_is_rejected(self):
        from coptic_examples.integrity import alignment_issue

        c = dict(
            tokens=[{}, {}, {}],
            original_groups=["[]ⲙⲡⲉϥⲉⲓⲱⲧ"],
            translation="And if he have no brethren ye shall give his inheritance to his father's brother",
        )
        self.assertEqual(alignment_issue(c), "lacuna_fragment_with_full_translation")
        c["original_groups"] = ["ⲙⲡⲉϥⲉⲓⲱⲧ"]
        self.assertIsNone(alignment_issue(c))

    def test_title_subcategory_is_retained_and_disambiguates_gloss(self):
        from coptic_examples.lexicon import load

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "lex.xml"
            p.write_text(
                '<TEI><entry xml:id="title"><form><orth>ⲉⲓⲱⲧ</orth></form><gramGrp><pos>Subst.</pos><subc>Titel</subc></gramGrp><sense><quote xml:lang="en">father</quote></sense></entry></TEI>'
            )
            e = load(p)["title"]
        self.assertEqual(e.subcategories, ("Titel",))
        self.assertEqual(e.gloss, "father (title)")


class NegativeWeightTests(unittest.TestCase):
    def test_class_and_entry_totals(self):
        from coptic_examples.learning import sample_weights

        rows = (
            [dict(tla="common", y=-1)] * 100
            + [dict(tla="rare", y=-1)]
            + [dict(tla="positive", y=1)] * 5
        )
        w = sample_weights(rows, "entry_class")
        self.assertAlmostEqual(sum(w[:100]), w[100])
        self.assertAlmostEqual(sum(w[:101]), sum(w[101:]))


class OriginalSpanTests(unittest.TestCase):
    def test_parser_preserves_missing_text_markers(self):
        from coptic_examples.corpus import parse_tt
        from coptic_examples.integrity import alignment_issue

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sample.tt"
            p.write_text("""<meta languages="Sahidic">
<verse_n verse_n="10" translation="And if he have no brethren ye shall give his inheritance to his father brother">
<norm_group orig_group="[]ⲙⲡⲉϥⲉⲓⲱⲧ" norm_group="ⲙⲡⲉϥⲉⲓⲱⲧ">
<norm xml:id="u1" norm="ⲙ" lemma="ⲛ" pos="PREP">
<norm xml:id="u2" norm="ⲡⲉϥ" lemma="ⲡⲉϥ" pos="PPOS">
<norm xml:id="u3" norm="ⲉⲓⲱⲧ" lemma="ⲉⲓⲱⲧ" pos="N">
</verse_n>""")
            row = list(parse_tt(p, Path(d)))[0]
        self.assertTrue(row["has_lacuna"])
        self.assertEqual(row["original_groups"], ["[]ⲙⲡⲉϥⲉⲓⲱⲧ"])
        self.assertEqual(alignment_issue(row), "lacuna_fragment_with_full_translation")


class ModelOnlySelectionTests(unittest.TestCase):
    def test_tab_rules_and_source_flags_do_not_override_predictions(self):
        from unittest.mock import patch, MagicMock
        from collections import defaultdict
        import numpy as np
        from coptic_examples.learning import recommend
        from coptic_examples.config import Config

        entry = Entry("entry", "ⲉⲓⲱⲧ", "NOUN", "father", ("ⲉⲓⲱⲧ",))
        candidates = [
            dict(
                id=t,
                sent=s + " ⲉⲓⲱⲧ",
                translation="This is a translated example.",
                dialect_id="sahidic",
                tokens=[
                    dict(id="u0", form=s, lemma=s, pos="ADV", xpos="ADV"),
                    dict(id="u1", form="ⲉⲓⲱⲧ", lemma="ⲉⲓⲱⲧ", pos="NOUN", xpos="N"),
                ],
            )
            for t, s in [("a", "ⲁ"), ("b", "ⲃ"), ("c", "ⲅ")]
        ]
        model = MagicMock()
        model.metadata = {"thresholds": {"two_models": 0.5}}
        model.score.return_value = (
            np.array([0.8, 0.9, 0.4]),
            np.array([2.0, 3.0, 4.0]),
        )
        model.features.vectors.cosine.return_value = 0.4
        gate = MagicMock()
        gate.citations = defaultdict(list)
        manual = [
            dict(candidate=candidates[0], annotation={"priority": "1"}),
            dict(candidate=candidates[2], annotation={"priority": "2"}),
        ]
        with (
            tempfile.TemporaryDirectory() as d,
            patch("coptic_examples.learning.Corpus"),
            patch(
                "coptic_examples.learning.load_manual",
                return_value=[{"TLA": "entry", "URN": "X", "forbidden_urns": "ALL"}],
            ),
            patch(
                "coptic_examples.learning.native_candidates", return_value=candidates
            ),
            patch(
                "coptic_examples.annotations.resolve_manual", return_value=(manual, [])
            ),
            patch("coptic_examples.relevance.DefinitionGate", return_value=gate),
            patch("coptic_examples.learning.LearningModel", return_value=model),
            patch(
                "coptic_examples.learning.competing_senses",
                side_effect=lambda m, r, *args: r,
            ),
            patch(
                "coptic_examples.shortlist.numbers",
                return_value=([], {"features": {"information_score": 0}}, None),
            ),
        ):
            result = recommend(
                Config(work=Path(d)),
                {"entry": entry},
                entry,
                "sahidic",
                10,
                "two_models",
            )
        self.assertEqual([r["id"] for r in result["shortlist"]], ["b", "a"])


class TokenSenseTests(unittest.TestCase):
    def test_ambiguous_pairs_have_target_role_and_entry_type_inputs(self):
        from coptic_examples.token_senses import dataset, COLUMNS

        lex, features, rows = AugmentationTests().make()
        features.vector = lambda c, e: [0.0] * 62
        for r in rows:
            r["candidate"]["tokens"][0]["dep"] = "vocative"
        lex["a"].subcategories = ("Titel",)
        out = dataset(rows, features, lex)
        self.assertEqual(len(out), 4)
        self.assertEqual(sum(r["y"] for r in out), 2)
        self.assertTrue(all(len(r["x"]) == len(COLUMNS) for r in out))
        a = next(r for r in out if r["tla"] == "a")
        self.assertEqual(a["x"][62:64], [1.0, 1.0])

    def test_known_dual_positive_sentence_supplies_no_exclusive_pair(self):
        from coptic_examples.token_senses import dataset

        lex, features, rows = AugmentationTests().make()
        rows[1]["text_key"] = rows[0]["text_key"]
        self.assertEqual(dataset(rows, features, lex), [])


class SourceLabelTests(unittest.TestCase):
    def test_only_source_grade_and_printed_position_are_needed(self):
        from coptic_examples.ui import source_label

        row = dict(
            manual_selected=True,
            source_grades=[dict(source="tab", grade=3)],
            crum_citations=[dict(position=4, total=20)],
        )
        self.assertEqual(source_label(row), "TAB · 3.0 | Crum · 4/20")
