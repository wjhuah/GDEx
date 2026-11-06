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
