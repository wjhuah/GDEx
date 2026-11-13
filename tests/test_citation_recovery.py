import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.citation_recovery import (
    expanded_heading_forms,
    bind_forms,
    token_spans,
    has_lexical_heading,
    has_dialect_alternative,
    explicit_dialect_forms,
    add_cdo_only_entries,
    contextual_dialect_reading,
    explicit_bible_location,
    has_government_alternative,
)
from coptic_examples.lexicon import Entry
from coptic_examples.crum_audit import preceding_dialect_codes


class CitationRecoveryTests(unittest.TestCase):
    def test_derivative_expands_article_root(self):
        forms = expanded_heading_forms("¶ ⟨ⲣ⟩, ⟨ⲉⲣ ⲟⲩ.⟩, ⟨gloss: shine⟩:", "ⲟⲩⲟⲉⲓⲛ")
        self.assertIn("ⲣⲟⲩⲟⲉⲓⲛ", forms)
        self.assertNotIn("ⲣ", forms)

    def test_gloss_continuation_keeps_lexical_parent(self):
        self.assertFalse(has_lexical_heading("¶ ⟨gloss: light⟩ of eyes"))
        self.assertTrue(has_lexical_heading("¶ ⟨ⲕⲟⲧ⟩, ⟨gloss: basket⟩"))

    def test_parenthetical_reading_is_scoped_to_its_dialect(self):
        raw = "⟦Eph 1 18⟧{bible: Ephesians 1:18}"
        para = raw + " ⟪B⟫ (⟪S⟫ ⟨ϫⲓ ⲟⲩ.⟩), "
        self.assertTrue(has_dialect_alternative(para, raw, ["S"]))
        self.assertFalse(has_dialect_alternative(para, raw, ["B"]))

    def test_inflection_list_keeps_explicit_lemma(self):
        forms = expanded_heading_forms("¶ ⟨ϣⲓϯ⟩, ⟨ϣⲁⲧ-⟩, ⟨gloss: leap⟩", "ϣⲓϯ")
        self.assertIn("ϣⲓϯ", forms)
        self.assertIn("ϣⲁⲧ", forms)
        self.assertNotIn("ϣⲁⲧϣⲓϯ", forms)

    def test_explicit_sahidic_alternative_expands_with_root(self):
        raw = "⟦Eph⟧{bible: Ephesians 1:18}"
        para = raw + " ⟪B⟫ (⟪S⟫ ⟨ϫⲓ ⲟⲩ.⟩)"
        forms = explicit_dialect_forms(para, raw, ["S"], "ⲟⲩⲟⲉⲓⲛ")
        self.assertIn("ϫⲓⲟⲩⲟⲉⲓⲛ", forms)
        self.assertEqual(explicit_dialect_forms(para, raw, ["B"], "ⲟⲩⲟⲉⲓⲛ"), [])

    def test_ditto_refers_to_previous_reading_in_same_dialect(self):
        first = "⟦Ps⟧{bible: Psalms 1:1}"
        current = "⟦Is⟧{bible: Isaiah 1:1}"
        paragraph = (
            first
            + " ⟪B⟫ (⟪S⟫ ⟨ϣⲓⲛⲉ⟩), "
            + current
            + " ⟪B⟫ (⟪S⟫ ⟦do⟧{annotation: ditto, same as last word cited in this dialect})"
        )
        forms, evidence = contextual_dialect_reading(paragraph, current, ["S"], "ⲕⲱⲧⲉ")
        self.assertEqual(forms, ["ϣⲓⲛⲉ"])
        self.assertEqual(evidence[0]["kind"], "ditto")
        self.assertEqual(
            contextual_dialect_reading(paragraph, current, ["B"], "ⲕⲱⲧⲉ")[0], []
        )

    def test_unclosed_parenthesis_preserves_explicit_reading(self):
        raw = "⟦Lev⟧{bible: Leviticus 1:1}"
        forms, _ = contextual_dialect_reading(raw + " (⟪S⟫ ⟨ϥⲱϭⲉ⟩", raw, ["S"], "ϣⲓϯ")
        self.assertEqual(forms, ["ϥⲱϭⲉ"])

    def test_ditto_without_antecedent_does_not_invent_spelling(self):
        raw = "⟦Is⟧{bible: Isaiah 1:1}"
        self.assertEqual(
            contextual_dialect_reading(
                raw + " (⟪S⟫ ⟦do⟧{annotation: ditto})", raw, ["S"], "ⲕⲱⲧⲉ"
            )[0],
            [],
        )

    def test_full_verse_label_overrides_default_chapter_metadata(self):
        self.assertEqual(
            explicit_bible_location(
                "urn:cts:copticLit:nt.1cor.sahidica", "1", "1 Corinthians 3:10"
            ),
            ("1cor", "3", "10"),
        )
        self.assertEqual(
            explicit_bible_location("urn:cts:copticLit:ot.Jonah.bohairic", "02", "03"),
            ("jonah", "2", "3"),
        )
        self.assertIsNone(
            explicit_bible_location("urn:cts:copticLit:shenoute.fox", "1", "3")
        )

    def test_adjacent_dialect_group_is_not_reduced_to_last_code(self):
        self.assertEqual(preceding_dialect_codes("MK 8 28 ⟪S⟫⟪F⟫, "), ["S", "F"])
        self.assertEqual(preceding_dialect_codes("⟪S⟫ an older example ⟪B⟫, "), ["B"])

    def test_governed_preposition_keeps_main_lexeme(self):
        raw = "⟦Ps⟧{bible: Psalms 76:19}"
        paragraph = (
            "¶ ⟦c⟧{annotation: constructed with (of verbs)} ⟨ⲛ-⟩: "
            + raw
            + " ⟪B⟫ (⟪S⟫ ⟨ⲉ-⟩)"
        )
        self.assertTrue(has_government_alternative(paragraph, raw, ["S"]))
        self.assertFalse(has_government_alternative(raw + " (⟪S⟫ ⟨ϩⲛⲁ⸗⟩)", raw, ["S"]))

    def test_basket_has_own_heading(self):
        forms = expanded_heading_forms("¶ ⟨ⲕⲟⲧ⟩, ⟨gloss: basket⟩:", "ⲕⲱⲧⲉ")
        self.assertEqual(forms, ["ⲕⲟⲧ"])

    def test_page_limits_binding(self):
        lex = {"C1": object(), "C2": object()}
        self.assertEqual(
            bind_forms(
                ["ⲁ"], "1a", lex, {"ⲁ": ["C1", "C2"]}, {"C1": {"1a"}, "C2": {"2b"}}
            ),
            ["C1"],
        )

    def test_same_page_homographs_stay_explicit(self):
        lex = {"C1": object(), "C2": object()}
        self.assertEqual(
            bind_forms(
                ["ⲁ"], "1a", lex, {"ⲁ": ["C1", "C2"]}, {"C1": {"1a"}, "C2": {"1a"}}
            ),
            ["C1", "C2"],
        )

    def test_native_token_split_can_cover_complete_form(self):
        c = {"tokens": [{"id": "u1", "form": "ϫⲓ"}, {"id": "u2", "form": "ⲟⲩⲟⲉⲓⲛ"}]}
        e = Entry("C4962", "ϫⲓⲟⲩⲟⲉⲓⲛ", "VERB", "get light", ("ϫⲓⲟⲩⲟⲉⲓⲛ",))
        self.assertEqual(token_spans(c, e), [["u1", "u2"]])

    def test_inflected_surface_uses_complete_native_lemma(self):
        c = {"tokens": [{"id": "u1", "form": "ⲕⲟⲧ", "lemma": "ⲕⲱⲧⲉ"}]}
        e = Entry("C1416", "ⲕⲱⲧⲉ", "VERB", "turn", ("ⲕⲱⲧⲉ",))
        self.assertEqual(token_spans(c, e), [["u1"]])

    def test_source_attested_bound_stem_uses_complete_token(self):
        c = {"tokens": [{"id": "u1", "form": "ϣⲁⲧ", "lemma": "ϣⲁⲧ"}]}
        e = Entry("C1", "ϣⲓϯ", "VERB", "vomit", ("ϣⲓϯ",))
        self.assertEqual(token_spans(c, e), [])
        self.assertEqual(token_spans(c, e, ["ϣⲁⲧ"]), [["u1"]])

    def test_online_only_tla_is_retained_with_dialect_forms(self):
        lex = {}
        records = [
            dict(
                id="C13001",
                lemma="ϩⲁ-",
                pos="V",
                senses=[dict(translations=dict(en=["begin"]))],
                forms=[dict(orth="ϩⲁ-", dialect="B")],
            )
        ]
        self.assertEqual(add_cdo_only_entries(lex, records), ["C13001"])
        self.assertEqual(lex["C13001"].gloss, "begin")
        self.assertIn("B", lex["C13001"].form_dialects["ϩⲁ"])
        self.assertEqual(add_cdo_only_entries(lex, records), [])

    def test_punctuation_cannot_enter_or_bridge_target_span(self):
        e = Entry("C1", "ϫⲓⲟⲩⲟⲉⲓⲛ", "VERB", "", ("ϫⲓⲟⲩⲟⲉⲓⲛ",))
        c = {
            "tokens": [
                {"id": "u1", "form": "ϫⲓ"},
                {"id": "u2", "form": "."},
                {"id": "u3", "form": "ⲟⲩⲟⲉⲓⲛ"},
            ]
        }
        self.assertEqual(token_spans(c, e), [])
        e.forms = ("ϫⲓ",)
        self.assertEqual(token_spans(c, e), [["u1"]])

    def test_partial_lemma_is_excluded(self):
        c = {"tokens": [{"id": "u1", "form": "ⲁⲃ", "lemma": "ⲁⲃ"}]}
        e = Entry("C1", "ⲁ", "NOUN", "", ("ⲁ",))
        self.assertEqual(token_spans(c, e), [])

    def test_substring_is_excluded(self):
        c = {"tokens": [{"id": "u1", "form": "ⲁⲃ"}]}
        e = Entry("C1", "ⲁ", "NOUN", "", ("ⲁ",))
        self.assertEqual(token_spans(c, e), [])


if __name__ == "__main__":
    unittest.main()
