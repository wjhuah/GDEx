import sys, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.homographs import homograph_groups


def record(tla, word, meaning, code="S", pos="N", forms=None, subc=""):
    return dict(
        id=tla,
        lemma=word,
        pos=pos,
        subc=subc,
        forms=forms if forms is not None else [dict(orth=word, dialect=code)],
        senses=[dict(translations=dict(en=[meaning]))] if meaning else [],
    )


class HomographTests(unittest.TestCase):
    def test_father_barley_and_title_have_distinct_entries(self):
        rows = [
            record("C992", "ⲉⲓⲱⲧ", "father"),
            record("C993", "ⲉⲓⲱⲧ", "father", subc="Titel"),
            record("C998", "ⲉⲓⲱⲧ", "barley"),
        ]
        group = homograph_groups(rows)[0]
        self.assertEqual(group["shared_dialects"]["sahidic"], ["C992", "C993", "C998"])
        self.assertIn("(title)", group["senses"][1]["gloss"])
        self.assertFalse(group["automatic_negative_labels"])

    def test_cross_dialect_only_is_separate(self):
        group = homograph_groups(
            [record("C1", "ⲁ", "one", "S"), record("C2", "ⲁ", "prefix", "B")]
        )[0]
        self.assertEqual(group["category"], "cross_dialect_or_unresolved")
        self.assertEqual(group["shared_dialects"], {})

    def test_duplicate_forms_do_not_create_a_new_sense(self):
        self.assertEqual(
            homograph_groups(
                [record("C1", "ⲁ", "one", forms=[dict(orth="ⲁ", dialect="S")] * 2)]
            ),
            [],
        )

    def test_complete_forms_and_word_boundaries(self):
        self.assertEqual(
            homograph_groups(
                [
                    record("C1", "ⲁ", "a"),
                    record("C2", "ⲁⲃ", "ab"),
                    record("C3", "ⲁ ⲃ", "a b"),
                ]
            ),
            [],
        )

    def test_variant_collision_has_its_own_inventory(self):
        rows = [
            record("C1", "ⲁ", "one"),
            record(
                "C2",
                "ⲃ",
                "two",
                forms=[dict(orth="ⲃ", dialect="S"), dict(orth="ⲁ", dialect="S")],
            ),
        ]
        self.assertEqual(homograph_groups(rows, True), [])
        self.assertEqual(homograph_groups(rows, False)[0]["word"], "ⲁ")

    def test_case_insensitive_grouping(self):
        group = homograph_groups(
            [record("C1", "Ϩⲟⲟⲩ", "day"), record("C2", "ϩⲟⲟⲩ", "bad", pos="V")]
        )[0]
        self.assertEqual(group["category"], "grammatical_and_possible_semantic")

    def test_same_gloss_keeps_review_status(self):
        group = homograph_groups([record("C1", "ⲁ", "one"), record("C2", "ⲁ", "one")])[
            0
        ]
        self.assertEqual(group["category"], "same_gloss_review_required")
        self.assertFalse(group["expert_verified"])

    def test_unshared_pos_does_not_change_shared_dialect_category(self):
        rows = [
            record("C1", "ⲁ", "first", "S"),
            record("C2", "ⲁ", "second", "S"),
            record("C3", "ⲁ", "verb", "B", pos="V"),
        ]
        self.assertEqual(
            homograph_groups(rows)[0]["category"], "same_pos_different_gloss"
        )

    def test_grammatical_subcategories_fill_missing_gloss(self):
        rows = [
            record("C616", "ⲉ-", "", subc="Umstandsatzkonverter", pos="C"),
            record("C617", "ⲉ-", "", subc="Relativkonverter", pos="C"),
        ]
        group = homograph_groups(rows)[0]
        self.assertEqual(group["category"], "same_pos_different_gloss")
        self.assertEqual(group["senses"][0]["gloss"], "circumstantial converter")


if __name__ == "__main__":
    unittest.main()
