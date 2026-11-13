import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.qualifier_scope import rarity_scope


class QualifierScopeTests(unittest.TestCase):
    def test_c4327_sahidic_form_only(self):
        r = rarity_scope("⟨ⲧⲱⲙⲧ⟩ ⟪B⟫, ⟨-ⲙⲛⲧ⟩ ⟪S⟫ (", "), ⟨ⲧⲟⲙⲧ†⟩ ⟪B⟫")
        self.assertEqual(r["scope"], "方言词形")
        self.assertEqual(r["dialect_codes"], ["S"])
        self.assertIn("-ⲙⲛⲧ", r["form"])

    def test_form_before_dialect(self):
        r = rarity_scope("⟨ⲙⲧⲟ⟩ (", ") ⟪S⟫, ⟨ⲉⲙⲑⲱ⟩ ⟪B⟫")
        self.assertEqual(r["dialect_codes"], ["S"])

    def test_number_usage_does_not_ban_a_dialect(self):
        r = rarity_scope(
            "⟪F⟫ & ⟦sg⟧{annotation: singular} as ⟦pl⟧{annotation: plural} (", "), ⟦nn⟧"
        )
        self.assertEqual(r["scope"], "单数形式用作复数")
        self.assertEqual(r["dialect_codes"], [])

    def test_transitivity_is_local(self):
        r = rarity_scope("¶ ⟦tr⟧{annotation: transitive} (", "), ⟨gloss: destroy⟩")
        self.assertEqual(r["scope"], "及物用法")

    def test_construction_is_not_general_rarity(self):
        r = rarity_scope("¶ ", " without ⟨ⲉⲃ.⟩:")
        self.assertEqual(r["scope"], "局部构式或搭配")

    def test_unclear_scope_stays_open(self):
        self.assertEqual(rarity_scope("Unclear ", ".")["scope"], "待核定")


if __name__ == "__main__":
    unittest.main()
