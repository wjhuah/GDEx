import sys, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.sense import targets
from coptic_examples.lexicon import Entry
from coptic_examples.features import Features
from coptic_examples.text import pos


class SenseTests(unittest.TestCase):
    def c(self, form, lemma, xpos, upos):
        return dict(
            sent=form,
            translation="example",
            dialect_id="sahidic",
            tokens=[
                dict(
                    id="u1",
                    form=form,
                    lemma=lemma,
                    xpos=xpos,
                    pos=upos,
                    head="",
                    dep="root",
                )
            ],
        )

    def test_no_substring(self):
        e = Entry("test", "wa", "NOUN", "word", ("wa",))
        self.assertEqual(targets(self.c("watch", "watch", "N", "NOUN"), e), [])
        f = Features({}).extract(self.c("watch", "watch", "N", "NOUN"), "wa")
        self.assertEqual(f["target_repetition"], 0)

    def test_converter_vs_preposition_same_surface(self):
        c = self.c("ⲉ", "ⲉⲣⲉ", "CCIRC", "SCONJ")
        self.assertEqual(targets(c, Entry("C616", "ⲉ", "SCONJ", "circ", ("ⲉ",))), [0])
        self.assertEqual(targets(c, Entry("C615", "ⲉ", "ADP", "to", ("ⲉ",))), [])

    def test_copula_not_possessive(self):
        c = self.c("ⲡⲉ", "ⲡⲉ", "COP", "PRON")
        self.assertEqual(
            targets(c, Entry("C2788", "ⲡⲉ", "PRON", "copula", ("ⲡⲉ",))), [0]
        )
        self.assertEqual(
            targets(c, Entry("C2786", "ⲡⲉ", "DET", "possessive", ("ⲡⲉ",))), []
        )

    def test_dictionary_pos_mapping(self):
        for raw, want in [
            ("Präp.", "ADP"),
            ("Partikel", "PART"),
            ("Possessivartikel", "DET"),
            ("Satzkonverter", "SCONJ"),
        ]:
            self.assertEqual(pos(raw), want)

    def test_multiword_requires_contiguous_tokens(self):
        e = Entry("test", "ⲣ ⲡⲉ", "VERB", "fixture", ("ⲣ ⲡⲉ",))
        c = self.c("ⲣ", "ⲣ", "V", "VERB")
        c["tokens"] += self.c("ⲡⲉ", "ⲡⲉ", "COP", "PRON")["tokens"]
        self.assertEqual(targets(c, e), [0, 1])
        c["tokens"].insert(1, self.c("ⲛ", "ⲛ", "PREP", "ADP")["tokens"][0])
        self.assertEqual(targets(c, e), [])


if __name__ == "__main__":
    unittest.main()
