import json
import sys
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.config import Config
from coptic_examples.corpus import Corpus, parse_tt
from coptic_examples.annotations import location_matches, contains_lemma, forbidden
from coptic_examples.text import (
    pos,
    real_translation,
    sm_freq,
    work_urn,
    editorial_heading,
)
from coptic_examples.features import Features, GDEX, UD
from coptic_examples.training import ndcg, make_negatives, prepare_rows


def tt(root):
    p = root / "test.tt"
    p.write_text(
        """<meta document_cts_urn="urn:cts:copticLit:ot.ruth.coptot:1" chapter="1" title="Ruth 1" translation="Translator credit">
<verse_n verse_n="22">
<translation translation="The barley harvest began.">
<norm_group norm_group="ⲡⲉⲓⲱⲧ">
<norm xml:id="u1" lemma="ⲡ" norm="ⲡ" pos="ART" func="det" head="#u2">
</norm>
<norm xml:id="u2" lemma="ⲉⲓⲱⲧ" norm="ⲉⲓⲱⲧ" pos="N" func="nsubj" head="#u3">
</norm>
</norm_group>
<norm_group norm_group="ⲁϥϣⲱⲡⲉ">
<norm xml:id="u3" lemma="ϣⲱⲡⲉ" norm="ϣⲱⲡⲉ" pos="V" func="root">
</norm>
</norm_group>
</translation>
</verse_n>
</meta>""",
        encoding="utf-8",
    )
    return p


class PipelineTests(unittest.TestCase):
    def test_native_auxiliary_and_portmanteau_mapping(self):
        for tag, expected in [
            ("APST", "AUX"),
            ("ART", "DET"),
            ("PPOS", "DET"),
            ("VSTAT", "VERB"),
            ("NPROP", "PROPN"),
            ("APREC", "SCONJ"),
            ("APST_PPERS", "PRON"),
        ]:
            self.assertEqual(pos(tag), expected)

    def test_placeholder_and_english_detection(self):
        for value in ["", "...", "ؤ", "…"]:
            self.assertFalse(real_translation(value))
        self.assertTrue(real_translation("Barley."))

    def test_uppercase_document_heading_not_recommended_as_sentence(self):
        self.assertTrue(
            editorial_heading(
                "THE FOURTH MARTYRDOM OF APA VICTOR. IN THE PEACE OF GOD. BLESS US. AMEN."
            )
        )
        self.assertFalse(editorial_heading("Bless me, my father!"))
        self.assertFalse(editorial_heading("The life of a man is short."))

    def test_sm_freq_is_not_rouge_or_empty_gloss(self):
        self.assertEqual(sm_freq("father", "The father spoke."), 1)
        self.assertEqual(sm_freq("barley", "The father spoke."), 0)
        self.assertEqual(sm_freq("", "anything"), 0)

    def test_parser_preserves_reference_and_arcs_not_translation_credit(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            rows = list(parse_tt(tt(root), root))
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual((r["chapter"], r["verse"]), ("1", "22"))
        self.assertEqual(r["translation"], "The barley harvest began.")
        self.assertEqual(r["tokens"][1]["head"], "u3")
        self.assertEqual(r["tokens"][1]["xpos"], "N")

    def test_manual_alias_and_wrong_location_rejected(self):
        r = dict(work="urn:cts:copticLit:ot.ruth.coptot", chapter="1", verse="22")
        a = dict(URN="urn:cts:copticLit:ot.ruth.coptot_ed:1", chapter="1", verse="22")
        self.assertTrue(location_matches(a, r))
        self.assertFalse(location_matches(dict(a, verse="23"), r))
        self.assertFalse(location_matches(dict(a, URN="X"), r))

    def test_forbidden_is_tla_scoped_and_manual_only_exception(self):
        r = dict(
            work="urn:cts:copticLit:ot.ruth.coptot",
            urn="urn:cts:copticLit:ot.ruth.coptot:1",
            chapter="1",
            verse="22",
        )
        self.assertTrue(forbidden(r, {"ALL"}))
        self.assertFalse(forbidden(r, {"ALL"}, True))
        self.assertFalse(forbidden(r, set()))
        self.assertTrue(forbidden(r, {"urn:cts:copticLit:ot.ruth.coptot_ed:1"}))

    def test_index_all_candidates_without_top50_cap(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            corpora = root / "corpora"
            corpora.mkdir()
            tt(corpora)
            cfg = Config(source=root, work=root / "work")
            index = Corpus(cfg)
            self.assertEqual(index.build()["sentences"], 1)
            self.assertEqual(len(index.candidates("ⲉⲓⲱⲧ", "NOUN")), 1)
            self.assertEqual(len(index.candidates("ⲉⲓⲱⲧ", "VERB")), 0)
            self.assertEqual(index.build()["sentences"], 1)

    def test_dependency_cycle_does_not_hang(self):
        c = dict(
            sent="ⲉⲓⲱⲧ ⲉⲓⲱⲧ",
            translation="His father.",
            tokens=[
                dict(id="a", head="b", dep="nsubj", pos="NOUN", lemma="ⲉⲓⲱⲧ"),
                dict(id="b", head="a", dep="acl", pos="VERB", lemma="ⲉⲓⲱⲧ"),
            ],
        )
        f = Features({"ⲉⲓⲱⲧ": 10}).extract(c, "ⲉⲓⲱⲧ")
        self.assertEqual(set(GDEX + UD), set(f))
        self.assertLess(f["tree_depth"], 10)
        self.assertEqual(f["target_in_main_clause"], 0)

    def test_linear_ndcg_and_signed_negative_policy(self):
        self.assertAlmostEqual(ndcg([30, 0], [1, 0]), 1)
        self.assertLess(ndcg([30, 0], [0, 1]), 1)
        self.assertIsNone(ndcg([0, 0], [0, 1]))

    def test_unranked_expert_selection_is_binary_positive_not_invented_order(self):
        r = dict(TLA="C998", evidence="manual", grade=None, group="C998|manual")
        rows = prepare_rows([r])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["group"], "C998|manual_acceptance")
        self.assertEqual(rows[0]["grade"], 10)
        self.assertIsNone(r["grade"])

    def test_ranked_expert_selection_keeps_order_separate_from_acceptance(self):
        r = dict(TLA="C998", evidence="manual", grade=20, group="C998|manual")
        rows = prepare_rows([r])
        self.assertEqual([x["grade"] for x in rows], [20, 10])
        self.assertEqual(
            [x["group"] for x in rows], ["C998|manual", "C998|manual_acceptance"]
        )

    def test_negative_labels_use_target_gloss_and_never_override_positive(self):
        class Encoder:
            def encode(self, texts, role="sentence"):
                import numpy as np

                return np.asarray(
                    [[1.0, 0.0] if "barley" in x else [0.0, 1.0] for x in texts]
                )

        rows = []
        fs = []
        for tla, g, text in [
            ("C998", "barley", "barley grew"),
            ("C992", "father", "father spoke"),
            ("C993", "father as title", "father prayed"),
        ]:
            rows.append(
                dict(
                    TLA=tla,
                    lemma="ⲉⲓⲱⲧ",
                    gloss=g,
                    group=tla + "|manual",
                    text_key=text,
                    grade=30,
                    candidate=dict(sent="ⲉⲓⲱⲧ", translation=text),
                )
            )
            fs.append(
                dict(
                    TLA=tla,
                    lemma="ⲉⲓⲱⲧ",
                    group=tla + "|manual",
                    text_key=text,
                    grade=30,
                    weight=1,
                    sm_freq=1,
                    sm_cosine=1,
                )
            )
        nr, nf = make_negatives(rows, fs, "conservative", Encoder())
        self.assertEqual(len(nr), 2)
        self.assertTrue(
            all({r["TLA"], r["source_tla"]} == {"C998", "C992"} for r in nr)
        )
        self.assertTrue(
            all(
                f["grade"] == 0 and f["sm_freq"] == 0 and f["sm_cosine"] == 0
                for f in nf
            )
        )
        # Identical target sentence already positively labeled: do not manufacture a contradiction.
        identical = [dict(r, text_key="same") for r in rows]
        self.assertEqual(make_negatives(identical, fs, "naive", Encoder()), ([], []))


if __name__ == "__main__":
    unittest.main()
