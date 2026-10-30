import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from coptic_examples.ablation import expected_top_metric,train_gain,family,crum4_module
from coptic_examples.config import Config


class AblationTests(unittest.TestCase):
    def test_ties_do_not_receive_source_order_credit(self):
        a=expected_top_metric([1,0],[0,0],1)
        b=expected_top_metric([0,1],[0,0],1)
        self.assertAlmostEqual(a,.5);self.assertEqual(a,b)
        self.assertEqual(expected_top_metric([1,0],[0,0],1,False),.5)

    def test_continuous_and_legacy_floor_are_distinct(self):
        a=dict(task='crum_style',gain=.4,rank=50)
        self.assertEqual(train_gain(a,'continuous'),.4)
        self.assertAlmostEqual(train_gain(a,'legacy321'),1/3)

    def test_spelling_family_keeps_homographs_and_vowel_variants_together(self):
        self.assertEqual(family('ⲉⲓⲱⲧ'),family('ⲉⲓⲱⲱⲧ'))

    def test_actual_crum4_formula_uses_frequency(self):
        old=crum4_module(Config())
        rows=[dict(sent='ⲁ',translation='barley',mean_frequency=99,row_index=0),
              dict(sent='ⲃ',translation='father',mean_frequency=9,row_index=1)]
        vectors={'barley':[1.,0.],'father':[0.,1.]}
        engine=object.__new__(old.RetrievalEngine)
        scored=engine.score_candidates_preembedded(rows,'barley',vectors,[1.,0.])
        self.assertAlmostEqual(scored[0]['score'],1.)
        self.assertAlmostEqual(scored[1]['score'],.075)

    def test_ui_error_escapes_markup_and_guide_is_readable(self):
        from coptic_examples.ui import search_page,guide_page
        page=search_page(Config(),error='<script>bad()</script>')
        self.assertNotIn('<script>bad()',page)
        self.assertIn('&lt;script&gt;',page)
        guide=guide_page(Config())
        self.assertIn('id="step10"',guide)
        self.assertIn('0.785 不等于',guide)


if __name__=='__main__':unittest.main()
