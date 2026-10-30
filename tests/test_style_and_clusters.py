import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from coptic_examples.dialects import Dialects
from coptic_examples.crum_audit import parse_reference,page_set,extract_file
from coptic_examples.lexicon import Entry
from coptic_examples.selection import profile,choose,metrics
from coptic_examples.shortlist import pair_data


class DialectAndExtractionTests(unittest.TestCase):
    def test_unknown_is_never_sahidic_and_metadata_beats_folder(self):
        d=Dialects()
        self.assertEqual(d.identify('Bohairic Coptic','sahidic.ot/a.tt'),('bohairic','metadata_path_conflict'))
        self.assertEqual(d.identify('Lycopolitan Coptic','sahidic.ot/a.tt')[0],'unknown')
        self.assertEqual(d.identify('','bohairic.mark/a.tt')[0],'bohairic')

    def test_register_new_dialect_without_algorithm_change(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'dialects.toml';p.write_text('[dialects.testdialect]\nname="Test Dialect"\naliases=["New Coptic"]\ncodes=["Z"]\npath_globs=["new/*"]\n')
            d=Dialects(p);self.assertEqual(d.require('New Coptic'),'testdialect')
            self.assertEqual(d.from_codes(['Z']),['testdialect'])
            self.assertEqual(d.identify('New Coptic','anything')[0],'testdialect')

    def test_reference_types_not_discarded_or_guessed(self):
        for text,kind in [('Genesis 1:2','verse'),('Daniel A:48','lettered_section'),('Psalms 92 titulus','titulus'),('Esther D','lettered_section_only'),('Epistle to the Romans 16 subscriptio','subscriptio'),('Genesis','book_only'),('first_samuel second_samuel first_kings second_kings','ambiguous_book_group')]:self.assertEqual(parse_reference(text)['kind'],kind)

    def test_printed_page_ranges_and_ab(self):
        self.assertEqual(page_set('87ab'),{'87a','87b'})
        self.assertEqual(page_set('86b-87a'),{'86b','87a'})

    def test_all_dialects_no_page_and_source_offsets_survive(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'sample.txt';text='⟦Gen⟧{bible: Genesis}\n¶ ⟨ⲉⲓⲱⲧ⟩ ⟨gloss: barley⟩: ⟦Ge 1 2⟧{bible: Genesis 1:2} ⟪B⟫, ⟦Su⟧{bible: Daniel A:48}'
            p.write_text(text)
            rows=extract_file(p,{}, {}, {})
            self.assertEqual(len(rows),3)
            self.assertTrue(all(text[r['char_start']:r['char_end']]==r['raw'] for r in rows))
            self.assertIn('B',rows[1]['dialect_codes'])
            self.assertEqual(rows[0]['tla_binding'],'outside_paragraph')

    def test_unresolved_new_head_does_not_inherit_previous_tla(self):
        entry=Entry('C1','ⲉⲓⲱⲧ','NOUN','barley',('ⲉⲓⲱⲧ',))
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'sample.txt';p.write_text('87a\n¶ ⟨ⲉⲓⲱⲧ⟩ ⟨gloss: barley⟩ ⟦Ge⟧{bible: Genesis 1:2}\n¶ ⟨ⲥⲁⲃⲟⲩ⟩ ⟦Ge⟧{bible: Genesis 1:3}')
            rows=extract_file(p,{'C1':entry},{'ⲉⲓⲱⲧ':['C1']},{'C1':{'87a'}})
        self.assertEqual(rows[0]['tla_candidates'],['C1']);self.assertEqual(rows[1]['tla_candidates'],[])


def token(id,lemma,pos,dep,head='',xpos=''):
    return dict(id=id,lemma=lemma,form=lemma,pos=pos,dep=dep,head=head,xpos=xpos)


class SelectionTests(unittest.TestCase):
    def test_information_prefers_descriptive_context_not_just_target_presence(self):
        # English toy dependency fixtures, not invented Coptic translations.
        plain=dict(sent='I like apples.',tokens=[token('1','I','PRON','nsubj','2'),token('2','like','VERB','root'),token('3','apple','NOUN','obj','2')])
        rich=dict(sent='This apple is large, red and delicious.',tokens=[token('1','this','DET','det','2'),token('2','apple','NOUN','nsubj','3'),token('3','large','ADJ','root'),token('4','red','ADJ','conj','3'),token('5','delicious','ADJ','conj','3')])
        a=profile(plain,'apple')['features']['information_score'];b=profile(rich,'apple')['features']['information_score']
        self.assertGreater(b,a)

    def rows(self):
        out=[]
        for i,(role,q) in enumerate([('obj',.9),('obj',.89),('obj',.88),('nsubj',.87),('obl',.86)]):
            out.append(dict(id=str(i),sent=str(i),dialect_id='sahidic',tla='C1',shortlist_score=q,tokens=[token('1',str(i),'VERB','root')],syntax=dict(signature=['role:'+role])))
        return out

    def test_three_example_coverage_and_quality_both_reported(self):
        rows=self.rows();a=choose(rows,3,'rank');b=choose(rows,3,'coverage')
        self.assertGreater(metrics(b,rows)['weighted_syntax_coverage'],metrics(a,rows)['weighted_syntax_coverage'])
        self.assertGreaterEqual(min(r['shortlist_score'] for r in b),.86)

    def test_quality_metric_does_not_confuse_style_score_with_acceptance(self):
        r=self.rows()[0];r['acceptance_score']=.2;r['shortlist_score']=.9
        m=metrics([r],[r])
        self.assertEqual(m['mean_quality'],.2);self.assertEqual(m['mean_combined_score'],.9)

    def test_no_cross_dialect_or_cross_entry_set(self):
        rows=self.rows();rows[0]['dialect_id']='bohairic'
        with self.assertRaises(ValueError):choose(rows,3)
        rows=self.rows();rows[0]['tla']='C2'
        with self.assertRaises(ValueError):choose(rows,3)

    def test_crum_pair_order_not_clipped_after_twentieth_example(self):
        rows=[dict(group='Crum article',rank=rank,text_key=str(rank),x=[rank,1]) for rank in [99,100,101]]
        x,y=pair_data(rows)
        self.assertEqual(len(y),6);self.assertEqual(set(y),{0,1})
        self.assertTrue(all(delta[0]<0 for delta,label in zip(x,y) if label==1))


if __name__=='__main__':unittest.main()
