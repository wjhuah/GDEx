import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from coptic_examples.config import Config
from coptic_examples.lexicon import Entry
from coptic_examples.sense import targets, align_candidate, recommendation_alignment
from coptic_examples.token_review import decisions


class TokenReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lex = {t: Entry(**e) for t, e in json.loads((Path(__file__).parent / 'fixtures/review_lexicon.json').read_text()).items()}
        cls.cases = json.loads((Path(__file__).parent / 'fixtures/top20_context_reviews.json').read_text())

    def test_context_corrections_survive_relaxed_pos_matching(self):
        for case in self.cases:
            for tla in case['scope']:
                with self.subTest(source=case['candidate']['id'], tla=tla):
                    c = copy.deepcopy(case['candidate'])
                    ii = align_candidate(c, self.lex[tla], allow_pos_mismatch=True)
                    actual = {c['tokens'][i]['id'] for i in ii}
                    checked = set(case['token_ids'])
                    if tla in case['accepted']:
                        self.assertTrue(checked.issubset(actual))
                    else:
                        self.assertFalse(checked.intersection(actual))

    def test_correction_requires_matching_coptic_and_dialect(self):
        c = copy.deepcopy(self.cases[0]['candidate'])
        self.assertTrue(decisions(c, 'C771'))
        c['sent'] += ' ⲁⲩⲱ'
        self.assertFalse(decisions(c, 'C771'))
        c = copy.deepcopy(self.cases[0]['candidate'])
        c['dialect_id'] = 'bohairic'
        self.assertFalse(decisions(c, 'C771'))

    def test_multiple_occurrences_are_reviewed_individually(self):
        c = next(x['candidate'] for x in self.cases if x['candidate']['id'].endswith('27_Daniel_02.tt#31'))
        ii = targets(c, self.lex['C2578'])
        ids = {c['tokens'][i]['id'] for i in ii}
        self.assertIn('u1185', ids)
        self.assertNotIn('u1219', ids)

    def test_old_alignment_does_not_restore_rejected_day_as_cow(self):
        c = copy.deepcopy(self.cases[0]['candidate'])
        tid = self.cases[0]['token_ids'][0]
        c['target_alignment'] = {'C771': [tid]}
        c['crum_pos_alignment'] = {'C771': [tid]}
        self.assertEqual(targets(c, self.lex['C771']), [])
        self.assertTrue(targets(c, self.lex['C6965']))

    def test_pos_relaxation_requires_source_support_for_recommendation(self):
        c = dict(sent='ⲡϫⲱ', dialect_id='sahidic', tokens=[
            dict(id='u1', form='ϫⲱ', lemma='ϫⲱ', pos='VERB', xpos='V')
        ])
        e = self.lex['C7097']
        align_candidate(c, e, True)
        supported = copy.deepcopy(c)
        self.assertEqual(recommendation_alignment(c, e), [])
        self.assertEqual(recommendation_alignment(supported, e, source_selected=True), [0])
        self.assertEqual(supported['tokens'][0]['xpos'], 'V')

    def test_numbers_31_keeps_oxen_separate_from_asses_in_the_same_verse(self):
        c = next(x['candidate'] for x in self.cases if x['candidate']['id'].endswith('04_Numeri_31.tt#28'))
        ox = {c['tokens'][i]['form'] for i in targets(c, self.lex['C772'])}
        ass = {c['tokens'][i]['form'] for i in targets(c, self.lex['C876'])}
        self.assertIn('ⲉϩⲱⲟⲩ', ox)
        self.assertNotIn('ⲉϩⲉⲩ', ox)
        self.assertIn('ⲉϩⲉⲩ', ass)


if __name__ == '__main__':
    unittest.main()
