import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from coptic_examples.lexicon import Entry
from coptic_examples.token_senses import resolve


class TokenSenseBatchTests(unittest.TestCase):
    def test_one_batch_keeps_independent_decisions_for_repeated_tokens(self):
        own = Entry('a', 'ⲉⲓⲱⲧ', 'NOUN', 'father', ('ⲉⲓⲱⲧ',))
        rival = Entry('b', 'ⲉⲓⲱⲧ', 'NOUN', 'barley', ('ⲉⲓⲱⲧ',))
        def row(ids):
            return dict(sent=' '.join(['ⲉⲓⲱⲧ'] * len(ids)), dialect_id='sahidic',
                        tokens=[dict(id=i, form='ⲉⲓⲱⲧ', lemma='ⲉⲓⲱⲧ', pos='NOUN', xpos='N') for i in ids])
        rows = [row(['u1', 'u2']), row(['u3']), row(['u4'])]
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            (folder / 'token_sense.txt').touch()
            (folder / 'token_sense.json').write_text(json.dumps({'available': True}))
            model = SimpleNamespace(folder=folder, metadata={},
                                    features=SimpleNamespace(rivals=lambda e: [rival]),
                                    score=Mock(return_value=(np.array([.9, .8]), np.array([2., 1.]))))
            scorer = Mock()
            scorer.predict.return_value = np.array([.9, .1, .2, .8, .7, .3, .994309, .994264])
            with patch('coptic_examples.token_senses.lgb.Booster', return_value=scorer), patch('coptic_examples.token_senses.vector', return_value=[0.]):
                result = resolve(model, rows, own, {'a': own, 'b': rival}, 'two_models')
        scorer.predict.assert_called_once()
        self.assertEqual(len(result), 2)
        self.assertEqual([t['id'] for t in result[0]['target_token_evidence']], ['u1'])
        self.assertEqual([t['id'] for t in result[1]['target_token_evidence']], ['u3'])
        self.assertEqual(len(result[0]['token_sense_predictions']), 2)


if __name__ == '__main__':
    unittest.main()
