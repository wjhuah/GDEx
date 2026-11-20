import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from coptic_examples.context_sets import context_bags, distances, joint_bag_selection, sum_selection, objective

def token(i, form, pos='NOUN', dep='obl', head='t4'):
    return dict(id='t' + str(i), form=form, pos=pos, dep=dep, head=head)

def test_offsets_follow_tt_tokens_inside_bound_groups():
    c = dict(sent='ⲡⲁϣⲏⲣⲉ ⲙⲡϩⲏⲧ', tokens=[token(1, 'ⲡ', 'DET'), token(2, 'ⲁ', 'PRON'), token(3, 'ϣⲏⲣⲉ'), token(4, 'ⲙ', 'ADP'), token(5, 'ⲡ', 'DET'), token(6, 'ϩⲏⲧ')])
    words, syntax, deps, windows = context_bags(c, [2])
    assert words['ⲁ_#-1'] == 1
    assert words['ⲙ_#+1'] == 1
    assert words['ϩⲏⲧ_#+3'] == 1
    assert len(words) == 5
    assert all(('ϣⲏⲣⲉ' not in x for x in words))
    assert len(syntax) == 5

def test_multiple_occurrences_are_averaged_and_punctuation_skipped():
    c = dict(tokens=[token(1, 'ⲉ', 'ADP'), token(2, 'ϣⲁ'), token(3, '...', 'X'), token(4, 'ϣⲁ'), token(5, 'ⲉ', 'ADP')])
    words, _, _, windows = context_bags(c, [1, 3])
    assert len(windows) == 2
    assert all(('.' not in x for x in words))
    assert words['ⲉ_#-1'] == 0.5

def test_diversity_is_distance_and_preserves_quality_floor():
    d = distances([{'a': 1}, {'a': 1}, {'b': 1}, {'c': 1}, {'d': 1}, {'e': 1}])
    assert d[0, 1] < 1e-08 and abs(d[0, 2] - 1) < 1e-08
    q = [0.95, 0.94, 0.93, 0.9, 0.89, 0.01]
    selected = sum_selection(q, d)
    assert np.mean([q[i] for i in selected['three']]) >= np.mean(q[:3]) - 0.15 - 1e-12
    assert set(selected['three']) <= set(selected['five'])
    bag = joint_bag_selection(q, d, d)
    assert len(bag['five']) == 5
    assert set(bag['lexical']) | set(bag['syntax']) <= set(bag['five'])

def test_normalized_fsum_matches_paper_ordered_pairs():
    q = [0.8, 0.7, 0.6]
    d = np.array([[0, 0.2, 0.4], [0.2, 0, 0.6], [0.4, 0.6, 0]])
    k = 3
    lam = 0.35
    raw = (k - 1) * sum(q) + lam * sum((d[i, j] for i in range(k) for j in range(k) if i != j))
    assert abs(raw / (k * (k - 1)) - objective((0, 1, 2), q, d)) < 1e-12

def load_tests(loader, tests, pattern):
    return unittest.TestSuite((unittest.FunctionTestCase(value) for name, value in globals().items() if name.startswith('test_') and callable(value)))
