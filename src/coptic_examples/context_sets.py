pass
from collections import Counter
from itertools import combinations
import unicodedata
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import euclidean_distances

def context_bags(candidate, indices, radius=3):
    tokens = candidate['tokens']
    words = [i for i, t in enumerate(tokens) if t['pos'] != 'PUNCT' and any((ch.isalpha() for ch in t['form']))]
    positions = {i: j for j, i in enumerate(words)}
    lexical, syntax, dependency = (Counter(), Counter(), Counter())
    windows = []
    byid = {t['id']: i for i, t in enumerate(tokens)}
    weight = 1 / max(1, len(indices))
    for index in indices:
        if index not in positions:
            continue
        window = {'target_id': tokens[index]['id'], 'word': [], 'syntax': []}
        for offset in range(-radius, radius + 1):
            j = positions[index] + offset
            if offset == 0 or not 0 <= j < len(words):
                continue
            i = words[j]
            token = tokens[i]
            tag = f'_#{offset:+d}'
            form = unicodedata.normalize('NFKC', token['form']).casefold()
            head = byid.get(token['head'])
            direction = 'root' if head is None else 'left' if head < i else 'right'
            lexical[form + tag] += weight
            syntag = token['pos'] + ':' + (token['dep'] or 'unknown') + ':' + direction + tag
            syntax[syntag] += weight
            window['word'].append(form + tag)
            window['syntax'].append(syntag)
        target = tokens[index]
        dependency['in:' + (target['dep'] or 'unknown')] += weight
        for child in tokens:
            if child['head'] == target['id']:
                dependency['out:' + (child['dep'] or 'unknown')] += weight
        windows.append(window)
    return (lexical, syntax, dependency, windows)

def distances(bags):
    if not any(bags):
        return np.zeros((len(bags), len(bags)))
    matrix = DictVectorizer().fit_transform(bags)
    matrix = TfidfTransformer().fit_transform(matrix)
    return euclidean_distances(matrix) / np.sqrt(2)

def objective(indices, quality, distance, diversity=0.35):
    relevance = float(np.mean([quality[i] for i in indices]))
    pairs = list(combinations(indices, 2))
    difference = float(np.mean([distance[i, j] for i, j in pairs])) if pairs else 0.0
    return relevance + diversity * difference

def admissible_sets(quality, size, regret=0.15):
    floor = float(np.mean(sorted(quality, reverse=True)[:size])) - regret
    return [s for s in combinations(range(len(quality)), size) if np.mean([quality[i] for i in s]) >= floor - 1e-12]

def joint_bag_selection(quality, lexical, syntax):
    triples = admissible_sets(quality, 3)
    a = {s: objective(s, quality, lexical) for s in triples}
    b = {s: objective(s, quality, syntax) for s in triples}
    first, second = max(((x, y) for x in triples for y in triples if len(set(x) | set(y)) <= 5), key=lambda xy: a[xy[0]] + b[xy[1]])
    union = set(first) | set(second)
    five = sorted(union)
    for i in np.argsort(-np.asarray(quality), kind='stable'):
        if int(i) not in union:
            five.append(int(i))
        if len(five) == 5:
            break
    return dict(five=five, lexical=list(first), syntax=list(second), lexical_objective=a[first], syntax_objective=b[second], unconstrained_lexical=list(max(triples, key=a.get)), unconstrained_syntax=list(max(triples, key=b.get)))

def sum_selection(quality, distance):
    triples = admissible_sets(quality, 3)
    three = max(triples, key=lambda s: objective(s, quality, distance))
    fives = [s for s in combinations(range(len(quality)), 5) if set(three) <= set(s)]
    five = max(fives, key=lambda s: objective(s, quality, distance))
    return dict(five=list(five), three=list(three), objective=objective(three, quality, distance))
