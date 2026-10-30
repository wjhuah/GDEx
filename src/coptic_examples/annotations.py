import csv
from collections import defaultdict
from .text import lemma_key, location_key, norm, variants, work_urn


def load_manual(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        result = []
        for line,row in enumerate(csv.DictReader(f,delimiter='\t'),2):
            row = {k:norm(v) for k,v in row.items()}
            row['row_id'] = line
            result.append(row)
        return result


def location_matches(annotation, candidate):
    if annotation['URN'].upper() == 'X' or work_urn(annotation['URN']) != candidate['work']:
        return False
    chapter, verse = annotation['chapter'],annotation['verse']
    # Bible work URNs may encode chapter / verse; explicit table columns take priority.
    tail = annotation['URN'].split(':')[4:]
    if not chapter and tail and '-' not in tail[-1]:
        chapter = tail[-1].split('.')[0]
    if not verse and tail and '.' in tail[-1] and '-' not in tail[-1]:
        verse = tail[-1].split('.',1)[1]
    if not chapter and not verse:
        return False  # A whole document is not a precise positive sentence annotation.
    return all(not wanted or location_key(wanted)==location_key(candidate[key]) for wanted,key in [(chapter,'chapter'),(verse,'verse')])


def contains_lemma(row, lemma, string_match=False):
    forms = set(variants(lemma))
    if any(t['lemma'] in forms for t in row['tokens']):
        return True
    if string_match:
        return any(f in lemma_key(row['sent']) for f in forms)
    # MWE: require contiguous annotation lemmas.
    if ' ' in lemma_key(lemma):
        return lemma_key(lemma) in ' '.join(t['lemma'] for t in row['tokens'])
    return False


def resolve_manual(annotations, corpus):
    resolved, audit = [], []
    work_cache = {}
    for a in annotations:
        if a['URN'].upper() == 'X':
            audit.append(dict(row_id=a['row_id'],TLA=a['TLA'],status='rule_only',matches=0))
            continue
        key=work_urn(a['URN'])
        if key not in work_cache:
            work_cache[key]=corpus.citation(a['URN'])
        matches = [r for r in work_cache[key] if location_matches(a,r) and contains_lemma(r,a['lemma'],a['string_match'].lower().startswith('y'))]
        # Keep duplicate witnesses visibly; downstream splits group by text and lemma.
        for r in matches:
            resolved.append(dict(TLA=a['TLA'],annotation=a,candidate=r))
        audit.append(dict(row_id=a['row_id'],TLA=a['TLA'],URN=a['URN'],chapter=a['chapter'],verse=a['verse'],status='matched' if matches else 'unresolved',matches=len(matches)))
    return resolved,audit


def forbidden_rules(annotations):
    result = defaultdict(set)
    for row in annotations:
        result[row['TLA']].update(x.strip() for x in row['forbidden_urns'].split(';') if x.strip())
    return result


def forbidden(candidate, rules, manual_positive=False):
    for rule in rules:
        if rule.upper() in {'ALL','ONLY'}:
            if not manual_positive:
                return True
        elif work_urn(rule)==candidate['work']:
            tail = rule.split(':')[4:]
            if not tail or rule==candidate['urn'] or (tail[-1]==candidate['chapter']):
                return True
            if tail[-1]==f"{candidate['chapter']}.{candidate['verse']}":
                return True
    return False
