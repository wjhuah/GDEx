pass
import sys, json, math, random, sqlite3
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import joblib
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from coptic_examples.compact_features import compact_vector
from coptic_examples.config import Config
from coptic_examples.lexicon import load
from coptic_examples.source_pool import SupplementedCorpus
from coptic_examples.sense import native_candidates, recommendation_alignment, targets
from coptic_examples.inclusion import EntryFeatures
from coptic_examples.text import real_translation, editorial_heading, sm_freq
from coptic_examples.dataset import text_key
from coptic_examples.integrity import alignment_issue
from coptic_examples.recommendation_policy import parse_bias
from coptic_examples.context_sets import context_bags, distances, joint_bag_selection, sum_selection
cfg = Config()
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
lex = load(cfg.lexicon)
corpus = SupplementedCorpus(cfg)
features = EntryFeatures(cfg, lex, 'sahidic')
freq = features.extractor.freq if hasattr(features.extractor, 'freq') else json.loads((cfg.work / 'models/sahidic/frequencies.json').read_text())
classifier = joblib.load(cfg.work / 'models/tuned15_20261120/RF/classification.joblib')
ranker = joblib.load(cfg.work / 'models/tuned15_20261120/LightGBM/ranking.joblib')
bias = parse_bias((cfg.work / 'dialect_bias.md').read_text())
positive, negative, grades = (set(), set(), defaultdict(list))
counts = Counter()
for line in (cfg.derived / 'model_rows_sahidic.jsonl').open():
    r = json.loads(line)
    key = (r['tla'], r['text_key'])
    (positive if r['y'] > 0 else negative).add(key)
    if r['y'] > 0:
        counts[r['tla']] += 1
        grades[key] = r.get('ranks', [])
cache_path = out / 'candidate_pools.json'
pools = json.loads(cache_path.read_text()) if cache_path.exists() else {}
attempts = []

def pool(tla):
    if tla in pools:
        cached = pools[tla]
        if cached.get('score_model') == 'tuned15' and cached.get('hard_limit') == 20:
            return cached
        e = lex[tla]
        candidates = [c for c in cached['candidates'] if c['bound_groups'] <= 20 and recommendation_alignment(c, e, (tla, text_key(c)) in positive)]
        matrix = np.asarray([compact_vector(c, e, features.vector(c, e), freq) for c in candidates], dtype=float)
        p = classifier.predict_proba(matrix)[:, 1] if len(matrix) else []
        s = ranker.predict(matrix) if len(matrix) else []
        updated = []
        for c, inclusion, score in zip(candidates, p, s):
            if inclusion < 0.5:
                continue
            adjusted = float(score) - 0.01 * max(0, c['bound_groups'] - 20)
            c.update(inclusion=float(inclusion), rank=float(score), adjusted_rank=adjusted, quality=float(inclusion) * float(np.clip((adjusted - 1) / 2, 0, 1)))
            updated.append(c)
        features.memo.clear()
        if len(updated) >= 5:
            cached['candidates'] = sorted(updated, key=lambda c: (-c['quality'], c['id']))
            cached['score_model'] = 'tuned15'
            cached['hard_limit'] = 20
            cached['pool_origin'] = 'compact16 shortlist, frozen before compact15 rescoring'
            cache_path.write_text(json.dumps(pools, ensure_ascii=False))
            return cached
        del pools[tla]
    e = lex[tla]
    if bias.get(tla, {}).get('sahidic'):
        return None
    raw = native_candidates(cfg, corpus, e, 'sahidic', True)
    eligible, seen = ([], set())
    excluded = Counter()
    for c in raw:
        key = text_key(c)
        if key in seen:
            excluded['duplicate'] += 1
            continue
        if alignment_issue(c) or not real_translation(c['translation']) or editorial_heading(c['translation']):
            excluded['translation_or_source'] += 1
            continue
        if len(c['sent'].split()) > 20:
            excluded['over_20_groups'] += 1
            continue
        if c.get('has_lacuna'):
            excluded['damaged_span'] += 1
            continue
        if not recommendation_alignment(c, e, (tla, key) in positive):
            excluded['target_alignment'] += 1
            continue
        seen.add(key)
        co = features.vectors.cosine(e.gloss, c['translation'])
        cheap = 0.6 * (co if co is not None else 0.0) + 0.4 * sm_freq(e.gloss, c['translation']) - 0.001 * max(0, len(c['sent'].split()) - 20)
        eligible.append((cheap, c))
    eligible.sort(key=lambda x: (-x[0], x[1]['id']))
    pre, works = ([], Counter())
    for cheap, c in eligible:
        work = c.get('urn', c.get('source', '')).split(':')[:4]
        work = ':'.join(work) if isinstance(work, list) else work
        if works[work] >= 30:
            continue
        pre.append(c)
        works[work] += 1
        if len(pre) >= 240:
            break
    if len(pre) < 5:
        attempts.append(dict(tla=tla, raw=len(raw), eligible=len(eligible), reason='fewer_than_five'))
        return None
    matrix = np.asarray([compact_vector(c, e, features.vector(c, e), freq) for c in pre], dtype=float)
    p = classifier.predict_proba(matrix)[:, 1]
    s = ranker.predict(matrix)
    ranked = []
    for c, inclusion, score in zip(pre, p, s):
        if inclusion < 0.5:
            excluded['model_below_05'] += 1
            continue
        length = len(c['sent'].split())
        adjusted = float(score) - 0.01 * max(0, length - 20)
        quality = float(inclusion) * float(np.clip((adjusted - 1) / 2, 0, 1))
        c.update(inclusion=float(inclusion), rank=float(score), adjusted_rank=adjusted, quality=quality, bound_groups=length, tt_target_indices=targets(c, e), source_grades=grades.get((tla, text_key(c)), []), training_label='positive' if (tla, text_key(c)) in positive else 'negative' if (tla, text_key(c)) in negative else 'unlabelled')
        ranked.append(c)
    ranked.sort(key=lambda c: (-c['quality'], c['id']))
    features.memo.clear()
    attempts.append(dict(tla=tla, raw=len(raw), eligible=len(eligible), prescreen=len(pre), admitted=len(ranked)))
    if len(ranked) < 5:
        print('Skip', tla, len(ranked), flush=True)
        return None
    result = dict(tla=tla, lemma=e.lemma, gloss=e.gloss, raw=len(raw), eligible=len(eligible), prescreen=len(pre), admitted=len(ranked), excluded=dict(excluded), candidates=ranked[:10], score_model='tuned15', hard_limit=20, pool_origin='compact15 native retrieval')
    pools[tla] = result
    cache_path.write_text(json.dumps(pools, ensure_ascii=False))
    print('Pool', tla, len(raw), len(ranked), flush=True)
    return result
selected, used_words = ([], set())

def add(tla, category, family=None):
    if tla in {r['tla'] for r in selected} or lex[tla].lemma in used_words:
        return False
    if category == 'random' and 'unknown' in lex[tla].gloss.lower():
        return False
    candidate = pool(tla)
    if not candidate:
        return False
    selected.append(dict(tla=tla, category=category, family=family or lex[tla].lemma))
    used_words.add(lex[tla].lemma)
    return True
families = json.loads((cfg.work / 'tests/fixtures/homographs_top20.json').read_text())
unavailable = []
for group in families:
    senses = sorted(group['senses'], key=lambda s: (-counts[s['tla']], s['tla']))
    if not any((add(s['tla'], 'homograph', group['word']) for s in senses if s['tla'] in lex)):
        unavailable.append(group['word'])
byhead = defaultdict(list)
for e in lex.values():
    if 'S' in {x for v in (e.form_dialects or {}).values() for x in v}:
        byhead[e.lemma].append(e.tla)
replacement = []
for word, ids in sorted(byhead.items(), key=lambda item: -sum((counts[t] for t in item[1]))):
    if sum((r['category'] == 'homograph' for r in selected)) >= 20:
        break
    if len({lex[t].gloss for t in ids}) < 2 or word in used_words or word in {g['word'] for g in families}:
        continue
    for tla in sorted(ids, key=lambda t: -counts[t]):
        if add(tla, 'homograph', word):
            replacement.append(dict(word=word, tla=tla, replaces=unavailable[len(replacement)]))
            break
with sqlite3.connect(cfg.work / 'cache/native_tokens_extended.sqlite3') as db:
    lemma_counts = dict(db.execute("SELECT lemma, COUNT(DISTINCT sentence_id) FROM token WHERE dialect='sahidic' GROUP BY lemma"))
eligible_entries = [e for e in lex.values() if e.gloss and e.pos and ('S' in {x for v in (e.form_dialects or {}).values() for x in v}) and (lemma_counts.get(e.lemma, 0) >= 5)]
for e in sorted([e for e in eligible_entries if e.pos in {'NOUN', 'VERB', 'ADJ', 'ADV'}], key=lambda e: (-lemma_counts.get(e.lemma, 0), e.tla)):
    if sum((r['category'] == 'frequent' for r in selected)) >= 15:
        break
    add(e.tla, 'frequent')
random.Random(20261120).shuffle(eligible_entries)
for e in eligible_entries:
    if sum((r['category'] == 'random' for r in selected)) >= 15:
        break
    add(e.tla, 'random')
assert Counter((r['category'] for r in selected)) == dict(homograph=20, frequent=15, random=15)
assert len({r['tla'] for r in selected}) == 50
audit = dict(seed=20261120, selection=selected, unavailable_families=unavailable, replacement=replacement, attempts=attempts, methods={})
packets = [[], []]
intro = ['# Position-tagged lexical and syntactic bags', ' # Relevance and diversity: fSUM']
for i in range(2):
    packets[i] = [intro[i].strip(), '', '50 Sahidic entries × 5 distinct candidates. Model suggestions for review. Maximum 20 whitespace-separated groups. All methods use the same frozen candidate pool (up to 10 per entry).', '', 'Quality: r = inclusion × clip((rank − 1) / 2, 0, 1). Triple quality may fall at most 0.15 below the best-quality triple. Diversity weight = 0.35.', '', 'Context: the nearest three TT word tokens on each side, excluding punctuation; function words retained. Syntax is a UD dependency proxy.', '']
for number, target in enumerate(selected, 1):
    p = pools[target['tla']]
    rows = p['candidates']
    bags = [context_bags(c, c['tt_target_indices']) for c in rows]
    ld, sd, dd = [distances([b[j] for b in bags]) for j in range(3)]
    combined = (ld + dd) / 2
    quality = [c['quality'] for c in rows]
    bag = joint_bag_selection(quality, ld, sd)
    summed = sum_selection(quality, combined)
    audit['methods'][target['tla']] = dict(bags=bag, fsum=summed, lexical_distance=ld.tolist(), syntax_window_distance=sd.tolist(), target_dependency_distance=dd.tolist(), fsum_distance=combined.tolist(), windows=[b[3] for b in bags])
    for which, choice in enumerate([bag, summed]):
        text = packets[which]
        text += [f"## {number:02d}. {p['lemma']} · [{p['tla']}](https://coptic-dictionary.org/entry/{p['tla']})", '', f"{p['gloss']} · {target['category']} · Sahidic · pool {len(rows)}", '']
        if which == 0:
            text += ['Lexical triple: ' + ', '.join(('#' + str(i + 1) for i in bag['lexical'])) + '; syntactic triple: ' + ', '.join(('#' + str(i + 1) for i in bag['syntax'])) + '. Both triples fit within the five displayed candidates.', '']
        else:
            text += ['Recommended triple: ' + ', '.join(('#' + str(i + 1) for i in summed['three'])) + '.', '']
        for index in choice['five']:
            c = rows[index]
            marks = ('lexical ✓ ' if index in bag['lexical'] else '') + ('syntax ✓' if index in bag['syntax'] else '') if which == 0 else 'recommended ✓' if index in summed['three'] else ''
            evidence = ' | '.join((t['id'] + ':' + t['form'] for j, t in enumerate(c['tokens']) if j in c['tt_target_indices']))
            text += [f'### Candidate #{index + 1} · {marks}', '', c['sent'], '', '*' + c['translation'].replace('\n', ' ') + '*', '', f"r={c['quality']:.3f} · inclusion={c['inclusion']:.3f} · rank={c['rank']:.3f} · groups={c['bound_groups']} · {c['training_label']}", '', f"Source: {c.get('source', '')} · {c.get('urn', '')} · {c.get('ref', '')} · {c['id']}", '', f'Target TT: {evidence}', '']
            for window in bags[index][3]:
                text += ['Word offsets: ' + ' | '.join(window['word']), '', 'UD offsets: ' + ' | '.join(window['syntax']), '']
names = ['01_BOW_UD_50x5.md', '02_fSUM_50x5.md']
for name, lines in zip(names, packets):
    (out / name).write_text('\n'.join(lines) + '\n')
(out / 'selection_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
print('Exported', len(selected), 'entries;', replacement, flush=True)
features.close()
