pass
import csv, hashlib, json, re, sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from coptic_examples.annotations import forbidden, forbidden_rules, load_manual
from coptic_examples.config import Config
from coptic_examples.corpus import Corpus
from coptic_examples.crum_audit import lexical_index, GLOSS
from coptic_examples.integrity import alignment_issue
from coptic_examples.sense import targets
from coptic_examples.text import english, lemma_key, real_translation, editorial_heading
cfg = Config()
out = cfg.reports / ''
lex, forms, pages = lexical_index(cfg.lexicon)
base = {r['id']: r for l in (out / '').open() if (r := json.loads(l))['status'] == ''}
previous = {json.loads(l)['id'] for l in (out / '').open()}
assert len(base) == 30033 and (not previous.intersection(base))
rules = forbidden_rules(load_manual(cfg.manual))
ban = {(r['TLA'], r['sentence_id']) for r in csv.DictReader((cfg.derived / 'forbidden_matches.tsv').open(), delimiter='\t')}
stop = set('the a as his its she her him who with for from into over again one make made makes have had has some other about without before after under down take give come get up out off see also often mostly be being become sb sth lit causative transitive intransitive'.split())

def terms(s):
    return {w for w in english(s) if len(w) >= 3 and w not in stop}

def match_translation(words, s):
    words_in = set(english(s))
    return sorted((w for w in words if words_in & {w, w + 's', w + 'es', w + 'ed', w + 'ing', w.rstrip('e') + 'ing'}))

@lru_cache(128)
def original(file):
    return (cfg.source / 'Crum_plain_text' / file).read_bytes()
counts = Counter()
seen = set()
passed = set()
with Corpus(cfg).connect() as db, (out / '').open('w') as detail, (out / '').open('w') as sheet:

    @lru_cache(1024)
    def sentence(sid):
        return json.loads(db.execute('select payload from sentence where id=?', (sid,)).fetchone()[0])
    w = csv.writer(sheet, delimiter='\t')
    w.writerow(['', '', '', '', '', '', '', ''])
    for line in (cfg.derived / 'complete_citations/citations.jsonl').open():
        r = json.loads(line)
        oid = r['id']
        if oid not in base:
            continue
        assert oid not in seen
        seen.add(oid)
        raw = r['raw'].encode()
        start = r['byte_start']
        assert original(r['file'])[start:start + len(raw)] == raw
        para = r['paragraph'] or ''
        before, _, after = para.partition(r['raw'])
        heading = lemma_key(r['heading_form'])
        headids = set(forms.get(heading, []))
        preceding = GLOSS.findall(before)
        gloss_terms = set().union(*(terms(g) for g in preceding[-1:]))
        pool = set(r['tla_candidates']) | set(base[oid]['legacy_tlas']) | headids
        pool = sorted((t for t in pool if t in lex and r['page'] in pages.get(t, set())))
        checks = []
        ok = []
        reasons = Counter()
        if r['reference']['kind'] != 'verse':
            why = ''
        elif not r['local_links']:
            why = {'dialect_not_local': '', 'no_local_verse': ''}.get(r['link_status'], '')
        elif not pool:
            why = ''
        else:
            for link in r['local_links']:
                c = sentence(link['sentence_id'])
                matches = []
                for t in pool:
                    ii = targets(c, lex[t])
                    if ii:
                        matches.append((t, ii))
                if not matches:
                    reasons[''] = 1
                for t, ii in matches:
                    entry = lex[t]
                    tokens = [c['tokens'][i] for i in ii]
                    peers = sorted({p for tok in tokens for key in (tok['lemma'], lemma_key(tok['form'])) for p in forms.get(key, []) if p != t and lex[p].pos == entry.pos and (r['page'] in pages.get(p, set())) and set(targets(c, lex[p])) & set(ii)})
                    overlap = match_translation(gloss_terms & terms(entry.gloss), c['translation'])
                    issues = []
                    if forbidden(c, rules[t]) or (t, c['id']) in ban:
                        issues.append('')
                    if alignment_issue(c) or not real_translation(c['translation']) or editorial_heading(c['translation']):
                        issues.append('')
                    if t not in headids:
                        issues.append('')
                    if not overlap:
                        issues.append('')
                    if peers or len(matches) > 1:
                        issues.append('')
                    if len(ii) != 1:
                        issues.append('')
                    if re.search('⟦diff⟧|⟦om⟧', after.split('{bible:')[0].split('⟦;⟧')[0]):
                        issues.append('')
                    reasons.update(issues)
                    item = dict(tla=t, sid=c['id'], dialect=c['dialect_id'], tokens=[v['id'] for v in tokens], text=c['sent'], translation=c['translation'], terms=overlap, peers=peers, issues=issues)
                    checks.append(item)
                    if not issues:
                        ok.append(item)
            if len({p['tla'] for p in ok}) > 1:
                ok = []
                reasons[''] += 1
            why = '' if ok else '；'.join(reasons)
        decision = '' if ok else ''
        counts[decision] += 1
        for p in ok:
            passed.add((p['tla'], p['sid']))
        row = dict(id=oid, file=r['file'], byte_start=start, page=r['page'], reference=r['normalized_reference'], heading=heading, paragraph=para, raw=r['raw'], original_status=r['training_status'], dialects=r['dialect_codes'], local_links=r['local_links'], candidates=pool, checks=checks, decision=decision, reason=why, accepted=ok)
        detail.write(json.dumps(row, ensure_ascii=False) + '\n')
        w.writerow([oid, row['reference'], r['file'], r['page'], heading, decision, why, ' | '.join((p['tla'] + ' ' + p['sid'] + ' ' + ','.join(p['tokens']) for p in ok))])
assert seen == set(base)
for ext in ['tsv', 'jsonl']:
    (out / f'{ext}.tmp').replace(out / f'{ext}')
print(counts, len(passed))
