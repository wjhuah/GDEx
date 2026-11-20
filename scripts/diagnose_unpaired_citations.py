pass
import collections
import csv
import json
import re
import sqlite3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from coptic_examples.config import Config
from coptic_examples.crum_audit import lexical_index
from coptic_examples.citation_recovery import add_cdo_only_entries, explicit_bible_location
from coptic_examples.dialects import Dialects
cfg = Config()
folder = cfg.reports / 'local_2218a127/local_aa724350'
folder.mkdir(exist_ok=True)
lex, byform, pages = lexical_index(cfg.lexicon)
added = add_cdo_only_entries(lex, json.loads((cfg.work / 'cache/cdo_index.json').read_text()))
for t in added:
    for f in lex[t].forms:
        byform[f].append(t)
registry = Dialects()
pending = {}
for line in (cfg.reports / 'local_2218a127/local_a9162ef4/local_b415e7f6').open():
    r = json.loads(line)
    if r['old_status'] != 'training_pair_available' and (not r['accepted']):
        pending[r['id']] = r
assert len(pending) == 26767
db = sqlite3.connect(cfg.index)
sentences = {}
lookup = collections.defaultdict(list)
books = set()
chapters = set()
for sid, work, ch, v, dialect, payload in db.execute('select id,work,chapter,verse,dialect,payload from sentence'):
    key = explicit_bible_location(work, ch, v)
    if key:
        sentences[sid] = json.loads(payload)
        lookup[key].append(sid)
        books.add((key[0], dialect))
        chapters.add((key[0], key[1], dialect))
counts = collections.Counter()
examples = {}
seen = set()
with (folder / '').open('w') as detailed, (folder / '').open('w') as tab:
    writer = csv.writer(tab, delimiter='\t')
    writer.writerow(['', '', '', '', '', '', '', '', '', '', '', '', ''])
    for line in (cfg.derived / 'complete_citations/citations.jsonl').open():
        source = json.loads(line)
        oid = source['id']
        if oid not in pending:
            continue
        assert oid not in seen
        seen.add(oid)
        r = pending[oid]
        ref = source['reference']
        links = r.get('expanded_local_links', source['local_links'])
        current = [sentences[l['sentence_id']] for l in links if l['sentence_id'] in sentences]
        kind = ref['kind']
        evidence = []
        if kind != 'verse':
            reason = ''
            evidence.append('' + kind + '' + source['raw'])
        elif r['status'] == '':
            direct = {t for f in r['forms'] for t in byform.get(f, [])}
            roots = byform.get(r['root'], [])
            if direct:
                reason = ''
                evidence.append('' + ','.join(sorted(direct)))
            elif roots:
                reason = ''
                evidence.append('' + ','.join(sorted(roots)) + '')
            else:
                reason = ''
                evidence.append('' + '/'.join(r['forms']))
        elif not current:
            allowed = registry.from_codes(source['dialect_codes'])
            key = (ref.get('book_key'), ref.get('chapter'), ref.get('verse'))
            other = [sentences[s]['dialect_id'] for s in lookup.get(key, [])]
            if source['dialect_codes'] and (not allowed):
                reason = ''
            elif other and allowed and (not set(other) & set(allowed)):
                reason = ''
            elif not any(((key[0], d) in books for d in allowed or registry.data)):
                reason = ''
            elif not any(((key[0], key[1], d) in chapters for d in allowed or registry.data)):
                reason = ''
            else:
                reason = ''
            evidence.append('' + ','.join(source['dialect_codes']) + '' + ','.join(sorted(set(other))))
        elif r['checks']:
            issues = sorted({x for p in r['checks'] for x in p['issues']})
            if issues:
                reason = ''
                evidence.extend(issues)
            else:
                reason = ''
                evidence.append('' + ','.join(sorted({p['tla'] for p in r['checks']})))
        else:
            reason = ''
            evidence.append('' + ','.join(r['proposed_tlas']))
            if re.search('annotation: (different reading|variant|omits)', source['paragraph'] or ''):
                evidence.append('')
        corpus_present = bool(current)
        target_forms = {t: dict(gloss=lex[t].gloss, pos=lex[t].pos, forms=lex[t].forms, dialects=lex[t].form_dialects) for t in r['proposed_tlas']}
        neighbor_ids = []
        for c in current:
            loc = explicit_bible_location(c['work'], c['chapter'], c['verse'])
            if loc and loc[2].isdigit():
                for delta in [-2, -1, 1, 2]:
                    for sid in lookup.get((loc[0], loc[1], str(int(loc[2]) + delta)), []):
                        if sentences[sid]['source'] == c['source']:
                            neighbor_ids.append(sid)
        neighbors = [sentences[s] for s in dict.fromkeys(neighbor_ids)]
        row = dict(id=oid, reason=reason, evidence=evidence, reference=source['normalized_reference'], file=source['file'], page=source['page'], raw=source['raw'], root=r['root'], forms=r['forms'], target_entries=target_forms, paragraph=source['paragraph'], corpus_present=corpus_present, current_sentences=current, neighbor_sentences=neighbors, checks=r['checks'])
        detailed.write(json.dumps(row, ensure_ascii=False) + '\n')
        writer.writerow([oid, reason, row['reference'], row['file'], row['page'], r['root'], '/'.join(r['forms']), ','.join(r['proposed_tlas']), '；'.join(evidence), row['paragraph'], ' | '.join((c['sent'] for c in current)), ' | '.join((c['translation'] for c in current)), ','.join((c['id'] for c in neighbors))])
        counts[reason] += 1
        if reason not in examples:
            examples[reason] = row
assert seen == set(pending) and sum(counts.values()) == 26767
(folder / '').write_text(json.dumps(counts, ensure_ascii=False, indent=2) + '\n')
lines = ['', '', '', '', '', '|---|---:|']
lines += [f'| {reason} | {n:,} |' for reason, n in counts.most_common()]
for reason, n in counts.most_common():
    r = examples[reason]
    lines += ['', '## ' + reason, '', f"{n:,}{r['id']}{r['file']}{r['page']}{r['reference']}。", '', '' + '；'.join(r['evidence']), '', '', r['paragraph'] or '']
    for c in r['current_sentences'][:1]:
        lines += ['', f"{c['id']}：", c['sent'], '', c['translation']]
    lines += ['', '']
    if '' in reason or '' in reason or '' in reason:
        lines += ['']
    elif '' in reason or '' in reason or '' in reason:
        lines += ['']
    elif '' in reason or '' in reason:
        lines += ['']
    elif '' in reason or '' in reason:
        lines += ['']
    else:
        lines += ['']
(folder / '').write_text('\n'.join(lines) + '\n')
print(json.dumps(counts, ensure_ascii=False, indent=2))
