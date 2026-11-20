import json, html, re, sys, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from coptic_examples.token_highlight import spans
from coptic_examples.text import STOP
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/annotation_20261120'
OUT.mkdir(parents=True, exist_ok=True)
SRC = Path(sys.argv[1])

def marked_coptic(c):
    positions = spans(c['sent'], c['tokens'])
    wanted = {c['tokens'][i]['id'] for i in c['tt_target_indices']}
    if positions is None:
        return html.escape(c['sent'])
    result = []
    cursor = 0
    for tid, a, b in positions:
        if tid in wanted:
            result.extend([html.escape(c['sent'][cursor:a]), '<mark>' + html.escape(c['sent'][a:b]) + '</mark>'])
            cursor = b
    result.append(html.escape(c['sent'][cursor:]))
    return ''.join(result)

def marked_english(text, gloss):
    terms = set(re.findall('[a-z]+', gloss.lower())) - STOP - {'object', 'none', 'sb', 'sth', 's', 'o', 'p', 'f', 'm', 'sg', 'plural', 'introduces', 'indicating'}
    if not terms:
        terms = set(re.findall('[a-z]+', gloss.lower()))
    irregular = {'man': ['men'], 'give': ['gave', 'given'], 'see': ['saw', 'seen'], 'go': ['went', 'gone'], 'be': ['is', 'are', 'was', 'were'], 'say': ['said'], 'fall': ['fell', 'fallen'], 'come': ['came']}
    variants = set(terms)
    for word in terms:
        variants.update([word + 's', word + 'es', word + 'ed', word + 'ing', word.rstrip('e') + 'ing'])
        variants.update(irregular.get(word, []))
    return re.sub('[A-Za-z]+', lambda m: '<mark>' + html.escape(m[0]) + '</mark>' if m[0].lower() in variants else html.escape(m[0]), text)

def main():
    audit = json.loads((SRC / 'selection_audit.json').read_text())
    pools = json.loads((SRC / 'candidate_pools.json').read_text())
    template = (ROOT / 'assets/annotation.html').read_text()
    for method, name in [('bags', 'A'), ('fsum', 'B')]:
        entries = []
        for target in audit['selection']:
            p = pools[target['tla']]
            choice = audit['methods'][target['tla']][method]['five']
            rows = []
            for i in choice:
                c = p['candidates'][i]
                assert c['bound_groups'] <= 20
                rows.append(dict(id=c['id'], coptic=marked_coptic(c), english=marked_english(c['translation'], p['gloss'])))
            rows.sort(key=lambda r: hashlib.sha256((name + target['tla'] + r['id']).encode()).hexdigest())
            entries.append(dict(tla=p['tla'], lemma=p['lemma'], gloss=p['gloss'], rows=rows))
        data = json.dumps(dict(packet=name, entries=entries), ensure_ascii=False).replace('<', '\\u003c')
        (OUT / f'packet_{name}.html').write_text(template.replace('__DATA__', data))
    print('Exported two annotation packets', flush=True)
if __name__ == '__main__':
    main()
