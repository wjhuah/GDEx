pass
import sys, json, sqlite3
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from coptic_examples.config import Config
from coptic_examples.source_pool import SupplementedCorpus
from coptic_examples.annotations import load_manual
from coptic_examples.dataset import text_key
from coptic_examples.text import real_translation, editorial_heading
cfg = Config()
out = cfg.reports / 'local_2218a127/local_bde2e0cb'
out.mkdir(parents=True, exist_ok=True)
corpus = defaultdict(Counter)
unique = defaultdict(set)
with SupplementedCorpus(cfg).connect() as db:
    for payload, in db.execute('SELECT payload FROM sentence'):
        c = json.loads(payload)
        d = c['dialect_id']
        corpus[d]['spans'] += 1
        corpus[d]['tt_tokens'] += len(c['tokens'])
        unique[d].add(text_key(c))
        corpus[d]['nonempty_translation'] += bool(c['translation'].strip())
        corpus[d]['translation_checks_passed'] += bool(real_translation(c['translation']) and (not editorial_heading(c['translation'])))
for d in corpus:
    corpus[d]['distinct_coptic_texts'] = len(unique[d])
training = {}
for d in ['sahidic', 'bohairic']:
    c = Counter()
    rank_groups = defaultdict(list)
    for r in map(json.loads, (cfg.derived / f'model_rows_{d}.jsonl').open()):
        c['positive' if r['y'] > 0 else 'negative'] += 1
        for source in ['tab', 'crum']:
            ranks = [v for v in r['ranks'] if v['source'] == source]
            if r['y'] > 0 and ranks:
                c[source + '_ranked_rows'] += 1
                rank_groups[source, r['tla']].extend((v['grade'] for v in ranks))
    for source in ['tab', 'crum']:
        c[source + '_ranked_entries'] = sum((s == source for s, t in rank_groups))
        c[source + '_entries_with_multiple_grades'] = sum((s == source and len(set(v)) > 1 for (s, t), v in rank_groups.items()))
    training[d] = c
manual = load_manual(cfg.manual)
priorities = Counter((r['priority'] for r in manual))
tab = dict(source_rows=len(manual), rows_with_explicit_priority=sum((r['priority'].isdigit() for r in manual)), priority_counts=priorities, entries_with_ALL_rule=len({r['TLA'] for r in manual if 'ALL' in r['forbidden_urns'].upper()}))
coverage = json.loads((cfg.reports / 'local_2218a127/local_5ae67291/local_3d2c3caf').read_text())
crum = json.loads((cfg.reports / 'local_2218a127/local_40c6713e/local_4310a6c0').read_text())
augmentation = json.loads((cfg.reports / 'local_2218a127/local_77165ac0/local_1d640cde').read_text())
homographs = json.loads((out / '').read_text())
result = dict(corpus=corpus, training=training, tab=tab, online=coverage['online'], online_alignment=coverage['alignment'], crum_occurrences=crum['exclusive_statuses'], augmentation_fold_rows=sum((r['rows'] for r in augmentation)), augmentation_homograph_fold_rows=sum((r['homograph'] for r in augmentation)), augmentation_unrelated_fold_rows=sum((r['unrelated'] for r in augmentation)), experimental_recovered_training_rows=4422, homographs=homographs)
(out / '').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
lines = ['', '', '', '', '', '', '', '|---|---:|---|', '', '', '', '', '', '', '', '', '', '', '', '', '', '|---|---:|---:|---:|---:|---:|']
for d, c in corpus.items():
    lines.append(f"| {d} | {c['spans']:,} | {c['tt_tokens']:,} | {c['distinct_coptic_texts']:,} | {c['nonempty_translation']:,} | {c['translation_checks_passed']:,} |")
lines += ['', '', '', '', '', '', '', '', '|---|---:|---:|---:|', '| Sahidic | 11,486 | 86,625 | 98,111 |', '| Bohairic | 502 | 2,155 | 2,657 |', '', '', '', '', '', '', '', '', '', '|---|---:|---:|---:|---:|---:|']
for d, c in training.items():
    lines.append(f"| {d} | {c['tab_ranked_rows']} | {c['tab_ranked_entries']} | {c['tab_entries_with_multiple_grades']} | {c['crum_ranked_rows']:,} | {c['crum_ranked_entries']:,} |")
lines += ['', '', '', '', '', '', '|---|---:|---|', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', f"{homographs['canonical_families']:,}{homographs['canonical_entries']:,}{homographs['canonical_within_dialect']:,}", '', f"{homographs['all_form_collision_groups']:,}{homographs['all_form_within_dialect']:,}{homographs['all_form_entries']:,}", '', '', '', '', '']
(out / '').write_text('\n'.join(lines) + '\n')
print(json.dumps(dict(corpus=corpus, tab=tab), ensure_ascii=False, indent=2))
