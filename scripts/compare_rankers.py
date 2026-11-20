pass
import json
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from coptic_examples.config import Config, write_json
from coptic_examples.retrieval import Search

def main():
    config = Config()
    search = Search(config)
    summary = []
    md = ['', '', '', '']
    for tla in ['C998', 'C992', 'C3519', 'C5945']:
        results = {}
        for mode in ['linear', 'learned']:
            print(f'Search {tla} / {mode}', flush=True)
            r = search.search(tla, ranker=mode)
            write_json(config.reports / f'{tla}_{mode}.json', r)
            results[mode] = r
            assert all((x['translation'].strip() not in {'', '...', 'ؤ'} for x in r['results']))
        md += [f"## {tla} — {results['learned']['entry']['gloss']}", '', f"{results['learned']['candidate_count']}{results['learned']['semantic_scored_count']}{json.dumps(results['learned']['excluded'])}`。", '', '| Rank | Linear | Learned |', '|---:|---|---|']
        for i in range(max((len(r['results']) for r in results.values()))):
            cells = []
            for mode in ['linear', 'learned']:
                rs = results[mode]['results']
                if i >= len(rs):
                    cells.append('—')
                    continue
                c = rs[i]
                cells.append(c['translation'].replace('|', '/') + f"<br>{c['title']} {c['chapter']}:{c['verse']} · {c['score']:.3f}")
            md.append(f'| {i + 1} | {cells[0]} | {cells[1]} |')
        ids = [{r['id'] for r in results[m]['results']} for m in ['linear', 'learned']]
        summary.append(dict(TLA=tla, candidate_count=results['learned']['candidate_count'], semantic_scored_count=results['learned']['semantic_scored_count'], top10_overlap=len(ids[0] & ids[1]), manual_examples=len(results['learned']['manual_examples'])))
    md += ['', '', '', '', '']
    (config.reports / 'CORPUS_COMPARISON.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    write_json(config.reports / 'corpus_comparison_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
