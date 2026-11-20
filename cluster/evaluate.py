
pass
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from coptic_examples.config import Config,write_json
from coptic_examples.dialects import Dialects
from coptic_examples.corpus import Corpus
from coptic_examples.shortlist import Shortlist,export


def main():
    config=Config();summary=[];ranker='research' if '--research' in sys.argv else 'cheap'
    for dialect in Dialects().data:
        engine=Shortlist(config,dialect)
        with Corpus(config).connect() as db:
            cohort=[r[0] for r in db.execute('SELECT tla FROM candidate WHERE dialect=? GROUP BY tla HAVING COUNT(*)>=10 ORDER BY COUNT(*) DESC,tla LIMIT 12',(dialect,))]
        if dialect=='sahidic':cohort=list(dict.fromkeys(['C998','C992','C3519','C5945',*cohort]))
        for tla in cohort:
            r=engine.search(tla,ranker=ranker)
            packet=export(config,r,'research_shortlists' if ranker=='research' else 'shortlists')
            if len(r['shortlist'])<3:continue
            for method,selected in packet['local_selection_experiments'].items():
                summary.append(dict(TLA=tla,dialect=dialect,method=method,shortlist_size=len(r['shortlist']),**selected['metrics']))
            print(dialect,tla,len(r['shortlist']),flush=True)
    write_json(config.work/('cluster/research_results.json' if ranker=='research' else 'cluster/results.json'),summary)
    md=['','',
        '','',
        '','|---|---|---:|---:|---:|---:|']
    for dialect in Dialects().data:
        for method in ['rank','kmeans','coverage']:
            rows=[r for r in summary if r['dialect']==dialect and r['method']==method]
            if not rows:continue
            mean=lambda key:sum(r[key] for r in rows)/len(rows)
            md.append(f"| {dialect} | {method} | {len(rows)} | {mean('mean_quality'):.4f} | {mean('weighted_syntax_coverage'):.4f} | {mean('mean_lexical_redundancy'):.4f} |")
    md+=['','',
         '',
         '']
    if ranker=='research':
        md.insert(2,'')
        md[-1]=''
    (config.work/('cluster/RESEARCH_RESULTS.md' if ranker=='research' else 'cluster/RESULTS.md')).write_text('\n'.join(md)+'\n',encoding='utf-8')


if __name__=='__main__':main()
