"""Build versioned comparison inputs while retaining the original rows and folds."""
import sys,json,hashlib,sqlite3,copy
from pathlib import Path
from collections import Counter,defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from coptic_examples.config import Config
from coptic_examples.lexicon import load
from coptic_examples.inclusion import EntryFeatures
from coptic_examples.token_review import decisions
from coptic_examples.sense import align_candidate
from coptic_examples.integrity import alignment_issue
from coptic_examples.dataset import text_key
cfg=Config();lex=load(cfg.lexicon);dest=cfg.derived/'adjusted_20261113';dest.mkdir(exist_ok=True)
report=cfg.reports/'本地复盘/调整后算法比较';report.mkdir(parents=True,exist_ok=True)
versions=defaultdict(list)
p=cfg.work/'cache/upstream_corpus.sqlite3'
if p.exists():
 with sqlite3.connect(p) as db:
  for (payload,) in db.execute('select payload from sentence'):
   c=json.loads(payload);versions[c['source'],str(c['chapter']),str(c['verse'])].append(c)
summary={}
for dialect in ['sahidic','bohairic']:
 source=cfg.derived/f'model_rows_{dialect}.jsonl';manifest=json.loads((cfg.derived/f'fold_manifest_{dialect}.json').read_text());new_manifest=[];counts=Counter();features=EntryFeatures(cfg,lex,dialect)
 with source.open() as inp,(dest/source.name).open('w') as out:
  for index,(line,m) in enumerate(zip(inp,manifest)):
   r=json.loads(line);c=r['candidate'];assert (r['tla'],r['text_key'],c['id'])==(m['tla'],m['text_key'],m['sentence_id']);counts['original']+=1
   choices=versions.get((c['source'],str(c['chapter']),str(c['verse'])),[])
   exact=[x for x in choices if x['translation']==c['translation']]
   replacement=exact[0] if len(exact)==1 else choices[0] if len(choices)==1 else None
   changed=False
   if replacement and (replacement['sent']!=c['sent'] or replacement['tokens']!=c['tokens']):
    c=copy.deepcopy(replacement);changed=True;counts['updated_source_span']+=1
   issue=alignment_issue(c)
   if issue:counts['excluded_'+issue]+=1;continue
   review=decisions(c,r['tla'])
   if changed or review:
    ii=align_candidate(c,lex[r['tla']],True)
    if r['y']>0 and not ii:counts['excluded_positive_target_link']+=1;continue
    r['x']=features.vector(c,lex[r['tla']]);counts['recomputed_features']+=1
   key=text_key(c) if changed else r['text_key']
   row=dict(tla=r['tla'],family=r['family'],text_key=key,candidate={'id':c['id']},x=r['x'],y=r['y'],ranks=r['ranks'],original_row=index)
   out.write(json.dumps(row,ensure_ascii=False,allow_nan=True)+'\n');new_manifest.append(dict(m,tla=r['tla'],text_key=key,sentence_id=c['id']));counts['retained']+=1;counts['positive' if r['y']>0 else 'negative']+=1
 features.close();(dest/f'fold_manifest_{dialect}.json').write_text(json.dumps(new_manifest,ensure_ascii=False));summary[dialect]=dict(counts);print(dialect,dict(counts),flush=True)
(report/'input_changes.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
