"""Recompute within-entry citation-order correlation from original corpus texts."""
import sys,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr,rankdata
root=Path(__file__).resolve().parents[1]
out=root/'reports/本地复盘/空格分组相关性';out.mkdir(parents=True,exist_ok=True)
entries=[];summary={}
for dialect in ['sahidic','bohairic']:
 groups=defaultdict(list)
 for r in map(json.loads,(root/f'data/derived/model_rows_{dialect}.jsonl').open()):
  if r['y']<=0 or 'crum' not in r['sources']:continue
  positions=[v['position'] for v in r['ranks'] if v['source']=='crum' and 'position' in v]
  if not positions:continue
  c=r['candidate'];groups[r['tla']].append(dict(position=min(positions),bound_groups=len(c['sent'].split()),id=c['id'],text=c['sent']))
 for tla,rr in groups.items():
  rr.sort(key=lambda r:r['position']);x=[r['position'] for r in rr];y=[r['bound_groups'] for r in rr]
  reason='可计算' if len(rr)>=3 and len(set(x))>1 and len(set(y))>1 else '少于三条' if len(rr)<3 else '顺序或长度无变化'
  rho=float(spearmanr(x,y).statistic) if reason=='可计算' else None
  length_ranks=rankdata(y,method='average').tolist()
  for r,rank in zip(rr,length_ranks):r['length_rank']=rank
  entries.append(dict(dialect=dialect,tla=tla,n=len(rr),reason=reason,rho=rho,examples=rr))
 vals=[r['rho'] for r in entries if r['dialect']==dialect and r['rho'] is not None]
 summary[dialect]=dict(aligned_examples=sum(len(rr) for rr in groups.values()),entries=len(groups),effective_entries=len(vals),median=float(np.median(vals)),mean=float(np.mean(vals)),positive=sum(v>0 for v in vals),negative=sum(v<0 for v in vals),zero=sum(v==0 for v in vals))
(out/'逐词条重算.json').write_text(json.dumps(entries,ensure_ascii=False,indent=2)+'\n')
(out/'汇总.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
for r in entries:
 if r['tla']=='C998' and r['dialect']=='sahidic':print('C998',[(v['position'],v['bound_groups'],v['length_rank']) for v in r['examples']],r['rho'])
