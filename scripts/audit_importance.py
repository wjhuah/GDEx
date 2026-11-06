#!/usr/bin/env python3
"""Matched held-out diagnosis of punctuation and pronouns, with source and sense checks."""
import sys,json,copy
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from coptic_examples.config import Config,write_json
from coptic_examples.ablation import CHEAP,fit_ranker,predict,evaluate,summary,difference,CachedVectors
from coptic_examples.lexicon import load
cfg=Config();rows=[json.loads(l) for l in (cfg.derived/'ablation_rows.jsonl').open() if json.loads(l)['dialect']=='sahidic']
setups={'full':CHEAP,'without_n_punctuation':[c for c in CHEAP if c!='n_punctuation'],'without_pronoun_ratio':[c for c in CHEAP if c!='pronoun_ratio'],'without_both':[c for c in CHEAP if c not in {'n_punctuation','pronoun_ratio'}]}
families=sorted({r['family'] for r in rows});np.random.default_rng(42).shuffle(families);folds={f:i%5 for i,f in enumerate(families)};metrics={k:[] for k in setups};pm={};fold_records=[]
for f in range(5):
 val=[r for r in rows if folds[r['family']]==f];held={r['text_key'] for r in val};tr=[r for r in rows if folds[r['family']]!=f and r['text_key'] not in held]
 for name,cols in setups.items():
  m=fit_ranker(tr,cols);scores=predict(m,val,cols);metrics[name].extend(evaluate(val,scores))
  if name=='full':
   groups=defaultdict(list)
   for i,r in enumerate(val):groups[r['group']].append(i)
   for feature in ['n_punctuation','pronoun_ratio']:
    key='permute_'+feature;pm.setdefault(key,[]);idx=CHEAP.index(feature);copied=[dict(r,x=list(r['x'])) for r in val];rng=np.random.default_rng(422+f)
    for inds in groups.values():
     perm=rng.permutation(inds)
     for dest,source in zip(inds,perm):copied[dest]['x'][idx]=val[source]['x'][idx]
    pm[key].extend(evaluate(val,predict(m,copied,cols)))
 fold_records.append(dict(fold=f,train=len(tr),validation=len(val),held_texts=len(held)))
 print('fold',f,'complete',flush=True)
allmetrics={**metrics,**pm};stats=[]
for task in ['acceptance','crum_style']:
 for label in ([0,1] if task=='acceptance' else [None]):
  sub=[r for r in rows if r['task']==task and (label is None or r['gain']==label)]
  stats.append(dict(task=task,label=label,n=len(sub),n_punctuation_mean=float(np.mean([r['x'][CHEAP.index('n_punctuation')] for r in sub])),punctuation_zero=float(np.mean([r['x'][CHEAP.index('n_punctuation')]==0 for r in sub])),pronoun_ratio_mean=float(np.mean([r['x'][CHEAP.index('pronoun_ratio')] for r in sub]))))
bycorpus=defaultdict(list);tags=Counter();seen=set()
for r in rows:
 c=r['candidate'];key=c['id']
 if key not in seen:
  tags.update(t['xpos'] for t in c['tokens'] if t['pos']=='PRON');seen.add(key)
 if r['task']=='acceptance':bycorpus[c['source'].split('/')[0]].append(r)
source=[]
for name,rr in bycorpus.items():
 source.append(dict(corpus=name,n=len(rr),positive_rate=float(np.mean([r['gain'] for r in rr])),punctuation_zero=float(np.mean([r['x'][CHEAP.index('n_punctuation')]==0 for r in rr])),pronoun_ratio_mean=float(np.mean([r['x'][CHEAP.index('pronoun_ratio')] for r in rr]))))
lex=load(cfg.lexicon);vec=CachedVectors(cfg);sense=[]
for r in rows:
 if r['task']!='acceptance' or r['gain']<=0 or r['TLA'] not in {'C992','C998'}:continue
 other=lex[{'C992':'C998','C998':'C992'}[r['TLA']]];x=vec.target_features(r['candidate'],other,'sahidic')
 sense.append(dict(own=r['TLA'],cross=other.tla,id=r['candidate']['id'],translation=r['candidate']['translation'],identical_nonsemantic_features=bool(np.array_equal(r['x'][:-1],x[:-1])),identical_all_45=bool(np.array_equal(r['x'],x)),own_overlap=r['x'][-1],cross_overlap=x[-1],own_cosine=r['cosine'],cross_cosine=vec.cosine(other.gloss,r['candidate']['translation'])))
vec.connection.close()
output=dict(protocol='Fixed existing extracted rows; spelling-family five-fold split; held-out Coptic texts purged from training; identical validation candidates; removal models retrained; permutation within query groups with one fixed shuffle. Exploratory.',folds=fold_records,summaries={k:summary(v) for k,v in allmetrics.items()},comparisons={k:{task:difference(v,metrics['full'],task) for task in ['acceptance','manual_order','crum_style']} for k,v in allmetrics.items() if k!='full'},label_feature_statistics=stats,source_statistics=sorted(source,key=lambda r:-r['n']),pronoun_xpos_counts=dict(tags.most_common()),father_barley_feature_pairs=sense)
write_json(cfg.reports/'importance_audit/results.json',output)
print(json.dumps(output['summaries'],indent=2),flush=True)
