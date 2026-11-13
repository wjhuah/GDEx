#!/usr/bin/env python3
"""Matched-fold ranking tests for context frequency and lexical sense overlap."""
import json,sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from coptic_examples.config import Config,write_json
from coptic_examples.ablation import CHEAP,fit_ranker,predict,evaluate,summary,difference
cfg=Config();rows=[json.loads(l) for l in (cfg.derived/'ablation_rows.jsonl').open()]
frequency={'mean_word_frequency','min_word_frequency','mean_log_word_frequency','rare_word_ratio','top1k_ratio','top5k_ratio','top10k_ratio'}
setups={'full':CHEAP,'without_frequency':[c for c in CHEAP if c not in frequency],'without_sense_overlap':[c for c in CHEAP if c!='sense_overlap']}
results=[]
for d in ['sahidic','bohairic']:
 data=[r for r in rows if r['dialect']==d];families=sorted({r['family'] for r in data});np.random.default_rng(42).shuffle(families);folds={f:i%5 for i,f in enumerate(families)}
 ms={k:[] for k in setups}
 for f in range(5):
  val=[r for r in data if folds[r['family']]==f];held={r['text_key'] for r in val};train=[r for r in data if folds[r['family']]!=f and r['text_key'] not in held]
  if not val:continue
  for name,cols in setups.items():
   m=fit_ranker(train,cols);ms[name].extend(evaluate(val,predict(m,val,cols)))
  print(d,f,'done',flush=True)
 results.append(dict(dialect=d,summary={k:summary(v) for k,v in ms.items()},comparisons={k:{task:difference(ms[k],ms['full'],task) for task in ['acceptance','manual_order','crum_style']} for k in setups if k!='full'}))
write_json(cfg.reports/'sense_scope_comparison/frequency_ablation.json',dict(protocol='Same rows, spelling-family folds and purged Coptic spans as current full ranking experiment; remove a feature block and refit. Not a token-sense accuracy test.',removed_frequency=sorted(frequency),results=results))
