#!/usr/bin/env python3
"""Crum ordering model using UD features only; definition gate handles relevance."""
import sys,json
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from coptic_examples.config import Config,write_json
from coptic_examples.ablation import CHEAP
from coptic_examples.features import UD
from coptic_examples.shortlist import pair_data,fit_linear,decision
cfg=Config();rows=[json.loads(l) for l in (cfg.derived/'ablation_rows.jsonl').open()]
indices=[CHEAP.index(k) for k in UD];reports=[]
for d in ['sahidic','bohairic']:
 data=[dict(r,x=[r['x'][i] for i in indices]) for r in rows if r['dialect']==d and r['task']=='crum_style']
 families=sorted({r['family'] for r in rows if r['dialect']==d});np.random.default_rng(42).shuffle(families);fm={v:i%5 for i,v in enumerate(families)};cv=[]
 for fold in range(5):
  val=[r for r in data if fm[r['family']]==fold];held={r['text_key'] for r in rows if r['dialect']==d and fm[r['family']]==fold}
  tr=[r for r in data if fm[r['family']]!=fold and r['text_key'] not in held];x,y=pair_data(tr);vx,vy=pair_data(val)
  if len(y) and len(vy):
   m=fit_linear(x,y,False);s=decision(m,vx);acc=float(np.mean(np.where(s==0,.5,(s>0)==vy)))
   cv.append(dict(fold=fold,pairs=len(vy)//2,accuracy=acc))
 x,y=pair_data(data)
 if len(y):
  m=fit_linear(x,y,False);m['columns']=UD;m['provisional']=len(data)<100;m['training_rows']=len(data);m['validation']=cv
  write_json(cfg.work/'models'/d/'syntax_order.json',m)
  reports.append(dict(dialect=d,rows=len(data),validation=cv,mean_accuracy=float(np.mean([r['accuracy'] for r in cv])) if cv else None,provisional=m['provisional'],standardized_pair_coefficients=sorted(zip(UD,m['coefficients']),key=lambda v:-abs(v[1]))))
 # Matched-fold ablation of the token-count input.
 reduced=[dict(r,x=[v for k,v in zip(UD,r['x']) if k!='n_tokens']) for r in data];no_length=[]
 for fold in range(5):
  val=[r for r in reduced if fm[r['family']]==fold];held={r['text_key'] for r in rows if r['dialect']==d and fm[r['family']]==fold}
  tr=[r for r in reduced if fm[r['family']]!=fold and r['text_key'] not in held];x,y=pair_data(tr);vx,vy=pair_data(val)
  if len(y) and len(vy):
   model=fit_linear(x,y,False);s=decision(model,vx)
   no_length.append(dict(fold=fold,pairs=len(vy)//2,accuracy=float(np.mean(np.where(s==0,.5,(s>0)==vy)))))
 if reports and reports[-1]['dialect']==d:
  reports[-1]['without_n_tokens']=dict(validation=no_length,mean_accuracy=float(np.mean([v['accuracy'] for v in no_length])) if no_length else None)
  reports[-1]['evaluation_scope']='Existing extracted Crum preference rows; spelling-family folds with held-out Coptic spans purged. Exploratory; expert token-sense validation pending.'
write_json(cfg.reports/'definition_first_syntax_models.json',reports)
print('UD ordering models saved')
