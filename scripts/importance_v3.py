#!/usr/bin/env python3
import json,sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import lightgbm as lgb
from coptic_examples.config import Config,write_json
cfg=Config();rows=[]
paths=list((cfg.source/'crum3').glob('lgbm_*.txt'))+list((cfg.work/'models').rglob('*.txt'))
for p in paths:
 try:m=lgb.Booster(model_file=str(p))
 except Exception:continue
 gains=m.feature_importance('gain');splits=m.feature_importance('split');total=max(1.,sum(gains))
 descriptor=p.with_suffix('.json');meta=json.loads(descriptor.read_text()) if descriptor.exists() else {}
 active=meta.get('kind')=='lightgbm' and (meta.get('file')==p.name or meta.get('model_file')==p.name)
 rows.append(dict(model=str(p),status='deployed' if active else 'historical_or_optional_not_currently_selected',cosine_present='sm_cosine' in m.feature_name() or 'sense_cosine' in m.feature_name(),features=[dict(name=n,gain=float(g),gain_fraction=float(g/total),split=int(s)) for n,g,s in sorted(zip(m.feature_name(),gains,splits),key=lambda t:-t[1])]))
write_json(cfg.reports/'sense_v3/importance.json',rows)
print('Saved actual feature importances for',len(rows),'models')
