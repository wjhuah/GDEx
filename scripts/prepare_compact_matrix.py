import sys,json,joblib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from coptic_examples.config import Config
from coptic_examples.lexicon import load
from coptic_examples.compact_features import compact_vector
cfg=Config()
lex=load(cfg.lexicon)
freq=json.loads((cfg.work/'models/sahidic/frequencies.json').read_text())
rows=[]
vectors=[]
for a,b in zip((cfg.derived/'model_rows_sahidic.jsonl').open(),(cfg.derived/'adjusted_20261113/model_rows_sahidic.jsonl').open()):
 old=json.loads(a)
 r=json.loads(b)
 full=[float(v) if v is not None else float('nan') for v in r['x']]
 vectors.append(compact_vector(old['candidate'],lex[r['tla']],full,freq))
 r.pop('x')
 rows.append(r)
folds=np.array([r['fold'] for r in json.loads((cfg.derived/'adjusted_20261113/fold_manifest_sahidic.json').read_text())])
assert len(rows)==len(folds)
path=Path(sys.argv[1])
path.parent.mkdir(parents=True,exist_ok=True)
joblib.dump((rows,np.asarray(vectors),np.array([r['y']>0 for r in rows]),folds),path)
