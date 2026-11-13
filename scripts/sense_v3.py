#!/usr/bin/env python3
"""Auditable cross-sense/unrelated augmentation and document-held-out evaluation."""
import hashlib,json,sys,pickle
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score
from coptic_examples.config import Config,write_json,database
from coptic_examples.lexicon import load
from coptic_examples.dataset import read_rows,text_key
from coptic_examples.corpus import Corpus
from coptic_examples.sense import targets,sense_text
from coptic_examples.text import english
from coptic_examples.dialects import Dialects

WORDS=['ϩⲟⲟⲩ','ⲧⲱⲙⲧ','ⲉϩⲉ','ⲡⲉ','ⲉ']


def run():
 cfg=Config();lex=load(cfg.lexicon);corpus=Corpus(cfg);registry=Dialects();out=cfg.reports/'sense_v3';out.mkdir(exist_ok=True)
 comparison='--scope-comparison' in sys.argv
 if comparison:out=cfg.reports/'sense_scope_comparison';out.mkdir(exist_ok=True)
 focus_tlas={e.tla for e in lex.values() if e.lemma in WORDS+['ϣⲁ']}|{'C2787','C4008'}
 sentences={c['id']:c for c in corpus.all() if c['dialect_id'] in {'sahidic','bohairic'}}
 families=defaultdict(list)
 for e in lex.values():families[e.lemma].append(e)
 homographs={k:v for k,v in families.items() if len(v)>1 and len({e.gloss for e in v})>1}
 byform=defaultdict(dict)
 for e in lex.values():
  for form in e.forms:byform[form][e.tla]=e
 collisions={k:v for k,v in byform.items() if len({e.gloss for e in v.values()})>1}
 positives=[];negatives=[];seen=set()
 for r in read_rows(cfg):
  if r['TLA'] not in lex or r['evidence'] not in {'manual','manual_forbidden_rule'}:continue
  c=r['candidate'];e=lex[r['TLA']];key=(e.tla,c['dialect_id'],text_key(c))
  if key in seen:continue
  seen.add(key)
  item=dict(tla=e.tla,cid=c['id'],text_key=key[-1],dialect=c['dialect_id'],work=c['work'],y=int(r['evidence']=='manual'),kind=r['evidence'],weight=1.)
  (positives if item['y'] else negatives).append(item)
 # A unique Crum binding is weak positive evidence, never expert token gold.
 with database(cfg.work/'cache/crum_bible.sqlite3') as db:
  for payload,sid in db.execute("SELECT o.payload,l.sentence_id FROM occurrence o JOIN link l ON o.id=l.occurrence_id WHERE l.status='verified_target_form'"):
   r=json.loads(payload)
   if len(r['tla_candidates'])!=1 or sid not in sentences:continue
   tla=r['tla_candidates'][0];e=lex[tla];c=sentences[sid];key=(tla,c['dialect_id'],text_key(c))
   if key in seen or not targets(c,e):continue
   seen.add(key);positives.append(dict(tla=tla,cid=sid,text_key=key[-1],dialect=c['dialect_id'],work=c['work'],y=1,kind='crum_weak',weight=.35))
 data=[r for r in positives+negatives if r['cid'] in sentences]
 # Exact translated/Coptic duplicates are kept together during splitting.
 def fold(r):return int(hashlib.sha256(r['work'].encode()).hexdigest()[:8],16)%5
 vec=HashingVectorizer(n_features=2**17,alternate_sign=False,lowercase=False,token_pattern=r'[^ ]+',norm='l2')
 def matrix(rows):return vec.transform(sense_text(sentences[r['cid']],lex[r['tla']]) for r in rows)
 def augment(train,mode,held):
  focused=mode.startswith('focus_');mode=mode.removeprefix('focus_')
  if mode=='none':return []
  positive=[r for r in train if r['y'] and targets(sentences[r['cid']],lex[r['tla']])]
  known={(r['tla'],r['text_key']) for r in train};generated=[];counts=Counter()
  for source in positive:
   if focused and source['tla'] not in focus_tlas:continue
   own=lex[source['tla']]
   competing={tla:e for form in own.forms for tla,e in collisions.get(form,{}).items()}
   if not competing:continue
   for target in competing.values():
    if focused and target.tla not in focus_tlas:continue
    codes={code for values in (target.form_dialects or {}).values() for code in values}
    if codes and source['dialect'] not in registry.from_codes(codes):continue
    key=(target.tla,source['text_key'])
    if target.tla==own.tla or target.gloss==own.gloss or key in known or counts[target.tla,source['dialect'],'cross']>=20:continue
    c=sentences[source['cid']]
    # If another exact target token is possible, do NOT label the whole sentence negative.
    compatible=targets(c,target)
    kind='cross_native_incompatible';weight=.35
    if compatible and mode!='naive_all':
     if compatible!=targets(c,own):continue # another token elsewhere is not a competing sense
     own_terms=set(english(own.gloss));target_terms=set(english(target.gloss));context=set(english(c['translation']))
     if source['kind']!='manual' or len(compatible)>1 or own_terms&target_terms or not own_terms&context or target_terms&context:continue
     kind='cross_lexical_inferred';weight=.15
    known.add(key);counts[target.tla,source['dialect'],'cross']+=1
    generated.append(dict(source,tla=target.tla,y=0,kind=kind,weight=weight,source_tla=own.tla))
  if mode in {'cross_random','cross_random_capped'}:
   bytarget={r['tla']:r for r in positive if any(form in collisions for form in lex[r['tla']].forms)}
   pool=sorted(train,key=lambda r:hashlib.sha256(r['cid'].encode()).hexdigest())
   for tla,ref in sorted(bytarget.items()):
    if focused and tla not in focus_tlas:continue
    e=lex[tla];n=0
    for source in pool:
     c=sentences[source['cid']];key=(tla,source['text_key'])
     if source['dialect']!=ref['dialect'] or key in known or source['text_key'] in held:continue
     if targets(c,e) or any(t['lemma'] in e.forms or t['form'] in e.forms for t in c['tokens']):continue
     known.add(key);generated.append(dict(source,tla=tla,y=0,kind='unrelated_no_target',weight=.2));n+=1
     if n>=10:break
  if mode=='cross_random_capped':
   caps=Counter((r['tla'],r['dialect']) for r in positive);used=Counter();bounded=[]
   for r in generated:
    key=(r['tla'],r['dialect']);limit=caps[key]
    if used[key]>=limit:continue
    used[key]+=1;bounded.append(r)
   generated=bounded
  assert not any(r['text_key'] in held for r in generated)
  return generated
 metrics=[];audit=[]
 for d in ['sahidic','bohairic']:
  subset=[r for r in data if r['dialect']==d]
  for f in range(5):
   val=[r for r in subset if r['kind'].startswith('manual') and fold(r)==f];held={r['text_key'] for r in val}
   train=[r for r in subset if fold(r)!=f and r['text_key'] not in held]
   if not val or len({r['y'] for r in train})<2:continue
   modes=['none','naive_all','cross','cross_random_capped','focus_naive_all','focus_cross','focus_cross_random_capped'] if comparison else ['none','naive_all','cross','cross_random','cross_random_capped']
   for mode in modes:
    added=augment(train,mode,held);tr=train+added
    model=LogisticRegression(C=2,max_iter=300,solver='liblinear',random_state=42)
    model.fit(matrix(tr),[r['y'] for r in tr],sample_weight=[r['weight'] for r in tr]);pred=model.predict_proba(matrix(val))[:,1]
    valid=len({r['y'] for r in val})==2
    ii=[i for i,r in enumerate(val) if r['tla'] in focus_tlas];fv=len({val[i]['y'] for i in ii})==2
    metrics.append(dict(dialect=d,fold=f,mode=mode,train=len(train),validation=len(val),focus_validation=len(ii),focus_positive=sum(val[i]['y'] for i in ii),focus_auc=float(roc_auc_score([val[i]['y'] for i in ii],pred[ii])) if fv else None,generated=dict(Counter(r['kind'] for r in added)),auc=float(roc_auc_score([r['y'] for r in val],pred)) if valid else None,ap=float(average_precision_score([r['y'] for r in val],pred)) if valid else None))
    audit.extend(dict(r,fold=f,mode=mode) for r in added)
   print(d,f,'done',flush=True)
 write_json(out/'augmentation.json',dict(protocol='Work-held-out; exact held-out Coptic text purged across all sources; validation only original manual labels. All exact dictionary form collisions considered within dialect. Generated negatives are hypotheses, not new expert gold. Guarded same-POS lexical negatives require the same target token, manual source and disjoint lexical support; naive_all deliberately removes these compatibility guards.',inventory=dict(original_manual_positive=sum(r['kind']=='manual' for r in data),original_manual_negative=sum(r['kind']=='manual_forbidden_rule' for r in data),crum_weak_positive=sum(r['kind']=='crum_weak' for r in data),canonical_homograph_families=len(homographs),surface_collision_groups=len(collisions)),folds=metrics))
 if comparison:
  write_json(out/'focus_definition.json',dict(words=WORDS+['ϣⲁ'],tlas=sorted(focus_tlas),both_source_and_target_restricted=True))
  print('Scope comparison complete',flush=True);return
 write_json(cfg.derived/'sense_v3_negative_audit.json',audit)
 # Standalone experimental models; not silently substituted for the ranking models.
 for d in ['sahidic','bohairic']:
  subset=[r for r in data if r['dialect']==d]
  if len({r['y'] for r in subset})<2:continue
  model=LogisticRegression(C=2,max_iter=300,solver='liblinear',random_state=42).fit(matrix(subset),[r['y'] for r in subset],sample_weight=[r['weight'] for r in subset])
  p=cfg.work/'models'/d/'sense_v3.pkl';p.write_bytes(pickle.dumps((vec,model)))
 # Exhaustive TT token audit of review's list, with original local index counts for comparison.
 entries=[e for e in lex.values() if any(w==e.lemma for w in WORDS) or e.tla in {'C2787','C4008'}]
 inventory=[]
 with corpus.connect() as db:
  for e in entries:
   for d in ['sahidic','bohairic']:
    hits=[(c,targets(c,e)) for c in sentences.values() if c['dialect_id']==d];hits=[(c,ii) for c,ii in hits if ii]
    old=db.execute('SELECT count(*) FROM candidate WHERE tla=? AND dialect=?',(e.tla,d)).fetchone()[0]
    positive_keys={r['text_key'] for r in data if r['tla']==e.tla and r['dialect']==d and r['kind']=='manual'}
    inventory.append(dict(tla=e.tla,lemma=e.lemma,gloss=e.gloss,pos=e.pos,dialect=d,old_candidate_count=old,strict_TT_count=len(hits),manual_positive_texts=len(positive_keys),status='native category separation; same-category sense unresolved unless independent evidence' if hits else 'no eligible TT token',examples=[dict(id=c['id'],translation=c['translation'],source=c['source'],tokens=[c['tokens'][i] for i in ii]) for c,ii in hits[:3]],cdo='https://coptic-dictionary.org/entry/'+e.tla))
 write_json(out/'review_words.json',inventory)
 # Meaning of feature importance: gain is not a coefficient or causal effect.
 import lightgbm as lgb
 importance=[]
 for path in sorted((cfg.work/'models').glob('*/*.txt')):
  try:
   model=lgb.Booster(model_file=str(path));g=model.feature_importance('gain');s=model.feature_importance('split');total=max(1.,sum(g))
   importance.append(dict(model=str(path.relative_to(cfg.work)),features=[dict(feature=n,gain=float(a),gain_share=float(a/total),split=int(b)) for n,a,b in sorted(zip(model.feature_name(),g,s),key=lambda v:-v[1])]))
  except Exception:continue
 write_json(out/'importance.json',importance)
 print('Sense audit and augmentation evaluation complete',flush=True)

if __name__=='__main__':run()
