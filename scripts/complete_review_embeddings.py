"""Cache genuine BGE vectors for lexical contrast retrieval in a separate database."""
import sys,json,re,hashlib,sqlite3
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from coptic_examples.config import Config
from coptic_examples.lexicon import load
from coptic_examples.source_pool import SupplementedCorpus
from coptic_examples.ablation import CachedVectors
from coptic_examples.embeddings import Embeddings
cfg=Config(retrieval_vectors=True);lex=load(cfg.lexicon)
selected=set(re.findall(r'/entry/(C[0-9]+)',(cfg.work/'homographs_top20.md').read_text()))
selected-={'C615','C616','C617','C618','C614','C2788'}
forms={f for t in selected for f in lex[t].forms if ' ' not in f}
index=sqlite3.connect(cfg.work/'cache/native_tokens_extended.sqlite3')
ids=set()
for form in forms:
 ids.update(r[0] for r in index.execute('SELECT sentence_id FROM token WHERE form=? OR lemma=?',(form,form)))
vectors=CachedVectors(cfg);missing=[]
for c in SupplementedCorpus(cfg).all():
 if c['id'] in ids and len(c['sent'].split())<=30 and c['translation'].strip():
  if vectors.get(c['translation'],'sentence') is None:missing.append(c['translation'])
missing=sorted(set(missing)); print('Missing translations',len(missing),flush=True)
model=Embeddings(cfg.work/'cache/retrieval_embeddings.sqlite3',device='mps')
model.encode(sorted({lex[t].gloss for t in selected}),'definition')
for start in range(0,len(missing),256):
 model.encode(missing[start:start+256]);print('Encoded',min(start+256,len(missing)), '/',len(missing),flush=True)
