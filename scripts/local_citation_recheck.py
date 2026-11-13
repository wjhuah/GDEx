"""Audit source spans and recover independently supported candidate alignments."""
import csv, hashlib, json, sys
from pathlib import Path
from collections import Counter, defaultdict
from functools import lru_cache
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from coptic_examples.config import Config
from coptic_examples.crum_audit import lexical_index
from coptic_examples.text import english, lemma_key
from coptic_examples.sense import targets
from coptic_examples.corpus import Corpus

cfg=Config()
out=cfg.reports/'本地复盘';out.mkdir(exist_ok=True)
lex,forms,pages=lexical_index(cfg.lexicon)
legacy=defaultdict(set)
for r in csv.DictReader((cfg.source/'crum3/training_data_v15.tsv').open(),delimiter='\t'):
    for ref in r['bible_refs'].split(';'):
        legacy[r['file'],ref.strip()].add(r['TLA'])
def terms(s):
    return set(english(s))-set('to of in on by an a the be as with one make it or and'.split())
source={p.name:p.read_bytes() for p in (cfg.source/'Crum_plain_text').glob('*.txt')}
hashes={k:hashlib.sha256(v).hexdigest() for k,v in source.items()}
records=[json.loads(l) for l in (cfg.derived/'complete_citations/citations.jsonl').open()]
records.sort(key=lambda r:(r['file'],r['char_start']))
counts=Counter();sizes=Counter((r['file'],r.get('section_id')) for r in records);order=Counter()
seen=set();offsets=set()
with Corpus(cfg).connect() as db,(out/'逐条核对.jsonl').open('w') as f:
    @lru_cache(2000)
    def sentence(sid):
        return json.loads(db.execute('select payload from sentence where id=?',(sid,)).fetchone()[0])
    for r in records:
        file=r['file'];start=r['byte_start'];raw=r['raw'].encode()
        assert hashes[file]==r['file_sha256'] and source[file][start:start+len(raw)]==raw
        assert r['id'] not in seen and (file,start) not in offsets
        seen.add(r['id']);offsets.add((file,start))
        group=(file,r.get('section_id'));order[group]+=1
        section_size=sizes[group];position=order[group]
        prior=set(r['tla_candidates']);old=legacy[file,r['normalized_reference']]
        pool=(prior&old) if prior else old
        evidence=[]
        for tla in sorted(pool):
            e=lex.get(tla)
            if not e or r['page'] not in pages.get(tla,set()):continue
            overlap=terms(e.gloss)&terms(r['heading_gloss'])
            if not overlap:continue
            for link in r['local_links']:
                c=sentence(link['sentence_id'])
                ids=[c['tokens'][i]['id'] for i in targets(c,e)]
                if ids:evidence.append(dict(tla=tla,sentence=c['id'],dialect=c['dialect_id'],tokens=ids,common_gloss=sorted(overlap)))
        if r['training_pairs']:status='已有训练配对'
        elif len({v['tla'] for v in evidence})==1:status='新增可复核候选'
        else:status='待补证'
        counts[status]+=1
        row=dict(id=r['id'],file=file,offset=start,page=r['page'],heading=r['heading_form'],gloss=r['heading_gloss'],reference=r['normalized_reference'],raw=r['raw'],section=r.get('section_id'),position=position,total=section_size,selection=1,order_grade=3 if section_size==1 else 3-1.5*(position-1)/(section_size-1),status=status,original_status=r['training_status'],link_status=r['link_status'],original_tlas=sorted(prior),legacy_tlas=sorted(old),local_links=r['local_links'],evidence=evidence,training_pairs=r['training_pairs'])
        f.write(json.dumps(row,ensure_ascii=False)+'\n')
assert len(seen)==sum(b.lower().count(b'{bible:') for b in source.values())==41808
print(counts)
