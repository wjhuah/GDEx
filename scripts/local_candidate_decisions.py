"""Produce local admission decisions for the November citation audit."""
import csv,json,re,sys
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from coptic_examples.config import Config
from coptic_examples.crum_audit import lexical_index
from coptic_examples.corpus import Corpus
from coptic_examples.text import english,lemma_key
from coptic_examples.integrity import alignment_issue
from coptic_examples.sense import targets
cfg=Config(); out=cfg.reports/'本地复盘'
lex,byform,pages=lexical_index(cfg.lexicon)
base={r['id']:r for l in (out/'逐条核对.jsonl').open() if (r:=json.loads(l))['status']=='新增可复核候选'}
bans={(r['TLA'],r['sentence_id']) for r in csv.DictReader((cfg.derived/'forbidden_matches.tsv').open(),delimiter='\t')}
allban={r['TLA'] for r in csv.DictReader(cfg.manual.open(),delimiter='\t') if r['forbidden_urns'].strip().upper()=='ALL'}
counts=Counter();outputs=[]
with Corpus(cfg).connect() as db:
 for line in (cfg.derived/'complete_citations/citations.jsonl').open():
    r=json.loads(line)
    if r['id'] not in base:continue
    b=base[r['id']]; pairs=[]
    for ev in b['evidence']:
        sid=ev['sentence'];tla=ev['tla'];e=lex[tla]
        c=json.loads(db.execute('select payload from sentence where id=?',(sid,)).fetchone()[0])
        ts=[t for t in c['tokens'] if t['id'] in ev['tokens']]
        paragraph=r['paragraph'];before=paragraph.partition(r['raw'])[0]
        checks=[];corrected=False
        if r['id']=='5a5dfbebb754926a33f10db1' and sid=='sahidic.ot/20_Proverbs_04.tt#18':
            tla='C4958';e=lex[tla];ts=[t for t in c['tokens'] if t['id'] in ['u331','u343']];assert len(ts)==2 and all(t['xpos']=='V' for t in ts)
            corrected=True
        else:
            peers={t for tok in ts for form in [lemma_key(tok['form']),tok['lemma']] for t in byform.get(form,[]) if t!=tla and lex[t].pos==e.pos and r['page'] in pages.get(t,set()) and targets(c,lex[t])}
            if peers:checks.append('同页同词性竞争词条 '+','.join(sorted(peers)))
            if not (set(e.forms)&{lemma_key(r['heading_form'])}):checks.append('引文标题与词条词形尚缺直接对应')
            vocab=set(english(c['translation']));support=[w for w in ev['common_gloss'] if len(w)>2 and any(v in {w,w+'s',w+'es',w+'ed',w+'ing',w.rstrip('e')+'ing'} for v in vocab)]
            if not support:checks.append('译文缺少相应释义的直接佐证')
            if len(ts)!=1:checks.append('多个目标位置需要分别辨义')
            if not any('⟨gloss: '+g.strip()+'⟩' in before for g in r['heading_gloss'].split('|')):checks.append('引文所在段前文缺少对应释义标记')
            if not r['tla_candidates'] or tla not in r['tla_candidates']:checks.append('词条归属仅由旧版匹配提出')
        if alignment_issue(c):checks.append('语料原文与译文跨度异常')
        if tla in allban or (tla,sid) in bans:checks.append('人工禁用记录')
        decision='排除本次配对' if checks else ('纠正后收录' if corrected else '收录')
        pairs.append(dict(result=decision,tla=tla,proposed_tla=ev['tla'],gloss=e.gloss,sentence=sid,dialect=ev['dialect'],tokens=ts,translation=c['translation'],text=c['sent'],reason='；'.join(checks) or ('引文指向动词发光，句内两个 V 对应 shine。' if corrected else '标题、词条、同页竞争词、TT 位置和译文通过核查。')))
    status='纠正后收录' if any(p['result']=='纠正后收录' for p in pairs) else '收录' if any(p['result']=='收录' for p in pairs) else '排除本次配对'
    counts[status]+=1
    outputs.append(dict(id=r['id'],result=status,ref=r['normalized_reference'],file=r['file'],page=r['page'],raw=r['raw'],paragraph=r['paragraph'],pairs=pairs))
assert len(outputs)==len(base)==1182
with (out/'1182处最终判定.jsonl').open('w') as f,(out/'1182处最终判定.tsv').open('w') as t:
 w=csv.writer(t,delimiter='\t');w.writerow(['引文ID','引用判定','经文','页','词条','词义','方言','TT位置','配对判定','理由','译文'])
 for r in outputs:
    f.write(json.dumps(r,ensure_ascii=False)+'\n')
    for p in r['pairs']:w.writerow([r['id'],r['result'],r['ref'],r['page'],p['tla'],p['gloss'],p['dialect'],','.join(x['id'] for x in p['tokens']),p['result'],p['reason'],p['translation']])
print(counts)
for r in outputs:
 if r['id']=='8860df021548bc2b29e0e1d5':print(r['result'],r['pairs'][0]['reason'])
