"""Collect an offset-verified Crum witness and aligned TT text for every outstanding citation."""
import json, re, sqlite3, hashlib
from pathlib import Path
from collections import Counter
from functools import lru_cache
root=Path(__file__).resolve().parents[1]
dest=root/'reports/本地复盘'
def records(path):
    with path.open() as f:
        for line in f:yield json.loads(line)
previous={r['id'] for r in records(dest/'AI语文学判读_首批.jsonl')}
cohort={r['id']:r for r in records(dest/'30033处判定.jsonl') if r['decision']=='本轮不收录' and r['id'] not in previous}
assert len(cohort)==29998
@lru_cache(16)
def source(name):
    raw=(root.parent/'Crum_plain_text'/name).read_bytes()
    return raw.decode(),hashlib.sha256(raw).hexdigest()
db=sqlite3.connect(root/'cache/corpus.sqlite3')
@lru_cache(512)
def witness(sid):
    return json.loads(db.execute('select payload from sentence where id=?',(sid,)).fetchone()[0])
seen=set();stats=Counter()
path=dest/'29998处原文与见证.jsonl'
with path.with_suffix('.tmp').open('w') as f:
 for r in records(root/'data/derived/complete_citations/citations.jsonl'):
    oid=r['id']
    if oid not in cohort:continue
    assert oid not in seen;seen.add(oid)
    text,digest=source(r['file']);a,b=r['char_start'],r['char_end']
    assert digest==r['file_sha256'] and text[a:b]==r['raw']
    left=text.rfind('\n',0,a)+1;right=text.find('\n',b)
    following=text[b:right if right>=0 else len(text)]
    tail=re.split(r'⟦;⟧|⟦[^⟧]*⟧\{(?:bible|reference):',following,1)[0]
    versions=[]
    for link in r['local_links']:
        c=witness(link['sentence_id'])
        versions.append(dict(id=c['id'],dialect=c['dialect_id'],text=c['sent'],translation=c['translation'],tokens=[(t['id'],t['form'],t['lemma'],t['xpos']) for t in c['tokens']]))
    row=dict(id=oid,file=r['file'],page=r['page'],offset=a,reference=r['normalized_reference'],raw=r['raw'],heading=r['heading_form'],paragraph=r['paragraph'],following_annotation=tail,versions=versions,previous_reason=cohort[oid]['reason'],direct_reading=None)
    if oid=='cbe1e0779fef6a358d25923c':
        assert 'not do again' in r['paragraph'] and '⟪S⟫' in tail
        assert all('ⲛⲅⲧⲙⲕⲟⲧⲕ' in v['text'] and any(t[0]=='u804' for t in v['tokens']) for v in versions)
        row['direct_reading']=dict(tla='C1416',dialect='sahidic',token='u804',decision='收录',reason='耶稣命污灵出去，并禁止再次进入。否定式 ⲛⲅⲧⲙⲕⲟⲧⲕ 后接进入动作，与 Crum 的 not do again 相符。三个语料版本保留同一处引用。')
    stats['有本地经文' if versions else '缺对应本地经文']+=1
    if row['direct_reading']:stats['新增直接判读']+=1
    f.write(json.dumps(row,ensure_ascii=False)+'\n')
assert seen==set(cohort)
path.with_suffix('.tmp').replace(path);print(stats)
