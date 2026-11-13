"""Keep the pinned official TT variants alongside locally modified originals."""
import sys,json,hashlib,sqlite3
from pathlib import Path
from urllib.request import urlopen
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from coptic_examples.config import Config
from coptic_examples.corpus import parse_tt
cfg=Config();folder=cfg.reports/'本地复盘/线上例句覆盖';manifest=json.loads((folder/'官方TT文件清单.json').read_text());commit=manifest['sha'];root=cfg.work/'cache/upstream_corpora';root.mkdir(exist_ok=True)
files=[]
for r in manifest['files']:
 if not r['path'].endswith('.tt'):continue
 local=cfg.corpus/r['path'];data=local.read_bytes() if local.exists() else b'';gitsha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
 if gitsha==r['sha']:continue
 url=f'https://raw.githubusercontent.com/CopticScriptorium/corpora/{commit}/{r["path"]}'
 with urlopen(url,timeout=40) as response:data=response.read()
 assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==r['sha']
 path=root/r['path'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);files.append(dict(r,url=url))
path=cfg.work/'cache/upstream_corpus.sqlite3';tmp=path.with_suffix('.building.sqlite3');tmp.unlink(missing_ok=True);db=sqlite3.connect(tmp);db.executescript('CREATE TABLE sentence(id TEXT PRIMARY KEY,work TEXT,chapter TEXT,verse TEXT,source TEXT,dialect TEXT,payload TEXT); CREATE TABLE token(sentence_id TEXT,lemma TEXT,pos TEXT); CREATE INDEX token_lemma ON token(lemma); CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);');count=0
for r in files:
 for c in parse_tt(root/r['path'],root):
  c['id']='upstream:'+c['id'];c['upstream_source']=dict(url=r['url'],commit=commit,blob_sha=r['sha']);db.execute('INSERT INTO sentence VALUES(?,?,?,?,?,?,?)',(c['id'],c['work'],c['chapter'],c['verse'],c['source'],c['dialect_id'],json.dumps(c,ensure_ascii=False)));db.executemany('INSERT INTO token VALUES(?,?,?)',[(c['id'],t['lemma'],t['pos']) for t in c['tokens'] if t['lemma']]);count+=1
fingerprint=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest();db.execute('INSERT INTO metadata VALUES(?,?)',('fingerprint',fingerprint));db.commit();assert db.execute('pragma integrity_check').fetchone()[0]=='ok';db.close();tmp.replace(path)
(folder/'官方版本补充统计.json').write_text(json.dumps(dict(commit=commit,files=files,spans=count),ensure_ascii=False,indent=2)+'\n');print('official variant files',len(files),'spans',count)
