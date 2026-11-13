"""Index TT archives omitted by the original unpacked-only corpus, additively."""

import sys, json, zipfile, hashlib, sqlite3
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.config import Config
from coptic_examples.corpus import parse_tt

cfg = Config()
extract = cfg.work / "cache/archived_corpora"
extract.mkdir(parents=True, exist_ok=True)
out = cfg.reports / "本地复盘/线上例句覆盖"
out.mkdir(parents=True, exist_ok=True)
path = cfg.work / "cache/supplemental_corpus.sqlite3"
temp = path.with_suffix(".building.sqlite3")
temp.unlink(missing_ok=True)
db = sqlite3.connect(temp)
db.executescript(
    "CREATE TABLE sentence(id TEXT PRIMARY KEY,work TEXT,chapter TEXT,verse TEXT,source TEXT,dialect TEXT,payload TEXT); CREATE TABLE token(sentence_id TEXT,lemma TEXT,pos TEXT); CREATE INDEX token_lemma ON token(lemma); CREATE INDEX sentence_location ON sentence(work,chapter,verse); CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);"
)
manifest = []
counts = Counter()
digest = hashlib.sha256()
for archive in sorted(cfg.corpus.rglob("*TT*.zip")):
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not name.lower().endswith(".tt") or name.startswith("__MACOSX"):
                continue
            relative = Path(archive.parent.name) / name
            if (archive.parent / name).exists():
                continue
            target = (extract / relative).resolve()
            assert extract.resolve() in target.parents
            data = z.read(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            sha = hashlib.sha256(data).hexdigest()
            digest.update((str(relative) + sha).encode())
            manifest.append(
                dict(
                    archive=str(archive), member=name, source=str(relative), sha256=sha
                )
            )
            for c in parse_tt(target, extract):
                c["id"] = "archive:" + c["id"]
                c["supplemental_source"] = dict(
                    archive=str(archive), member=name, sha256=sha
                )
                db.execute(
                    "INSERT INTO sentence VALUES(?,?,?,?,?,?,?)",
                    (
                        c["id"],
                        c["work"],
                        c["chapter"],
                        c["verse"],
                        c["source"],
                        c["dialect_id"],
                        json.dumps(c, ensure_ascii=False),
                    ),
                )
                db.executemany(
                    "INSERT INTO token VALUES(?,?,?)",
                    [
                        (c["id"], t["lemma"], t["pos"])
                        for t in c["tokens"]
                        if t["lemma"]
                    ],
                )
                counts["sentences"] += 1
                counts["tokens"] += len(c["tokens"])
                counts[c["dialect_id"]] += 1
            if len(manifest) % 100 == 0:
                db.commit()
                print("files", len(manifest), dict(counts), flush=True)
db.execute("INSERT INTO metadata VALUES(?,?)", ("fingerprint", digest.hexdigest()))
db.commit()
assert db.execute("pragma integrity_check").fetchone()[0] == "ok"
db.close()
temp.replace(path)
(out / "压缩语料清单.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
)
(out / "补充语料统计.json").write_text(
    json.dumps(
        dict(files=len(manifest), counts=counts, fingerprint=digest.hexdigest()),
        ensure_ascii=False,
        indent=2,
    )
    + "\n"
)
print(dict(files=len(manifest), counts=counts), flush=True)
