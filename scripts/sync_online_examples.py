"""Fetch every indexed CDO entry and atomically publish a separate source database."""

import sys, json, time, hashlib, sqlite3, re, threading
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen

root = Path(__file__).resolve().parents[1]
cache = root / "cache/cdo_online"
cache.mkdir(parents=True, exist_ok=True)
report = root / "reports/本地复盘/线上例句覆盖"
report.mkdir(parents=True, exist_ok=True)
refresh = "--refresh" in sys.argv
BASE = "https://coptic-dictionary.org"
lock = threading.Lock()
next_request = 0


def get(url):
    global next_request
    with lock:
        wait = max(0, next_request - time.monotonic())
        next_request = max(time.monotonic(), next_request) + 1 / 30
    if wait:
        time.sleep(wait)
    with urlopen(
        Request(url, headers={"User-Agent": "GDEx-local-example-coverage/1.0"}),
        timeout=30,
    ) as response:
        return response.read()


index = get(BASE + "/index.json")
(cache / "index.json").write_bytes(index)
records = json.loads(index)
ids = [r["id"] for r in records]
assert len(ids) == len(set(ids))
assert all(re.fullmatch("C[0-9]+", t) for t in ids)


def fetch(tla):
    path = cache / (tla + ".json")
    last = None
    for attempt in range(4):
        try:
            data = (
                path.read_bytes()
                if path.exists() and not refresh
                else get(BASE + "/details/" + tla + ".json")
            )
            r = json.loads(data)
            assert r.get("id") == tla
            assert isinstance(r.get("examples", []), list)
            if not path.exists() or refresh:
                tmp = path.with_suffix(".tmp")
                tmp.write_bytes(data)
                tmp.replace(path)
            return dict(
                tla=tla,
                ok=True,
                examples=len(r.get("examples", [])),
                sha256=hashlib.sha256(data).hexdigest(),
            )
        except Exception as e:
            last = str(e)
            if attempt < 3:
                time.sleep(attempt + 1)
    return dict(tla=tla, ok=False, error=last)


manifest = []
with ThreadPoolExecutor(max_workers=6) as pool:
    for future in as_completed([pool.submit(fetch, tla) for tla in ids]):
        manifest.append(future.result())
        if len(manifest) % 250 == 0:
            print(
                "entries",
                len(manifest),
                "/",
                len(ids),
                "examples",
                sum(r.get("examples", 0) for r in manifest),
                "failed",
                sum(not r["ok"] for r in manifest),
                flush=True,
            )
(report / "抓取清单.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
)
failures = [r for r in manifest if not r["ok"]]
assert not failures, failures
path = root / "cache/online_examples.sqlite3"
tmp = path.with_suffix(".building.sqlite3")
tmp.unlink(missing_ok=True)
db = sqlite3.connect(tmp)
db.executescript(
    "CREATE TABLE entry(tla TEXT PRIMARY KEY,lemma TEXT,pos TEXT,gloss TEXT,payload TEXT); CREATE TABLE example(id TEXT PRIMARY KEY,tla TEXT REFERENCES entry(tla),dialect TEXT,ordinal INTEGER,marked_text TEXT,text TEXT,translation TEXT,title TEXT,urn TEXT,payload TEXT); CREATE INDEX example_entry ON example(tla,dialect,ordinal); CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);"
)
counts = {}
total = 0
entries_with_examples = 0
for entry in records:
    tla = entry["id"]
    detail = json.loads((cache / (tla + ".json")).read_text())
    gloss = " | ".join(
        g
        for sense in entry.get("senses", [])
        for g in sense.get("translations", {}).get("en", [])
    )
    db.execute(
        "INSERT INTO entry VALUES(?,?,?,?,?)",
        (
            tla,
            entry.get("lemma", ""),
            entry.get("pos", ""),
            gloss,
            json.dumps(entry, ensure_ascii=False),
        ),
    )
    examples = detail.get("examples", [])
    entries_with_examples += bool(examples)
    for ordinal, e in enumerate(examples, 1):
        marked = e.get("coptic", "")
        text = " ".join(marked.replace("**", "").replace("*", "").split())
        rawdialect = str(e.get("dialect", ""))
        dialect = {
            "s": "sahidic",
            "b": "bohairic",
            "sahidic coptic": "sahidic",
            "bohairic coptic": "bohairic",
        }.get(rawdialect.casefold(), rawdialect.casefold())
        eid = hashlib.sha256(f"{tla}|{dialect}|{ordinal}".encode()).hexdigest()[:24]
        db.execute(
            "INSERT INTO example VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                eid,
                tla,
                dialect,
                ordinal,
                marked,
                text,
                e.get("english", ""),
                e.get("source", ""),
                e.get("urn", ""),
                json.dumps(e, ensure_ascii=False),
            ),
        )
        counts[dialect] = counts.get(dialect, 0) + 1
        total += 1
summary = dict(
    index_entries=len(ids),
    fetched_entries=len(manifest),
    failed_entries=len(failures),
    entries_with_examples=entries_with_examples,
    examples=total,
    by_dialect=counts,
    index_sha256=hashlib.sha256(index).hexdigest(),
    completed_at=datetime.now(timezone.utc).isoformat(),
    source=BASE,
    license="CC BY-SA 4.0",
    training_labels_added=0,
)
db.execute(
    "INSERT INTO metadata VALUES(?,?)",
    ("snapshot", json.dumps(summary, ensure_ascii=False)),
)
db.commit()
assert db.execute("SELECT COUNT(*) FROM entry").fetchone()[0] == len(ids)
assert db.execute("SELECT COUNT(*) FROM example").fetchone()[0] == sum(
    r["examples"] for r in manifest
)
assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
db.close()
tmp.replace(path)
(report / "覆盖统计.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
)
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
