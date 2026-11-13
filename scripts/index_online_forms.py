"""Index source-only entries and aliases without changing the trained lexicon."""

import sys, json, sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.config import Config
from coptic_examples.text import lemma_key

cfg = Config()
db = sqlite3.connect(cfg.work / "cache/online_examples.sqlite3")
db.executescript(
    "DROP TABLE IF EXISTS online_form; CREATE TABLE online_form(form TEXT,tla TEXT,PRIMARY KEY(form,tla));"
)
for tla, payload in db.execute("SELECT tla,payload FROM entry").fetchall():
    r = json.loads(payload)
    forms = {lemma_key(r.get("lemma", "")).casefold()} | {
        lemma_key(f.get("orth", "")).casefold() for f in r.get("forms", [])
    }
    db.executemany(
        "INSERT OR IGNORE INTO online_form VALUES(?,?)", [(f, tla) for f in forms if f]
    )
db.commit()
print("source form index", db.execute("SELECT COUNT(*) FROM online_form").fetchone()[0])
db.close()
