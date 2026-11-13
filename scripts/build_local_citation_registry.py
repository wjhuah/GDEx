"""Store every Crum occurrence with its source and current alignment status."""

import json
import sys
import sqlite3
from pathlib import Path

root = Path(__file__).resolve().parents[1]
folder = (
    root / "reports/本地复盘" / (sys.argv[1] if len(sys.argv) > 1 else "批量修复第四轮")
)
target = folder / "完整引用库.sqlite3"
temporary = target.with_suffix(".building.sqlite3")
if temporary.exists():
    temporary.unlink()
decisions = {}
scoped = (folder / "异写表来源.json").exists()
for line in (folder / "逐条修复.jsonl").open():
    r = json.loads(line)
    decisions[r["id"]] = r
db = sqlite3.connect(temporary)
db.execute("PRAGMA foreign_keys=ON")
with db:
    db.executescript("""
    CREATE TABLE citation(
        id TEXT PRIMARY KEY, file TEXT NOT NULL, char_start INTEGER NOT NULL,
        char_end INTEGER NOT NULL, file_sha256 TEXT NOT NULL, page TEXT,
        raw TEXT NOT NULL, reference TEXT, heading TEXT, gloss TEXT,
        paragraph TEXT, dialects TEXT, status TEXT NOT NULL,
        UNIQUE(file,char_start));
    CREATE TABLE alignment(
        citation_id TEXT REFERENCES citation(id), tla TEXT, dialect TEXT,
        sentence_id TEXT, origin TEXT, tokens TEXT, expert_verified INTEGER DEFAULT 0,
        PRIMARY KEY(citation_id,tla,dialect,sentence_id,origin));
    CREATE INDEX citation_status ON citation(status);
    CREATE INDEX alignment_tla ON alignment(tla,dialect);
    """)
    for line in (root / "data/derived/complete_citations/citations.jsonl").open():
        r = json.loads(line)
        d = decisions[r["id"]]
        record_only = scoped and "S" not in d["original_dialect_codes"]
        existing = [
            p
            for p in r["training_pairs"]
            if not scoped or (not record_only and p["dialect"] == "sahidic")
        ]
        status = (
            "原记录未明确含S，仅记录"
            if record_only
            else (
                "原有训练配对"
                if existing
                else "自动恢复配对" if d["accepted"] else d["status"]
            )
        )
        db.execute(
            "INSERT INTO citation VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                r["id"],
                r["file"],
                r["char_start"],
                r["char_end"],
                r["file_sha256"],
                r["page"],
                r["raw"],
                r["normalized_reference"],
                r["heading_form"],
                r["heading_gloss"],
                r["paragraph"],
                json.dumps(r["dialect_codes"]),
                status,
            ),
        )
        for p in existing:
            db.execute(
                "INSERT OR IGNORE INTO alignment VALUES(?,?,?,?,?,?,?)",
                (
                    r["id"],
                    p["tla"],
                    p["dialect"],
                    p["sentence_id"],
                    "已有训练",
                    "[]",
                    0,
                ),
            )
        for p in d["accepted"]:
            db.execute(
                "INSERT OR IGNORE INTO alignment VALUES(?,?,?,?,?,?,?)",
                (
                    r["id"],
                    p["tla"],
                    p["dialect"],
                    p["sentence_id"],
                    "自动恢复",
                    json.dumps(p["spans"]),
                    0,
                ),
            )
    assert db.execute("SELECT COUNT(*) FROM citation").fetchone()[0] == 41808
    assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert not db.execute("PRAGMA foreign_key_check").fetchall()
    stats = dict(db.execute("SELECT status,COUNT(*) FROM citation GROUP BY status"))
db.close()
temporary.replace(target)
(folder / "入库统计.json").write_text(
    json.dumps(stats, ensure_ascii=False, indent=2) + "\n"
)
print(json.dumps(stats, ensure_ascii=False))
