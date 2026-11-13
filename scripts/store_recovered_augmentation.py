"""Persist training-fold-only negative contrasts with their positive source records."""

import sys, json, sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.config import Config, write_json
from coptic_examples.learning import augment
from coptic_examples.lexicon import load
from coptic_examples.citation_recovery import add_cdo_only_entries
from coptic_examples.inclusion import EntryFeatures

cfg = Config()
folder = cfg.reports / "本地复盘/新增数据模型比较"
db = sqlite3.connect(folder / "实验数据.sqlite3")
rows = [
    json.loads(v[0])
    for v in db.execute("select payload from example where dialect='sahidic'")
]
base = list(rows)
lex = load(cfg.lexicon)
add_cdo_only_entries(lex, json.loads((cfg.work / "cache/cdo_index.json").read_text()))
ef = EntryFeatures(cfg, lex, "sahidic")
memo = {}
db.executescript(
    "DROP TABLE IF EXISTS augmentation; CREATE TABLE augmentation(fold INTEGER,tla TEXT,text_key TEXT,label INTEGER,source_tla TEXT,sentence_id TEXT,reason TEXT,features TEXT,PRIMARY KEY(fold,tla,text_key));"
)
counts = []
experiments = json.loads((folder / "逐折结果.json").read_text())
for fold in range(5):
    val = [
        r for r in base if r["fold"] == fold and "crum_automatic" not in r["sources"]
    ]
    held = {r["text_key"] for r in val}
    families = {r["family"] for r in val}
    train = [r for r in rows if r["fold"] != fold and r["text_key"] not in held]
    extra = augment(train, 0.5, "mixed", ef, lex, memo)
    extra = [
        r for r in extra if r["text_key"] not in held and r["family"] not in families
    ]
    expected = next(
        r["augmented"]
        for r in experiments
        if r["dialect"] == "sahidic"
        and r["version"] == "recovered_augmented"
        and r["fold"] == fold
    )
    assert len(extra) == expected, (fold, len(extra), expected)
    db.executemany(
        "insert into augmentation values(?,?,?,?,?,?,?,?)",
        [
            (
                fold,
                r["tla"],
                r["text_key"],
                -1,
                r["source_tla"],
                r["candidate"]["id"],
                ",".join(r["sources"]),
                json.dumps(r["x"]),
            )
            for r in extra
        ],
    )
    db.commit()
    counts.append(
        dict(
            fold=fold,
            rows=len(extra),
            homograph=sum("homograph" in r["sources"] for r in extra),
            unrelated=sum("unrelated" in r["sources"] for r in extra),
        )
    )
    print(counts[-1], flush=True)
assert db.execute("pragma integrity_check").fetchone()[0] == "ok"
db.close()
write_json(folder / "增广数据库统计.json", counts)
