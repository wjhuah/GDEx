"""Audit every online example against both original and supplemental corpus spans."""

import sys, json, sqlite3, re
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.config import Config
from coptic_examples.source_pool import SupplementedCorpus
from coptic_examples.dataset import text_key
from coptic_examples.text import lemma_key, norm, pos

cfg = Config()
report = cfg.reports / "本地复盘/线上例句覆盖"
path = cfg.work / "cache/online_examples.sqlite3"
db = sqlite3.connect(path)
db.row_factory = sqlite3.Row
exact = defaultdict(list)
tolerant = defaultdict(list)
with SupplementedCorpus(cfg).connect() as corpus:
    for row in corpus.execute("SELECT id,dialect,payload FROM sentence"):
        c = json.loads(row["payload"])
        key = (c["dialect_id"], text_key(c))
        exact[key].append(c["id"])
        tolerant[
            (c["dialect_id"], lemma_key(c["sent"]).replace(" ", "").casefold())
        ].append(c["id"])
    db.executescript(
        "DROP TABLE IF EXISTS corpus_alignment; CREATE TABLE corpus_alignment(example_id TEXT,sentence_id TEXT,match_type TEXT,target_ids TEXT,pos_review INTEGER,PRIMARY KEY(example_id,sentence_id)); CREATE INDEX align_example ON corpus_alignment(example_id);"
    )
    counts = Counter()
    pending = []
    entries = {r["tla"]: pos(r["pos"]) for r in db.execute("SELECT tla,pos FROM entry")}
    examples = db.execute("SELECT * FROM example").fetchall()
    for e in examples:
        ids = exact.get((e["dialect"], text_key({"sent": e["text"]})), [])
        method = "exact_text"
        if not ids:
            ids = tolerant.get(
                (e["dialect"], lemma_key(e["text"]).replace(" ", "").casefold()), []
            )
            method = "punctuation_or_marks"
        if not ids:
            counts["无本地对应片段"] += 1
            pending.append(dict(e))
            continue
        counts[method] += 1
        for sid in ids:
            c = json.loads(
                corpus.execute(
                    "SELECT payload FROM sentence WHERE id=?", (sid,)
                ).fetchone()[0]
            )
            marked = norm(e["marked_text"])
            pieces = re.split(r"(\*\*.*?\*\*)", marked)
            ranges = []
            offset = 0
            for piece in pieces:
                bold = piece.startswith("**") and piece.endswith("**")
                value = piece[2:-2] if bold else piece
                compact = "".join(value.split()).casefold()
                if bold:
                    ranges.append((offset, offset + len(compact)))
                offset += len(compact)
            tokens = c["tokens"]
            token_compact = "".join(
                "".join(norm(t["form"]).split()) for t in tokens
            ).casefold()
            plain = "".join(e["text"].split()).casefold()
            target = []
            position = 0
            if token_compact == plain:
                for t in tokens:
                    length = len("".join(norm(t["form"]).split()).casefold())
                    end = position + length
                    if length and any(a <= position and end <= b for a, b in ranges):
                        target.append(t["id"])
                    position = end
            mismatch = any(
                t["pos"] != entries[e["tla"]] for t in tokens if t["id"] in target
            )
            db.execute(
                "INSERT INTO corpus_alignment VALUES(?,?,?,?,?)",
                (e["id"], sid, method, json.dumps(target), int(mismatch)),
            )
    db.commit()
    assert db.execute("pragma integrity_check").fetchone()[0] == "ok"
    summary = dict(
        online_examples=len(examples),
        alignment_counts=counts,
        with_corpus_match=len(examples) - counts["无本地对应片段"],
        all_preserved=True,
        training_labels_added=0,
        C3553=[
            dict(r)
            for r in db.execute(
                "SELECT e.title,e.ordinal,COUNT(a.sentence_id) AS corpus_matches FROM example e LEFT JOIN corpus_alignment a ON e.id=a.example_id WHERE e.tla='C3553' GROUP BY e.id ORDER BY e.ordinal"
            )
        ],
    )
    db.execute(
        "INSERT OR REPLACE INTO metadata VALUES(?,?)",
        ("alignment_summary", json.dumps(summary, ensure_ascii=False)),
    )
    db.commit()
(report / "语料对齐统计.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
)
(report / "尚无对应TT的线上例句.json").write_text(
    json.dumps(pending, ensure_ascii=False, indent=2) + "\n"
)
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
db.close()
