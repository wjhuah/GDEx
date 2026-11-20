pass

import collections
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.config import Config
from coptic_examples.lexicon import load
from coptic_examples.text import lemma_key
from coptic_examples.citation_recovery import add_cdo_only_entries

cfg = Config()
dest = cfg.reports / 'local_2218a127/local_7de64c76'
online = (dest / "index.json").read_bytes()
snapshot = (cfg.work / "cache/cdo_index.json").read_bytes()
assert hashlib.sha256(online).digest() == hashlib.sha256(snapshot).digest()
lex = load(cfg.lexicon)
data = json.loads(online)
online_only = add_cdo_only_entries(lex, data)
missing = []
for r in data:
    if r["id"] not in lex:
        continue
    e = lex[r["id"]]
    for f in r.get("forms", []):
        form = lemma_key(f.get("orth", ""))
        dialect = f.get("dialect", "")
        if form and (
            form not in e.forms
            or (dialect and dialect not in (e.form_dialects or {}).get(form, []))
        ):
            missing.append(dict(tla=r["id"], form=form, dialect=dialect))
assert not missing
oldrows = {}
cohorts = collections.Counter()
for line in (cfg.reports / 'local_2218a127/local_f2365691/local_b415e7f6').open():
    r = json.loads(line)
    if r["old_status"] == "training_pair_available" or r["accepted"]:
        continue
    if r["status"] in ['', '']:
        oldrows[r["id"]] = r
for line in (cfg.derived / "complete_citations/citations.jsonl").open():
    r = json.loads(line)
    if r["id"] in oldrows:
        cohorts[
            (
                oldrows[r["id"]]["status"],
                '' if r["local_links"] else '',
            )
        ] += 1
e = lex["C5257"]
forms = {d: {f for f in e.forms if d in e.form_dialects[f]} for d in ["S", "B"]}
stats = collections.Counter()
db = sqlite3.connect(cfg.index)
placeholders = ",".join("?" for f in e.forms)
ids = [
    r[0]
    for r in db.execute(
        f"SELECT DISTINCT sentence_id FROM token WHERE lemma IN ({placeholders})",
        e.forms,
    )
]
for sid in ids:
    r = json.loads(
        db.execute("SELECT payload FROM sentence WHERE id=?", (sid,)).fetchone()[0]
    )
    code = {"sahidic": "S", "bohairic": "B"}.get(r["dialect_id"])
    if not code:
        continue
    for t in r["tokens"]:
        f = lemma_key(t["form"])
        if f in forms[code] and t["pos"] == "VERB":
            stats[(code, f)] += 1
summary = dict(
    sha256=hashlib.sha256(online).hexdigest(),
    cdo_entries=len(data),
    missing_normalized_forms=missing,
    online_only_entries_added=online_only,
    unresolved_cohort=[
        dict(status=s, corpus=c, occurrences=n) for (s, c), n in cohorts.items()
    ],
    C5257=[
        dict(dialect=d, form=f, verb_tagged_surface_tokens=stats[(d, f)])
        for d in ["S", "B"]
        for f in sorted(forms[d])
    ],
)
(dest / '').write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
)
lines = [
    '',
    "",
    '' + summary["sha256"] + "。",
    '',
    "",
    '',
    "|---|---|---:|",
]
lines += [f"| {s} | {c} | {n:,} |" for (s, c), n in cohorts.items()]
lines += [
    "",
    '',
    "",
    "## C5257",
    "",
    '',
    "|---|---|---:|",
]
lines += [
    f"| {d} | {f} | {stats[(d,f)]:,} |" for d in ["S", "B"] for f in sorted(forms[d])
]
lines += [
    "",
    '',
    '',
]
(dest / '').write_text("\n".join(lines) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
