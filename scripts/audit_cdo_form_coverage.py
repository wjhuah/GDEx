"""Compare live CDO forms with the local lexicon and trace unresolved citations."""

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
dest = cfg.reports / "本地复盘/CDO词形核验"
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
for line in (cfg.reports / "本地复盘/批量修复第三轮/逐条修复.jsonl").open():
    r = json.loads(line)
    if r["old_status"] == "training_pair_available" or r["accepted"]:
        continue
    if r["status"] in ["标题扩展后仍待定位词条", "已有词条线索但目标词形不符"]:
        oldrows[r["id"]] = r
for line in (cfg.derived / "complete_citations/citations.jsonl").open():
    r = json.loads(line)
    if r["id"] in oldrows:
        cohorts[
            (
                oldrows[r["id"]]["status"],
                "有本地方言经文" if r["local_links"] else "缺对应经文",
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
(dest / "核验结果.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
)
lines = [
    "# CDO 词形与 Crum 漏配核验",
    "",
    "在线 index.json 与本地快照 SHA-256 完全一致：" + summary["sha256"] + "。",
    "旧载入流程跳过 283 个在线新增 TLA。本轮将这些条目补入恢复专用词典后，再逐项核对全部非空词形与方言，遗漏数为 0。连字符和 ⸗ 归入规范化前的原始词形证据。",
    "",
    "| 上轮问题 | 本地语料覆盖 | 引用处数 |",
    "|---|---|---:|",
]
lines += [f"| {s} | {c} | {n:,} |" for (s, c), n in cohorts.items()]
lines += [
    "",
    "词条未定位发生在 Crum 标题到 TLA 的映射阶段。即使本地经文已存在，派生词、词性续段、缩写和同形词仍可使映射失败。表中语料存在与否单独核验。",
    "",
    "## C5257",
    "",
    "| 方言 | 规范词形 | 原生 VERB 标签的实际词形 token 数 |",
    "|---|---|---:|",
]
lines += [
    f"| {d} | {f} | {stats[(d,f)]:,} |" for d in ["S", "B"] for f in sorted(forms[d])
]
lines += [
    "",
    "统计从 lemma 命中集合读取完整 TT token；计数仍受已有 lemma 与词性标注质量影响。S 与 B 分开，A/F/L 词形独立保留。",
    "Crum 的名词 desire/love 续段应另归 C5258；同一字形的动词 C5257 与名词 C5258 分开定位。部分经文有括注明确替换词或原译节号错位，补入更多词形无法单独解决这些情况。",
]
(dest / "中文核验说明.md").write_text("\n".join(lines) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
