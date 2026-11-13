"""Export a concrete evidence record for every still-unpaired source occurrence."""

import collections
import csv
import json
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.config import Config
from coptic_examples.crum_audit import lexical_index
from coptic_examples.citation_recovery import (
    add_cdo_only_entries,
    explicit_bible_location,
)
from coptic_examples.dialects import Dialects

cfg = Config()
folder = cfg.reports / "本地复盘/未配对逐条诊断"
folder.mkdir(exist_ok=True)
lex, byform, pages = lexical_index(cfg.lexicon)
added = add_cdo_only_entries(
    lex, json.loads((cfg.work / "cache/cdo_index.json").read_text())
)
for t in added:
    for f in lex[t].forms:
        byform[f].append(t)
registry = Dialects()
pending = {}
for line in (cfg.reports / "本地复盘/批量修复第五轮/逐条修复.jsonl").open():
    r = json.loads(line)
    if r["old_status"] != "training_pair_available" and not r["accepted"]:
        pending[r["id"]] = r
assert len(pending) == 26767
db = sqlite3.connect(cfg.index)
sentences = {}
lookup = collections.defaultdict(list)
books = set()
chapters = set()
for sid, work, ch, v, dialect, payload in db.execute(
    "select id,work,chapter,verse,dialect,payload from sentence"
):
    key = explicit_bible_location(work, ch, v)
    if key:
        sentences[sid] = json.loads(payload)
        lookup[key].append(sid)
        books.add((key[0], dialect))
        chapters.add((key[0], key[1], dialect))
counts = collections.Counter()
examples = {}
seen = set()
with (
    (folder / "26767处断点与例子.jsonl").open("w") as detailed,
    (folder / "26767处断点与例子.tsv").open("w") as tab,
):
    writer = csv.writer(tab, delimiter="\t")
    writer.writerow(
        [
            "引文ID",
            "主要原因",
            "出处",
            "文件",
            "页",
            "词根",
            "标题候选词形",
            "候选TLA",
            "明确证据",
            "Crum原段",
            "当前科普特语",
            "当前译文",
            "前后经节ID",
        ]
    )
    for line in (cfg.derived / "complete_citations/citations.jsonl").open():
        source = json.loads(line)
        oid = source["id"]
        if oid not in pending:
            continue
        assert oid not in seen
        seen.add(oid)
        r = pending[oid]
        ref = source["reference"]
        links = r.get("expanded_local_links", source["local_links"])
        current = [
            sentences[l["sentence_id"]] for l in links if l["sentence_id"] in sentences
        ]
        kind = ref["kind"]
        evidence = []
        if kind != "verse":
            reason = "出处缺唯一节号或使用特殊分节"
            evidence.append("原出处类型=" + kind + "；原文=" + source["raw"])
        elif r["status"] == "标题扩展后仍待定位词条":
            direct = {t for f in r["forms"] for t in byform.get(f, [])}
            roots = byform.get(r["root"], [])
            if direct:
                reason = "标题有词典候选，定位证据仍有歧义"
                evidence.append("完整标题的全局词典候选=" + ",".join(sorted(direct)))
            elif roots:
                reason = "母词可定位，派生词或构式标题未定位"
                evidence.append(
                    "母词候选=" + ",".join(sorted(roots)) + "；展开形式仍未匹配词典"
                )
            else:
                reason = "Crum标题及母词拼写未命中词典索引"
                evidence.append(
                    "已查全部在线和XML词形；当前展开形式=" + "/".join(r["forms"])
                )
        elif not current:
            allowed = registry.from_codes(source["dialect_codes"])
            key = (ref.get("book_key"), ref.get("chapter"), ref.get("verse"))
            other = [sentences[s]["dialect_id"] for s in lookup.get(key, [])]
            if source["dialect_codes"] and not allowed:
                reason = "引文指定当前未收录的方言"
            elif other and allowed and not set(other) & set(allowed):
                reason = "经节存在，缺指定方言版本"
            elif not any((key[0], d) in books for d in allowed or registry.data):
                reason = "本地缺该卷的指定方言语料"
            elif not any(
                (key[0], key[1], d) in chapters for d in allowed or registry.data
            ):
                reason = "本地缺该章的指定方言语料"
            else:
                reason = "该章已收录，当前出处节号未命中"
            evidence.append(
                "引文方言代码="
                + ",".join(source["dialect_codes"])
                + "；同节已找到方言="
                + ",".join(sorted(set(other)))
            )
        elif r["checks"]:
            issues = sorted({x for p in r["checks"] for x in p["issues"]})
            if issues:
                reason = "目标有对应，来源或标注冲突"
                evidence.extend(issues)
            else:
                reason = "目标有对应，多个同形词条仍并存"
                evidence.append(
                    "候选=" + ",".join(sorted({p["tla"] for p in r["checks"]}))
                )
        else:
            reason = "词条已定位，当前经文完整词形与lemma未命中"
            evidence.append("待找TLA=" + ",".join(r["proposed_tlas"]))
            if re.search(
                r"annotation: (different reading|variant|omits)",
                source["paragraph"] or "",
            ):
                evidence.append(
                    "Crum所在原段含异文或省略标记，需按具体引文进一步定范围"
                )
        corpus_present = bool(current)
        target_forms = {
            t: dict(
                gloss=lex[t].gloss,
                pos=lex[t].pos,
                forms=lex[t].forms,
                dialects=lex[t].form_dialects,
            )
            for t in r["proposed_tlas"]
        }
        neighbor_ids = []
        for c in current:
            loc = explicit_bible_location(c["work"], c["chapter"], c["verse"])
            if loc and loc[2].isdigit():
                for delta in [-2, -1, 1, 2]:
                    for sid in lookup.get(
                        (loc[0], loc[1], str(int(loc[2]) + delta)), []
                    ):
                        if sentences[sid]["source"] == c["source"]:
                            neighbor_ids.append(sid)
        neighbors = [sentences[s] for s in dict.fromkeys(neighbor_ids)]
        row = dict(
            id=oid,
            reason=reason,
            evidence=evidence,
            reference=source["normalized_reference"],
            file=source["file"],
            page=source["page"],
            raw=source["raw"],
            root=r["root"],
            forms=r["forms"],
            target_entries=target_forms,
            paragraph=source["paragraph"],
            corpus_present=corpus_present,
            current_sentences=current,
            neighbor_sentences=neighbors,
            checks=r["checks"],
        )
        detailed.write(json.dumps(row, ensure_ascii=False) + "\n")
        writer.writerow(
            [
                oid,
                reason,
                row["reference"],
                row["file"],
                row["page"],
                r["root"],
                "/".join(r["forms"]),
                ",".join(r["proposed_tlas"]),
                "；".join(evidence),
                row["paragraph"],
                " | ".join(c["sent"] for c in current),
                " | ".join(c["translation"] for c in current),
                ",".join(c["id"] for c in neighbors),
            ]
        )
        counts[reason] += 1
        if reason not in examples:
            examples[reason] = row
assert seen == set(pending) and sum(counts.values()) == 26767
(folder / "原因数量.json").write_text(
    json.dumps(counts, ensure_ascii=False, indent=2) + "\n"
)
lines = [
    "# 26,767 处未配对引用的断点诊断",
    "",
    "每个原始引用 ID 均有一条诊断，附 Crum 原段、候选词条全部词形、当前经文及前后两节的同源语料。分类表示可观察到的断点；版本差异和经节错位需要进一步比对，尚未自动断定原因。",
    "",
    "| 断点 | 引用处数 |",
    "|---|---:|",
]
lines += [f"| {reason} | {n:,} |" for reason, n in counts.most_common()]
for reason, n in counts.most_common():
    r = examples[reason]
    lines += [
        "",
        "## " + reason,
        "",
        f"共有 {n:,} 处。例子 ID 为 {r['id']}，Crum 文件 {r['file']}，页 {r['page']}，出处 {r['reference']}。",
        "",
        "可核对证据：" + "；".join(r["evidence"]),
        "",
        "Crum 原段：",
        r["paragraph"] or "原段缺失",
    ]
    for c in r["current_sentences"][:1]:
        lines += ["", f"语料 {c['id']}：", c["sent"], "", c["translation"]]
    lines += ["", "需要的处理："]
    if "词典" in reason or "母词" in reason or "标题" in reason:
        lines += [
            "重建 Crum 母条目、子义项和派生词的关系，核对原拼写与同页候选；现有经文可帮助验证词义，词条身份仍需来源依据。"
        ]
    elif "方言" in reason or "语料" in reason or "收录" in reason:
        lines += [
            "补对应方言版本或核对出处编号。现有其他方言的同节文本可作平行见证，记录其各自方言身份。"
        ]
    elif "节号" in reason or "分节" in reason:
        lines += [
            "核对具体版本的章节与分节。整卷引用先保留为文献见证，缩小到具体经文后再构建例句配对。"
        ]
    elif "并存" in reason or "冲突" in reason:
        lines += [
            "结合原文词性、性别、义项及方言括注判定；人工标签冲突另列，避免把未定项设成明确负例。"
        ]
    else:
        lines += [
            "先比对所附前后两节，检查版本读法、原译错位和未识别词形；只有对应关系有实际证据时才更换句子或词条。"
        ]
(folder / "逐类原理与真实例子.md").write_text("\n".join(lines) + "\n")
print(json.dumps(counts, ensure_ascii=False, indent=2))
