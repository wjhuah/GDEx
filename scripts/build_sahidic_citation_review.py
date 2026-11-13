import sys, json, csv, re
from pathlib import Path
from collections import Counter

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from coptic_examples.config import Config
from coptic_examples.corpus import Corpus

folder = root / "reports/本地复盘/仅S异写兼容"
raw = {
    r["id"]: r
    for r in map(
        json.loads, (root / "data/derived/complete_citations/citations.jsonl").open()
    )
}
rows = list(map(json.loads, (folder / "逐条修复.jsonl").open()))
exact = {
    r["id"]: r
    for r in map(json.loads, (folder.parent / "仅S精确匹配/逐条修复.jsonl").open())
}
pairs = {}
gains = Counter()
for r in rows:
    if (
        not any(p["dialect"] == "sahidic" for p in raw[r["id"]]["training_pairs"])
        and r["accepted"]
    ):
        gains[r["old_status"]] += 1
        for p in r["accepted"]:
            key = (p["tla"], p["dialect"], p["sentence_id"])
            pair = pairs.setdefault(
                key,
                dict(
                    p,
                    citation_ids=[],
                    label=1,
                    label_basis="Crum唯一标题与完整词形自动对齐",
                    expert_verified=False,
                ),
            )
            pair["citation_ids"].append(r["id"])
(folder / "恢复配对_待验证.jsonl").write_text(
    "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs.values())
)
result = json.loads((folder / "结果.json").read_text())
result["recovered_occurrences_by_previous_status"] = gains
result["new_staging_pairs"] = len(pairs)
(folder / "结果.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n"
)
new = [r for r in rows if r["accepted"] and not exact[r["id"]]["accepted"]]
lost = [r for r in rows if not r["accepted"] and exact[r["id"]]["accepted"]]
(folder / "异写新增引用.jsonl").write_text(
    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in new)
)
(folder / "异写新增冲突.jsonl").write_text(
    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in lost)
)
(folder / "异写对照结果.json").write_text(
    json.dumps(
        dict(newly_aligned=len(new), new_conflict=len(lost), net=len(new) - len(lost)),
        ensure_ascii=False,
        indent=2,
    )
    + "\n"
)
reasons = {
    "已有词条线索但目标词形不符": "已经知道要查哪个词，但当前经文里尚未找到这个词的完整写法或已有词形。要继续查词形、经文版本和前后节。",
    "标题扩展后仍待定位词条": "Crum把词头省略了，或写的是派生词、固定搭配；需要先还原完整表达，再确定词典编号。",
    "词形对齐但词条或标注有冲突": "句子里已经找到了词，但多个词条共用这个写法，或与来源、词性、已有人工记录发生冲突；要继续判断具体用法。",
    "已有词条线索但缺对应经文": "当前方言的本地经文尚未定位；要检查卷名、章节号和版本覆盖。其他方言经文留作线索。",
}
with Corpus(Config()).connect() as db, (folder / "剩余逐处中文说明.tsv").open("w") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(
        [
            "引文编号",
            "文件",
            "原始位置",
            "方言代码",
            "目标词形",
            "候选词条",
            "卡在哪里",
            "大白话原理",
            "Crum科普特语摘录",
            "本地对应科普特语",
            "语料位置",
        ]
    )
    n = 0
    for r in rows:
        if (
            "S" not in r["original_dialect_codes"]
            or any(p["dialect"] == "sahidic" for p in raw[r["id"]]["training_pairs"])
            or r["accepted"]
        ):
            continue
        coptic = []
        positions = []
        for link in r["expanded_local_links"]:
            if link["dialect"] != "sahidic":
                continue
            item = db.execute(
                "SELECT payload FROM sentence WHERE id=?", (link["sentence_id"],)
            ).fetchone()
            if item:
                coptic.append(json.loads(item[0])["sent"])
                positions.append(link["sentence_id"])
        source = raw[r["id"]]
        quotes = [
            q
            for q in re.findall("⟨([^⟩]+)⟩", source["paragraph"])
            if re.search("[\u03e2-\u03ef\u2c80-\u2cff]", q)
        ]
        writer.writerow(
            [
                r["id"],
                r["file"],
                source["char_start"],
                ",".join(r["original_dialect_codes"]),
                " / ".join(r["forms"]),
                ",".join(r["proposed_tlas"]),
                r["status"],
                reasons[r["status"]],
                " / ".join(quotes),
                " || ".join(dict.fromkeys(coptic)),
                " || ".join(dict.fromkeys(positions)),
            ]
        )
        n += 1
    assert (
        n
        == json.loads((folder / "剩余数量.json").read_text())["exclusive_statuses"][
            "仍未形成配对引用"
        ]
    ), n
print(
    "scoped staging",
    len(pairs),
    "pending examples",
    n,
    "new",
    len(new),
    "conflict",
    len(lost),
)
