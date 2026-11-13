"""Produce an exclusive occurrence-level balance after each recovery pass."""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parents[1]
folder = (
    root / "reports/本地复盘" / (sys.argv[1] if len(sys.argv) > 1 else "批量修复第二轮")
)
statuses = Counter()
remaining = Counter()
causes = Counter()
seen = set()
scoped = (folder / "异写表来源.json").exists()
original = {}
if scoped:
    for line in (root / "data/derived/complete_citations/citations.jsonl").open():
        record = json.loads(line)
        original[record["id"]] = record
with (folder / "剩余引用.tsv").open("w") as output:
    writer = csv.writer(output, delimiter="\t")
    writer.writerow(["引文ID", "出处", "原始文件", "页", "候选词条", "剩余原因"])
    for line in (folder / "逐条修复.jsonl").open():
        r = json.loads(line)
        assert r["id"] not in seen
        seen.add(r["id"])
        if scoped and "S" not in r["original_dialect_codes"]:
            statuses["原记录未明确含S，仅记录"] += 1
            continue
        existing = (
            any(p["dialect"] == "sahidic" for p in original[r["id"]]["training_pairs"])
            if scoped
            else r["old_status"] == "training_pair_available"
        )
        if existing:
            statuses["原有训练配对引用"] += 1
        elif r["accepted"]:
            statuses["累计自动恢复引用"] += 1
            causes[r["binding_basis"]] += 1
        else:
            statuses["仍未形成配对引用"] += 1
            remaining[r["status"]] += 1
            writer.writerow(
                [
                    r["id"],
                    r["reference"],
                    r["file"],
                    r["page"],
                    ",".join(r["proposed_tlas"]),
                    r["status"],
                ]
            )
assert len(seen) == sum(statuses.values()) == 41808
result = dict(
    total_occurrences=41808,
    exclusive_statuses=statuses,
    pending_reasons=remaining,
    automatic_binding_evidence=causes,
)
(folder / "剩余数量.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n"
)
new = json.loads((folder / "结果.json").read_text())["deduplicated_recovered_pairs"]
lines = [
    "# 批量整理余额",
    "",
    "统计单位为原始 Crum 引用位置。每个 ID 仅归入以下一种状态。",
    "自动恢复指来源定位与 TT 对齐通过当前流程，expert_verified=false，尚未并入训练。",
    "",
    "| 状态 | 引用处数 |",
    "|---|---:|",
]
lines += [f"| {k} | {n:,} |" for k, n in statuses.items()]
lines += ["", "| 剩余主因 | 引用处数 |", "|---|---:|"]
lines += [f"| {k} | {n:,} |" for k, n in remaining.items()]
lines += ["", "| 去重文本配对 | 数量 |", "|---|---:|"]
lines += [f"| {k} | {n:,} |" for k, n in new.items()]
lines += [
    "",
    "此轮增加完整 lemma 对齐、唯一完整标题跨页定位、明确语法性别筛选、母条目下缩写构式归属。",
    "长词的 TT 词性冲突保留为 warnings，短功能词原生词性、人工标签冲突、方言括号替换读法继续保留独立检查。",
    "篇章序号错位与纯文本省略的替换读法仍需补充版本证据；缺文本记录保留原出处。",
]
(folder / "第二轮整理结果.md").write_text("\n".join(lines) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
