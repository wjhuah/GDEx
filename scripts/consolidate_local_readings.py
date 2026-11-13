"""Reconcile citation evidence and preserve the provenance of each decision."""
import csv
import json
from collections import Counter
from pathlib import Path

folder = Path(__file__).resolve().parents[1] / "reports" / "本地复盘"

def read(name):
    with (folder / name).open() as stream:
        for line in stream:
            yield json.loads(line)

cohort = {r["id"]: r for r in read("29998处原文与见证.jsonl")}
readings = {}
for name in ["AI语文学判读_第二批.jsonl", "AI语文学判读_第三批.jsonl"]:
    for r in read(name):
        assert r["id"] not in readings
        readings[r["id"]] = r["direct_reading"]
assert len(cohort) == 29998 and len(readings) == 31
assert set(readings) <= set(cohort)
counts = Counter()
dest = folder / "29998处审阅状态总表"
with dest.with_suffix(".jsonl").open("w") as detail, dest.with_suffix(".tsv").open("w") as table:
    writer = csv.writer(table, delimiter="\t")
    writer.writerow(["引文ID", "经文", "页", "标题", "审阅状态", "结论或缺口", "词条", "目标位置", "经文版本数"])
    for oid, record in cohort.items():
        reading = readings.get(oid)
        if reading:
            status = "已记录AI逐条判读"
            reason = reading["reason"]
        elif not record["versions"]:
            status = "待补文本后判读"
            reason = "当前材料缺少对应方言经文，需补原版本或页影。"
        else:
            status = "待逐条判读"
            reason = record["previous_reason"]
        counts[status] += 1
        row = dict(record, review_status=status, direct_reading=reading)
        detail.write(json.dumps(row, ensure_ascii=False) + "\n")
        writer.writerow([oid, record["reference"], record["page"], record["heading"], status, reason,
                         reading.get("tla", "") if reading else "",
                         " ".join(reading.get("tokens", [reading.get("token", "")])) if reading else "",
                         len(record["versions"])])
assert sum(counts.values()) == 29998
report = [
    "# 全量审阅状态", "",
    "本表逐项区分原文定位、自动对齐和直接义项判读，所有记录保留原始引文 ID。",
    "文件写入每一行证明记录已处理；语文学审定完成情况由审阅状态列表示。", "",
    "| 状态 | 引用处数 |", "| --- | ---: |",
    *[f"| {k} | {v:,} |" for k, v in counts.items()], "",
    "已记录 AI 判读的 31 处保留词性、分词、原译错位及方言构式差异等具体结论。",
    "8,616 处需要补齐对应方言经文或原版本页影；这些记录的词条义项仍待核读。",
    "其余记录需要直接阅读各自的 Crum 义项段和经文，当前规则核查结果保留为线索。",
]
(folder / "全量审阅状态.md").write_text("\n".join(report) + "\n")
print(json.dumps(counts, ensure_ascii=False))
