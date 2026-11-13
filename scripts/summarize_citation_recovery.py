"""Quantify recovery without conflating source occurrences with new training texts."""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.config import Config
from coptic_examples.dataset import text_key

cfg = Config()
folder = cfg.reports / "本地复盘" / (sys.argv[1] if len(sys.argv) > 1 else "批量修复")
known = {}
for line in (cfg.derived / "learning_examples.jsonl").open():
    r = json.loads(line)
    known[(r["tla"], r["dialect"], r["text_key"])] = r["y"]
unique = {}
seen = Counter()
for line in (folder / "恢复配对_待验证.jsonl").open():
    r = json.loads(line)
    key = (r["tla"], r["dialect"], text_key({"sent": r["text"]}))
    state = (
        "已有正例"
        if known.get(key, 0) > 0
        else "与既有负例冲突" if known.get(key, 0) < 0 else "新增文本配对"
    )
    seen[state] += 1
    item = unique.setdefault(key, dict(r, training_status=state))
    item["citation_ids"] = sorted(set(item["citation_ids"]) | set(r["citation_ids"]))
with (folder / "去重恢复配对.jsonl").open("w") as f:
    for r in unique.values():
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
counts = Counter(r["training_status"] for r in unique.values())
summary = json.loads((folder / "结果.json").read_text())
summary["deduplicated_recovered_pairs"] = counts
summary["deduplicated_pair_total"] = len(unique)
(folder / "结果.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
)
groups = defaultdict(Counter)
for line in (folder / "逐条修复.jsonl").open():
    r = json.loads(line)
    groups[r["old_status"]][r["status"]] += 1
lines = [
    "# Crum 成批漏配修复",
    "",
    "本次重放全部 41,808 处原始引文，逐条验证文件哈希和原文位置。引文数与去重词条—文本配对数分别计数。",
    "",
    "## 共同问题与处理",
    "",
    "| 共同原因 | 处理 | 验证与边界 |",
    "|---|---|---|",
    "| 派生词缩写 | 用母条目完整词根展开，优先完整派生词形 | 精确词形匹配，保留原缩写 |",
    "| 只含释义的续段 | 继承最近的有效科普特语标题 | 英文 gloss 标签不会重置标题 |",
    "| 页码粒度差异 | 唯一完整派生词形加母条目页码范围 | 单独记录此定位依据，供查页影 |",
    "| TT 将复合词拆开 | 连续 1–4 个完整 token 拼合核对 | 不用词内子串 |",
    "| 同形异义或词性冲突 | 保留候选及冲突位置 | 不据某句有目标词就消掉其他义项 |",
    "| 方言替换读法 | 识别本条引文旁当前方言的括号替换读法并隔离 | 已识别的替换读法留待单独归词条；沿用原经节链接 |",
    "| 原译节号错位 | 已记录诗篇两例的相邻原文对应 | 此轮没有对全书自动平移，单独保留原译疑义 |",
    "| 缺对应方言或特殊节号 | 保留原引文及缺口 | 需补实际版本或节号换算依据 |",
    "",
    "## 全量结果",
    "",
    "| 修复后状态 | 引用处数 |",
    "|---|---:|",
]
lines += [f"| {k} | {v:,} |" for k, v in summary["statuses"].items()]
lines += [
    "",
    "此前未进入训练配对、现找到唯一标题与完整词形对齐的引用共 "
    + str(sum(summary["recovered_occurrences_by_previous_status"].values()))
    + " 处。",
    "",
    "| 去重后的恢复配对 | 数量 |",
    "|---|---:|",
]
lines += [f"| {k} | {v:,} |" for k, v in counts.items()]
lines += [
    "",
    "这些记录属于自动恢复的 Crum 来源配对，expert_verified=false。现有训练集和已部署模型仍使用原版本；增量文件可单独核验、做消融后再并入。直接判读文件作为回归依据保留。恢复范围未覆盖全部语义歧义、方言括注和节号错位。",
]
(folder / "成批修复说明.md").write_text("\n".join(lines) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
