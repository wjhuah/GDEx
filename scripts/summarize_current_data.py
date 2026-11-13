"""Current data inventory with separate units for sources, corpus spans and labels."""

import sys, json, sqlite3
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.config import Config
from coptic_examples.source_pool import SupplementedCorpus
from coptic_examples.annotations import load_manual
from coptic_examples.dataset import text_key
from coptic_examples.text import real_translation, editorial_heading

cfg = Config()
out = cfg.reports / "本地复盘/数据规模与同形词"
out.mkdir(parents=True, exist_ok=True)
corpus = defaultdict(Counter)
unique = defaultdict(set)
with SupplementedCorpus(cfg).connect() as db:
    for (payload,) in db.execute("SELECT payload FROM sentence"):
        c = json.loads(payload)
        d = c["dialect_id"]
        corpus[d]["spans"] += 1
        corpus[d]["tt_tokens"] += len(c["tokens"])
        unique[d].add(text_key(c))
        corpus[d]["nonempty_translation"] += bool(c["translation"].strip())
        corpus[d]["translation_checks_passed"] += bool(
            real_translation(c["translation"])
            and not editorial_heading(c["translation"])
        )
for d in corpus:
    corpus[d]["distinct_coptic_texts"] = len(unique[d])
training = {}
for d in ["sahidic", "bohairic"]:
    c = Counter()
    rank_groups = defaultdict(list)
    for r in map(json.loads, (cfg.derived / f"model_rows_{d}.jsonl").open()):
        c["positive" if r["y"] > 0 else "negative"] += 1
        for source in ["tab", "crum"]:
            ranks = [v for v in r["ranks"] if v["source"] == source]
            if r["y"] > 0 and ranks:
                c[source + "_ranked_rows"] += 1
                rank_groups[source, r["tla"]].extend(v["grade"] for v in ranks)
    for source in ["tab", "crum"]:
        c[source + "_ranked_entries"] = sum(s == source for s, t in rank_groups)
        c[source + "_entries_with_multiple_grades"] = sum(
            s == source and len(set(v)) > 1 for (s, t), v in rank_groups.items()
        )
    training[d] = c
manual = load_manual(cfg.manual)
priorities = Counter(r["priority"] for r in manual)
tab = dict(
    source_rows=len(manual),
    rows_with_explicit_priority=sum(r["priority"].isdigit() for r in manual),
    priority_counts=priorities,
    entries_with_ALL_rule=len(
        {r["TLA"] for r in manual if "ALL" in r["forbidden_urns"].upper()}
    ),
)
coverage = json.loads((cfg.reports / "本地复盘/线上例句覆盖/最终核验.json").read_text())
crum = json.loads((cfg.reports / "本地复盘/仅S异写兼容/剩余数量.json").read_text())
augmentation = json.loads(
    (cfg.reports / "本地复盘/新增数据模型比较/增广数据库统计.json").read_text()
)
homographs = json.loads((out / "规模.json").read_text())
result = dict(
    corpus=corpus,
    training=training,
    tab=tab,
    online=coverage["online"],
    online_alignment=coverage["alignment"],
    crum_occurrences=crum["exclusive_statuses"],
    augmentation_fold_rows=sum(r["rows"] for r in augmentation),
    augmentation_homograph_fold_rows=sum(r["homograph"] for r in augmentation),
    augmentation_unrelated_fold_rows=sum(r["unrelated"] for r in augmentation),
    experimental_recovered_training_rows=4422,
    homographs=homographs,
)
(out / "完整数据规模.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n"
)
lines = [
    "# 当前数据规模与同形词清单",
    "",
    "日期：2026-11-13。原始训练数据库与模型保留；新增线上来源和完整检索语料采用独立数据库。下面各表的计数单位分别说明，互有重叠，直接相加会重复。",
    "",
    "## 词典和线上例句",
    "",
    "| 项目 | 数量 | 计数单位 |",
    "|---|---:|---|",
    "| 当前线上索引 | 11,539 | TLA词条 |",
    "| 线上有展示例句的词条 | 6,036 | TLA词条 |",
    "| 已保存线上展示例句 | 18,669 | 词条＋方言＋线上例句位置 |",
    "| 其中Sahidic | 14,001 | 来源记录 |",
    "| 其中Bohairic | 4,668 | 来源记录 |",
    "| 已对应到本地语料 | 18,540 | 线上例句记录 |",
    "| 尚待对应TT，原文仍全部保留 | 129 | 线上例句记录 |",
    "",
    "## 完整检索语料",
    "",
    "原数据库55,634个片段，补充897个压缩TT文件得到22,819个片段，再保存4个官方差异文件的860个版本片段，共79,313个片段。不同版本和树库副本保留，因此也给出去重原文数量。一个片段可能是一节经文、一个句子或多个分句。TT token按语料既有分词计数。",
    "",
    "| 方言 | 片段数 | TT词数 | 方言内去重原文 | 非空译文片段 | 通过现有译文检查 |",
    "|---|---:|---:|---:|---:|---:|",
]
for d, c in corpus.items():
    lines.append(
        f"| {d} | {c['spans']:,} | {c['tt_tokens']:,} | {c['distinct_coptic_texts']:,} | {c['nonempty_translation']:,} | {c['translation_checks_passed']:,} |"
    )
lines += [
    "",
    "合计TT词数为2,413,306。译文检查只识别空白、占位或标题等形式问题，具体对齐和词义另行核查。unknown方言保持独立。完整匹配可从网页Browse all corpus matches浏览，推荐仍按既有模型与30空格分组规则筛选。",
    "",
    "## 当前冻结训练数据",
    "",
    "单位为目标TLA＋方言＋去重原文。同一句原文用于两个词条，会形成两条记录。负例含TAB禁用规则展开，88个TLA具有ALL规则。",
    "",
    "| 方言 | 正例 | 负例 | 总记录 |",
    "|---|---:|---:|---:|",
    "| Sahidic | 11,486 | 86,625 | 98,111 |",
    "| Bohairic | 502 | 2,155 | 2,657 |",
    "| 合计 | 11,988 | 88,780 | 100,768 |",
    "",
    "输入特征为62项。新补齐的检索语料和线上自动例句尚未自动转成训练正例；模型保持原版本。",
    "",
    "## 人工TAB和排序证据",
    "",
    "人工TAB有1,714个源行，其中58行明确写priority：1有23行，2有21行，3有14行；其余1,656行的priority为空。当前冻结数据按对齐、去重和标签处理后，可用于明确TAB排序的记录见下表。源行数与去重训练记录分别统计，58与43的差额还需结合逐行对齐与合并记录解释。",
    "",
    "| 方言 | 有TAB排序分的正例记录 | 涉及词条 | 有多个分值的词条 | Crum排序记录 | Crum词条 |",
    "|---|---:|---:|---:|---:|---:|",
]
for d, c in training.items():
    lines.append(
        f"| {d} | {c['tab_ranked_rows']} | {c['tab_ranked_entries']} | {c['tab_entries_with_multiple_grades']} | {c['crum_ranked_rows']:,} | {c['crum_ranked_entries']:,} |"
    )
lines += [
    "",
    "TAB明确排序共43条记录，涉及21个方言—词条组合，其中12个组合具有不同优先级。因此TAB排序得分需要同时报告有效组数，样本规模有限。未写priority的正例仍可支持收录判断。",
    "",
    "## Crum来源和恢复数据",
    "",
    "| 项目 | 数量 | 单位 |",
    "|---|---:|---|",
    "| 原书圣经引用 | 41,808 | 原书引用位置，允许重复引用 |",
    "| 当前原训练Crum配对 | 10,871 | 词条—去重语料原文 |",
    "| 原记录明确含S的处理范围 | 33,752 | 引用位置 |",
    "| 其中原有S训练配对引用 | 10,185 | 引用位置 |",
    "| 其中累计自动恢复引用 | 5,090 | 引用位置，待审定 |",
    "| 其中继续查证 | 18,477 | 引用位置 |",
    "| 原记录未明确含S，仅记录 | 8,056 | 引用位置 |",
    "| 自动恢复形成的配对 | 5,233 | 词条—语料句子记录 |",
    "| 进一步按原文去重 | 4,951 | 词条—原文记录 |",
    "| 最终进入新增实验数据 | 4,422 | 新记录，尚未部署 |",
    "",
    "Crum恢复余额沿用上一轮仅S快照；新增检索语料尚未重新运行这一恢复流程。原训练100,768条＋新增实验4,422条，共105,190条实验数据。自动恢复、线上自动展示和人工审定保持来源区别。41,808处引用均留档。",
    "",
    "## 负例增广",
    "",
    "本轮原数据八算法对照未使用新增数据或增广。此前“原数据＋自动恢复＋混合负例”的五折实验保存19,046条折内增广记录：同形义项交叉负例3,741条，无关负例15,305条。不同训练折可重复使用同一负例，因此19,046按折计数。已核查与对应测试折原文重合为0。",
    "",
    "## 同形词清单",
    "",
    f"完整索引产生{homographs['canonical_families']:,}组同词头候选，涉及{homographs['canonical_entries']:,}个不同TLA；其中{homographs['canonical_within_dialect']:,}组在同一已知方言内发生碰撞。",
    "同方言主表中，309组是同词性而英文释义不同，704组含词性差异，153组词性与英文释义相同、仍需核定词条划分。另有82组仅跨方言或方言证据不足，放在附录。",
    f"将变化词形纳入后，共{homographs['all_form_collision_groups']:,}组完整词形冲突，其中{homographs['all_form_within_dialect']:,}组发生在同一方言内，涉及{homographs['all_form_entries']:,}个不同TLA。同一对词条可通过多个词形重复出现。",
    "同词性按词典原始标签判断，保留V、VSTAT等区别。相同字母写法的词性、义项和方言证据均保留。同形候选仅说明有消歧需求。同义、性别差异或同词条划分方式均可能造成多个TLA，未自动生成互斥负例。多词表达保留词间边界，完整词形比较，避免子串匹配。",
    "",
    "文件：regression_vocabulary.md为20组重点词与固定错误案例；homograph_vocabulary.md为完整同词头候选表；homograph_forms.md为全部变化词形冲突表。机器可读逐条记录为同词头候选.json与全部词形冲突.json。",
    "",
    "20组精选表见homographs_top20.md，包含74个选定TLA义项。完整词表继续保留，精选表作为优先回归检查范围。131项测试通过。已补充父亲/称号/大麦、方言隔离、词性差异、重复词形、变化词形碰撞、大小写、完整词边界、同释义复核与语法义项的单元测试。测试结果见测试记录.txt。原模型、原训练数据保持，GitHub未推送。",
]
(out / "中文数据规模.md").write_text("\n".join(lines) + "\n")
print(json.dumps(dict(corpus=corpus, tab=tab), ensure_ascii=False, indent=2))
