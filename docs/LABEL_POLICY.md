> 历史第一版说明。当前按方言分开的双模型与集合选择流程以 [PIPELINE_V2.md](PIPELINE_V2.md) 为准。

# 标签、负例与实验协议

| 证据 | signed_label | 排序 relevance | 可信范围 |
|---|---:|---:|---|
| 人工 priority 1/2/3 | 3/2/1 | 30/20/10 | 仅对应 TLA 和已对齐片段 |
| 人工选中但 priority 空白 | null | 顺序任务 null；收录任务 10 | 知道被选中，不捏造 3 分或严格名次 |
| 明确 forbidden_urns 匹配 | -1 | 0 | 按 TLA 限定；ALL/ONLY 保留人工选中例外 |
| 没有人工标注的全语料候选 | null | null | unknown，不是负例 |
| C992/C998 交叉候选 | -1 | 0 | 自动推导假设，不是新增专家标注 |
| Crum 引用顺序 | null | max(10,31-rank) | 低权重的可撤销弱监督假设 |

LightGBM lambdarank 需要非负 relevance，因此数据中保留 -1 的含义，训练时明确转换到 0；不把 -1 样本直接剔除后仍声称“模型学过坏例句”。

三类 query 分开：`TLA|manual` 仅学习明确的人工顺序；`TLA|manual_acceptance` 用全部人工选中例作为收录正例（10），与明确禁用例（0）比较；`TLA|crum:sense_idx` 学习弱引用顺序。这样未填写 priority 的人工好例句仍参与训练，同时不会被捏造成严格排名。人工样本权重 1.0，Crum 0.35 是本次公开的实验选择，不是调参得出的最优比例。

完整 forbidden 匹配关系导出到 `forbidden_matches.tsv`；用于训练的负例按句子 ID 的固定 SHA-256 顺序每 TLA 最多选 20 条，避免海量功能词负例压过少数正例。检索和完整负例关系不受此训练抽样上限影响。

重复 witness 在同一 query 按 Coptic 文本去重。若同一个片段有多个不同 relevance，保留较高值并把冲突写入 `label_conflicts.json`，供人工判断，而不是隐藏。正例/禁用范围的命中冲突也会计数，不在此自动决定专家原意。

## 负例对照

`none` 不新增推导负例。`naive` 从同主 lemma 的其他 TLA 正例生成负例，作为待检验的朴素策略。`conservative` 只使用已核对的 C992/C998 互斥义项对；father / father-as-title 不在允许名单。两种策略都不覆盖目标 TLA 已有正例，最多每个正例一条，且使用目标 TLA 的 gloss 重新计算语义特征。

负例只从当前训练折的数据生成。验证候选固定，不把模型刚生成的“负标签”加入验证后宣称获得新的人工真值。

## 划分与指标

主 lemma 经连续双元音折叠后分折，同形 TLA 同折。每折额外清除训练集中与验证集共享同一 Coptic 片段的行。未声称解决所有不规则拼写族、文本版本近重复或翻译改写的泄漏问题。

报告线性 relevance 的 NDCG@6、成对顺序准确率、random 对照、人工子集、Crum 子集及按 lemma 聚类的探索性 bootstrap 区间。Crum relevance 差小且 query 常很短，随机 NDCG 也很高，不能把该指标解释为全语料义项准确率。

原表实际有 1,714 行，只有 58 行填写 priority 1/2/3（23/21/14）。大多数“好例子”只表达被选中，不能像旧代码那样一概赋成 3。人工严格排序验证样本因此很少。
