> 历史第一版说明。当前按方言分开的双模型与集合选择流程以 [PIPELINE_V2.md](PIPELINE_V2.md) 为准。

# 重建复盘与研究结论

## 首要假设

四个人工锚点的真实 BGE 对照见 `reports/BARLEY_FATHER.md`。它支持把不相容词义的正例作为另一义项的候选负例。这个结果只涉及义项区分，不证明负例能改善全局排序；真实五折对照见 `reports/RANKING_EXPERIMENT.md`。

father 和 barley 共享词形，但 father 和 father-as-title 的语义相近。不能把“不同 TLA”直接当作“互斥义项”，也不能把英文 gloss 没有共同单词当成互斥关系的证明。

## Crum 顺序的含义：已核对原序言

本机 `preface.pdf` 是扫描件，普通文本提取为空。已实际渲染并阅读原书前言印刷页 vii–viii（PDF 第 3–4 页）。第 vii 页说明分号组织意义或用法组，圣经引用采用已发表 Coptic 版本的节号，未必与 LXX 或英文版本相同。第 viii 页说明先以圣经及能用希腊原文/方言平行文本较可靠确认意义的例证开头，再列其他文献类别。

这给出了两个直接的研究约束：

1. Bible 原始节号不能直接拿来对齐英文/其他版本；目标 lemma 检查失败应列为待复核，不直接断言 Crum 错了。
2. 引用排列包含文献组织策略，没有足够依据把它解释成从“最好教学例句”到“最差教学例句”的严格排序。

因此本项目仍保留学习该排列的实验，但明确称作低权重弱监督。模型是否更适合词典用户，必须用独立人工偏好验证。

来源：[本机 Crum 序言](../../preface.pdf)。对应页面渲染保留在被忽略的 `cache/literature/`，没有把整本 OCR 或 PDF 复制进项目源码。

## 对旧工作做了什么核验

已遍历旧目录 132 个 Python 文件的声明和依赖，记录哈希；重点阅读 `dictionary/utils/get_examples.py`、`ML4CD/generate_gdex_dataset.py`、Crum v14/v15 解析/合并/特征/训练、Crum4 检索/负例/评测及已有验证文档。函数级现场清单保存在 `reports/source_manifest.json`，不是声称 132 个文件都逐行重写。

发现并避免：

- 原始人工表与 annotated 表的引用/句子错配；新对齐以原始表和当前 TT 为准。
- 空白 gloss 导致大片 sm=0；新训练/检索统一使用词典 gloss。
- 同一句话的重复行、相近义项被机械当负例、unknown 混成 0。
- 旧脚本删掉 forbidden=-1，导致所谓“坏例句训练”并未实际发生。
- 旧 Crum 模型导出最后一折；新模型在全部清洁数据上重训。
- 训练按 TSV 表面词形统计词频、检索按 lemma 统计词频造成的分布不一致。
- 占位译文参与语义评分、候选 cap 隐藏以及旧 POS 映射遗漏。

## 相关资料的作用

查阅了本机 `相關工作/` 的论文和词典编纂章节，重点关注 GDEX 标准、不同类型例句的评价差异及自足性。其结论用于设计评价维度；没有用他人报告的数值冒充本项目实验。

外部一手资料：

- [GDEX 原始论文](https://euralex.org/elx_proceedings/Euralex2008/095_Euralex_2008_Adam%20Kilgarriff_Milos%20Husak_Katy%20McAdam_Michael%20Rundell_Pavel%20Rychly_GDEX_Automatically%20Finding%20Good%20Di.pdf)
- [开源 GDEX 规则实现](https://github.com/zentrum-lexikographie/gdex)
- [Coptic UD 词性映射](https://universaldependencies.org/cop/morphology.html)

## 仍需完成的研究验证

这次完成的是可运行、可审计的完整研究基线，并没有取得新的全语料人工真值。还需人工处理未对齐引用和冲突、核对 Coptic Bible 节号、标注隐藏模型身份的 top-10 样本，并扩大只含少量明确 priority 的人工排序集。不能用自动相似度为自己生成真值，再拿这个真值证明自己的相似度正确。
