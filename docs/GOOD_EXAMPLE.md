> 历史第一版说明。当前按方言分开的双模型与集合选择流程以 [PIPELINE_V2.md](PIPELINE_V2.md) 为准。

# 什么是好的科普特语词典例句

## 两个必须分开的维度

**词义有效性**：这个片段中目标词的用法是否对应所查 TLA？C992 father 与 C998 barley 同形，字符串命中无法区分。英文译文和定义的相似度可提供线索，但句子可能同时涉及两个意思，且译文可能错位，因此不能等同于目标 token 的人工义项标注。

**展示质量**：在词义有效的前提下，该片段是否典型、信息充分、便于理解、自足且适合展示？长度适中、常见词比例、主句中的目标词、较低的依存复杂度可以作为可解释特征；不是越短越好，也不能因为 Coptic 没有英语式大写就扣分。

原始 GDEX 论文提出典型性、释义信息和可理解性，并用长度、频率、指代及句子完整性等作近似指标。不能原样搬用英语的长度阈值和词频门槛到 Coptic。参考：[GDEX 2008](https://euralex.org/elx_proceedings/Euralex2008/095_Euralex_2008_Adam%20Kilgarriff_Milos%20Husak_Katy%20McAdam_Michael%20Rundell_Pavel%20Rychly_GDEX_Automatically%20Finding%20Good%20Di.pdf)。

[开源 GDEX 实现](https://github.com/zentrum-lexikographie/gdex) 提供可配置规则和句法条件，面向 spaCy 处理后的文本及其语言资源。这里使用可从 Coptic TT 原生标注计算的 GDEX 启发特征，没有声称直接运行该德语实现，也没有证据声称它是唯一开源方案。

## 本次特征

- 24 个表面特征：长度、字符、标点、目标词位置/重复、固定全语料句面词频、词汇多样性、top-k 词表覆盖、译文存在性。
- 12 个依存特征：标注可用性、token 数、主语/宾语/从句/并列数、深度、依存距离、谓语存在、代词比例、目标是否位于主句。
- 2 个义项特征：与训练时相同定义的词集合覆盖率、真实 BGE cosine。

TT 自带的 `func` / `head` 提供依存关系，`parsing` 元数据说明自动/校验状态；它们不全等于人工 UD gold。保留原 native POS，不把未映射词性悄悄当成名词。映射参考 [UD Coptic morphology](https://universaldependencies.org/cop/morphology.html)。官方表与注释在 COP/CFOC 的细节上存在不一致，本实现跟随表格并保留 xpos 供追溯。

## 人工复核表建议

对每个候选独立记录：目标 token 义项是否正确（是/否/不确定）、译文是否对齐、上下文是否自足、是否典型、是否信息充分、阅读难度、整体偏好和理由。未知不要用 0 代替。

评审时隐藏模型名称和分数，至少分别覆盖同形名词、动词、虚词、低频词以及有/无可靠依存标注的文本。四个诊断 TLA 不能替代这个独立评测。
