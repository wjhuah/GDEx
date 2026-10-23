# CDO examples 数据说明

## 当前状态

本目录目前**没有** `examples_sahidic.tab` 或 `examples_bohairic.tab`。本轮没有从已有 TSV 复制/改名伪装成新导出，也没有编造新的例句。

已找到 CDO 项目内的权威生成实现：[dictionary/utils/get_examples.py](../../../dictionary/utils/get_examples.py)。其输入包括：

- `/Users/whua/Desktop/coptic_dictionary/corpora/**/*.tt` 中的源语料；
- `/Users/whua/Desktop/coptic_dictionary/dictionary/utils/citations_manual_merged.tab` 中人工指定的 TLA/引用/优先级及禁用引用；
- `alpha_kyima_rc1.db` 中 `entries` 表的 lemma、POS、English definition 和 TLA 映射；
- Coptic NLP `mwe.tab` 与可能存在的 UD cache。

`dictionary/README.md` 的 “Building lexical data” 一节说明 `examples_sahidic.tab` / `examples_bohairic.tab` 属于本地数据构建产物，`dictionary/make_json.py` 会从 examples TSV 加载词条例句。因此这里找到的是**生成器和部分原始输入**，不是可直接从 `public/` 获取的完整例句 JSON/TSV 导出。
`dictionary/public/` 未找到可直接用于重新合成 TSV 的 CDO examples 导出；`make_json.py` 显示词条详情 JSON 是从 examples TSV 构建，而不是反向提供生成数据。

## 为什么本轮不能重新生成

生成器当前需要的数据库入口无法用本机已有文件满足：

- `dictionary/alpha_kyima_rc1.db` 是指向 `/Users/whua/Desktop/coptic_dictionary/script/alpha_kyima_rc1.db` 的符号链接，但该目标路径不存在。
- `/Users/whua/Desktop/coptic_dictionary/alpha_kyima_rc1.db` 虽存在，实测大小为 0 bytes，没有 `entries` 表，不能作为生成器所需数据库。
- `dictionary/utils/update_dict.py` 依赖的数据库构建模块 `dictionary_reader.py` 不在本机 `dictionary/utils/` 文件列表中。
- 生成脚本导入的 NLTK 在项目 venv 中未安装。

因此，本轮结论是：**权威生成脚本已找到，但其必要的词典数据库输入缺失，无法重新生成；新 TSV 尚未产出。** 旧 TSV 保留在原位置，未被覆盖。若补齐可用 `alpha_kyima_rc1.db`（含原脚本期望的 `entries` 表）及生成依赖后，建议在 `data/` 以外的隔离临时工作目录运行 `get_examples.py --tab_delim`，审查输出后再复制为本目录对应文件名。

## 旧文件检查

旧源文件为：

- `/Users/whua/Desktop/coptic_dictionary/dictionary/examples_sahidic.tab`
- `/Users/whua/Desktop/coptic_dictionary/dictionary/examples_bohairic.tab`

旧文件均为无表头、每行 9 列的 TSV。依据 `get_examples.py` 的写出代码，列依次是 lemma、TLA、POS、sentence、translation、title、URN、priority、dialect。

| 旧源 TSV | 行数 | 唯一 TLA | dialect 列计数 |
|---|---:|---:|---|
| `dictionary/examples_sahidic.tab` | 3,833 | 1,536 | sahidic：3,833 |
| `dictionary/examples_bohairic.tab` | 1,463 | 623 | bohairic：1,463 |
| `dictionary/utils/examples_sahidic.tab`（旧副本） | 4,058 | 1,612 | sahidic：4,058 |
| `dictionary/utils/examples_bohairic.tab`（旧副本） | 1,463 | 623 | bohairic：1,463 |

根目录与 `utils/` 下 Sahidic TSV 行数分别是 3,833 和 4,058，且逐行及行集合都不相同；Bohairic 行数相同，但内容/顺序也不完全相同。因此不能把其中一份未经核实地称作另一份的准确副本或新生成权威版。

对根目录两份旧 TSV 各以固定随机种子抽查 10 行：Sahidic 10/10、Bohairic 10/10 的 TLA 在 KELLIA lexicon 中存在，row lemma 与 KELLIA lemma 匹配，句子（去掉 `**` 标记后）含目标 lemma，且译文非空。抽查只说明样本，不代表全表所有行都通过。

旧文件中 C998/C992/C6/C19/C20 计数：

| 文件 | C998 | C992 | C6 | C19 | C20 |
|---|---:|---:|---:|---:|---:|
| Sahidic 根目录 TSV | 0 | 0 | 2 | 3 | 0 |
| Bohairic 根目录 TSV | 0 | 0 | 0 | 3 | 0 |
| Sahidic `utils/` TSV | 0 | 0 | 2 | 3 | 0 |
| Bohairic `utils/` TSV | 0 | 0 | 0 | 3 | 0 |

**C998 判断：**本轮没有新 TSV，故“新 TSV 是否含 C998”不适用；现有两 dialect TSV 都没有 C998。但数据源本身并非没有 C998：人工引用源 `citations_manual_merged.tab` 明列 3 条 C998 barley 例句引用（包含 Ruth 和 Shenoute 来源）。现有导出却为零，说明缺失发生在导出链路/数据库构建或这些旧导出版本，而不是没有定义过 C998 例句；由于期望的 SQLite 数据库缺失，无法断定究竟是哪一环造成遗漏。

注意：`dictionary/utils/examples_*.tab` 是另一组旧输出副本，不能在缺少数据库生成步骤时当作权威输入。两组旧 TSV 的行数不同。

## 字段定义（由 `get_examples.py` 生成代码核实）

| 列（从 0 开始） | 字段 | 含义 |
|---:|---|---|
| 0 | `lemma` | 目标词典 lemma |
| 1 | `TLA` | KELLIA/TLA 词条 ID |
| 2 | `POS` | 例句中的词性标签（如 `N`、`V`、`A`） |
| 3 | `sent` | Coptic 例句；命中词形以 `**` 包围 |
| 4 | `translation` | 例句英文译文 |
| 5 | `title` | 文献/作品标题 |
| 6 | `urn` | 文献 CTS URN |
| 7 | `priority` | 人工/自动选择后的例句顺序 |
| 8 | `dialect` | `sahidic` 或 `bohairic` |

Crum4 的 `app.py` 会优先读取本目录同名 TSV；文件缺失时逐文件回退到 `dictionary/` 下旧 TSV。因新文件尚未生成，本机当前运行行为会回退到旧文件。
