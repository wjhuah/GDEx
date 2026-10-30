# Crum 全部 Bible 标记提取与匹配审计

目标是完整保留 Crum 的专业编纂顺序，用于学习他的风格。所有方言、不能定位到节的引用和暂未归入 TLA 的引用都保留；没有先过滤 Sahidic。

| 检查 | 数量 |
|---|---:|
| files | 3415 |
| source_occurrences | 41808 |
| ref_book_only | 267 |
| binding_exact_form_page_gloss | 20575 |
| link_non_verse_reference | 646 |
| ref_verse | 41162 |
| link_linked | 31880 |
| link_dialect_not_local | 6341 |
| binding_unresolved_heading | 16559 |
| link_no_local_verse | 2941 |
| ref_chapter_only | 123 |
| binding_ambiguous_form_page | 2860 |
| binding_exact_form_page | 1805 |
| ref_lettered_section | 200 |
| ref_titulus | 50 |
| ref_ambiguous_book_group | 3 |
| ref_lettered_section_only | 2 |
| binding_no_heading | 9 |
| ref_subscriptio | 1 |
| raw_marker_count_independent | 41808 |
| extraction_recall_of_explicit_markers | 1.0 |
| verified_links | 11988 |
| training_coverage_is_not_extraction_coverage | True |

## 完整性保证与正确性边界

逐文件独立数 `{bible:` 标记；逐条核对字符范围可切回完全相同的原始标记。每条保存文件哈希、行号、字节位置、印刷页、原始引用、规范化引用、词条/段落顺序与方言证据。
这保证当前数字化文本中所有显式 Bible 标记均被提取，不等于证明扫描本所有未标记引用也已由数字化作者正确标注。没有凭空补章节或跨版本节号。
book-only、chapter-only、titulus 和字母章节是有效的引用形态，不应因不能映射一个句子而从提取结果删除。字母章节暂不猜测成 Daniel 的数字章节。
TLA 不唯一时保留候选清单；没有像旧 pick_best 一样静默取第一个。只有目标形式实际出现在对应方言 TT 标注中的链接才标 verified_target_form；仍不是逐 token 专家义项判断。
旧 v15 输出 30,466 条引用，不能代替本次完整原始清单。旧流程还会把尚未识别的新词头继承为上一词条；本次新词头会重置绑定，避免错误继承。
完整清单：`data/derived/crum_bible_occurrences.jsonl`。逐链接：`cache/crum_bible.sqlite3`。待复核状态均可按字段查询，不用从日志猜测。
