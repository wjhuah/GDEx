#!/usr/bin/env python3
"""Human-readable results derived from actual fold outputs, not hand-entered scores."""
import json
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from coptic_examples.config import Config

NAMES={
 'quality_first_joint':'质量优先 + 联合树','two_head_baseline':'原双任务组合（重做验证）',
 'random':'随机排序','gdex_rule':'固定 GDEX 启发式','tree_gdex':'树：GDEX',
 'tree_ud':'树：仅 UD','tree_gdex_ud':'树：GDEX + UD',
 'tree_full_no_sense':'树：GDEX + UD + 信息量','tree_full':'树：完整便宜特征（默认研究版）',
 'tree_full_no_gdex':'完整减去 GDEX','tree_full_no_ud':'完整减去 UD','tree_full_no_info':'完整减去信息量',
 'tree_extended_ud':'完整 + Crum3 式细 UD 计数','tree_tab_only':'完整：只用 TAB 训练',
 'tree_crum_only':'完整：只用 Crum 训练','tree_321_floor':'完整：旧 Crum 递减至 1 后持平',
 'tree_naive_homograph':'完整 + 朴素同形词负例','tree_barley_father':'完整 + barley/father 负例',
 'crum4_original':'Crum4 原公式','tree_cached_cheap':'同语义子集：便宜树',
 'tree_cached_bge':'同语义子集：树 + BGE','tree_cached_bge_barley':'同语义子集：树 + BGE + barley/father 负例'}


def main():
    cfg=Config();obj=json.loads((cfg.reports/'ablation_results.json').read_text())
    md=['# 实测消融：TAB、Crum、特征、赋分和负例','',
        '所有表格来自 ablation_results.json 的实际五折预测。先固定验证集，再改变一个训练设置；未调用大模型，也未重新计算 BGE。',
        '同拼写家族（包括 C992/C998）留在同一折，验证 Coptic 原文从训练集清除。分数相同时使用随机打破并列的期望 DCG，避免原始引用顺序偷偷给常数模型加分。','',
        'NDCG@6 均用固定的连续评价 gain；不会因为某个训练标签方案不同而替换评价标准。TAB 收录、TAB 有序、Crum 顺序分开报告，不把多数 Crum 组的高分掩盖人工小样本。','']
    for result in obj['results']:
        md += [f"## {result['dialect']} / {result['scope']}",'',f"{result['rows']:,} 行。语义子集仅保留真实 BGE 缓存完整的行，子集各方法候选相同；不能直接把子集数字与全便宜数据数字相减。",'',
            '| 设置 | 特征数 | TAB 收录 NDCG@6 | TAB 收录 P@3 | TAB 有序 NDCG@6 | Crum 顺序 NDCG@6 |','|---|---:|---:|---:|---:|---:|']
        fmt=lambda v:'—' if v is None else f'{v:.4f}'
        for r in result['summaries']:
            md.append(f"| {NAMES.get(r['setup'],r['setup'])} | {r['features']} | {fmt(r['acceptance']['ndcg_at_6'])} | {fmt(r['acceptance']['known_precision_at_3'])} | {fmt(r['manual_order']['ndcg_at_6'])} | {fmt(r['crum_style']['ndcg_at_6'])} |")
        sample=result['summaries'][0]
        md += ['',f"可评估组数：TAB 收录 {sample['acceptance']['groups']}；TAB 明确有序 {sample['manual_order']['groups']}；Crum 有序 {sample['crum_style']['groups']}。P@3 只针对已知好坏候选，不是全语料 top-3 的实测正确率。",'',
            '| 改动（左减右） | TAB 收录 ΔNDCG [95%区间] | TAB 有序 ΔNDCG [95%区间] | Crum 顺序 ΔNDCG [95%区间] |','|---|---|---|---|']
        for c in result['comparisons']:
            cells=[]
            for task in ['acceptance','manual_order','crum_style']:
                d=c['by_task'][task]
                cells.append('—' if d['delta_ndcg'] is None else f"{d['delta_ndcg']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]")
            md.append('| '+NAMES[c['setup']]+' − '+NAMES[c['reference']]+' | '+' | '.join(cells)+' |')
        md.append('')
    diag=json.loads((cfg.reports/'barley_loo_diagnostic.json').read_text())
    md += ['## barley / father 四个锚点的留一诊断','',
        '每次将一个人工例句的 Coptic 原文从所有训练来源及派生负例中清除，其余同词家族允许参与训练。跨义项负标签是研究假设，没有新增人工 token 义项标注；这个诊断不能代替独立的真实负例测试。','',
        '| 方法 | 自身义项高于交叉义项 | 平均原始分差 |','|---|---:|---:|']
    for name,r in diag['summaries'].items():md.append(f"| {name} | {r['correct']}/{r['total']} | {r['mean_margin']:.4f} |")
    md += ['','原始分差只在同一模型内有意义，不能用较大数值跨模型证明更好。四个锚点的有无负例正确数相同，未显示额外收益。','',
        '严格主实验把 C992/C998 放在同一折：验证这对词时不能使用该对的训练负例。因此全表增广效果主要衡量对其他未见词形的推广，四锚点留一才是同词内诊断；两者均不等于独立人工 token 义项测试。','',
        '## 本次采用的设计','',
        '- 默认网页：45 特征联合树排序，配独立单句质量门槛。另测质量优先组合（收录分 0.60 + 联合顺序分 0.25 + 信息量 0.15）；其留出排序较低，只保留为可切换对照，不采用为默认。',
        '- 不采用 36 个额外细 UD 计数：实测未改善 TAB 收录；保留代码和消融。',
        '- 不默认采用 barley/father 推导负例：便宜模型略降、语义子集略升，区间均跨零。朴素把所有同形词互作负例在本次 TAB 收录上下降。',
        '- 不把昂贵语义模型放进默认十条生成器；Crum4 和缓存 BGE 对照保留在独立结果中。',
        '- Bohairic 无可评估的正负混合收录组，只有 1 个人工有序组、4 个 Crum 组；不能宣称验证成功，研究树不部署，回退到原方言独立暂定基线。','',
        '## 解释边界','',
        '所有模型选择使用了这组验证折；没有选择后的独立测试，也没有对大量消融比较作多重检验修正。置信区间是按词形家族聚类的配对 bootstrap（1000 次），主要用来显示不确定性，不能当作最终显著性证明。',
        '指标评价已标注候选间的偏好；实际全语料推荐还受到义项、译文、引用匹配及句法标注误差影响。最终需要专家盲评十条候选和三句集合。','',
        '原始逐组指标及每折负例数量：[ablation_results.json](ablation_results.json)。逐条训练负例：[barley_negative_training_audit.json](barley_negative_training_audit.json)。四锚点：[barley_loo_diagnostic.json](barley_loo_diagnostic.json)。重跑：`python scripts/run_ablations.py --rebuild`，然后 `python scripts/finish_research.py` 和 `python scripts/publish_research.py`。']
    (cfg.reports/'ABLATIONS.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    comparison=cfg.reports/'deployed_cohort_comparison.json'
    if comparison.exists():
        cohort=json.loads(comparison.read_text())
        extra=['','## 实际全语料候选的取舍','',
            '固定 28 个词条（Sahidic 16、Bohairic 12），分别查询三种已部署设计。以下仅是同一个未校准单句模型的平均分，不是新的人工正确率。研究树更符合留出排序，但它选出的十条平均单句分较低；质量优先组合提高这个代理分，留出 NDCG 却下降，因此仅作对照。','',
            '| 方言 | 词条数 | 双任务基线均分 | 研究树均分 | 质量优先组合均分 |','|---|---:|---:|---:|---:|']
        for r in cohort['summary']:extra.append(f"| {r['dialect']} | {r['entries']} | {r['cheap_mean_quality']:.4f} | {r['research_mean_quality']:.4f} | {r['quality_first_mean_quality']:.4f} |")
        extra+=['','这说明模型目标确实有取舍，不能用自己优化的代理分证明推荐更好。默认纯联合树配独立质量门槛，但仍需对真实候选做专家盲评。逐词条见 [deployed_cohort_comparison.json](deployed_cohort_comparison.json)。']
        with (cfg.reports/'ABLATIONS.md').open('a',encoding='utf-8') as f:f.write('\n'.join(extra)+'\n')
    print('Published actual ablation tables.')


if __name__=='__main__':main()
