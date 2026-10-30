#!/usr/bin/env python3
"""Compare target-syntax selection methods, always within a dialect."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from coptic_examples.config import Config,write_json
from coptic_examples.dialects import Dialects
from coptic_examples.corpus import Corpus
from coptic_examples.shortlist import Shortlist,export


def main():
    config=Config();summary=[];ranker='research' if '--research' in sys.argv else 'cheap'
    for dialect in Dialects().data:
        engine=Shortlist(config,dialect)
        with Corpus(config).connect() as db:
            cohort=[r[0] for r in db.execute('SELECT tla FROM candidate WHERE dialect=? GROUP BY tla HAVING COUNT(*)>=10 ORDER BY COUNT(*) DESC,tla LIMIT 12',(dialect,))]
        if dialect=='sahidic':cohort=list(dict.fromkeys(['C998','C992','C3519','C5945',*cohort]))
        for tla in cohort:
            r=engine.search(tla,ranker=ranker)
            packet=export(config,r,'research_shortlists' if ranker=='research' else 'shortlists')
            if len(r['shortlist'])<3:continue
            for method,selected in packet['local_selection_experiments'].items():
                summary.append(dict(TLA=tla,dialect=dialect,method=method,shortlist_size=len(r['shortlist']),**selected['metrics']))
            print(dialect,tla,len(r['shortlist']),flush=True)
    write_json(config.work/('cluster/research_results.json' if ranker=='research' else 'cluster/results.json'),summary)
    md=['# 三条例句的句法覆盖实验','',
        '候选由便宜的方言独立模型选出约十条；本地比较 rank、句法 K-means、穷举三元组合。没有调用大模型。覆盖率是这十条候选中自动句法特征的覆盖代理，不是独立人工质量评价。','',
        '| 方言 | 方法 | TLA 数 | 平均单句质量 | 句法覆盖 | 词汇重复 |','|---|---|---:|---:|---:|---:|']
    for dialect in Dialects().data:
        for method in ['rank','kmeans','coverage']:
            rows=[r for r in summary if r['dialect']==dialect and r['method']==method]
            if not rows:continue
            mean=lambda key:sum(r[key] for r in rows)/len(rows)
            md.append(f"| {dialect} | {method} | {len(rows)} | {mean('mean_quality'):.4f} | {mean('weighted_syntax_coverage'):.4f} | {mean('mean_lexical_redundancy'):.4f} |")
    md+=['','精确优化只保证在已给定十条和公开目标函数下的最优三元组合。它可能牺牲一点单句模型分数换取不同结构；不能据此宣称语言学专家也会同意。',
         '质量和覆盖分开报告。K-means 只是对照方法，不预设三种自然句法类别一定存在。新增方言由 dialects.toml 控制，不会合并已有两种方言的候选或模型。',
         '每个 TLA 的十条原文、引用、模型分项和本地三元组合详见 reports/shortlists/{dialect}/{TLA}/packet.json 与 review.html。']
    if ranker=='research':
        md.insert(2,'本表使用当前 45 特征研究树；Bohairic 自动回退到本方言暂定基线。候选包位于 reports/research_shortlists/，不覆盖原双任务基线。')
        md[-1]='每个 TLA 的候选原文、引用和集合指标详见 reports/research_shortlists/{dialect}/{TLA}/packet.json。'
    (config.work/('cluster/RESEARCH_RESULTS.md' if ranker=='research' else 'cluster/RESULTS.md')).write_text('\n'.join(md)+'\n',encoding='utf-8')


if __name__=='__main__':main()
