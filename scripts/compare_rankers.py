#!/usr/bin/env python3
"""Full-corpus side-by-side output for the four historical diagnostic TLAs."""
import json
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from coptic_examples.config import Config,write_json
from coptic_examples.retrieval import Search


def main():
    config=Config();search=Search(config);summary=[]
    md=['# 全语料检索：线性与学习排序对照','',
        '所有候选来自新建的全局 TT 索引。没有语义候选截断。C992/C998 两种 ranker 共用同一个实验性义项对比过滤，避免把过滤带来的改变归功于 LightGBM。这里没有全语料相关性真值，因此不能计算有效的全语料 NDCG。','']
    for tla in ['C998','C992','C3519','C5945']:
        results={}
        for mode in ['linear','learned']:
            print(f'Search {tla} / {mode}',flush=True)
            r=search.search(tla,ranker=mode)
            write_json(config.reports/f'{tla}_{mode}.json',r);results[mode]=r
            assert all(x['translation'].strip() not in {'','...','ؤ'} for x in r['results'])
        md += [f'## {tla} — {results["learned"]["entry"]["gloss"]}','',
               f"检索 {results['learned']['candidate_count']} 个片段，语义评分 {results['learned']['semantic_scored_count']} 个；排除计数：`{json.dumps(results['learned']['excluded'])}`。",'',
               '| Rank | Linear | Learned |','|---:|---|---|']
        for i in range(max(len(r['results']) for r in results.values())):
            cells=[]
            for mode in ['linear','learned']:
                rs=results[mode]['results']
                if i>=len(rs):cells.append('—');continue
                c=rs[i];cells.append(c['translation'].replace('|','/')+f"<br>{c['title']} {c['chapter']}:{c['verse']} · {c['score']:.3f}")
            md.append(f'| {i+1} | {cells[0]} | {cells[1]} |')
        ids=[{r['id'] for r in results[m]['results']} for m in ['linear','learned']]
        summary.append(dict(TLA=tla,candidate_count=results['learned']['candidate_count'],semantic_scored_count=results['learned']['semantic_scored_count'],top10_overlap=len(ids[0]&ids[1]),manual_examples=len(results['learned']['manual_examples'])))
    md+=['','## 如何解释','',
         '重排改变了同一候选池中的先后顺序。义项明确只是好例句的必要条件；语法复杂度、上下文自足性和译文对齐还需复核。本表只用于人工评审，不能自动断言 learned 比 linear 更好。',
         '后续请对隐藏模型名称后的例句按“目标词义是否正确 / 是否自足 / 是否典型 / 是否适读”分别标注，保留分歧，不先看模型总分。']
    (config.reports/'CORPUS_COMPARISON.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    write_json(config.reports/'corpus_comparison_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
