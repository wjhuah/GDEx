#!/usr/bin/env python3
"""A corpus retrieval diagnostic; no invented human correctness labels."""
import json
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from coptic_examples.config import Config,write_json
from coptic_examples.shortlist import Shortlist


def main():
    cfg=Config();rows=[];engines={}
    for p in sorted((cfg.reports/'research_shortlists').glob('*/*/packet.json')):
        research=json.loads(p.read_text());d=research['dialect'];tla=research['TLA']
        cheap=json.loads((cfg.reports/'shortlists'/d/tla/'packet.json').read_text())
        if d not in engines:engines[d]=Shortlist(cfg,d)
        q=engines[d].search(tla,ranker='quality_first')
        quality=lambda xs:sum(xs)/max(1,len(xs))
        rows.append(dict(dialect=d,TLA=tla,
            research_count=len(research['candidates']),cheap_count=len(cheap['candidates']),quality_first_count=len(q['shortlist']),
            research_mean_quality=quality([r['quality_score'] for r in research['candidates']]),
            cheap_mean_quality=quality([r['quality_score'] for r in cheap['candidates']]),
            quality_first_mean_quality=quality([r['acceptance_score'] for r in q['shortlist']]),
            research_cheap_top10_overlap=len({r['candidate_id'] for r in research['candidates']}&{r['candidate_id'] for r in cheap['candidates']})))
        print(d,tla,flush=True)
    summary=[]
    for d in engines:
        sub=[r for r in rows if r['dialect']==d]
        summary.append(dict(dialect=d,entries=len(sub),**{key:sum(r[key] for r in sub)/len(sub) for key in ['research_mean_quality','cheap_mean_quality','quality_first_mean_quality','research_cheap_top10_overlap']}))
    write_json(cfg.reports/'deployed_cohort_comparison.json',dict(protocol='Same fixed 28-entry retrieval cohort as cluster experiment; scores are the same uncalibrated acceptance model proxy, NOT human precision. Different ten-candidate sets. No LLM.',summary=summary,rows=rows))


if __name__=='__main__':main()
