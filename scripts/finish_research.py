#!/usr/bin/env python3
"""Train the selected cheap research baseline and audit the four sense anchors."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from coptic_examples.config import Config,write_json
from coptic_examples.ablation import CHEAP,SEMANTIC,CachedVectors,fit_ranker,predict,generate_negatives
from coptic_examples.lexicon import load
from coptic_examples.dataset import text_key
from coptic_examples.text import sm_freq
from coptic_examples.dialects import Dialects


def main():
    cfg=Config();rows=[json.loads(l) for l in (cfg.derived/'ablation_rows.jsonl').open()]
    vectors=CachedVectors(cfg);lex=load(cfg.lexicon)
    anchors=json.loads((cfg.reports/'barley_father.json').read_text())['rows'];diagnostic=[]
    data=[r for r in rows if r['dialect']=='sahidic']
    for anchor in anchors:
        held=text_key(dict(sent=anchor['sent']))
        own=next(r for r in data if r['task']=='acceptance' and r['TLA']==anchor['expert_tla'] and r['text_key']==held)
        cross=lex[anchor['cross_tla']]
        wrong=dict(own,TLA=cross.tla,gloss=cross.gloss,x=own['x'][:-1]+[sm_freq(cross.gloss,own['candidate']['translation'])],cosine=vectors.cosine(cross.gloss,own['candidate']['translation']))
        # Seen-family diagnostic, not the strict unseen-family main evaluation.
        training=[r for r in data if r['text_key']!=held]
        row=dict(expert_tla=own['TLA'],cross_tla=cross.tla,translation=anchor['translation'],held_out_text=held,scores={})
        for semantic in [False,True]:
            train=[r for r in training if not semantic or r['cosine'] is not None]
            for mode in ['none','barley_father']:
                neg=generate_negatives(train,mode,lex,vectors)
                if semantic:neg=[r for r in neg if r['cosine'] is not None]
                assert all(r['text_key']!=held for r in train+neg)
                columns=SEMANTIC if semantic else CHEAP
                model=fit_ranker(train+neg,columns)
                scores=predict(model,[own,wrong],columns)
                name=('cached_bge' if semantic else 'cheap')+'_'+mode
                row['scores'][name]=dict(own=float(scores[0]),cross=float(scores[1]),margin=float(scores[0]-scores[1]),correct=bool(scores[0]>scores[1]),training_negatives=len(neg))
        diagnostic.append(row)
    vectors.connection.close()
    write_json(cfg.reports/'barley_loo_diagnostic.json',dict(protocol='Leave one of four expert-anchor Coptic texts out everywhere, allow same family in training; cross labels inferred, diagnostic only; no independent token-level gold',rows=diagnostic,
        summaries={name:dict(correct=sum(r['scores'][name]['correct'] for r in diagnostic),total=len(diagnostic),mean_margin=float(np.mean([r['scores'][name]['margin'] for r in diagnostic]))) for name in diagnostic[0]['scores']}))
    for dialect in Dialects().data:
        directory=cfg.work/'models'/dialect
        selected=[r for r in rows if r['dialect']==dialect]
        acceptance=[r for r in selected if r['task']=='acceptance']
        group_gains={}
        for r in selected:group_gains.setdefault((r['task'],r['group']),set()).add(r['gain'])
        acceptance_groups=sum(task=='acceptance' and len(gains)>1 for (task,group),gains in group_gains.items())
        style_groups=sum(task=='crum_style' and len(gains)>1 for (task,group),gains in group_gains.items())
        sufficient=len(acceptance)>=100 and acceptance_groups>=20 and style_groups>=20
        if sufficient:
            model=fit_ranker(selected,CHEAP);model.save_model(str(directory/'research_ranker.txt'))
            write_json(directory/'research_ranker.json',dict(dialect=dialect,columns=CHEAP,kind='lightgbm',file='research_ranker.txt',
                augmentation='none; barley/father not adopted because the measured benefit is inconclusive',
                labels='continuous within-source preferences; expert acceptance/order and Crum style remain distinct queries',
                provisional=False,validation='exploratory same-fold selection; no independent final test',
                manual_sha256=hashlib.sha256(cfg.manual.read_bytes()).hexdigest(),
                corpus_fingerprint=json.loads((directory/'shortlist_metadata.json').read_text())['corpus_fingerprint']))
        else:
            (directory/'research_ranker.txt').unlink(missing_ok=True)
            write_json(directory/'research_ranker.json',dict(dialect=dialect,provisional=True,fallback='cheap',
                acceptance_rows=len(acceptance),acceptance_groups=acceptance_groups,style_groups=style_groups,
                reason='Insufficient independent ordering/inclusion evidence. Requires at least 100 acceptance rows, 20 mixed acceptance groups and 20 ordered Crum groups; do not deploy an unvalidated constant ranker.'))
    print('Research model trained; four leave-one-text-out diagnostics saved.')


if __name__=='__main__':main()
