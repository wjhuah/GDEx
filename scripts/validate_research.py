#!/usr/bin/env python3
"""Validate fold equality, negative provenance, deployed inference and real HTTP."""
import hashlib
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from coptic_examples.config import Config,write_json
from coptic_examples.ablation import CHEAP,feature_fingerprint
from coptic_examples.retrieval import Search
from coptic_examples.dialects import Dialects


def main():
    cfg=Config();checks=[]
    def check(name,value,detail=None):
        checks.append(dict(check=name,passed=bool(value),detail=detail))
        if not value:raise AssertionError(name)
    data=json.loads((cfg.reports/'ablation_results.json').read_text())
    check('features_and_input_fingerprints_current',data['provenance']==feature_fingerprint(cfg))
    configured=Dialects().data
    present={r['dialect'] for r in data['results']}
    check('all_dialect_scopes_present',present<=configured.keys() and {(r['dialect'],r['scope']) for r in data['results']}=={(d,s) for d in present for s in ['full_cheap','cached_BGE_complete_case']})
    rows=[json.loads(l) for l in (cfg.derived/'ablation_rows.jsonl').open()]
    scopes={}
    for result in data['results']:
        identity=result['dialect']+'/'+result['scope'];ms=result['group_metrics'];base=next(iter(ms.values()))
        original={(r['group'],r['n'],r['fold']) for r in base}
        check(identity+'_identical_validation_groups',all({(r['group'],r['n'],r['fold']) for r in group}==original for group in ms.values()))
        check(identity+'_five_folds',len(result['folds'])==5)
        check(identity+'_finite_metrics',all(0<=r['ndcg_at_6']<=1+1e-10 and 0<=r['pair_accuracy']<=1 for values in ms.values() for r in values))
        if result['scope']=='full_cheap':
            check(identity+'_all_required_ablations',set(['tree_gdex','tree_ud','tree_full_no_ud','tree_full_no_gdex','tree_full_no_info','tree_full_no_sense','tree_extended_ud','tree_tab_only','tree_crum_only','tree_321_floor','tree_naive_homograph','tree_barley_father','quality_first_joint','two_head_baseline'])<=ms.keys())
        subset=[r for r in rows if r['dialect']==result['dialect'] and (result['scope']=='full_cheap' or r['cosine'] is not None)]
        scopes[(result['dialect'],result['scope'])]=subset
        check(identity+'_row_count',len(subset)==result['rows'])
        if result['scope']!='full_cheap':check(identity+'_no_fake_BGE',all(r['cosine'] is not None and -1.00001<=r['cosine']<=1.00001 for r in subset))
    audit=json.loads((cfg.reports/'barley_negative_training_audit.json').read_text())
    foldmaps={}
    for d in present:
        families=sorted({r['family'] for r in rows if r['dialect']==d});np.random.default_rng(42).shuffle(families)
        foldmaps[d]={f:i%5 for i,f in enumerate(families)}
    for d,s in scopes:
        subset=scopes[d,s]
        for fold in range(5):
            held={r['text_key'] for r in subset if foldmaps[d][r['family']]==fold}
            generated=[r for r in audit if r['dialect']==d and r['scope']==s and r['fold']==fold]
            check(d+'/'+s+f'_fold{fold}_negatives_never_validation',all(r['text_key'] not in held for r in generated))
            check(d+'/'+s+f'_fold{fold}_only_disjoint_pair',all({r['target'],r['source']}=={'C992','C998'} for r in generated))
    check('barley_negatives_actually_entered_training',bool(audit),len(audit))
    packets=list((cfg.reports/'research_shortlists').glob('*/*/packet.json'))
    cohort={(r['dialect'],r['TLA']) for r in json.loads((cfg.work/'cluster/research_results.json').read_text())}
    check('research_candidate_packets_exported',{(json.loads(p.read_text())['dialect'],json.loads(p.read_text())['TLA']) for p in packets}>=cohort,len(packets))
    for p in packets:
        r=json.loads(p.read_text());c=r['candidates'];label=r['dialect']+'/'+r['TLA']
        check(label+'_research_packet_invariants',len(c)<=10 and len({v['candidate_id'] for v in c})==len(c) and all(v['dialect']==r['dialect'] for v in c) and r['no_llm_called'] and r['no_embedding_model_loaded'])
    search=Search(cfg)
    for tla,dialect in [('C998','sahidic'),('C992','sahidic'),('C9683','bohairic'),('C998','bohairic')]:
        r=search.search(tla,ranker='research',dialect=dialect)
        check(tla+'/'+dialect+'_actual_research_query',r['dialect']==dialect and len(r['results'])==(0 if dialect=='bohairic' and tla=='C998' else 10))
        if dialect=='bohairic' and tla=='C9683':check('bohairic_explicit_provisional_fallback',r['ranker']=='cheap' and r['shortlist_evidence']['model_status']['provisional'])
    check('live_inference_loaded_no_embedding_or_LLM_runtime','torch' not in sys.modules and 'sentence_transformers' not in sys.modules and search.embeddings is None)
    import lightgbm as lgb
    m=lgb.Booster(model_file=str(cfg.work/'models/sahidic/research_ranker.txt'))
    check('deployed_research_feature_schema',m.feature_name()==CHEAP)
    http=[]
    for route in ['/','/search?query=C998&dialect=sahidic&ranker=research','/search?query=C992&dialect=sahidic&ranker=quality_first','/guide','/experiments','/experiments?dialect=bohairic&task=crum_style','/review.css','/font.woff','/reports/ABLATIONS.md']:
        with urllib.request.urlopen('http://127.0.0.1:8012'+route,timeout=60) as response:
            body=response.read();check('HTTP '+route,response.status==200 and len(body)>100)
            http.append(dict(route=route,status=response.status,bytes=len(body)))
    for payload in [[],dict(query='C998',dialect='unknown'),dict(query='C998',ranker='invented')]:
        try:urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8012/api/search',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'}))
        except urllib.error.HTTPError as e:check('invalid API request rejected',e.code==400)
        else:raise AssertionError('invalid API request accepted')
    for payload in [dict(query='C998',dialect='sahidic',ranker='research'),dict(query='C9683',dialect='bohairic',ranker='research')]:
        with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8012/api/search',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})) as response:
            r=json.load(response);check('JSON API '+payload['query'],len(r['results'])==10 and r['dialect']==payload['dialect'])
    write_json(cfg.reports/'research_validation.json',dict(passed=len(checks),checks=checks,http=http))
    print(len(checks),'research artifact and HTTP checks passed')


if __name__=='__main__':main()
