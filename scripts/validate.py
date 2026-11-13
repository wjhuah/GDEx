#!/usr/bin/env python3
"""Validate current dialect-separated artifacts and lossless Crum extraction."""
import hashlib
import json
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from coptic_examples.config import Config,write_json
from coptic_examples.corpus import Corpus,fingerprint
from coptic_examples.crum_audit import MARKER
from coptic_examples.dialects import Dialects
from coptic_examples.shortlist import Shortlist,COLUMNS


def main():
    cfg=Config();registry=Dialects();checks=[]
    def check(name,value,detail=None):
        checks.append(dict(check=name,passed=bool(value),detail=detail))
        if not value:raise AssertionError(name)
    manifest=json.loads((cfg.reports/'source_manifest.json').read_text())
    check('original_inputs_unchanged',all(hashlib.sha256((cfg.source/r['path']).read_bytes()).hexdigest()==r['sha256'] for r in manifest['inputs'] if r['exists']))
    check('legacy_implementations_unchanged',all(hashlib.sha256((cfg.source/r['path']).read_bytes()).hexdigest()==r['sha256'] for r in manifest['legacy_implementations']))
    source={p.name:p.read_bytes().decode('utf-8') for p in (cfg.source/'Crum_plain_text').glob('*.txt')}
    raw_count=sum(len(MARKER.findall(t)) for t in source.values());ids=set();roundtrip=True;hashes=True;count=0
    for line in (cfg.derived/'crum_bible_occurrences.jsonl').open():
        r=json.loads(line);text=source[r['file']];count+=1
        if r['id'] in ids:raise AssertionError('Duplicate Bible occurrence ID')
        ids.add(r['id'])
        roundtrip &= text[r['char_start']:r['char_end']]==r['raw'] and len(text[:r['char_start']].encode())==r['byte_start']
        hashes &= hashlib.sha256(text.encode()).hexdigest()==r['file_sha256']
    check('all_explicit_Bible_markers_preserved',count==raw_count,count)
    check('all_reference_source_spans_roundtrip',roundtrip)
    check('all_reference_source_hashes_correct',hashes)
    with Corpus(cfg).connect() as db:
        check('corpus_fingerprint_current',db.execute("SELECT value FROM metadata WHERE key='fingerprint'").fetchone()[0]==fingerprint(cfg.corpus)[1])
        check('candidate_dialect_matches_source',db.execute('SELECT COUNT(*) FROM candidate c JOIN sentence s ON s.id=c.sentence_id WHERE c.dialect!=s.dialect').fetchone()[0]==0)
        check('unknown_dialect_not_silently_published',db.execute("SELECT COUNT(*) FROM candidate WHERE dialect='unknown'").fetchone()[0]==0)
        dialects=dict(db.execute('SELECT dialect,COUNT(*) FROM sentence GROUP BY dialect'))
        check('both_current_dialects_present',all(dialects.get(d,0)>0 for d in ['sahidic','bohairic']),dialects)
    for dialect in registry.data:
        directory=cfg.work/'models'/dialect;meta=json.loads((directory/'shortlist_metadata.json').read_text())
        check(dialect+'_separate_model_identity',meta['dialect']==dialect)
        check(dialect+'_registry_fingerprint',meta['dialect_registry_fingerprint']==registry.fingerprint)
        for kind in ['acceptance','crum_style']:
            p=directory/(kind+'.json')
            if p.exists():check(dialect+'_'+kind+'_feature_schema',json.loads(p.read_text()).get('columns',COLUMNS)==COLUMNS)
    packets=list((cfg.reports/'shortlists').glob('*/*/packet.json'))
    check('shortlist_packets_exist',bool(packets),len(packets))
    for path in packets:
        r=json.loads(path.read_text());dialect=r['dialect']
        check(dialect+'_'+r['TLA']+'_single_dialect',all(c['dialect']==dialect for c in r['candidates']))
        check(dialect+'_'+r['TLA']+'_at_most_ten',len(r['candidates'])<=10)
        best=max((c['quality_score'] for c in r['candidates']),default=0.)
        check(dialect+'_'+r['TLA']+'_single_quality_floor_before_diversity',all(c['quality_score']+1e-12>=best*.5 for c in r['candidates']))
        check(dialect+'_'+r['TLA']+'_no_LLM_or_embedding',r['no_llm_called'] and r['no_embedding_model_loaded'])
        ids={c['candidate_id'] for c in r['candidates']}
        check(dialect+'_'+r['TLA']+'_three_selection_uses_only_candidates',all(set(s['candidate_ids'])<=ids and len(s['candidate_ids'])<=3 for s in r['local_selection_experiments'].values()))
    result=Shortlist(cfg,'sahidic').search('C998')
    check('C998_returns_ten_current_candidates',len(result['shortlist'])==10)
    check('C998_candidates_support_barley',all('barley' in r['translation'].lower() for r in result['shortlist']))
    check('inference_never_imported_torch_or_sentence_transformers','torch' not in sys.modules and 'sentence_transformers' not in sys.modules)
    write_json(cfg.reports/'validation.json',dict(passed=len(checks),checks=checks))
    print(f'{len(checks)} current artifact checks passed')


if __name__=='__main__':main()
