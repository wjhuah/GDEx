#!/usr/bin/env python3
import hashlib,json,sys,urllib.request
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from coptic_examples.config import Config,write_json
from coptic_examples.dataset import read_rows,text_key
cfg=Config();checks=[]
def check(name,value,detail=None):
 checks.append(dict(check=name,passed=bool(value),detail=detail))
 if not value:raise AssertionError(name)
manifest=json.loads((cfg.reports/'source_manifest.json').read_text())
check('original_inputs_unchanged',all(hashlib.sha256((cfg.source/r['path']).read_bytes()).hexdigest()==r['sha256'] for r in manifest['inputs'] if r['exists']))
historical_drift=[r['path'] for r in manifest['legacy_implementations'] if hashlib.sha256((cfg.source/r['path']).read_bytes()).hexdigest()!=r['sha256']]
check('historical_implementation_reference_comparison_recorded',bool(manifest['legacy_implementations']),dict(differences=historical_drift,note='October reference predates user changes. train_v15_sm.py has an October 30 mtime; preserve it, do not restore the old file.'))
held={};positive=set()
for r in read_rows(cfg):
 if r['evidence'] not in {'manual','manual_forbidden_rule'}:continue
 c=r['candidate'];d=c['dialect_id'];f=int(hashlib.sha256(c['work'].encode()).hexdigest()[:8],16)%5
 held.setdefault((d,f),set()).add(text_key(c))
 if r['evidence']=='manual':positive.add((r['TLA'],text_key(c),d))
audit=json.loads((cfg.derived/'sense_v3_negative_audit.json').read_text())
check('augmentation_nonempty',bool(audit),len(audit))
check('no_generated_validation_text',all(r['text_key'] not in held.get((r['dialect'],r['fold']),set()) for r in audit))
check('no_generated_validation_work',all(int(hashlib.sha256(r['work'].encode()).hexdigest()[:8],16)%5!=r['fold'] for r in audit))
check('no_manual_positive_overwritten',all((r['tla'],r['text_key'],r['dialect']) not in positive for r in audit))
live=json.loads((cfg.reports/'sense_v3/live_regression.json').read_text())
check('sixteen_live_entries_tested',len(live)==16 and not any('error' in r for r in live))
check('target_token_ids_returned',all(all(c['tokens'] and all(t['id'] for t in c['tokens']) for c in r['examples']) for r in live))
check('same_POS_tomt_abstains_from_sense_claim',all(r['sense_review'] for r in live if r['TLA'] in {'C4324','C4327'}))
for route in ['/sense-review','/reports/sense_v3/REPORT.md','/search?query=C616&dialect=sahidic&ranker=research','/search?query=C4327&dialect=sahidic&ranker=research']:
 with urllib.request.urlopen('http://localhost:8012'+route,timeout=90) as response:
  body=response.read().decode();check('HTTP '+route,response.status==200 and len(body)>100)
  if 'C616' in route:check('CCIRC_evidence_visible','CCIRC' in body and 'TT target tokens' in body)
  if 'C4327' in route:check('uncertainty_visible','Sense unresolved' in body)
write_json(cfg.reports/'sense_v3/validation.json',dict(passed=len(checks),checks=checks))
print(len(checks),'sense validation checks passed')
