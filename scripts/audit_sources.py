#!/usr/bin/env python3
"""Hash original inputs and older implementations without modifying them."""
import ast
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from coptic_examples.config import Config,write_json


def main():
    cfg=Config();records=[]
    inputs=['citations_manual_merged.tab','citations_manual_annotated.tab','Comprehensive_Coptic_Lexicon-v1.2-2020.xml','alpha_kyima_rc1.db','coptic-nlp/data/mwe.tab','crum3/training_data_v15.tsv','crum3/crum_senses_matched_v15.tsv','crum3/training_v15_merged.tsv','crum3/training_v15_with_sm_bge.tsv','preface.pdf']
    for rel in inputs:
        p=cfg.source/rel
        records.append(dict(path=rel,exists=p.is_file(),bytes=p.stat().st_size if p.is_file() else None,sha256=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None))
    implementations=[]
    for d,dirs,files in os.walk(cfg.source):
        dirs[:]=[x for x in dirs if x not in {cfg.work.name,'.git','.venv','venv','__pycache__','node_modules'}]
        for name in sorted(files):
            if not name.endswith('.py'):continue
            p=Path(d)/name;source=p.read_bytes();entry=dict(path=p.relative_to(cfg.source).as_posix(),sha256=hashlib.sha256(source).hexdigest())
            try:
                tree=ast.parse(source.decode('utf-8-sig'))
                entry['functions']=[dict(name=n.name,line=n.lineno) for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
                entry['imports']=sorted({ast.unparse(n) for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))})
            except (SyntaxError,UnicodeError) as exc:entry['syntax_note']=str(exc)
            implementations.append(entry)
    write_json(cfg.reports/'source_manifest.json',dict(inputs=records,legacy_implementations=implementations))
    print(f'Manifest: {len(inputs)} inputs; {len(implementations)} old Python files')


if __name__=='__main__':main()
