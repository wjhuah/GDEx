#!/usr/bin/env python3
"""Replay the two audited legacy OCR stages into this project's derived folder only.

The fragile annotated OCR grammar and Bible book aliases are upstream inputs, not
silently rewritten heuristics. This adapter records hashes and overrides every
output path before executing main. It never runs the old merger or feature code.
"""
import hashlib
import importlib.util
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from coptic_examples.config import Config,write_json


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    cfg=Config();cfg.derived.mkdir(parents=True,exist_ok=True)
    parser_path=cfg.source/'crum3/parse_v15.py';matcher_path=cfg.source/'crum3/match_crum_to_corpus_v15.py'
    parse=load(parser_path,'audited_crum_parser')
    parse.KELLIA_XML=cfg.lexicon;parse.CRUM_DIR=cfg.source/'Crum_plain_text';parse.OUT_TSV=cfg.derived/'crum_ocr_references.tsv'
    old_argv=sys.argv
    try:
        sys.argv=[str(parser_path)];parse.main()
    finally:sys.argv=old_argv
    match=load(matcher_path,'audited_crum_matcher')
    match.TRAINING_TSV=parse.OUT_TSV;match.OUT_TSV=cfg.derived/'crum_senses_matched.tsv'
    match.OT_DIR=cfg.corpus/'sahidic.ot';match.NT_DIR=cfg.corpus/'sahidica.nt';match.main()
    write_json(cfg.reports/'crum_rebuild_provenance.json',dict(
        strategy='Reuse audited upstream OCR grammar and Bible aliases; write new outputs; revalidate links in dataset.py',
        implementations=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [parser_path,matcher_path]],
        output=str(match.OUT_TSV),output_sha256=hashlib.sha256(match.OUT_TSV.read_bytes()).hexdigest()))


if __name__=='__main__':main()
